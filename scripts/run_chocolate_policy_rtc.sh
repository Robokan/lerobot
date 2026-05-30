#!/usr/bin/env bash
# Run the converted pi0.5 chocolate-bars policy on the OpenArm followers using
# LeRobot's Real-Time Chunking (RTC) path: examples/rtc/eval_with_real_robot.py.
#
# Why RTC (and not the synchronous lerobot-record path, nor the async_inference
# weighted-average blending path):
#   pi0.5 is a flow-matching action-chunking policy. The Physical Intelligence
#   RTC method inpaints each new chunk so it stays consistent with the part of
#   the previous chunk still being executed - that is the ONLY chunk-stitching
#   scheme PI report as working for these policies; naive blending (weighted
#   average across chunk seams) produces the jerky / shaking motion we saw.
#   examples/rtc/eval_with_real_robot.py is LeRobot's own RTC-on-real-hardware
#   reference and natively supports pi05 + bi_openarm_follower.
#
# What we changed in the example (and ONLY this):
#   The example assumes a NATIVE lerobot checkpoint, where the model action
#   space == the robot wire space. Our checkpoint is converted from
#   openpi/SparkJAX, so the stamped processors do three extra things the stock
#   example didn't account for in its RTC-prefix re-anchoring:
#     * arm halves swapped   ([right,left] wire  <->  [left,right] model)
#     * joints deg <-> rad
#     * (images are already handled correctly - the example feeds full-res
#       frames and the model resizes internally, so no 3px-wide bug here)
#   The forward path (observation in, fresh chunk out) already goes through the
#   stamped pre/post-processors, so it was fine. The one gap was
#   _reanchor_relative_rtc_prefix, which re-expressed the leftover prefix in
#   WIRE space before normalizing - we now run it through the same stamped
#   arm-swap + angle steps so the RTC prefix lands in MODEL space. That change
#   is a no-op for native lerobot checkpoints. We also added an opt-in
#   --lift_arms that reuses the SparkJAX lift spine, off by default.
#
# Prereqs (run each in a fresh shell session):
#   1. CAN buses up:           sudo bash scripts/bring_up_can.sh
#   2. lerobot venv active:    source .venv/bin/activate
#   3. Cameras identified:     v4l2-ctl --list-devices  (or `ls /dev/v4l/by-id/`)
#
# Stopping:
#   - Press Ctrl-C once: the example's signal handler sets the shutdown event,
#     the threads drain, and robot.disconnect() disables torque. tee ignores
#     SIGINT (see below) so the cleanup logs are still captured.

set -euo pipefail

POLICY_DIR="/home/evaughan/sparkpack/lerobot/outputs/pi05_chocolate_v4_from_openpi"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# RTC + timing knobs (overridable via env).
#
# Sizing for pi0.5 here: chunk = 50 actions (1.0 s @ 50 Hz) and measured
# inference latency ~0.4 s = ~20 steps of inference_delay. The first tuning
# pass (horizon=25, threshold=40, no interpolation) ran but was jerky; the log
# showed (a) "action_queue_size_to_get_new_actions Too small" every chunk
# (40 < horizon 25 + delay 20 = 45) and (b) "Indexes diff is not equal to real
# delay" (e.g. indexes_diff=5 vs real_delay=19) - i.e. the queue drained and a
# replace skipped ~14 actions forward, a visible jump at the chunk seam.
#
#   FPS:                action (policy) rate. 50 to MATCH the training data rate
#                       (the openpi LoRA was fine-tuned on 50 Hz data).
#   EXECUTION_HORIZON:  # of already-committed steps RTC freezes/guides when
#                       inpainting the next chunk. Set ~= inference_delay (20):
#                       it must cover the steps that execute *during* inference,
#                       and making it bigger only inflates the replan-threshold
#                       requirement below.
#   MAX_GUIDANCE_WEIGHT/PREFIX_ATTENTION_SCHEDULE: RTC inpainting strength + how
#                       prefix attention decays across the chunk (LINEAR matches
#                       the stock bi_openarm example).
#   QUEUE_GET_NEW:      replan when the queue drains to this size. After each RTC
#                       replace the queue holds ~chunk - delay = ~30 usable
#                       steps, so a threshold of 48 keeps the requester replanning
#                       essentially continuously from the freshest observation -
#                       which is the regime where indexes_diff tracks real_delay
#                       (smooth seams) and the queue never stalls.
#   INTERP_MULT:        control-rate upsampling. KEEP AT 1 (50 Hz) on this rig:
#                       at 2 (100 Hz) the SocketCAN TX queue overflowed with
#                       "No buffer space available" (ENOBUFS) - the kernel CAN
#                       txqueuelen (default 10) can't drain 16 motor frames x
#                       100 Hz in bursts. The synchronous path runs at 50 Hz,
#                       which is why it never hit this. The latency fix already
#                       collapsed the chunk seams to ~15 deg, so interpolation is
#                       no longer needed for smoothness. To re-enable >1 later,
#                       first raise the bus TX queue:
#                       sudo ip link set can2 txqueuelen 1000 (and can3).
# ---------------------------------------------------------------------------
FPS="${FPS:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-20}"
MAX_GUIDANCE_WEIGHT="${MAX_GUIDANCE_WEIGHT:-5.0}"
PREFIX_ATTENTION_SCHEDULE="${PREFIX_ATTENTION_SCHEDULE:-LINEAR}"
QUEUE_GET_NEW="${QUEUE_GET_NEW:-48}"
INTERP_MULT="${INTERP_MULT:-1}"
DURATION="${DURATION:-120}"
TASK="${TASK:-put the chocolate bars in the container}"

# Action safety: 3-gate hard-abort run on every (raw, 50 Hz) policy action
# before it hits the motors (finite / absolute-envelope / per-step delta). On a
# violation the actor holds the last good pose and shuts the session down.
#
# Conservative defaults (0.5 rad/step = 28.65 deg joints, 1.0 rad/step =
# 57.30 deg gripper). These deliberately ABORT on the large discontinuities our
# RTC pipeline is currently producing - that abort is the hardware protection,
# not a bug to tune away. Loosen via env only once the motion is calm.
ACTION_SAFETY_ENABLED="${ACTION_SAFETY_ENABLED:-true}"
ACTION_SAFETY_MAX_JOINT_DELTA_DEG="${ACTION_SAFETY_MAX_JOINT_DELTA_DEG:-28.647889756541159}"
# Gripper per-step delta gate is DISABLED by default ("none"). The gripper is a
# fast, human-teleoperated, near-binary actuator: 60-80 deg/step open/close is
# normal teleop motion (verified in the recordings), not a fault, and slamming a
# gripper open/shut can't whip the arm. The gripper is still bounded by the
# finite + absolute-envelope gates. Set to a number to re-enable a bound.
ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG="${ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG:-none}"

# Only forward the gripper-delta flag when a real bound is requested; leaving it
# off lets the script default (None = disabled) stand.
GRIPPER_DELTA_ARG=()
if [[ "${ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG,,}" != "none" && -n "${ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG}" ]]; then
  GRIPPER_DELTA_ARG=(--action_safety_max_gripper_delta_deg="${ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG}")
fi

# RTC on/off. With RTC enabled the queue is REPLACED from the freshest
# observation every replan (continuous re-planning + inpainting). With it
# disabled the policy runs OPEN-LOOP: execute a whole chunk, then append the
# next - one seam per chunk instead of continuous replace. Open-loop is the
# closest analogue to the (jerky-but-functional) SparkJAX baseline, so it is a
# useful A/B: RTC_ENABLED=false bash scripts/run_chocolate_policy_rtc.sh
RTC_ENABLED="${RTC_ENABLED:-true}"

# ---------------------------------------------------------------------------
# Log mirroring (same Ctrl-C/tee rationale as run_chocolate_policy.sh: tee
# ignores SIGINT/SIGTERM so it flushes the final cleanup logs on pipe close).
# ---------------------------------------------------------------------------
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_chocolate_policy_rtc_${RUN_TS}.log"
exec > >(trap '' INT TERM; exec tee -a "${LOG_FILE}") 2>&1
echo "[run_chocolate_policy_rtc] mirroring stdout+stderr to: ${LOG_FILE}"

# ---------------------------------------------------------------------------
# HF cache (same as the sync script: avoid root-owned lock dirs, stay offline -
# the policy weights live in $POLICY_DIR, not on the hub).
# ---------------------------------------------------------------------------
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface_user_cache/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# Camera + CAN mapping (identical to run_chocolate_policy.sh).
#   ego -> /dev/video0, left_wrist -> /dev/video4, right_wrist -> /dev/video2
#   can2 = Lumpa right, can3 = Lumpa left.
# Cameras run at 60 Hz (no native 50 Hz mode); the loop samples the freshest
# frame at FPS.
# ---------------------------------------------------------------------------
EGO_CAM=/dev/video0
LEFT_WRIST_CAM=/dev/video4
RIGHT_WRIST_CAM=/dev/video2
LEFT_CAN="can3"
RIGHT_CAN="can2"

if [ ! -d "$POLICY_DIR" ]; then
  echo "Policy dir not found: $POLICY_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Block system sleep/idle while the policy runs (see run_chocolate_policy.sh:
# an s2idle mid-inference drops CUDA and the loop falls back to CPU at ~9 s/chunk).
# ---------------------------------------------------------------------------
INHIBIT_PREFIX=()
if command -v systemd-inhibit >/dev/null 2>&1; then
  INHIBIT_PREFIX=(
    systemd-inhibit
    --what=sleep:idle:handle-lid-switch:handle-power-key:handle-suspend-key
    --who="rtc-chocolate"
    --why="Running RTC robot policy: blocking system sleep to keep GPU resident"
    --mode=block
    --
  )
  echo "[run_chocolate_policy_rtc] systemd-inhibit will block sleep/idle for the session."
else
  echo "[run_chocolate_policy_rtc] WARNING: systemd-inhibit not available; system may sleep mid-run." >&2
fi

"${INHIBIT_PREFIX[@]}" \
  python "${REPO_ROOT}/examples/rtc/eval_with_real_robot.py" \
  --policy.path="${POLICY_DIR}" \
  --policy.device=cuda \
  --device=cuda \
  --rtc.enabled="${RTC_ENABLED}" \
  --rtc.execution_horizon="${EXECUTION_HORIZON}" \
  --rtc.max_guidance_weight="${MAX_GUIDANCE_WEIGHT}" \
  --rtc.prefix_attention_schedule="${PREFIX_ATTENTION_SCHEDULE}" \
  --fps="${FPS}" \
  --interpolation_multiplier="${INTERP_MULT}" \
  --action_queue_size_to_get_new_actions="${QUEUE_GET_NEW}" \
  --duration="${DURATION}" \
  --lift_arms=true \
  --action_safety_enabled="${ACTION_SAFETY_ENABLED}" \
  --action_safety_max_joint_delta_deg="${ACTION_SAFETY_MAX_JOINT_DELTA_DEG}" \
  "${GRIPPER_DELTA_ARG[@]}" \
  --task="${TASK}" \
  --robot.type=bi_openarm_follower \
  --robot.id=lumpa \
  --robot.left_arm_config.port="${LEFT_CAN}" \
  --robot.left_arm_config.side=left \
  --robot.left_arm_config.use_can_fd=false \
  --robot.left_arm_config.can_bitrate=1000000 \
  --robot.right_arm_config.port="${RIGHT_CAN}" \
  --robot.right_arm_config.side=right \
  --robot.right_arm_config.use_can_fd=false \
  --robot.right_arm_config.can_bitrate=1000000 \
  --robot.cameras="{
    ego: {type: opencv, index_or_path: ${EGO_CAM}, width: 640, height: 480, fps: 60, fourcc: MJPG},
    left_wrist: {type: opencv, index_or_path: ${LEFT_WRIST_CAM}, width: 640, height: 480, fps: 60, fourcc: MJPG},
    right_wrist: {type: opencv, index_or_path: ${RIGHT_WRIST_CAM}, width: 640, height: 480, fps: 60, fourcc: MJPG}
  }"
