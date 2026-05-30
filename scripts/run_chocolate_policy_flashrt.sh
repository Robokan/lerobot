#!/usr/bin/env bash
# Drive the OpenArm followers from a FlashRT policy server (synchronous bridge).
#
# Unlike scripts/run_chocolate_policy.sh (which loads the converted pi0.5 policy
# IN-PROCESS via lerobot-record), this launcher runs NO model locally. The
# FlashRT server is the policy; scripts/run_flashrt_bridge.py is a thin
# synchronous driver that:
#   - reads the BiOpenArmFollower observation (wire layout [R8,L8], joints deg),
#   - swaps arm halves + converts joints deg->rad to the openpi model layout,
#   - JPEG-encodes ego/left_wrist/right_wrist and calls the server over the
#     openpi websocket protocol,
#   - converts the returned chunk back to wire layout/degrees,
#   - runs the same 3-gate action safety checker before each send_action.
#
# Two consumer paths (MODE), both running on FlashRT's AsyncChunkRunner:
#   rtc  (default) — server-side RTC soft-guidance. This is the port of
#                    lerobot's RTCProcessor (examples/rtc/eval_with_real_robot.py).
#                    Params below mirror scripts/run_chocolate_policy_rtc.sh, the
#                    lerobot RTC run that worked: execution_horizon=20,
#                    schedule=LINEAR, full 50-step chunk, continuous replan.
#                    That run only jerked because lerobot's in-process inference
#                    pushed the delay d past execution_horizon; FlashRT's ~246 ms
#                    keeps d (~13) inside the merge window -> smooth seam.
#   sync           — full 50-step synchronous replan + 5-step seam blend. Debug/A-B.
#
# Prereqs (each in its own shell):
#   1. CAN buses up:        sudo bash scripts/bring_up_can.sh
#   2. lerobot venv active: source .venv/bin/activate
#   3. FlashRT server up on $SERVER_PORT, e.g. (in the FlashRT repo, BF16 +
#      prompt-length bucket caching — correctness-verified path):
#        python scripts/serve_policy_flashrt.py \
#          --checkpoint <orbax-ckpt> --framework jax --no-fp8 \
#          --robot-action-dim 16 --num-views 3 --chunk-size 50 \
#          --runtime-lora <lora> --prewarm-prompt-lens 74,76,78,80,82 \
#          --port 8011
#
# Stopping: Ctrl-C once; the bridge's finally-block disconnects the bus and
# disables torque.

set -euo pipefail

RUN_TS="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Server (FlashRT). Defaults match the BF16 + prompt-bucket-cache server we
# verified at ~246 ms steady-state latency on this task.
# ---------------------------------------------------------------------------
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_PORT="${SERVER_PORT:-8011}"
# rtc (default, lerobot-equivalent) | sync (debug full-chunk replan)
MODE="${MODE:-rtc}"
# RTC params — match scripts/run_chocolate_policy_rtc.sh (the lerobot RTC run).
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-20}"
RTC_SCHEDULE="${RTC_SCHEDULE:-linear}"
EXPECTED_LATENCY_MS="${EXPECTED_LATENCY_MS:-246}"

# ---------------------------------------------------------------------------
# Camera mapping (same device map as run_chocolate_policy.sh / SparkJAX).
#   ego        -> /dev/video0
#   left_wrist -> /dev/video4
#   right_wrist-> /dev/video2
# UVC cameras run at 60 Hz MJPG; the bridge samples at --target-hz (50).
# Re-check device order after re-plugging: v4l2-ctl --list-devices
# ---------------------------------------------------------------------------
EGO_CAM="${EGO_CAM:-/dev/video0}"
LEFT_WRIST_CAM="${LEFT_WRIST_CAM:-/dev/video4}"
RIGHT_WRIST_CAM="${RIGHT_WRIST_CAM:-/dev/video2}"

# ---------------------------------------------------------------------------
# CAN mapping (matches scripts/bring_up_can.sh: can2=Lumpa right, can3=Lumpa left)
# ---------------------------------------------------------------------------
LEFT_CAN="${LEFT_CAN:-can3}"
RIGHT_CAN="${RIGHT_CAN:-can2}"

# ---------------------------------------------------------------------------
# Control loop. fps=50 matches the training data rate (the openpi LoRA was
# fine-tuned on 50 Hz data), so the server's 50-action chunk maps cleanly to
# 1 s of wall-clock.
# ---------------------------------------------------------------------------
TARGET_HZ="${TARGET_HZ:-50}"
DURATION="${DURATION:-120}"
PROMPT="${PROMPT:-put the chocolate bars in the container}"
LIFT_ARMS="${LIFT_ARMS:-true}"

# ---------------------------------------------------------------------------
# Action safety (same 3-gate checker / SparkJAX thresholds as the RTC path).
# Gripper per-step delta gate disabled by default ("none"); set to a float to
# re-enable it.
# ---------------------------------------------------------------------------
ACTION_SAFETY_ENABLED="${ACTION_SAFETY_ENABLED:-true}"
ACTION_SAFETY_MAX_JOINT_DELTA_DEG="${ACTION_SAFETY_MAX_JOINT_DELTA_DEG:-28.647889756541159}"
ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG="${ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG:-none}"
# Fast distal joints: the terminal wrist roll (joint_7) is low-inertia and can
# snap quickly at a chunk seam without whipping the arm, so it's exempted from
# the standard per-step delta gate (like the gripper). It stays bounded by the
# finite + absolute-envelope gates and the follower's own ±limit clamp. Set the
# delta to a float to re-enable a bound, or clear ACTION_SAFETY_FAST_JOINTS to
# treat joint_7 as a normal joint again.
ACTION_SAFETY_FAST_JOINTS="${ACTION_SAFETY_FAST_JOINTS:-joint_7}"
ACTION_SAFETY_MAX_FAST_JOINT_DELTA_DEG="${ACTION_SAFETY_MAX_FAST_JOINT_DELTA_DEG:-none}"

# ---------------------------------------------------------------------------
# Log mirroring (same pattern + Ctrl-C-safe tee as run_chocolate_policy.sh).
# ---------------------------------------------------------------------------
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_chocolate_policy_flashrt_${RUN_TS}.log"
exec > >(trap '' INT TERM; exec tee -a "${LOG_FILE}") 2>&1
echo "[run_chocolate_policy_flashrt] mirroring stdout+stderr to: ${LOG_FILE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Assemble optional flags.
EXTRA_ARGS=()
if [ "${LIFT_ARMS}" != "true" ]; then
  EXTRA_ARGS+=(--no-lift)
fi
if [ "${ACTION_SAFETY_ENABLED}" != "true" ]; then
  EXTRA_ARGS+=(--no-action-safety)
fi
# Only pass the gripper-delta gate if it is a real number (the script's default
# is None = gate disabled). "none" leaves the gate off.
if [ "${ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG}" != "none" ]; then
  EXTRA_ARGS+=(--action-safety-max-gripper-delta-deg "${ACTION_SAFETY_MAX_GRIPPER_DELTA_DEG}")
fi
# Always pass the fast-joint set (may be empty to disable); only pass a numeric
# fast-joint delta when one is requested (default "none" = gate disabled).
EXTRA_ARGS+=(--action-safety-fast-joints "${ACTION_SAFETY_FAST_JOINTS}")
if [ "${ACTION_SAFETY_MAX_FAST_JOINT_DELTA_DEG}" != "none" ]; then
  EXTRA_ARGS+=(--action-safety-max-fast-joint-delta-deg "${ACTION_SAFETY_MAX_FAST_JOINT_DELTA_DEG}")
fi

# Block system sleep during the run so the GPU/server stays resident (same
# rationale as run_chocolate_policy.sh).
INHIBIT_PREFIX=()
if command -v systemd-inhibit >/dev/null 2>&1; then
  INHIBIT_PREFIX=(
    systemd-inhibit
    --what=sleep:idle:handle-lid-switch:handle-power-key:handle-suspend-key
    --who="flashrt-bridge"
    --why="Running robot policy bridge: blocking system sleep"
    --mode=block
    --
  )
fi

"${INHIBIT_PREFIX[@]}" \
  python "${SCRIPT_DIR}/run_flashrt_bridge.py" \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --mode "${MODE}" \
  --rtc-execution-horizon "${RTC_EXECUTION_HORIZON}" \
  --rtc-schedule "${RTC_SCHEDULE}" \
  --expected-latency-ms "${EXPECTED_LATENCY_MS}" \
  --robot-id lumpa \
  --left-can "${LEFT_CAN}" \
  --right-can "${RIGHT_CAN}" \
  --ego-cam "${EGO_CAM}" \
  --left-wrist-cam "${LEFT_WRIST_CAM}" \
  --right-wrist-cam "${RIGHT_WRIST_CAM}" \
  --target-hz "${TARGET_HZ}" \
  --duration "${DURATION}" \
  --prompt "${PROMPT}" \
  --action-safety-max-joint-delta-deg "${ACTION_SAFETY_MAX_JOINT_DELTA_DEG}" \
  "${EXTRA_ARGS[@]}"
