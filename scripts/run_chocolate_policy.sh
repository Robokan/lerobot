#!/usr/bin/env bash
# Run the converted pi0.5 chocolate-bars policy on the OpenArm followers.
#
# Prereqs (run each in a fresh shell session):
#   1. CAN buses up:           sudo bash scripts/bring_up_can.sh
#   2. lerobot venv active:    source .venv/bin/activate
#   3. Cameras identified:     v4l2-ctl --list-devices  (or `ls /dev/v4l/by-id/`)
#
# What this script does:
#   - Drives both Lumpa follower arms via can2 (right) and can3 (left), classic CAN.
#   - Loads the converted lerobot pi0.5 policy at $POLICY_DIR, with runtime LoRA
#     auto-installed via the lora_runtime_marker.json next to it.
#   - Lifts the arms along the SparkJAX lift spine before handing control to
#     the policy (3 s zero-pose ramp + 3 s pre-ramp to spine[0] + 1.7 s
#     10-waypoint table-clearing arc that ends at the HIGH-cluster
#     training-distribution-center pose + 0.5 s hold at 50 Hz). MID-cluster
#     READY was tried on 28-May; the lift was visually imperceptible
#     (~17 deg max motion from zero) so we reverted to HIGH while keeping
#     fps=50 and 120 s episodes (cheapest unexplored combo: HIGH + native
#     training fps + enough wall-clock for task completion).
#   - Runs the policy at fps=50 to MATCH the training data rate (the
#     openpi LoRA was fine-tuned on 50 Hz data; running at 30 Hz would
#     stretch every learned trajectory by 1.67x). The 1.0s chunk =
#     50 actions then maps cleanly to 1 actual second of wall-clock.
#   - Long episodes (120 s) so even slow task progression (e.g. the
#     model takes ~10 s of "settle" before commanding visible motion)
#     has time to finish a full chocolate-grasp cycle within one episode.
#   - Dumps the first 10 policy actions per episode (state, command, delta)
#     so we can see exactly what the model wants to do right after lift.
#   - Also dumps the first 120 fresh-inference outputs per episode (one log
#     line per chunk, i.e. roughly every 1.0 s at fps=50). Reveals long-
#     horizon drift of the model's intent that the first-10-frames dump
#     can't show, e.g. arms "slowly moving down" = each new chunk targets
#     a lower pose. 120 chunks at ~1.0 s/chunk = 120 s, exactly one
#     episode, so we capture the whole run.
#   - Hard-aborts on bad policy outputs via the 3-gate action safety checker
#     (NaN/Inf, abs envelope, per-step delta), with SparkJAX-equivalent thresholds.
#   - Logs every frame + action to a fresh dataset under
#     ~/.cache/huggingface/lerobot/local/<EVAL_DATASET_NAME>/ so you can replay
#     the run later. (lerobot-record always records, even with a policy.)
#
# Stopping:
#   - Press the right arrow once to end the current episode cleanly.
#   - Press Escape once to stop the whole session cleanly.
#   - Press Ctrl-C once to kill the script; the try/finally in record() will
#     still disconnect the bus and disable torque (a few seconds of cleanup).

set -euo pipefail

POLICY_DIR="/home/evaughan/sparkpack/lerobot/outputs/pi05_chocolate_v4_from_openpi"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
EVAL_DATASET_NAME="eval_pi05_chocolate_${RUN_TS}"
HF_USER="${HF_USER:-evaughan}"

# ---------------------------------------------------------------------------
# Log mirroring. Capture the full stdout+stderr to a timestamped file under
# logs/ while still printing to the terminal. Makes post-mortem inspection
# painless (search the log, share it, diff between runs) instead of relying
# on terminal scrollback.
#
# Ctrl-C safety:
#   By default a ^C delivers SIGINT to every process in the foreground
#   process group, which means `tee` would die at exactly the same instant
#   as `lerobot-record`. Anything still in tee's stdio buffer (final
#   Damiao cleanup logs, the disconnect traceback if cleanup raises, etc.)
#   would be lost. To prevent that, the tee subshell installs
#   ``trap '' INT TERM`` so it ignores both signals. Because POSIX exec
#   preserves SIG_IGN across the exec call, the actual ``tee`` process
#   continues to ignore SIGINT/SIGTERM. It only exits when its stdin pipe
#   reaches EOF — i.e. when this script itself exits and bash closes the
#   write end of the pipe — at which point tee flushes everything and
#   terminates normally.
#
# What gets captured:
#   EVERYTHING this script writes after the exec line, including child
#   processes (systemd-inhibit, lerobot-record, its SVT-AV1 encoders, any
#   Python tracebacks, the [profile_loop] lines, …). Append (-a) so
#   simultaneous re-runs with the same timestamp don't overwrite.
# ---------------------------------------------------------------------------
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_chocolate_policy_${RUN_TS}.log"
exec > >(trap '' INT TERM; exec tee -a "${LOG_FILE}") 2>&1
echo "[run_chocolate_policy] mirroring stdout+stderr to: ${LOG_FILE}"
echo "[run_chocolate_policy] (Ctrl-C is safe — tee ignores SIGINT/SIGTERM and flushes on pipe close)"

# ---------------------------------------------------------------------------
# Hugging Face cache. The default `~/.cache/huggingface/hub/` has root-owned
# .locks/ from a previous `sudo` install, which makes HF Hub fail with
# PermissionError when it tries to grab a lock to download paligemma's
# tokenizer files. The sibling cache below was populated as user evaughan and
# has the 5 tokenizer files we need (~17 MB). HF_HUB_OFFLINE=1 short-circuits
# the lock-creation path entirely — the policy weights live in $POLICY_DIR,
# not on the hub, so we never need network.
# ---------------------------------------------------------------------------
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface_user_cache/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# Camera mapping. Taken from SparkJAX's recorded device map at
# ~/.config/sparkjax/cameras.yaml (the same source openpi_runner_node uses):
#   ego        -> /dev/video0
#   left_wrist -> /dev/video4
#   right_wrist-> /dev/video2
# If you've re-plugged USB cameras since SparkJAX recording, re-check with:
#   v4l2-ctl --list-devices    or    ls /dev/v4l/by-id/
#
# Camera fps note: at 640x480 MJPG these UVC cameras only support 30 and 60
# fps natively (no 50 Hz mode). We run the cameras at 60 Hz and let the
# dataset loop sample at fps=50 below - the OpenCVCamera background thread
# reads frames as fast as the device delivers them and stores the most
# recent in latest_frame; the main loop's async_read() always gets the
# freshest frame at its own (slower) sample rate. Result: capture cadence
# matches the loop, no buffer build-up, no double-counted frames.
# ---------------------------------------------------------------------------
EGO_CAM=/dev/video0
LEFT_WRIST_CAM=/dev/video4
RIGHT_WRIST_CAM=/dev/video2

# ---------------------------------------------------------------------------
# CAN mapping (matches scripts/bring_up_can.sh: can2=Lumpa right, can3=Lumpa left)
# ---------------------------------------------------------------------------
LEFT_CAN="can3"
RIGHT_CAN="can2"

if [ ! -d "$POLICY_DIR" ]; then
  echo "Policy dir not found: $POLICY_DIR" >&2
  exit 1
fi

# Sanity: confirm we're inside the lerobot venv
which lerobot-record | grep -q "lerobot/.venv/bin/lerobot-record" || {
  echo "lerobot-record is not the venv one. Run: source .venv/bin/activate" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Wrap lerobot-record in systemd-inhibit so the system can NOT enter s2idle /
# suspend while the policy is running. Without this, the OS will sometimes
# auto-sleep during a long chunk inference (no keyboard/mouse activity for a
# few minutes), which triggers nvidia-suspend.service. On resume, CUDA stays
# unavailable for a while and torch silently falls back to CPU — at which
# point Pi 0.5 chunk inference becomes ~9 s instead of ~100 ms and the
# control loop drops to ~4 Hz. We block the specific sleep/idle handlers
# instead of masking system-wide targets so normal sleep behavior is
# preserved outside the policy session.
#
# If systemd-inhibit isn't available (non-systemd system, missing tool, etc.),
# we fall through and just run the command directly with a warning.
# ---------------------------------------------------------------------------
INHIBIT_PREFIX=()
if command -v systemd-inhibit >/dev/null 2>&1; then
  INHIBIT_PREFIX=(
    systemd-inhibit
    --what=sleep:idle:handle-lid-switch:handle-power-key:handle-suspend-key
    --who="lerobot-record"
    --why="Running robot policy: blocking system sleep to keep GPU resident"
    --mode=block
    --
  )
  echo "[run_chocolate_policy] systemd-inhibit will block sleep/idle for the session."
else
  echo "[run_chocolate_policy] WARNING: systemd-inhibit not available; system may sleep mid-run." >&2
fi

"${INHIBIT_PREFIX[@]}" \
  lerobot-record \
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
  }" \
  --policy.path="${POLICY_DIR}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.repo_id="${HF_USER}/${EVAL_DATASET_NAME}" \
  --dataset.single_task="put the chocolate bars in the container" \
  --dataset.fps=50 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=5 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --lift_arms_before_policy=true \
  --action_safety_enabled=true \
  --profile_loop=true \
  --profile_log_period_s=1.0 \
  --debug_first_actions=10 \
  --debug_log_each_inference=120 \
  --display_data=false
