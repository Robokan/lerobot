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
#                    lerobot RTC run: execution_horizon=20, schedule=LINEAR, full
#                    50-step chunk, continuous replan. Latency is now solved
#                    (FlashRT ~200 ms, no graph-capture spikes), so d (~10-13)
#                    stays inside the merge window. The server's guidance-weight
#                    ceiling is 5.0 (FLASHRT_RTC_MAX_GW), matching lerobot.
#   sync           — full 50-step synchronous replan + 5-step seam blend, NO RTC
#                    guidance and NO continuous replan. Use as the A/B isolation:
#                    if sync is smooth (just slower) the jerk is the RTC layer;
#                    if sync is also jerky it's the model chunks / obs conversion.
#                    Run with:  MODE=sync bash scripts/run_chocolate_policy_flashrt.sh
#
# Prereqs (each in its own shell):
#   1. CAN buses up:        sudo bash scripts/bring_up_can.sh
#   2. lerobot venv active: source .venv/bin/activate
#   3. FlashRT server up on $SERVER_PORT. VERIFIED CONFIG — FP8-QAT torch
#      frontend, ~4.8 deg arm MAE @ p50 173ms / p99 194ms (matches the eager
#      reference; see notes below). Two non-obvious requirements:
#        - the merged checkpoint MUST ship assets/openarm/norm_stats.json
#          (16-DoF). Without it the frontend silently falls back to stale
#          LIBERO 7-DoF stats in ~/.cache/openpi -> ~13 deg regression. A
#          dim guard in pi05_rtx._load_norm_stats now hard-fails on this.
#        - do NOT set FLASHRT_PAD_STATE=1. State-prompt padding corrupts the
#          conditioning (~13 deg). Use --prewarm-prompt-lens instead to kill
#          the per-length CUDA-graph rebuild spikes (no accuracy cost).
#        - FLASHRT_FIXED_NOISE (EXPERIMENTAL, default OFF). The pi0.5 decode is
#          inherently noise-sensitive (~2 deg BF16 / ~5 deg FP8 chunk shift per
#          noise draw; eager and this frontend agree chunk-for-chunk given the
#          SAME noise). Fresh noise every inference is the source of the closed-
#          loop seam jerk. Pinning to one seed removes the jerk (0.000 deg
#          run-to-run) but a single sample is not a guaranteed-coherent grasp
#          trajectory (seed 0 -> robot made no attempt). Left OFF until a good
#          sample / noise-averaging scheme is found. Set =1 to experiment.
#        (in the FlashRT repo)
#        PYTHONPATH=.:$HOME/sparkpack/openpi/src:$HOME/sparkpack/openpi/packages/openpi-client/src \
#        python scripts/serve_policy_flashrt.py \
#          --checkpoint runs/openarm_fp8_slow2x/merged_serve --framework torch \
#          --robot-action-dim 16 --num-views 3 --chunk-size 50 \
#          --delta-action-mask 7,-1,7,-1 \
#          --calib-data runs/openarm_fp8_slow2x/calib_openarm_merged_64.npz \
#          --default-prompt "put the chocolate bars in the container" \
#          --runtime-lora 0 --prewarm-prompt-lens 60-100 --port 8011
#
# Stopping: Ctrl-C once; the bridge's finally-block disconnects the bus and
# disables torque.

set -euo pipefail

RUN_TS="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Server (FlashRT). Defaults match the FP8-QAT torch + prompt-bucket-prewarm
# server we verified at ~173 ms steady-state latency (4.8 deg arm) on this task.
# ---------------------------------------------------------------------------
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_PORT="${SERVER_PORT:-8011}"
# rtc (default, lerobot-equivalent) | sync (debug full-chunk replan)
MODE="${MODE:-rtc}"
# RTC params — match scripts/run_chocolate_policy_rtc.sh (the lerobot RTC run).
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-20}"
RTC_SCHEDULE="${RTC_SCHEDULE:-linear}"
EXPECTED_LATENCY_MS="${EXPECTED_LATENCY_MS:-175}"  # FP8 torch p50 ~173ms (was 246 on BF16 JAX)

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
echo "[run_chocolate_policy_flashrt] MODE=${MODE} (rtc=soft-guidance replan | sync=full-chunk A/B baseline)  exec_horizon=${RTC_EXECUTION_HORIZON} schedule=${RTC_SCHEDULE} fps=${TARGET_HZ}"

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
