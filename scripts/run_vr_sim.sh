#!/usr/bin/env bash
# VR motion-capture teleop / recording for the MuJoCo-simulated bimanual OpenArm.
#
# Drives a fully simulated BiOpenArm (no hardware, no CAN) with the VR mocap
# teleoperator through the standard lerobot loop:
#
#   robot  = mujoco_bi_openarm   (dynamic MuJoCo sim, 16 right-first *.pos deg)
#   teleop = vr_mocap            (pose source -> IK -> joint targets)
#
# Two modes (MODE):
#   teleop  (default) — lerobot-teleoperate: arms track the pose source live.
#   record            — lerobot-record: capture episodes into a LeRobotDataset
#                       whose schema matches the real chocolate datasets
#                       (observation.state (16,), action (16,), image features).
#
# Pose driver (DRIVER):
#   scripted  (default) — deterministic headless motion (no input device).
#   keyboard            — single-char terminal control (needs an interactive TTY).
#   openxr              — real VR headset (Phase 2; run on the headset machine).
#
# Prereqs:
#   1. lerobot venv with the sim extra:  uv pip install -e ".[openarm-sim]"
#      (or just `uv pip install mujoco`), then: source .venv/bin/activate
#   2. Headless rendering backend: MUJOCO_GL=egl (default below). Use osmesa if
#      EGL is unavailable on your machine.
#
# Watching the sim (VIEWER=1) — opens an on-screen MuJoCo window. CAMERAS=0
# additionally skips the offscreen camera renders, which nothing in teleop
# consumes; on this box that is the difference between ~50 Hz and ~40 Hz.
#
# Keyboard driver keys (active hand only; Tab switches hands).
# With VIEWER=1, focus the MuJoCo window — keys are read from there too:
#   w/s +x/-x   a/d +y/-y   r/f +z/-z
#   i/k pitch   j/l yaw     u/o roll
#   [ / ] gripper open/close       space  reset targets to current pose
#   c     cycle viewer camera (ego → right_wrist → left_wrist → free)
#
# Record mode (MODE=record) — teleop continuously; frames only while armed:
#   y     start recording episode
#   t     stop recording and save episode
#   n     end episode early (while recording)
#   ←     re-record (Left arrow only — letter r is teleop +z, not re-record)
#   q     quit
#   Note: OpenXR headset Y is tracking toggle; use keyboard/viewer Y/T for record.
#
# OpenXR driver (Quest / WiVRn) — keep CAMERAS=1 (default) so the headset can
# show the MuJoCo ego + wrist feeds:
#   X  toggle tracking (delta teleop from current controller pose)
#   Y  toggle passthrough ↔ MuJoCo camera view
#   thumbstick  left = left wrist cam, right = right wrist cam, center = chest
#   B  start recording        A  stop recording & save
#   right grip squeeze  cancel (re-record) the current episode
#   triggers = grippers
#   (keyboard y/t/n/←/q still work as a fallback in record mode)
#
# Examples:
#   bash scripts/run_vr_sim.sh                                  # scripted teleop
#   VIEWER=1 CAMERAS=0 DRIVER=keyboard bash scripts/run_vr_sim.sh   # watch + drive by keyboard
#   MODE=record NUM_EPISODES=2 bash scripts/run_vr_sim.sh       # record 2 episodes
#   DRIVER=keyboard bash scripts/run_vr_sim.sh                  # keyboard teleop
#   VIEWER=1 DRIVER=openxr bash scripts/run_vr_sim.sh           # VR teleop (headset)
#   DRIVER=openxr MODE=record bash scripts/run_vr_sim.sh        # VR record (headset)
#
# Note: MuJoCo 3.9.0 segfaults during GL teardown at interpreter exit on this
# aarch64 box whenever the viewer has been open (reproducible with plain mujoco,
# no lerobot involved). It happens AFTER a clean disconnect — not while you are
# teleoperating. VIEWER=1 sets MUJOCO_SAFE_EXIT_AFTER_VIEWER so we os._exit(0)
# and skip that teardown (otherwise apport writes ~1GB crash dumps every run).

set -euo pipefail

MODE="${MODE:-teleop}"
DRIVER="${DRIVER:-scripted}"
MODEL_PATH="${MODEL_PATH:-$HOME/sparkpack/openarm_mujoco/v1/scene.xml}"

# VIEWER=1 opens an on-screen MuJoCo window to watch the arms (debugging).
# egl is offscreen-only so it cannot present a window; glx serves both the
# window and the offscreen camera renders.
VIEWER="${VIEWER:-0}"
if [[ "${VIEWER}" == "1" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glx}"
  export MUJOCO_SAFE_EXIT_AFTER_VIEWER="${MUJOCO_SAFE_EXIT_AFTER_VIEWER:-1}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

# CAMERAS=0 skips the offscreen camera renders. Nothing in teleop consumes the
# images, so dropping them buys frame time while debugging motion.
CAMERAS="${CAMERAS:-1}"

# Target control / dataset FPS. With VIEWER=1 + cameras, ~30 Hz is typical on
# this box; claiming 50 Hz floods slow-loop warnings and mislabels timestamps.
if [[ -z "${FPS:-}" ]]; then
  if [[ "${VIEWER}" == "1" && "${CAMERAS}" != "0" ]]; then
    FPS=30
  else
    FPS=50
  fi
fi

# Teleop-only. Empty/unset = run until Ctrl-C (do not default to a short
# timeout — holding keys for ~20s used to look like a crash on the next press).
TELEOP_TIME_S="${TELEOP_TIME_S-}"

# Record-only
REPO_ID="${REPO_ID:-local/openarm-sim-vr}"
SINGLE_TASK="${SINGLE_TASK:-teleoperate the simulated openarm}"
NUM_EPISODES="${NUM_EPISODES:-1}"
# 0 = no auto-stop; recording runs until T (or n/q). Set e.g. 60 to re-enable a cap.
EPISODE_TIME_S="${EPISODE_TIME_S:-0}"
RESET_TIME_S="${RESET_TIME_S:-2}"

ROBOT_ARGS=(
  --robot.type=mujoco_bi_openarm
  --robot.id=mujoco_bi_openarm
  --robot.model_path="${MODEL_PATH}"
)
if [[ "${VIEWER}" == "1" ]]; then
  ROBOT_ARGS+=(--robot.viewer=true)
fi
if [[ "${CAMERAS}" == "0" ]]; then
  ROBOT_ARGS+=(--robot.cameras='{}')
fi
TELEOP_ARGS=(
  --teleop.type=vr_mocap
  --teleop.id=vr_mocap
  --teleop.model_path="${MODEL_PATH}"
  --teleop.driver="${DRIVER}"
  --teleop.vr_hz="${FPS}"
)

if [[ "${MODE}" == "record" ]]; then
  echo "[run_vr_sim] RECORD: repo_id=${REPO_ID} episodes=${NUM_EPISODES} fps=${FPS} driver=${DRIVER}"
  exec lerobot-record \
    "${ROBOT_ARGS[@]}" \
    "${TELEOP_ARGS[@]}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.single_task="${SINGLE_TASK}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.fps="${FPS}" \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.reset_time_s="${RESET_TIME_S}" \
    --dataset.push_to_hub=false \
    --display_data=false \
    --play_sounds=false
else
  TELEOP_CMD=(
    lerobot-teleoperate
    "${ROBOT_ARGS[@]}"
    "${TELEOP_ARGS[@]}"
    --fps="${FPS}"
    --display_data=false
  )
  if [[ -n "${TELEOP_TIME_S}" ]]; then
    TELEOP_CMD+=(--teleop_time_s="${TELEOP_TIME_S}")
    echo "[run_vr_sim] TELEOP: fps=${FPS} driver=${DRIVER} for ${TELEOP_TIME_S}s (Ctrl-C to stop)"
  else
    echo "[run_vr_sim] TELEOP: fps=${FPS} driver=${DRIVER} until Ctrl-C"
  fi
  exec "${TELEOP_CMD[@]}"
fi
