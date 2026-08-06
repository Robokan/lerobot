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
# Keyboard driver keys (active hand only; Tab switches hands):
#   w/s +x/-x   a/d +y/-y   r/f +z/-z
#   i/k pitch   j/l yaw     u/o roll
#   [ / ] gripper open/close       space  reset targets to current pose
#
# Examples:
#   bash scripts/run_vr_sim.sh                                  # scripted teleop
#   VIEWER=1 CAMERAS=0 DRIVER=keyboard bash scripts/run_vr_sim.sh   # watch + drive by keyboard
#   MODE=record NUM_EPISODES=2 bash scripts/run_vr_sim.sh       # record 2 episodes
#   DRIVER=keyboard bash scripts/run_vr_sim.sh                  # keyboard teleop
#   DRIVER=openxr MODE=record bash scripts/run_vr_sim.sh        # VR record (headset)
#
# Note: MuJoCo 3.9.0 segfaults during GL teardown at interpreter exit on this
# aarch64 box whenever the viewer has been open (reproducible with plain mujoco,
# no lerobot involved). It happens after all work completes and after the robot
# disconnects cleanly, so it is cosmetic — but do not mistake it for a crash in
# the teleop loop.

set -euo pipefail

MODE="${MODE:-teleop}"
DRIVER="${DRIVER:-scripted}"
MODEL_PATH="${MODEL_PATH:-$HOME/sparkpack/openarm_mujoco/v1/scene.xml}"
FPS="${FPS:-50}"

# VIEWER=1 opens an on-screen MuJoCo window to watch the arms (debugging).
# egl is offscreen-only so it cannot present a window; glx serves both the
# window and the offscreen camera renders.
VIEWER="${VIEWER:-0}"
if [[ "${VIEWER}" == "1" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glx}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

# CAMERAS=0 skips the offscreen camera renders. Nothing in teleop consumes the
# images, so dropping them buys frame time while debugging motion.
CAMERAS="${CAMERAS:-1}"

# Teleop-only
TELEOP_TIME_S="${TELEOP_TIME_S:-20}"

# Record-only
REPO_ID="${REPO_ID:-local/openarm-sim-vr}"
SINGLE_TASK="${SINGLE_TASK:-teleoperate the simulated openarm}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
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
    --display_data=false \
    --play_sounds=false
else
  echo "[run_vr_sim] TELEOP: fps=${FPS} driver=${DRIVER} (Ctrl-C to stop)"
  exec lerobot-teleoperate \
    "${ROBOT_ARGS[@]}" \
    "${TELEOP_ARGS[@]}" \
    --fps="${FPS}" \
    --teleop_time_s="${TELEOP_TIME_S}" \
    --display_data=false
fi
