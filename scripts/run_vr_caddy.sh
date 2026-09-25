#!/usr/bin/env bash
# Record HUMAN demonstrations of the caddy-picking task in the MuJoCo sim with
# a VR headset (Quest via WiVRn). Same loop as run_vr_sim.sh, with two changes:
#
#   robot = mujoco_bi_openarm_caddy  — lays out N stacks of bars on coloured
#           pads before EVERY episode and publishes the prompt for that layout
#           ("get bar from <colour> pad"), so each frame is stamped with the
#           right task. The layout draws are the scripted generator's own, so
#           the dataset schema and scene distribution match the scripted sets
#           exactly (16-dim state/action in degrees, ego + two wrist cameras).
#   record  is the default mode, with the headset driver.
#
# Why: 900 planner episodes -> 0-1/20 closed loop; 100 human episodes on the
# real robot -> occasional successes. The demonstrations have to come from a
# person.
#
# Before you start (each session):
#   1. flatpak run io.github.wivrn.wivrn        (leave the dashboard running)
#   2. put the headset on, open the WiVRn client, connect to this machine
#      (same Wi-Fi; if the server list is empty: systemctl status avahi-daemon)
#   3. run this script from a terminal you can see; the MuJoCo window opens too
#
# In the headset you also SEE the prompt: a banner on the camera view reads
# "EP 7 · get bar from purple pad · RIGHT arm", and a red "● REC" while
# recording. With SPEAK=1 (default) each prompt is spoken as well.
#
# In the headset (OpenXR driver):
#   X            toggle tracking — press once with your hands where you want
#                the grippers to be; the arms then follow controller DELTAS
#   Y            passthrough <-> MuJoCo camera view
#   thumbstick   left = left wrist cam, right = right wrist cam, centre = ego
#   triggers     grippers
#   B            START recording the episode      A   STOP and SAVE it
#   right grip   cancel and re-record the current episode
#   (keyboard y / t / n / <- / q in the terminal do the same)
#
# Each episode: the terminal and MuJoCo window show the new layout and the
# prompt. Pick ONE bar from the named pad, put it on the pile at the centre,
# press A. If you knocked something over, right-grip to redo it. Between
# episodes the scene is re-laid out automatically (RESET_TIME_S seconds).
#
# Examples:
#   bash scripts/run_vr_caddy.sh                       # record 20 episodes
#   NUM_EPISODES=50 bash scripts/run_vr_caddy.sh
#   REPO_ID=local/openarm_caddy6_vr_b SEED=2 bash scripts/run_vr_caddy.sh
#   MODE=teleop bash scripts/run_vr_caddy.sh           # practise, record nothing
#   DRIVER=keyboard bash scripts/run_vr_caddy.sh       # no headset: keys in the MuJoCo window
#
# Dataset naming: this fork's lerobot-record stamps the time onto the repo id,
# so each session writes a NEW dataset, e.g. local/openarm_caddy6_vr_20260924_183012.
# To add episodes to an existing one, name it exactly and set RESUME=1 (and use
# a NEW SEED, or the same scene sequence is drawn again):
#   RESUME=1 REPO_ID=local/openarm_caddy6_vr_20260924_183012 SEED=2 bash scripts/run_vr_caddy.sh

set -euo pipefail

MODE="${MODE:-record}"
DRIVER="${DRIVER:-openxr}"
MODEL_PATH="${MODEL_PATH:-$HOME/sparkpack/openarm_mujoco/v1/scene.xml}"
VIEWER="${VIEWER:-1}"
STACKS="${STACKS:-6}"
SEED="${SEED:-0}"
REPO_ID="${REPO_ID:-local/openarm_caddy6_vr}"
NUM_EPISODES="${NUM_EPISODES:-20}"
FPS="${FPS:-30}"                 # what the scripted datasets and the policies use
EPISODE_TIME_S="${EPISODE_TIME_S:-0}"   # 0 = until A / t
RESET_TIME_S="${RESET_TIME_S:-3}"       # the scene re-lays out at the start of this
RESUME="${RESUME:-0}"                   # 1 = append to the exact REPO_ID given (no time stamp)
# IK feel (see src/lerobot/teleoperators/vr_mocap/ik.py). Seven values, J1..J7
# = shoulder(3), elbow, wrist(3). Leave unset for the defaults.
#   IK_JOINT_WEIGHTS  willingness of each joint to serve ORIENTATION changes
#                     (default 0.15,0.15,0.15,0.35,1,1,1: the wrist does the
#                     turning, the shoulder barely helps until the wrist hits
#                     its limit). Raise the first three to bring the shoulder
#                     in earlier, e.g. 0.6,0.6,0.6,0.6,1,1,1.
#   IK_SPRING_WEIGHTS spring stiffness toward the rest pose (default
#                     1,1,1,0.6,0.02,0.02,0.02: stiff shoulder, slack wrist).
#   IK_SPRING_GAIN    overall spring strength (default 0.15).
IK_JOINT_WEIGHTS="${IK_JOINT_WEIGHTS:-}"
IK_SPRING_WEIGHTS="${IK_SPRING_WEIGHTS:-}"
IK_SPRING_GAIN="${IK_SPRING_GAIN:-}"
#   IK_LIMIT_MARGIN_DEG  how far before a joint's limit its willingness starts
#                     fading so the next joint out takes over (default 15).
#                     Widen it to bring the shoulder in EARLIER in a wrist turn:
#                     45 = about halfway through the wrist's travel.
IK_LIMIT_MARGIN_DEG="${IK_LIMIT_MARGIN_DEG:-}"
#   IK_HANDOVER_DEG   wind-up hardening per joint, J1..J7 (default 0,0,0,0,45,45,45):
#                     a wrist joint gives up its share of a turn over this many
#                     degrees from rest, so the shoulder takes over progressively.
#                     Smaller = shoulder joins sooner. 0 = off for that joint.
IK_HANDOVER_DEG="${IK_HANDOVER_DEG:-}"
#   IK_CHAIN_WEIGHTS  keyboard rotation keys as a spring chain: stiffness J1..J7
#                     (default 1,1,1,1,0.33,0.33,0.33 = wrist takes 3x the
#                     shoulder's share, both from the first degree). A joint at
#                     its limit in the needed direction drops out automatically.
#   CHAIN_ROTATION=0  turn the chain off (rotation keys pin the wrist again).
IK_CHAIN_WEIGHTS="${IK_CHAIN_WEIGHTS:-}"
CHAIN_ROTATION="${CHAIN_ROTATION:-1}"
SPEAK="${SPEAK:-1}"                     # 1 = say each prompt aloud (spd-say); route audio to the WiVRn sink to hear it in the headset

cd "$(dirname "$0")/.."

# --- OpenXR runtime: WiVRn ships its manifest inside the flatpak. Register it
# for THIS process only (no change to your system config).
if [[ "${DRIVER}" == "openxr" ]]; then
  if [[ -z "${XR_RUNTIME_JSON:-}" && ! -f "$HOME/.config/openxr/1/active_runtime.json" ]]; then
    XR_RUNTIME_JSON="$(ls /var/lib/flatpak/app/io.github.wivrn.wivrn/*/*/*/files/share/openxr/1/openxr_wivrn.json 2>/dev/null | head -1 || true)"
    if [[ -z "${XR_RUNTIME_JSON}" ]]; then
      echo "ERROR: WiVRn runtime manifest not found — is the flatpak installed? (flatpak install flathub io.github.wivrn.wivrn)" >&2
      exit 1
    fi
    export XR_RUNTIME_JSON
  fi
  if ! flatpak ps 2>/dev/null | grep -q io.github.wivrn.wivrn; then
    echo "WARNING: WiVRn is not running. Start it first:  flatpak run io.github.wivrn.wivrn" >&2
    echo "         (then connect the headset). Continuing in 5 s..." >&2
    sleep 5
  fi
  .venv/bin/python -c "import xr, glfw, OpenGL" 2>/dev/null || {
    echo "ERROR: VR python deps missing:  uv pip install --python .venv/bin/python pyopenxr glfw PyOpenGL" >&2; exit 1; }
fi

if [[ "${VIEWER}" == "1" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glx}"
  export MUJOCO_SAFE_EXIT_AFTER_VIEWER="${MUJOCO_SAFE_EXIT_AFTER_VIEWER:-1}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

ROBOT_ARGS=(
  --robot.type=mujoco_bi_openarm_caddy
  --robot.id=mujoco_bi_openarm_caddy
  --robot.model_path="${MODEL_PATH}"
  --robot.fps="${FPS}"
  --robot.stacks="${STACKS}"
  --robot.seed="${SEED}"
)
[[ "${VIEWER}" == "1" ]] && ROBOT_ARGS+=(--robot.viewer=true)
TELEOP_ARGS=(
  --teleop.type=vr_mocap
  --teleop.id=vr_mocap
  --teleop.model_path="${MODEL_PATH}"
  --teleop.driver="${DRIVER}"
  --teleop.vr_hz="${FPS}"
)
[[ -n "${IK_JOINT_WEIGHTS}" ]]  && TELEOP_ARGS+=(--teleop.joint_weights="[${IK_JOINT_WEIGHTS}]")
[[ -n "${IK_SPRING_WEIGHTS}" ]] && TELEOP_ARGS+=(--teleop.spring_weights="[${IK_SPRING_WEIGHTS}]")
[[ -n "${IK_SPRING_GAIN}" ]]    && TELEOP_ARGS+=(--teleop.spring_gain="${IK_SPRING_GAIN}")
[[ -n "${IK_LIMIT_MARGIN_DEG}" ]] && TELEOP_ARGS+=(--teleop.limit_margin_deg="${IK_LIMIT_MARGIN_DEG}")
[[ -n "${IK_HANDOVER_DEG}" ]]     && TELEOP_ARGS+=(--teleop.handover_deg="[${IK_HANDOVER_DEG}]")
[[ -n "${IK_CHAIN_WEIGHTS}" ]]    && TELEOP_ARGS+=(--teleop.chain_weights="[${IK_CHAIN_WEIGHTS}]")
[[ "${CHAIN_ROTATION}" == "0" ]]   && TELEOP_ARGS+=(--teleop.chain_rotation=false)

if [[ "${MODE}" == "record" ]]; then
  echo "[run_vr_caddy] RECORD -> ${REPO_ID}  episodes=${NUM_EPISODES}  stacks=${STACKS}  seed=${SEED}  fps=${FPS}  driver=${DRIVER}"
  echo "[run_vr_caddy] B = start   A = stop & save   right grip = redo   (keyboard: y / t / <- / q)"
  exec .venv/bin/lerobot-record \
    "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.single_task="get bar from the named pad" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.fps="${FPS}" \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.reset_time_s="${RESET_TIME_S}" \
    --dataset.push_to_hub=false \
    --resume="$([[ "${RESUME}" == "1" ]] && echo true || echo false)" \
    --display_data=false \
    --play_sounds="$([[ "${SPEAK}" == "1" ]] && echo true || echo false)"
else
  echo "[run_vr_caddy] TELEOP (practice, nothing recorded)  driver=${DRIVER}  Ctrl-C to stop"
  exec .venv/bin/lerobot-teleoperate \
    "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" \
    --fps="${FPS}" --display_data=false
fi
