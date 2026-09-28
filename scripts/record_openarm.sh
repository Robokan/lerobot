#!/usr/bin/env bash
# Record bimanual OpenArm teleop episodes into a LeRobot dataset (for pi0.5
# finetuning — see scripts/train_pi05_openarm.sh).
#
# Rig roles on this machine:
#   UMPA  = leader  (the pair you move by hand; torque off)
#   LUMPA = follower (the pair that mirrors and gets recorded)
#
# Device identity is resolved at runtime, never hard-coded:
#   - CAN interfaces come from bring_up_can.sh --export (adapter USB serial ->
#     whatever canN it landed on). Run `sudo bash scripts/bring_up_can.sh`
#     once after every reboot before using this script.
#   - Camera device nodes come from identify_cameras.sh --export. The
#     EXPECTED_DEVICE map in that script must be filled in once (run it with
#     --list with all three cameras plugged in) or this script fails loudly.
#
# The CANable 2 adapters run classic (non-FD) firmware, so every arm config
# gets use_can_fd=false to match the 1 Mbps classic-CAN bring-up.
#
# First run: each arm that has no calibration file triggers lerobot's
# interactive calibration on connect — follow the prompts once per arm; the
# files are stored under the robot/teleop id and reused afterwards.
#
# Usage:
#   REPO_ID=eric/openarm_pick_cube TASK="Pick up the cube and place it in the bin" \
#     bash scripts/record_openarm.sh
#
# Optional env overrides (defaults in parentheses):
#   NUM_EPISODES (10)  EPISODE_TIME_S (60)  RESET_TIME_S (15)  FPS (30)
#   CAM_WIDTH (640)  CAM_HEIGHT (480)  PUSH_TO_HUB (false)  RESUME (false)

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ID="${REPO_ID:?Set REPO_ID, e.g. REPO_ID=eric/openarm_pick_cube}"
TASK="${TASK:?Set TASK, e.g. TASK='Pick up the cube and place it in the bin'}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-15}"
FPS="${FPS:-30}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
RESUME="${RESUME:-false}"

# Resolve hardware identities (both fail loudly on missing/ambiguous devices).
eval "$(bash scripts/bring_up_can.sh --export)"
eval "$(bash scripts/identify_cameras.sh --export)"

# shellcheck disable=SC1091
source .venv/bin/activate

exec lerobot-record \
    --robot.type=bi_openarm_follower \
    --robot.id=lumpa_follower \
    --robot.left_arm_config.port="$LUMPA_LEFT_CAN" \
    --robot.left_arm_config.side=left \
    --robot.left_arm_config.use_can_fd=false \
    --robot.right_arm_config.port="$LUMPA_RIGHT_CAN" \
    --robot.right_arm_config.side=right \
    --robot.right_arm_config.use_can_fd=false \
    --robot.cameras="{
        ego:         {type: opencv, index_or_path: $EGO_CAM,         width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS, fourcc: MJPG},
        left_wrist:  {type: opencv, index_or_path: $LEFT_WRIST_CAM,  width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS, fourcc: MJPG},
        right_wrist: {type: opencv, index_or_path: $RIGHT_WRIST_CAM, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS, fourcc: MJPG}}" \
    --teleop.type=bi_openarm_leader \
    --teleop.id=umpa_leader \
    --teleop.left_arm_config.port="$UMPA_LEFT_CAN" \
    --teleop.left_arm_config.use_can_fd=false \
    --teleop.right_arm_config.port="$UMPA_RIGHT_CAN" \
    --teleop.right_arm_config.use_can_fd=false \
    --dataset.repo_id="$REPO_ID" \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.fps="$FPS" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --resume="$RESUME" \
    --display_data=true
