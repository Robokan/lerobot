#!/usr/bin/env bash
# VR teleop of the REAL bimanual OpenArm follower (LUMPA) with the headset.
#
# Same teleoperator as run_vr_caddy.sh (vr_mocap: controller pose -> IK on the
# MuJoCo model -> 16 joint angles in degrees), but the robot is the real
# bi_openarm_follower on CAN instead of the sim. SparkJAX drove these arms the
# same way (MuJoCo IK angles sent straight to the motors, no sign flips), so
# the only corrections here are measured ones:
#
#   zero offsets   each joint's zero error, from scripts/openarm_shadow.py
#                  --save-offsets. The follower subtracts them from readings and
#                  adds them to commands, so the IK model, joint limits and
#                  datasets all see true joint angles. Motors are never re-zeroed.
#   gripper range  -157..+5 deg (SparkJAX's measured open/closed), not the
#                  -65..0 lerobot default that stops the gripper ~40% open.
#   speed caps     MAX_SPEED / GRIPPER_SPEED deg/s (per-tick max_relative_target), plus vr_mocap's
#                  own leash that keeps the command within 10 deg of the arm.
#
# DRY RUN IS THE DEFAULT. With DRY_RUN=1 nothing is ever sent to a motor except
# status requests (enforced in the follower: the bus refuses any other frame).
# The full VR pipeline runs against the real arms' readings, and every second
# the terminal shows how far the command is from where each arm actually is.
# Go live only after a dry run looks right:
#
#   bash scripts/run_vr_real.sh                          # dry run, both arms, headset
#   DRIVER=scripted bash scripts/run_vr_real.sh          # dry run, no headset (canned motion)
#   DRY_RUN=0 ARMS=right bash scripts/run_vr_real.sh     # LIVE: right arm only, left stays off
#   DRY_RUN=0 ARMS=both bash scripts/run_vr_real.sh      # LIVE: both arms
#
# LIVE checklist: arms resting in a safe pose, clear workspace, a hand on the
# e-stop / power switch. When torque comes on the arms GLIDE slowly to the home
# pose (HOME_POSE, a few seconds), then hold it and do not follow the headset
# until you press X (tracking starts from where your hands are, as deltas). Ctrl-C disables the motors, and the arms go LIMP
# and drop, so support them or rest them on the table first.
#
# Before the first run (each boot):
#   sudo ip link set can0 up type can bitrate 1000000   # and can1; or bring_up_can.sh
#   flatpak run io.github.wivrn.wivrn                   # headset connected
#   uv run python scripts/openarm_shadow.py --right can0 --left can1 --save-offsets   # once

set -euo pipefail
cd "$(dirname "$0")/.."

# Everything (including the dry-run "command - actual" lines) is also kept in a
# log, since the terminal scrolls past the start of the run within seconds.
LOG_DIR="${LOG_DIR:-outputs/vr_real_logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "[run_vr_real] logging to ${LOG}"

DRY_RUN="${DRY_RUN:-1}"
ARMS="${ARMS:-both}"                 # arms that go LIVE when DRY_RUN=0: right | left | both
DRIVER="${DRIVER:-openxr}"           # openxr (headset) | scripted | keyboard
# 60 Hz control: the motors' PD chases a new position target every tick, so at
# 30 Hz the arm moved in visible 33 ms steps (shaky). Cameras still run at 30 fps.
FPS="${FPS:-30}"
CAM_FPS="${CAM_FPS:-30}"
# Speed caps (deg/s), turned into the per-tick max_relative_target. The 90 deg/s
# (3 deg/tick at 30 Hz) that also applied to the gripper made both feel slow.
MAX_SPEED="${MAX_SPEED:-150}"        # arm joints
GRIPPER_SPEED="${GRIPPER_SPEED:-400}"
KP_SCALE="${KP_SCALE:-1.0}"          # scales the follower's MIT position gains
# Damping of the wrist motors J5..J7 (MIT kd). lerobot's default 0.3 (vs 3-5 on
# J1..J4) let them ring around the VR target: in the trace they reversed 61-81
# times while their commands kept going one way. KD="k1,...,k8" sets all eight.
WRIST_KD="${WRIST_KD:-0.8}"
KD="${KD:-}"
MODEL_PATH="${MODEL_PATH:-$HOME/sparkpack/openarm_mujoco/v1/scene.xml}"
OFFSETS="${OFFSETS:-$HOME/.cache/huggingface/lerobot/calibration/robots/openarm_follower/lumpa_zero_offsets.json}"
ROBOT_ID="${ROBOT_ID:-lumpa_follower}"
GRIPPER_LIMITS="${GRIPPER_LIMITS:-[-157, 5]}"
# Where the arms go when the motors come on, J1..J7 degrees, left mirrored: the
# start pose tested in the headset in the sim (run_vr_caddy.sh START_POSE with
# ELBOW_HEIGHT=35). They GLIDE there at HOME_SPEED deg/s, then teleop starts.
HOME_POSE="${HOME_POSE:-0,0,-35,0,35,0,0}"
HOME_SPEED="${HOME_SPEED:-15}"
# The pose the IK springs pull toward: the home pose with the elbow bent, as in
# run_vr_caddy.sh (a straight-elbow rest would pull the arm into its singularity).
REST_POSE="${REST_POSE:-$(IFS=, read -r -a p <<< "${HOME_POSE}"; p[3]=90; IFS=,; echo "${p[*]}")}"
CAMERAS="${CAMERAS:-1}"              # 1 = ego + wrist cameras to the headset (and datasets)
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"

case "${ARMS}" in right|left|both) ;; *) echo "ARMS must be right, left or both" >&2; exit 1;; esac

# --- CAN: find each follower arm by adapter serial (same map as bring_up_can.sh).
serial_for() { sed -n "s/^\s*\[\([0-9A-Fa-f]*\)\]=\"$1\".*/\1/p" scripts/bring_up_can.sh; }
iface_for_serial() {
  local want="$1" i d p
  for i in /sys/class/net/can*; do
    [[ -e "$i" ]] || continue
    d=$(readlink -f "$i/device"); p="$d"
    while [[ "$p" != "/" ]]; do
      if [[ -f "$p/serial" ]]; then [[ "$(cat "$p/serial")" == "$want" ]] && { basename "$i"; return; }; break; fi
      p=$(dirname "$p")
    done
  done
}
RIGHT_CAN="${RIGHT_CAN:-$(iface_for_serial "$(serial_for lumpa_right)")}"
LEFT_CAN="${LEFT_CAN:-$(iface_for_serial "$(serial_for lumpa_left)")}"
for pair in "right:${RIGHT_CAN}" "left:${LEFT_CAN}"; do
  side="${pair%%:*}"; iface="${pair#*:}"
  if [[ -z "${iface}" ]]; then echo "ERROR: ${side} follower CAN adapter not found (see scripts/bring_up_can.sh)" >&2; exit 1; fi
  if [[ "$(cat /sys/class/net/${iface}/operstate)" != "up" ]]; then
    echo "ERROR: ${iface} (${side} arm) is down:  sudo ip link set ${iface} up type can bitrate 1000000" >&2; exit 1
  fi
done
echo "[run_vr_real] right arm = ${RIGHT_CAN}, left arm = ${LEFT_CAN}"

# --- Cameras, resolved by USB port (identify_cameras.sh). All three are USB 2.0
# devices sharing one 480 Mbit/s hub, so they must stream MJPEG: uncompressed,
# the second one to open fails with "Not enough bandwidth".
CAMERA_ARGS=()
if [[ "${CAMERAS}" == "1" ]]; then
  eval "$(bash scripts/identify_cameras.sh --export)"
  cam() { echo "$1: {type: opencv, index_or_path: $2, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: MJPG}"; }
  CAMERA_ARGS=(--robot.cameras="{$(cam ego "${EGO_CAM}"), $(cam left_wrist "${LEFT_WRIST_CAM}"), $(cam right_wrist "${RIGHT_WRIST_CAM}")}")
  echo "[run_vr_real] cameras: ego=${EGO_CAM} left_wrist=${LEFT_WRIST_CAM} right_wrist=${RIGHT_WRIST_CAM}"
fi

# --- Per-arm settings (offsets, limits, dry run) and calibration files, so that
# connect() never lands in lerobot's interactive calibration: that prompt
# re-zeroes the motors on ENTER, which would destroy the zero SparkJAX and the
# offsets are measured against.
mapfile -t ARM_ARGS < <(WRIST_KD="${WRIST_KD}" KD="${KD}" VEL_FF="${VEL_FF:-0}" LOG="${LOG}" DRY_RUN="${DRY_RUN}" ARMS="${ARMS}" OFFSETS="${OFFSETS}" ROBOT_ID="${ROBOT_ID}" \
  KP_SCALE="${KP_SCALE}" GRIPPER_LIMITS="${GRIPPER_LIMITS}" .venv/bin/python - <<'EOF'
import json, os, sys
from pathlib import Path
import draccus
from lerobot.motors import MotorCalibration
from lerobot.robots.openarm_follower import OpenArmFollowerConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

dry, arms = os.environ["DRY_RUN"] == "1", os.environ["ARMS"]
offsets = {}
if os.path.isfile(os.environ["OFFSETS"]):
    offsets = json.load(open(os.environ["OFFSETS"]))
elif not dry:
    sys.exit(f"ERROR: no zero offsets at {os.environ['OFFSETS']} -- run openarm_shadow.py --save-offsets first")
defaults = OpenArmFollowerConfig(port="x")
kp = [round(k * float(os.environ["KP_SCALE"]), 3) for k in defaults.position_kp]
grip = json.loads(os.environ["GRIPPER_LIMITS"])
if os.environ["KD"]:
    kd = [float(v) for v in os.environ["KD"].split(",")]
    assert len(kd) == 8, "KD needs 8 values, J1..J7 and the gripper"
else:
    kd = list(defaults.position_kd)
    kd[4:7] = [float(os.environ["WRIST_KD"])] * 3
print(f"[run_vr_real] kd {kd}", file=sys.stderr)
for side in ("right", "left"):
    live = not dry and arms in (side, "both")
    p = f"--robot.{side}_arm_config"
    print(f"{p}.dry_run={'false' if live else 'true'}")
    print(f"{p}.zero_offsets={json.dumps(offsets.get(side, {}))}")
    print(f'{p}.joint_limits_override={json.dumps({"gripper": grip})}')
    print(f"{p}.position_kp={json.dumps(kp)}")
    print(f"{p}.position_kd={json.dumps(kd)}")
    print(f"{p}.velocity_feedforward={'true' if os.environ.get('VEL_FF', '0') == '1' else 'false'}")
    print(f"{p}.trace_path={os.environ['LOG'].removesuffix('.log')}_{side}.csv")
    if live:
        cal = HF_LEROBOT_CALIBRATION / "robots" / "openarm_follower" / f"{os.environ['ROBOT_ID']}_{side}.json"
        if not cal.is_file():
            cal.parent.mkdir(parents=True, exist_ok=True)
            data = {m: MotorCalibration(id=mid, drive_mode=0, homing_offset=0, range_min=-90, range_max=90)
                    for m, (mid, _, _) in defaults.motor_config.items()}
            with open(cal, "w") as f, draccus.config_type("json"):
                draccus.dump(data, f, indent=4)
            print(f"[run_vr_real] wrote {cal} (ranges only; motors are not re-zeroed)", file=sys.stderr)
    print(f"[run_vr_real] {side}: {'LIVE' if live else 'dry run'}, offsets {offsets.get(side, 'none')}", file=sys.stderr)
EOF
)

# --- OpenXR runtime (WiVRn flatpak), for this process only.
if [[ "${DRIVER}" == "openxr" ]]; then
  if [[ -z "${XR_RUNTIME_JSON:-}" && ! -f "$HOME/.config/openxr/1/active_runtime.json" ]]; then
    XR_RUNTIME_JSON="$(ls /var/lib/flatpak/app/io.github.wivrn.wivrn/*/*/active/files/share/openxr/1/openxr_wivrn.json \
      "$HOME"/.local/share/flatpak/app/io.github.wivrn.wivrn/*/*/active/files/share/openxr/1/openxr_wivrn.json 2>/dev/null | head -1 || true)"
    [[ -n "${XR_RUNTIME_JSON}" ]] || { echo "ERROR: WiVRn runtime manifest not found" >&2; exit 1; }
    export XR_RUNTIME_JSON
  fi
  flatpak ps 2>/dev/null | grep -q io.github.wivrn.wivrn || echo "WARNING: WiVRn is not running (flatpak run io.github.wivrn.wivrn)" >&2
fi

STEP_CAP="$(awk -v a="${MAX_SPEED}" -v g="${GRIPPER_SPEED}" -v f="${FPS}" 'BEGIN{
  s = "{"; for (i = 1; i <= 7; i++) s = s sprintf("\"joint_%d\": %.3f, ", i, a / f)
  printf "%s\"gripper\": %.3f}", s, g / f }')"
echo "[run_vr_real] ${FPS} Hz, step cap ${MAX_SPEED} deg/s arm, ${GRIPPER_SPEED} deg/s gripper"
TELEOP_EXTRA=(
  --teleop.home_pose_deg="[${HOME_POSE}]"
  --teleop.home_speed_deg_s="${HOME_SPEED}"
  --teleop.rest_pose_deg="[${REST_POSE}]"
)
[[ "${SMOOTH:-0}" == "1" ]] && TELEOP_EXTRA+=(--teleop.smooth_controllers=true)
echo "[run_vr_real] home pose [${HOME_POSE}] at ${HOME_SPEED} deg/s, springs toward [${REST_POSE}]"
if [[ "${DRY_RUN}" == "1" ]]; then
  # The arms never move in a dry run, so the leash toward the actual arm would
  # pin the command to it. Let it run free to show what WOULD be commanded.
  TELEOP_EXTRA+=(--teleop.actual_leash_deg=1000)
else
  echo
  echo "  LIVE on: ${ARMS} arm(s). Hand on the e-stop. Ctrl-C disables the motors (arms go limp)."
  read -r -p "  Type 'live' to continue: " ok
  [[ "${ok}" == "live" ]] || { echo "aborted"; exit 1; }
fi

# TELEOP_CMD: e.g. ".venv/bin/python -m cProfile -o prof.out .venv/bin/lerobot-teleoperate" to profile.
exec ${TELEOP_CMD:-.venv/bin/lerobot-teleoperate} \
  --robot.type=bi_openarm_follower \
  --robot.id="${ROBOT_ID}" \
  --robot.right_arm_config.port="${RIGHT_CAN}" \
  --robot.right_arm_config.side=right \
  --robot.right_arm_config.use_can_fd=false \
  --robot.right_arm_config.max_relative_target="${STEP_CAP}" \
  --robot.left_arm_config.port="${LEFT_CAN}" \
  --robot.left_arm_config.side=left \
  --robot.left_arm_config.use_can_fd=false \
  --robot.left_arm_config.max_relative_target="${STEP_CAP}" \
  "${ARM_ARGS[@]}" \
  "${CAMERA_ARGS[@]}" \
  --teleop.type=vr_mocap \
  --teleop.id=vr_mocap \
  --teleop.model_path="${MODEL_PATH}" \
  --teleop.driver="${DRIVER}" \
  --teleop.vr_hz="${FPS}" \
  "${TELEOP_EXTRA[@]}" \
  --fps="${FPS}" \
  --display_data=false
