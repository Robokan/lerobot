#!/usr/bin/env bash
# Bring up the 4 OpenArm CAN buses on this DGX Spark.
#
# Why classic CAN (not CAN FD):
#   The CANable 2 USB-CAN boards on this machine run the classic
#   (non-FD) candleLight firmware, so the kernel refuses
#   `ip link set canN type can ... fd on` with "Operation not supported".
#   We therefore configure classic CAN at 1 Mbps. Lerobot OpenArm configs
#   must match: pass `--robot.use_can_fd=False` (and `--teleop.use_can_fd=False`)
#   on the CLI, or set `use_can_fd=False` when building the dataclass.
#
# What this script checks (and why it matters):
#   Nothing in CAN or lerobot identifies an arm. The serial of each
#   USB-CAN adapter is the only stable physical identity, and the
#   serial-to-arm mapping is a human convention. We hard-code the
#   expected canN <-> serial pairing below and refuse to call the
#   setup successful if any adapter has been swapped. That turns a
#   silent miscalibration risk into a loud, early error.
#
# Usage:
#   sudo bash scripts/bring_up_can.sh
#
# Re-run after every reboot until/unless we install a systemd unit.

set -euo pipefail

BITRATE=1000000  # 1 Mbps nominal, classic CAN

# Expected canN -> (USB serial, physical role).
# Derived from the SparkJAX ROS2 launch arguments observed on this machine.
# If any of these no longer match, STOP and re-verify wiring before
# running calibration or teleop.
declare -A EXPECTED_SERIAL=(
  [can0]=001B00523630501120353355
  [can1]=004500533630501120353355
  [can2]=003100553630501120353355
  [can3]=004900303945501620303651
)
declare -A ROLE=(
  [can0]="Umpa  left"
  [can1]="Umpa  right"
  [can2]="Lumpa right"
  [can3]="Lumpa left"
)

if [[ ${EUID} -ne 0 ]]; then
  echo "This script must be run as root. Try: sudo bash $0" >&2
  exit 1
fi

mismatch=0

for i in can0 can1 can2 can3; do
  if ! ip link show "$i" >/dev/null 2>&1; then
    echo "[FAIL] $i not present (USB-CAN adapter unplugged or driver missing)" >&2
    mismatch=1
    continue
  fi

  serial=$(udevadm info -q property -p "/sys/class/net/$i" 2>/dev/null \
           | awk -F= '/^ID_SERIAL_SHORT/{print $2}')
  expected=${EXPECTED_SERIAL[$i]}
  if [[ "$serial" != "$expected" ]]; then
    echo "[WARN] $i serial=$serial  (expected $expected for ${ROLE[$i]})" >&2
    echo "       Cable/adapter swap suspected. Re-verify wiring before calibrating." >&2
    mismatch=1
  fi

  ip link set "$i" down 2>/dev/null || true
  ip link set "$i" type can bitrate "$BITRATE"
  # Enlarge the kernel TX queue. The SocketCAN default (10) is fine for the
  # single-threaded synchronous control loop, but the async/RTC path drives the
  # bus from two threads (actor writes + observation reads), so 8-frame batch
  # writes can collide with read bursts and overflow the queue -> "No buffer
  # space available" (ENOBUFS), which drops a motor and makes the arm go wild.
  # 1000 frames drain in well under a ms at 1 Mbps, so this just absorbs bursts.
  ip link set "$i" txqueuelen 1000
  ip link set "$i" up

  state=$(ip -details link show "$i" | awk '/can state/{print $3; exit}')
  printf "[OK]   %s  bitrate=%dMbps  state=%-13s  role=%s  serial=%s\n" \
    "$i" $((BITRATE/1000000)) "$state" "${ROLE[$i]}" "$serial"
done

echo
if [[ $mismatch -eq 0 ]]; then
  echo "All 4 CAN buses up. Serials match the expected wiring."
  echo "Next: source .venv/bin/activate && \\"
  echo "      lerobot-setup-can --mode=test --interfaces=can0,can1,can2,can3 --use_fd=False"
else
  echo "One or more mismatches above. Do NOT proceed with calibration or teleop" >&2
  echo "until the physical wiring matches the expected canN <-> serial map." >&2
  exit 2
fi
