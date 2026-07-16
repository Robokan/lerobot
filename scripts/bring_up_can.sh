#!/usr/bin/env bash
# Bring up the 4 OpenArm CAN buses, resolving each arm by adapter serial.
#
# Why serial-first (not fixed canN names):
#   The kernel assigns can0..can3 in USB enumeration order, which is a
#   property of the computer/hub topology and plug order — NOT of the
#   hardware. The USB serial of each CANable 2 adapter, by contrast, is
#   burned into the board and travels with it to any machine and any port.
#   So we treat the serial as the identity: find each expected serial on
#   whatever canN it landed on, bring that interface up, and report/export
#   the role -> canN mapping. Plugging the same adapters into a different
#   computer Just Works; only replacing an adapter (new serial) requires
#   editing the map below.
#
# TODO(cameras): figure out how to get the same plug-and-play story for the
#   cameras. scripts/identify_cameras.sh has to pin cameras to physical USB
#   ports (/dev/v4l/by-path) because these UVC cameras reported empty serial
#   strings in the SparkJAX device map — there may be nothing on the camera
#   hardware to identify it by. Next time the cameras are attached, run
#   `bash scripts/identify_cameras.sh --list`: if ID_SERIAL_SHORT turns out
#   non-empty and unique per camera, switch the camera map to by-id and the
#   cameras become machine-portable like the CAN adapters. If not, options
#   are: buy cameras that expose serials, flash UVC serials where supported,
#   or accept per-machine by-path mapping.
#
# Why classic CAN (not CAN FD):
#   The CANable 2 USB-CAN boards on this machine run the classic
#   (non-FD) candleLight firmware, so the kernel refuses
#   `ip link set canN type can ... fd on` with "Operation not supported".
#   We therefore configure classic CAN at 1 Mbps. Lerobot OpenArm configs
#   must match: pass `--robot.use_can_fd=False` (and `--teleop.use_can_fd=False`)
#   on the CLI, or set `use_can_fd=False` when building the dataclass.
#
# Usage:
#   sudo bash scripts/bring_up_can.sh             # verify + bring up all 4 buses
#   bash scripts/bring_up_can.sh --check          # verify only (no root, no changes)
#   eval "$(bash scripts/bring_up_can.sh --export)"
#       # sets UMPA_LEFT_CAN / UMPA_RIGHT_CAN / LUMPA_LEFT_CAN / LUMPA_RIGHT_CAN
#       # to the canN each arm is actually on (no root, no changes)
#
# Re-run the bring-up after every reboot until/unless we install a systemd unit.

set -euo pipefail

BITRATE=1000000  # 1 Mbps nominal, classic CAN

# Adapter USB serial -> arm. The serial is the hardware identity; the canN
# name is resolved at runtime. Derived from the SparkJAX ROS2 launch
# arguments observed on this machine. Update ONLY when an adapter is
# physically replaced or rewired to a different arm.
declare -A ROLE_OF_SERIAL=(
  [001B00523630501120353355]="umpa_left"
  [004500533630501120353355]="umpa_right"
  [003100553630501120353355]="lumpa_right"
  [004900303945501620303651]="lumpa_left"
)
declare -A EXPORT_VAR=(
  [umpa_left]="UMPA_LEFT_CAN"
  [umpa_right]="UMPA_RIGHT_CAN"
  [lumpa_right]="LUMPA_RIGHT_CAN"
  [lumpa_left]="LUMPA_LEFT_CAN"
)

MODE="${1:-up}"
if [[ "$MODE" != "up" && "$MODE" != "--check" && "$MODE" != "--export" ]]; then
  echo "Usage: sudo bash $0 | bash $0 --check | bash $0 --export" >&2
  exit 1
fi

if [[ "$MODE" == "up" && ${EUID} -ne 0 ]]; then
  echo "Bringing up interfaces requires root. Try: sudo bash $0" >&2
  echo "(--check and --export are read-only and don't need root.)" >&2
  exit 1
fi

# Discover: read the USB serial of every canN present and match it to a role.
declare -A IFACE_OF_ROLE=()
fail=0
for path in /sys/class/net/can*; do
  [[ -e "$path" ]] || continue
  iface=$(basename "$path")
  serial=$(udevadm info -q property -p "$path" 2>/dev/null \
           | awk -F= '/^ID_SERIAL_SHORT/{print $2}')
  role="${ROLE_OF_SERIAL[$serial]:-}"
  if [[ -z "$role" ]]; then
    echo "[WARN] $iface serial=$serial is not in the adapter map (foreign or" >&2
    echo "       replaced adapter?). Leaving it untouched." >&2
    continue
  fi
  if [[ -n "${IFACE_OF_ROLE[$role]:-}" ]]; then
    echo "[FAIL] role $role matched twice (${IFACE_OF_ROLE[$role]} and $iface)" >&2
    echo "       — duplicate serial in the map or cloned adapter firmware." >&2
    fail=1
    continue
  fi
  IFACE_OF_ROLE[$role]="$iface"
done

for serial in "${!ROLE_OF_SERIAL[@]}"; do
  role="${ROLE_OF_SERIAL[$serial]}"
  if [[ -z "${IFACE_OF_ROLE[$role]:-}" ]]; then
    echo "[FAIL] $role adapter (serial $serial) not found on any canN" >&2
    echo "       (unplugged, no power, or gs_usb driver missing)." >&2
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo >&2
  echo "CAN identification FAILED. Do NOT proceed with calibration or teleop" >&2
  echo "until all four arm adapters are found." >&2
  exit 2
fi

if [[ "$MODE" == "--export" ]]; then
  for role in umpa_left umpa_right lumpa_left lumpa_right; do
    printf '%s=%s\n' "${EXPORT_VAR[$role]}" "${IFACE_OF_ROLE[$role]}"
  done
  exit 0
fi

for role in umpa_left umpa_right lumpa_left lumpa_right; do
  iface="${IFACE_OF_ROLE[$role]}"

  if [[ "$MODE" == "up" ]]; then
    ip link set "$iface" down 2>/dev/null || true
    ip link set "$iface" type can bitrate "$BITRATE"
    # Enlarge the kernel TX queue. The SocketCAN default (10) is fine for the
    # single-threaded synchronous control loop, but the async/RTC path drives the
    # bus from two threads (actor writes + observation reads), so 8-frame batch
    # writes can collide with read bursts and overflow the queue -> "No buffer
    # space available" (ENOBUFS), which drops a motor and makes the arm go wild.
    # 1000 frames drain in well under a ms at 1 Mbps, so this just absorbs bursts.
    ip link set "$iface" txqueuelen 1000
    ip link set "$iface" up
  fi

  state=$(ip -details link show "$iface" 2>/dev/null | awk '/can state/{print $3; exit}' || true)
  printf '[OK]   %-12s -> %s  bitrate=%dMbps  state=%s\n' \
    "$role" "$iface" $((BITRATE/1000000)) "${state:-unknown}"
done

echo
if [[ "$MODE" == "--check" ]]; then
  echo "All 4 arm adapters found (check only — interfaces not (re)configured)."
else
  interfaces=$(for r in umpa_left umpa_right lumpa_left lumpa_right; do
                 printf '%s,' "${IFACE_OF_ROLE[$r]}"; done)
  echo "All 4 CAN buses up, resolved by adapter serial."
  echo "Next: source .venv/bin/activate && \\"
  echo "      lerobot-setup-can --mode=test --interfaces=${interfaces%,} --use_fd=False"
fi
