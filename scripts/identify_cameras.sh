#!/usr/bin/env bash
# Identify the OpenArm cameras by stable identity and resolve them to
# /dev/videoN — the camera analogue of scripts/bring_up_can.sh.
#
# Why this exists:
#   /dev/videoN indices are assigned in USB enumeration order, so they shuffle
#   across reboots and re-plugs. Feeding the wrong index into lerobot-record
#   silently swaps the ego / wrist views — the policy still runs, it just sees
#   the world through the wrong eyes. The only stable identities are:
#     /dev/v4l/by-id/    device model + serial   (empty serial on cheap UVC
#                        cams => identical models collide and by-id is useless)
#     /dev/v4l/by-path/  physical USB port chain (survives reboots; changes
#                        only if you move the plug to a different port)
#   This script hard-codes the expected role <-> identity mapping and refuses
#   to report success if a camera is missing or ambiguous, turning a silent
#   view-swap risk into a loud, early error.
#
# Usage:
#   bash scripts/identify_cameras.sh            # verify + human-readable report
#   bash scripts/identify_cameras.sh --list     # show everything plugged in
#                                               # (use this to fill in the map)
#   eval "$(bash scripts/identify_cameras.sh --export)"
#                                               # sets EGO_CAM / LEFT_WRIST_CAM /
#                                               # RIGHT_WRIST_CAM for run scripts
#
# Filling in the map:
#   Plug in all cameras, run `--list`, and paste each camera's by-id name
#   (preferred, if its serial is non-empty and unique) or by-path name into
#   EXPECTED_DEVICE below. A unique substring of the symlink name is enough.
#
# TODO(portability): unlike the CAN adapters (whose USB serials are burned
#   into the boards, letting scripts/bring_up_can.sh resolve arms on any
#   machine with no per-machine setup), these UVC cameras recorded EMPTY
#   serial strings in the SparkJAX device map. If that holds (verify with
#   --list next time the cameras are attached), identical cameras are only
#   distinguishable by physical USB port (by-path), which makes this map
#   inherently per-machine. We still need a way to make cameras
#   plug-and-play: cameras that expose real serials, flashing UVC serials
#   where the vendor tool supports it, or some content-based fingerprint at
#   startup. Until then: re-run --list and refill the map on every new
#   machine (and after moving a plug to a different port).

set -euo pipefail

# Expected role -> stable device identity (substring of a /dev/v4l/by-id/ or
# /dev/v4l/by-path/ symlink name). The role names match the *_CAM variables
# used by the run scripts and the SparkJAX map at ~/.config/sparkjax/cameras.yaml
# (ego -> /dev/video0, left_wrist -> /dev/video4, right_wrist -> /dev/video2 at
# the time of recording; indices are NOT stable, which is the point of this
# script).
#
# TODO: fill these in from `bash scripts/identify_cameras.sh --list` with all
# three cameras plugged into their usual ports.
declare -A EXPECTED_DEVICE=(
  [ego]=""
  [left_wrist]=""
  [right_wrist]=""
)

# Role -> environment variable emitted by --export.
declare -A EXPORT_VAR=(
  [ego]="EGO_CAM"
  [left_wrist]="LEFT_WRIST_CAM"
  [right_wrist]="RIGHT_WRIST_CAM"
)

MODE="${1:-verify}"

# Collect candidate symlinks: capture nodes only. UVC cameras expose two
# /dev/videoN nodes (capture + metadata); the by-id/by-path names distinguish
# them with an "-index0" (capture) / "-index1" (metadata) suffix.
candidates() {
  local dir
  for dir in /dev/v4l/by-id /dev/v4l/by-path; do
    [[ -d "$dir" ]] || continue
    local link
    for link in "$dir"/*-index0; do
      [[ -e "$link" ]] && printf '%s\n' "$link"
    done
  done
}

describe() {  # $1 = /dev/videoN — one-line udev identity summary
  udevadm info -q property -n "$1" 2>/dev/null \
    | awk -F= '/^(ID_MODEL|ID_SERIAL_SHORT)=/{printf "%s=%s  ", $1, $2}'
}

if [[ "$MODE" == "--list" ]]; then
  found=0
  while IFS= read -r link; do
    found=1
    dev=$(readlink -f "$link")
    printf '%-16s <- %s\n' "$dev" "$link"
    printf '                   %s\n' "$(describe "$dev")"
  done < <(candidates)
  if [[ $found -eq 0 ]]; then
    echo "No V4L capture devices found. Plug the cameras in and re-run." >&2
    exit 2
  fi
  exit 0
fi

if [[ "$MODE" != "verify" && "$MODE" != "--export" ]]; then
  echo "Usage: $0 [--list | --export]" >&2
  exit 1
fi

mapfile -t links < <(candidates)

fail=0
declare -A RESOLVED=()
for role in ego left_wrist right_wrist; do
  want="${EXPECTED_DEVICE[$role]}"
  if [[ -z "$want" ]]; then
    echo "[FAIL] $role: EXPECTED_DEVICE not filled in yet." >&2
    echo "       Run: bash $0 --list   and paste the identity into this script." >&2
    fail=1
    continue
  fi

  matches=()
  for link in "${links[@]}"; do
    [[ "$(basename "$link")" == *"$want"* ]] && matches+=("$link")
  done

  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "[FAIL] $role: no /dev/v4l symlink matches '$want' (camera unplugged," >&2
    echo "       moved to a different USB port, or the map is stale)." >&2
    fail=1
    continue
  fi
  # The same physical camera legitimately matches twice when it appears under
  # both by-id and by-path — collapse to the resolved device before calling
  # it ambiguous.
  mapfile -t devs < <(for m in "${matches[@]}"; do readlink -f "$m"; done | sort -u)
  if [[ ${#devs[@]} -gt 1 ]]; then
    echo "[FAIL] $role: '$want' matches ${#devs[@]} different devices:" >&2
    printf '       %s\n' "${devs[@]}" >&2
    echo "       Use a longer, unique substring (serial or full port path)." >&2
    fail=1
    continue
  fi

  RESOLVED[$role]="${devs[0]}"
done

# One camera matched by two roles = the map is wrong (e.g. same substring
# pasted twice). Catch it before someone records a dataset with it.
for a in "${!RESOLVED[@]}"; do
  for b in "${!RESOLVED[@]}"; do
    if [[ "$a" < "$b" && "${RESOLVED[$a]}" == "${RESOLVED[$b]}" ]]; then
      echo "[FAIL] roles '$a' and '$b' resolved to the same device ${RESOLVED[$a]}" >&2
      fail=1
    fi
  done
done

if [[ $fail -ne 0 ]]; then
  echo >&2
  echo "Camera identification FAILED. Do NOT record or run a policy until every" >&2
  echo "role above resolves — a swapped view corrupts datasets silently." >&2
  exit 2
fi

if [[ "$MODE" == "--export" ]]; then
  for role in ego left_wrist right_wrist; do
    printf '%s=%s\n' "${EXPORT_VAR[$role]}" "${RESOLVED[$role]}"
  done
else
  for role in ego left_wrist right_wrist; do
    dev="${RESOLVED[$role]}"
    printf '[OK]   %-12s %s  (%s)\n' "$role" "$dev" "$(describe "$dev")"
  done
  echo
  echo "All cameras identified. Use in run scripts with:"
  echo '  eval "$(bash scripts/identify_cameras.sh --export)"'
fi
