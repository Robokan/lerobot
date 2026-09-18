#!/usr/bin/env bash
# Runs on the WORKSTATION, not the pod: stop a Runpod pod as soon as its
# training run is done, and unconditionally after a deadline.
#
# Why a second mechanism: pod_push_checkpoints.sh stops the pod itself when the
# last checkpoint is on the Hub, but that needs runpodctl (or RUNPOD_API_KEY) to
# work from inside the pod, which depends on the image. And runpodctl 2.14 has
# no --stop-after at creation, so nothing else caps the bill. This uses the
# runpodctl already authenticated on this machine, so no key is ever copied
# anywhere, and it is detached, so it outlives the shell that started it.
#
# Usage:
#   nohup scripts/pod_autostop_watchdog.sh <pod-id> <hf-user>/<repo> <final-step> [max-hours] \
#       > ~/pod_watchdog.log 2>&1 &
set -uo pipefail

POD="${1:?usage: pod_autostop_watchdog.sh <pod-id> <hf-repo> <final-step> [max-hours]}"
REPO="${2:?need the HF repo the pod pushes to}"
FINAL="${3:?need the final step number, e.g. 070000}"
MAX_H="${4:-12}"
RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
PY="${PY:-$HOME/sparkpack/lerobot/.venv/bin/python}"
INTERVAL="${INTERVAL:-300}"

deadline=$(( $(date +%s) + $(printf '%.0f' "$(echo "$MAX_H * 3600" | bc)") ))
echo "[watchdog] pod $POD | waiting for $REPO checkpoint $FINAL | hard stop $(date -d "@$deadline" -Is)"

stopped() {  # pod already not running?
    local st
    st=$("$RUNPODCTL" pod get "$POD" -o json 2>/dev/null | "$PY" -c \
        'import sys,json;print(json.load(sys.stdin).get("runtimeStatus","?"))' 2>/dev/null)
    [ "$st" = "stopped" ] || [ "$st" = "exited" ] || [ -z "$st" ]
}

stop_pod() {
    echo "[watchdog] stopping pod $POD ($1) at $(date -Is)"
    "$RUNPODCTL" stop pod "$POD" && echo "[watchdog] stop requested" || \
        echo "[watchdog] STOP FAILED — stop it from the console" >&2
}

while true; do
    if stopped; then
        echo "[watchdog] pod is no longer running — nothing to do"; exit 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        stop_pod "deadline reached"; exit 0
    fi
    # The final checkpoint landing on the Hub is the real "training is done"
    # signal: it is written by the pod but observable from here without ssh.
    if "$PY" - "$REPO" "$FINAL" <<'PYEOF'
import sys
from huggingface_hub import HfApi
repo, final = sys.argv[1], sys.argv[2]
try:
    files = HfApi().list_repo_files(repo)
except Exception:
    sys.exit(1)          # repo not there yet, or offline: keep waiting
sys.exit(0 if any(f.startswith(f"checkpoints/{final}/") for f in files) else 1)
PYEOF
    then
        echo "[watchdog] checkpoint $FINAL is on the Hub"
        sleep 180        # let the push script finish its prune/squash
        stop_pod "final checkpoint pushed"; exit 0
    fi
    sleep "$INTERVAL"
done
