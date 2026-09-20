#!/usr/bin/env bash
# Last line of defence: stop any pod that is running without healthy training,
# regardless of who started it or whether its watchdog is alive.
#
#   scripts/pod_guard.sh            # report only
#   scripts/pod_guard.sh --enforce  # stop anything unhealthy
#
# Run it from cron every 15 minutes:
#   */15 * * * * $HOME/sparkpack/lerobot/scripts/pod_guard.sh --enforce >> $HOME/pod_guard.log 2>&1
#
# The per-run watchdog covers the normal case. This covers the cases it cannot:
# the watchdog process was killed, the workstation rebooted, a pod was created
# by hand, or a session ended while a pod was up. It knows nothing about any
# particular run — it only asks "is this pod doing useful work?" and stops it if
# not. Judgement is deliberately crude, because the cost of a false stop is one
# restarted run and the cost of a false pass is measured in dollars per hour.
set -uo pipefail

ENFORCE=0
[ "${1:-}" = "--enforce" ] && ENFORCE=1
RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
PY="${PY:-$HOME/sparkpack/lerobot/.venv/bin/python}"
MAX_IDLE_MIN="${MAX_IDLE_MIN:-45}"   # a pod may exist this long without training
STATE_DIR="${STATE_DIR:-$HOME/.cache/pod_guard}"
mkdir -p "$STATE_DIR"

pods=$("$RUNPODCTL" pod list -o json 2>/dev/null) || { echo "$(date -Is) cannot reach runpod"; exit 0; }
n=$(echo "$pods" | "$PY" -c 'import sys,json; print(len(json.load(sys.stdin)))' 2>/dev/null || echo 0)
[ "${n:-0}" -gt 0 ] || { echo "$(date -Is) no running pods"; exit 0; }

echo "$pods" | "$PY" -c '
import sys, json
for p in json.load(sys.stdin):
    print(p["id"], p.get("name","?"), p.get("costPerHr",0))
' | while read -r id name cost; do
    info=$("$RUNPODCTL" ssh info "$id" 2>/dev/null)
    ip=$(echo "$info"   | grep -oE '"ip": "[^"]*"'   | cut -d'"' -f4)
    port=$(echo "$info" | grep -oE '"port": [0-9]*'  | awk '{print $2}')
    key=$(echo "$info"  | grep -oE '"path": "[^"]*"' | cut -d'"' -f4)
    alive=0
    if [ -n "$ip" ] && [ -n "$port" ] && [ -n "$key" ]; then
        out=$(ssh -i "$key" -p "$port" -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
              -o UserKnownHostsFile=/dev/null "root@$ip" \
              'pgrep -fc "[l]erobot-train|[p]ython.*train" || true' 2>/dev/null)
        [[ "$out" =~ ^[0-9]+$ ]] && alive=$out
    fi
    mark="$STATE_DIR/$id.idle_since"
    if [ "$alive" -gt 0 ]; then
        rm -f "$mark"
        echo "$(date -Is) $id ($name) OK — training running, \$$cost/hr"
        continue
    fi
    [ -f "$mark" ] || date +%s > "$mark"
    idle_min=$(( ( $(date +%s) - $(cat "$mark") ) / 60 ))
    echo "$(date -Is) $id ($name) IDLE ${idle_min}m — no training, \$$cost/hr"
    if [ "$idle_min" -ge "$MAX_IDLE_MIN" ]; then
        if [ "$ENFORCE" = 1 ]; then
            echo "$(date -Is) $id STOPPING — idle ${idle_min}m >= ${MAX_IDLE_MIN}m"
            "$RUNPODCTL" pod stop "$id" 2>&1 | tail -1
            rm -f "$mark"
        else
            echo "$(date -Is) $id WOULD STOP (run with --enforce)"
        fi
    fi
done
