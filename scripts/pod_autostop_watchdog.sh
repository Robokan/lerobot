#!/usr/bin/env bash
# A dead-man's switch for a Runpod pod. Runs on the WORKSTATION, not the pod.
#
#   nohup scripts/pod_autostop_watchdog.sh <pod-id> <hf-repo> <final-step> [max-hours] \
#       > ~/pod_watchdog.log 2>&1 &
#
# WHY IT IS SHAPED LIKE THIS. The previous version only started once training
# was already running, and only watched for the final checkpoint. So when the
# launcher died BEFORE training started — a bad bundle download — nothing was
# watching, the launcher's "|| exit 1" left the GPU up, and the pod billed for
# 18 hours and $64 before a human noticed.
#
# The rule now is: the pod is stopped by DEFAULT, and only positive, repeated
# evidence of healthy training keeps it alive. Three independent conditions
# stop it, and the watchdog never needs the launcher to be correct:
#
#   1. GRACE       — training is not running within GRACE_MIN of arming.
#   2. STALL       — training was running and then stopped, or its log stopped
#                    growing for STALL_MIN.
#   3. DONE / CAP  — the final checkpoint reaches the Hub, or max-hours passes.
#
# Arm it IMMEDIATELY after `pod create` returns, before anything else is
# attempted on the pod. That is the whole point: it must already be watching
# while the risky setup steps run.
set -uo pipefail

POD="${1:?usage: pod_autostop_watchdog.sh <pod-id> <hf-repo> <final-step> [max-hours]}"
REPO="${2:?need the HF repo the pod pushes to}"
FINAL="${3:?need the final step number, e.g. 030000}"
MAX_H="${4:-12}"
RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
PY="${PY:-$HOME/sparkpack/lerobot/.venv/bin/python}"
INTERVAL="${INTERVAL:-180}"
# Setup must show PROGRESS, not finish, inside this. A venv build with the pi
# and peft extras ran past 45 minutes and a fixed deadline killed it mid-install,
# so the timer resets whenever the venv grows. A build that is genuinely wedged
# stops growing and still dies.
GRACE_MIN="${GRACE_MIN:-30}"
STALL_MIN="${STALL_MIN:-25}"     # training may go this long without the log growing

deadline=$(( $(date +%s) + MAX_H * 3600 ))
grace_until=$(( $(date +%s) + GRACE_MIN * 60 ))
seen_training=0
last_log_size=-1
last_growth=$(date +%s)
last_venv_size=-1

echo "[watchdog] pod $POD | grace ${GRACE_MIN}m | stall ${STALL_MIN}m | cap ${MAX_H}h | final $FINAL"

stop_pod() {
    echo "[watchdog] STOPPING pod $POD — $1 — $(date -Is)"
    "$RUNPODCTL" pod stop "$POD" 2>&1 | tail -1 || \
        echo "[watchdog] STOP FAILED — stop it yourself: runpodctl pod stop $POD" >&2
    exit 0
}

pod_state() {
    "$RUNPODCTL" pod get "$POD" -o json 2>/dev/null | "$PY" -c \
        'import sys,json
try: print(json.load(sys.stdin).get("runtimeStatus","?"))
except Exception: print("?")' 2>/dev/null
}

ssh_to_pod() {   # $1 = remote command; prints output, non-zero if unreachable
    local info ip port key
    info=$("$RUNPODCTL" ssh info "$POD" 2>/dev/null) || return 1
    ip=$(echo "$info"   | grep -oE '"ip": "[^"]*"'   | cut -d'"' -f4)
    port=$(echo "$info" | grep -oE '"port": [0-9]*'  | awk '{print $2}')
    key=$(echo "$info"  | grep -oE '"path": "[^"]*"' | cut -d'"' -f4)
    [ -n "$ip" ] && [ -n "$port" ] && [ -n "$key" ] || return 1
    ssh -i "$key" -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null "root@$ip" ". /etc/rp_environment; $1" 2>/dev/null
}

final_on_hub() {
    "$PY" - "$REPO" "$FINAL" <<'PYEOF'
import sys
from huggingface_hub import HfApi
try:
    files = HfApi().list_repo_files(sys.argv[1])
except Exception:
    sys.exit(1)
sys.exit(0 if any(f.startswith(f"checkpoints/{sys.argv[2]}/") for f in files) else 1)
PYEOF
}

while true; do
    now=$(date +%s)
    state=$(pod_state)
    case "$state" in
        stopped|exited|"") echo "[watchdog] pod is $state — nothing to do"; exit 0 ;;
    esac

    [ "$now" -ge "$deadline" ] && stop_pod "hard cap of ${MAX_H}h reached"

    # Is training alive, and is its log still growing?
    #
    # '[l]erobot-train', not 'lerobot-train': the ssh command line carries the
    # pattern onto the pod, so a plain pgrep -f MATCHES ITSELF and reports
    # training running when nothing is. That misfire once stopped a pod in the
    # middle of a healthy venv build — pgrep said "alive", train.log did not
    # exist yet, and the stall timer ran out. The bracket makes the pattern not
    # match its own literal text.
    alive=$(ssh_to_pod 'pgrep -fc "[l]erobot-train" || true')
    size=$(ssh_to_pod 'stat -c%s /workspace/train.log 2>/dev/null || echo -1')
    # Setup has not reached training yet; that is what the grace period is for,
    # and the stall timer must not run during it.
    if [ "$size" = "-1" ] && [ "$seen_training" = 0 ]; then
        alive=0
        # Still installing? Then it is making progress and the grace period is
        # extended. Only a build that has STOPPED growing runs out of time.
        venv=$(ssh_to_pod 'du -sb /workspace/lerobot/.venv 2>/dev/null | cut -f1 || echo 0')
        [[ "$venv" =~ ^[0-9]+$ ]] || venv=0
        if [ "$venv" -gt "$last_venv_size" ]; then
            [ "$last_venv_size" -ge 0 ] && echo "[watchdog] setup progressing: venv $(( venv / 1000000 )) MB — extending grace"
            last_venv_size=$venv
            grace_until=$(( now + GRACE_MIN * 60 ))
        fi
    fi
    [[ "$alive" =~ ^[0-9]+$ ]] || alive=0
    [[ "$size"  =~ ^-?[0-9]+$ ]] || size=-1

    if [ "$alive" -gt 0 ]; then
        if [ "$seen_training" = 0 ]; then
            echo "[watchdog] training is running — grace period satisfied"
            seen_training=1
        fi
        if [ "$size" -gt "$last_log_size" ]; then
            last_log_size=$size
            last_growth=$now
        elif [ $(( now - last_growth )) -ge $(( STALL_MIN * 60 )) ]; then
            stop_pod "training alive but its log has not grown for ${STALL_MIN}m"
        fi
    else
        if [ "$seen_training" = 1 ]; then
            # It ran and is now gone. Give the checkpoint pusher a little time
            # to finish its last upload, then stop.
            echo "[watchdog] training has exited — waiting 5m for the last push"
            sleep 300
            stop_pod "training exited"
        elif [ "$now" -ge "$grace_until" ]; then
            stop_pod "training never started within ${GRACE_MIN}m of arming"
        fi
    fi

    if final_on_hub; then
        echo "[watchdog] final checkpoint $FINAL is on the Hub"
        sleep 180
        stop_pod "final checkpoint pushed"
    fi
    sleep "$INTERVAL"
done
