#!/usr/bin/env bash
# Runs ON a Runpod pod: push every permanent checkpoint to a private HF repo
# as soon as it appears.
#
# Why this exists: a pod with no network volume is TERMINATED when the account
# balance hits zero, and Runpod keeps no backups — "gone for good" in their
# words. Pulling a 12.6 GB checkpoint down over a ~1 MB/s home link takes ~3.5 h,
# which is longer than the gap between checkpoints, so downloading can never
# keep up. The pod, however, has datacenter bandwidth: the same upload to
# HuggingFace takes minutes. So the pod pushes, and nothing is ever only on the
# pod for long.
#
# Auth: run `hf auth login` on the pod first with a WRITE token. This
# script never takes a token as an argument — it reads whatever the HF CLI
# stored, so the token is never in a command line, a log, or this file.
#
# Usage (on the pod):
#   nohup bash scripts/pod_push_checkpoints.sh <hf-user>/<repo> > /workspace/push.log 2>&1 &
#
# When training has finished and everything is on the Hub, this STOPS THE POD
# (set ON_DONE=terminate to delete it instead, ON_DONE=none to leave it up).
set -uo pipefail

REPO="${1:?usage: pod_push_checkpoints.sh <hf-user>/<repo-name> [output-dir] [interval-s] [keep-n] [min-step]}"
OUT="${2:-/workspace/lerobot/outputs/groot_caddy6_3cam}"
INTERVAL="${3:-120}"
KEEP="${4:-4}"          # how many of the newest checkpoints to keep on the Hub
# Early checkpoints of a from-scratch run are never the one you deploy, and each
# costs 12.6 GB of a 100 GB Hub quota. Below this step a checkpoint is skipped
# and DELETED from the pod (freeing its ~24 GB, weights + optimiser state, so a
# long run cannot fill the disk the way the 50k colour run did).
MIN_STEP="${5:-${MIN_STEP:-40000}}"
ON_DONE="${ON_DONE:-stop}"      # stop | terminate | none
STATE="/workspace/.pushed_checkpoints"
VENV="/workspace/lerobot/.venv/bin"

touch "$STATE"
export PATH="$VENV:$HOME/.local/bin:$PATH"
# The pod image exports HF_HOME from its interactive shell profile, which a
# non-interactive ssh command never sources — so `hf auth login` writes the
# token to /workspace/.cache/huggingface while anything run over ssh looks in
# ~/.cache/huggingface and sees nothing. Pin it.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# Creating the repo IS the write test. Checking the token's reported role is
# not: a fine-grained token returns role=None even with full write permission,
# so a role check refuses a perfectly good token.
if ! "$VENV/python" - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
try:
    who = HfApi().whoami()
except Exception as e:
    print("NOT AUTHENTICATED:", type(e).__name__); sys.exit(1)
try:
    HfApi().create_repo(repo, repo_type="model", private=True, exist_ok=True)
except Exception as e:
    print("TOKEN CANNOT WRITE:", type(e).__name__, str(e)[:160]); sys.exit(2)
print(f"hf user: {who.get('name')} | private repo ready: {repo}")
PY
then
    echo "ERROR: no usable HuggingFace write token on this pod." >&2
    echo "Run:  /workspace/lerobot/.venv/bin/hf auth login   (token needs write access)" >&2
    exit 1
fi

# Stop (or delete) this pod. Nothing here ever sees a key we supply: runpodctl
# on the pod image is already authenticated for its own pod, and RUNPOD_API_KEY
# is only read if the pod was created with one in its environment. If neither
# works the pod stays up and says so — the pod's own --stop-after backstop,
# set at creation, is what guarantees it cannot bill forever.
shut_down() {
    case "$ON_DONE" in
        none) echo "[push] ON_DONE=none — leaving the pod running"; return 0 ;;
        terminate) verb=remove ;;
        *) verb=stop ;;
    esac
    local pod="${RUNPOD_POD_ID:-}"
    [ -n "$pod" ] || { echo "[push] RUNPOD_POD_ID not set — cannot $verb the pod; STOP IT YOURSELF" >&2; return 1; }
    echo "[push] ${verb}ping pod $pod ..."
    if command -v runpodctl >/dev/null && runpodctl "$verb" pod "$pod"; then
        echo "[push] pod $verb requested via runpodctl"; return 0
    fi
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        local url="https://rest.runpod.io/v1/pods/$pod/stop"
        [ "$verb" = remove ] && url="https://rest.runpod.io/v1/pods/$pod"
        if [ "$verb" = remove ]; then
            curl -fsS -X DELETE "$url" -H "Authorization: Bearer $RUNPOD_API_KEY" && \
                { echo "[push] pod deleted via REST"; return 0; }
        else
            curl -fsS -X POST "$url" -H "Authorization: Bearer $RUNPOD_API_KEY" && \
                { echo "[push] pod stopped via REST"; return 0; }
        fi
    fi
    echo "[push] COULD NOT $verb THE POD — it is still billing. Stop it from the console." >&2
    return 1
}

# "No lerobot-train process" means training has FINISHED only if it was ever
# seen running. Without this the script starts while cloud_train_setup.sh is
# still doing uv sync, sees no trainer and no checkpoints, concludes the run is
# complete and stops the pod — which is exactly what happened on the first try.
SEEN_TRAINING=0

echo "[push] watching $OUT every ${INTERVAL}s -> $REPO"
echo "[push] min step to push: $MIN_STEP | keep newest $KEEP on the Hub | when done: $ON_DONE"
while true; do
    if [ -d "$OUT/checkpoints" ]; then
        for d in "$OUT"/checkpoints/[0-9]*/; do
            [ -d "$d" ] || continue
            step=$(basename "$d")
            grep -qx "$step" "$STATE" && continue
            # A LoRA run writes adapter_model.safetensors; a full fine-tune writes
            # model.safetensors. Requiring only the latter made this script skip
            # EVERY checkpoint of the pi0.5 LoRA run, silently, and a completed
            # 20,000-step run ended with an empty Hub repo. Accept either, and
            # say so when neither is there instead of `continue`ing in silence.
            w="$d/pretrained_model/model.safetensors"
            [ -f "$w" ] || w="$d/pretrained_model/adapter_model.safetensors"
            if [ ! -f "$w" ]; then
                echo "[push] $step: no model.safetensors or adapter_model.safetensors — skipping this pass"
                continue
            fi
            # Too early to be worth Hub storage: drop it from the pod instead.
            if [ "$((10#$step))" -lt "$MIN_STEP" ]; then
                echo "[push] $step < MIN_STEP $MIN_STEP — not pushing; freeing $(du -sh "$d" | cut -f1) on the pod"
                echo "$step" >> "$STATE"
                rm -rf "$d"
                continue
            fi
            # Only push once the file has stopped growing, or we ship a half-written checkpoint.
            s1=$(stat -c%s "$w")
            sleep 20
            s2=$(stat -c%s "$w")
            [ "$s1" = "$s2" ] || { echo "[push] $step still being written, skipping this pass"; continue; }

            echo "[push] uploading $step ($(du -sh "$d/pretrained_model" | cut -f1)) ..."
            # Python API rather than the CLI: `huggingface-cli` was renamed to `hf`
            # and the old name now refuses to run, so calling either by name is a
            # hostage to the next rename. upload_folder is stable.
            if "$VENV/python" - "$REPO" "$d/pretrained_model" "$step" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder, step = sys.argv[1], sys.argv[2], sys.argv[3]
HfApi().upload_folder(repo_id=repo, folder_path=folder,
                      path_in_repo=f"checkpoints/{step}",
                      commit_message=f"checkpoint {step}")
PY
            then
                echo "$step" >> "$STATE"
                echo "[push] $step DONE at $(date -Is)"
                "$VENV/python" - "$REPO" "$KEEP" <<'PY'
import sys
from huggingface_hub import HfApi
repo, keep = sys.argv[1], int(sys.argv[2])
api = HfApi()
steps = sorted({f.split("/")[1] for f in api.list_repo_files(repo)
                if f.startswith("checkpoints/") and len(f.split("/")) > 2})
stale = steps[:-keep] if len(steps) > keep else []
for s in stale:
    api.delete_folder(f"checkpoints/{s}", repo_id=repo,
                      commit_message=f"prune checkpoint {s}")
    print(f"[push] pruned {s} from the Hub")
if stale:
    # Deleting an LFS file in a new commit does NOT reclaim its storage — the
    # blob stays in history. Squashing collapses history to one commit, which
    # is what actually frees the space. Safe here: this repo is a checkpoint
    # drop, its history has no value.
    api.super_squash_history(repo_id=repo)
    print(f"[push] squashed history — storage reclaimed, keeping {keep} newest")
PY
            else
                echo "[push] $step FAILED — will retry next pass" >&2
            fi
        done
    fi
    # Stop once training has exited and everything on disk has been pushed.
    if pgrep -f "lerobot-train" >/dev/null; then
        [ "$SEEN_TRAINING" = 1 ] || echo "[push] training is running"
        SEEN_TRAINING=1
    elif [ "$SEEN_TRAINING" = 0 ]; then
        echo "[push] no trainer yet (still setting up?) — waiting, not concluding anything"
    else
        pending=0
        for d in "$OUT"/checkpoints/[0-9]*/; do
            [ -d "$d" ] || continue
            grep -qx "$(basename "$d")" "$STATE" || pending=1
        done
        if [ "$pending" = 0 ]; then
            echo "[push] training finished, all checkpoints pushed at $(date -Is)"
            shut_down
            exit 0
        fi
    fi
    sleep "$INTERVAL"
done
