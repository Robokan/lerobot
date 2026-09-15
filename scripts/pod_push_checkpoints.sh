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
set -uo pipefail

REPO="${1:?usage: pod_push_checkpoints.sh <hf-user>/<repo-name> [output-dir] [interval-s] [keep-n]}"
OUT="${2:-/workspace/lerobot/outputs/groot_color_3cam_aug}"
INTERVAL="${3:-120}"
KEEP="${4:-2}"          # how many of the newest checkpoints to keep on the Hub
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

echo "[push] watching $OUT every ${INTERVAL}s -> $REPO"
while true; do
    if [ -d "$OUT/checkpoints" ]; then
        for d in "$OUT"/checkpoints/[0-9]*/; do
            [ -d "$d" ] || continue
            step=$(basename "$d")
            grep -qx "$step" "$STATE" && continue
            [ -f "$d/pretrained_model/model.safetensors" ] || continue
            # Only push once the file has stopped growing, or we ship a half-written checkpoint.
            s1=$(stat -c%s "$d/pretrained_model/model.safetensors")
            sleep 20
            s2=$(stat -c%s "$d/pretrained_model/model.safetensors")
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
    if ! pgrep -f "lerobot-train" >/dev/null; then
        pending=0
        for d in "$OUT"/checkpoints/[0-9]*/; do
            [ -d "$d" ] || continue
            grep -qx "$(basename "$d")" "$STATE" || pending=1
        done
        [ "$pending" = 0 ] && { echo "[push] training finished, all checkpoints pushed"; exit 0; }
    fi
    sleep "$INTERVAL"
done
