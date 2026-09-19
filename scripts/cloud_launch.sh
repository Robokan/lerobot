#!/usr/bin/env bash
# Take a recorded LeRobot dataset to a running GR00T training job on a fresh
# Runpod pod, from this workstation, in one command.
#
#   scripts/cloud_launch.sh \
#       --dataset    local/openarm_caddy6_pick_all_300 \
#       --hf-dataset evaughan69/openarm_caddy6_pick_all_300 \
#       --hf-ckpt    evaughan69/groot_caddy6_3cam \
#       --out        groot_caddy6_3cam \
#       --steps 70000 --min-step 40000
#
# PREREQUISITE, once per account: a Runpod SECRET named HF_TOKEN holding a
# HuggingFace WRITE token, created at console.runpod.io/user/secrets. The pod is
# created referencing it as {{ RUNPOD_SECRET_HF_TOKEN }}, so the token goes from
# the console straight into the container and never touches this machine, a
# command line, or a log. Without it the run stops before training starts.
#
# See CLOUD_TRAINING.md for the whole procedure, the costs, and what each of the
# failure modes looks like.
set -uo pipefail

SPARK="${SPARK:-$HOME/sparkpack}"
RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
PY="${PY:-$SPARK/lerobot/.venv/bin/python}"

DATASET=""            # local repo id, e.g. local/openarm_caddy6_pick_all_300
HF_DATASET=""         # Hub repo to push it to / train from
HF_CKPT=""            # Hub repo the pod pushes checkpoints to
OUT=""                # output dir name under outputs/
STEPS=70000
MIN_STEP=40000        # checkpoints below this are skipped and deleted on the pod
KEEP=4                # newest checkpoints kept on the Hub
BATCH=16
AUGMENT=0
POLICY=groot          # groot | pi05
LORA=0                # pi05 only: LoRA instead of a full finetune
GPUS=("NVIDIA H100 80GB HBM3" "NVIDIA H100 NVL" "NVIDIA H100 PCIe")
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
VOLUME=200
MAX_HOURS=11          # watchdog's unconditional stop
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --hf-dataset) HF_DATASET="$2"; shift 2 ;;
        --hf-ckpt)    HF_CKPT="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --min-step)   MIN_STEP="$2"; shift 2 ;;
        --keep)       KEEP="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --augment)    AUGMENT=1; shift ;;
        --policy)     POLICY="$2"; shift 2 ;;
        --lora)       LORA=1; shift ;;
        --volume)     VOLUME="$2"; shift 2 ;;
        --max-hours)  MAX_HOURS="$2"; shift 2 ;;
        --dry-run)    DRY=1; shift ;;
        -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for v in DATASET HF_DATASET HF_CKPT OUT; do
    [ -n "${!v}" ] || { echo "missing --${v,,} " | tr '_' '-' >&2; exit 2; }
done
STATE="${STATE:-$HOME/.cache/cloud_launch}"
mkdir -p "$STATE"
say() { echo "$(date +%H:%M:%S) $*"; }

# --- 1. dataset and code onto the Hub -------------------------------------
# Everything the pod needs travels via the Hub, never ssh. Measured on this
# link: workstation -> Hub 4.7 MB/s, Hub -> pod datacenter speed (129 MB in 5 s),
# workstation -> pod over ssh 24 KB/s on a bad pod. Doing this BEFORE the pod
# exists also means nothing is billing while the slow leg runs.
say "pushing the dataset to $HF_DATASET (skips what is already there)"
"$PY" "$SPARK/lerobot/scripts/hf_upload_dataset.py" "$DATASET" --to "$HF_DATASET" || exit 1

# lerobot resolves a Hub dataset by a tag matching codebase_version in info.json.
"$PY" - "$DATASET" "$HF_DATASET" <<'PYEOF' || exit 1
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi
local, repo = sys.argv[1], sys.argv[2]
root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
tag = json.loads((root / local / "meta" / "info.json").read_text())["codebase_version"]
api = HfApi()
if tag not in [t.name for t in api.list_repo_refs(repo, repo_type="dataset").tags]:
    api.create_tag(repo, tag=tag, repo_type="dataset")
    print(f"tagged {repo} {tag}")
else:
    print(f"{repo} already tagged {tag}")
PYEOF

BUNDLE="$STATE/lerobot.bundle"
say "bundling lerobot @ $(git -C "$SPARK/lerobot" rev-parse --short HEAD) and pushing it to the Hub"
git -C "$SPARK/lerobot" bundle create "$BUNDLE" main 2>/dev/null || exit 1
"$PY" - "$HF_DATASET" "$BUNDLE" <<'PYEOF' || exit 1
import sys
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=sys.argv[2], path_in_repo="code/lerobot.bundle",
                    repo_id=sys.argv[1], repo_type="dataset",
                    commit_message="lerobot bundle for the pod")
print("bundle on the Hub")
PYEOF

if [ "$DRY" = 1 ]; then say "--dry-run: stopping before anything bills"; exit 0; fi

# --- 2. the pod ------------------------------------------------------------
# H100 stock is usually "Low" in every datacenter, so try the alternatives.
POD=""
for gpu in "${GPUS[@]}"; do
    for attempt in 1 2 3; do
        say "creating a pod on '$gpu' (attempt $attempt)"
        out=$("$RUNPODCTL" pod create --name "${OUT//_/-}" --image "$IMAGE" \
                --gpu-id "$gpu" --cloud-type SECURE \
                --container-disk-in-gb 30 --volume-in-gb "$VOLUME" \
                --volume-mount-path /workspace --ports "22/tcp" \
                --env '{"HF_TOKEN":"{{ RUNPOD_SECRET_HF_TOKEN }}"}' \
                --wait --wait-timeout 12m 2>&1)
        POD=$(echo "$out" | grep -oE '"id": "[^"]*"' | head -1 | cut -d'"' -f4)
        [ -n "$POD" ] && break 2
        echo "$out" | tail -2
        sleep 60
    done
done
[ -n "$POD" ] || { say "no pod could be created on any GPU — nothing spent"; exit 1; }
say "pod $POD is up — BILLING STARTS NOW"
echo "$POD" > "$STATE/pod_id"

read -r IP PORT KEY < <("$RUNPODCTL" ssh info "$POD" | "$PY" -c \
    'import json,sys;d=json.load(sys.stdin);print(d["ip"],d["port"],d["key"]["path"] if isinstance(d.get("key"),dict) else d["path"])' 2>/dev/null) \
    || read -r IP PORT KEY < <("$RUNPODCTL" ssh info "$POD" | tr ',' '\n' | grep -oE '"(ip|port|path)": ?"?[^",]*' | cut -d: -f2- | tr -d ' "' | tr '\n' ' ')
[ -n "${IP:-}" ] || { say "could not read ssh info for $POD — STOP IT YOURSELF"; exit 1; }
printf '%s\n' "$IP" "$PORT" "$KEY" > "$STATE/pod_ssh"
# Runpod keeps the container env (including the substituted HF_TOKEN) in
# /etc/rp_environment, sourced only by an INTERACTIVE shell. A plain
# `ssh host cmd` never reads it, so every command sources it first.
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$IP . /etc/rp_environment;"
SCP="scp -i $KEY -P $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

say "checking the secret reached the container"
$SSH 'test -n "$HF_TOKEN" && [ "${HF_TOKEN#*RUNPOD_SECRET}" = "$HF_TOKEN" ]' || {
    say "HF_TOKEN missing or unsubstituted — create the Runpod secret named HF_TOKEN. Stopping the pod."
    "$RUNPODCTL" stop pod "$POD"; exit 3; }

say "pulling code and building the venv (~10 min)"
$SCP "$SPARK/lerobot/scripts/cloud_train_setup.sh" "$SPARK/lerobot/scripts/pod_push_checkpoints.sh" \
     "root@$IP:/workspace/" || exit 1
# -b main matters: a bundle made from a branch carries no HEAD ref, and a plain
# clone of one leaves an EMPTY working tree.
$SSH "set -e
cd /workspace
curl -sL -H 'Authorization: Bearer \$HF_TOKEN' \
  'https://huggingface.co/datasets/$HF_DATASET/resolve/main/code/lerobot.bundle' -o lerobot.bundle
rm -rf lerobot && git clone -q -b main /workspace/lerobot.bundle lerobot
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\$HOME/.local/bin:\$PATH
cd lerobot && uv sync --locked --extra dataset --extra training --extra core_scripts --extra groot 2>&1 | tail -2" || exit 1

say "proving the token can read the dataset and write checkpoints"
$SSH "cd /workspace/lerobot && .venv/bin/python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo('$HF_CKPT', private=True, exist_ok=True)
n = len(api.list_repo_files('$HF_DATASET', repo_type='dataset'))
print(f\"hf {api.whoami()['name']} | write OK | dataset {n} files\")
PY" || { say "token cannot do the job — stopping the pod"; "$RUNPODCTL" stop pod "$POD"; exit 3; }

# --- 3. training, pusher, watchdog ----------------------------------------
AUG_FLAG=""; [ "$AUGMENT" = 1 ] && AUG_FLAG="--augment"
POL_FLAGS="--policy $POLICY"; [ "$LORA" = 1 ] && POL_FLAGS="$POL_FLAGS --lora"
say "starting training: $STEPS steps, policy $POLICY$([ "$LORA" = 1 ] && echo ' (LoRA)')"
# From /workspace, not /workspace/lerobot: cloud_train_setup.sh expects the
# bundle and the clone as siblings of its working directory.
$SSH "cd /workspace && export PATH=\$HOME/.local/bin:\$PATH && \
      nohup bash /workspace/cloud_train_setup.sh --scratch --repo-id '$HF_DATASET' \
        --dataset /pull-from-hub --out outputs/$OUT --steps $STEPS --batch $BATCH \
        --save-freq 1000 $AUG_FLAG $POL_FLAGS > /workspace/train.log 2>&1 & sleep 5; echo ok" || exit 1

say "starting the checkpoint pusher (skip < $MIN_STEP, keep $KEEP, stop the pod when done)"
$SSH "cd /workspace && MIN_STEP=$MIN_STEP ON_DONE=stop \
      nohup bash /workspace/pod_push_checkpoints.sh '$HF_CKPT' \
      /workspace/lerobot/outputs/$OUT 180 $KEEP $MIN_STEP > /workspace/push.log 2>&1 & sleep 2; echo ok"

FINAL=$(printf '%06d' "$STEPS")
say "starting the local watchdog (stops the pod on checkpoint $FINAL, or after ${MAX_HOURS}h)"
setsid nohup "$SPARK/lerobot/scripts/pod_autostop_watchdog.sh" "$POD" "$HF_CKPT" "$FINAL" "$MAX_HOURS" \
    > "$STATE/watchdog.log" 2>&1 < /dev/null &
disown

cat <<EOF

pod        $POD  ($IP:$PORT)
train log  ssh -i $KEY -p $PORT root@$IP 'tail -f /workspace/train.log'
checkpoints https://huggingface.co/$HF_CKPT
watchdog   $STATE/watchdog.log

The pod stops itself when checkpoint $FINAL is on the Hub. Verify with
'$RUNPODCTL pod list' afterwards — a stopped pod still bills for its volume.
EOF
