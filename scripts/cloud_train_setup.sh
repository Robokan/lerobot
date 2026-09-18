#!/usr/bin/env bash
# Bring a fresh cloud GPU box (RunPod H100 80GB, x86, CUDA 12.8+) up to a
# running GR00T training job.
#
# Upload these to the pod first (see DEPLOY_TENSORRT.md for how the bundles are
# made):
#   lerobot.bundle                          self-contained clone, no network needed
#   <dataset>/                              the 3-camera dataset (~3 GB)
#   <checkpoint>/                           optional warm-start checkpoint (12.6 GB)
#
# Then, e.g. the caddy picker from scratch:
#   bash cloud_train_setup.sh --scratch \
#       --dataset ./openarm_caddy6_pick_all_300 \
#       --repo-id local/openarm_caddy6_pick_all_300 \
#       --out outputs/groot_caddy6_3cam --steps 50000
#   bash cloud_train_setup.sh --warm-start ./050000_3cam
set -euo pipefail

BUNDLE="${BUNDLE:-./lerobot.bundle}"
DATASET_SRC="${DATASET_SRC:-./openarm_caddy6_pick_all_300}"
REPO_ID="${REPO_ID:-local/openarm_caddy6_pick_all_300}"
OUT="${OUT:-outputs/groot_caddy6_3cam}"
STEPS="${STEPS:-50000}"
BATCH="${BATCH:-16}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
# Image augmentation is OFF by default. It exists to bridge sim-to-real
# lighting, and this is a pure simulation study — and on the caddy task it
# would be actively harmful, because the PAD COLOUR is what the prompt names:
# at the default hue jitter of +-18 deg orange drifts toward red and yellow,
# purple toward pink, i.e. the transform relabels the task.
AUG_ON=0
WARM=""
MODE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --warm-start) WARM="$(readlink -f "$2")"; MODE=warm; shift 2 ;;
        --scratch)    MODE=scratch; shift ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --dataset)    DATASET_SRC="$2"; shift 2 ;;
        --repo-id)    REPO_ID="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --save-freq)  SAVE_FREQ="$2"; shift 2 ;;
        --augment)    AUG_ON=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODE" ] || { echo "pass --warm-start <dir> or --scratch" >&2; exit 2; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "no GPU visible — wrong pod image?" >&2; exit 1; }

# --- code ------------------------------------------------------------------
if [ ! -d lerobot ]; then
    [ -f "$BUNDLE" ] || { echo "no bundle at $BUNDLE" >&2; exit 1; }
    git clone "$BUNDLE" lerobot
fi
cd lerobot
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked

# --- dataset ---------------------------------------------------------------
# lerobot resolves datasets by repo id under this cache root.
CACHE="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
if [ ! -f "$CACHE/$REPO_ID/meta/info.json" ]; then
    mkdir -p "$CACHE/$(dirname "$REPO_ID")"
    cp -r "$DATASET_SRC" "$CACHE/$REPO_ID"
fi
echo "dataset: $(du -sh "$CACHE/$REPO_ID" | cut -f1) at $CACHE/$REPO_ID"

# --- policy source ---------------------------------------------------------
if [ "$MODE" = warm ]; then
    POLICY_ARGS=(--policy.path="$WARM")
    echo "warm start from $WARM"
else
    POLICY_ARGS=(
        --policy.type=groot
        --policy.base_model_path=nvidia/GR00T-N1.7-3B
        --policy.embodiment_tag=new_embodiment
        --policy.chunk_size=16
        --policy.n_action_steps=16
        --policy.use_relative_actions=true
        --policy.relative_exclude_joints='["gripper"]'
        --policy.use_bf16=true
        --policy.push_to_hub=false
        --policy.device=cuda
    )
    echo "training from the base model"
fi

# --- augmentation ----------------------------------------------------------
# Off on the Spark because the sharpness jitter's CPU depthwise conv crashes
# oneDNN's aarch64 JIT crashes on the sharpness jitter, so the Spark never ran
# augmentation; x86 has no such problem. --augment turns the tfs back on minus
# the affine transform, which rotates +-5 deg and translates up to 5% of the
# frame while leaving the action labels untouched: for a fixed-camera
# manipulation policy that is not augmentation, it is label noise.
if [ "$AUG_ON" = 1 ]; then
    AUG=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.tfs.affine.weight=0.0
    )
    echo "image augmentation ON (affine excluded)"
else
    AUG=(--dataset.image_transforms.enable=false)
    echo "image augmentation OFF (simulation study; pad colour is the task cue)"
fi

set -x
uv run lerobot-train \
    "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$REPO_ID" \
    "${AUG[@]}" \
    --output_dir="$OUT" \
    --job_name="$(basename "$OUT")" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --checkpoint_keep_every=10000 \
    --log_freq=100
