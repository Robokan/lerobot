#!/usr/bin/env bash
# Bring a fresh cloud GPU box (RunPod H100 80GB, x86, CUDA 12.8+) up to a
# running colour-sort training job.
#
# Upload these to the pod first (see DEPLOY_TENSORRT.md for how the bundles are
# made):
#   lerobot.bundle                          self-contained clone, no network needed
#   openarm_color_sort_all_300/             the 3-camera dataset (~3 GB)
#   050000_3cam/                            optional warm-start checkpoint (12.6 GB)
#
# Then:
#   bash cloud_train_setup.sh --warm-start ./050000_3cam
#   bash cloud_train_setup.sh --scratch            # from the base model instead
set -euo pipefail

BUNDLE="${BUNDLE:-./lerobot.bundle}"
DATASET_SRC="${DATASET_SRC:-./openarm_color_sort_all_300}"
REPO_ID="local/openarm_color_sort_all_300"
OUT="${OUT:-outputs/groot_color_3cam_aug}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-16}"
WARM=""
MODE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --warm-start) WARM="$(readlink -f "$2")"; MODE=warm; shift 2 ;;
        --scratch)    MODE=scratch; shift ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
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
# oneDNN's aarch64 JIT. x86 has no such problem, so it goes on here — EXCEPT
# the affine transform.
#
# RandomAffine rotates +-5 deg and translates up to 5% of the frame while
# leaving the action labels untouched. For a fixed-camera manipulation policy
# that is not augmentation, it is label noise: it teaches the model that the
# same action is correct for a cube that appears somewhere else. This task
# already fails on grasp precision, and needs the gripper placed within about a
# centimetre, so a 5% frame shift is far larger than the error we are trying to
# remove. Colour and sharpness jitter change appearance without moving
# anything, which is the kind of invariance we actually want.
AUG=(
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.tfs.affine.weight=0.0
)

set -x
uv run lerobot-train \
    "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$REPO_ID" \
    "${AUG[@]}" \
    --output_dir="$OUT" \
    --job_name="$(basename "$OUT")" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --save_freq=1000 \
    --checkpoint_keep_every=10000 \
    --log_freq=100
