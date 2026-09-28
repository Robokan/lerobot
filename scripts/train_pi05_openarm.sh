#!/usr/bin/env bash
# Finetune pi0.5 (pi05) on a bimanual OpenArm dataset recorded with
# scripts/record_openarm.sh.
#
# Memory notes for this machine (RTX 4090, 24 GB per GPU): pi0.5 is a ~3 B
# parameter model (PaliGemma 2B VLM + 300M action expert). A full finetune
# with AdamW does not fit in 24 GB, so the defaults below train the action
# expert + projections only (train_expert_only=true, VLM frozen) with
# gradient checkpointing and bfloat16 — the documented reduced-memory recipe.
# Override TRAIN_EXPERT_ONLY=false only on a bigger GPU.
#
# The dataset must have quantile stats for pi0.5's default normalization.
# Datasets recorded with current lerobot have them; if training aborts with a
# quantile-stats error, either run
#   python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py --repo-id=<repo_id>
# or add:
#   --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
#
# Usage:
#   REPO_ID=eric/openarm_pick_cube bash scripts/train_pi05_openarm.sh
#
# Optional env overrides (defaults in parentheses):
#   STEPS (30000)  BATCH_SIZE (4)  JOB_NAME (pi05_openarm)
#   PRETRAINED (lerobot/pi05_base)  TRAIN_EXPERT_ONLY (true)
#   WANDB (false)  DEVICE (cuda)  EXTRA_ARGS (appended verbatim)

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ID="${REPO_ID:?Set REPO_ID to the dataset to train on, e.g. REPO_ID=eric/openarm_pick_cube}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
JOB_NAME="${JOB_NAME:-pi05_openarm}"
PRETRAINED="${PRETRAINED:-lerobot/pi05_base}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
WANDB="${WANDB:-false}"
DEVICE="${DEVICE:-cuda}"

# shellcheck disable=SC1091
source .venv/bin/activate

# shellcheck disable=SC2086
exec lerobot-train \
    --dataset.repo_id="$REPO_ID" \
    --policy.type=pi05 \
    --policy.pretrained_path="$PRETRAINED" \
    --policy.device="$DEVICE" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
    --policy.push_to_hub=false \
    --output_dir="outputs/train/$JOB_NAME" \
    --job_name="$JOB_NAME" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --wandb.enable="$WANDB" \
    ${EXTRA_ARGS:-}
