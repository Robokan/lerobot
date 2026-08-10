#!/usr/bin/env bash
# Finetune pi0.5 on the simulated cube-pick dataset.
#
# Generate data first, e.g. 300 successful episodes with all 3 cameras:
#   MUJOCO_GL=egl python scripts/random_cube_pick.py --no-viewer \
#     --record local/openarm_sim_cube --episodes 300 --cameras all --seed 1
#
# pi0.5 consumes exactly 3 views (base + two wrists) — same layout as the
# real chocolate runs, so this checkpoint drops into the existing eval
# tooling (run_chocolate_policy*.sh on the flash-rt branch).
set -euo pipefail

DATASET="${DATASET:-local/openarm_sim_cube}"
OUT="${OUT:-outputs/pi05_sim_cube}"
# pi0.5 consumes 3 views. For a chest-only dataset (--cameras chest), pad the
# two missing views:  EMPTY_CAMERAS=2 DATASET=local/..._chest bash $0

lerobot-train \
  --dataset.repo_id="${DATASET}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
  --policy.n_action_steps=10 \
  --policy.empty_cameras="${EMPTY_CAMERAS:-0}" \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="${OUT}" \
  --job_name=pi05_sim_cube \
  --batch_size="${BATCH:-32}" \
  --steps="${STEPS:-30000}" \
  --save_freq=5000 \
  --log_freq=100 \
  "$@"
