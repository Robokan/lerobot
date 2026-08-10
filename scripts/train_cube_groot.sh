#!/usr/bin/env bash
# Finetune GR00T N1.7 on the simulated cube-pick dataset.
#
# Same dataset as train_cube_pi05.sh. GR00T handles arbitrary camera counts;
# 'chest'-only datasets (--cameras chest) work too.
set -euo pipefail

DATASET="${DATASET:-local/openarm_sim_cube}"
OUT="${OUT:-outputs/groot_sim_cube}"

# Image augmentation is off by default: the default transform set includes a
# sharpness jitter whose CPU depthwise conv sporadically crashes oneDNN's
# aarch64 JIT ("Xbyak::Error: label is too far") in spawned dataloader
# workers on this Grace machine. Sim renders are visually uniform anyway.
# To re-enable everything EXCEPT sharpness on other hardware, pass
# --dataset.image_transforms.enable=true and a custom tfs dict.
lerobot-train \
  --dataset.repo_id="${DATASET}" \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.use_bf16=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir="${OUT}" \
  --job_name=groot_sim_cube \
  --batch_size="${BATCH:-16}" \
  --steps="${STEPS:-20000}" \
  --save_freq=5000 \
  --log_freq=100 \
  "$@"
