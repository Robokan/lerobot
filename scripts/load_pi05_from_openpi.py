"""
Load a lerobot PI05 policy that was converted from an openpi-PT runtime-LoRA
checkpoint, installing the LoRA forward patches automatically.

Usage as a CLI for quick smoke tests:

    python scripts/load_pi05_from_openpi.py \
        --policy-dir /path/to/converted_dir \
        --dataset-root /path/to/lerobot/dataset

Or as a library:

    from scripts.load_pi05_from_openpi import load_pi05_with_runtime_lora
    policy, preprocessor, postprocessor = load_pi05_with_runtime_lora(
        policy_dir, ds_meta=meta, device="cuda", dtype="bfloat16",
    )
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.lora_runtime import (
    install_runtime_lora,
    load_lora_from_safetensors,
)


def load_pi05_with_runtime_lora(
    policy_dir: str | Path,
    *,
    ds_meta: LeRobotDatasetMetadata,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    """Load PI05Policy + processors, then install runtime LoRA if present.

    Returns (policy, preprocessor, postprocessor). The policy is in eval mode
    and ready for `select_action` / `predict_action_chunk`.
    """
    policy_dir = Path(policy_dir)

    cfg = PreTrainedConfig.from_pretrained(policy_dir)
    cfg.pretrained_path = str(policy_dir)
    cfg.device = device
    cfg.dtype = dtype
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.eval()

    marker = policy_dir / "lora_runtime_marker.json"
    lora_file = policy_dir / "lora.safetensors"
    if marker.exists() and lora_file.exists():
        info = json.loads(marker.read_text())
        if info.get("runtime_lora"):
            print(f"  installing runtime LoRA from {lora_file.name}")
            lora_sd = load_lora_from_safetensors(str(lora_file))
            n = install_runtime_lora(policy, lora_sd, base_path="model")
            print(f"  patched {n} projection modules")
            # Move freshly-attached buffers to the policy device + correct dtype
            policy = policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    return policy, preprocessor, postprocessor


def _run_smoke(policy_dir: Path, ds_root: Path, n_frames: int = 4) -> None:
    print(f"[1/4] loading dataset {ds_root.name}")
    meta = LeRobotDatasetMetadata(repo_id=ds_root.name, root=ds_root)
    ds = LeRobotDataset(repo_id=ds_root.name, root=ds_root)
    print(f"      episodes={meta.total_episodes}  frames={meta.total_frames}  fps={meta.fps}")

    print(f"\n[2/4] loading policy + runtime LoRA from {policy_dir}")
    t0 = time.time()
    policy, preprocessor, postprocessor = load_pi05_with_runtime_lora(
        policy_dir, ds_meta=meta, device="cuda", dtype="bfloat16",
    )
    print(f"      ready in {time.time()-t0:.1f}s")

    print(f"\n[3/4] testing on first {n_frames} frames of episode 0 "
          f"(reset between frames for fresh chunk prediction)")
    for idx in range(n_frames):
        sample = ds[idx]
        batch: dict = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            elif isinstance(v, str):
                batch[k] = [v]
            else:
                batch[k] = [v]
        if "task" not in batch or not isinstance(batch["task"][0], str):
            batch["task"] = ["put the chocolate bars in the container"]

        policy.reset()
        t1 = time.time()
        with torch.no_grad():
            preproc_batch = preprocessor(batch)
            action = policy.select_action(preproc_batch)
            final = postprocessor(action)
        dt = time.time() - t1

        pred = final[0].cpu().float().numpy()
        targ = sample["action"].numpy()
        state = sample["observation.state"].numpy()
        err = pred - targ
        print(f"\n  --- frame {idx} ({dt*1000:.0f} ms) ---")
        print(f"    state[:8]  : {np.round(state[:8], 3)}")
        print(f"    state[8:]  : {np.round(state[8:], 3)}")
        print(f"    target[:8] : {np.round(targ[:8], 3)}")
        print(f"    target[8:] : {np.round(targ[8:], 3)}")
        print(f"    pred[:8]   : {np.round(pred[:8], 3)}")
        print(f"    pred[8:]   : {np.round(pred[8:], 3)}")
        print(f"    abs err:  mean={np.abs(err).mean():.4f}  max={np.abs(err).max():.4f}")
        print(f"    ||pred-state||={np.linalg.norm(pred-state):.4f}  "
              f"||targ-state||={np.linalg.norm(targ-state):.4f}")

    print("\n[4/4] full chunk vs full chunk (50 frames) for episode 0 frame 0")
    sample = ds[0]
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = [v]
    if "task" not in batch or not isinstance(batch["task"][0], str):
        batch["task"] = ["put the chocolate bars in the container"]

    with torch.no_grad():
        preproc_batch = preprocessor(batch)
        chunk = policy.predict_action_chunk(preproc_batch)  # (1, T, A)

    # Compare against the next T target actions from the dataset (same episode)
    T = chunk.shape[1]
    targets = torch.stack([ds[i]["action"] for i in range(T)])
    # Push chunk through unnormalizer one frame at a time
    final_chunk = []
    for t in range(T):
        a = postprocessor(chunk[:, t])
        final_chunk.append(a[0].cpu().float())
    final_chunk = torch.stack(final_chunk)
    err = (final_chunk - targets).abs()
    print(f"      chunk shape: {chunk.shape}  ->  unnormalized {final_chunk.shape}")
    print(f"      vs ground truth chunk:")
    print(f"        mean abs err: {err.mean().item():.4f}")
    print(f"        max  abs err: {err.max().item():.4f}")
    print(f"        per-dim mean: {err.mean(0).numpy().round(3)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=3)
    args = p.parse_args()
    _run_smoke(args.policy_dir, args.dataset_root, n_frames=args.n_frames)


if __name__ == "__main__":
    main()
