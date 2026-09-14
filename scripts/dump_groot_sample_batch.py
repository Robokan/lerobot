#!/usr/bin/env python3
"""Capture one preprocessed GR00T batch from lerobot's inference pipeline.

TensorRT engines are built for static shapes, so the ONNX export needs a real
example of what the model will be handed at runtime. Our runtime preprocessing
lives in lerobot (image resize/patching, prompt tokenisation, state packing),
not in Isaac-GR00T's native processor, and the two produce different sequence
lengths — export from the native processor and the engines silently expect a
tensor shape the eval will never send.

This runs a single dataset frame through the checkpoint's own saved processor
pipeline and saves exactly the dict ``predict_action_chunk`` would pass to the
model, for ``export_sim_cube.py --sample-batch``.

Only the processor is loaded, never the 12 GB of weights, so this is cheap and
safe to run next to a training job.

Usage:
    python scripts/dump_groot_sample_batch.py \
        --checkpoint outputs/groot_color_1cam/checkpoints/030000/pretrained_model \
        --dataset local/openarm_color_sort_chest_300 \
        --out /tmp/color_sample_batch.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# The model-side key filter (GrootPolicy._filter_groot_inputs). Kept here as a
# literal so no policy weights have to be loaded just to learn the key set; the
# --check flag verifies it still agrees with the policy class.
GROOT_INPUT_KEYS = {
    "state", "state_mask", "action_mask", "embodiment_id",
    "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
    "mm_token_type_ids", "pixel_values_videos", "video_grid_thw",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a lerobot checkpoint's pretrained_model/ dir")
    ap.add_argument("--dataset", required=True, help="the repo_id the checkpoint was trained on")
    ap.add_argument("--out", required=True, help="path of the .pt to write")
    ap.add_argument("--frame", type=int, default=0, help="dataset frame index to use")
    ap.add_argument("--task", default=None,
                    help="prompt to tokenise (default: the dataset's own task string)")
    ap.add_argument("--device", default="cpu",
                    help="device the processor emits tensors on; cpu avoids touching a busy GPU")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference

    ds = LeRobotDataset(args.dataset)
    sample = ds[args.frame]
    task = args.task or sample.get("task") or ds.meta.tasks.index[0]
    print(f"dataset {args.dataset}: frame {args.frame}, task {task!r}")

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = args.device
    pre, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.checkpoint,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    # Rebuild the raw observation the robot would hand in: state vector plus one
    # HWC uint8 frame per camera the checkpoint declares.
    obs: dict = {}
    for key, feat in cfg.input_features.items():
        if key not in sample:
            raise SystemExit(f"checkpoint wants {key}, which {args.dataset} does not have")
        v = sample[key]
        if key.startswith("observation.images."):
            img = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW float [0,1] -> HWC uint8
                img = (img.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8)
            obs[key] = img.numpy()
        else:
            obs[key] = (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).numpy()
        print(f"  {key}: {tuple(feat.shape)}")

    with torch.inference_mode():
        batch = pre(prepare_observation_for_inference(obs, torch.device(args.device), task, None))

    dropped = sorted(k for k in batch if k not in GROOT_INPUT_KEYS)
    kept = {k: v for k, v in batch.items() if k in GROOT_INPUT_KEYS and isinstance(v, torch.Tensor)}
    if not kept:
        raise SystemExit(f"processor produced none of the model inputs; got {sorted(batch)}")
    print(f"\nmodel inputs ({len(kept)} tensors; dropped {dropped}):")
    for k, v in sorted(kept.items()):
        print(f"  {k:24s} {tuple(v.shape)} {v.dtype}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in kept.items()}, out)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
