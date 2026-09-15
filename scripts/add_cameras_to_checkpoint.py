#!/usr/bin/env python3
"""Re-point a trained GR00T checkpoint at a different set of cameras.

Warm-starting a 1-camera checkpoint on a 3-camera dataset does NOT pick up the
new views. Two things pin the camera list to what the checkpoint was trained
with, and both survive `--policy.path`:

  * ``config.json``'s ``input_features`` — the training factory only fills
    these from the dataset ``if not cfg.input_features``, i.e. never for a
    checkpoint;
  * the saved preprocessor's ``video_modality_keys`` — the GR00T packer feeds
    exactly those views and logs the rest as "unused".

The run then trains happily on the old camera and the extra footage is simply
never seen. This writes a patched checkpoint directory that lists the cameras
you ask for, symlinking the weights so nothing is duplicated.

The model itself needs no surgery: the vision tower and LLM are frozen, and
each view is encoded by the same frozen ViT through the same projector, so more
cameras mean more tokens in the same embedding space — no weight shape changes.

Usage:
    python scripts/add_cameras_to_checkpoint.py \
        --checkpoint outputs/groot_color_1cam/checkpoints/050000/pretrained_model \
        --cameras ego left_wrist right_wrist \
        --output outputs/groot_color_1cam/checkpoints/050000_3cam

Then train with --policy.path=<output>. Verify before committing GPU-days:
    python scripts/dump_groot_sample_batch.py --checkpoint <output> \
        --dataset <3-camera repo_id> --out /tmp/check.pt
and confirm pixel_values has 256 patches PER CAMERA.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

OBS_IMAGES = "observation.images"


def set_video_keys(node, cameras: list[str], hits: list[str]) -> None:
    """Rewrite every video modality key list found anywhere in the pipeline config."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "video_modality_keys" and isinstance(value, list):
                node[key] = list(cameras)
                hits.append(key)
            elif key == "video" and isinstance(value, dict) and "modality_keys" in value:
                value["modality_keys"] = list(cameras)
                hits.append("modality_config.video.modality_keys")
            else:
                set_video_keys(value, cameras, hits)
    elif isinstance(node, list):
        for item in node:
            set_video_keys(item, cameras, hits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a trained pretrained_model/ directory")
    ap.add_argument("--cameras", required=True, nargs="+",
                    help="modality names in feed order, e.g. ego left_wrist right_wrist")
    ap.add_argument("--output", required=True, help="patched checkpoint directory to write")
    ap.add_argument("--copy-weights", action="store_true",
                    help="copy model.safetensors instead of symlinking it")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.output).expanduser()
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists (use --force)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # --- config.json: the feature list the policy is built from ---------------
    cfg = json.loads((src / "config.json").read_text())
    old = [k for k in cfg.get("input_features", {}) if k.startswith(OBS_IMAGES)]
    template = cfg["input_features"][old[0]] if old else {"type": "VISUAL", "shape": [3, 480, 640]}
    for key in old:
        cfg["input_features"].pop(key)
    for cam in args.cameras:
        cfg["input_features"][f"{OBS_IMAGES}.{cam}"] = json.loads(json.dumps(template))
    (out / "config.json").write_text(json.dumps(cfg, indent=4))
    print(f"config.json: {old} -> {[f'{OBS_IMAGES}.{c}' for c in args.cameras]}")

    # --- preprocessor: the packer's view list --------------------------------
    pre_path = src / "policy_preprocessor.json"
    pre = json.loads(pre_path.read_text())
    hits: list[str] = []
    set_video_keys(pre, args.cameras, hits)
    if not hits:
        raise SystemExit("no video modality keys found in policy_preprocessor.json — "
                         "the packer would still feed the original cameras")
    (out / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2))
    print(f"policy_preprocessor.json: rewrote {len(hits)} key list(s) -> {args.cameras}")

    # --- everything else rides along -----------------------------------------
    for item in sorted(src.iterdir()):
        if item.name in {"config.json", "policy_preprocessor.json"}:
            continue
        dst = out / item.name
        if item.name == "model.safetensors" and not args.copy_weights:
            dst.symlink_to(item)
            print(f"{item.name}: symlinked ({item.stat().st_size / 1e9:.1f} GB not copied)")
        else:
            shutil.copy2(item, dst)
            print(f"{item.name}: copied")

    print(f"\ndone: {out}")
    print("Verify the packer really feeds every camera before spending GPU time:")
    print(f"  python scripts/dump_groot_sample_batch.py --checkpoint {out} \\")
    print("      --dataset <3-camera repo_id> --out /tmp/check.pt")


if __name__ == "__main__":
    main()
