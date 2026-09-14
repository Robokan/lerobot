#!/usr/bin/env python3
"""Convert a lerobot GR00T checkpoint into a native Isaac-GR00T checkpoint.

The TensorRT toolchain (export_sim_cube.py -> build_trt_pipeline.py ->
sim_cube_trt_server.py) loads the model with Isaac-GR00T's
``Gr00tN1d7.from_pretrained``, which wants a plain native checkpoint. lerobot
saves the same weights wrapped in its policy class. The two differ in exactly
two ways:

  1. every tensor is prefixed ``_groot_model.``;
  2. lerobot ties the unused Qwen LM head to the token embedding and therefore
     saves the tensor once (``backbone.model.lm_head.weight``), while the
     native checkpoint stores both names.

Everything else — 1030 tensors, identical names and shapes, fp32 on disk — is
the same, so the conversion is a rename plus one alias.

The non-weight files (config.json, embodiment_id.json, processor_config.json,
statistics.json) describe the architecture and the embodiment, not the trained
weights, so they are copied from a reference native checkpoint of the same
architecture. They are also not used at inference in this deployment:
normalization and packing run on the lerobot side and the server receives an
already-preprocessed batch (see sim_cube_trt_server.py's docstring). The script
refuses to run if the reference's tensor names don't match the source, which is
what would catch a genuine architecture difference.

Usage:
    python scripts/export_groot_native.py \
        --checkpoint outputs/groot_color_1cam/checkpoints/030000/pretrained_model \
        --output ~/sparkpack/Isaac-GR00T-n17/checkpoints/groot_color_1cam_native
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

PREFIX = "_groot_model."
TIED_SOURCE = "backbone.model.lm_head.weight"
TIED_ALIAS = "backbone.model.model.language_model.embed_tokens.weight"
SIDECAR = ("config.json", "embodiment_id.json", "processor_config.json", "statistics.json")

DEFAULT_TEMPLATE = Path.home() / "sparkpack/Isaac-GR00T-n17/checkpoints/groot_new_sim_cube_300_native"


def safetensors_header(path: Path) -> dict:
    """Tensor names/shapes/dtypes without reading 12 GB of weights."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="a lerobot checkpoint's pretrained_model/ directory")
    ap.add_argument("--output", required=True, help="native checkpoint directory to write")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                    help="native checkpoint to take the non-weight files from")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --output")
    args = ap.parse_args()

    src = Path(args.checkpoint).expanduser()
    out = Path(args.output).expanduser()
    tpl = Path(args.template).expanduser()

    src_weights = src / "model.safetensors"
    if not src_weights.exists():
        raise SystemExit(f"no model.safetensors in {src}")
    if not tpl.is_dir():
        raise SystemExit(
            f"template checkpoint {tpl} not found — pass --template pointing at any native "
            "GR00T N1.7 checkpoint of this architecture"
        )
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists (use --force to overwrite)")
        shutil.rmtree(out)

    # Verify the architecture matches the template BEFORE spending minutes on I/O.
    src_hdr = safetensors_header(src_weights)
    tpl_hdr = safetensors_header(tpl / "model.safetensors")
    renamed = {k[len(PREFIX):]: v for k, v in src_hdr.items() if k.startswith(PREFIX)}
    if len(renamed) != len(src_hdr):
        raise SystemExit(f"{len(src_hdr) - len(renamed)} tensors lack the '{PREFIX}' prefix — "
                         "is this really a lerobot GR00T checkpoint?")
    renamed.setdefault(TIED_ALIAS, renamed.get(TIED_SOURCE))
    missing = sorted(set(tpl_hdr) - set(renamed))
    extra = sorted(set(renamed) - set(tpl_hdr))
    if missing or extra:
        raise SystemExit(
            "checkpoint does not match the template architecture.\n"
            f"  missing {len(missing)}: {missing[:5]}\n  extra {len(extra)}: {extra[:5]}"
        )
    bad = [k for k in tpl_hdr if renamed[k]["shape"] != tpl_hdr[k]["shape"]]
    if bad:
        raise SystemExit(f"{len(bad)} tensors differ in shape from the template, e.g. {bad[:5]}")
    print(f"architecture matches {tpl.name}: {len(tpl_hdr)} tensors")

    import torch
    from safetensors.torch import load_file, save_file

    print(f"loading {src_weights} ...", flush=True)
    state = load_file(str(src_weights))
    native = {k[len(PREFIX):]: v for k, v in state.items()}
    if TIED_ALIAS not in native:
        # lerobot saved the tied weight once; the native loader expects both names.
        native[TIED_ALIAS] = native[TIED_SOURCE].clone()
        print(f"untied {TIED_SOURCE} -> {TIED_ALIAS}")

    out.mkdir(parents=True)
    print(f"writing {out}/model.safetensors ({sum(v.numel() * v.element_size() for v in native.values()) / 1e9:.1f} GB) ...",
          flush=True)
    save_file(native, str(out / "model.safetensors"), metadata={"format": "pt"})
    del native, state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for name in SIDECAR:
        srcf = tpl / name
        if srcf.exists():
            shutil.copy2(srcf, out / name)
            print(f"copied {name} from {tpl.name}")
        else:
            print(f"WARNING: {name} missing from the template")

    print(f"\ndone: {out}")


if __name__ == "__main__":
    main()
