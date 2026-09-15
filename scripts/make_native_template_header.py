#!/usr/bin/env python3
"""Build a header-only safetensors template from a sharded GR00T N1.7 checkpoint.

export_groot_native.py verifies the source checkpoint against a native template
of the same architecture, and reads only that template's safetensors *header* —
tensor names, shapes and dtypes — never its weights. The Elements payload ships
the template's sidecars but not its 13.8 GB of weights, and the stock
nvidia/GR00T-N1.7-3B checkpoint is sharded rather than a single file, so neither
can be passed to --template directly.

This merges the shard headers into one header-only file that satisfies the
check. It is a reference for the shape comparison, not a loadable model: no
tensor data is written.

Usage:
    python scripts/make_native_template_header.py \
        --snapshot ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/<hash> \
        --output ~/groot/native_template_hdr/model.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct


# safetensors dtype -> bytes per element, for laying out offsets consistently.
DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True,
                    help="directory holding model-*-of-*.safetensors (or a single model.safetensors)")
    ap.add_argument("--output", required=True, help="header-only safetensors file to write")
    args = ap.parse_args()

    snap = Path(args.snapshot).expanduser()
    out = Path(args.output).expanduser()

    shards = sorted(snap.glob("model-*-of-*.safetensors")) or sorted(snap.glob("model.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors files in {snap}")

    merged: dict[str, dict] = {}
    for shard in shards:
        h = header(shard)
        print(f"{shard.name}: {len(h)} tensors")
        merged.update(h)

    offset = 0
    for name, entry in merged.items():
        count = 1
        for dim in entry["shape"]:
            count *= dim
        size = count * DTYPE_SIZE[entry["dtype"]]
        merged[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                        "data_offsets": [offset, offset + size]}
        offset += size

    blob = json.dumps(merged, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # the header must be 8-byte aligned
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(struct.pack("<Q", len(blob)) + blob)
    print(f"wrote {out} — {len(merged)} tensors, header only ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
