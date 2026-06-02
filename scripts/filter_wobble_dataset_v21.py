#!/usr/bin/env python
"""Produce a wobble-filtered copy of a **LeRobot v2.1** dataset (per-episode parquet).

This is the v2.1 sibling of ``filter_wobble_dataset.py`` (which only handles the
v3.0 single-concatenated-parquet layout). It applies the *same* zero-phase
Butterworth low-pass (``filtfilt``) to ``observation.state`` and ``action``,
per-episode and per-dimension, so the policy is no longer trained to reproduce
underdamped follower ringing that a stiff deployment arm then chases.

Only ``observation.state`` and ``action`` are filtered. The per-episode stats in
``meta/episodes_stats.jsonl`` are recomputed for those two features (all other
features -- images, timestamp, indices -- are preserved verbatim). Video files
are symlinked (not copied), so the smoothed dataset costs a few MB, not GB.

Use this to smooth the single-prompt v2.1 dataset directly, e.g.:

    python scripts/filter_wobble_dataset_v21.py \
        --src ~/.cache/huggingface/lerobot/local/openarm-chocolate-v4_old \
        --dst ~/.cache/huggingface/lerobot/local/openarm-chocolate-v4-smoothed-v21 \
        --cutoff-hz 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt

# Only these two features carry the wobble and get filtered + restatted.
FILTERED_FEATURES = ["observation.state", "action"]
_EP_RE = re.compile(r"episode_(\d+)\.parquet$")


def _load_info(src: Path) -> dict:
    return json.loads((src / "meta" / "info.json").read_text())


def _butter_lowpass(cutoff_hz: float, fps: float, order: int):
    nyq = fps / 2.0
    wn = cutoff_hz / nyq
    if not 0.0 < wn < 1.0:
        raise ValueError(f"cutoff {cutoff_hz} Hz invalid for fps {fps} (Wn={wn:.3f}); need 0<Wn<1")
    return butter(order, wn, btype="low")


def _set_list_column(table: pa.Table, name: str, rows: list) -> pa.Table:
    """Replace ``name`` with ``rows`` (list of length-D lists), preserving the
    column's existing arrow type (e.g. fixed_size_list<float>[16])."""
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    new_col = pa.array(rows, type=field.type)
    return table.set_column(idx, field, new_col)


def _ep_stats(arr: np.ndarray) -> dict:
    """min/max/mean/std (ddof=0) per-dim + count -- matches LeRobot's episode stats."""
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="source LeRobot v2.1 dataset dir")
    p.add_argument("--dst", type=Path, required=True, help="output dir (must not exist)")
    p.add_argument("--cutoff-hz", type=float, default=4.0,
                   help="low-pass cutoff in Hz (default 4.0; reach is <2 Hz, wobble >3 Hz)")
    p.add_argument("--order", type=int, default=4, help="Butterworth order (default 4)")
    p.add_argument("--fps", type=float, default=None, help="override fps (default: from info.json)")
    args = p.parse_args()

    src: Path = args.src.expanduser()
    dst: Path = args.dst.expanduser()
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; remove it or choose another --dst")

    info = _load_info(src)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"this script is for v2.1 datasets; {src} is {info.get('codebase_version')!r} "
            f"(use filter_wobble_dataset.py for v3.0)"
        )
    fps = float(args.fps if args.fps is not None else info["fps"])
    b, a = _butter_lowpass(args.cutoff_hz, fps, args.order)
    padlen = 3 * max(len(a), len(b))
    print(f"[setup] fps={fps:.0f}  cutoff={args.cutoff_hz} Hz  order={args.order}  "
          f"(filtfilt -> zero-phase, effective order {2 * args.order})  padlen={padlen}")

    data_files = sorted(src.glob("data/chunk-*/episode_*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no per-episode parquet found under {src}/data/chunk-*/")
    print(f"[data] {len(data_files)} episode parquet files")

    # --- filter each episode parquet, collecting fresh state/action stats -----
    filtered_stats: dict[int, dict] = {}
    n_filt = n_skip = 0
    max_rms = 0.0
    for f in data_files:
        ep = int(_EP_RE.search(f.name).group(1))
        tbl = pq.read_table(f)
        frame_index = tbl.column("frame_index").to_numpy()
        order_idx = np.argsort(frame_index, kind="stable")  # temporal order

        new_stats = {}
        for feat in FILTERED_FEATURES:
            raw = np.asarray(tbl.column(feat).to_pylist(), dtype=np.float64)  # [T, D]
            out = raw.copy()
            if raw.shape[0] > padlen:
                out[order_idx] = filtfilt(b, a, raw[order_idx], axis=0)
                if feat == FILTERED_FEATURES[0]:
                    n_filt += 1
            elif feat == FILTERED_FEATURES[0]:
                n_skip += 1  # episode too short to filter; left as-is
            max_rms = max(max_rms, float(np.sqrt(np.mean((raw - out) ** 2))))
            tbl = _set_list_column(tbl, feat, out.astype(np.float32).tolist())
            new_stats[feat] = _ep_stats(out)
        filtered_stats[ep] = new_stats

        dst_file = dst / f.relative_to(src)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, dst_file)

    print(f"[data] filtered {n_filt} episodes (skipped {n_skip} too-short)  "
          f"max RMS change={max_rms:.4f} rad")

    # --- regenerate meta/episodes_stats.jsonl (state/action only) -------------
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    src_stats = (src / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    out_lines = []
    for line in src_stats:
        if not line.strip():
            continue
        e = json.loads(line)
        ep = int(e["episode_index"])
        for feat in FILTERED_FEATURES:
            e["stats"][feat] = filtered_stats[ep][feat]
        out_lines.append(json.dumps(e))
    (dst_meta / "episodes_stats.jsonl").write_text("\n".join(out_lines) + "\n")
    print(f"[stats] wrote meta/episodes_stats.jsonl ({len(out_lines)} episodes, state/action restatted)")

    # --- copy the small invariant meta verbatim ------------------------------
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
        shutil.copy2(src / "meta" / name, dst_meta / name)
    # optional global stats.json (this dataset family doesn't ship one)
    src_stats_json = src / "meta" / "stats.json"
    if src_stats_json.exists():
        shutil.copy2(src_stats_json, dst_meta / "stats.json")
    print("[meta] copied info.json / tasks.jsonl / episodes.jsonl")

    # --- symlink the big videos (RELATIVE so it resolves inside containers,
    #     e.g. openpi's docker that bind-mounts ~/.cache/huggingface -> /root/...) -
    src_videos = (src / "videos").resolve()
    if src_videos.exists():
        rel_target = os.path.relpath(src_videos, start=dst.resolve())
        os.symlink(rel_target, dst / "videos")
        print(f"[done] smoothed dataset at {dst}\n       videos symlinked -> {rel_target}")
    else:
        print(f"[done] smoothed dataset at {dst}  (no videos/ dir in source to link)")


if __name__ == "__main__":
    main()
