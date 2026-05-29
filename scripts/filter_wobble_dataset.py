#!/usr/bin/env python
"""Produce a wobble-filtered copy of a LeRobot dataset.

Teleop captured on a low-stiffness / low-damping follower bakes underdamped
ringing into ``observation.state`` AND ``action`` (the policy is trained to
reproduce the action wobble, then a stiff deployment arm chases every wiggle ->
jerky execution). This script low-pass filters the joint/gripper trajectories
with a ZERO-PHASE Butterworth filter (``filtfilt``, no time shift), per-episode
(no bleed across episode boundaries) and per-dimension.

Only ``observation.state`` and ``action`` are touched. Video files are
symlinked (not copied), so the smoothed dataset costs ~17 MB, not ~4 GB. The
per-episode stats (``meta/episodes``) and global stats (``meta/stats.json``) for
state/action are recomputed with LeRobot's own utilities so the copy is fully
self-consistent for retraining.

Usage:
    python scripts/filter_wobble_dataset.py \
        --src ~/.cache/huggingface/lerobot/local/openarm-chocolate-v4 \
        --dst ~/.cache/huggingface/lerobot/local/openarm-chocolate-v4-smoothed \
        --cutoff-hz 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

# Only these two features carry the wobble and are filtered + restatted.
FILTERED_FEATURES = ["observation.state", "action"]
# This dataset stores q01/q99 in stats.json; match that exactly.
QUANTILES = [0.01, 0.99]


def _load_info(src: Path) -> dict:
    return json.loads((src / "meta" / "info.json").read_text())


def _butter_lowpass(cutoff_hz: float, fps: float, order: int):
    nyq = fps / 2.0
    wn = cutoff_hz / nyq
    if not 0.0 < wn < 1.0:
        raise ValueError(f"cutoff {cutoff_hz} Hz invalid for fps {fps} (Wn={wn:.3f}); need 0<Wn<1")
    return butter(order, wn, btype="low")


def filter_per_episode(
    arr: np.ndarray, episode_index: np.ndarray, frame_index: np.ndarray, b, a
) -> tuple[np.ndarray, int, int]:
    """Zero-phase low-pass each episode's [T, D] block independently."""
    out = arr.copy()
    padlen = 3 * max(len(a), len(b))
    n_filt = 0
    n_skip = 0
    for e in np.unique(episode_index):
        pos = np.where(episode_index == e)[0]
        pos = pos[np.argsort(frame_index[pos])]  # temporal order
        if pos.size <= padlen:
            n_skip += 1
            continue
        out[pos] = filtfilt(b, a, arr[pos], axis=0)
        n_filt += 1
    return out, n_filt, n_skip


def _set_list_column(table: pa.Table, name: str, rows: list, pa_type: pa.DataType) -> pa.Table:
    idx = table.schema.get_field_index(name)
    new_col = pa.array(rows, type=pa_type)
    return table.set_column(idx, table.schema.field(idx), new_col)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="source LeRobot dataset dir")
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
    fps = float(args.fps if args.fps is not None else info["fps"])
    b, a = _butter_lowpass(args.cutoff_hz, fps, args.order)
    print(f"[setup] fps={fps:.0f}  cutoff={args.cutoff_hz} Hz  order={args.order}  "
          f"(filtfilt -> zero-phase, effective order {2 * args.order})")

    # --- locate the single data parquet and episodes parquet -----------------
    data_files = sorted(src.glob("data/**/*.parquet"))
    ep_files = sorted(src.glob("meta/episodes/**/*.parquet"))
    if len(data_files) != 1 or len(ep_files) != 1:
        raise NotImplementedError(
            f"this script assumes a single data + episodes parquet "
            f"(found {len(data_files)} data, {len(ep_files)} episodes)"
        )
    data_file, ep_file = data_files[0], ep_files[0]

    # --- filter the data parquet ---------------------------------------------
    print(f"[data] reading {data_file.relative_to(src)}")
    tbl = pq.read_table(data_file)
    episode_index = tbl.column("episode_index").to_numpy()
    frame_index = tbl.column("frame_index").to_numpy()

    filtered: dict[str, np.ndarray] = {}
    for feat in FILTERED_FEATURES:
        raw = np.asarray(tbl.column(feat).to_pylist(), dtype=np.float64)  # [N, D]
        out, n_filt, n_skip = filter_per_episode(raw, episode_index, frame_index, b, a)
        filtered[feat] = out
        rms_removed = float(np.sqrt(np.mean((raw - out) ** 2)))
        print(f"[data] {feat}: filtered {n_filt} eps (skipped {n_skip} too-short), "
              f"RMS change={rms_removed:.4f} rad")
        tbl = _set_list_column(tbl, feat, out.astype(np.float32).tolist(), pa.list_(pa.float32()))

    dst_data = dst / data_file.relative_to(src)
    dst_data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, dst_data)
    print(f"[data] wrote {dst_data.relative_to(dst)}")

    # --- recompute per-episode stats (in episodes-parquet row order) ----------
    print("[stats] recomputing per-episode stats on filtered state/action")
    features = {f: {"dtype": "float32", "shape": [filtered[f].shape[1]]} for f in FILTERED_FEATURES}
    ep_tbl = pq.read_table(ep_file)
    ep_order = ep_tbl.column("episode_index").to_numpy()

    per_ep_full: dict[int, dict] = {}
    for e in ep_order:
        pos = np.where(episode_index == e)[0]
        pos = pos[np.argsort(frame_index[pos])]
        ep_data = {f: filtered[f][pos] for f in FILTERED_FEATURES}
        per_ep_full[int(e)] = compute_episode_stats(ep_data, features, quantile_list=QUANTILES)

    # write only the 5 base stats columns the episodes parquet stores
    for feat in FILTERED_FEATURES:
        for stat, pa_type in (
            ("min", pa.list_(pa.float64())),
            ("max", pa.list_(pa.float64())),
            ("mean", pa.list_(pa.float64())),
            ("std", pa.list_(pa.float64())),
            ("count", pa.list_(pa.int64())),
        ):
            col = f"stats/{feat}/{stat}"
            rows = [np.asarray(per_ep_full[int(e)][feat][stat]).tolist() for e in ep_order]
            ep_tbl = _set_list_column(ep_tbl, col, rows, pa_type)

    dst_ep = dst / ep_file.relative_to(src)
    dst_ep.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(ep_tbl, dst_ep)
    print(f"[stats] wrote {dst_ep.relative_to(dst)}")

    # --- recompute global stats.json (state/action only) ----------------------
    agg = aggregate_stats([per_ep_full[int(e)] for e in ep_order])
    stats_json = json.loads((src / "meta" / "stats.json").read_text())
    for feat in FILTERED_FEATURES:
        for k, v in agg[feat].items():
            stats_json[feat][k] = np.asarray(v).tolist()
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "stats.json").write_text(json.dumps(stats_json))
    print("[stats] wrote meta/stats.json (state/action updated, others preserved)")

    # --- copy small meta verbatim, symlink the big videos --------------------
    shutil.copy2(src / "meta" / "info.json", dst / "meta" / "info.json")
    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    os.symlink((src / "videos").resolve(), dst / "videos")
    print(f"[done] smoothed dataset at {dst}\n       videos symlinked -> {(src / 'videos').resolve()}")


if __name__ == "__main__":
    main()
