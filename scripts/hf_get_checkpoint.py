#!/usr/bin/env python
"""Pull a checkpoint (and optionally its dataset) from the Hub onto this machine.

The inference side of the loop. The Spark records data and pushes it to the Hub,
a pod trains and pushes checkpoints to the Hub, and this brings both down
wherever evaluation happens. Nothing needs a USB drive.

  # list what is on the Hub
  python scripts/hf_get_checkpoint.py evaughan69/groot_caddy6_3cam --list

  # fetch one checkpoint and the dataset it needs for normalisation stats
  python scripts/hf_get_checkpoint.py evaughan69/groot_caddy6_3cam 070000 \
      --dataset evaughan69/openarm_caddy6_pick_all_300

The dataset lands in the LeRobot cache under its Hub repo id, so the eval takes
that id directly as --dataset.
"""

from __future__ import annotations

import argparse
import collections
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def checkpoints(api: HfApi, repo: str) -> dict[str, int]:
    tot: dict[str, int] = collections.defaultdict(int)
    for f in api.repo_info(repo, files_metadata=True).siblings:
        parts = f.rfilename.split("/")
        if len(parts) > 2 and parts[0] == "checkpoints":
            tot[parts[1]] += f.size or 0
    return dict(tot)


def fetch(repo: str, patterns: str, dest: Path, repo_type: str, retries: int = 8) -> None:
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(repo, repo_type=repo_type, allow_patterns=patterns,
                              local_dir=str(dest), max_workers=4)
            return
        except Exception as e:  # noqa: BLE001 — a long pull reliably meets one transport error
            print(f"  attempt {attempt}/{retries}: {type(e).__name__}: {str(e)[:110]}", flush=True)
            if attempt == retries:
                raise
            time.sleep(min(60, 5 * attempt))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="checkpoint repo, e.g. evaughan69/groot_caddy6_3cam")
    ap.add_argument("step", nargs="?", help="checkpoint to fetch, e.g. 070000 (default: the newest)")
    ap.add_argument("--dataset", help="also fetch this Hub dataset into the LeRobot cache")
    ap.add_argument("--out", default=str(Path.home() / "checkpoints"),
                    help="where checkpoints land (default ~/checkpoints)")
    ap.add_argument("--list", action="store_true", help="show what is on the Hub and stop")
    args = ap.parse_args()

    api = HfApi()
    have = checkpoints(api, args.repo)
    if not have:
        print(f"{args.repo} has no checkpoints/<step>/ directories")
        return 1
    print(f"{args.repo}:")
    for s in sorted(have):
        print(f"  {s}  {have[s] / 1e9:5.2f} GB")
    if args.list:
        return 0

    step = args.step or sorted(have)[-1]
    if step not in have:
        print(f"no checkpoint {step}; have {sorted(have)}")
        return 2

    dest = Path(args.out) / args.repo.split("/")[-1] / step
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nfetching checkpoint {step} ({have[step] / 1e9:.1f} GB) -> {dest}")
    t = time.time()
    fetch(args.repo, f"checkpoints/{step}/*", dest.parent / ".staging", "model")
    staged = dest.parent / ".staging" / "checkpoints" / step
    for f in staged.iterdir():
        f.replace(dest / f.name)
    for p in sorted((dest.parent / ".staging").rglob("*"), reverse=True):
        p.rmdir() if p.is_dir() else p.unlink()
    (dest.parent / ".staging").rmdir()
    print(f"  {sum(f.stat().st_size for f in dest.iterdir()) / 1e9:.1f} GB in {(time.time() - t) / 60:.1f} min")

    if args.dataset:
        root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
        dpath = root / args.dataset
        print(f"\nfetching dataset {args.dataset} -> {dpath}")
        t = time.time()
        fetch(args.dataset, "*", dpath, "dataset")
        print(f"  done in {(time.time() - t) / 60:.1f} min")

    print(f"\neval with:\n  --policy {dest}" + (f" --dataset {args.dataset}" if args.dataset else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
