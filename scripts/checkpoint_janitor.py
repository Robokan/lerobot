#!/usr/bin/env python3
"""Keep a training run's checkpoints from filling the disk.

A GR00T checkpoint is ~24 GB (12 GB weights + 12 GB optimizer state) and
lerobot prunes nothing, so a fine save_freq fills a 3.6 TB disk in a day (it
already killed one run mid-flight). This watches a run's checkpoint directory
and keeps:

  * every checkpoint at a multiple of --keep-every steps (default 5000),
  * the --keep-recent most recent ones (default 5),
  * optimizer state ONLY on the newest checkpoint, so the run can still be
    resumed but old optimizer states do not each cost 12 GB.

Everything else is deleted. Runs until the training process exits.

Usage:
    python scripts/checkpoint_janitor.py outputs/groot_color_scratch
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path


def steps_of(d: Path) -> int | None:
    try:
        return int(d.name)
    except ValueError:
        return None


def sweep(root: Path, keep_every: int, keep_recent: int, verbose: bool) -> None:
    ckpts = sorted(
        (d for d in root.glob("*") if d.is_dir() and steps_of(d) is not None),
        key=lambda d: steps_of(d),
    )
    if not ckpts:
        return
    keep = {c for c in ckpts if steps_of(c) % keep_every == 0}
    keep |= set(ckpts[-keep_recent:])
    newest = ckpts[-1]
    for c in ckpts:
        if c not in keep:
            shutil.rmtree(c, ignore_errors=True)
            if verbose:
                print(f"  pruned {c.name}", flush=True)
            continue
        state = c / "training_state"
        if c is not newest and state.exists():
            shutil.rmtree(state, ignore_errors=True)
            if verbose:
                print(f"  dropped optimizer state of {c.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="a run's --output_dir (its checkpoints/ is watched)")
    ap.add_argument("--keep-every", type=int, default=5000)
    ap.add_argument("--keep-recent", type=int, default=5)
    ap.add_argument("--interval", type=float, default=120.0, help="seconds between sweeps")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.output_dir).expanduser() / "checkpoints"
    print(f"[janitor] watching {root}: keeping every {args.keep_every} steps, "
          f"the {args.keep_recent} most recent, and one optimizer state", flush=True)
    while True:
        if root.exists():
            sweep(root, args.keep_every, args.keep_recent, not args.quiet)
        if subprocess.run(["pgrep", "-f", "lerobot-trai[n]"], capture_output=True).returncode != 0:
            # training gone: sweep once more so the run leaves a tidy directory
            if root.exists():
                sweep(root, args.keep_every, args.keep_recent, not args.quiet)
            print("[janitor] training finished — done", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
