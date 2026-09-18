#!/usr/bin/env python
"""Push a local LeRobot dataset to a private HuggingFace dataset repo.

This exists to keep the slow half of the transfer off the clock. Uploading a
3 GB dataset from this workstation takes the best part of an hour on a home
link; doing it by rsync to a running pod bills the pod for every minute of it.
Pushed to the Hub first, the pod pulls the same data at datacenter speed in
minutes, and lerobot can take the Hub repo id directly as --dataset.repo_id.

  python scripts/hf_upload_dataset.py local/openarm_caddy6_pick_all_300 \
      --to evaughan69/openarm_caddy6_pick_all_300

Re-running resumes: upload_folder skips files already on the Hub.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_id", help="local dataset repo id, e.g. local/openarm_caddy6_pick_all_300")
    ap.add_argument("--to", required=True, help="Hub repo, e.g. evaughan69/openarm_caddy6_pick_all_300")
    ap.add_argument("--root", default=os.environ.get("HF_LEROBOT_HOME", str(Path.home() / ".cache/huggingface/lerobot")))
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    args = ap.parse_args()

    src = Path(args.root) / args.repo_id
    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        print(f"no dataset at {src}")
        return 1
    info = json.loads(info_path.read_text())
    size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    cams = [k.split(".")[-1] for k in info["features"] if "images" in k]
    print(f"{src}\n  {info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{info['fps']} fps, cameras {cams}, {size / 1e9:.2f} GB")

    api = HfApi()
    api.create_repo(args.to, repo_type="dataset", private=not args.public, exist_ok=True)
    print(f"uploading to {args.to} (private={not args.public}) — resumable, re-run if it drops")
    api.upload_folder(repo_id=args.to, repo_type="dataset", folder_path=str(src),
                      commit_message=f"{info['total_episodes']} episodes, {info['total_frames']} frames")
    files = api.list_repo_files(args.to, repo_type="dataset")
    print(f"done: {len(files)} files on the Hub")
    print(f"train with:  --dataset.repo_id={args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
