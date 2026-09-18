#!/usr/bin/env python
"""Delete checkpoints from a private HuggingFace checkpoint-mirror repo.

Deleting an LFS file in a new commit does NOT reclaim its storage — the blob
stays in the repo's history — so this squashes history afterwards, which is
what actually frees the quota. That makes it irreversible: everything deleted
here is gone from the Hub, and so is the history that referenced it. Only point
it at a checkpoint drop whose contents exist elsewhere.

  # see what is there
  python scripts/hf_prune_checkpoints.py evaughan69/groot_color_3cam_aug --list

  # keep one checkpoint, delete the rest
  python scripts/hf_prune_checkpoints.py evaughan69/groot_color_3cam_aug \
      --delete 010000 020000 030000 040000 --yes
"""

from __future__ import annotations

import argparse
import collections
import sys

from huggingface_hub import HfApi


def sizes(api: HfApi, repo: str) -> dict[str, int]:
    info = api.repo_info(repo, files_metadata=True)
    tot: dict[str, int] = collections.defaultdict(int)
    for f in info.siblings:
        parts = f.rfilename.split("/")
        if len(parts) > 2 and parts[0] == "checkpoints":
            tot[parts[1]] += f.size or 0
    return dict(tot)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--list", action="store_true", help="show checkpoints and sizes, change nothing")
    ap.add_argument("--delete", nargs="*", default=[], metavar="STEP", help="checkpoint dirs to delete")
    ap.add_argument("--yes", action="store_true", help="actually delete (otherwise this is a dry run)")
    args = ap.parse_args()

    api = HfApi()
    have = sizes(api, args.repo)
    print(f"{args.repo}: {len(have)} checkpoints, {sum(have.values()) / 1e9:.1f} GB")
    for step in sorted(have):
        mark = " <- DELETE" if step in args.delete else ""
        print(f"  {step}  {have[step] / 1e9:5.2f} GB{mark}")
    if args.list or not args.delete:
        return 0

    missing = [s for s in args.delete if s not in have]
    if missing:
        print(f"not in the repo: {missing}", file=sys.stderr)
        return 2
    freed = sum(have[s] for s in args.delete) / 1e9
    keep = sorted(set(have) - set(args.delete))
    print(f"\nwould free {freed:.1f} GB, leaving {keep or 'NOTHING'}")
    if not args.yes:
        print("dry run — pass --yes to do it")
        return 0

    for step in args.delete:
        api.delete_folder(path_in_repo=f"checkpoints/{step}", repo_id=args.repo,
                          commit_message=f"delete checkpoint {step}")
        print(f"deleted checkpoints/{step}")
    print("squashing history (this is what reclaims the storage)…")
    api.super_squash_history(repo_id=args.repo, commit_message=f"keep checkpoints {keep}")
    after = sizes(api, args.repo)
    print(f"now: {sum(after.values()) / 1e9:.1f} GB across {sorted(after)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
