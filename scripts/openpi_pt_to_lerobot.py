"""
Convert an openpi-PyTorch safetensors checkpoint (output of
`openpi/examples/convert_jax_model_to_pytorch.py`) into a lerobot
PI05Policy directory that `lerobot-eval` / `RobotPolicy.from_pretrained`
can load directly.

Why this is just a key rename:
    openpi's PI0Pytorch root state_dict and lerobot's PI05Policy.model
    state_dict are the same module tree (lerobot pi05 is a port of
    openpi pi0_pytorch). The only difference is lerobot wraps the
    flow-matching network under `self.model`, so every key needs the
    `model.` prefix added.

    The tied embed_tokens<->lm_head weight (deduped in the openpi
    safetensors via metadata) is re-tied after load.

Usage:
    python scripts/openpi_pt_to_lerobot.py \\
        --openpi-ckpt /path/to/openpi_pytorch/model.safetensors \\
        --norm-stats /path/to/openpi_jax_ckpt/assets/<robot_name>/norm_stats.json \\
        --dataset-root /home/$USER/.cache/huggingface/lerobot/local/openarm-chocolate-v4 \\
        --output-dir /home/$USER/sparkpack/lerobot/outputs/pi05_chocolate_v4_from_openpi

LoRA handling:
    Per openpi's `JAX_TO_PYTORCH_LORA_CONVERSION.md`, pre-merging LoRA
    into base weights and storing in bf16 introduces a systematic ~8%
    magnitude bias (robot drifts upward). The fix is **runtime LoRA**:
    keep `lora_a`/`lora_b` separate and add the LoRA contribution as
    two matmuls on every forward, matching JAX numerics exactly.

    This script ports openpi's `lora_runtime.install_runtime_lora` to
    lerobot. Pass `--lora-ckpt` pointing to the `lora.safetensors` that
    sits next to the base `model.safetensors` in
    `chocolate_bars_pi05_pytorch/` (the variant produced when openpi was
    converted with `OPENPI_PT_RUNTIME_LORA=1`).

    The output directory carries `lora.safetensors` plus a tiny
    `lora_runtime_marker.json` so the load helper can re-apply patches
    automatically.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_model

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def _build_features_from_dataset(root: Path):
    """Read dataset meta to pin down state/action shapes + camera names."""
    meta = LeRobotDatasetMetadata(repo_id=root.name, root=root)
    input_features = {}
    output_features = {}

    state = meta.features["observation.state"]
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=tuple(state["shape"])
    )
    for k, v in meta.features.items():
        if k.startswith("observation.images."):
            input_features[k] = PolicyFeature(
                type=FeatureType.VISUAL, shape=tuple(v["shape"])
            )

    action = meta.features["action"]
    output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION, shape=tuple(action["shape"])
    )

    # action.names in v3.0 datasets is nested: [[name1, name2, ...]] (1 list per
    # axis). lerobot's RelativeActionsProcessorStep._build_mask iterates this
    # directly; if we pass it nested, it sees a single "name" that stringifies
    # to "[..., 'gripper']" — contains "gripper" → masks dim 0 as excluded and
    # silently leaves all others as deltas. Flatten to a list of strings.
    raw_action_names = action.get("names") or []
    if raw_action_names and isinstance(raw_action_names[0], list):
        action_names = list(raw_action_names[0])
    else:
        action_names = list(raw_action_names)
    return input_features, output_features, action_names, meta.stats


def _retie_embed_tokens(policy: PI05Policy) -> None:
    """openpi safetensors dedupes tied weights. Re-tie after load."""
    pg = policy.model.paligemma_with_expert.paligemma
    pg.model.language_model.embed_tokens.weight = pg.lm_head.weight
    # gemma_expert has no embed_tokens (action expert processes continuous
    # action vectors, not language tokens). It carries an lm_head only because
    # it inherits the Gemma class; pi0.5 never uses it. No retie needed.


def _inject_norm_stats(policy_stats: dict, openpi_norm: dict) -> dict:
    """
    Merge openpi norm_stats.json into a lerobot-style policy_stats dict.

    openpi norm_stats schema (per key, e.g. `state`, `actions`):
        { "mean": [...], "std": [...], "q01": [...], "q99": [...] }

    lerobot stats dict (per feature, e.g. `observation.state`, `action`):
        { "mean": tensor|ndarray, "std": ..., "q01": ..., "q99": ...,
          "min": ..., "max": ..., "count": ... }

    We overwrite mean/std/q01/q99 from openpi if present, leave dataset-derived
    min/max/count untouched. Returns torch tensors for normalizer compatibility.
    """
    import numpy as np

    def _to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.to(torch.float32)
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).to(torch.float32)
        return torch.tensor(v, dtype=torch.float32)

    merged: dict = {}
    for feat, d in policy_stats.items():
        merged[feat] = {k: _to_tensor(v) if k != "count" else v for k, v in d.items()}

    mapping = {
        "observation.state": ["state"],
        "action": ["actions", "action"],
    }
    for lr_key, op_candidates in mapping.items():
        for cand in op_candidates:
            if cand in openpi_norm:
                src = openpi_norm[cand]
                for stat in ("mean", "std", "q01", "q99"):
                    if stat in src:
                        merged.setdefault(lr_key, {})[stat] = torch.tensor(
                            src[stat], dtype=torch.float32
                        )
                break
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--openpi-ckpt", type=Path, required=True,
                   help="path to openpi-PT base model.safetensors")
    p.add_argument("--lora-ckpt", type=Path, default=None,
                   help="path to openpi-PT lora.safetensors (runtime LoRA). "
                        "If provided, LoRA is kept separate and applied at "
                        "forward time (recommended; matches JAX numerics).")
    p.add_argument("--norm-stats", type=Path, default=None,
                   help="optional openpi assets/<robot>/norm_stats.json to copy in")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="lerobot dataset root to derive feature shapes (state/action/cameras)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="where to write the lerobot-compatible policy directory")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float32"])
    args = p.parse_args()

    print(f"[1/6] reading dataset meta from {args.dataset_root}")
    input_features, output_features, action_names, ds_stats = _build_features_from_dataset(
        args.dataset_root
    )
    print(f"      state shape:   {input_features['observation.state'].shape}")
    print(f"      action shape:  {output_features['action'].shape}  ({len(action_names)} names)")
    print(f"      cameras:       {[k for k in input_features if 'images' in k]}")

    # We build the in-memory policy on CPU because conversion is a state_dict
    # remap (no forward pass), but we write `device="cuda"` into the SAVED
    # config so downstream consumers (lerobot-record, eval scripts) default
    # to GPU. Without this, the saved policy config has `device: cpu`, which
    # causes ~11s/inference and a 0.1 Hz control loop on first launch -
    # users then have to remember to pass `--policy.device=cuda` to override.
    print(f"[2/6] building empty PI05Policy on CPU (dtype={args.dtype}, ~90s, ~8GB) "
          f"-> saved cfg will default device=cuda")
    cfg = PI05Config(dtype=args.dtype, device="cpu")
    cfg.input_features = input_features
    cfg.output_features = output_features
    cfg.action_feature_names = list(action_names)
    cfg.empty_cameras = max(0, 3 - sum(1 for k in input_features if "images" in k))
    policy = PI05Policy(cfg)

    print(f"[3/6] loading openpi safetensors {args.openpi_ckpt}")
    src = {}
    with safe_open(args.openpi_ckpt, framework="pt") as f:
        for k in f.keys():
            src[k] = f.get_tensor(k)
    print(f"      loaded {len(src)} tensors, total params "
          f"{sum(t.numel() for t in src.values())/1e9:.2f}B")

    print("[4/6] remapping keys (prefix `model.`) and loading into PI05Policy")
    remapped = {f"model.{k}": v for k, v in src.items()}
    missing, unexpected = policy.load_state_dict(remapped, strict=False)
    print(f"      missing keys:    {len(missing)}")
    if missing:
        for k in missing[:5]:
            print(f"        {k}")
        if len(missing) > 5:
            print(f"        ... ({len(missing)-5} more)")
    print(f"      unexpected keys: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:5]:
            print(f"        {k}")

    _retie_embed_tokens(policy)
    print("      re-tied embed_tokens<->lm_head")

    # Verify tying worked
    pg = policy.model.paligemma_with_expert.paligemma
    et = pg.model.language_model.embed_tokens.weight
    lh = pg.lm_head.weight
    if et.data_ptr() != lh.data_ptr():
        print("      WARNING: re-tie failed, embed_tokens and lm_head are distinct tensors")
    else:
        print("      OK: embed_tokens and lm_head share storage")

    if args.norm_stats is not None and args.norm_stats.exists():
        print(f"[5/6] merging norm_stats from {args.norm_stats}")
        with open(args.norm_stats) as f:
            openpi_norm = json.load(f)
        # norm_stats.json from openpi is sometimes wrapped under "norm_stats"
        if "norm_stats" in openpi_norm:
            openpi_norm = openpi_norm["norm_stats"]
        new_stats = _inject_norm_stats({k: dict(v) for k, v in ds_stats.items()}, openpi_norm)
    else:
        print("[5/6] no --norm-stats given, will use dataset-derived stats")
        new_stats = ds_stats

    print(f"[6/6] saving lerobot policy directory to {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Flip the saved default device from "cpu" (the value we used to
    # construct the in-memory policy) to "cuda" so downstream consumers
    # don't accidentally run inference on CPU and wonder why the control
    # loop is at 0.1 Hz. Users on CPU-only boxes can still override at
    # runtime with ``--policy.device=cpu``.
    cfg.device = "cuda"
    cfg.save_pretrained(args.output_dir)
    # save_model (not save_file) handles tied embed_tokens<->lm_head by
    # recording the alias in metadata, identical to what openpi does.
    save_model(policy, str(args.output_dir / "model.safetensors"))

    # Copy the lora.safetensors verbatim (no remapping needed; load helper
    # re-applies the openpi-format keys via lora_runtime.install_runtime_lora).
    if args.lora_ckpt is not None:
        shutil.copyfile(args.lora_ckpt, args.output_dir / "lora.safetensors")
        (args.output_dir / "lora_runtime_marker.json").write_text(json.dumps({
            "runtime_lora": True,
            "lora_file": "lora.safetensors",
            "comment": (
                "This policy was converted from an openpi-PT runtime-LoRA "
                "checkpoint. Load with "
                "`lerobot.policies.pi05.lora_runtime.install_runtime_lora` "
                "after calling `make_policy(...)` (or use "
                "`scripts/load_pi05_from_openpi.py`)."
            ),
        }, indent=2))

    # Build + save the pre/post processors with the (openpi-injected) stats,
    # so `lerobot-eval` / `RobotPolicy.from_pretrained` loads with the right
    # normalization out of the box.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=new_stats
    )

    # IMPORTANT: openpi's `LeRobotOpenArmDataConfig` defaults to
    # `use_delta_joint_actions=True` with mask [7 delta, 1 abs, 7 delta, 1 abs].
    # The model learned to predict deltas for arm joints and absolute positions
    # for the two grippers. lerobot mirrors this via
    # `delta_actions_processor(enabled, exclude_joints=["gripper"])` on the pre
    # side and `absolute_actions_processor(enabled)` on the post side. The
    # default pi05 processor factory leaves both disabled, so flip them here.
    # (make_pre_post_processors's `preprocessor_overrides` kwarg only takes
    # effect when loading from a pretrained path; we're building fresh.)
    for step in preprocessor.steps:
        if type(step).__name__ == "RelativeActionsProcessorStep":
            step.enabled = True
            step.exclude_joints = ["gripper"]
            # Force flat action_names; lerobot may have populated it from the
            # dataset as the nested [[...]] form (see _build_features_from_dataset
            # comment). _build_mask iterates this list directly.
            if (
                step.action_names
                and isinstance(step.action_names, list)
                and step.action_names
                and isinstance(step.action_names[0], list)
            ):
                step.action_names = list(step.action_names[0])
            print(f"      enabled delta_actions_processor "
                  f"(exclude_joints={step.exclude_joints}, "
                  f"action_names[0..2]={step.action_names[:2] if step.action_names else None})")
    for step in postprocessor.steps:
        if type(step).__name__ == "AbsoluteActionsProcessorStep":
            step.enabled = True
            print(f"      enabled absolute_actions_processor")

    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)

    print(f"\nDONE. Output directory:")
    print(f"  {args.output_dir}")
    for f in sorted(args.output_dir.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
