"""Regenerate an already-converted PI05 checkpoint's config + processor files so
it picks up the ``input_angle_unit`` flag and the AngleUnitProcessorStep,
WITHOUT rebuilding the 8GB model weights.

Reuses the norm stats already baked into the saved normalizer safetensors, so
this is fast (seconds). Mirrors the relative/absolute-action enabling that
``openpi_pt_to_lerobot.py`` does.

Usage:
    python scripts/restamp_angle_unit.py \
        --policy-dir outputs/pi05_chocolate_v4_from_openpi \
        --input-angle-unit radians
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

def _find_norm_file(policy_dir: Path) -> Path:
    """Locate the preprocessor normalizer safetensors regardless of its step
    index (the index shifts when steps are added/removed)."""
    candidates = sorted(policy_dir.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No preprocessor normalizer safetensors found in {policy_dir}"
        )
    # Prefer the highest step index (most recent layout) if several exist.
    return candidates[-1]


def _load_stats(policy_dir: Path) -> dict[str, dict]:
    """Reconstruct the nested {feature: {stat: tensor}} stats dict from the
    saved normalizer safetensors (keys are 'feature.stat')."""
    norm_file = _find_norm_file(policy_dir)
    print(f"      reading stats from {norm_file.name}")
    flat = load_file(str(norm_file))
    stats: dict[str, dict] = defaultdict(dict)
    for key, tensor in flat.items():
        feature, _, stat = key.rpartition(".")
        stats[feature][stat] = tensor
    return dict(stats)


def _cleanup_stale_normalizers(policy_dir: Path) -> None:
    """Remove normalizer safetensors no longer referenced by the saved JSONs."""
    import json as _json

    referenced: set[str] = set()
    for jf in policy_dir.glob("policy_*processor.json"):
        data = _json.loads(jf.read_text())
        for step in data.get("steps", []):
            sf = step.get("state_file")
            if sf:
                referenced.add(sf)
    for f in policy_dir.glob("policy_*_normalizer_processor.safetensors"):
        if f.name not in referenced:
            print(f"      removing stale {f.name}")
            f.unlink()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-dir", type=Path, required=True)
    p.add_argument("--input-angle-unit", default="radians",
                   choices=["radians", "degrees"])
    p.add_argument("--swap-arm-halves", action="store_true",
                   help="Swap the two 8-D arm halves at the policy boundary "
                        "(openpi/SparkJAX OpenArm is left-first; the lerobot "
                        "follower streams right-first). Leave OFF for native "
                        "lerobot-recorded+trained checkpoints.")
    p.add_argument("--convert-gripper-angle", action="store_true",
                   help="Include the gripper in the deg<->rad conversion "
                        "(angle_unit_exclude_joints=[]). Needed for openpi "
                        "OpenArm: the gripper is degrees on the wire, radians "
                        "in training.")
    args = p.parse_args()

    policy_dir = args.policy_dir
    print(f"[1/4] loading config from {policy_dir}")
    cfg = PreTrainedConfig.from_pretrained(policy_dir)
    print(f"      use_relative_actions={cfg.use_relative_actions}  "
          f"action_feature_names[:2]={(cfg.action_feature_names or [])[:2]}")

    print(f"[2/4] stamping config: input_angle_unit={args.input_angle_unit!r}, "
          f"swap_arm_halves={args.swap_arm_halves}, "
          f"convert_gripper_angle={args.convert_gripper_angle}")
    cfg.input_angle_unit = args.input_angle_unit
    cfg.swap_arm_halves = bool(args.swap_arm_halves)
    if args.convert_gripper_angle:
        cfg.angle_unit_exclude_joints = []
    cfg.save_pretrained(policy_dir)

    print("[3/4] reconstructing norm stats from saved normalizer safetensors")
    stats = _load_stats(policy_dir)
    print(f"      features: {sorted(stats)}")

    print("[4/4] rebuilding + saving processors (fresh, with angle steps)")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=stats
    )

    # Mirror openpi_pt_to_lerobot.py EXACTLY: it force-enables the delta/absolute
    # action steps unconditionally (the openpi OpenArm checkpoint is a delta-joint
    # model: mask [7 delta, 1 abs, 7 delta, 1 abs]). The saved config leaves
    # use_relative_actions=False, but the runtime processors MUST have these
    # enabled or the model's delta output is never added back to the anchor
    # state (arms collapse toward zero). Do not gate on cfg.use_relative_actions.
    for step in preprocessor.steps:
        if type(step).__name__ == "RelativeActionsProcessorStep":
            step.enabled = True
            step.exclude_joints = ["gripper"]
            if (
                step.action_names
                and isinstance(step.action_names, list)
                and isinstance(step.action_names[0], list)
            ):
                step.action_names = list(step.action_names[0])
            print(f"      enabled delta_actions_processor (exclude={step.exclude_joints})")
        if type(step).__name__ == "AngleUnitProcessorStep":
            print(f"      preprocessor AngleUnitProcessorStep: scale={step.scale:.6f} "
                  f"exclude={step.exclude_joints} "
                  f"(apply_obs={step.apply_to_observation}, apply_act={step.apply_to_action})")
        if type(step).__name__ == "ArmSwapProcessorStep":
            print(f"      preprocessor ArmSwapProcessorStep: enabled={step.enabled} "
                  f"(apply_obs={step.apply_to_observation}, apply_act={step.apply_to_action})")
    for step in postprocessor.steps:
        if type(step).__name__ == "AbsoluteActionsProcessorStep":
            step.enabled = True
            print(f"      enabled absolute_actions_processor")
        if type(step).__name__ == "AngleUnitProcessorStep":
            print(f"      postprocessor AngleUnitProcessorStep: scale={step.scale:.6f} "
                  f"exclude={step.exclude_joints} "
                  f"(apply_obs={step.apply_to_observation}, apply_act={step.apply_to_action})")
        if type(step).__name__ == "ArmSwapProcessorStep":
            print(f"      postprocessor ArmSwapProcessorStep: enabled={step.enabled} "
                  f"(apply_obs={step.apply_to_observation}, apply_act={step.apply_to_action})")

    preprocessor.save_pretrained(policy_dir)
    postprocessor.save_pretrained(policy_dir)
    _cleanup_stale_normalizers(policy_dir)

    print("\nDONE. Updated files:")
    for f in sorted(policy_dir.glob("policy_*")):
        print(f"    {f.name}")
    print(f"    config.json")


if __name__ == "__main__":
    main()
