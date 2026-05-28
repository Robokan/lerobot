#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stand-alone tester for the *Phase 0 / "default hanging" move* used by
``lerobot_record --lift_arms_before_policy=true``.

Connects to the bimanual OpenArm follower, prints the current pose, then
slowly ramps to the operator-calibrated symmetric default pose
(``DEFAULT_BIMANUAL_START_POSE_DEG`` in
``lerobot.robots.bi_openarm_follower.lift_arms``) using the soft gains
(``SAFE_RAMP_KP`` / ``SAFE_RAMP_KD``). NO spine sweep, NO hold — just the
single "go-to-default-hanging" move so you can see whether it does what
you expect in isolation.

Usage (from a shell where the lerobot venv is active and CAN buses are up):

    python scripts/test_default_pose.py
    python scripts/test_default_pose.py --duration_s 4.0
    python scripts/test_default_pose.py --left_can can3 --right_can can2

Stopping:
    Ctrl-C at any time aborts the ramp and disconnects (which disables
    torque on both arms via ``disable_torque_on_disconnect=True``).
"""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

import lerobot.robots.bi_openarm_follower.lift_arms as _lift_arms_mod
from lerobot.robots.bi_openarm_follower import (
    BiOpenArmFollower,
    BiOpenArmFollowerConfig,
)
from lerobot.robots.bi_openarm_follower.lift_arms import (
    DEFAULT_BIMANUAL_START_POSE_DEG,
    DEFAULT_LIFT_HZ,
    DEFAULT_PRE_ZERO_S,
    SAFE_RAMP_KD,
    SAFE_RAMP_KP,
)
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4",
           "joint_5", "joint_6", "joint_7", "gripper")
_LEFT_KEYS = tuple(f"left_{j}.pos" for j in _JOINTS)
_RIGHT_KEYS = tuple(f"right_{j}.pos" for j in _JOINTS)
_KEY_ORDER = _LEFT_KEYS + _RIGHT_KEYS


def _read_pose_deg(robot: BiOpenArmFollower) -> np.ndarray:
    obs = robot.get_observation()
    arr = np.zeros(len(_KEY_ORDER), dtype=np.float64)
    for i, k in enumerate(_KEY_ORDER):
        v = obs.get(k)
        if v is None:
            print(f"  WARN: observation missing {k}; using 0.0", file=sys.stderr)
        else:
            arr[i] = float(v)
    return arr


def _print_pose(name: str, arr: np.ndarray) -> None:
    print(f"\n--- {name} (deg) ---")
    print(f"  {'joint':<10} {'left':>9} {'right':>9}")
    for i, j in enumerate(_JOINTS):
        L = arr[i]
        R = arr[i + 8]
        print(f"  {j:<10} {L:>+9.2f} {R:>+9.2f}")


def _default_pose_array() -> np.ndarray:
    arr = np.zeros(len(_KEY_ORDER), dtype=np.float64)
    for i, k in enumerate(_KEY_ORDER):
        arr[i] = float(DEFAULT_BIMANUAL_START_POSE_DEG[k])
    return arr


def _to_action_dict(pose: np.ndarray) -> dict[str, float]:
    return {k: float(pose[i]) for i, k in enumerate(_KEY_ORDER)}


def _format_pose_dict_literal(pose: np.ndarray) -> str:
    """Format the 16-DOF pose as the exact Python source for
    ``DEFAULT_BIMANUAL_START_POSE_DEG`` in ``lift_arms.py`` — same column
    layout the rest of the file uses (matched L/R per joint, one row per joint).
    """
    lines = ["DEFAULT_BIMANUAL_START_POSE_DEG: dict[str, float] = {"]
    for i, j in enumerate(_JOINTS):
        L = pose[i]
        R = pose[i + 8]
        # Match the existing 4-space indent + column widths used in lift_arms.py.
        # Sign-aware width so negatives line up with positives.
        l_key = f'"left_{j}.pos":'
        r_key = f'"right_{j}.pos":'
        lines.append(f"    {l_key:<17} {L:>+6.2f},  {r_key:<18} {R:>+6.2f},")
    lines.append("}")
    return "\n".join(lines)


def _rewrite_default_pose_in_lift_arms(pose: np.ndarray) -> Path:
    """Replace the ``DEFAULT_BIMANUAL_START_POSE_DEG = { ... }`` literal in
    ``lift_arms.py`` in place with the captured pose. Returns the file path
    that was edited.

    Uses a multiline regex that anchors on the dict header ``DEFAULT_BIMANUAL_START_POSE_DEG:
    dict[str, float] = {`` and consumes through the matching closing ``}``,
    so it doesn't depend on the specific whitespace / number formatting of
    the previous values.
    """
    src_path = Path(_lift_arms_mod.__file__)
    src = src_path.read_text()
    pattern = re.compile(
        r"DEFAULT_BIMANUAL_START_POSE_DEG:\s*dict\[str,\s*float\]\s*=\s*\{[^}]*\}",
        re.DOTALL,
    )
    matches = list(pattern.finditer(src))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 DEFAULT_BIMANUAL_START_POSE_DEG dict literal "
            f"in {src_path}, found {len(matches)}. Refusing to edit."
        )
    new_literal = _format_pose_dict_literal(pose)
    new_src = src[: matches[0].start()] + new_literal + src[matches[0].end():]
    src_path.write_text(new_src)
    return src_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left_can", default="can3",
                        help="CAN interface for the left arm (default: can3 — matches run_chocolate_policy.sh)")
    parser.add_argument("--right_can", default="can2",
                        help="CAN interface for the right arm (default: can2 — matches run_chocolate_policy.sh)")
    parser.add_argument("--duration_s", type=float, default=DEFAULT_PRE_ZERO_S,
                        help=f"Ramp duration in seconds (default: {DEFAULT_PRE_ZERO_S}s, the production Phase-0 value)")
    parser.add_argument("--hz", type=float, default=DEFAULT_LIFT_HZ,
                        help=f"Command rate in Hz (default: {DEFAULT_LIFT_HZ})")
    parser.add_argument("--hold_s", type=float, default=2.0,
                        help="Seconds to hold the default pose after the ramp before disconnecting (default: 2.0)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Connect, print current pose + planned target, do NOT command motion.")
    parser.add_argument("--capture", action="store_true",
                        help="Connect, read the current physical pose, and OVERWRITE "
                             "DEFAULT_BIMANUAL_START_POSE_DEG in lift_arms.py with it. "
                             "No motion is commanded. Use this after you've physically "
                             "positioned the arms in the start pose you want.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    left_cfg = OpenArmFollowerConfigBase(
        port=args.left_can,
        side="left",
        use_can_fd=False,
        can_bitrate=1_000_000,
    )
    right_cfg = OpenArmFollowerConfigBase(
        port=args.right_can,
        side="right",
        use_can_fd=False,
        can_bitrate=1_000_000,
    )
    cfg = BiOpenArmFollowerConfig(
        id="lumpa",
        left_arm_config=left_cfg,
        right_arm_config=right_cfg,
        cameras={},
    )

    robot = BiOpenArmFollower(cfg)

    abort = {"flag": False}

    def _on_sigint(_signo, _frame):
        if abort["flag"]:
            print("\n[test_default_pose] second Ctrl-C — exiting hard.", file=sys.stderr)
            sys.exit(1)
        abort["flag"] = True
        print("\n[test_default_pose] Ctrl-C caught — finishing this step and disconnecting...",
              file=sys.stderr)

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[test_default_pose] connecting (left={args.left_can}, right={args.right_can})...")
    robot.connect(calibrate=False)
    try:
        start_deg = _read_pose_deg(robot)
        target_deg = _default_pose_array()
        delta = target_deg - start_deg

        _print_pose("current pose", start_deg)

        if args.capture:
            print("\n--- captured pose (Python literal, ready to paste) ---")
            print(_format_pose_dict_literal(start_deg))
            edited = _rewrite_default_pose_in_lift_arms(start_deg)
            print(f"\n[test_default_pose] --capture: wrote new DEFAULT_BIMANUAL_START_POSE_DEG "
                  f"into {edited}")
            print("[test_default_pose] no motion was commanded. Done.")
            return 0

        _print_pose("target (DEFAULT_BIMANUAL_START_POSE_DEG)", target_deg)
        _print_pose("delta target - current", delta)
        max_abs_delta = float(np.max(np.abs(delta)))
        max_idx = int(np.argmax(np.abs(delta)))
        max_key = _KEY_ORDER[max_idx]
        print(f"\n  Max |delta| = {max_abs_delta:.2f} deg on {max_key}")
        print(f"  Peak avg speed during ramp ~ {max_abs_delta / max(args.duration_s, 1e-6):.2f} deg/s")
        print(f"  Soft gains: kp_shoulder=50, kp_wrist=10, kd_shoulder=1.2, kd_wrist≈0.3")

        if args.dry_run:
            print("\n[test_default_pose] --dry_run: not commanding any motion. Done.")
            return 0

        print(f"\n[test_default_pose] Phase 0 ramp: {args.duration_s:.2f}s @ {args.hz:.0f} Hz "
              "(soft gains, current -> default)")
        n = max(2, int(round(args.duration_s * args.hz)))
        ts = np.linspace(0.0, 1.0, n)
        step_dt = 1.0 / args.hz
        last_print = time.monotonic()
        for k, alpha in enumerate(ts):
            if abort["flag"]:
                print("[test_default_pose] aborted mid-ramp.", file=sys.stderr)
                break
            pose = start_deg * (1.0 - alpha) + target_deg * alpha
            robot.send_action(
                _to_action_dict(pose),
                custom_kp=SAFE_RAMP_KP,
                custom_kd=SAFE_RAMP_KD,
            )
            if time.monotonic() - last_print > 0.5:
                actual = _read_pose_deg(robot)
                err = float(np.max(np.abs(actual - pose)))
                print(f"    step {k+1:>4}/{n}  alpha={alpha:.2f}  "
                      f"max_tracking_err={err:.2f} deg")
                last_print = time.monotonic()
            time.sleep(step_dt)

        if not abort["flag"] and args.hold_s > 0:
            print(f"\n[test_default_pose] holding default pose {args.hold_s:.1f}s "
                  "(soft gains) to let it settle...")
            t_end = time.monotonic() + args.hold_s
            action = _to_action_dict(target_deg)
            while time.monotonic() < t_end and not abort["flag"]:
                robot.send_action(action, custom_kp=SAFE_RAMP_KP, custom_kd=SAFE_RAMP_KD)
                time.sleep(step_dt)

        final_deg = _read_pose_deg(robot)
        final_err = final_deg - target_deg
        _print_pose("final pose (actual)", final_deg)
        _print_pose("final error (actual - target)", final_err)
        print(f"\n  Max final |error| = {float(np.max(np.abs(final_err))):.2f} deg")
        if float(np.max(np.abs(final_err))) > 5.0:
            print("  WARNING: tracking error >5 deg — check for stalls, "
                  "wrong flash zero, joint-limit clipping, or a fault-latched motor.",
                  file=sys.stderr)

        return 0
    finally:
        print("[test_default_pose] disconnecting (torque will be disabled)...")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[test_default_pose] disconnect raised: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
