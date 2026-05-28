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

"""Stand-alone tester for the full LIFT sequence used by
``lerobot-record --lift_arms_before_policy=true``.

Runs all four phases of ``lift_arms_to_ready`` in isolation:

    Phase 0 (3.0 s) - slow soft-gain ramp from current pose -> zero pose
                       (arms hanging straight, grippers closed)
    Phase 1 (3.0 s) - soft-gain ramp from zero pose -> LIFT_SPINE[0]
    Phase 2 (1.5 s) - stiff-gain sweep through the 9 SparkJAX spine
                       waypoints -> arms-up, tucked, table-cleared pose
    Phase 3 (0.5 s) - hold the final pose so the arms settle

Does NOT load the policy, open the cameras, or write a dataset. Just the
motion. Use this to verify the lift trajectory before committing to a full
policy run.

Prereqs:
    1. CAN buses up:           sudo bash scripts/bring_up_can.sh
    2. lerobot venv active:    source .venv/bin/activate
    3. Flash zero calibrated:  python scripts/recalibrate_zero.py
       (verify with ``python scripts/test_default_pose.py --dry_run`` ->
        every joint should read ~0.0 deg with arms physically hanging
        straight and grippers closed)

Usage:
    python scripts/test_lift_arms.py
    python scripts/test_lift_arms.py --skip_zero          # skip Phase 0
    python scripts/test_lift_arms.py --final_hold_s 10    # hold lifted pose 10s

Stopping:
    Ctrl-C aborts the sequence. The bus disconnect in the finally block
    will disable torque on both arms.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import numpy as np

from lerobot.robots.bi_openarm_follower import (
    BiOpenArmFollower,
    BiOpenArmFollowerConfig,
    lift_arms_to_ready,
)
from lerobot.robots.bi_openarm_follower.lift_arms import (
    DEFAULT_HOLD_S,
    DEFAULT_LIFT_DURATION_S,
    DEFAULT_LIFT_HZ,
    DEFAULT_PRE_RAMP_S,
    DEFAULT_PRE_ZERO_S,
    LIFT_SPINE_RAD,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left_can", default="can3",
                        help="CAN interface for the left arm (default: can3 — matches run_chocolate_policy.sh)")
    parser.add_argument("--right_can", default="can2",
                        help="CAN interface for the right arm (default: can2 — matches run_chocolate_policy.sh)")
    parser.add_argument("--pre_zero_s", type=float, default=DEFAULT_PRE_ZERO_S,
                        help=f"Phase 0 duration (current -> zero pose). Default {DEFAULT_PRE_ZERO_S}s.")
    parser.add_argument("--skip_zero", action="store_true",
                        help="Shortcut for --pre_zero_s 0. Skips Phase 0 entirely.")
    parser.add_argument("--pre_ramp_s", type=float, default=DEFAULT_PRE_RAMP_S,
                        help=f"Phase 1 duration (zero pose -> spine[0]). Default {DEFAULT_PRE_RAMP_S}s.")
    parser.add_argument("--spine_s", type=float, default=DEFAULT_LIFT_DURATION_S,
                        help=f"Phase 2 duration (spine sweep). Default {DEFAULT_LIFT_DURATION_S}s.")
    parser.add_argument("--hold_s", type=float, default=DEFAULT_HOLD_S,
                        help=f"Phase 3 duration (settle on final spine pose). Default {DEFAULT_HOLD_S}s.")
    parser.add_argument("--hz", type=float, default=DEFAULT_LIFT_HZ,
                        help=f"Command rate (Hz). Default {DEFAULT_LIFT_HZ}.")
    parser.add_argument("--final_hold_s", type=float, default=5.0,
                        help="EXTRA hold time (s) AFTER the spine completes and before disconnecting. "
                             "Lets you inspect the lifted pose before torque is dropped. Default 5.0s.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Connect, print the planned phase summary, do NOT command motion.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.skip_zero:
        args.pre_zero_s = 0.0

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
            print("\n[test_lift_arms] second Ctrl-C — exiting hard.", file=sys.stderr)
            sys.exit(1)
        abort["flag"] = True
        print("\n[test_lift_arms] Ctrl-C caught — disconnecting (torque will drop)...",
              file=sys.stderr)

    signal.signal(signal.SIGINT, _on_sigint)

    spine0_deg = np.rad2deg(LIFT_SPINE_RAD[0])
    spine_end_deg = np.rad2deg(LIFT_SPINE_RAD[-1])

    total_motion_s = args.pre_zero_s + args.pre_ramp_s + args.spine_s + args.hold_s
    print("[test_lift_arms] planned sequence:")
    print(f"    Phase 0  current -> zero pose                 {args.pre_zero_s:5.2f}s  (soft gains)")
    print(f"    Phase 1  zero pose -> spine[0]                {args.pre_ramp_s:5.2f}s  (soft gains)")
    print(f"    Phase 2  spine sweep -> spine[-1]             {args.spine_s:5.2f}s  (stiff gains)")
    print(f"    Phase 3  hold final pose                      {args.hold_s:5.2f}s  (stiff gains)")
    print(f"    + extra inspect-hold AFTER sequence           {args.final_hold_s:5.2f}s  (stiff gains)")
    print(f"    -------------------------------------------")
    print(f"    total motion time                              {total_motion_s:5.2f}s")
    print(f"    command rate                                   {args.hz:.1f} Hz")
    print()
    print("[test_lift_arms] connecting (left={}, right={})...".format(args.left_can, args.right_can))
    robot.connect(calibrate=False)
    try:
        start_deg = _read_pose_deg(robot)
        _print_pose("current pose (before lift)", start_deg)
        _print_pose("spine[0] target (after Phase 1)", spine0_deg)
        _print_pose("spine[-1] target (after Phase 2)", spine_end_deg)

        if args.dry_run:
            print("\n[test_lift_arms] --dry_run: not commanding any motion. Done.")
            return 0

        print(f"\n[test_lift_arms] running lift_arms_to_ready (total motion ~{total_motion_s:.1f}s)...")
        t0 = time.monotonic()
        lift_arms_to_ready(
            robot,
            pre_zero_s=args.pre_zero_s,
            pre_ramp_s=args.pre_ramp_s,
            spine_duration_s=args.spine_s,
            hz=args.hz,
            hold_s=args.hold_s,
            log_fn=logging.info,
        )
        dt = time.monotonic() - t0
        print(f"[test_lift_arms] lift_arms_to_ready returned after {dt:.2f}s.")

        final_deg = _read_pose_deg(robot)
        _print_pose("final pose (after lift)", final_deg)
        err = final_deg - spine_end_deg
        _print_pose("final error (actual - spine[-1])", err)
        max_err = float(np.max(np.abs(err)))
        print(f"\n  Max |error vs spine[-1]| = {max_err:.2f} deg")

        if args.final_hold_s > 0 and not abort["flag"]:
            print(f"\n[test_lift_arms] holding lifted pose {args.final_hold_s:.1f}s "
                  "(stiff gains — disconnect at end will drop torque)...")
            step_dt = 1.0 / args.hz
            action = {k: float(spine_end_deg[i]) for i, k in enumerate(_KEY_ORDER)}
            t_end = time.monotonic() + args.final_hold_s
            while time.monotonic() < t_end and not abort["flag"]:
                robot.send_action(action)
                time.sleep(step_dt)

        return 0
    finally:
        print("[test_lift_arms] disconnecting (torque will be disabled — arms will drop)...")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[test_lift_arms] disconnect raised: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
