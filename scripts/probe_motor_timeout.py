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

"""Probe the Damiao TIMEOUT register (address 9) on every motor.

Why this matters:
    Damiao DM-series motors have a communication-watchdog register at
    address 9 (named TIMEOUT). If the motor does not receive a control
    command within TIMEOUT (in milliseconds) of the previous one, the
    firmware auto-disables the motor and drops torque. This is a safety
    feature: if the controller crashes or wedges, the arm goes limp
    instead of holding whatever last pose it was commanded to.

    The same feature kills a policy-driven setup the first time the
    policy's inference takes longer than the watchdog window:
      - lift_arms streams MIT frames at 50 Hz (every 20ms) -> motors
        stay armed and hold the lifted pose at the end.
      - Then the policy fires its FIRST inference; on cold start that
        can be ~1000ms (compilation, warm-up).
      - During that 1000ms no MIT frames are sent.
      - If TIMEOUT < 1000ms, motors auto-disable, gravity wins, arms
        drop. The user sees "arms lifted then immediately fell after
        the loop started" -- which is exactly what happened.

This script:
    1. Reads the TIMEOUT register (address 9) on every motor on can3
       and can2 via the Damiao param protocol (cmd 0x33, write 0x55).
       NO writes -- read-only probe.
    2. Reads CTRL_MODE (address 10) at the same time as a sanity check.
    3. Prints both per motor, plus a global recommendation.

Usage:
    .venv/bin/python scripts/probe_motor_timeout.py
    .venv/bin/python scripts/probe_motor_timeout.py --can can3
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

import can

CAN_PARAM_ID = 0x7FF
READ_REGISTER_CODE = 0x33

REG_TIMEOUT = 9
REG_CTRL_MODE = 10

CTRL_MODE_NAMES = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "TORQUE_POS"}

JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4",
               "joint_5", "joint_6", "joint_7", "gripper")
ALL_IDS = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)


def _drain(bus: can.Bus, timeout_s: float = 0.05) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if bus.recv(timeout=0.01) is None:
            return


def read_register_int(bus: can.Bus, slave_id: int, reg: int,
                       timeout_s: float = 0.2) -> int | None:
    data = struct.pack("<HBBBBBB", slave_id, READ_REGISTER_CODE, reg,
                        0, 0, 0, 0)
    bus.send(can.Message(arbitration_id=CAN_PARAM_ID, data=data,
                          is_extended_id=False))
    master_id = slave_id + 16
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = bus.recv(timeout=0.02)
        if m is None:
            continue
        if m.arbitration_id != master_id:
            continue
        payload = bytes(m.data)
        if len(payload) < 8:
            continue
        try:
            _, _, _, value = struct.unpack("<HBBI", payload)
            return int(value)
        except struct.error:
            return None
    return None


def scan(channel: str, label: str, slave_ids: list[int]) -> None:
    bus = can.interface.Bus(channel=channel, interface="socketcan",
                             bitrate=1_000_000)
    try:
        print(f"\n=== {label} ({channel}) ===")
        print(f"  {'joint':<8} {'CTRL_MODE':>11}  {'TIMEOUT (ms)':>15}")
        any_timeout_small = False
        for sid in slave_ids:
            joint = JOINT_NAMES[sid - 1] if 1 <= sid <= 8 else f"id_{sid}"
            _drain(bus, 0.03)
            ctrl = read_register_int(bus, sid, REG_CTRL_MODE)
            _drain(bus, 0.03)
            timeout = read_register_int(bus, sid, REG_TIMEOUT)

            ctrl_str = (f"{ctrl} [{CTRL_MODE_NAMES.get(ctrl, '?')}]"
                        if ctrl is not None else "NO_REPLY")
            timeout_str = f"{timeout}" if timeout is not None else "NO_REPLY"
            note = ""
            if timeout is not None and 0 < timeout < 2000:
                note = "  <-- too short for cold policy inference (~1s)"
                any_timeout_small = True
            elif timeout is None:
                note = "  <-- no reply, motor unreachable"
            print(f"  {joint:<8} {ctrl_str:>11}  {timeout_str:>15}{note}")
        if any_timeout_small:
            print(f"  {label}: at least one motor has TIMEOUT < 2000ms;")
            print(f"  these motors WILL auto-disable during the first policy")
            print(f"  inference (~1s warmup) and drop the arms. Fix by setting")
            print(f"  TIMEOUT to 0 (no watchdog) or a large value (e.g. 5000ms).")
    finally:
        bus.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="both", choices=("both", "can2", "can3"))
    parser.add_argument("--motors", default="1-8",
                        help="Motor IDs to probe. Default 1-8.")
    args = parser.parse_args()

    slave_ids: list[int] = []
    for token in args.motors.split(","):
        if "-" in token:
            a, b = token.split("-")
            slave_ids.extend(range(int(a), int(b) + 1))
        else:
            slave_ids.append(int(token))
    slave_ids = sorted(set(slave_ids))

    if args.can in ("both", "can3"):
        scan("can3", "LEFT  arm", slave_ids)
    if args.can in ("both", "can2"):
        scan("can2", "RIGHT arm", slave_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
