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

"""Force all Damiao motors into MIT control mode and save to flash.

Why this exists:
    Damiao DM-series motors have a CTRL_MODE register (address 10) that
    determines which CAN frame format the motor accepts as a control
    command:
        1 = MIT mode      (frame ID = slave_id, payload = kp+kd+pos+vel+tau)
        2 = POS_VEL mode  (frame ID = slave_id + 0x100, payload = pos+vel)
        3 = VEL mode      (frame ID = slave_id + 0x200, payload = vel)
        4 = TORQUE_POS    (frame ID = slave_id + 0x300, payload = pos+force)

    The OpenArm setup tools (`python -m openarm.damiao ...`) configure
    motors in POS_VEL mode by default. lerobot's DamiaoMotorsBus, however,
    ONLY speaks MIT — `_mit_control_batch` sends frames on ID=slave_id
    with the MIT payload. A motor in POS_VEL mode silently ignores those
    frames because they don't match its expected ID / format. End result:
    handshake succeeds, encoder reads work, ENABLE is acknowledged
    (`fault=0x1 ENABLED`), but no torque is ever applied no matter how
    long we stream commands — the motor never even sees them as a control
    command. This script fixes that.

What it does:
    1. Connects to both CAN buses (can3 = left arm, can2 = right arm) via
       raw python-can. No lerobot wrappers.
    2. For every motor (slave IDs 1..8 on each bus):
        a. Reads CTRL_MODE (register 10) and prints the current value.
        b. If not already MIT (=1), writes 1 to CTRL_MODE.
        c. Sends SAVE_PARAMETERS (0xAA) so the change persists across
           power cycles.
        d. Re-reads CTRL_MODE to verify.
    3. Prints a per-motor + global verdict.

Damiao register protocol (per openarm/openarm/damiao/encoding.py):

    Write int register:  send to 0x7FF, data = struct.pack(
                              "<HBBI", slave_id, 0x55, reg_addr, value)
    Read register:       send to 0x7FF, data = struct.pack(
                              "<HBBBBBB", slave_id, 0x33, reg_addr,
                              0x00, 0x00, 0x00, 0x00)
    Save to flash:       send to 0x7FF, data = struct.pack(
                              "<HBBBBBB", slave_id, 0xAA, 0x00,
                              0x00, 0x00, 0x00, 0x00)
    Response (all):      arbitration_id = slave_id + 16 (master ID)
                         data = struct.pack("<HBBI", slave_id, cmd,
                                            reg_addr, value)

Usage:
    .venv/bin/python scripts/set_motors_to_mit_mode.py
    .venv/bin/python scripts/set_motors_to_mit_mode.py --dry-run
        (probes only — does NOT write any registers)
    .venv/bin/python scripts/set_motors_to_mit_mode.py --no-save
        (writes CTRL_MODE in RAM but does NOT save to flash; resets on
        the next 24V power cycle)
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

import can

CAN_PARAM_ID = 0x7FF
WRITE_REGISTER_CODE = 0x55
READ_REGISTER_CODE = 0x33
SAVE_PARAMETERS_CODE = 0xAA

REG_CTRL_MODE = 10

CTRL_MODE_NAMES = {
    1: "MIT",
    2: "POS_VEL",
    3: "VEL",
    4: "TORQUE_POS",
}

JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4",
               "joint_5", "joint_6", "joint_7", "gripper")
ALL_SLAVE_IDS = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)


def _drain(bus: can.Bus, timeout_s: float = 0.05) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if bus.recv(timeout=0.01) is None:
            return


def _wait_master_reply(bus: can.Bus, master_id: int,
                        timeout_s: float = 0.2) -> can.Message | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = bus.recv(timeout=0.02)
        if m and m.arbitration_id == master_id:
            return m
    return None


def read_register_int(bus: can.Bus, slave_id: int, reg_addr: int,
                       timeout_s: float = 0.2) -> int | None:
    """Read an integer register from a Damiao motor. Returns None on no reply."""
    data = struct.pack("<HBBBBBB", slave_id, READ_REGISTER_CODE, reg_addr,
                        0, 0, 0, 0)
    bus.send(can.Message(arbitration_id=CAN_PARAM_ID, data=data,
                          is_extended_id=False))
    master_id = slave_id + 16
    msg = _wait_master_reply(bus, master_id, timeout_s)
    if msg is None:
        return None
    payload = bytes(msg.data)
    if len(payload) < 8:
        return None
    try:
        _, _, _, value = struct.unpack("<HBBI", payload)
    except struct.error:
        return None
    return int(value)


def write_register_int(bus: can.Bus, slave_id: int, reg_addr: int,
                        value: int, timeout_s: float = 0.2) -> bool:
    """Write an integer register. Returns True if the motor acknowledged."""
    data = struct.pack("<HBBI", slave_id, WRITE_REGISTER_CODE, reg_addr,
                        int(value))
    bus.send(can.Message(arbitration_id=CAN_PARAM_ID, data=data,
                          is_extended_id=False))
    master_id = slave_id + 16
    msg = _wait_master_reply(bus, master_id, timeout_s)
    return msg is not None


def save_to_flash(bus: can.Bus, slave_id: int,
                   timeout_s: float = 0.5) -> bool:
    """Send SAVE_PARAMETERS (0xAA) so register changes persist across power
    cycles. Per Damiao protocol the flash write takes ~50-200ms; we wait up
    to 0.5s for the ack."""
    data = struct.pack("<HBBBBBB", slave_id, SAVE_PARAMETERS_CODE, 0,
                        0, 0, 0, 0)
    bus.send(can.Message(arbitration_id=CAN_PARAM_ID, data=data,
                          is_extended_id=False))
    master_id = slave_id + 16
    msg = _wait_master_reply(bus, master_id, timeout_s)
    return msg is not None


def process_one_motor(bus: can.Bus, slave_id: int, *, dry_run: bool,
                       save_flash: bool) -> tuple[bool, int | None, int | None]:
    """Returns (ok, mode_before, mode_after).

    ok == True   means motor is in MIT after we're done (or already was).
    mode_before  is the CTRL_MODE register value before any write.
    mode_after   is the CTRL_MODE register value after the write (None if
                 we didn't write).
    """
    joint = JOINT_NAMES[slave_id - 1]
    _drain(bus, 0.05)

    mode_before = read_register_int(bus, slave_id, REG_CTRL_MODE)
    mode_before_name = (CTRL_MODE_NAMES.get(mode_before, f"UNKNOWN(0x{mode_before:X})")
                        if mode_before is not None else "NO_REPLY")
    print(f"  {joint:>8} (0x{slave_id:02X})  "
          f"current CTRL_MODE = {mode_before} [{mode_before_name}]",
          end="")

    if mode_before is None:
        print("    [SKIP - no reply on read]")
        return (False, None, None)

    if mode_before == 1:
        print("    -> already MIT, no change")
        return (True, mode_before, mode_before)

    if dry_run:
        print(f"    -> would write 1 (MIT){' + save' if save_flash else ''}")
        return (False, mode_before, None)

    print(f"    -> writing 1 (MIT)", end="")
    ok_write = write_register_int(bus, slave_id, REG_CTRL_MODE, 1)
    if not ok_write:
        print("    [FAIL - write not acked]")
        return (False, mode_before, None)
    print("    [ok]", end="")

    if save_flash:
        print("    saving to flash", end="")
        ok_save = save_to_flash(bus, slave_id)
        if not ok_save:
            print("    [WARN - save not acked, change will be lost on next power cycle]")
        else:
            print("    [ok]", end="")

    _drain(bus, 0.1)
    mode_after = read_register_int(bus, slave_id, REG_CTRL_MODE)
    mode_after_name = (CTRL_MODE_NAMES.get(mode_after, f"UNKNOWN(0x{mode_after:X})")
                       if mode_after is not None else "NO_REPLY")
    print(f"    verify: now {mode_after} [{mode_after_name}]")
    return (mode_after == 1, mode_before, mode_after)


def scan_and_fix_bus(channel: str, label: str, slave_ids: list[int],
                      *, dry_run: bool, save_flash: bool) -> int:
    bus = can.interface.Bus(channel=channel, interface="socketcan",
                             bitrate=1_000_000)
    try:
        print(f"\n=== {label} ({channel}) ===")
        ok_count = 0
        for sid in slave_ids:
            ok, _before, _after = process_one_motor(
                bus, sid, dry_run=dry_run, save_flash=save_flash,
            )
            if ok:
                ok_count += 1
        return ok_count
    finally:
        bus.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="both",
                        choices=("both", "can2", "can3"),
                        help="CAN bus(es) to act on. Default: both.")
    parser.add_argument("--motors", default="1-8",
                        help="Motor slave IDs to act on. Default 1-8.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read CTRL_MODE on every motor but do NOT write.")
    parser.add_argument("--no-save", action="store_true",
                        help="Write CTRL_MODE in RAM only. Does NOT save to flash "
                             "(change will revert on the next 24V power cycle).")
    args = parser.parse_args()

    slave_ids: list[int] = []
    for token in args.motors.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-")
            slave_ids.extend(range(int(a), int(b) + 1))
        else:
            slave_ids.append(int(token))
    slave_ids = sorted(set(slave_ids))

    print(f"[set_mit_mode] target slave_ids = {slave_ids}")
    print(f"[set_mit_mode] dry_run = {args.dry_run}, save_to_flash = {not args.no_save}")
    if args.dry_run:
        print("[set_mit_mode] DRY RUN: reading CTRL_MODE only, no writes.")

    save_flash = not args.no_save

    bus_specs: list[tuple[str, str]] = []
    if args.can in ("both", "can3"):
        bus_specs.append(("can3", "LEFT  arm"))
    if args.can in ("both", "can2"):
        bus_specs.append(("can2", "RIGHT arm"))

    total_motors = 0
    total_ok = 0
    for chan, lab in bus_specs:
        ok_n = scan_and_fix_bus(chan, lab, slave_ids,
                                 dry_run=args.dry_run, save_flash=save_flash)
        total_motors += len(slave_ids)
        total_ok += ok_n

    print()
    print("=" * 70)
    print(f"SUMMARY: {total_ok}/{total_motors} motors are now in MIT control mode.")
    print("=" * 70)

    if total_ok == total_motors and not args.dry_run:
        print()
        print("Next steps:")
        print("  1. Re-verify with:")
        print("       python scripts/diag_single_motor_mit.py --can can2 --motor 1 --delta_deg 15")
        print("     The right-arm shoulder pan should now visibly swing ~15 deg.")
        print("  2. If that works, the lift sequence will too:")
        print("       python scripts/test_lift_arms.py")

    return 0 if (total_ok == total_motors) else 2


if __name__ == "__main__":
    sys.exit(main())
