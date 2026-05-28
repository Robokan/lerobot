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

"""Decode and print the Damiao DM-series MCU fault codes for every motor.

Damiao motor state CAN frames pack a 4-bit error/state code in the HIGH
nibble of byte 0 (the LOW nibble is the motor ID). lerobot's
``_decode_motor_state`` ignores that nibble entirely, so when the motor
LED is solid red and lerobot still says "Handshake successful" you can't
tell WHY the motor is red without a separate decoder.

This script:
  1. Opens both CAN buses (can2 = right arm, can3 = left arm)
  2. Pings every motor (DISABLE -> ENABLE -> read state response)
  3. Decodes byte 0's high nibble against the documented fault table
  4. Prints a per-motor verdict and a one-line global verdict

Code table (Damiao DM-J4310/J4340/J8009 manual, "CAN feedback frame"):

      0x0   Reset state (just powered on, not yet armed)
      0x1   Enabled (OK, MIT mode armed)
      0x2   Calibration in progress
      0x8   Overvoltage  (V_BUS > ~30 V)
      0x9   Undervoltage (V_BUS < ~18 V)
      0xA   Overcurrent
      0xB   MOSFET over-temperature
      0xC   Motor over-temperature
      0xD   Lost CAN-communication watchdog
      0xE   Overload (sustained high-torque)

If you see code 0x9 on every motor: the 24V supply is sagging.
If you see code 0xB or 0xC: a motor is thermally protected — let it cool
~5 minutes with 24V OFF.
If you see codes 0x0 / 0x1 on a motor whose LED is still red, the LED
state and the CAN feedback disagree; that usually means the firmware
has a hardware fault not exposed over CAN (e.g. a blown driver) and the
motor needs to be replaced.

Usage:
    python scripts/diag_motor_faults.py
    python scripts/diag_motor_faults.py --can can2 --motors 1-8
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import can

CAN_PARAM_ID = 0x7FF
CAN_CMD_DISABLE = 0xFD
CAN_CMD_ENABLE = 0xFC
CAN_CMD_REFRESH = 0xCC

# Same encoding as lerobot/motors/damiao/tables.py for DM4310/DM4340/DM8009
# (all three share PMAX=12.5 rad).
PMAX = 12.5

JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4",
               "joint_5", "joint_6", "joint_7", "gripper")
ALL_IDS = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)

CODE_TABLE = {
    0x0: ("RESET",        "Just powered on, NOT yet armed. ENABLE was not seen / acknowledged."),
    0x1: ("ENABLED (OK)", "MIT mode is armed. Motor should accept MIT control commands."),
    0x2: ("CAL_IN_PROG",  "Motor is busy with internal calibration / SET_ZERO write. Wait ~100 ms."),
    0x8: ("OVERVOLTAGE",  "V_BUS > ~30V. Check the 24V supply and any in-line caps/regulators."),
    0x9: ("UNDERVOLTAGE", "V_BUS < ~18V. 24V supply is sagging, current-limited, or wiring resistance is too high. THIS MUST BE FIXED IN HARDWARE."),
    0xA: ("OVERCURRENT",  "Sustained current above motor rating. Mechanical jam, motor stalled, or short circuit."),
    0xB: ("MOS_OVERTEMP", "The H-bridge MOSFETs are too hot. Power off 24V and wait 5+ minutes."),
    0xC: ("MOTOR_OVERTEMP","Rotor / windings too hot. Power off 24V and wait 5+ minutes."),
    0xD: ("LOST_COMM",    "CAN watchdog timeout. No control frames received within the watchdog window. Software bug or bus error."),
    0xE: ("OVERLOAD",     "Sustained high torque. Reduce load or kp."),
}


def _u_to_deg(uint16: int) -> float:
    pos_rad = float(uint16) / 65535.0 * (2 * PMAX) - PMAX
    return math.degrees(pos_rad)


def _send_simple(bus: can.Bus, motor_id: int, cmd_byte: int) -> None:
    bus.send(can.Message(
        arbitration_id=motor_id,
        data=[0xFF] * 7 + [cmd_byte],
        is_extended_id=False,
    ))


def _drain(bus: can.Bus, timeout_s: float = 0.05) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if bus.recv(timeout=0.01) is None:
            return


def _wait(bus: can.Bus, recv_id: int, timeout_s: float = 0.15) -> can.Message | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = bus.recv(timeout=0.02)
        if m and m.arbitration_id == recv_id:
            return m
    return None


def query_one(bus: can.Bus, motor_id: int) -> dict | None:
    """Send DISABLE -> ENABLE -> REFRESH, return decoded state and fault."""
    # Drain any prior frames so we definitely read the freshest reply.
    _drain(bus, 0.05)

    # First DISABLE: clears soft fault latch (if any). We ignore its reply.
    _send_simple(bus, motor_id, CAN_CMD_DISABLE)
    time.sleep(0.05)
    _drain(bus, 0.05)

    # ENABLE: brings the motor up. The response carries the post-enable status.
    _send_simple(bus, motor_id, CAN_CMD_ENABLE)
    msg = _wait(bus, motor_id + 0x10, timeout_s=0.2)
    if msg is None:
        return None

    data = bytes(msg.data)
    state_nibble = (data[0] & 0xF0) >> 4
    id_nibble = data[0] & 0x0F
    q_uint = (data[1] << 8) | data[2]
    pos_deg = _u_to_deg(q_uint)
    t_mos = data[6]
    t_rotor = data[7]
    return {
        "fault_code": state_nibble,
        "fault_name": CODE_TABLE.get(state_nibble, ("UNKNOWN", "unrecognized code"))[0],
        "fault_help": CODE_TABLE.get(state_nibble, ("UNKNOWN", "unrecognized code"))[1],
        "id_byte": id_nibble,
        "pos_deg": pos_deg,
        "t_mos_c": int(t_mos),
        "t_rotor_c": int(t_rotor),
        "raw_byte0": data[0],
    }


def scan_bus(channel: str, label: str, motor_ids: list[int]) -> dict[int, dict]:
    bus = can.interface.Bus(channel=channel, interface="socketcan",
                            bitrate=1_000_000)
    try:
        print(f"\n=== {label} ({channel}) ===")
        results: dict[int, dict] = {}
        for mid in motor_ids:
            joint_name = JOINT_NAMES[mid - 1] if 1 <= mid <= 8 else f"id_{mid}"
            res = query_one(bus, mid)
            if res is None:
                print(f"  {joint_name:>8} (0x{mid:02X})  NO RESPONSE "
                      f"(motor unpowered, CAN unplugged, or wrong ID)")
                results[mid] = {"fault_name": "NO_RESPONSE", "fault_code": -1}
                continue
            tag = res["fault_name"]
            marker = "OK" if res["fault_code"] == 0x1 else "FAULT"
            print(f"  {joint_name:>8} (0x{mid:02X})  [{marker:<5}]  "
                  f"fault=0x{res['fault_code']:X} {tag:<14}  "
                  f"pos={res['pos_deg']:+7.2f}deg  "
                  f"T_mos={res['t_mos_c']}C  T_rotor={res['t_rotor_c']}C")
            results[mid] = res
        return results
    finally:
        # Leave everything disabled so the arms don't lurch when the script exits.
        for mid in motor_ids:
            try:
                _send_simple(bus, mid, CAN_CMD_DISABLE)
            except Exception:
                pass
        time.sleep(0.1)
        bus.shutdown()


def _parse_motor_range(spec: str) -> list[int]:
    """'1-8', '1,3,5', or '1' -> [1,..8] / [1,3,5] / [1]."""
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="both",
                        help="CAN interface to scan: 'can2', 'can3', or "
                             "'both' for both arms. Default 'both'.")
    parser.add_argument("--motors", default="1-8",
                        help="Motor IDs to scan on each bus. Default '1-8' = all 8.")
    args = parser.parse_args()

    motor_ids = _parse_motor_range(args.motors)
    print(f"[diag_motor_faults] scanning motor IDs {motor_ids} on {args.can}")

    all_results: dict[str, dict[int, dict]] = {}
    if args.can in ("both", "can3"):
        all_results["can3 (LEFT)"] = scan_bus("can3", "LEFT  arm", motor_ids)
    if args.can in ("both", "can2"):
        all_results["can2 (RIGHT)"] = scan_bus("can2", "RIGHT arm", motor_ids)

    print()
    print("=" * 70)
    fault_counts: dict[str, int] = {}
    enabled_count = 0
    total = 0
    for _bus_label, res in all_results.items():
        for mid, info in res.items():
            total += 1
            name = info.get("fault_name", "NO_RESPONSE")
            fault_counts[name] = fault_counts.get(name, 0) + 1
            if name == "ENABLED (OK)":
                enabled_count += 1

    print(f"SUMMARY: {enabled_count}/{total} motors enabled (OK).")
    print("Fault code distribution:")
    for name, n in sorted(fault_counts.items(), key=lambda kv: -kv[1]):
        print(f"   {n:2d}x  {name}")

    print()
    if enabled_count == total:
        print("All motors are armed. Try `scripts/diag_single_motor_mit.py` next.")
    else:
        # Print remediation help for any fault we saw.
        unique_faults = {code for res in all_results.values()
                                for code, _name in [(info["fault_code"], info.get("fault_name", "?"))
                                                    for info in res.values()]}
        for code in sorted(unique_faults):
            if code in CODE_TABLE and code != 0x1:
                name, help_text = CODE_TABLE[code]
                print(f"FAULT 0x{code:X} ({name}):")
                print(f"   {help_text}")
                print()
    print("=" * 70)
    return 0 if enabled_count == total else 2


if __name__ == "__main__":
    sys.exit(main())
