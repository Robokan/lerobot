#!/usr/bin/env python
"""Capture a fresh motor-flash zero on both OpenArm followers.

Why this exists:
    OpenArmFollower.connect() used to call bus.set_zero_position() on
    every connect, which overwrote the motor flash's saved zero with
    whatever physical pose the arms happened to be in when the operator
    started the script. That made every session start with a different
    encoder reference and broke the "go to default pose" step of
    lift_arms. The connect-time re-zero has been removed (see
    src/lerobot/robots/openarm_follower/openarm_follower.py), so once
    you've captured a clean zero with this script the motors REMEMBER
    that zero across power cycles and across lerobot-record runs.

What this script does:
    1. Connects directly to can2 + can3 via raw python-can (no lerobot
       wrappers, no policy load, no cameras).
    2. Sends DISABLE+ENABLE to every motor to clear any fault latches
       (specifically the gripper motors that fault-latch easily).
    3. Disables torque on every motor so the arms go limp.
    4. Prompts you to physically position the arms in the default pose:
         - arms hanging straight down
         - wrists neutral (no roll/twist)
         - grippers fully closed
       and press ENTER when ready.
    5. Sends CAN_CMD_SET_ZERO (0xFE) to every motor. This writes the
       current encoder reading to the motor's flash as the new zero.
    6. Probes every motor again; all positions should now read ~0.0 deg
       (since we just defined the current pose as zero).
    7. Re-disables torque so the arms stay limp on exit (no surprise
       movement when you let go).

Usage:
    .venv/bin/python scripts/recalibrate_zero.py
"""

from __future__ import annotations

import math
import sys
import time

import can

LEFT_CAN = "can3"
RIGHT_CAN = "can2"

JOINT_NAMES = (
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "joint_7", "gripper",
)
JOINT_IDS = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)

# Damiao protocol
CAN_PARAM_ID = 0x7FF
CAN_CMD_DISABLE = 0xFD
CAN_CMD_ENABLE = 0xFC
CAN_CMD_SET_ZERO = 0xFE
CAN_CMD_REFRESH = 0xCC

# DM motor position encoding range (rad) - same for all DM-series motors used here
PMAX = 12.5


def _u_to_deg(uint16: int) -> float:
    """Decode the 16-bit position field from a Damiao state response."""
    pos_rad = float(uint16) / 65535.0 * (2 * PMAX) - PMAX
    return math.degrees(pos_rad)


def _send_simple(bus: can.Bus, motor_id: int, cmd_byte: int) -> None:
    bus.send(
        can.Message(
            arbitration_id=motor_id,
            data=[0xFF] * 7 + [cmd_byte],
            is_extended_id=False,
        )
    )


def _drain(bus: can.Bus, timeout_s: float = 0.05) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if bus.recv(timeout=0.01) is None:
            return


def wake_motor(bus: can.Bus, motor_id: int) -> bool:
    """DISABLE+ENABLE sequence to clear any fault latch. Returns True if the
    motor responded to ENABLE."""
    _send_simple(bus, motor_id, CAN_CMD_DISABLE)
    time.sleep(0.03)
    _drain(bus, 0.05)
    _send_simple(bus, motor_id, CAN_CMD_ENABLE)
    deadline = time.time() + 0.15
    while time.time() < deadline:
        m = bus.recv(timeout=0.02)
        if m and m.arbitration_id == motor_id + 0x10:
            return True
    return False


def probe(bus: can.Bus, motor_id: int) -> float | None:
    bus.send(
        can.Message(
            arbitration_id=CAN_PARAM_ID,
            data=[motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0],
            is_extended_id=False,
        )
    )
    deadline = time.time() + 0.08
    while time.time() < deadline:
        m = bus.recv(timeout=0.02)
        if m and m.arbitration_id == motor_id + 0x10:
            q_uint = (m.data[1] << 8) | m.data[2]
            return _u_to_deg(q_uint)
    return None


def step_1_wake_all(bus: can.Bus, label: str) -> list[str]:
    print(f"  [{label}] waking all motors (DISABLE+ENABLE)...")
    missing: list[str] = []
    for name, mid in zip(JOINT_NAMES, JOINT_IDS):
        ok = wake_motor(bus, mid)
        if not ok:
            missing.append(name)
            print(f"    {name:>8}: DID NOT WAKE")
        else:
            print(f"    {name:>8}: woke OK")
    return missing


def step_2_disable_all(bus: can.Bus, label: str) -> None:
    print(f"  [{label}] disabling torque on all motors (arms will be limp)...")
    for mid in JOINT_IDS:
        _send_simple(bus, mid, CAN_CMD_DISABLE)
        time.sleep(0.02)
    _drain(bus, 0.1)


def step_3_save_zero(bus: can.Bus, label: str) -> None:
    print(f"  [{label}] saving current encoder reading as motor flash zero...")
    for name, mid in zip(JOINT_NAMES, JOINT_IDS):
        _send_simple(bus, mid, CAN_CMD_SET_ZERO)
        time.sleep(0.02)
    _drain(bus, 0.1)
    print(f"  [{label}] zero saved to flash. Verifying...")


def step_4_verify(bus: can.Bus, label: str) -> bool:
    """All positions should read approximately 0 after a fresh SET_ZERO."""
    print(f"  [{label}] verifying positions (should all be ~0 deg now):")
    all_ok = True
    for name, mid in zip(JOINT_NAMES, JOINT_IDS):
        pos = probe(bus, mid)
        if pos is None:
            print(f"    {name:>8}: NO RESPONSE")
            all_ok = False
            continue
        marker = "ok" if abs(pos) < 0.5 else "OFF"
        if marker == "OFF":
            all_ok = False
        print(f"    {name:>8}: {pos:+7.3f} deg  [{marker}]")
    return all_ok


def run_one(channel: str, label: str, prompt_phase: bool) -> bool:
    bus = can.interface.Bus(channel=channel, interface="socketcan", bitrate=1_000_000)
    try:
        print(f"\n=== {label} ({channel}) ===")
        # Step 1 - wake everyone (clears any fault latch from a stalled gripper)
        missing = step_1_wake_all(bus, label)
        if missing:
            print(f"  WARNING: {label}: motors did not wake: {missing}")
            print(f"  Check 24V power and CAN wiring. Continuing anyway with the rest.")

        # Step 2 - go limp so the operator can move the arms by hand
        step_2_disable_all(bus, label)
        return True
    finally:
        bus.shutdown()


def save_and_verify(channel: str, label: str) -> bool:
    bus = can.interface.Bus(channel=channel, interface="socketcan", bitrate=1_000_000)
    try:
        print(f"\n=== {label} ({channel}): saving + verifying ===")
        # Re-wake (some motors return to a sleep state after disable+a few seconds)
        step_1_wake_all(bus, label)
        # Critical: motors must be DISABLED when we send SET_ZERO so the flash
        # write captures the operator-held physical pose, not whatever pose the
        # PD loop would otherwise drift toward.
        step_2_disable_all(bus, label)
        step_3_save_zero(bus, label)
        ok = step_4_verify(bus, label)
        # Leave motors disabled so the arms don't lurch when the script exits.
        step_2_disable_all(bus, label)
        return ok
    finally:
        bus.shutdown()


def main() -> int:
    print("Recalibration of motor flash zero on both OpenArm followers.")
    print("This will go limp, prompt you to position the arms, and save the")
    print("operator-held pose as the new encoder zero permanently.\n")

    # Phase A - wake + go limp on both arms so the operator can hand-position them.
    run_one(LEFT_CAN, "LEFT  arm", prompt_phase=False)
    run_one(RIGHT_CAN, "RIGHT arm", prompt_phase=False)

    print("\n" + "=" * 70)
    print("ARMS ARE NOW LIMP.")
    print("Physically position BOTH arms in the default pose:")
    print("    * arms hanging straight down (joint_1..7 all relaxed to neutral)")
    print("    * wrists neutral (no twist)")
    print("    * BOTH grippers FULLY CLOSED")
    print("Then press ENTER (here in this terminal) to capture this pose as zero.")
    print("=" * 70)
    try:
        input("> press ENTER when arms are positioned: ")
    except (KeyboardInterrupt, EOFError):
        print("\nAborted; no zero saved.")
        return 1

    # Phase B - save zero on both arms and verify
    ok_left = save_and_verify(LEFT_CAN, "LEFT  arm")
    ok_right = save_and_verify(RIGHT_CAN, "RIGHT arm")

    print("\n" + "=" * 70)
    if ok_left and ok_right:
        print("SUCCESS: both arms now have a clean flash zero.")
        print("The arms are LIMP. You may now run the policy with:")
        print("    bash scripts/run_chocolate_policy.sh")
    else:
        print("PARTIAL SUCCESS: some motors did not reach ~0 deg after SET_ZERO.")
        print("Check the per-motor verification above and re-run if needed.")
    print("=" * 70)
    return 0 if (ok_left and ok_right) else 2


if __name__ == "__main__":
    sys.exit(main())
