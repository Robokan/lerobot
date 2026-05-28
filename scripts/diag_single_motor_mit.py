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

"""One-motor MIT diagnostic for Damiao DM-series motors.

Bypasses ALL of lerobot. Talks directly to ONE motor on a chosen CAN bus
using raw python-can frames. Wakes the motor (DISABLE -> ENABLE), reads
its current position, then sends a couple of MIT control frames with soft
gains commanding a small position delta. Re-reads position after each
command and prints the actual encoder reading.

Use this AFTER a hardware change (24V power-cycle, motor swap, etc.) to
confirm the motor is in MIT mode and responds to control commands. If
this script makes the motor visibly move a few degrees, the motor itself
is fine and the bug is elsewhere in the lerobot stack. If it doesn't,
the motor (or its firmware-stored Control_Mode register) is in a state
that won't accept MIT commands.

Usage:
    python scripts/diag_single_motor_mit.py
        --can can2          # right arm bus by default
        --motor 1           # joint_1 = the easiest one to see move
        --delta_deg 15      # how far to ask it to move
        --kp 80 --kd 2.0    # soft, safe gains

Stop:
    Ctrl-C at any time. The script always finishes with a DISABLE so the
    motor goes limp and the arm doesn't stay stiff.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import can

CAN_PARAM_ID = 0x7FF
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_REFRESH = 0xCC

# DM-series shared encoding limits (matches lerobot.motors.damiao.tables
# MOTOR_LIMIT_PARAMS for DM4310/DM4340/DM8009 — they all share PMAX=12.5).
PMAX = 12.5     # rad
VMAX = 30.0     # rad/s
TMAX = 10.0     # N*m
KP_MAX = 500.0  # MIT kp upper bound
KD_MAX = 5.0    # MIT kd upper bound


def _f_to_u(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(x_min, min(x_max, x))
    span = x_max - x_min
    return int((x - x_min) / span * ((1 << bits) - 1))


def _u_to_f(u: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    return float(u) / ((1 << bits) - 1) * span + x_min


def encode_mit(kp: float, kd: float, pos_rad: float,
               vel_rad_s: float = 0.0, tau_nm: float = 0.0) -> list[int]:
    """Encode an MIT-mode CAN frame: (kp, kd, pos, vel, tau) -> 8 bytes."""
    q = _f_to_u(pos_rad, -PMAX, PMAX, 16)
    dq = _f_to_u(vel_rad_s, -VMAX, VMAX, 12)
    kp_u = _f_to_u(kp, 0.0, KP_MAX, 12)
    kd_u = _f_to_u(kd, 0.0, KD_MAX, 12)
    tau_u = _f_to_u(tau_nm, -TMAX, TMAX, 12)
    data = [0] * 8
    data[0] = (q >> 8) & 0xFF
    data[1] = q & 0xFF
    data[2] = (dq >> 4) & 0xFF
    data[3] = ((dq & 0xF) << 4) | ((kp_u >> 8) & 0xF)
    data[4] = kp_u & 0xFF
    data[5] = (kd_u >> 4) & 0xFF
    data[6] = ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF)
    data[7] = tau_u & 0xFF
    return data


def decode_state(data: bytes) -> tuple[float, float, float, int, int]:
    """Decode an 8-byte motor response into (pos_deg, vel_deg_s, tau_nm, t_mos, t_rotor)."""
    if len(data) < 8:
        raise ValueError(f"short motor response: {len(data)} bytes")
    q = (data[1] << 8) | data[2]
    dq = (data[3] << 4) | (data[4] >> 4)
    tau = ((data[4] & 0xF) << 8) | data[5]
    pos_rad = _u_to_f(q, -PMAX, PMAX, 16)
    vel_rad_s = _u_to_f(dq, -VMAX, VMAX, 12)
    tau_nm = _u_to_f(tau, -TMAX, TMAX, 12)
    return (math.degrees(pos_rad), math.degrees(vel_rad_s), tau_nm,
            data[6], data[7])


def send_simple(bus: can.Bus, motor_id: int, cmd: int) -> None:
    bus.send(can.Message(
        arbitration_id=motor_id,
        data=[0xFF] * 7 + [cmd],
        is_extended_id=False,
    ))


def wait_for_response(bus: can.Bus, recv_id: int, timeout_s: float = 0.15) -> can.Message | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = bus.recv(timeout=0.02)
        if msg and msg.arbitration_id == recv_id:
            return msg
    return None


def drain(bus: can.Bus, timeout_s: float = 0.05) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline and bus.recv(timeout=0.01) is not None:
        pass


def probe_position(bus: can.Bus, motor_id: int) -> float | None:
    """Send a refresh request and decode the response position in degrees."""
    bus.send(can.Message(
        arbitration_id=CAN_PARAM_ID,
        data=[motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH,
              0, 0, 0, 0, 0],
        is_extended_id=False,
    ))
    msg = wait_for_response(bus, motor_id + 0x10, timeout_s=0.1)
    if msg is None:
        return None
    pos_deg, *_ = decode_state(bytes(msg.data))
    return pos_deg


def send_mit(bus: can.Bus, motor_id: int, kp: float, kd: float,
             pos_deg: float) -> tuple[float, float, float, int, int] | None:
    """Send one MIT control frame and return decoded response or None."""
    pos_rad = math.radians(pos_deg)
    data = encode_mit(kp, kd, pos_rad)
    bus.send(can.Message(
        arbitration_id=motor_id, data=data, is_extended_id=False,
    ))
    msg = wait_for_response(bus, motor_id + 0x10, timeout_s=0.15)
    if msg is None:
        return None
    return decode_state(bytes(msg.data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can2",
                        help="CAN interface (default: can2 = right arm)")
    parser.add_argument("--motor", type=int, default=1,
                        help="Damiao motor ID 1..8 (1=joint_1 ... 8=gripper). Default 1.")
    parser.add_argument("--delta_deg", type=float, default=15.0,
                        help="How many degrees to ask the motor to move from its current pose. Default 15.")
    parser.add_argument("--kp", type=float, default=80.0,
                        help="MIT kp gain (0..500). Default 80 (soft).")
    parser.add_argument("--kd", type=float, default=2.0,
                        help="MIT kd gain (0..5). Default 2.0.")
    parser.add_argument("--hold_s", type=float, default=1.5,
                        help="How long to hold the commanded position at 100 Hz before returning. Default 1.5s.")
    args = parser.parse_args()

    print(f"[diag] opening {args.can} (socketcan, 1 Mbps, classic)...")
    bus = can.interface.Bus(channel=args.can, interface="socketcan",
                            bitrate=1_000_000)
    try:
        motor_id = args.motor
        recv_id = motor_id + 0x10
        print(f"[diag] target motor: id=0x{motor_id:02X} (recv id 0x{recv_id:02X})")

        # 1. Drain any stale frames
        drain(bus, 0.1)

        # 2. DISABLE then ENABLE: clears any fault latch and re-arms MIT mode.
        print("[diag] sending DISABLE...")
        send_simple(bus, motor_id, CAN_CMD_DISABLE)
        drain(bus, 0.1)
        print("[diag] sending ENABLE...")
        send_simple(bus, motor_id, CAN_CMD_ENABLE)
        resp = wait_for_response(bus, recv_id, timeout_s=0.2)
        if resp is None:
            print("[diag] FAIL: motor did not respond to ENABLE. "
                  "Check 24V, CAN wiring, and that the motor ID is correct.",
                  file=sys.stderr)
            return 2
        pos0_deg, vel0, tau0, tmos0, trot0 = decode_state(bytes(resp.data))
        print(f"[diag] motor responded after ENABLE: pos={pos0_deg:+.3f} deg, "
              f"vel={vel0:+.3f} deg/s, tau={tau0:+.3f} Nm, "
              f"T_mos={tmos0} C, T_rotor={trot0} C")

        # 3. Probe via REFRESH to confirm READS still work after ENABLE
        pos_probe = probe_position(bus, motor_id)
        if pos_probe is None:
            print("[diag] WARNING: REFRESH (read-only) probe got no response.",
                  file=sys.stderr)
        else:
            print(f"[diag] REFRESH probe pos={pos_probe:+.3f} deg "
                  f"(should match ENABLE response above)")

        # 4. Send MIT command commanding pos0 + delta. Watch for motion.
        target_deg = pos0_deg + args.delta_deg
        print()
        print(f"[diag] sending MIT command: kp={args.kp} kd={args.kd} "
              f"target={target_deg:+.3f} deg (delta = {args.delta_deg:+.2f} deg)")
        print(f"[diag] WATCH THE MOTOR -- it should move ~{args.delta_deg:.1f} deg "
              f"over the next {args.hold_s:.1f}s")

        # Stream MIT commands at 100 Hz for the hold duration so the PD loop
        # has time to actually drive the motor to the target.
        step_dt = 0.01
        n_steps = max(1, int(args.hold_s / step_dt))
        last_print = 0.0
        for k in range(n_steps):
            t0 = time.perf_counter()
            decoded = send_mit(bus, motor_id, args.kp, args.kd, target_deg)
            elapsed = time.perf_counter() - t0
            if decoded is not None and (time.perf_counter() - last_print) > 0.25:
                pos, vel, tau, tmos, trot = decoded
                print(f"    t={k*step_dt:5.2f}s  pos={pos:+8.3f} deg  "
                      f"err={pos - target_deg:+7.3f}  vel={vel:+7.2f} deg/s  "
                      f"tau={tau:+6.3f} Nm")
                last_print = time.perf_counter()
            time.sleep(max(0.0, step_dt - elapsed))

        # 5. Return to original position
        print()
        print(f"[diag] returning motor to original pose {pos0_deg:+.3f} deg "
              f"over {args.hold_s:.1f}s...")
        n_steps = max(1, int(args.hold_s / step_dt))
        for k in range(n_steps):
            t0 = time.perf_counter()
            decoded = send_mit(bus, motor_id, args.kp, args.kd, pos0_deg)
            elapsed = time.perf_counter() - t0
            if decoded is not None and (time.perf_counter() - last_print) > 0.25:
                pos, vel, tau, tmos, trot = decoded
                print(f"    t={k*step_dt:5.2f}s  pos={pos:+8.3f} deg  "
                      f"err={pos - pos0_deg:+7.3f}  vel={vel:+7.2f} deg/s  "
                      f"tau={tau:+6.3f} Nm")
                last_print = time.perf_counter()
            time.sleep(max(0.0, step_dt - elapsed))

        # 6. Final readout: did the motor's encoder change at all relative to start?
        final = probe_position(bus, motor_id)
        if final is None:
            print("[diag] final REFRESH got no response.", file=sys.stderr)
        else:
            drift = final - pos0_deg
            print(f"[diag] final pos={final:+.3f} deg (drift vs start = {drift:+.3f} deg)")
            print()
            print("=" * 70)
            if abs(drift) < 0.5:
                print(f"RESULT: motor did NOT move. The encoder is responsive but the motor")
                print(f"is ignoring MIT commands (tau stayed near 0, position unchanged).")
                print(f"Likely cause: Control_Mode register not set to MIT mode after the")
                print(f"24V power-cycle, or motor is in a soft-disabled state.")
                print(f"Next step: power-cycle the motor again with logic power up FIRST,")
                print(f"then 24V, OR re-run the Damiao manufacturer config tool to verify")
                print(f"Control_Mode = MIT (=1).")
            else:
                print(f"RESULT: motor DID move {drift:+.1f} deg in response to MIT commands.")
                print(f"The motor itself is healthy. The lerobot motion path must be the bug.")
            print("=" * 70)
        return 0
    except KeyboardInterrupt:
        print("\n[diag] Ctrl-C caught", file=sys.stderr)
        return 130
    finally:
        # Always disable so the motor doesn't stay holding torque.
        print("[diag] sending final DISABLE (motor will go limp)...")
        try:
            send_simple(bus, args.motor, CAN_CMD_DISABLE)
            time.sleep(0.05)
        except Exception:
            pass
        bus.shutdown()


if __name__ == "__main__":
    sys.exit(main())
