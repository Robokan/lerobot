#!/usr/bin/env python
"""Show the REAL follower arms' pose on the MuJoCo model — read-only.

Why: the VR teleop (vr_mocap) solves IK on the MuJoCo OpenArm model and sends
the resulting joint angles to the robot as-is (degrees -> radians, no sign flips
or offsets; see MujocoBiOpenArm.send_action). That is only safe on hardware if
the model's zero pose and joint directions match the real arm's calibration.
This script checks exactly that: move the unpowered arm by hand and watch the
model follow. Any joint that turns the wrong way, or sits at an offset, shows
up immediately — with no motor ever enabled.

Safety: the only CAN frame this script sends is the Damiao REFRESH status
request. The bus is opened with handshake=False (the handshake sends ENABLE),
the follower's connect() is never called (it enables torque), and the bus is
closed with disable_torque=False. A guard wraps the CAN bus's send() and
aborts if anything other than a refresh request is about to go out.

Usage (bring the buses up first; follower = LUMPA):
    uv run python scripts/openarm_shadow.py --right can0 --left can1
    uv run python scripts/openarm_shadow.py --right can0              # one arm
    uv run python scripts/openarm_shadow.py --right can0 --no-viewer  # table only

The table shows each joint in degrees, the model's range for it, and a flag
when the real angle is outside the model's range or the real follower's limit.

Zero offsets (the real arm's zero error, fed to the follower as zero_offsets):
    # arms hanging straight down, grippers closed, wrists held neutral:
    uv run python scripts/openarm_shadow.py --right can0 --left can1 --save-offsets
    # then check the corrected pose: the model should now hang straight too
    uv run python scripts/openarm_shadow.py --right can0 --left can1 --offsets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from lerobot.motors.damiao.damiao import CAN_CMD_REFRESH, CAN_PARAM_ID
from lerobot.robots.mujoco_bi_openarm.mujoco_bi_openarm import (
    ARM_JOINT_NAMES,
    gripper_deg_to_m,
)
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
)

DEFAULT_MODEL = os.path.expanduser("~/sparkpack/openarm_mujoco/v1/scene.xml")
DEFAULT_OFFSETS = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/openarm_follower/lumpa_zero_offsets.json"
)
# Gripper: SparkJAX's measured open/closed, as used by scripts/run_vr_real.sh.
REAL_LIMITS = {
    "right": {**RIGHT_DEFAULT_JOINTS_LIMITS, "gripper": (-157.0, 5.0)},
    "left": {**LEFT_DEFAULT_JOINTS_LIMITS, "gripper": (-157.0, 5.0)},
}


def guard_bus(bus) -> None:
    """Refuse to send anything but a REFRESH status request."""
    send = bus.canbus.send

    def guarded(msg, *a, **kw):
        d = bytes(msg.data)
        if not (msg.arbitration_id == CAN_PARAM_ID and len(d) >= 3 and d[2] == CAN_CMD_REFRESH):
            raise RuntimeError(
                f"openarm_shadow: blocked non-refresh CAN frame id=0x{msg.arbitration_id:X} data={d.hex()}"
            )
        return send(msg, *a, **kw)

    bus.canbus.send = guarded


def open_arm(side: str, port: str):
    cfg = OpenArmFollowerConfig(id=f"lumpa_follower_{side}", port=port, side=side, use_can_fd=False)
    bus = OpenArmFollower(cfg).bus  # constructing it touches no hardware
    bus.connect(handshake=False)
    guard_bus(bus)
    return bus


def read_arm(bus) -> tuple[dict[str, float], list[str]]:
    """One refresh round. Returns positions (deg) and the motors that did not answer."""
    import can

    motors = list(bus.motors)
    try:
        for m in motors:
            bus.canbus.send(bus_msg(bus, bus._get_motor_id(m)))
    except can.CanOperationError:
        # Nothing ACKs on the bus (arm unplugged / unpowered): the TX queue fills.
        pos = {m: float(bus._last_known_states[m]["position"]) for m in motors}
        return pos, motors
    want = [bus._get_motor_recv_id(m) for m in motors]
    got = bus._recv_all_responses(want, timeout=0.02)
    missing = []
    for m in motors:
        msg = got.get(bus._get_motor_recv_id(m))
        if msg is None:
            missing.append(m)
        else:
            bus._process_response(m, msg)
    pos = {m: float(bus._last_known_states[m]["position"]) for m in motors}
    return pos, missing


def bus_msg(bus, motor_id: int):
    import can

    data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
    return can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=bus.use_can_fd)


def save_offsets(buses, path: str) -> int:
    """Average 2 s of readings per joint (arm joints only) and write them as offsets."""
    samples = {side: {m: [] for m in ARM_JOINT_NAMES} for side in buses}
    for _ in range(40):
        for side, bus in buses.items():
            pos, missing = read_arm(bus)
            if missing:
                print(f"[shadow] {side}: no reply from {missing}; not saving")
                return 1
            for m in ARM_JOINT_NAMES:
                samples[side][m].append(pos[m])
        time.sleep(0.05)
    out = {}
    if os.path.isfile(path):
        with open(path) as f:
            out = json.load(f)  # keep the other arm's offsets when saving one
    for side, per in samples.items():
        spread = {m: max(v) - min(v) for m, v in per.items()}
        if max(spread.values()) > 1.0:
            print(f"[shadow] {side}: arm moved during capture ({spread}); hold it still and retry")
            return 1
        out[side] = {m: round(float(np.mean(v)), 2) for m, v in per.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[shadow] wrote {path}:")
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--right", help="CAN interface of the right follower arm (e.g. can0)")
    ap.add_argument("--left", help="CAN interface of the left follower arm (e.g. can1)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument(
        "--offsets",
        nargs="?",
        const=DEFAULT_OFFSETS,
        default=None,
        help="subtract saved zero offsets (default file: %(const)s)",
    )
    ap.add_argument(
        "--save-offsets",
        nargs="?",
        const=DEFAULT_OFFSETS,
        default=None,
        help="average 2 s of readings in the zero pose, write them as offsets, exit",
    )
    args = ap.parse_args()
    ports = {s: p for s, p in (("right", args.right), ("left", args.left)) if p}
    if not ports:
        ap.error("give --right and/or --left")

    import mujoco

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    qadr, rng = {}, {}
    for side in ports:
        for i, name in enumerate(ARM_JOINT_NAMES):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i + 1}")
            qadr[(side, name)] = int(model.jnt_qposadr[jid])
            rng[(side, name)] = np.degrees(model.jnt_range[jid])
        fingers = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint{k}")
            for k in (1, 2)
        ]
        qadr[(side, "gripper")] = [int(model.jnt_qposadr[j]) for j in fingers if j >= 0]

    offsets = {}
    if args.offsets:
        with open(args.offsets) as f:
            offsets = json.load(f)
        print(f"[shadow] applying zero offsets from {args.offsets}")

    buses = {}
    try:
        for side, port in ports.items():
            buses[side] = open_arm(side, port)
            print(f"[shadow] {side} arm on {port}: bus open, read-only")

        if args.save_offsets:
            return save_offsets(buses, args.save_offsets)

        viewer = None
        if not args.no_viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(model, data)

        period = 1.0 / args.hz
        while viewer is None or viewer.is_running():
            t0 = time.perf_counter()
            lines = []
            for side, bus in buses.items():
                pos, missing = read_arm(bus)
                pos = {m: v - offsets.get(side, {}).get(m, 0.0) for m, v in pos.items()}
                for name in ARM_JOINT_NAMES:
                    deg = pos[name]
                    data.qpos[qadr[(side, name)]] = math.radians(deg)
                for a in qadr[(side, "gripper")]:
                    data.qpos[a] = gripper_deg_to_m(pos["gripper"])
                lines.append(f"{side.upper():5s}  {'joint':8s} {'real°':>8s}   model range     real limit")
                for name in [*ARM_JOINT_NAMES, "gripper"]:
                    deg = pos[name]
                    rl = REAL_LIMITS[side][name]
                    if name == "gripper":
                        mr = "   (-157..5)  "
                        flag = ""
                    else:
                        lo, hi = rng[(side, name)]
                        mr = f"{lo:7.0f}..{hi:<5.0f}"
                        flag = " OUTSIDE MODEL" if not (lo - 0.5 <= deg <= hi + 0.5) else ""
                    if not (rl[0] - 0.5 <= deg <= rl[1] + 0.5):
                        flag += " OUTSIDE REAL LIMIT"
                    if name in missing:
                        flag += " NO REPLY"
                    lines.append(f"       {name:8s} {deg:8.1f}   {mr}  {rl[0]:6.0f}..{rl[1]:<5.0f}{flag}")
            mujoco.mj_forward(model, data)
            if viewer is not None:
                viewer.sync()
            sys.stdout.write(
                "\x1b[H\x1b[2J" + "\n".join(lines) + "\n\nCtrl-C (or close the viewer) to quit.\n"
            )
            sys.stdout.flush()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        for bus in buses.values():
            bus.disconnect(disable_torque=False)
        print("[shadow] buses closed (no motor was ever enabled).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
