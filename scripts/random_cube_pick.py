#!/usr/bin/env python3
"""Randomize the table cube and script a bimanual OpenArm pick.

Each trial:
  1. Place a reachable cube; choose left/right; park the other arm
  2. Move over the cube (open gripper, level)
  3. Pitch the wrist and put the fingertip midpoint around the cube
  4. Close, then lift

Usage:
  source .venv/bin/activate
  MUJOCO_GL=glx MUJOCO_SAFE_EXIT_AFTER_VIEWER=1 DISPLAY=:1 \\
    python scripts/random_cube_pick.py --trials 5
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.robots.mujoco_bi_openarm import (
    FINGER_OPEN_M,
    MujocoBiOpenArm,
    MujocoBiOpenArmConfig,
    apply_base_pose,
    gripper_m_to_deg,
)
from lerobot.utils.robot_utils import precise_sleep

TABLE_TOP_Z = 0.37
CUBE_HALF = 0.025
CUBE_Z = TABLE_TOP_Z + CUBE_HALF

# Full table workspace (keep a margin for the 5 cm cube). Neg-y = body right.
CUBE_X_RANGE = (0.26, 0.48)
CUBE_Y_RANGE = (-0.32, 0.32)

HOVER_CLEARANCE = 0.08
# Tip-mid height above cube center. Measured at the pitched grasp pose: the pad
# face extends 2.5 cm below the estimated tip, so +3.0 cm here made the pads
# grip only the cube's TOP 2 cm. +1.0 cm centers the contact patch on the
# cube's middle (pad bottom stays ~1.5 cm above the table).
GRASP_CLEARANCE = 0.010
# Extra grasp height applied on a placement RETRY: at some cube spots the
# mid-cube depth is geometrically unreachable (the finger corner catches the
# table first, deterministically). Upper-middle grip beats a failed episode.
_GRASP_Z_RELIEF = {"m": 0.0}
# Horizontal inset so the cube sits a bit inside the jaws (XY only).
GRASP_INSET_M = 0.010
GRASP_PITCH_RAD = math.radians(25.0)
# Commanded opening while squeezing a 5 cm cube (pads block at ~25-30 mm).
# Commanding only 3-4 mm past the block point gives kp*err ≈ 3 N per finger --
# soft enough that a 1 mm cube shift unloads one pad and the cube pops out
# (traced: grip snapped 26.6 -> 5.2 mm at 18% of a lift). 13 mm past block is
# ~10 N per finger, still far inside the ±333 N force limit.
GRASP_HOLD_M = 0.012
# Live opening at or below this means the gripper has finished closing on the cube.
GRASP_CLOSED_M = 0.032
# Commanded-opening ramp rate. Stepping the target straight to the goal slams
# the pads shut at whatever the position servo can deliver; ramping the target
# closes gently and lets the cube settle between the pads instead of being
# batted. 30 mm/s: full open -> hold in ~1.1 s.
GRIP_RAMP_MPS = 0.060
PITCH_DURATION_S = 1.3
# Lift the cube to double its own height (5 cm cube -> +10 cm), then hold.
LIFT_CLEARANCE = 0.10
LIFT_HOLD_S = 1.0
SUCCESS_CUBE_Z = CUBE_Z + 0.05
# True finger bottom hangs ~3.2 cm below the estimated tip-mid (measured), so
# table + 0.035 put the pads exactly AT the table. 0.042 keeps ~1 cm of air.
MIN_TIP_Z = TABLE_TOP_Z + 0.033
GRASP_TIP_TOL = 0.015
TRANSIT_CLEARANCE = 0.10
APPROACH_SPEED_MPS = 0.24
LIFT_DURATION_S = 1.1
REACH_TIP_TOL = 0.035

# Quiet idle seeds for IK / teleport.
RIGHT_IDLE_DEG = np.array([40.0, 35.0, 0.0, 85.0, 0.0, 15.0, 0.0])
LEFT_IDLE_DEG = np.array([-40.0, 35.0, 0.0, 85.0, 0.0, 15.0, 0.0])
# Side parks (mirrored): joint1 sign flips so the unused arm sits low beside the torso,
# not overhead. Left uses +74°, right uses −74°.
RIGHT_PARK_DEG = np.array([-74.0, 0.0, 0.0, 140.0, 0.0, 0.0, 0.0])
LEFT_PARK_DEG = np.array([74.0, 0.0, 0.0, 140.0, 0.0, 0.0, 0.0])

# TCP axes for a table-parallel pinch (OpenArm hang pose).
#   TCP x → world +Z, TCP y → world −Y, TCP z → world +X.
_LEVEL_R0 = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
)


@dataclass(frozen=True)
class ArmSpec:
    """Per-arm names, park/idle poses, and start tip ranges."""

    side: str
    idle_deg: np.ndarray
    park_deg: np.ndarray
    tuck_deg: np.ndarray
    start_x: tuple[float, float]
    start_y: tuple[float, float]
    start_z: tuple[float, float] = (0.46, 0.64)

    @property
    def other(self) -> str:
        return "left" if self.side == "right" else "right"

    @property
    def joints(self) -> list[str]:
        return [f"openarm_{self.side}_joint{i}" for i in range(1, 8)]

    @property
    def tcp_body(self) -> str:
        return f"openarm_{self.side}_hand_tcp"

    @property
    def finger_geoms(self) -> tuple[str, str]:
        return (
            f"openarm_{self.side}_right_finger_collision",
            f"openarm_{self.side}_left_finger_collision",
        )

    @property
    def hand_geom(self) -> str:
        """Collision mesh on the hand body the fingers slide out of."""
        return f"openarm_{self.side}_hand_collision"

    @property
    def geom_prefix(self) -> str:
        return f"openarm_{self.side}"


# In-episode retreat pose for the arm NOT picking: like its side park, rotated
# a bit forward so the hand hovers just over the table's near edge
# (FK: hand ~(0.22, ±0.15, 0.47)). Stays by the robot's side — never swings
# outside the table.
RIGHT_TUCK_DEG = np.array([-50.0, 5.0, 0.0, 125.0, 0.0, 0.0, 0.0])
LEFT_TUCK_DEG = np.array([50.0, 5.0, 0.0, 125.0, 0.0, 0.0, 0.0])

RIGHT_ARM = ArmSpec(
    side="right",
    idle_deg=RIGHT_IDLE_DEG,
    park_deg=RIGHT_PARK_DEG,
    tuck_deg=RIGHT_TUCK_DEG,
    start_x=(0.32, 0.48),
    start_y=(-0.30, -0.10),
)
LEFT_ARM = ArmSpec(
    side="left",
    idle_deg=LEFT_IDLE_DEG,
    park_deg=LEFT_PARK_DEG,
    tuck_deg=LEFT_TUCK_DEG,
    start_x=(0.32, 0.48),
    start_y=(0.10, 0.30),
)
ARMS = (RIGHT_ARM, LEFT_ARM)
ARMS_BY_SIDE = {arm.side: arm for arm in ARMS}


def level_rot(yaw: float = 0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ _LEVEL_R0


def grasp_rot(yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """Table-parallel pinch, then pitch about TCP Y (pitch>0 tips down)."""
    c, s = math.cos(-pitch), math.sin(-pitch)
    ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return level_rot(yaw) @ ry


def _mat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    angle = math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        return np.zeros(3)
    return axis / n * angle


def clamp_tip_target(tip_mid_target: np.ndarray) -> np.ndarray:
    t = np.asarray(tip_mid_target, dtype=float).copy()
    t[2] = max(float(t[2]), MIN_TIP_Z)
    return t


def _geom_name(robot: MujocoBiOpenArm, gid: int) -> str:
    import mujoco

    return mujoco.mj_id2name(robot._model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""


@dataclass
class ContactHit:
    geom_a: str
    geom_b: str
    dist: float  # MuJoCo: negative = penetration
    force_n: float  # contact force magnitude (N)


# Name of the body/geom the grasp machinery targets. The cube sim leaves this
# as "cube"; other scenarios (e.g. the chocolate-bars sim) retarget it per pick
# via set_target_body() — geoms are matched by substring, so body and geom
# names must share this string.
_TARGET_BODY = "cube"


def set_target_body(name: str) -> None:
    global _TARGET_BODY
    _TARGET_BODY = name


def contacts_between(
    robot: MujocoBiOpenArm, a_substr: str, b_substr: str
) -> list[ContactHit]:
    """Active MuJoCo contacts whose geom names contain the two substrings."""
    import mujoco

    hits: list[ContactHit] = []
    for i in range(robot._data.ncon):
        c = robot._data.contact[i]
        g1, g2 = _geom_name(robot, c.geom1), _geom_name(robot, c.geom2)
        if not (
            (a_substr in g1 and b_substr in g2) or (a_substr in g2 and b_substr in g1)
        ):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(robot._model, robot._data, i, force)
        hits.append(
            ContactHit(
                geom_a=g1,
                geom_b=g2,
                dist=float(c.dist),
                force_n=float(np.linalg.norm(force[:3])),
            )
        )
    hits.sort(key=lambda h: h.dist)
    return hits


def arm_hits_table(robot: MujocoBiOpenArm, arm: ArmSpec) -> list[ContactHit]:
    return contacts_between(robot, arm.geom_prefix, "table")


def arm_hits_cube(robot: MujocoBiOpenArm, arm: ArmSpec) -> list[ContactHit]:
    return contacts_between(robot, arm.geom_prefix, _TARGET_BODY)


def finger_hits_cube(robot: MujocoBiOpenArm, arm: ArmSpec) -> list[ContactHit]:
    """Contacts between this arm's finger collision pads and the cube."""
    return [
        h
        for h in arm_hits_cube(robot, arm)
        if "finger" in h.geom_a or "finger" in h.geom_b
    ]


def hand_hits_cube(robot: MujocoBiOpenArm, arm: ArmSpec) -> list[ContactHit]:
    """Contacts between the hand/palm body (gripper mount) and the cube."""
    return contacts_between(robot, f"{arm.side}_hand", _TARGET_BODY)


def hand_hits_table(robot: MujocoBiOpenArm, arm: ArmSpec) -> list[ContactHit]:
    return contacts_between(robot, f"{arm.side}_hand", "table")


def grasp_contact_summary(robot: MujocoBiOpenArm, arm: ArmSpec) -> dict[str, object]:
    """Structured view of gripper↔cube / hand↔cube / table contact state."""
    cube_hits = arm_hits_cube(robot, arm)
    finger_hits = finger_hits_cube(robot, arm)
    hand_cube = hand_hits_cube(robot, arm)
    hand_table = hand_hits_table(robot, arm)
    table_hits = arm_hits_table(robot, arm)
    left_pad = any("left_finger" in h.geom_a or "left_finger" in h.geom_b for h in finger_hits)
    right_pad = any(
        "right_finger" in h.geom_a or "right_finger" in h.geom_b for h in finger_hits
    )
    # Both finger pads pinching the cube is the signal we want before lift.
    pinching = left_pad and right_pad
    return {
        "cube": cube_hits,
        "fingers_cube": finger_hits,
        "hand_cube": hand_cube,
        "hand_table": hand_table,
        "table": table_hits,
        "left_pad_on_cube": left_pad,
        "right_pad_on_cube": right_pad,
        "hand_on_cube": bool(hand_cube),
        "pinching": pinching,
        "cube_force_n": float(sum(h.force_n for h in finger_hits)),
        "hand_force_n": float(sum(h.force_n for h in hand_cube)),
        "deepest_cube_mm": (
            float(min(h.dist for h in finger_hits) * 1000.0) if finger_hits else 0.0
        ),
    }


def format_contacts(hits: list[ContactHit], limit: int = 4) -> str:
    if not hits:
        return "none"
    parts = [
        f"{h.geom_a.split('openarm_')[-1]}↔{h.geom_b} "
        f"d={h.dist * 1000:.1f}mm F={h.force_n:.1f}N"
        for h in hits[:limit]
    ]
    extra = f" (+{len(hits) - limit})" if len(hits) > limit else ""
    return "; ".join(parts) + extra


def log_grasp_contacts(robot: MujocoBiOpenArm, arm: ArmSpec, label: str) -> dict[str, object]:
    s = grasp_contact_summary(robot, arm)
    pinch = "PINCH" if s["pinching"] else "no-pinch"
    hand = "HAND" if s["hand_on_cube"] else "hand-clear"
    print(
        f"  contacts[{label}]: {pinch} {hand} "
        f"pads L={s['left_pad_on_cube']} R={s['right_pad_on_cube']} "
        f"cubeF={s['cube_force_n']:.1f}N handF={s['hand_force_n']:.1f}N "
        f"deepest={s['deepest_cube_mm']:.1f}mm"
    )
    print(f"    fingers↔cube: {format_contacts(s['fingers_cube'])}")  # type: ignore[arg-type]
    print(f"    hand↔cube:    {format_contacts(s['hand_cube'])}")  # type: ignore[arg-type]
    print(f"    hand↔table:   {format_contacts(s['hand_table'])}")  # type: ignore[arg-type]
    print(f"    arm↔table:    {format_contacts(s['table'])}")  # type: ignore[arg-type]
    return s


def gripper_hits_table(robot: MujocoBiOpenArm, arm: ArmSpec) -> list[ContactHit]:
    """Hand body or finger pads contacting the table (wrist/gripper collision)."""
    return [
        h
        for h in arm_hits_table(robot, arm)
        if "hand" in h.geom_a
        or "hand" in h.geom_b
        or "finger" in h.geom_a
        or "finger" in h.geom_b
    ]


def grasp_table_fault(robot: MujocoBiOpenArm, arm: ArmSpec) -> tuple[str, str] | None:
    """Hard fail on wrist/hand↔table, or a hard finger dig into the table."""
    hand = hand_hits_table(robot, arm)
    if hand:
        return hand[0].geom_a, hand[0].geom_b
    fingers = [
        h
        for h in gripper_hits_table(robot, arm)
        if ("finger" in h.geom_a or "finger" in h.geom_b) and h.force_n >= 15.0
    ]
    if fingers:
        return fingers[0].geom_a, fingers[0].geom_b
    return None


def finger_table_graze(robot: MujocoBiOpenArm, arm: ArmSpec, min_force_n: float = 1.0) -> bool:
    """A finger pad lightly touching the table — below the hard-fault threshold.

    The descent loops treat this as \"stop going down / climb a little\" so force
    never builds to the 15 N fault that aborts the whole trial.
    """
    return any(
        ("finger" in h.geom_a or "finger" in h.geom_b) and h.force_n >= min_force_n
        for h in gripper_hits_table(robot, arm)
    )


class PositionOnlyIK:
    """DLS IK for tip-mid reaching with optional table-parallel orientation."""

    def __init__(self, model, data, arm: ArmSpec):
        import mujoco

        self.model = model
        self.data = data
        self.arm = arm
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm.joints]
        self.qadr = [model.jnt_qposadr[j] for j in self.jids]
        self.dadr = [model.jnt_dofadr[j] for j in self.jids]
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.tcp_body)
        self.hand_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, arm.hand_geom)
        self.tip_gids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in arm.finger_geoms
        ]
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])

    def q(self) -> np.ndarray:
        return np.array([self.data.qpos[i] for i in self.qadr])

    def set_q(self, q: np.ndarray) -> None:
        import mujoco

        for i, qi in zip(self.qadr, q, strict=True):
            self.data.qpos[i] = float(qi)
        mujoco.mj_forward(self.model, self.data)

    def ee(self) -> np.ndarray:
        return self.data.xpos[self.body].copy()

    def rot(self) -> np.ndarray:
        return self.data.xmat[self.body].reshape(3, 3).copy()

    def tip_mid(self) -> np.ndarray:
        return finger_tips_from_data(self.model, self.data, self.arm).mean(axis=0)

    def ori_err(self, yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
        return _mat_to_axis_angle(grasp_rot(yaw, pitch) @ self.rot().T)

    def level_err(self, yaw: float = 0.0) -> np.ndarray:
        return self.ori_err(yaw, pitch=0.0)

    def step(
        self,
        target: np.ndarray,
        max_dq: float = math.radians(5),
        damp: float = 1e-2,
        yaw: float | None = None,
        pitch: float = 0.0,
        ori_weight: float = 0.35,
        hold_proximal: np.ndarray | None = None,
        proximal_scale: float = 1.0,
    ) -> float:
        import mujoco

        err = target - self.ee()
        n = float(np.linalg.norm(err))
        if n > 1e-9:
            err = err * min(1.0, 0.025 / n)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.body)
        jp = jacp[:, self.dadr]
        if yaw is None:
            j = jp
            dx = err
        else:
            ori = self.ori_err(yaw, pitch)
            on = float(np.linalg.norm(ori))
            if on > 1e-9:
                ori = ori * min(1.0, 0.2 / on)
            jr = jacr[:, self.dadr]
            j = np.vstack([jp, ori_weight * jr])
            dx = np.concatenate([err, ori_weight * ori])
        dq = j.T @ np.linalg.solve(j @ j.T + damp**2 * np.eye(j.shape[0]), dx)
        dq = np.clip(dq, -max_dq, max_dq)
        if hold_proximal is not None:
            dq[:4] = 0.0
        else:
            dq[:4] *= float(np.clip(proximal_scale, 0.0, 1.0))
        q = np.clip(self.q() + dq, self.lo, self.hi)
        if hold_proximal is not None:
            q[:4] = hold_proximal
        q[3] = max(q[3], math.radians(8.0))
        self.set_q(q)
        return float(np.linalg.norm(target - self.ee()))

    def step_tip_mid(
        self,
        tip_mid_target: np.ndarray,
        max_dq: float = math.radians(5),
        damp: float = 1e-2,
        yaw: float = 0.0,
        pitch: float = 0.0,
        level: bool = True,
        freeze_wrist: np.ndarray | None = None,
        hold_proximal: np.ndarray | None = None,
        proximal_scale: float = 1.0,
    ) -> float:
        tip_mid_target = clamp_tip_target(tip_mid_target)
        tcp_target = self.ee() + (tip_mid_target - self.tip_mid())
        tcp_target[2] = max(float(tcp_target[2]), TABLE_TOP_Z + 0.015)
        self.step(
            tcp_target,
            max_dq=max_dq,
            damp=damp,
            yaw=None if freeze_wrist is not None else (yaw if level else None),
            pitch=pitch,
            ori_weight=0.45 if abs(pitch) > 1e-6 else 0.25,
            hold_proximal=None if freeze_wrist is not None else hold_proximal,
            proximal_scale=proximal_scale,
        )
        if freeze_wrist is not None:
            q = self.q()
            q[4:7] = freeze_wrist
            self.set_q(q)
        return float(np.linalg.norm(self.tip_mid() - tip_mid_target))


def _arm_q_real(robot: MujocoBiOpenArm, ik: PositionOnlyIK) -> np.ndarray:
    """Actual joint angles straight from sim state (cheap; no camera renders)."""
    return np.array([float(robot._data.qpos[a]) for a in ik.qadr])


def _arm_q_from_obs(obs: dict, side: str) -> np.ndarray:
    return np.array([obs[f"{side}_joint_{i}.pos"] * math.pi / 180.0 for i in range(1, 8)])


# Per-tick retreat rate for the arm that is NOT picking: it starts at a random
# pose like the active arm and drifts back to its tucked side pose while the
# pick happens (12 deg/s at 30 fps — deliberate, unhurried).
_RETREAT_STEP_RAD = math.radians(0.8)
_OTHER_GRIP: dict[str, float] = {}
# Where the idle arm retreats this trial: its half-tuck by default, or the full
# park when the cube spawned too close to the half-tuck spot to be safe.
_RETREAT_TARGET: dict[str, np.ndarray] = {}
# Half-tuck hand position (xy) per side, from the IK plan behind *_TUCK_DEG.
_TUCK_TIP_XY = {"right": np.array([0.22, -0.15]), "left": np.array([0.22, 0.15])}
_TUCK_CLEARANCE_M = 0.22


def _park_action(side: str) -> dict[str, float]:
    """One slow step of the unused arm toward its retreat pose (half-tuck over
    the table edge, or the full park when the cube spawned too close)."""
    park = _RETREAT_TARGET.get(side)
    if park is None:
        park = np.deg2rad(ARMS_BY_SIDE[side].tuck_deg)
    cur = _LAST_CMD.get(side)
    if cur is None:
        cur = park.copy()
    q = cur + np.clip(park - cur, -_RETREAT_STEP_RAD, _RETREAT_STEP_RAD)
    _LAST_CMD[side] = q
    grip = _OTHER_GRIP.get(side, 0.0)
    grip = max(0.0, grip - GRIP_RAMP_MPS / 30.0)
    _OTHER_GRIP[side] = grip
    return {f"{side}_joint_{i}.pos": float(math.degrees(q[i - 1])) for i in range(1, 8)} | {
        f"{side}_gripper.pos": gripper_m_to_deg(grip)
    }


def set_cube_xy(robot: MujocoBiOpenArm, x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    import mujoco

    model, data = robot._model, robot._data
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    if jid < 0:
        raise RuntimeError("cube_free joint missing — use openarm_mujoco/v1/scene.xml")
    adr = model.jnt_qposadr[jid]
    dof = model.jnt_dofadr[jid]
    data.qpos[adr : adr + 3] = [x, y, CUBE_Z]
    data.qpos[adr + 3 : adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    return cube_pos(robot)


def cube_pos(robot: MujocoBiOpenArm) -> np.ndarray:
    import mujoco

    bid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_BODY, _TARGET_BODY)
    return robot._data.xpos[bid].copy()


def vertical_tilt_rad(ik: PositionOnlyIK) -> float:
    """Angle of the hand's down-pointing axis off vertical (0 = fingers vertical).

    Unlike level_err this excludes yaw error, which is harmless to finger
    height — gating translation on yaw-inclusive error deadlocks centering
    whenever the wrist is still winding toward a yawed grasp.
    """
    return float(math.acos(np.clip(ik.rot()[2, 0], -1.0, 1.0)))


def cube_yaw(robot: MujocoBiOpenArm) -> float:
    """Cube yaw folded to (-45°, 45°] — the nearest face-aligned grasp yaw.

    The pads must align with the cube's faces: a 5 cm square rotated 24°
    presents a 66 mm support width to axis-aligned pads (50 mm at 0°), which
    reads as \"never closed\" to the 32 mm closed-check.
    """
    import mujoco

    bid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_BODY, _TARGET_BODY)
    R = robot._data.xmat[bid].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4


def tcp_pos(robot: MujocoBiOpenArm, arm: ArmSpec) -> np.ndarray:
    import mujoco

    bid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_BODY, arm.tcp_body)
    return robot._data.xpos[bid].copy()


def _mesh_tip_world(model, data, finger_gid: int, hand_gid: int) -> np.ndarray:
    """Pad-side fingertip point: farthest from the hand in the horizontal plane.

    Euclidean-farthest mesh verts are the pitched lower corners and dig into the
    table; horizontal reach keeps tip-mid at grasp height on the pad faces.
    """
    import mujoco

    if model.geom_type[finger_gid] != mujoco.mjtGeom.mjGEOM_MESH:
        return data.geom_xpos[finger_gid].copy()
    mid = int(model.geom_dataid[finger_gid])
    vadr = int(model.mesh_vertadr[mid])
    vnum = int(model.mesh_vertnum[mid])
    verts = model.mesh_vert[vadr : vadr + vnum]
    xpos = data.geom_xpos[finger_gid]
    xmat = data.geom_xmat[finger_gid].reshape(3, 3)
    world = xpos + (xmat @ verts.T).T
    hand = data.geom_xpos[hand_gid]
    d_xy = np.linalg.norm(world[:, :2] - hand[:2], axis=1)
    # Among far verts, prefer those near the finger geom mid-height (pad), not
    # the table-scraping corner.
    far = d_xy >= float(np.percentile(d_xy, 85.0))
    candidates = np.flatnonzero(far)
    z_err = np.abs(world[candidates, 2] - float(xpos[2]))
    return world[int(candidates[int(np.argmin(z_err))])].copy()


def finger_tips_from_data(model, data, arm: ArmSpec) -> np.ndarray:
    import mujoco

    hand_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, arm.hand_geom)
    if hand_gid < 0:
        raise RuntimeError(f"geom {arm.hand_geom} not found")
    tips = []
    for gname in arm.finger_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            raise RuntimeError(f"geom {gname} not found")
        tips.append(_mesh_tip_world(model, data, gid, hand_gid))
    return np.asarray(tips)


def finger_tips_world(robot: MujocoBiOpenArm, arm: ArmSpec) -> np.ndarray:
    return finger_tips_from_data(robot._model, robot._data, arm)


def finger_lowest_z(robot: MujocoBiOpenArm, arm: ArmSpec) -> list[float]:
    """True lowest mesh-vertex height of each finger — what actually hits the table."""
    import mujoco

    model, data = robot._model, robot._data
    out = []
    for gname in arm.finger_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        mid = int(model.geom_dataid[gid])
        vadr, vnum = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[vadr : vadr + vnum]
        world_z = data.geom_xpos[gid][2] + (data.geom_xmat[gid].reshape(3, 3) @ verts.T).T[:, 2]
        out.append(float(world_z.min()))
    return out


def tip_mid_world(robot: MujocoBiOpenArm, arm: ArmSpec) -> np.ndarray:
    return finger_tips_world(robot, arm).mean(axis=0)


def hand_pos_world(robot: MujocoBiOpenArm, arm: ArmSpec) -> np.ndarray:
    import mujoco

    gid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_GEOM, arm.hand_geom)
    return robot._data.geom_xpos[gid].copy()


def grasp_tip_target(robot: MujocoBiOpenArm, arm: ArmSpec, cube: np.ndarray) -> np.ndarray:
    """Tip-mid goal: above cube center, inset horizontally so the cube nests in-hand.

    Inset is XY-only. Applying it in 3D along hand→tip pulled Z down under pitch
    and drove the wrist into the table.
    """
    tip = tip_mid_world(robot, arm)
    hand = hand_pos_world(robot, arm)
    out_xy = tip[:2] - hand[:2]
    cube = cube.copy()
    cube[2] += _GRASP_Z_RELIEF["m"]
    n = float(np.linalg.norm(out_xy))
    target = np.asarray(cube, dtype=float).copy()
    if n > 1e-6:
        target[:2] = target[:2] + (out_xy / n) * GRASP_INSET_M
    target[2] = float(cube[2] + GRASP_CLEARANCE)
    return clamp_tip_target(target)


def tips_straddle_cube(robot: MujocoBiOpenArm, arm: ArmSpec, cube: np.ndarray) -> tuple[bool, str]:
    tips = finger_tips_world(robot, arm)
    mid = tips.mean(axis=0)
    axis = tips[0] - tips[1]
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-6:
        return False, "tips coinciding"
    axis = axis / axis_n
    s0 = float(np.dot(tips[0] - cube, axis))
    s1 = float(np.dot(tips[1] - cube, axis))
    # With grasp inset, tip-mid sits past the cube; judge centering in the pad plane.
    mid_err_xy = float(np.linalg.norm(mid[:2] - cube[:2]))
    imbalance = abs(s0 + s1)
    opposite = s0 * s1 < 0.0
    wide_enough = abs(s0) > 0.012 and abs(s1) > 0.012
    ok = opposite and wide_enough and mid_err_xy < 0.022 and imbalance < 0.022
    msg = (
        f"tip-mid xy_err={mid_err_xy * 100:.1f} cm, "
        f"side offsets=({s0 * 100:.1f}, {s1 * 100:.1f}) cm, "
        f"sep={axis_n * 100:.1f} cm"
    )
    return ok, msg


def build_ik(robot: MujocoBiOpenArm, arm: ArmSpec) -> PositionOnlyIK:
    import mujoco

    path = str(Path(robot.config.model_path).expanduser())
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    apply_base_pose(mujoco, model, data, 90.0)
    for fi in (1, 2):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm.side}_finger_joint{fi}")
        data.qpos[model.jnt_qposadr[jid]] = FINGER_OPEN_M
    mujoco.mj_forward(model, data)
    return PositionOnlyIK(model, data, arm)


# Last commanded arm target per side. Phases must chain their command streams
# from this — re-seeding a phase from *observation* snaps the command backward
# by the PD tracking lag at every phase boundary, which is exactly the jerk
# visible at each transition.
_LAST_CMD: dict[str, np.ndarray] = {}


def _cmd_seed(robot: MujocoBiOpenArm, side: str) -> np.ndarray:
    """Where a new phase's command stream should start: the previous phase's
    last command (continuity), falling back to observation only when no
    command has been issued since the last teleport."""
    q = _LAST_CMD.get(side)
    if q is not None:
        return q.copy()
    return _arm_q_from_obs(robot.get_observation(), side)


def send_q(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    grip_m: float,
) -> None:
    """Command active arm from IK; keep the other arm parked."""
    q = ik.q()
    _LAST_CMD[ik.arm.side] = np.asarray(q, dtype=float).copy()
    action = _park_action(ik.arm.other)
    for i, qi in enumerate(q, start=1):
        action[f"{ik.arm.side}_joint_{i}.pos"] = float(math.degrees(qi))
    action[f"{ik.arm.side}_gripper.pos"] = gripper_m_to_deg(grip_m)
    robot.send_action(action)
    if _RECORDER is not None:
        _RECORDER.tick(action)


def _set_arm_qpos(robot: MujocoBiOpenArm, side: str, q_rad: np.ndarray) -> None:
    import mujoco

    model, data = robot._model, robot._data
    for i, qi in enumerate(q_rad, start=1):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}")
        data.qpos[model.jnt_qposadr[jid]] = float(qi)
        data.qvel[model.jnt_dofadr[jid]] = 0.0


def _set_gripper_qpos(robot: MujocoBiOpenArm, side: str, grip_m: float) -> None:
    import mujoco

    model, data = robot._model, robot._data
    for fi in (1, 2):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint{fi}")
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        dof = model.jnt_dofadr[jid]
        data.qpos[adr] = float(grip_m)
        data.qvel[dof] = 0.0


def zero_sim_velocity(robot: MujocoBiOpenArm) -> None:
    robot._data.qvel[:] = 0.0
    robot._data.qacc[:] = 0.0


def teleport_arms(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    q_active: np.ndarray,
    grip_m: float,
) -> None:
    """Park the unused arm at its side pose; snap the active arm to ``q_active``."""
    import mujoco

    other = ARMS_BY_SIDE[ik.arm.other]
    _set_arm_qpos(robot, other.side, np.deg2rad(other.park_deg))
    _set_gripper_qpos(robot, other.side, 0.0)
    _set_arm_qpos(robot, ik.arm.side, q_active)
    _set_gripper_qpos(robot, ik.arm.side, grip_m)
    zero_sim_velocity(robot)
    mujoco.mj_forward(robot._model, robot._data)
    ik.set_q(q_active)
    _LAST_CMD[ik.arm.side] = np.asarray(q_active, dtype=float).copy()
    _LAST_CMD[other.side] = np.deg2rad(other.park_deg)


def park_both_arms(robot: MujocoBiOpenArm, iks: dict[str, PositionOnlyIK]) -> None:
    """Snap both arms to their low side-park poses."""
    import mujoco

    for arm in ARMS:
        _set_arm_qpos(robot, arm.side, np.deg2rad(arm.park_deg))
        _set_gripper_qpos(robot, arm.side, 0.0)
    zero_sim_velocity(robot)
    mujoco.mj_forward(robot._model, robot._data)
    for arm in ARMS:
        iks[arm.side].set_q(np.deg2rad(arm.park_deg))
        _LAST_CMD[arm.side] = np.deg2rad(arm.park_deg)


def settle_pose(
    robot: MujocoBiOpenArm, ik: PositionOnlyIK, grip_m: float, fps: int, hold_s: float = 0.25
) -> None:
    n = max(2, int(hold_s * fps))
    for _ in range(n):
        zero_sim_velocity(robot)
        t0 = time.perf_counter()
        send_q(robot, ik, grip_m)
        zero_sim_velocity(robot)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def random_start_tip(arm: ArmSpec, rng: np.random.Generator) -> np.ndarray:
    return clamp_tip_target(
        np.array(
            [
                float(rng.uniform(*arm.start_x)),
                float(rng.uniform(*arm.start_y)),
                float(rng.uniform(*arm.start_z)),
            ]
        )
    )


def plan_q_to_tip_mid(
    ik: PositionOnlyIK,
    q_start: np.ndarray,
    tip_mid_target: np.ndarray,
    max_iters: int = 1000,
    tol: float = 0.01,
    accept_err: float | None = None,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray | None:
    tip_mid_target = clamp_tip_target(tip_mid_target)
    accept = tol * 2.5 if accept_err is None else accept_err
    ik.set_q(q_start.copy())
    best_q = ik.q().copy()
    best_err = float(np.linalg.norm(ik.tip_mid() - tip_mid_target))
    for _ in range(max_iters):
        err = ik.step_tip_mid(tip_mid_target, yaw=yaw, pitch=pitch, level=True)
        if err < best_err:
            best_err = err
            best_q = ik.q().copy()
        if err <= tol and float(np.linalg.norm(ik.ori_err(yaw, pitch))) < 0.35:
            return ik.q().copy()
    ik.set_q(best_q)
    tilt = float(np.linalg.norm(ik.ori_err(yaw, pitch)))
    return best_q if best_err < accept and tilt < 0.6 else None


def plan_q_to_tip_mid_robust(
    ik: PositionOnlyIK,
    q_start: np.ndarray,
    tip_mid_target: np.ndarray,
    rng: np.random.Generator | None = None,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray | None:
    tip_mid_target = clamp_tip_target(tip_mid_target)
    seeds = [q_start.copy(), np.deg2rad(ik.arm.idle_deg)]
    for dq4, dq6 in ((0.0, 0.0), (0.2, 0.15), (-0.1, -0.2), (0.25, -0.1), (0.1, 0.25)):
        q = q_start.copy()
        q[3] = float(np.clip(q[3] + dq4, ik.lo[3], ik.hi[3]))
        q[5] = float(np.clip(q[5] + dq6, ik.lo[5], ik.hi[5]))
        seeds.append(q)
    if rng is not None:
        for _ in range(4):
            noise = rng.normal(0.0, 0.12, size=7)
            seeds.append(np.clip(q_start + noise, ik.lo, ik.hi))

    best_q = None
    best_score = 1e9
    for seed in seeds:
        q = plan_q_to_tip_mid(
            ik, seed, tip_mid_target, max_iters=800, tol=0.01, accept_err=0.04, yaw=yaw, pitch=pitch
        )
        if q is None:
            continue
        ik.set_q(q)
        err = float(np.linalg.norm(ik.tip_mid() - tip_mid_target))
        tilt = float(np.linalg.norm(ik.ori_err(yaw, pitch)))
        score = err + 0.05 * tilt
        if score < best_score:
            best_score = score
            best_q = q
        if err <= 0.012 and tilt < 0.25:
            return q
    return best_q if best_score < 0.05 else None


def arm_can_reach_cube(ik: PositionOnlyIK, cube: np.ndarray, grasp_yaw: float = 0.0) -> bool:
    """True if this arm's IK can hover over the cube AND reach the grasp pose.

    Hover alone is not enough: close to the torso an arm can hover fine but
    stall centimetres short of the pitched grasp pose (measured on the right
    arm at x=0.285 — servo timeout, trial lost). Checking the grasp pose here
    hands those cubes to the other arm up front.
    """
    hover = clamp_tip_target(np.array([cube[0], cube[1], cube[2] + HOVER_CLEARANCE]))
    q = plan_q_to_tip_mid_robust(ik, np.deg2rad(ik.arm.idle_deg), hover, yaw=0.0, pitch=0.0)
    if q is None:
        return False
    ik.set_q(q)
    if float(np.linalg.norm(ik.tip_mid() - hover)) > REACH_TIP_TOL:
        return False
    grasp = clamp_tip_target(np.array([cube[0], cube[1], cube[2] + GRASP_CLEARANCE]))
    q_g = plan_q_to_tip_mid_robust(ik, q, grasp, yaw=grasp_yaw, pitch=GRASP_PITCH_RAD)
    if q_g is None:
        return False
    ik.set_q(q_g)
    return float(np.linalg.norm(ik.tip_mid() - grasp)) <= 0.02


def reachable_arms(
    iks: dict[str, PositionOnlyIK], cube: np.ndarray, grasp_yaw: float = 0.0
) -> list[ArmSpec]:
    return [arm for arm in ARMS if arm_can_reach_cube(iks[arm.side], cube, grasp_yaw)]


def choose_arm(
    reachable: list[ArmSpec],
    rng: np.random.Generator,
    iks: dict[str, PositionOnlyIK] | None = None,
    cube: np.ndarray | None = None,
    grasp_yaw: float = 0.0,
) -> ArmSpec | None:
    """Deterministic side rule: the cube's side of the robot centerline picks
    the arm — left arm for y >= 0, right arm for y < 0.

    Earlier versions scored plan quality and coin-flipped ties, so central
    cubes were sometimes picked cross-body. That gives the policy an
    ambiguous visual mapping to learn. One consistent rule makes the
    demonstration data unambiguous: same cube position, same arm, always.
    If the side's arm cannot reach, return None so the caller resamples the
    cube pose instead of teaching a cross-body exception.
    """
    if not reachable:
        return None
    side = "left" if (cube is None or float(cube[1]) >= 0.0) else "right"
    for arm in reachable:
        if arm.side == side:
            return arm
    return None


def teleport_to_tip(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    tip_mid: np.ndarray,
    grip_m: float,
    yaw: float = 0.0,
) -> bool:
    tip_mid = clamp_tip_target(tip_mid)
    q = plan_q_to_tip_mid(
        ik, np.deg2rad(ik.arm.idle_deg), tip_mid, max_iters=1200, tol=0.015, yaw=yaw
    )
    if q is None:
        print(f"  teleport {ik.arm.side}: IK failed for tip-mid={tip_mid}")
        return False
    teleport_arms(robot, ik, q, grip_m)
    err = float(np.linalg.norm(tip_mid_world(robot, ik.arm) - tip_mid))
    tilt = float(np.linalg.norm(ik.level_err(yaw)))
    print(
        f"  teleport {ik.arm.side}: tip-mid err={err * 100:.1f} cm, tilt={math.degrees(tilt):.1f}°"
    )
    return err < 0.05


def make_robot(
    model_path: str, fps: int, viewer: bool, cameras: str = "none"
) -> MujocoBiOpenArm:
    from lerobot.cameras.mujoco import MujocoCameraConfig

    # Camera sets for recording: "chest" = ego only, "all" = ego + both wrists.
    # Names match the real chocolate-dataset schema (ego/left_wrist/right_wrist).
    cam_cfg: dict = {}
    if cameras in ("chest", "all"):
        cam_cfg["ego"] = MujocoCameraConfig(mujoco_name="ego_camera", fps=fps, width=640, height=480)
    if cameras == "all":
        cam_cfg["left_wrist"] = MujocoCameraConfig(
            mujoco_name="left_wrist_camera", fps=fps, width=640, height=480
        )
        cam_cfg["right_wrist"] = MujocoCameraConfig(
            mujoco_name="right_wrist_camera", fps=fps, width=640, height=480
        )
    robot = MujocoBiOpenArm(
        MujocoBiOpenArmConfig(
            viewer=viewer,
            cameras=cam_cfg,
            model_path=model_path,
            fps=fps,
            start_elbow_bend_deg=90.0,
        )
    )
    robot.connect(calibrate=False)
    return robot


class EpisodeRecorder:
    """Record every commanded tick into a LeRobotDataset episode.

    Frames are captured in send_q — the single funnel every phase's commands
    pass through — so the recorded action stream is exactly what drove the
    sim, including the idle arm's slow tuck. Failed trials are dropped.
    """

    def __init__(self, robot: MujocoBiOpenArm, repo_id: str, fps: int, task: str):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME
        from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

        self.robot = robot
        self.task = task
        self.features = combine_feature_dicts(
            hw_to_dataset_features(robot.observation_features, "observation", True),
            hw_to_dataset_features(robot.action_features, "action", True),
        )
        if (HF_LEROBOT_HOME / repo_id).exists():
            # Append to the existing dataset (generation happens in chunks).
            self.dataset = LeRobotDataset.resume(
                repo_id,
                root=HF_LEROBOT_HOME / repo_id,
                image_writer_threads=4 * max(1, len(robot.cameras)),
            )
            missing = set(self.features) - set(self.dataset.features)
            if missing:
                raise ValueError(
                    f"Existing dataset '{repo_id}' lacks features {sorted(missing)} — "
                    "it was probably recorded with a different --cameras setting. "
                    "Use a new repo id (or delete the old dataset)."
                )
            print(
                f"  appending to existing dataset ({self.dataset.num_episodes} episodes so far). "
                "NOTE: use a different --seed than previous runs or the same cube "
                "sequence will be repeated."
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id,
                fps,
                robot_type=robot.name,
                features=self.features,
                use_videos=True,
                image_writer_threads=4 * max(1, len(robot.cameras)),
            )
        self.active = False

    def tick(self, action: dict) -> None:
        if not self.active:
            return
        from lerobot.utils.feature_utils import build_dataset_frame

        obs = self.robot.get_observation()
        frame = build_dataset_frame(self.features, obs, prefix="observation")
        frame.update(build_dataset_frame(self.features, action, prefix="action"))
        self.dataset.add_frame({**frame, "task": self.task})

    def start(self) -> None:
        self.active = True

    def drop(self) -> None:
        self.active = False
        self.dataset.clear_episode_buffer()

    def save(self) -> bool:
        self.active = False
        try:
            self.dataset.save_episode()
            return True
        except Exception as e:  # noqa: BLE001
            # One bad episode (e.g. a staging PNG lost to a transient disk
            # error) must not kill a multi-hour generation run — drop it and
            # keep generating.
            print(f"  WARNING: episode save failed ({e}); dropping episode and continuing")
            try:
                self.dataset.clear_episode_buffer()
            except Exception:  # noqa: BLE001
                pass
            return False

    def finalize(self) -> None:
        self.dataset.finalize()


_RECORDER: EpisodeRecorder | None = None


def _random_start_offsets(rng: np.random.Generator) -> np.ndarray:
    """Whole-arm start randomization. Wrist gets the most (visible gripper
    orientation variety); proximal joints get some too, so starts vary in arm
    CONFIGURATION and not just tip placement — every IK plan from the same
    idle seed lands in the same configuration family otherwise."""
    return np.array([
        float(rng.uniform(-0.17, 0.17)),   # J1 ±10°
        float(rng.uniform(-0.14, 0.14)),   # J2 ±8°
        float(rng.uniform(-0.17, 0.17)),   # J3 ±10°
        float(rng.uniform(-0.17, 0.17)),   # J4 ±10°
        float(rng.uniform(-0.35, 0.35)),   # J5 wrist pitch ±20°
        float(rng.uniform(-0.25, 0.25)),   # J6 wrist yaw   ±14°
        float(rng.uniform(-0.50, 0.50)),   # J7 wrist roll  ±29°
    ])


def _random_start_q(
    ik: PositionOnlyIK, arm: ArmSpec, rng: np.random.Generator
) -> np.ndarray:
    """Random start configuration for one arm — a mixture, for VLA variety:
    75% a random pose in the workspace box, 25% a perturbed tucked pose (arms
    in real deployments often start at rest by the robot's side)."""
    def _tips_safe(q: np.ndarray) -> bool:
        # Estimated tips sit ~3.2 cm above the true finger bottom, so demand
        # est tip z > table + 8 cm: at 2x speed the PD lag is bigger and a low
        # random start sweeps the fingers into the table on the first move.
        ik.set_q(q)
        return float(ik.tip_mid()[2]) > TABLE_TOP_Z + 0.08

    for _ in range(12):
        if rng.uniform() < 0.25:
            q = np.clip(np.deg2rad(arm.park_deg) + _random_start_offsets(rng), ik.lo, ik.hi)
        else:
            tip = random_start_tip(arm, rng)
            q0 = plan_q_to_tip_mid(
                ik, np.deg2rad(arm.idle_deg), clamp_tip_target(tip), max_iters=800, tol=0.02, yaw=0.0
            )
            if q0 is None:
                q0 = np.deg2rad(arm.idle_deg)
            q = np.clip(q0 + _random_start_offsets(rng), ik.lo, ik.hi)
        if _tips_safe(q):
            return q
    return np.clip(np.deg2rad(arm.idle_deg), ik.lo, ik.hi)


def setup_start_pose(
    robot: MujocoBiOpenArm, ik: PositionOnlyIK, rng: np.random.Generator, fps: int
) -> np.ndarray:
    """Teleport BOTH arms to random start poses with slightly random gripper
    orientation. The unused arm will retreat to its park pose on its own
    (see _park_action); the active arm levels its gripper before approaching."""
    import mujoco

    q_active = _random_start_q(ik, ik.arm, rng)
    ik.set_q(q_active)
    tip = ik.tip_mid().copy()
    print(f"  {ik.arm.side} start tip-mid=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})")

    other = ARMS_BY_SIDE[ik.arm.other]
    other_ik = PositionOnlyIK(ik.model, ik.data, other)
    q_other = _random_start_q(other_ik, other, rng)

    # Grippers start in a random state too (anywhere from closed to fully
    # open). The active arm ramps open as its first act of the episode; the
    # idle arm's gripper ramps closed during its tuck.
    g_active = float(rng.uniform(0.0, FINGER_OPEN_M))
    g_other = float(rng.uniform(0.0, FINGER_OPEN_M))

    _set_arm_qpos(robot, ik.arm.side, q_active)
    _set_gripper_qpos(robot, ik.arm.side, g_active)
    _set_arm_qpos(robot, other.side, q_other)
    _set_gripper_qpos(robot, other.side, g_other)
    zero_sim_velocity(robot)
    mujoco.mj_forward(robot._model, robot._data)
    ik.set_q(q_active)
    _LAST_CMD[ik.arm.side] = q_active.copy()
    _LAST_CMD[other.side] = q_other.copy()
    _OTHER_GRIP[other.side] = g_other
    cube_now = cube_pos(robot)
    d_tuck = float(np.linalg.norm(cube_now[:2] - _TUCK_TIP_XY[other.side]))
    if d_tuck < _TUCK_CLEARANCE_M:
        _RETREAT_TARGET[other.side] = np.deg2rad(other.park_deg)
        print(
            f"  cube is {d_tuck * 100:.0f} cm from {other.side}'s half-tuck spot "
            f"— {other.side} will retreat to full park instead"
        )
    else:
        _RETREAT_TARGET[other.side] = np.deg2rad(other.tuck_deg)
    print(
        f"  {other.side} starts random too (grip {g_other * 1000:.0f} mm); "
        f"will half-tuck over the table edge during the pick"
    )
    settle_pose(robot, ik, g_active, fps, hold_s=0.2)
    return tip


def go_to_tips(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    tip_mid_target: np.ndarray,
    grip_m: float,
    fps: int,
    timeout_s: float,
    tol: float,
    label: str,
    max_step_m: float = 0.012,
    ik_iters: int = 2,
) -> bool:
    deadline = time.perf_counter() + timeout_s
    last = 1e9
    ticks = 0
    while time.perf_counter() < deadline:
        tip_mid_target = clamp_tip_target(tip_mid_target)
        tip_now = tip_mid_world(robot, ik.arm)
        delta = tip_mid_target - tip_now
        dist = float(np.linalg.norm(delta))
        tip_cmd = tip_now + delta * (min(1.0, max_step_m / dist) if dist > 1e-9 else 1.0)
        obs = robot.get_observation()
        ik.set_q(_arm_q_from_obs(obs, ik.arm.side))
        for _ in range(ik_iters):
            ik.step_tip_mid(tip_cmd, max_dq=math.radians(2.5), yaw=0.0, level=True)
        t0 = time.perf_counter()
        send_q(robot, ik, grip_m)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        last = float(np.linalg.norm(tip_mid_world(robot, ik.arm) - tip_mid_target))
        ticks += 1
        if last <= tol:
            print(f"  {label}: ok (tip-mid err={last * 100:.1f} cm, {ticks} ticks)")
            return True
    print(f"  {label}: timeout (tip-mid err={last * 100:.1f} cm)")
    return last < tol * 1.6


def _finger_opening_m(robot: MujocoBiOpenArm, side: str) -> float:
    """Live finger slide opening in meters (0 = closed)."""
    import mujoco

    jid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint1")
    return float(robot._data.qpos[robot._model.jnt_qposadr[jid]])


def _hold_fingers(robot: MujocoBiOpenArm, ik: PositionOnlyIK, hold_m: float) -> None:
    """Command gripper opening via the robot PD / position actuators only."""
    hold_m = float(np.clip(hold_m, 0.0, FINGER_OPEN_M))
    send_q(robot, ik, hold_m)


def set_gripper(
    robot: MujocoBiOpenArm, ik: PositionOnlyIK, grip_m: float, fps: int, hold_s: float
) -> float:
    """Drive gripper to ``grip_m`` and **do not return until it gets there**.

    Returns the commanded opening, or -1.0 if it never closed/opened in time.
    """
    start = float(np.clip(_finger_opening_m(robot, ik.arm.side), 0.0, FINGER_OPEN_M))
    ik.set_q(_cmd_seed(robot, ik.arm.side))
    q_hold = ik.q().copy()
    target = float(np.clip(grip_m, 0.0, FINGER_OPEN_M))
    closing = target < start - 1e-4
    max_wait_s = max(hold_s, 3.0 if closing else 0.5)
    deadline = time.perf_counter() + max_wait_s
    cmd = start
    step = GRIP_RAMP_MPS / fps
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        ik.set_q(q_hold)
        # Ramp the commanded opening toward the target instead of jumping.
        cmd = max(target, cmd - step) if closing else min(target, cmd + step)
        _hold_fingers(robot, ik, cmd)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        opened = _finger_opening_m(robot, ik.arm.side)
        if closing:
            # Cube blocks ~25–30 mm — require a real close, not a tiny twitch.
            done = opened <= GRASP_CLOSED_M and (start - opened) >= 0.010
        else:
            done = opened >= target - 0.004
        if done:
            for _ in range(max(8, fps // 3)):
                ik.set_q(q_hold)
                _hold_fingers(robot, ik, target)
                precise_sleep(1.0 / fps)
            opened = _finger_opening_m(robot, ik.arm.side)
            print(
                f"  gripper[{ik.arm.side}]: CLOSED opening={opened * 1000:.1f} mm "
                f"(cmd {target * 1000:.1f} mm)"
            )
            return target
    opened = _finger_opening_m(robot, ik.arm.side)
    print(
        f"  gripper[{ik.arm.side}]: NOT closed opening={opened * 1000:.1f} mm "
        f"(cmd {target * 1000:.1f} mm) — abort"
    )
    return -1.0


def _rate_limit_q(q: np.ndarray, q_prev: np.ndarray, max_step: float) -> np.ndarray:
    return np.clip(q, q_prev - max_step, q_prev + max_step)


def _command_q(
    robot: MujocoBiOpenArm, ik: PositionOnlyIK, q: np.ndarray, grip_m: float, fps: int
) -> None:
    ik.set_q(q)
    t0 = time.perf_counter()
    send_q(robot, ik, grip_m)
    precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def play_joint_path(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    q_start: np.ndarray,
    q_end: np.ndarray,
    grip_m: float,
    fps: int,
    duration_s: float,
    label: str,
    abort_on_table: bool = True,
) -> bool:
    n = max(2, int(duration_s * fps))
    print(f"  {label}: joint move ({duration_s:.1f}s)…")
    q_prev = q_start.copy()
    for k in range(n):
        u = (k + 1) / n
        s = 1.0 - (1.0 - u) ** 2
        q = _rate_limit_q((1.0 - s) * q_start + s * q_end, q_prev, math.radians(0.8))
        _command_q(robot, ik, q, grip_m, fps)
        q_prev = q.copy()
        if abort_on_table and grasp_table_fault(robot, ik.arm) is not None:
            print(f"  {label}: TABLE HIT {grasp_table_fault(robot, ik.arm)} — aborting")
            return False
    return True


def play_tip_cartesian(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    tip_end: np.ndarray,
    grip_m: float,
    fps: int,
    yaw: float,
    label: str,
    speed_mps: float = APPROACH_SPEED_MPS,
    lock_z: float | None = None,
    freeze_wrist: bool = False,
    min_z: float | None = None,
) -> np.ndarray | None:
    tip_end = clamp_tip_target(tip_end)
    tip_start = tip_mid_world(robot, ik.arm)
    z_floor = MIN_TIP_Z if min_z is None else float(min_z)
    if lock_z is not None:
        tip_start = tip_start.copy()
        tip_start[2] = float(lock_z)
        tip_end = tip_end.copy()
        tip_end[2] = float(lock_z)
    tip_end = tip_end.copy()
    tip_end[2] = max(float(tip_end[2]), z_floor)
    dist = float(np.linalg.norm(tip_end - tip_start))
    duration_s = max(0.5, dist / max(speed_mps, 1e-3))
    n = max(2, int(duration_s * fps))
    print(f"  {label}: cartesian tip move ({duration_s:.1f}s, {dist * 100:.0f} cm)…")
    q0 = _cmd_seed(robot, ik.arm.side)
    wrist = q0[4:7].copy() if freeze_wrist else None
    ik.set_q(q0)
    q_prev = q0.copy()
    for k in range(n):
        u = (k + 1) / n
        s = u * u * (3.0 - 2.0 * u)  # ease-in-out: starts and ends at rest
        tip_t = (1.0 - s) * tip_start + s * tip_end
        if lock_z is not None:
            tip_t[2] = float(lock_z)
        tip_t[2] = max(float(tip_t[2]), z_floor)
        tip_t = clamp_tip_target(tip_t)
        for _ in range(2):
            ik.step_tip_mid(
                tip_t,
                max_dq=math.radians(1.6),
                yaw=yaw,
                level=not freeze_wrist,
                freeze_wrist=wrist,
            )
        q = _rate_limit_q(ik.q(), q_prev, math.radians(1.4))
        _command_q(robot, ik, q, grip_m, fps)
        q_prev = q.copy()
        if grasp_table_fault(robot, ik.arm) is not None:
            print(f"  {label}: TABLE HIT {grasp_table_fault(robot, ik.arm)} — aborting")
            return None
        if float(np.linalg.norm(tip_mid_world(robot, ik.arm) - tip_end)) < 0.012:
            break
    return _arm_q_from_obs(robot.get_observation(), ik.arm.side)


def center_tip_over_cube(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    cube: np.ndarray,
    hold_z: float,
    fps: int,
    yaw: float,
    tol_xy: float = 0.01,
    timeout_s: float = 3.0,
) -> bool:
    deadline = time.perf_counter() + timeout_s
    hold_z = max(float(hold_z), MIN_TIP_Z)
    tips_dbg = finger_tips_world(robot, ik.arm)
    low_dbg = finger_lowest_z(robot, ik.arm)
    tilt_dbg = vertical_tilt_rad(ik)
    print(
        f"  centering tip over cube at z={hold_z:.3f}… "
        f"[entry: est tips z=({tips_dbg[0][2]:.3f},{tips_dbg[1][2]:.3f}) "
        f"true lowest z=({low_dbg[0]:.3f},{low_dbg[1]:.3f}) "
        f"tilt={math.degrees(tilt_dbg):.1f}°]"
    )
    ik.set_q(_cmd_seed(robot, ik.arm.side))
    q_prev = ik.q().copy()
    last_err = 1e9
    z_bump = 0.0
    tip_cmd = None
    while time.perf_counter() < deadline:
        cube = cube_pos(robot)
        tip = tip_mid_world(robot, ik.arm)
        err_xy = float(np.linalg.norm(tip[:2] - cube[:2]))
        last_err = err_xy
        if err_xy <= tol_xy:
            print(f"  centered: xy_err={err_xy * 100:.1f} cm")
            return True
        if finger_table_graze(robot, ik.arm) and z_bump < 0.02:
            z_bump += 0.005
            print(f"  center: finger grazed table — raising hold z by {z_bump * 1000:.0f} mm")
        # Tilt-compensated altitude: a tilted hand swings a fingertip below the
        # tip estimate by ~finger_len*sin(tilt), so translate at a raised height
        # while tilted instead of blocking translation (a hard gate deadlocks
        # when joint limits make leveling impossible at the current pose).
        tilt_now = vertical_tilt_rad(ik)
        z_tilt = 0.10 * math.sin(min(tilt_now, math.radians(45.0)))
        tip_t = np.array([cube[0], cube[1], hold_z + z_bump + z_tilt])
        tip_cmd = tip_t if tip_cmd is None else 0.75 * tip_cmd + 0.25 * tip_t
        tip_t = tip_cmd
        ik.set_q(_arm_q_real(robot, ik))  # model = reality; command stays chained
        for _ in range(2):
            ik.step_tip_mid(tip_t, max_dq=math.radians(1.6), yaw=yaw, level=True)
        q = _rate_limit_q(ik.q(), q_prev, math.radians(1.4))
        _command_q(robot, ik, q, FINGER_OPEN_M, fps)
        q_prev = q.copy()
        if grasp_table_fault(robot, ik.arm) is not None:
            print(f"  center: TABLE HIT {grasp_table_fault(robot, ik.arm)}")
            return False
    print(f"  center: timeout xy_err={last_err * 100:.1f} cm")
    return last_err < tol_xy * 1.8


def pitch_tips_onto_cube(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    fps: int,
    yaw: float,
    pitch_end: float = GRASP_PITCH_RAD,
    duration_s: float = PITCH_DURATION_S,
) -> bool:
    """Pitch the wrist down and lower tip-mid onto the cube center together.

    One rate-limited joint-space blend from the level hover pose to the pitched
    grasp pose — wrist angle and tip height move at the same time (no snap).
    """
    tip0 = tip_mid_world(robot, ik.arm)
    cube = cube_pos(robot)
    hold_z = max(float(tip0[2]), float(cube[2] + HOVER_CLEARANCE * 0.5), MIN_TIP_Z)

    q_hover = _cmd_seed(robot, ik.arm.side)
    ik.set_q(q_hover)

    grasp_tip = grasp_tip_target(robot, ik.arm, cube)
    print(
        f"  pitch+lower: plan {math.degrees(pitch_end):.0f}° grasp "
        f"(tip past cube by {GRASP_INSET_M * 100:.1f} cm, from hover z={hold_z:.3f})…"
    )
    q_grasp = plan_q_to_tip_mid_robust(ik, q_hover, grasp_tip, yaw=yaw, pitch=pitch_end)
    if q_grasp is None:
        print("  pitch+lower: IK could not plan grasp pose — aborting")
        return False
    ik.set_q(q_grasp)
    plan_xy = float(np.linalg.norm(ik.tip_mid()[:2] - grasp_tip[:2]))
    plan_z = float(ik.tip_mid()[2])
    if plan_xy > 0.03:
        print(f"  pitch+lower: grasp plan off-center ({plan_xy * 100:.1f} cm) — aborting")
        return False

    dq_plan = np.abs(q_grasp - q_hover)
    n = max(
        int(max(duration_s, 2.8) * fps),
        int(math.ceil(float(np.max(dq_plan)) / math.radians(1.1))),
    )
    print(
        f"  pitch+lower: joint blend over {n / fps:.1f}s "
        f"(Δprox≤{np.degrees(dq_plan[:4]).max():.1f}°, Δwrist≤{np.degrees(dq_plan[4:]).max():.1f}°, "
        f"tip z {hold_z:.3f}→{plan_z:.3f})…"
    )
    q_prev = q_hover.copy()
    for k in range(n):
        u = (k + 1) / n
        s = u * u * (3.0 - 2.0 * u)
        q = _rate_limit_q((1.0 - s) * q_hover + s * q_grasp, q_prev, math.radians(1.2))
        _command_q(robot, ik, q, FINGER_OPEN_M, fps)
        q_prev = q.copy()
        if grasp_table_fault(robot, ik.arm) is not None:
            print(f"  pitch+lower: TABLE HIT {grasp_table_fault(robot, ik.arm)} — aborting")
            return False
        if finger_table_graze(robot, ik.arm):
            # Stop the open-loop descent at first light touch; the closed-loop
            # servo below owns the last centimetre and can climb back off.
            print(f"  pitch+lower: finger grazed table at blend {100 * u:.0f}% — handing to servo")
            break
        tip = tip_mid_world(robot, ik.arm)
        tgt = grasp_tip_target(robot, ik.arm, cube_pos(robot))
        if float(np.linalg.norm(tip - tgt)) <= GRASP_TIP_TOL and float(tip[2]) >= tgt[2] - 0.01:
            break

    tip = tip_mid_world(robot, ik.arm)
    cube = cube_pos(robot)
    tgt = grasp_tip_target(robot, ik.arm, cube)
    err = float(np.linalg.norm(tip - tgt))
    xy_err = float(np.linalg.norm(tip[:2] - tgt[:2]))
    print(
        f"  pitch+lower blend: tip=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}), "
        f"target=({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}), "
        f"err={err * 100:.1f} cm (xy={xy_err * 100:.1f} cm), "
        f"pitch≈{math.degrees(pitch_end):.0f}°"
    )
    log_grasp_contacts(robot, ik.arm, "pitch+lower")
    if grasp_table_fault(robot, ik.arm) is not None:
        print(f"  pitch+lower: TABLE HIT {grasp_table_fault(robot, ik.arm)} — fail")
        return False
    # Open-loop joint blend drifts under PD — servo the *live* tip-mid onto the
    # grasp target (high enough that finger meshes clear the table).
    return servo_tip_to_cube_center(
        robot, ik, fps, yaw=yaw, pitch=pitch_end, label="servo tips to cube center"
    )


def servo_tip_to_cube_center(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    fps: int,
    yaw: float,
    pitch: float,
    tol: float = GRASP_TIP_TOL * 1.15,
    timeout_s: float = 4.0,
    label: str = "servo tip-mid",
) -> bool:
    """Closed-loop: drive measured fingertip midpoint onto the cube center.

    IK joint plans put tip-mid on center kinematically, but live PD tracking
    leaves the real tips short/low — this loop measures tip-mid each tick and
    corrects until both XY and Z are on the cube (or timeout).
    """
    deadline = time.perf_counter() + timeout_s
    ik.set_q(_cmd_seed(robot, ik.arm.side))
    q_prev = ik.q().copy()
    last_err = 1e9
    # Pads close along this axis: an error here decides whether the cube is
    # pinched or batted sideways by one pad, so it gets a much tighter budget
    # than the along-jaw component (where the pads are ~2 cm longer than needed).
    jaw_axis = np.array([math.sin(yaw), -math.cos(yaw)])
    across_tol = 0.007
    z_bump = 0.0
    tip_cmd = None
    # Steady-state droop compensation, targets-only: with stock gains the arm
    # sags ~2 cm below a held command (gravity vs soft wrist PD), so integrate
    # the residual into the command the way a human teleoperator keeps pulling
    # up until the arm actually arrives.
    droop_bias = np.zeros(3)
    print(f"  {label}: closed-loop on live tip-mid (inset {GRASP_INSET_M * 100:.1f} cm)…")
    tick = 0
    if True:
        while time.perf_counter() < deadline:
            tick += 1
            cube = cube_pos(robot)
            target = grasp_tip_target(robot, ik.arm, cube)
            tip = tip_mid_world(robot, ik.arm)
            err = float(np.linalg.norm(tip - target))
            last_err = err
            fault = grasp_table_fault(robot, ik.arm)
            if fault is not None:
                print(f"  {label}: TABLE HIT {fault} — fail")
                return False
            z_ok = abs(float(tip[2]) - float(target[2])) <= 0.010
            d_xy = tip[:2] - target[:2]
            xy_err = float(np.linalg.norm(d_xy))
            across = abs(float(d_xy @ jaw_axis))
            if xy_err <= tol and across <= across_tol and z_ok:
                print(
                    f"  {label}: OK tip-mid→target xy={xy_err * 100:.1f} cm "
                    f"(across-jaw {across * 1000:.0f} mm) "
                    f"z={(tip[2] - cube[2]) * 100:+.1f} cm "
                    f"(target_z={target[2]:.3f})"
                )
                return True
            if finger_table_graze(robot, ik.arm) and z_bump < 0.02:
                z_bump += 0.004
            tip_t = tip.copy()
            tip_t[:2] = target[:2]
            # Aim AT the grasp depth. The old +6 mm "table protection" bias made
            # every grasp shallow — pads wrapping only the cube's top centimetre
            # slip out under the slightest twist during lift.
            tip_t[2] = max(float(target[2]) + z_bump, MIN_TIP_Z)
            # Integrate the tracking residual into the command (droop_bias),
            # then low-pass so per-tick re-targeting cannot twitch the arm.
            # Gravity only pulls DOWN, so the z bias is one-sided: it may lift
            # the command, never push it below the grasp target (a negative z
            # bias walked the fingers into the table).
            droop_bias = droop_bias + 0.2 * (tip_t - tip)
            droop_bias[:2] = np.clip(droop_bias[:2], -0.02, 0.02)
            droop_bias[2] = float(np.clip(droop_bias[2], 0.0, 0.035))
            tip_t = clamp_tip_target(tip_t + droop_bias)
            tip_cmd = tip_t if tip_cmd is None else 0.75 * tip_cmd + 0.25 * tip_t
            tip_t = tip_cmd
            # Model anchored to REALITY every tick (the IK linearizes about the
            # true pose; a model chained off commands diverges and the servo
            # chases a phantom). The droop integrator above supplies the lead
            # the old command-chained model provided implicitly. The command
            # stream q_prev stays continuous — never snapped back to obs.
            ik.set_q(_arm_q_real(robot, ik))
            for _ in range(4):
                ik.step_tip_mid(
                    tip_t,
                    max_dq=math.radians(1.6),
                    yaw=yaw,
                    pitch=pitch,
                    level=True,
                    proximal_scale=0.4,
                )
            q = _rate_limit_q(ik.q(), q_prev, math.radians(1.4))
            _command_q(robot, ik, q, FINGER_OPEN_M, fps)
            q_prev = q.copy()
            fault = grasp_table_fault(robot, ik.arm)
            if fault is not None:
                print(f"  {label}: TABLE HIT {fault} — fail")
                return False
        tip = tip_mid_world(robot, ik.arm)
        cube = cube_pos(robot)
        target = grasp_tip_target(robot, ik.arm, cube)
        last_err = float(np.linalg.norm(tip - target))
        z_ok = abs(float(tip[2]) - float(target[2])) <= 0.012
        across = abs(float((tip[:2] - target[:2]) @ jaw_axis))
        print(
            f"  {label}: timeout err={last_err * 100:.1f} cm "
            f"(tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}), "
            f"target_z={target[2]:.3f}, z_ok={z_ok}, across-jaw {across * 1000:.0f} mm)"
        )
        return (
            last_err <= tol * 1.5
            and across <= across_tol
            and z_ok
            and grasp_table_fault(robot, ik.arm) is None
        )


def approach_above_cube(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    cube: np.ndarray,
    fps: int,
    yaw: float,
) -> np.ndarray | None:
    # First: orient the gripper parallel to the table, in place, up high. The
    # start pose deliberately has a bit of random wrist orientation; levelling
    # is its own visible, deliberate motion before the reach begins.
    q_now = _cmd_seed(robot, ik.arm.side)
    ik.set_q(q_now)
    if vertical_tilt_rad(ik) > math.radians(6.0):
        tip_here = clamp_tip_target(tip_mid_world(robot, ik.arm))
        # Level AT a safe altitude: leveling in place at a low start sweeps the
        # fingertips through the table (they trail the tip estimate by ~3 cm).
        tip_here[2] = max(float(tip_here[2]), float(cube[2] + TRANSIT_CLEARANCE))
        q_level = plan_q_to_tip_mid_robust(ik, q_now, tip_here, yaw=0.0, pitch=0.0)
        if q_level is not None:
            dq = float(np.max(np.abs(q_level - q_now)))
            dur = max(0.5, dq / math.radians(50.0))
            play_joint_path(robot, ik, q_now, q_level, FINGER_OPEN_M, fps, dur, "level gripper")

    tip0 = tip_mid_world(robot, ik.arm)
    transit_z = float(cube[2] + TRANSIT_CLEARANCE)
    slide_z = max(float(tip0[2]), transit_z + 0.01)
    hover_z = max(float(cube[2] + HOVER_CLEARANCE), MIN_TIP_Z)

    if float(tip0[2]) < transit_z - 0.01:
        if (
            play_tip_cartesian(
                robot,
                ik,
                np.array([tip0[0], tip0[1], slide_z]),
                FINGER_OPEN_M,
                fps,
                yaw,
                label="lift to slide height",
                freeze_wrist=True,
            )
            is None
        ):
            return None

    cube = cube_pos(robot)
    over = np.array([cube[0], cube[1], slide_z])
    tip_now = tip_mid_world(robot, ik.arm)
    if float(np.linalg.norm(tip_now[:2] - over[:2])) > 0.01:
        if (
            play_tip_cartesian(
                robot,
                ik,
                over,
                FINGER_OPEN_M,
                fps,
                yaw,
                label="slide over cube center",
                lock_z=slide_z,
                freeze_wrist=True,
                min_z=slide_z - 0.005,
            )
            is None
        ):
            return None

    # Planned (not servoed) move to a LEVEL hover pose over the cube, verified
    # and retried once. The frozen-wrist transits above accumulate up to ~30° of
    # hand tilt, and the centering servo cannot always level from there (joint
    # limits) — the robust planner either finds a level hover pose or tells us
    # now. A single blend can also under-deliver (PD lag), so measure and retry.
    for attempt in range(2):
        q_now = _cmd_seed(robot, ik.arm.side)
        ik.set_q(q_now)
        cube = cube_pos(robot)
        hover_tip = clamp_tip_target(np.array([cube[0], cube[1], hover_z]))
        # The robust planner ranks by position error, so its best can be a
        # tilted pose even when a level one exists — plan from the current pose
        # and from idle, and keep whichever is more level.
        candidates = []
        for seed_q in (q_now, np.deg2rad(ik.arm.idle_deg)):
            q_c = plan_q_to_tip_mid_robust(ik, seed_q, hover_tip, yaw=yaw, pitch=0.0)
            if q_c is None:
                continue
            ik.set_q(q_c)
            candidates.append((vertical_tilt_rad(ik), q_c))
        if not candidates:
            print("  hover: could not plan a level hover pose")
            return None
        plan_tilt, q_hover = min(candidates, key=lambda c: c[0])
        if plan_tilt > math.radians(8.0):
            print(f"  hover: best plan still {math.degrees(plan_tilt):.0f}° tilted — rejecting")
            return None
        dq = float(np.max(np.abs(q_hover - q_now)))
        dur = max(0.6, dq / math.radians(40.0))
        if not play_joint_path(robot, ik, q_now, q_hover, FINGER_OPEN_M, fps, dur, "level hover blend"):
            return None
        if center_tip_over_cube(robot, ik, cube_pos(robot), hover_z, fps, yaw, tol_xy=0.01):
            return _arm_q_from_obs(robot.get_observation(), ik.arm.side)
        if attempt == 0:
            print("  hover: centering failed — replanning once from current pose")
    return None


def place_reachable_cube(
    robot: MujocoBiOpenArm,
    iks: dict[str, PositionOnlyIK],
    rng: np.random.Generator,
    max_tries: int = 40,
) -> tuple[np.ndarray, ArmSpec]:
    """Drop a cube somewhere at least one arm can reach; return cube + chosen arm."""
    for _ in range(max_tries):
        x = float(rng.uniform(*CUBE_X_RANGE))
        y = float(rng.uniform(*CUBE_Y_RANGE))
        yaw = float(rng.uniform(-0.6, 0.6))
        cube0 = set_cube_xy(robot, x, y, yaw=yaw)
        gy = cube_yaw(robot)
        reachable = reachable_arms(iks, cube0, grasp_yaw=gy)
        arm = choose_arm(reachable, rng, iks=iks, cube=cube0, grasp_yaw=gy)
        if arm is None:
            continue
        print(
            f"  cube @ xy=({cube0[0]:.3f}, {cube0[1]:.3f}) yaw={math.degrees(yaw):.0f}° "
            f"reachable={[a.side for a in reachable]} → pick with {arm.side}"
        )
        return cube0, arm
    raise RuntimeError("could not sample a cube reachable by either arm")


def run_trial(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    fps: int,
    cube0: np.ndarray,
) -> bool:
    """Simple pick: over cube → pitch tips around cube → close → lift."""
    yaw = cube_yaw(robot)
    print(f"  grasp yaw={math.degrees(yaw):+.0f}° (align pads to cube faces)")
    pitch = GRASP_PITCH_RAD

    # Yaw-scaled grasp relief: at a yawed grasp the pad's leading corner swings
    # lower (half-gap * sin|yaw|), and past ~15° the mid-cube depth would put
    # that corner inside the table — deterministically unreachable. Face-on
    # grasps keep the full mid-cube depth.
    _GRASP_Z_RELIEF["m"] = 0.035 * max(0.0, abs(math.sin(yaw)) - 0.15)
    if _GRASP_Z_RELIEF["m"] > 0.0:
        print(f"  yawed grasp: relieving depth by {_GRASP_Z_RELIEF['m'] * 1000:.0f} mm")

    # 1+2) Approach and place tips, with ONE full re-approach retry — a servo
    # table strike usually means this particular descent geometry was bad, and
    # a fresh approach from above fixes it more often than not.
    placed = False
    base_relief = _GRASP_Z_RELIEF["m"]
    for attempt in range(2):
        if attempt == 1:
            _GRASP_Z_RELIEF["m"] = base_relief + 0.008  # retry shallower still
        set_gripper(robot, ik, FINGER_OPEN_M, fps, hold_s=0.2)
        if approach_above_cube(robot, ik, cube_pos(robot), fps, yaw) is None:
            print("  fail: could not move over cube")
            return False
        if grasp_table_fault(robot, ik.arm) is not None:
            print(f"  fail: TABLE HIT {grasp_table_fault(robot, ik.arm)}")
            return False
        if pitch_tips_onto_cube(robot, ik, fps, yaw, pitch_end=pitch):
            placed = True
            break
        if attempt == 0:
            print("  placement failed — lifting clear and re-approaching once")
            tip_up = tip_mid_world(robot, ik.arm) + np.array([0.0, 0.0, 0.07])
            play_tip_cartesian(
                robot, ik, tip_up, FINGER_OPEN_M, fps, yaw,
                label="retreat up", freeze_wrist=True,
            )
    if not placed:
        print("  fail: could not place tips around cube")
        return False
    tip = tip_mid_world(robot, ik.arm)
    cube = cube_pos(robot)
    target = grasp_tip_target(robot, ik.arm, cube)
    print(
        f"  tips around cube: tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) "
        f"target_z={target[2]:.3f} xy_err={np.linalg.norm(tip[:2] - cube[:2]) * 100:.1f} cm"
    )
    log_grasp_contacts(robot, ik.arm, "pre-close")
    if grasp_table_fault(robot, ik.arm) is not None:
        print(f"  fail: TABLE HIT {grasp_table_fault(robot, ik.arm)}")
        return False

    # 3) Close fully — lift is forbidden until this returns success.
    print(f"  close {ik.arm.side} (will not lift until closed)…")
    if True:
        for close_try in range(2):
            ik.set_q(_cmd_seed(robot, ik.arm.side))
            hold_grip = set_gripper(robot, ik, GRASP_HOLD_M, fps, hold_s=3.0)
            if hold_grip < 0.0:
                print("  fail: gripper never finished closing — not lifting")
                return False
            opened = _finger_opening_m(robot, ik.arm.side)
            if opened > GRASP_CLOSED_M:
                print(f"  fail: still open ({opened * 1000:.1f} mm) — not lifting")
                return False
            s_close = log_grasp_contacts(robot, ik.arm, "after-close")
            if s_close["pinching"] and float(s_close["cube_force_n"]) >= 6.0:
                break
            if close_try == 0:
                # One-sided or feeble contact lifts nothing — reopen, re-center
                # on the (possibly nudged) cube, and close once more.
                print("  close: weak/one-sided pinch — reopening to re-center")
                set_gripper(robot, ik, FINGER_OPEN_M, fps, hold_s=0.3)
                if not servo_tip_to_cube_center(
                    robot, ik, fps, yaw=yaw, pitch=pitch, label="re-center tips"
                ):
                    print("  fail: could not re-center for second close")
                    return False
        else:
            print("  fail: no solid pinch after retry — not lifting")
            return False

        # 4) Lift only after close confirmed.
        print(f"  closed OK ({opened * 1000:.1f} mm) — now lifting")
        q_now = _arm_q_from_obs(robot.get_observation(), ik.arm.side)
        cube = cube_pos(robot)
        # Straight-up cartesian lift that HOLDS the grasp orientation. A
        # joint-space blend to a lift IK solution rotates the hand several
        # degrees during the first ticks (traced: 10° -> 1° of orientation
        # residual), and pitched pads twisting against a cube still on the
        # table pry it out of the jaws — pinch 17 N at 2% of the lift, cube
        # gone by 18%. Stepping the tip vertically with yaw/pitch enforced
        # keeps the pads glued to the cube instead.
        q_now = _cmd_seed(robot, ik.arm.side)
        ik.set_q(q_now)
        tip0_l = tip_mid_world(robot, ik.arm)
        # Freeze the wrist at whatever orientation the grasp actually ended at.
        # The grasp can finish ~10° off nominal, and ANY orientation correction
        # while the cube still touches the table pries it out of the pads
        # (traced twice: pinch >8 N at 2% of lift, zero by 18%).
        wrist_hold = q_now[4:7].copy()
        n_lift = max(2, int(LIFT_DURATION_S * fps))
        print(f"  lift ({n_lift / fps:.1f}s, cartesian, wrist frozen)…")
        q_prev = q_now.copy()
        cube_held = cube_pos(robot)
        if True:
            for k in range(n_lift):
                u = (k + 1) / n_lift
                s_u = u * u * (3.0 - 2.0 * u)
                tip_t = tip0_l + np.array([0.0, 0.0, s_u * LIFT_CLEARANCE])
                # Wrist frozen while the cube can still touch the table (any
                # orientation correction there pries it out of the pads); once
                # clearly airborne, gently hold the grasp orientation so the
                # carry looks deliberate instead of drifting ~20°.
                airborne = float(cube_pos(robot)[2]) > cube_held[2] + 0.02
                for _ in range(3):
                    if airborne:
                        ik.step_tip_mid(tip_t, max_dq=math.radians(0.8), yaw=yaw, pitch=pitch, level=True)
                    else:
                        ik.step_tip_mid(tip_t, max_dq=math.radians(1.6), freeze_wrist=wrist_hold)
                q = _rate_limit_q(ik.q(), q_prev, math.radians(1.4))
                ik.set_q(q)
                _hold_fingers(robot, ik, hold_grip)
                precise_sleep(1.0 / fps)
                q_prev = q.copy()
                if k % max(1, n_lift // 6) == 0:
                    cnow = cube_pos(robot)
                    s_c = grasp_contact_summary(robot, ik.arm)
                    tilt = float(np.linalg.norm(ik.ori_err(yaw, pitch)))
                    print(
                        f"    lift {100 * u:3.0f}%: cube_z={cnow[2]:.3f} "
                        f"pinchF={s_c['cube_force_n']:.1f}N pads "
                        f"L={s_c['left_pad_on_cube']} R={s_c['right_pad_on_cube']} "
                        f"grip={_finger_opening_m(robot, ik.arm.side) * 1000:.1f}mm "
                        f"tilt={math.degrees(tilt):.0f}°"
                    )
                    if cnow[2] < cube_held[2] - 0.005 and not s_c["pinching"]:
                        print("    lift: cube LOST — aborting lift early")
                        break

        # Hold the cube at the top for a beat before the episode ends.
        n_hold = max(1, int(LIFT_HOLD_S * fps))
        print(f"  holding at the top for {LIFT_HOLD_S:.1f}s (cube_z={cube_pos(robot)[2]:.3f})…")
        for _ in range(n_hold):
            ik.set_q(q_prev)
            _hold_fingers(robot, ik, hold_grip)
            precise_sleep(1.0 / fps)

        cube_f = cube_pos(robot)
        ok = float(cube_f[2]) >= SUCCESS_CUBE_Z
        log_grasp_contacts(robot, ik.arm, "after-lift")
        print(f"  result: {'SUCCESS' if ok else 'FAIL'} ({ik.arm.side}) cube_z={cube_f[2]:.3f}")
        return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-path",
        default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"),
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--record",
        default=None,
        metavar="REPO_ID",
        help="Record successful picks into a LeRobotDataset (e.g. local/openarm_sim_cube).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="With --record: keep going until this many SUCCESSFUL episodes are saved.",
    )
    parser.add_argument(
        "--cameras",
        choices=["chest", "all"],
        default="all",
        help="Cameras to record: 'chest' (ego only) or 'all' (ego + both wrists).",
    )
    parser.add_argument(
        "--task",
        default="pick up the red cube and lift it",
        help="Task string stored with every recorded frame (VLA language conditioning).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose per-trial output (cube pose, IK plans, phase logs). "
        "Default is quiet: one 'episode N' line per saved episode.",
    )
    args = parser.parse_args()

    global _RECORDER
    rng = np.random.default_rng(args.seed)
    robot = make_robot(
        args.model_path,
        args.fps,
        viewer=not args.no_viewer,
        cameras=args.cameras if args.record else "none",
    )
    if args.record:
        _RECORDER = EpisodeRecorder(robot, args.record, args.fps, args.task)
        print(
            f"Recording to '{args.record}' (cameras={args.cameras}, task='{args.task}');"
            f" target {args.episodes or args.trials} successful episodes"
        )
    iks = {arm.side: build_ik(robot, arm) for arm in ARMS}

    # Park both arms at their low side poses immediately.
    park_both_arms(robot, iks)
    settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)

    target_eps = args.episodes if (args.record and args.episodes > 0) else 0
    max_trials = args.trials if not target_eps else max(args.trials, target_eps * 3)
    print("Running randomized bilateral pick trial(s)…")
    successes = 0
    used = {"left": 0, "right": 0}

    import contextlib
    import io

    def _trial_output():
        # Quiet by default: swallow the per-trial chatter so the console is
        # one clean 'episode N' line per saved episode. --debug restores it.
        if args.debug:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    try:
        t = 0
        while t < max_trials:
            t += 1
            with _trial_output():
                if target_eps:
                    print(f"\n=== Trial {t} — episodes saved {successes}/{target_eps} (trial cap {max_trials}) ===")
                else:
                    print(f"\n=== Trial {t}/{max_trials} ===")
                cube0, arm = place_reachable_cube(robot, iks, rng)
                ik = iks[arm.side]
                used[arm.side] += 1
                print(f"  start: {arm.other} side-parked, teleport {arm.side} to random pose")
                setup_start_pose(robot, ik, rng, args.fps)

                if _RECORDER is not None:
                    _RECORDER.start()
                ok = run_trial(robot, ik, args.fps, cube0)
                if ok:
                    if _RECORDER is not None:
                        if _RECORDER.save():
                            successes += 1
                            print(f"  episode {successes} saved")
                        else:
                            ok = False
                    else:
                        successes += 1
                    if ok:
                        print(f"  pick succeeded ({arm.side})")
                else:
                    if _RECORDER is not None:
                        _RECORDER.drop()
                        print("  failed trial — episode dropped")
                    print(f"  pick failed ({arm.side})")
                # Park both before the next drop (never recorded).
                park_both_arms(robot, iks)
                settle_pose(robot, ik, 0.0, args.fps, hold_s=0.15)
            if ok and not args.debug:
                print(f"episode {successes}", flush=True)
            if target_eps and successes >= target_eps:
                break
            if not target_eps and t >= args.trials:
                break
    finally:
        if _RECORDER is not None:
            _RECORDER.finalize()
            print(f"dataset finalized: {successes} episodes")
        print(
            f"\nDone: {successes}/{t} successful picks "
            f"(used left={used['left']}, right={used['right']})"
        )
        robot.disconnect()


if __name__ == "__main__":
    main()
