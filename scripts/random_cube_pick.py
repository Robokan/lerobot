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
CUBE_Y_DEADBAND = 0.04  # no cubes within this of the centreline (see place_reachable_cube)

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

    def hand(self) -> np.ndarray:
        """Palm position — the origin of the aim line (the TCP sits at the pads)."""
        return self.data.geom_xpos[self.hand_gid].copy()

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
# Retreat rate of the idle arm. 0.8 deg/tick (24 deg/s) was slow enough that
# the working arm could catch up with it mid-tuck and collide.
_RETREAT_STEP_RAD = math.radians(1.5)

# Where the idle arm parks: a wrist target BEHIND the table's near edge
# (the top spans x 0.055..0.555), so the tucked arm is off the table
# altogether. The IK solves the arm configuration; hand-picked joint angles
# kept the hand out over the table no matter how far the shoulder was pulled.
TUCK_TIP_TARGET = {"right": np.array([0.08, -0.28, 0.53]), "left": np.array([0.08, 0.28, 0.53])}
_TUCK_Q_CACHE: dict[str, np.ndarray] = {}


def tuck_q(ik: PositionOnlyIK) -> np.ndarray:
    """Joint angles that put this arm's wrist at its off-table park target."""
    side = ik.arm.side
    if side not in _TUCK_Q_CACHE:
        q = plan_q_to_tip_mid_robust(
            ik, np.deg2rad(ik.arm.tuck_deg), TUCK_TIP_TARGET[side], yaw=0.0, pitch=0.0
        )
        _TUCK_Q_CACHE[side] = np.deg2rad(ik.arm.tuck_deg) if q is None else q
    return _TUCK_Q_CACHE[side].copy()
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
        park = _TUCK_Q_CACHE.get(side, np.deg2rad(ARMS_BY_SIDE[side].tuck_deg))
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
    return finger_lowest_z_model(robot._model, robot._data, arm)


def finger_lowest_z_model(model, data, arm: ArmSpec) -> list[float]:
    import mujoco

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


# Centring tolerances for the straddle test (cube defaults). A wider-open
# gripper around a narrow bar can be judged more loosely — close+lift decides.
_STRADDLE_TOL = {"xy": 0.035, "imbalance": 0.035}


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
    tol = _STRADDLE_TOL["xy"]
    ok = opposite and wide_enough and mid_err_xy < tol and imbalance < _STRADDLE_TOL["imbalance"]
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


# Debug overlay: when set to a cube position, every commanded tick draws the
# wrist->cube line (green) and the gripper's approach ray (red) of the same
# length into the viewer. When the arm is aimed, the red ray lies on the green
# line. None disables and clears the overlay.
_AIM_OVERLAY: dict[str, np.ndarray | None] = {"cube": None, "goal_offset": None}
_AIM_OVERLAY_ENABLED = False  # set by --debug; the overlay is a diagnostic


def _draw_aim_overlay(robot: MujocoBiOpenArm, ik: PositionOnlyIK) -> None:
    viewer = getattr(robot, "_viewer", None)
    if viewer is None:
        return
    import mujoco

    scn = viewer.user_scn
    if not _AIM_OVERLAY_ENABLED or _AIM_OVERLAY["cube"] is None:
        scn.ngeom = 0
        return
    wrist = robot._data.geom_xpos[ik.hand_gid].copy()
    approach = robot._data.xmat[ik.body].reshape(3, 3)[:, 2]
    # Draw to where the target IS, read live every tick. _AIM_OVERLAY["cube"]
    # is only an on/off flag: it used to hold a snapshot taken when the trial
    # started, so the line kept pointing at the cube's old position while it
    # was being pushed. cube_pos() follows set_target_body(), so this is the
    # bar in the caddy scene and the cube everywhere else.
    off = _AIM_OVERLAY["goal_offset"]
    cube = aim_point(cube_pos(robot)) + (off if off is not None else 0.0)
    length = float(np.linalg.norm(cube - wrist))
    rays = [
        (cube, np.array([0.1, 0.9, 0.2, 0.9])),                       # wrist -> cube
        (wrist + approach * length, np.array([0.95, 0.2, 0.15, 0.9])),  # gripper approach
    ]
    scn.ngeom = 0
    for end, rgba in rays:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, wrist, np.asarray(end, dtype=float))
        scn.ngeom += 1
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.012), np.asarray(cube, dtype=float),
        np.eye(3).flatten(), np.array([0.1, 0.9, 0.2, 0.6]),
    )
    scn.ngeom += 1


# DART (Laskey et al. 2017): perturb what the robot EXECUTES, record what the
# planner INTENDED. Three runs on this task — GR00T and two pi0.5 fine-tunes —
# fit their demonstrations to ~1° open loop and still scored 0/50 closed loop,
# knocking a neighbouring bar in 17 of 20 trials. The demonstrations came from
# a planner that never errs, so they contain no recoveries: the first time the
# policy's own small error puts it a centimetre off the nominal path, it is in
# a state no demonstration visited and keeps executing the nominal path from
# the wrong place. A human teleoperator drifts and corrects constantly, and 100
# teleop episodes on the real robot did produce occasional successes where 300
# perfect ones here produced none. This puts the drift back in: the arm wanders
# off the path, the camera sees it off the path, and every label says how to
# get back on. The noise is an OU process (rho ~0.9 at 30 fps, ~1/3 s memory)
# rather than i.i.d. per tick, because a stiff position controller filters
# single-tick jitter into nothing — the perturbation has to persist long enough
# to actually displace the arm. Gripper and parked arm stay clean. Off unless
# a generator sets sigma_deg; the eval never sees it.
# Watched on screen, the first version "wobbled all over the place". Two causes,
# both fixed here. rho 0.9 at 30 fps is a third-of-a-second memory: that reads
# as tremor. A person leaning off the path and correcting does it over seconds,
# so rho is 0.98 (about 1.7 s). And independent noise on all seven joints
# flops the wrist around; a person's error is a coherent position offset, so
# only the four proximal joints (shoulder x3, elbow) drift and the wrist stays
# where the planner put it.
_DART: dict = {"sigma_deg": 0.0, "rho": 0.98, "scale": 1.0, "state": {},
               "joints": (0, 1, 2, 3), "rng": np.random.default_rng(0)}


def _dart_noise_deg(side: str, n: int) -> np.ndarray | None:
    d = _DART
    if d["sigma_deg"] <= 0.0 or _RECORDER is None or not _RECORDER.active:
        return None
    prev = d["state"].get(side)
    if prev is None or prev.shape[0] != n:
        prev = np.zeros(n)
    rho = d["rho"]
    sig = d["sigma_deg"] * d["scale"]
    cur = rho * prev + sig * math.sqrt(1.0 - rho * rho) * d["rng"].standard_normal(n)
    mask = np.zeros(n)
    for j in d["joints"]:
        if j < n:
            mask[j] = 1.0
    cur = cur * mask
    d["state"][side] = cur
    return cur


def dart_episode_reset(rng: np.random.Generator | None = None) -> None:
    """New episode: forget the noise history and draw this episode's amplitude.
    The per-episode scale is what makes the dataset cover both small and large
    deviations instead of one fixed band. 0.5x-1.5x: a 0.5x-3x range was tried
    and looked like the arm wobbling all over the place, so the top end is gone.
    Episodes the noise pushes into failure are simply not saved."""
    _DART["state"] = {}
    g = rng if rng is not None else _DART["rng"]
    _DART["scale"] = float(g.uniform(0.5, 1.5)) if _DART["sigma_deg"] > 0 else 1.0


def send_q(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    grip_m: float,
) -> None:
    """Command active arm from IK; keep the other arm parked."""
    _draw_aim_overlay(robot, ik)
    q = ik.q()
    _LAST_CMD[ik.arm.side] = np.asarray(q, dtype=float).copy()
    action = _park_action(ik.arm.other)
    for i, qi in enumerate(q, start=1):
        action[f"{ik.arm.side}_joint_{i}.pos"] = float(math.degrees(qi))
    action[f"{ik.arm.side}_gripper.pos"] = gripper_m_to_deg(grip_m)
    noise = _dart_noise_deg(ik.arm.side, len(q))
    if noise is None:
        robot.send_action(action)
    else:
        executed = dict(action)           # the robot gets the drift ...
        for i in range(len(q)):
            executed[f"{ik.arm.side}_joint_{i + 1}.pos"] += float(noise[i])
        robot.send_action(executed)
    if _RECORDER is not None:
        _RECORDER.tick(action)            # ... the dataset gets the intention


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
    model_path: str, fps: int, viewer: bool, cameras: str = "none", arm_gain_scale: float = 1.0
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
    cfg = MujocoBiOpenArmConfig(
        viewer=viewer,
        cameras=cam_cfg,
        model_path=model_path,
        fps=fps,
        start_elbow_bend_deg=90.0,
    )
    if arm_gain_scale != 1.0:
        # Stiffer position servos: the default gains let the arm sag ~1.5 cm
        # under gravity at mid reach (measured), which is more than the pad
        # clearance a 2.5 cm bar allows. kd scales as sqrt(kp) to keep the
        # damping ratio. Measured x3: 0.5 cm sag, no oscillation.
        cfg.arm_kp = [k * arm_gain_scale for k in cfg.arm_kp]
        cfg.arm_kd = [k * math.sqrt(arm_gain_scale) for k in cfg.arm_kd]
    robot = MujocoBiOpenArm(cfg)
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
        dart_episode_reset()

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


TUCKED_START_PROB = 0.75
# Variation of a "tucked" start: every joint jittered, the elbow/shoulder more
# than the wrist (a resting arm settles differently each time), so the policy
# never sees one canonical rest pose.
TUCK_JITTER_DEG = np.array([5.0, 5.0, 4.0, 6.0, 3.0, 3.0, 3.0])


def jittered_tuck(arm: ArmSpec, rng: np.random.Generator, ik: PositionOnlyIK | None = None) -> np.ndarray:
    """A varied tucked pose. With ``ik`` given, draws whose fingertips would
    end up within 3 cm of the table are resampled (the elbow jitter can drop
    the resting hand onto the table edge)."""
    for _ in range(12):
        base = tuck_q(ik) if ik is not None else np.deg2rad(arm.tuck_deg)
        q = base + np.deg2rad(rng.normal(0.0, 1.0, size=7) * TUCK_JITTER_DEG)
        if ik is None:
            return q
        q = np.clip(q, ik.lo, ik.hi)
        ik.set_q(q)
        # Check the REAL finger geometry, not the tip point: the pads hang
        # ~3.6 cm below it, so a tip 3 cm up put them through the table and the
        # episode faulted on its first move.
        if min(finger_lowest_z_model(ik.model, ik.data, arm)) > TABLE_TOP_Z + AIM_PAD_CLEARANCE_M:
            return q
    return tuck_q(ik) if ik is not None else np.deg2rad(arm.tuck_deg)


def setup_start_pose(
    robot: MujocoBiOpenArm, ik: PositionOnlyIK, rng: np.random.Generator, fps: int
) -> np.ndarray:
    """Teleport BOTH arms to random start poses with slightly random gripper
    orientation. The unused arm will retreat to its park pose on its own
    (see _park_action); the active arm levels its gripper before approaching."""
    import mujoco

    # Three quarters of the time the grabbing arm starts from its tucked pose
    # (varied) — the reach then begins from rest, like a fresh pick.
    active_tucked = rng.uniform() < TUCKED_START_PROB
    if active_tucked:
        q_active = np.clip(jittered_tuck(ik.arm, rng, ik), ik.lo, ik.hi)
    else:
        q_active = _random_start_q(ik, ik.arm, rng)
    ik.set_q(q_active)
    tip = ik.tip_mid().copy()
    print(
        f"  {ik.arm.side} start {'TUCKED (jittered)' if active_tucked else 'random'} "
        f"tip-mid=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})"
    )

    other = ARMS_BY_SIDE[ik.arm.other]
    other_ik = PositionOnlyIK(ik.model, ik.data, other)

    # Three quarters of the time the idle arm starts already tucked (a human
    # who just finished with that hand leaves it resting), varied so the pose
    # is never identical. Skipped when the cube sits near that arm's tuck
    # spot — the arm would be parked on top of the workspace.
    cube_now = cube_pos(robot)
    d_tuck = float(np.linalg.norm(cube_now[:2] - _TUCK_TIP_XY[other.side]))
    tucked_start = d_tuck >= _TUCK_CLEARANCE_M and rng.uniform() < TUCKED_START_PROB
    if tucked_start:
        q_other = np.clip(jittered_tuck(other, rng, other_ik), other_ik.lo, other_ik.hi)
    else:
        q_other = _random_start_q(other_ik, other, rng)

    # Grippers start in a random state too (anywhere from closed to fully
    # open). The active arm ramps open as its first act of the episode; the
    # idle arm's gripper ramps closed during its tuck (a tucked start is
    # already nearly closed).
    g_active = float(rng.uniform(0.0, 0.006)) if active_tucked else float(rng.uniform(0.0, FINGER_OPEN_M))
    g_other = float(rng.uniform(0.0, 0.006)) if tucked_start else float(rng.uniform(0.0, FINGER_OPEN_M))

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
    # Always the same deep tuck the arm can start in. There used to be a
    # fallback to the old half-park whenever the cube sat near the tuck spot,
    # which fired on most layouts and left the idle arm sitting over the table;
    # the tuck is now off the table entirely, so it can never conflict.
    _RETREAT_TARGET[other.side] = np.clip(
        jittered_tuck(other, rng, other_ik), other_ik.lo, other_ik.hi
    )
    if tucked_start:
        print(f"  {other.side} starts TUCKED (jittered, grip {g_other * 1000:.0f} mm)")
    else:
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
    stop_when=None,
) -> bool:
    """Stream a joint interpolation. ``stop_when`` (no-arg callable) is checked
    after every tick; returning True ends the move early (still a success)."""
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
        if stop_when is not None and stop_when():
            return True
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
        if abs(y) < CUBE_Y_DEADBAND:
            # A cube on the centreline is an either-arm case: the side rule is
            # a hard step there, and two cubes a centimetre apart would demand
            # opposite arms while looking identical to the camera. Every
            # trained checkpoint's arm mismatches were exactly these cubes.
            continue
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


AIM_TIP_ABOVE_CUBE_M = 0.035

# The aim machinery grasps whatever body set_target_body() points at. These two
# describe that target: how far above its centre the tips stop (the pads then
# straddle its upper part) and the surface beneath it that the pads must clear
# (the table, or the bar below in a stack). Cube defaults; set_aim_target()
# switches them for other objects.
_AIM_TARGET = {
    "tip_above": AIM_TIP_ABOVE_CUBE_M,
    "support_z": TABLE_TOP_Z,
    "pad_clearance": None,
    "xy": None,          # where the support surface is (None = everywhere, i.e. the table)
    "azimuth": None,     # preferred approach azimuth (rad); None = from the shoulder line
    "obstacles": [],     # (xyz, radius) keep-outs for fingertips and hand
    "in_extra": None,    # reach this far past the aim point (None = half a cube: pads centre on it)
    "max_pitch_deg": None,  # cap on the approach pitch (None = all candidates up to 70 deg)
    "squeeze_m": None,      # squeeze past the block point (None = AIM_SQUEEZE_PAST_BLOCK_M, sized for the cube)
}
_AIM_LAST_HOLD = {"m": 0.0}  # grip commanded by the last grasp_and_lift (for carrying on)
AIM_SUPPORT_FOOTPRINT_M = 0.08  # the raised support floor applies within this xy radius


def set_aim_target(
    tip_above_m: float,
    support_z: float,
    pad_clearance_m: float | None = None,
    xy: np.ndarray | None = None,
    azimuth: float | None = None,
    obstacles: list[tuple[np.ndarray, float]] | None = None,
    in_extra_m: float | None = None,
    max_pitch_deg: float | None = None,
    squeeze_m: float | None = None,
) -> None:
    """Describe the grasp target for the aim machinery. ``support_z`` is the
    surface the pads must clear over the target (the table, or the object
    below it in a stack); with ``xy`` given it only applies within
    AIM_SUPPORT_FOOTPRINT_M of that point and the table floor applies
    elsewhere. ``azimuth`` pins the approach direction (an elongated object
    must be approached along its length); ``obstacles`` are other objects the
    fingertips and hand must stay out of on the way."""
    _AIM_TARGET["tip_above"] = float(tip_above_m)
    _AIM_TARGET["support_z"] = float(support_z)
    _AIM_TARGET["pad_clearance"] = None if pad_clearance_m is None else float(pad_clearance_m)
    _AIM_TARGET["xy"] = None if xy is None else np.asarray(xy, dtype=float)[:2].copy()
    _AIM_TARGET["azimuth"] = azimuth
    _AIM_TARGET["obstacles"] = list(obstacles or [])
    _AIM_TARGET["in_extra"] = None if in_extra_m is None else float(in_extra_m)
    _AIM_TARGET["max_pitch_deg"] = max_pitch_deg
    _AIM_TARGET["squeeze_m"] = None if squeeze_m is None else float(squeeze_m)


def aim_pad_clearance() -> float:
    c = _AIM_TARGET["pad_clearance"]
    return AIM_PAD_CLEARANCE_M if c is None else c


def arm_probe_points(ik: PositionOnlyIK) -> list[tuple[np.ndarray, float]]:
    """(point, extra keep-out) along the arm beyond the fingertips: the hand and
    the wrist/forearm joints. The forearm sweeping through the pile or a
    neighbouring stack knocked bars flying while the fingertips were clear."""
    joints = [ik.hand()] + [ik.data.xpos[ik.model.jnt_bodyid[ik.jids[j]]].copy() for j in (6, 5, 4, 3)]
    pts = [(joints[0], 0.03)]
    # the link BETWEEN joints is what rests across a pile: sample it too
    for a, b in zip(joints[:-1], joints[1:], strict=True):
        pts.append((0.5 * (a + b), 0.05))
        pts.append((b, 0.05))
    return pts


def obstacles_clear(ik: PositionOnlyIK, tips: np.ndarray) -> bool:
    for pos, radius in _AIM_TARGET["obstacles"]:
        if float(np.linalg.norm(tips - pos, axis=1).min()) < radius:
            return False
        for pt, extra in arm_probe_points(ik):
            if float(np.linalg.norm(pt - pos)) < radius + extra:
                return False
    return True


def aim_pads_clear(ik: PositionOnlyIK) -> bool:
    """Pads above the local floor and fingertips/hand/forearm outside every
    obstacle, for the pose currently set on ``ik``."""
    tips = finger_tips_from_data(ik.model, ik.data, ik.arm)
    low = min(finger_lowest_z_model(ik.model, ik.data, ik.arm))
    xy = _AIM_TARGET["xy"]
    if xy is None:
        floor = TABLE_TOP_Z + aim_pad_clearance()
    elif float(np.linalg.norm(tips.mean(axis=0)[:2] - xy)) < AIM_SUPPORT_FOOTPRINT_M:
        # over the target: the (possibly tight) clearance above its support
        floor = _AIM_TARGET["support_z"] + aim_pad_clearance()
    else:
        # anywhere else the tight target clearance does not apply — keep the
        # normal safe margin above the table (a 2 mm floor let the turn dip)
        floor = TABLE_TOP_Z + max(aim_pad_clearance(), AIM_PAD_CLEARANCE_M)
    if low < floor:
        return False
    return obstacles_clear(ik, tips)


def aim_point(cube: np.ndarray) -> np.ndarray:
    """The point on the target the gripper aims at and the tips travel to: the
    grasp point between the pads when straddling. Aiming at the geometric
    centre while sending the tips above it is a built-in conflict for the IK."""
    goal = np.asarray(cube, dtype=float).copy()
    goal[2] += _AIM_TARGET["tip_above"]
    return goal


def aim_error_deg(ik: PositionOnlyIK, cube: np.ndarray) -> float:
    """Angle between the gripper approach axis (TCP z) and the hand -> cube line."""
    d = aim_point(cube) - ik.hand()
    d /= max(float(np.linalg.norm(d)), 1e-9)
    approach = ik.rot()[:, 2]
    return math.degrees(math.acos(float(np.clip(np.dot(approach, d), -1.0, 1.0))))


def aim_yaw_pitch_from_dir(aim_dir: np.ndarray) -> tuple[float, float]:
    """grasp_rot() parameters whose approach axis points along aim_dir."""
    d = np.asarray(aim_dir, dtype=float)
    return math.atan2(d[1], d[0]), math.atan2(-d[2], math.hypot(d[0], d[1]))


def pinch_tilt_deg(ik: PositionOnlyIK) -> float:
    """How far the finger-opening axis (TCP y) is from table-parallel."""
    return math.degrees(math.asin(float(np.clip(abs(ik.rot()[2, 1]), 0.0, 1.0))))


def shoulder_pos(ik: PositionOnlyIK) -> np.ndarray:
    return ik.data.xpos[ik.model.jnt_bodyid[ik.jids[0]]].copy()


# Pre-aim standoff: wrist this far from the cube, on the shoulder->cube line,
# looking down at this angle. A human aims from behind the object, not from
# wherever the hand happened to be.
AIM_STANDOFF_M = 0.22
AIM_PITCH_RAD = math.radians(30.0)  # measured: 30 deg plans in every scene, 22-26 deg in none


# Stereotyped strategy (default). A data generator must map similar-looking
# scenes to similar-looking motions: if it picks between discrete strategies
# (candidate standoffs, detour-or-not, yield-or-not), two scenes a centimetre
# apart can get visibly different motions, and from the policy's side that is
# indistinguishable from randomness. Everything below is a CONTINUOUS function
# of the geometry; when the single strategy does not plan, the caller resamples
# the scene instead of trying a different one.
AIM_STEREOTYPED = True
AIM_MIN_SHOULDER_CLEAR_M = 0.14  # standoff never closer than this to the shoulder
# A TRANSIT path only has to miss the table. AIM_PAD_CLEARANCE_M is for the
# final approach, where the arm sags under load.
AIM_TRANSIT_CLEAR_M = 0.012


def aim_standoff_candidates(ik: PositionOnlyIK, cube: np.ndarray) -> list[np.ndarray]:
    """Standoff point(s) at AIM_STANDOFF_M from the cube, preferred first.

    Stereotyped: exactly one, behind the cube on the shoulder->cube line (or on
    the target's own axis when one is set), looking down at a pitch that varies
    CONTINUOUSLY with how much room there is behind the cube — shallow when the
    cube is far from the shoulder, steepening smoothly as it gets closer.
    """
    sh = shoulder_pos(ik)
    h = np.asarray(cube[:2], dtype=float) - sh[:2]
    hd = float(np.linalg.norm(h))
    base_az = math.atan2(h[1], h[0])
    if _AIM_TARGET["azimuth"] is not None:
        base_az = float(_AIM_TARGET["azimuth"])  # elongated target: along its length

    # Shallowest pitch whose standoff still clears the shoulder, floored at the
    # nominal 22 deg and capped by any per-target limit.
    cos_max = (hd - AIM_MIN_SHOULDER_CLEAR_M) / AIM_STANDOFF_M
    pitch = max(AIM_PITCH_RAD, math.acos(float(np.clip(cos_max, -1.0, 1.0))))
    max_pitch = _AIM_TARGET["max_pitch_deg"]
    if max_pitch is not None:
        pitch = min(pitch, math.radians(float(max_pitch)))

    def at(pitch_rad: float, az: float) -> np.ndarray:
        horiz = AIM_STANDOFF_M * math.cos(pitch_rad)
        p = np.asarray(cube, dtype=float) - np.array([math.cos(az), math.sin(az), 0.0]) * horiz
        p[2] = float(cube[2]) + AIM_STANDOFF_M * math.sin(pitch_rad)
        return p

    if AIM_STEREOTYPED:
        return [at(pitch, base_az)]
    out = []
    for pd in (22.0, 30.0, 40.0, 55.0, 70.0):
        if max_pitch is not None and pd > max_pitch:
            break
        for daz in (0.0, 30.0, -30.0, 60.0, -60.0):
            out.append(at(math.radians(pd), base_az + math.radians(daz)))
    return out


def plan_aim_axis(
    ik: PositionOnlyIK,
    q_start: np.ndarray,
    wrist_target: np.ndarray,
    aim_dir: np.ndarray,
    rng: np.random.Generator | None = None,
    iters: int = 400,
) -> tuple[np.ndarray | None, float, float]:
    """DLS IK: wrist (TCP) to wrist_target, approach axis along aim_dir, and the
    pinch (finger-opening axis) table-parallel — i.e. the full grasp_rot()
    orientation whose approach axis is the aim line. Six constraints on seven
    joints; the standoff-candidate search supplies the reachability slack.
    Returns (q or None, wrist_err_m, aim_err_deg); q is set on ``ik``.
    """
    import mujoco

    aim_dir = np.asarray(aim_dir, dtype=float)
    aim_dir = aim_dir / max(float(np.linalg.norm(aim_dir)), 1e-9)
    yaw, pitch = aim_yaw_pitch_from_dir(aim_dir)
    seeds = [q_start.copy(), np.deg2rad(ik.arm.idle_deg)]
    if rng is not None:
        seeds += [np.clip(q_start + rng.normal(0.0, 0.15, 7), ik.lo, ik.hi) for _ in range(4)]
    best = None
    best_score = float("inf")
    for seed in seeds:
        ik.set_q(seed.copy())
        for _ in range(iters):
            perr = wrist_target - ik.ee()
            pn = float(np.linalg.norm(perr))
            if pn > 1e-9:
                perr = perr * min(1.0, 0.03 / pn)
            oerr = ik.ori_err(yaw, pitch)
            on = float(np.linalg.norm(oerr))
            if on > 1e-9:
                oerr = oerr * min(1.0, 0.25 / on)
            jacp = np.zeros((3, ik.model.nv))
            jacr = np.zeros((3, ik.model.nv))
            mujoco.mj_jacBody(ik.model, ik.data, jacp, jacr, ik.body)
            j = np.vstack([jacp[:, ik.dadr], 0.6 * jacr[:, ik.dadr]])
            dx = np.concatenate([perr, 0.6 * oerr])
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), dx)
            dq = np.clip(dq, -math.radians(4.0), math.radians(4.0))
            q = np.clip(ik.q() + dq, ik.lo, ik.hi)
            q[3] = max(q[3], math.radians(8.0))
            ik.set_q(q)
        pe = float(np.linalg.norm(wrist_target - ik.ee()))
        ae = math.degrees(math.acos(float(np.clip(np.dot(ik.rot()[:, 2], aim_dir), -1.0, 1.0))))
        score = pe + 0.01 * ae
        if score < best_score:
            best_score = score
            best = (ik.q().copy(), pe, ae)
        if pe < 0.015 and ae < 5.0:
            break
    q, pe, ae = best
    ik.set_q(q)
    return (q if (pe < 0.04 and ae < 10.0) else None), pe, ae


def plan_aim_at_cube(
    ik: PositionOnlyIK,
    q_start: np.ndarray,
    cube: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Wrist to the pre-aim standoff, approach axis along the TRUE wrist->cube line.

    Pass 1 heads for the standoff point; the wrist rarely lands exactly there,
    so two more passes hold the achieved wrist and re-aim along the line from
    where it actually is. Retried from the idle pose if the start seed diverges.
    """
    ik.set_q(q_start)
    hand_to_tcp = float(np.linalg.norm(ik.ee() - ik.hand()))
    goal = aim_point(cube)
    for standoff in aim_standoff_candidates(ik, cube):
        aim0 = goal - standoff
        tcp_target = standoff + aim0 / np.linalg.norm(aim0) * hand_to_tcp
        seeds = [q_start.copy()] if AIM_STEREOTYPED else [q_start.copy(), np.deg2rad(ik.arm.idle_deg)]
        for seed in seeds:
            plan_aim_axis(ik, seed, tcp_target, aim0, rng=None, iters=300)
            for _ in range(3):
                q = ik.q().copy()
                tcp = ik.ee().copy()
                plan_aim_axis(ik, q, tcp, goal - ik.hand(), rng=None, iters=150)
            hand = ik.hand()
            dist = float(np.linalg.norm(goal - hand))
            ok = (
                aim_error_deg(ik, cube) < 8.0
                and pinch_tilt_deg(ik) < 6.0
                and dist > 0.15
                and float(ik.tip_mid()[2]) > _AIM_TARGET["support_z"] + 0.06
                and float(np.linalg.norm(hand - standoff)) < 0.12
            )
            if not ok:
                continue
            q_standoff = ik.q().copy()
            # Go half a cube farther in so the pads centre on the cube rather
            # than stopping at its near face. Horizontal only: extending along
            # the tilted line would also lower the pads toward the table.
            dir_h = goal - ik.hand()
            dir_h[2] = 0.0
            dir_h /= max(float(np.linalg.norm(dir_h)), 1e-9)
            in_extra = _AIM_TARGET["in_extra"]
            goal_in = goal + dir_h * (CUBE_HALF if in_extra is None else in_extra)
            chain = plan_approach_chain(ik, q_standoff, goal_in)
            if chain is not None:
                _AIM_OVERLAY["goal_offset"] = (goal_in - aim_point(cube)).copy()
                _AIM_PLAN["cube"] = np.asarray(cube, dtype=float).copy()
                _AIM_PLAN["in_offset"] = (goal_in - goal).copy()
                return q_standoff, chain
    return None


# What the current approach chain was planned against, so a bumped cube can be
# detected and the chain re-planned toward where the cube actually is.
_AIM_PLAN: dict[str, np.ndarray | None] = {"cube": None, "in_offset": None}
AIM_REPLAN_BUMP_M = 0.008


# Physical arm sags ~1.5 cm below the commanded pose under PD tracking; keep
# the planned pad bottoms at least this far above the table.
AIM_PAD_CLEARANCE_M = 0.025


def plan_approach_chain(
    ik: PositionOnlyIK, q_standoff: np.ndarray, goal: np.ndarray
) -> list[np.ndarray] | None:
    """Aimed waypoints along the aim line from the standoff to the goal.

    Each waypoint has the tips on the line, the approach axis along the
    hand->goal line and the pinch level; consecutive waypoints are close
    enough that interpolating between them keeps the aim within a few
    degrees, so the arm re-aims continuously while closing in. The joint
    interpolation between waypoints is swept for pad/table clearance.
    """
    ik.set_q(q_standoff)
    tip0 = ik.tip_mid().copy()
    tcp_off = ik.ee() - ik.tip_mid()  # TCP sits a hair off tip-mid
    chain = []
    q_prev = q_standoff
    for f in (0.25, 0.5, 0.75, 1.0):
        tip_f = tip0 + f * (goal - tip0)
        q_f, pe, ae = plan_aim_axis(ik, q_prev, tip_f + tcp_off, goal - ik.hand(), rng=None, iters=300)
        if q_f is None or pe > 0.015 or ae > 8.0:
            return None
        ik.set_q(q_f)
        if pinch_tilt_deg(ik) > 6.0:
            return None
        if not aim_pads_clear(ik):
            return None
        for k in range(1, 12):
            ik.set_q((1.0 - k / 12) * q_prev + (k / 12) * q_f)
            if not aim_pads_clear(ik):
                return None
        chain.append(q_f)
        q_prev = q_f
    return chain


def joint_path_clear(ik: PositionOnlyIK, q_a: np.ndarray, q_b: np.ndarray, steps: int = 40) -> bool:
    """Kinematic sweep of the straight joint interpolation: fingertips must stay
    above the table the whole way (the executed path is this interpolation)."""
    for k in range(steps + 1):
        ik.set_q((1.0 - k / steps) * q_a + (k / steps) * q_b)
        tips = finger_tips_from_data(ik.model, ik.data, ik.arm)
        # Real pad geometry, not the tip points: the pads hang ~3.6 cm below
        # them, so the old 3.5 cm tip margin approved paths whose pads were
        # already through the table (seed 11, mid-turn).
        if min(finger_lowest_z_model(ik.model, ik.data, ik.arm)) < TABLE_TOP_Z + AIM_TRANSIT_CLEAR_M:
            return False
        if not obstacles_clear(ik, tips):
            return False
    return True


def plan_via_lift(ik: PositionOnlyIK, q_from: np.ndarray, q_to: np.ndarray) -> list[np.ndarray] | None:
    """Joint waypoints from q_from to q_to that don't sweep through the table:
    direct if clear, else via a straight lift of the wrist (orientation held)."""
    if joint_path_clear(ik, q_from, q_to):
        return [q_to]
    if AIM_STEREOTYPED:
        return None  # no detour variant: the caller resamples the scene
    ik.set_q(q_from)
    wrist, approach = ik.ee().copy(), ik.rot()[:, 2].copy()
    back = -approach.copy()
    back[2] = 0.0
    back /= max(float(np.linalg.norm(back)), 1e-9)
    # via-points: straight up, and up-and-back (retreating along the approach
    # line clears an obstacle the wrist is already over)
    for lift, retreat in ((0.10, 0.0), (0.16, 0.0), (0.22, 0.0), (0.12, 0.08), (0.18, 0.10)):
        target = wrist + np.array([0.0, 0.0, lift]) + back * retreat
        q_lift, _, _ = plan_aim_axis(ik, q_from, target, approach, iters=250)
        if q_lift is None:
            continue
        if joint_path_clear(ik, q_from, q_lift) and joint_path_clear(ik, q_lift, q_to):
            return [q_lift, q_to]
    return None


# Closing speed is a smooth function of aim error AND distance — no gate.
# Aim penalty: badly aimed -> slow. Its weight shrinks with distance: far out
# there is travel left to keep correcting, so a poor aim still approaches at a
# good fraction of full speed; close in, a poor aim brings it to a creep. A
# good aim is full speed at any range. The turn toward the aim runs at its own
# steady joint rate underneath, so the two blend into one motion.
AIM_SPEED_MAX_MPS = 0.08
AIM_SPEED_FLOOR = 0.08           # fraction of max at worst
AIM_SLOW_ANGLE_DEG = 30.0        # aim error at/above which the penalty saturates
AIM_NEAR_M, AIM_FAR_M = 0.08, 0.30
AIM_FAR_PENALTY_WEIGHT = 0.35    # at AIM_FAR_M a saturated aim penalty still leaves 65% speed
AIM_TURN_RATE_DEG_S = 45.0       # joint rate for the turn toward the aim


def approach_speed_mps(aim_err_deg: float, dist_m: float) -> float:
    aim_pen = min(1.0, max(0.0, aim_err_deg / AIM_SLOW_ANGLE_DEG))
    far = min(1.0, max(0.0, (dist_m - AIM_NEAR_M) / (AIM_FAR_M - AIM_NEAR_M)))
    weight = 1.0 - far * (1.0 - AIM_FAR_PENALTY_WEIGHT)
    return AIM_SPEED_MAX_MPS * max(AIM_SPEED_FLOOR, 1.0 - weight * aim_pen)


def _polyline_at(points: list[np.ndarray], cum: np.ndarray, x: float) -> np.ndarray:
    i = int(np.searchsorted(cum, x, side="right") - 1)
    i = max(0, min(i, len(points) - 2))
    seg = cum[i + 1] - cum[i]
    f = 0.0 if seg < 1e-9 else min(1.0, max(0.0, (x - cum[i]) / seg))
    return (1.0 - f) * points[i] + f * points[i + 1]


# Gripper command during the approach. Default: fully open the whole way in,
# then close_on_cube() does everything after the arm has arrived. A generator
# may install a function of approach progress (0..1) returning the opening in
# metres, so the fingers start closing over the last stretch and the squeeze
# lands as the arm arrives (the caddy picker does this: it is much quicker and
# it is how a person picks). The settle before the close is shortened likewise.
_APPROACH_GRIP_FN: dict = {"fn": None}
_APPROACH_SETTLE_S: dict = {"s": 0.5}


def _approach_grip(progress: float) -> float:
    fn = _APPROACH_GRIP_FN["fn"]
    return FINGER_OPEN_M if fn is None else float(np.clip(fn(float(np.clip(progress, 0.0, 1.0))), 0.0, FINGER_OPEN_M))


def execute_aim_and_approach(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    fps: int,
    turn_path: list[np.ndarray],
    chain: list[np.ndarray],
    rng: np.random.Generator,
    timeout_s: float = 20.0,
) -> bool:
    """One continuous motion: turn toward the aim while closing in along the
    aim line, the closing speed following the current aim error and distance.

    ``turn_path`` runs from the current pose to the aimed standoff pose (with
    any lift via-point); ``chain`` runs from the standoff to the goal through
    aimed waypoints. The command is the chain pose at the current approach
    progress plus the *remaining turn offset* (turn pose minus standoff pose),
    which decays to zero at the turn's own rate. Approach progress advances at
    approach_speed_mps(aim error of the commanded pose): while the turn is far
    from done the aim is poor and the tip creeps; as the arm comes onto the
    line the approach speeds up. Nothing switches.
    """
    arm = ik.arm
    q_aim = turn_path[-1]
    ref = [q_aim] + list(chain)
    tips = []
    for q in ref:
        ik.set_q(q)
        tips.append(ik.tip_mid().copy())
    arc = np.concatenate([[0.0], np.cumsum([np.linalg.norm(tips[i + 1] - tips[i]) for i in range(len(ref) - 1)])])
    total = float(arc[-1])
    tcum = np.concatenate([[0.0], np.cumsum([float(np.max(np.abs(turn_path[i + 1] - turn_path[i]))) for i in range(len(turn_path) - 1)])])
    turn_total = float(tcum[-1])
    print(f"  aim+approach: turn {math.degrees(turn_total):.0f}° while closing {total * 100:.0f} cm along the line…")

    # No per-episode pace randomness. A random pace scale / wobble is invisible
    # in the observation, so identical situations demanded different actions
    # and the policy learned to hesitate (aim dataset v1: 17% at 50k steps with
    # the lowest training loss of any run). Variety belongs in what the policy
    # can SEE: cube pose, start poses, disturbances and their recoveries.
    speed_scale = 1.0
    turn_scale = 1.0
    wob_hz = 0.0
    wob_phase = np.zeros(2)

    q_cmd = turn_path[0].copy()
    s_arc = 0.0
    tau = 0.0
    held = 0
    stalled = 0
    yielding = False
    replans = 0
    dt = 1.0 / fps
    last_log = -1.0
    for k in range(int(timeout_s * fps)):
        ik.set_q(q_cmd)
        cube_now = cube_pos(robot)
        # Bumped cube: once the turn is done, re-plan the rest of the approach
        # from the current pose toward where the cube actually is now.
        planned = _AIM_PLAN["cube"]
        moved = 0.0 if planned is None else float(np.linalg.norm(cube_now[:2] - planned[:2]))
        if (
            tau >= turn_total
            and planned is not None
            # the cube is not deliberately disturbed any more, but if the arm
            # brushes it the approach still re-plans toward where it now is
            and moved > AIM_REPLAN_BUMP_M
        ):
            goal_new = aim_point(cube_now) + _AIM_PLAN["in_offset"]
            new_chain = plan_approach_chain(ik, q_cmd, goal_new)
            ik.set_q(q_cmd)
            if new_chain is not None:
                q_aim = q_cmd.copy()
                ref = [q_aim] + list(new_chain)
                tips = []
                for q in ref:
                    ik.set_q(q)
                    tips.append(ik.tip_mid().copy())
                arc = np.concatenate(
                    [[0.0], np.cumsum([np.linalg.norm(tips[i + 1] - tips[i]) for i in range(len(ref) - 1)])]
                )
                total = float(arc[-1])
                s_arc = 0.0
                turn_path = [q_aim, q_aim]  # turn already complete
                tcum = np.array([0.0, 0.0])
                turn_total = 0.0
                tau = 0.0
                _AIM_PLAN["cube"] = cube_now.copy()
                _AIM_OVERLAY["goal_offset"] = (goal_new - aim_point(cube_now)).copy()
                replans += 1
                print(f"    cube moved {np.linalg.norm(cube_now[:2] - planned[:2]) * 100:.1f} cm — re-planned approach ({total * 100:.0f} cm to go)")
            else:
                # The full chain planner wants a standoff geometry it no longer
                # has mid-approach. Track the moved cube the cheap way instead:
                # shift the remaining tip targets by the cube's displacement,
                # re-solving each with the low-level axis IK.
                shift = cube_now - planned
                tail = []
                ik.set_q(q_cmd)
                tcp_off = ik.ee() - ik.tip_mid()
                q_prev_wp = q_cmd
                for q_wp in ref[1:]:
                    ik.set_q(q_wp)
                    tip_t = ik.tip_mid() + shift
                    q_new, pe, ae = plan_aim_axis(
                        ik, q_prev_wp, tip_t + tcp_off, aim_point(cube_now) - ik.hand(),
                        rng=None, iters=150,
                    )
                    if q_new is None or pe > 0.015 or ae > 8.0:
                        break
                    ik.set_q(q_new)
                    if not aim_pads_clear(ik):
                        break  # a shifted waypoint that dips is no plan at all
                    tail.append(q_new)
                    q_prev_wp = q_new
                if len(tail) == len(ref) - 1:
                    q_aim = q_cmd.copy()
                    ref = [q_aim] + tail
                    tips = []
                    for q in ref:
                        ik.set_q(q)
                        tips.append(ik.tip_mid().copy())
                    arc = np.concatenate([[0.0], np.cumsum(
                        [np.linalg.norm(tips[i + 1] - tips[i]) for i in range(len(ref) - 1)])])
                    total = float(arc[-1])
                    s_arc = min(s_arc, total)
                    turn_path = [q_aim, q_aim]
                    tcum = np.array([0.0, 0.0])
                    turn_total = 0.0
                    tau = 0.0
                    replans += 1
                _AIM_PLAN["cube"] = cube_now.copy()
                _AIM_OVERLAY["goal_offset"] = _AIM_PLAN["in_offset"].copy()
            ik.set_q(q_cmd)
        aim_err = aim_error_deg(ik, cube_now)
        dist = float(np.linalg.norm(aim_point(cube_now) - ik.hand()))
        t_now = k * dt
        v = approach_speed_mps(aim_err, dist) * speed_scale
        if turn_total > 1e-6:
            v *= min(1.0, tau / turn_total)
        turn_rate = math.radians(AIM_TURN_RATE_DEG_S) * turn_scale
        tau_prev = tau
        tau = min(turn_total, tau + turn_rate * dt)
        turn_off = _polyline_at(turn_path, tcum, tau) - q_aim
        # The turn path and the chain are each swept for pad clearance, but
        # their superposition is not: before advancing the approach, check the
        # candidate pose in the kinematic model. If it would dip the pads, hold
        # the approach this tick and let the turn continue — once the turn is
        # done the blend IS the verified chain pose, so progress always resumes.
        if yielding:
            if tau < turn_total:
                v = 0.0  # approach waits for the turn
            else:
                yielding = False
        s_try = min(total, s_arc + v * dt)

        def step_toward(q_goal: np.ndarray) -> np.ndarray:
            # One scaled step toward q_goal: the command stays on the straight
            # joint-space line to it. Per-joint clipping bent that line and put
            # the commanded pose somewhere no sweep had checked (traced: target
            # 3.5 cm clear, commanded 0.7 cm, pads into the table).
            dq = q_goal - q_cmd
            big = float(np.max(np.abs(dq)))
            return q_cmd + dq * min(1.0, math.radians(1.6) / max(big, 1e-9))

        def clear(q: np.ndarray) -> bool:
            ik.set_q(q)
            return aim_pads_clear(ik)

        # Blend the two verified paths as a WEIGHTED MIX, not by adding the
        # remaining turn as a joint offset onto a chain pose: with a large turn
        # outstanding that sum is an arbitrary pose belonging to neither path
        # (seed 7 ep 2 needs 93 deg and the arm stalled in front of the cube).
        # A convex mix is the turn path while the turn is young and the chain
        # once it is done, and always lies between the two.
        def blend(arc_s: float, turn_tau: float) -> np.ndarray:
            w = 1.0 if turn_total <= 1e-6 else min(1.0, turn_tau / turn_total)
            q_turn = _polyline_at(turn_path, tcum, turn_tau)
            return (1.0 - w) * q_turn + w * _polyline_at(ref, arc, arc_s)

        # Check the pose that will actually be COMMANDED. If advancing the
        # approach would dip the pads, hold the approach this tick; if even the
        # turn alone would, hold the turn too (the arm pauses a tick).
        q_target = blend(s_try, tau)
        q_next = step_toward(q_target)
        if clear(q_next):
            s_arc = s_try
            stalled = 0
        elif AIM_STEREOTYPED:
            # The turn path and the chain are each verified, but their BLEND is
            # not, and it can dip the pads into the table. Hold the approach for
            # this tick and let the turn continue: a pause caused by the arm's
            # own geometry is a deterministic function of the scene (unlike the
            # yield/back-off this replaces, which searched and was erratic).
            held += 1
            q_next = step_toward(blend(s_arc, tau))
        else:
            held += 1
            q_target = blend(s_arc, tau)
            q_next = step_toward(q_target)
            if not clear(q_next):
                stalled += 1
                # Both vetoed: the blend is stuck between two paths that were
                # each verified on their own. Yield: back the approach off
                # (toward s=0, where the blend IS the verified turn path) until
                # the turn has completed, then approach along the verified chain.
                yielding = True
                s_arc = max(0.0, s_arc - AIM_SPEED_MAX_MPS * dt)  # fixed rate: v is 0 while yielding
                q_target = blend(s_arc, tau)
                q_next = step_toward(q_target)
                if not clear(q_next):
                    # Hold position rather than command a pose no check has
                    # approved. This used to "force the verified turn through"
                    # after 1.5 s, which is precisely how the pads ended up in
                    # the table. If nothing clears, abandon the scene: a
                    # skipped trial costs nothing (failures are dropped), a
                    # collision costs a wrecked episode.
                    q_next = q_cmd
                    if stalled > int(1.5 * fps):
                        print("  aim+approach: no clear way in from here — skipping this scene")
                        return False
                if stalled == 1:
                    print("    clearance hold: yielding the approach until the turn completes")
        q_cmd = q_next
        _command_q(robot, ik, q_cmd, _approach_grip(s_arc / max(total, 1e-9)), fps)
        if s_arc >= 0.85 * total and finger_table_graze(robot, arm, min_force_n=2.0):
            # Pads already brushing the table on the last stretch: the target
            # is as low as it gets (thin bars). Stop here before the servo
            # drives the contact up to a hard fault.
            print("    approach: pads touching the table near the target — stopping here")
            break
        if grasp_table_fault(robot, arm) is not None:
            ik.set_q(q_cmd)
            cmd_low = min(finger_lowest_z_model(ik.model, ik.data, arm)) - TABLE_TOP_Z
            ik.set_q(q_target)
            tgt_low = min(finger_lowest_z_model(ik.model, ik.data, arm)) - TABLE_TOP_Z
            phys_low = min(finger_lowest_z(robot, arm)) - TABLE_TOP_Z
            print(
                f"  aim+approach: TABLE HIT {grasp_table_fault(robot, arm)} — aborting "
                f"[planned pad clearance: target {tgt_low * 100:.1f} cm, commanded {cmd_low * 100:.1f} cm; "
                f"physical {phys_low * 100:.1f} cm | turn {100 * tau / max(turn_total, 1e-9):.0f}% "
                f"approach {100 * s_arc / max(total, 1e-9):.0f}% aim {aim_err:.0f}°]"
            )
            return False
        t = k * dt
        if t - last_log >= 1.0:
            print(
                f"    t={t:4.1f}s aim={aim_err:5.1f}° dist={dist * 100:3.0f}cm "
                f"speed={v * 100:4.1f} cm/s progress={100 * s_arc / max(total, 1e-9):3.0f}%"
            )
            last_log = t
        if s_arc >= total and tau >= turn_total and float(np.max(np.abs(q_cmd - q_target))) < 1e-4:
            break
    else:
        print("  fail: aim+approach timed out")
        return False
    ik.set_q(ref[-1])
    planned_dz = float(ik.tip_mid()[2] - aim_point(cube_pos(robot))[2])
    ik.set_q(q_cmd)
    got_dz = float(tip_mid_world(robot, arm)[2] - aim_point(cube_pos(robot))[2])
    print(f"  aim+approach: grasp height vs aim point — planned {planned_dz * 100:+.1f} cm, physical {got_dz * 100:+.1f} cm")
    if held:
        print(f"  aim+approach: held the approach {held} ticks for pad clearance while turning")
    if replans:
        print(f"  aim+approach: re-planned {replans}x for a bumped cube")
    play_joint_path(robot, ik, q_cmd, q_cmd, _approach_grip(1.0), fps, _APPROACH_SETTLE_S["s"], "settle")  # physical arm trails
    ok, msg = tips_straddle_cube(robot, arm, cube_pos(robot))
    if ok:
        print(f"  fingers around the cube ({msg})")
        return True
    print(f"  fail: ended without the fingers around the cube ({msg})")
    return False


def run_aim_trial(
    robot: MujocoBiOpenArm,
    ik: PositionOnlyIK,
    fps: int,
    cube0: np.ndarray,
    rng: np.random.Generator,
    hold_s: float = 1.0,
) -> bool:
    """Stage 1 of the natural-motion pick: bring the wrist to a pre-aim standoff
    behind the cube and point the gripper straight at it, hold, done. Later
    stages extend this with the reach along the aim line."""
    q_now = _cmd_seed(robot, ik.arm.side)
    ik.set_q(q_now)
    cube = cube_pos(robot)
    print(f"  aim: before={aim_error_deg(ik, cube):.0f}° off the wrist->cube line")
    _AIM_OVERLAY["cube"] = cube.copy()
    try:
        return _run_aim_trial(robot, ik, fps, cube, q_now, rng, hold_s)
    finally:
        _AIM_OVERLAY["cube"] = None
        _AIM_OVERLAY["goal_offset"] = None
        _draw_aim_overlay(robot, ik)


# Recovery data comes from REAL misses only: a failed grasp is retried from
# wherever the arm ended up, instead of ending the episode. Injected
# disturbances (shoving the cube mid-approach) were removed — they muddied the
# demonstrations more than they taught recovery.
RECOVERY_MAX_ATTEMPTS = 3
_DISTURB: dict = {}


def _run_aim_trial(robot, ik, fps, cube, q_now, rng, hold_s) -> bool:
    # Only the nudge: a cube that visibly slides EXPLAINS the correction that
    # follows. The "drop" disturbance (release the lifted cube, re-pick) was
    # removed — the release had no visible cause, so it taught the policy that
    # letting go after a lift is sometimes right, keyed on a random number it
    # cannot see.
    for attempt in range(RECOVERY_MAX_ATTEMPTS):
        if attempt > 0:
            print(f"  RETRY {attempt}: re-aiming at the cube from here")
            set_gripper(robot, ik, FINGER_OPEN_M, fps, hold_s=0.6)
            q_now = _cmd_seed(robot, ik.arm.side)
            ik.set_q(q_now)
            cube = cube_pos(robot)
            _AIM_OVERLAY["cube"] = cube.copy()
        planned = plan_aim_at_cube(ik, q_now, cube, rng)
        if planned is None:
            print("  fail: no reachable aim pose + approach for this cube")
            return False
        q_aim, chain = planned
        waypoints = plan_via_lift(ik, q_now, q_aim)
        if waypoints is None:
            print("  fail: every path to the aim pose sweeps the fingers through the table")
            return False
        if not execute_aim_and_approach(robot, ik, fps, [q_now] + waypoints, chain, rng):
            if grasp_table_fault(robot, ik.arm) is not None:
                return False  # a hard table strike is not something to demonstrate
            continue  # fingers not around the cube: back off and try again
        ik.set_q(_cmd_seed(robot, ik.arm.side))
        print(
            f"  around cube: aim {aim_error_deg(ik, cube_pos(robot)):.1f}° off the line, "
            f"pinch tilt {pinch_tilt_deg(ik):.1f}°"
        )
        if not grasp_and_lift(robot, ik, fps, rng, hold_s=0.0):
            continue  # slipped or missed: retry
        # hold aloft; the episode ends here
        q_hold = _cmd_seed(robot, ik.arm.side)
        ik.set_q(q_hold)
        for _ in range(int(hold_s * fps)):
            _hold_fingers(robot, ik, _AIM_LAST_HOLD["m"])
            precise_sleep(1.0 / fps)
        return True
    print("  fail: out of attempts")
    return False


# Squeeze this far past where the pads block on the cube. The aim approach does
# not align the pads to the cube faces, so the cube may sit diagonally between
# them; a firm squeeze holds it regardless (legacy used 13 mm ~ 10 N/finger).
AIM_SQUEEZE_PAST_BLOCK_M = 0.018


def close_on_cube(robot: MujocoBiOpenArm, ik: PositionOnlyIK, fps: int) -> float:
    """Close until the pads block on the cube, then squeeze a fixed distance
    past the block point. Unlike set_gripper() this does not require the
    opening to reach a preset width — a diagonally-held cube blocks wider.
    Returns the commanded hold opening, or -1.0 (with a printed reason)."""
    arm = ik.arm
    ik.set_q(_cmd_seed(robot, arm.side))
    q_hold = ik.q().copy()
    cmd = float(np.clip(_finger_opening_m(robot, arm.side), 0.0, FINGER_OPEN_M))
    step = GRIP_RAMP_MPS / fps
    history: list[float] = []
    deadline = time.perf_counter() + 4.0
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        ik.set_q(q_hold)
        cmd = max(0.0, cmd - step)
        _hold_fingers(robot, ik, cmd)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        opened = _finger_opening_m(robot, arm.side)
        history.append(opened)
        stalled = len(history) >= 10 and (history[-10] - opened) < 0.0005
        if stalled:
            summary = grasp_contact_summary(robot, arm)
            # blocked on the target: both pad faces touching, or a firm
            # contact on the finger bodies (a bar held low on the plates)
            if summary["pinching"] or float(summary["cube_force_n"]) >= 3.0:
                squeeze = _AIM_TARGET["squeeze_m"]
                hold = max(0.0, opened - (AIM_SQUEEZE_PAST_BLOCK_M if squeeze is None else squeeze))
                # ramp the last bit of squeeze on, then confirm force
                while cmd > hold:
                    ik.set_q(q_hold)
                    cmd = max(hold, cmd - step)
                    _hold_fingers(robot, ik, cmd)
                    precise_sleep(1.0 / fps)
                for _ in range(max(8, fps // 3)):
                    ik.set_q(q_hold)
                    _hold_fingers(robot, ik, hold)
                    precise_sleep(1.0 / fps)
                summary = grasp_contact_summary(robot, arm)
                print(
                    f"  gripper[{arm.side}]: blocked at {opened * 1000:.1f} mm, squeezing to "
                    f"{hold * 1000:.1f} mm — pinch {summary['cube_force_n']:.1f} N"
                )
                if summary["pinching"] and float(summary["cube_force_n"]) >= 6.0:
                    return hold
                print("  fail: weak or one-sided pinch after squeeze — not lifting")
                return -1.0
            if cmd <= 0.0:
                tgt = cube_pos(robot)
                tips = finger_tips_world(robot, arm)
                mid = tips.mean(axis=0)
                print(
                    f"  fail: gripper closed to {opened * 1000:.1f} mm without the target between the pads "
                    f"[target-vs-tips: dxy={np.linalg.norm(mid[:2] - tgt[:2]) * 100:.1f} cm, "
                    f"tips z-above-target={(mid[2] - tgt[2]) * 100:.1f} cm, sep={np.linalg.norm(tips[0] - tips[1]) * 100:.1f} cm]"
                )
                return -1.0
    print(f"  fail: gripper still moving at {_finger_opening_m(robot, arm.side) * 1000:.1f} mm after 4 s")
    return -1.0


AIM_LIFT_M = 2.0 * CUBE_HALF        # lift the cube its own height off the table
AIM_LIFT_SPEED_MPS = 0.05


def grasp_and_lift(
    robot: MujocoBiOpenArm, ik: PositionOnlyIK, fps: int, rng: np.random.Generator, hold_s: float = 1.0
) -> bool:
    """Stage 3: close on the cube, then lift it its own height straight up.

    Close and lift reuse what the legacy grasp learned the hard way: do not
    lift until the fingers have actually closed on the cube, and keep the
    wrist frozen while the cube can still touch the table — any orientation
    correction there pries it out of the pads.
    """
    arm = ik.arm
    ik.set_q(_cmd_seed(robot, arm.side))
    print(f"  close {arm.side}…")
    hold_grip = close_on_cube(robot, ik, fps)
    if hold_grip < 0.0:
        return False
    _AIM_LAST_HOLD["m"] = hold_grip

    q_now = _cmd_seed(robot, arm.side)
    ik.set_q(q_now)
    tip0 = tip_mid_world(robot, arm)
    wrist_hold = q_now[4:7].copy()
    cube0 = cube_pos(robot)
    n = max(2, int(AIM_LIFT_M / AIM_LIFT_SPEED_MPS * fps))
    print(f"  lift {AIM_LIFT_M * 100:.0f} cm ({n / fps:.1f}s, wrist frozen)…")
    q_prev = q_now.copy()
    # Closed-loop on the MEASURED cube height: the PD-tracked arm sags under
    # load (tips reach ~3.5 of a commanded 5 cm), so after the nominal profile
    # keep raising the command until the cube has actually risen its height.
    extra = 0.0
    for k in range(n + int(2.0 * fps)):
        u = min(1.0, (k + 1) / n)
        s_u = u * u * (3.0 - 2.0 * u)
        if u >= 1.0:
            if float(cube_pos(robot)[2] - cube0[2]) >= AIM_LIFT_M:
                break
            extra = min(extra + 0.04 / fps, 0.05)  # 4 cm/s, at most 5 cm over
        tip_t = tip0 + np.array([0.0, 0.0, s_u * AIM_LIFT_M + extra])
        for _ in range(3):
            ik.step_tip_mid(tip_t, max_dq=math.radians(1.6), freeze_wrist=wrist_hold)
        q = _rate_limit_q(ik.q(), q_prev, math.radians(1.4))
        ik.set_q(q)
        _hold_fingers(robot, ik, hold_grip)
        precise_sleep(1.0 / fps)
        q_prev = q.copy()
    # hold aloft briefly (the cube demo ends here; a carry that follows passes 0), then judge
    for _ in range(int(hold_s * fps)):
        ik.set_q(q_prev)
        _hold_fingers(robot, ik, hold_grip)
        precise_sleep(1.0 / fps)
    rise = float(cube_pos(robot)[2] - cube0[2])
    tip_rise = float(tip_mid_world(robot, arm)[2] - tip0[2])
    print(f"  lift: tips rose {tip_rise * 100:.1f} cm, cube rose {rise * 100:.1f} cm, grip {_finger_opening_m(robot, arm.side) * 1000:.1f} mm")
    if rise >= 0.8 * AIM_LIFT_M:
        print(f"  lifted the cube {rise * 100:.1f} cm — success")
        return True
    print(f"  fail: cube only rose {rise * 100:.1f} cm (dropped or slipped)")
    return False


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
        "--motion",
        choices=["legacy", "aim"],
        default="legacy",
        help="legacy: the original over-the-top grasp. aim: natural-motion rebuild, "
        "stage 1 — turn the gripper to aim along the wrist->cube line, hold 3s, next cube.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose per-trial output (cube pose, IK plans, phase logs). "
        "Default is quiet: one 'episode N' line per saved episode.",
    )
    args = parser.parse_args()

    global _RECORDER, _AIM_OVERLAY_ENABLED
    _AIM_OVERLAY_ENABLED = bool(args.debug)
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
    import os

    if not args.debug:
        # The episode save path is noisy on stderr: HF datasets shows a Map
        # progress bar and the AV1 encoder (SVT, via libav) prints its whole
        # config banner per video. Silence both at the source where an API
        # exists; anything left is caught by the fd-level redirect below.
        try:
            import datasets

            datasets.disable_progress_bars()
        except Exception:
            pass
        try:
            import av

            av.logging.set_level(av.logging.PANIC)
        except Exception:
            pass

    @contextlib.contextmanager
    def _trial_output():
        # Quiet by default: swallow the per-trial chatter so the console is
        # one clean 'episode N' line per saved episode. --debug restores it.
        # stderr must be redirected at the file-descriptor level: the video
        # encoder logs from C code, below Python's sys.stderr.
        if args.debug:
            yield
            return
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_err = os.dup(2)
        try:
            os.close(2)
            os.dup2(devnull, 2)
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            os.dup2(saved_err, 2)
            os.close(saved_err)
            os.close(devnull)

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
                if args.motion == "aim":
                    ok = run_aim_trial(robot, ik, args.fps, cube0, rng)
                else:
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
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        if _RECORDER is not None:
            _RECORDER.finalize()
        import os as _os

        _os._exit(0)
