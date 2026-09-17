#!/usr/bin/env python3
"""Caddy picking: stacks of brown chocolate bars on coloured pads; "get bar from blue pad".

Each trial:
  1. Arrange N stacks (default 4, 1-3 bars each) of identical brown square bars
     (50 x 50 x 25 mm) on an arc in front of the arms, each stack on a coloured
     pad. The pad colours are a fresh random draw from the palette every trial,
     so a colour says nothing about position and position nothing about colour:
     the policy has to find the pad the prompt names.
  2. The prompt names one pad: "get bar from blue pad".
     Three quarters of episodes start with the arms exactly where the previous
     episode ended (as they would in deployment); the rest from the cube
     picker's random start poses.
  3. The arm is the one on the pad's side of the table — left arm for pads left
     of the robot, right arm for pads right of it. N is even by default so no
     pad sits on the centreline where that rule is ambiguous. (An odd N puts a
     slot there; it always goes to the LEFT arm.)
  4. The top bar of that stack is picked with the natural-motion machinery of
     the cube picker (aim, close in along the aim line, squeeze; retries after
     a miss); the fingers start closing over the last third of the approach so
     the squeeze lands as the arm arrives. Then straight up to a transit height
     that clears a pile in the middle, level across to the centre of the table
     in front of the robot, straight down, and drop.
     A pile of 0-4 bars already sits on the centre spot when the episode starts
     (the robot has usually fetched some before); the new bar goes on top.
  5. Success: exactly that bar rests on the pile (or the spot) and no other
     bar moved.

Recording (--record) stores each successful trial as one episode whose task
string is the prompt.

Usage:
  MUJOCO_GL=egl python scripts/random_caddy_pick.py --trials 5 --debug
  MUJOCO_GL=egl python scripts/random_caddy_pick.py --no-viewer \\
      --record local/openarm_caddy_pick_all_300 --episodes 300 --cameras all
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import random_cube_pick as rcp  # noqa: E402  (shared grasp machinery)

# The bar: a square slab of chocolate, all bars identical and brown.
BAR_HALF = np.array([0.025, 0.025, 0.0125])  # 50 x 50 x 25 mm
BAR_RGBA = (0.36, 0.22, 0.10, 1.0)
# A bar rests on the bar below through four tiny feet under its box (see
# scene.xml: bar boxes never collide with each other, which is what made
# MuJoCo's box-box collider overflow), so stacked bars sit 1 mm apart.
BAR_FOOT_M = 0.001
BAR_PITCH_M = 2 * BAR_HALF[2] + BAR_FOOT_M  # centre-to-centre height in a stack

# Grasp geometry for the aim machinery (same planner/executor as the cube
# picker; only the target description changes). The finger plates are ~7.2 cm
# tall centred on the "tip" point, so with the tips this far above the bar's
# centre the plate bottoms sit ~2-3 mm above whatever the bar rests on (the
# table, or the bar below it) and hold nearly the bar's full height. (3.8 cm
# was tried: the plates covered only the top 6 mm and closed past the bar;
# 3.0 cm gripped, but visibly high on the bar.)
#
# That is where the tips must PHYSICALLY end up. The arm sags under the cube
# picker's servo gains — at the grasp, measured against the plan: ~1.6 cm at
# the end slots (39 cm from the shoulder), ~2.2 cm at the middle slots (47 cm) —
# and a 2.5 cm bar has no room for that (the plates landed on the bar below
# the target: one-sided pinch, no lift). So the plan aims the tips higher by
# the expected sag for the reach FROM THE SHOULDER (base-relative reach put
# the end slots 6 mm too high: they are far from the base but near the arm).
# Stiffer servos were tried instead (3x) and cured the sag but made the carry
# visibly shaky: the arm snapped to each 30 Hz command step (frame-to-frame
# speed jitter 7.6 vs 3.0 cm/s). The gains stay at the cube picker's.
BAR_TIP_ABOVE_M = 0.025
BAR_PAD_CLEARANCE_M = 0.002
ARM_GAIN_SCALE = 1.0


def sag_estimate_m(shoulder_reach_m: float) -> float:
    """Expected droop of the fingertips below the planned height when the
    target is this far (xy) from the arm's shoulder."""
    return float(np.clip(0.016 + 0.07 * (shoulder_reach_m - 0.39), 0.010, 0.030))

MAX_STACKS = 9
MAX_PER_STACK = 3
MAX_BARS = MAX_STACKS * MAX_PER_STACK  # 27 bar bodies in the scene
TABLE_TOP_Z = rcp.TABLE_TOP_Z

# The arc of pads: centred on the robot base, left-most slot at +y. The radius
# puts the middle pad at the far edge of the table while every slot stays in
# reliable reach; the span keeps the end pads inside the table's side edges.
ARC_RADIUS = 0.48
ARC_SPAN_DEG = 46.0
ARC_RADIUS_JITTER = 0.015
ARC_ANGLE_JITTER_DEG = 1.5

# Where the bar goes: the centre of the table, straight in front of the robot.
# Fixed (nothing in the image marks it, so it must not vary).
DROP_XY = np.array([0.31, 0.0])
# Release with the bar's underside this far above whatever is at the spot. The
# arm sags ~1 cm more as it lowers, so this lands nearer 2 cm; and the descent
# stops early the moment the bar touches down (set onto a pile while still
# gripped, it shoved the top bars off).
DROP_HEIGHT_M = 0.03
TOUCHDOWN_M = 0.004
PLACE_TOL = 0.06  # bar counts as delivered within this radius of the spot
# Bars already collected sit in a pile on the spot when an episode starts (the
# robot has usually fetched some before this one), 0..PILE_MAX of them.
PILE_MAX = 4
PILE_XY_JITTER = 0.004
PILE_YAW_JITTER = 0.15

# Where an idle arm rests: hand well out over the table, level with the drop
# spot and 20 cm to its side, not the cube picker's deep park behind the table
# edge — the next pick then starts close to where it has to go. Pads are far
# out on the arc and the drop spot is on the centreline, so nothing conflicts.
TUCK_TIP_TARGET = {"right": np.array([0.30, -0.20, 0.50]), "left": np.array([0.30, 0.20, 0.50])}

# Fingers start closing over the last stretch of the approach, from fully open
# (44 mm) to just wider than the bar, so the squeeze lands as the arm arrives.
# 32 mm is ~7.6 cm between the pads for a 5 cm bar.
PRECLOSE_FROM = 0.65     # approach progress at which the fingers start closing
PRECLOSE_M = 0.032
APPROACH_SETTLE_S = 0.2


def approach_grip(progress: float) -> float:
    if progress <= PRECLOSE_FROM:
        return rcp.FINGER_OPEN_M
    u = (progress - PRECLOSE_FROM) / (1.0 - PRECLOSE_FROM)
    return rcp.FINGER_OPEN_M + (PRECLOSE_M - rcp.FINGER_OPEN_M) * u

PAD_HALF_XY = 0.045
PAD_Z = TABLE_TOP_Z + 0.0015  # 3 mm slab, visual only
PAD_PARK = (-0.9, 0.0, -0.5)  # unused pads hide under the floor
WAREHOUSE = [(-0.55, -0.66 + 0.12 * i, 0.0135) for i in range(MAX_BARS)]

# Nine well-separated colours (four used by default). Table is brown, bars are
# brown: none of these is anywhere near either.
PALETTE: list[tuple[str, tuple[float, float, float, float]]] = [
    ("red", (0.85, 0.15, 0.12, 1.0)),
    ("green", (0.12, 0.65, 0.18, 1.0)),
    ("blue", (0.12, 0.30, 0.90, 1.0)),
    ("yellow", (0.95, 0.85, 0.10, 1.0)),
    ("white", (0.96, 0.96, 0.94, 1.0)),
    ("black", (0.05, 0.05, 0.05, 1.0)),
    ("purple", (0.55, 0.15, 0.70, 1.0)),
    ("orange", (0.95, 0.50, 0.08, 1.0)),
    ("pink", (0.98, 0.55, 0.75, 1.0)),
]
# (no cyan: the wrist links' mirror finish reflects the sky and reads as cyan)


def make_prompt(colour: str) -> str:
    return f"get bar from {colour} pad"


# --- scene helpers -----------------------------------------------------------


def set_bar_pose(robot, i: int, x: float, y: float, z: float, yaw: float = 0.0) -> None:
    import mujoco

    m, d = robot._model, robot._data
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"bar_{i}_free")
    adr, dof = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    d.qpos[adr : adr + 3] = [x, y, z]
    d.qpos[adr + 3 : adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    d.qvel[dof : dof + 6] = 0.0


def bar_pos(robot, i: int) -> np.ndarray:
    import mujoco

    bid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_BODY, f"bar_{i}")
    return robot._data.xpos[bid].copy()


def set_pad(robot, i: int, xy: tuple[float, float] | None, rgba=None, half_xy: float = PAD_HALF_XY) -> None:
    """Place (or park, with xy=None) and colour visual pad geom ``cpad_i``."""
    import mujoco

    m = robot._model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"cpad_{i}")
    if xy is None:
        m.geom_pos[gid] = PAD_PARK
    else:
        m.geom_pos[gid] = [xy[0], xy[1], PAD_Z]
        m.geom_size[gid] = [half_xy, half_xy, 0.0015]
    if rgba is not None:
        m.geom_rgba[gid] = rgba


def hide_legacy_pads(robot) -> None:
    """The colour-sort scene's grey pads share the table; this task has its own."""
    import mujoco

    for name in ("pad_left", "pad_right"):
        gid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            robot._model.geom_pos[gid] = PAD_PARK  # alpha 0 still rendered as a ghost; move them away


def bar_centre_z(level: int) -> float:
    """Resting height of a bar's centre at ``level`` of a stack (0 = on the table)."""
    return TABLE_TOP_Z + BAR_FOOT_M + BAR_HALF[2] + BAR_PITCH_M * level


def stack_top_z(n_bars: int) -> float:
    """Top surface of a stack of ``n_bars`` (the table itself for 0)."""
    return TABLE_TOP_Z + BAR_PITCH_M * n_bars


# --- one trial ---------------------------------------------------------------


class Trial:
    """One arranged scene: N stacks on coloured pads, the prompt, the plan."""

    def __init__(self, robot, rng: np.random.Generator, n_stacks: int):
        import mujoco

        self.robot = robot
        self.n = n_stacks
        self.sizes = [int(rng.integers(1, MAX_PER_STACK + 1)) for _ in range(self.n)]
        self.pile_n = min(int(rng.integers(0, PILE_MAX + 1)), MAX_BARS - sum(self.sizes))

        # slots on the arc, left (+y) to right (-y), with a little jitter
        angles = np.deg2rad(np.linspace(ARC_SPAN_DEG, -ARC_SPAN_DEG, self.n))
        self.stack_xy: list[tuple[float, float]] = []
        self.stack_yaw: list[float] = []
        for th in angles:
            th_j = float(th + np.deg2rad(rng.uniform(-ARC_ANGLE_JITTER_DEG, ARC_ANGLE_JITTER_DEG)))
            r_j = ARC_RADIUS + float(rng.uniform(-ARC_RADIUS_JITTER, ARC_RADIUS_JITTER))
            self.stack_xy.append((r_j * math.cos(th_j), r_j * math.sin(th_j)))
            self.stack_yaw.append(th_j)  # radial direction from the robot base
        self.centre_slot = (self.n - 1) // 2 if self.n % 2 == 1 else None

        # pad colours: a random draw of n distinct palette entries
        ids = rng.permutation(len(PALETTE))[: self.n]
        self.colours = [PALETTE[i] for i in ids]
        # pads must not touch when the arc is crowded (9 slots -> ~9.6 cm pitch)
        pitch = 2 * ARC_RADIUS * math.sin(math.radians(2 * ARC_SPAN_DEG) / (self.n - 1) / 2)
        pad_half = min(PAD_HALF_XY, 0.42 * pitch)
        for i in range(MAX_STACKS):
            if i < self.n:
                set_pad(robot, i, self.stack_xy[i], self.colours[i][1], pad_half)
            else:
                set_pad(robot, i, None)

        m = robot._model
        for i in range(MAX_BARS):
            set_bar_pose(robot, i, *WAREHOUSE[i])
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"bar_{i}")
            # real bars are never quite identical: +-1%
            m.geom_size[gid] = BAR_HALF * (1.0 + rng.uniform(-0.01, 0.01, size=3))
            m.geom_rgba[gid] = BAR_RGBA
        self.stack_bars: list[list[int]] = []
        bar_i = 0
        for s_idx, (x, y) in enumerate(self.stack_xy):
            bars = []
            for level in range(self.sizes[s_idx]):
                jx, jy = rng.uniform(-0.0015, 0.0015, size=2)
                jyaw = float(rng.uniform(-0.05, 0.05))
                set_bar_pose(robot, bar_i, x + jx, y + jy, bar_centre_z(level),
                             yaw=self.stack_yaw[s_idx] + jyaw)
                bars.append(bar_i)
                bar_i += 1
            self.stack_bars.append(bars)
        # the pile of bars already fetched, on the drop spot
        self.pile_bars: list[int] = []
        for level in range(self.pile_n):
            jx, jy = rng.uniform(-PILE_XY_JITTER, PILE_XY_JITTER, size=2)
            set_bar_pose(robot, bar_i, DROP_XY[0] + jx, DROP_XY[1] + jy, bar_centre_z(level),
                         yaw=float(rng.uniform(-PILE_YAW_JITTER, PILE_YAW_JITTER)))
            self.pile_bars.append(bar_i)
            bar_i += 1
        rcp.zero_sim_velocity(robot)
        mujoco.mj_forward(robot._model, robot._data)
        self.initial_pos = {i: bar_pos(robot, i) for i in range(MAX_BARS)}
        self.initial_qpos = robot._data.qpos.copy()
        self.bar_qpos_adr = [
            (m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"bar_{i}_free")],
             m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"bar_{i}_free")])
            for i in range(MAX_BARS)
        ]

        self.target = int(rng.integers(0, self.n))
        self.colour = self.colours[self.target][0]
        self.prompt = make_prompt(self.colour)
        self.place_xy = DROP_XY.copy()
        self.side = self.arm_for(self.target)
        self.fail_stage: str | None = None
        self.moved: int | None = None

    def restore_bars(self) -> None:
        """Put every bar back exactly as arranged (after a start pose disturbed one)."""
        import mujoco

        d = self.robot._data
        for adr, dof in self.bar_qpos_adr:
            d.qpos[adr : adr + 7] = self.initial_qpos[adr : adr + 7]
            d.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.robot._model, d)

    def arm_for(self, s_idx: int) -> str:
        """Left arm for pads left of the robot (+y), right arm for pads right of
        it. The centreline slot of an odd arc goes left, always."""
        if s_idx == self.centre_slot:
            return "left"
        return "left" if self.stack_xy[s_idx][1] > 0.0 else "right"

    def top_bar(self) -> int:
        bars = self.stack_bars[self.target]
        return max(bars, key=lambda i: bar_pos(self.robot, i)[2])

    def level_of(self, bar: int) -> int:
        """Which level the bar currently rests at (0 = table), from its live height."""
        return int(round((float(bar_pos(self.robot, bar)[2]) - bar_centre_z(0)) / BAR_PITCH_M))

    def arranged_bars(self) -> list[int]:
        return [i for bars in self.stack_bars for i in bars] + self.pile_bars

    def describe(self) -> str:
        cols = " ".join(f"{c[0]}:{s}" for c, s in zip(self.colours, self.sizes, strict=True))
        return (f"\"{self.prompt}\"  slot {self.target + 1}/{self.n} y={self.stack_xy[self.target][1]:+.2f} "
                f"-> {self.side} arm   [{cols}]  pile:{self.pile_n}")


BAR_OBSTACLE_RADIUS_M = 0.055  # keep fingertips this far from other bars' centres


def aim_target_for_bar(robot, ik, trial: Trial, bar: int) -> None:
    """Point the aim machinery at ``bar`` where it is NOW: tip offset, the
    surface under it, approach along the slot's radial, every other bar an
    obstacle. Re-evaluated before each attempt, so a retry after a bar tumbled
    off its stack plans for where it landed."""
    level = max(0, trial.level_of(bar))
    support_z = stack_top_z(level)
    obstacles = [(bar_pos(robot, i), BAR_OBSTACLE_RADIUS_M) for i in trial.arranged_bars() if i != bar]
    bp = bar_pos(robot, bar)
    sag = sag_estimate_m(float(np.linalg.norm(bp[:2] - rcp.shoulder_pos(ik)[:2])))
    rcp.set_aim_target(
        BAR_TIP_ABOVE_M + sag, support_z, BAR_PAD_CLEARANCE_M,
        xy=bar_pos(robot, bar)[:2], azimuth=trial.stack_yaw[trial.target], obstacles=obstacles,
        # in_extra None: half a cube (= half a bar) so the pads centre on it.
        # squeeze None: the cube's squeeze; the bar is the cube's width.
    )


def grasp_plannable(robot, ik, trial: Trial, rng: np.random.Generator) -> bool:
    """Can the natural-motion planner reach the target's top bar from idle?"""
    bar = trial.top_bar()
    aim_target_for_bar(robot, ik, trial, bar)
    q_idle = np.deg2rad(ik.arm.idle_deg)
    planned = rcp.plan_aim_at_cube(ik, q_idle, bar_pos(robot, bar), rng)
    if planned is None:
        return False
    return rcp.plan_via_lift(ik, q_idle, planned[0]) is not None


def others_disturbed(robot, trial: Trial, exclude: set[int]) -> list[str]:
    """Bars (outside ``exclude``) that strayed from their arranged pose."""
    bad = []
    for i in trial.arranged_bars():
        if i in exclude:
            continue
        p0, p1 = trial.initial_pos[i], bar_pos(robot, i)
        dxy = float(np.linalg.norm(p1[:2] - p0[:2]))
        dz = abs(float(p1[2] - p0[2]))
        if dxy > 0.02 or dz > 0.006:
            bad.append(f"bar_{i} moved {dxy * 100:.1f}cm xy / {dz * 1000:.0f}mm z")
    return bad


def pick_bar(robot, ik, fps: int, trial: Trial, bar: int, rng: np.random.Generator) -> bool:
    """The cube picker's natural-motion pick (aim standoff, reach along the aim
    line, squeeze, lift), with its retries: a miss is re-aimed from wherever
    the arm is, and the retry is kept in the episode as recovery data. Same
    structure as rcp._run_aim_trial, but the aim target is refreshed from the
    bar's live pose before every attempt (a stack is not a lone cube)."""
    for attempt in range(rcp.RECOVERY_MAX_ATTEMPTS):
        if attempt > 0:
            print(f"  RETRY {attempt}: re-aiming at the bar from here")
            rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=0.6)
        if others_disturbed(robot, trial, {bar}):
            return False  # knocked something: nothing to recover into
        aim_target_for_bar(robot, ik, trial, bar)
        bp = bar_pos(robot, bar)
        rcp._AIM_OVERLAY["cube"] = bp.copy()
        q_now = rcp._cmd_seed(robot, ik.arm.side)
        ik.set_q(q_now)
        planned = rcp.plan_aim_at_cube(ik, q_now, bp, rng)
        if planned is None:
            print("  fail: no reachable aim pose + approach for this bar")
            trial.fail_stage = "plan"
            return False
        q_aim, chain = planned
        waypoints = rcp.plan_via_lift(ik, q_now, q_aim)
        if waypoints is None:
            print("  fail: every path to the aim pose sweeps the fingers through the table")
            trial.fail_stage = "plan"
            return False
        if not rcp.execute_aim_and_approach(robot, ik, fps, [q_now] + waypoints, chain, rng):
            if rcp.grasp_table_fault(robot, ik.arm) is not None:
                trial.fail_stage = "table strike"
                return False  # a hard table strike is not something to demonstrate
            trial.fail_stage = "approach"
            continue
        ik.set_q(rcp._cmd_seed(robot, ik.arm.side))
        print(f"  close {ik.arm.side}…")
        hold = rcp.close_on_cube(robot, ik, fps)
        if hold < 0.0:
            trial.fail_stage = "grasp"
            continue
        rcp._AIM_LAST_HOLD["m"] = hold
        outcome = carry_and_place(robot, ik, fps, trial.place_xy, hold, bar, stack_top_z(trial.pile_n))
        if outcome == "lost":
            trial.fail_stage = "grasp"  # slipped as it came off the stack: re-aim where it lies
            continue
        if outcome != "ok":
            trial.fail_stage = "carry"
            return False
        trial.fail_stage = None
        return True
    print("  fail: out of attempts")
    return False


# Lift straight up until the bar's underside is this high over the table, cross
# to the centre at that height, then straight down to the drop. High enough to
# clear a pile of bars already in the middle (the real box fills up).
TRANSIT_UNDERSIDE_M = 0.14
LIFT_SPEED_MPS = 0.12
CARRY_SPEED_MPS = 0.15
LOWER_SPEED_MPS = 0.12


def carry_and_place(robot, ik, fps: int, target_xy: np.ndarray, grip_m: float, bar: int,
                    spot_top_z: float = TABLE_TOP_Z) -> str:
    """Fingers are closed on the bar. Straight up until the bar's underside is
    TRANSIT_UNDERSIDE_M over the table, or higher if the pile at the spot needs
    it (dragging it sideways from rest shoved the rest of the stack along),
    level across to over the table centre, straight down to DROP_HEIGHT_M
    above ``spot_top_z`` (the table, or the top of the pile), release. Wrist frozen throughout: re-orienting a just-gripped bar pries it
    out of the pads (the cube picker learned that). The bar hangs wherever the
    pads caught it, so the tips are aimed at target + (tips - bar) and the BAR
    arrives over the spot.

    Returns "ok", "lost" (the bar parted from the pads early: a grasp failure,
    retryable) or "fail"."""
    arm = ik.arm
    q_prev = rcp._cmd_seed(robot, arm.side)
    ik.set_q(q_prev)
    wrist_hold = q_prev[4:7].copy()
    tip_start = rcp.tip_mid_world(robot, arm).copy()
    bar_start = bar_pos(robot, bar)

    def tick(tip_t: np.ndarray) -> None:
        nonlocal q_prev
        for _ in range(3):
            ik.step_tip_mid(tip_t, max_dq=math.radians(1.6), freeze_wrist=wrist_hold)
        q = rcp._rate_limit_q(ik.q(), q_prev, math.radians(1.4))
        ik.set_q(q)
        rcp._hold_fingers(robot, ik, grip_m)
        rcp.precise_sleep(1.0 / fps)
        q_prev = q.copy()

    def segment(tip_a: np.ndarray, tip_b: np.ndarray, speed: float, label: str,
                stop_below_z: float | None = None) -> float | None:
        """Ease-in-out straight tip move; returns the fraction travelled when the
        bar parted from the pads, or None if it stayed. With ``stop_below_z``
        the move ends as soon as the bar's underside reaches that height."""
        dist = float(np.linalg.norm(tip_b - tip_a))
        n = max(2, int(max(0.4, dist / speed) * fps))
        print(f"  {label}: {dist * 100:.0f} cm ({n / fps:.1f}s)…")
        for k in range(n):
            u = (k + 1) / n
            s_u = u * u * (3.0 - 2.0 * u)
            tick((1.0 - s_u) * tip_a + s_u * tip_b)
            if rcp.grasp_table_fault(robot, arm) is not None:
                print(f"  {label}: TABLE HIT {rcp.grasp_table_fault(robot, arm)} — aborting")
                return -1.0
            if float(np.linalg.norm(bar_pos(robot, bar) - rcp.tip_mid_world(robot, arm))) > 0.06:
                return u
            if stop_below_z is not None and float(bar_pos(robot, bar)[2]) - BAR_HALF[2] <= stop_below_z:
                print(f"  {label}: touched down at {u * 100:.0f}% — releasing here")
                break
        return None

    # 1. straight up to transit height, closed-loop on the bar's own height
    #    (the arm sags under load: keep raising the command until the bar is
    #    actually there)
    tip_bar_dz = float(tip_start[2] - bar_start[2])
    transit_bar_z = max(TABLE_TOP_Z + TRANSIT_UNDERSIDE_M, spot_top_z + 0.06) + BAR_HALF[2]
    lift_m = transit_bar_z - float(bar_start[2])
    n_lift = max(2, int(lift_m / LIFT_SPEED_MPS * fps))
    print(f"  lift: {lift_m * 100:.0f} cm straight up ({n_lift / fps:.1f}s)…")
    extra = 0.0
    for k in range(n_lift + int(1.5 * fps)):
        u = min(1.0, (k + 1) / n_lift)
        s_u = u * u * (3.0 - 2.0 * u)
        if u >= 1.0:
            if float(bar_pos(robot, bar)[2]) >= transit_bar_z - 0.005:
                break
            extra = min(extra + 0.04 / fps, 0.04)
        tick(tip_start + np.array([0.0, 0.0, s_u * lift_m + extra]))
        if k > int(0.5 * fps) and float(bar_pos(robot, bar)[2] - bar_start[2]) < 0.005:
            print("  lift: bar did not come up with the fingers — slipped")
            return "lost"
    rise = float(bar_pos(robot, bar)[2] - bar_start[2])
    if rise < 0.5 * lift_m:
        print(f"  lift: bar only rose {rise * 100:.1f} cm of {lift_m * 100:.0f} — slipped")
        return "lost"

    # 2. level across to over the centre, 3. straight down to the drop height
    tip0 = rcp.tip_mid_world(robot, arm).copy()
    bar0 = bar_pos(robot, bar)
    over = np.array([target_xy[0] + (tip0[0] - bar0[0]), target_xy[1] + (tip0[1] - bar0[1]), tip0[2]])
    lost_at = segment(tip0, over, CARRY_SPEED_MPS, "carry: across to the centre")
    if lost_at is None:
        tip2 = rcp.tip_mid_world(robot, arm).copy()
        bar2 = bar_pos(robot, bar)
        drop_bar_z = spot_top_z + DROP_HEIGHT_M + BAR_HALF[2]
        down = np.array([target_xy[0] + (tip2[0] - bar2[0]), target_xy[1] + (tip2[1] - bar2[1]),
                         drop_bar_z + float(tip2[2] - bar2[2])])
        lost_at = segment(tip2, down, LOWER_SPEED_MPS, "carry: down to the drop",
                          stop_below_z=spot_top_z + TOUCHDOWN_M)
    if lost_at is not None:
        if lost_at < 0:
            return "fail"
        print(f"  carry: bar parted from the pads {lost_at * 100:.0f}% of the way")
        return "lost"
    p = bar_pos(robot, bar)
    print(f"  drop: bar {np.linalg.norm(p[:2] - target_xy) * 100:.1f} cm off centre, "
          f"underside {(p[2] - BAR_HALF[2] - spot_top_z) * 100:.1f} cm above the spot")
    # open, and do not move until the fingers actually are
    if rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.0) < 0.0:
        rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.0)
    # back off to transit height: the next episode usually starts from here,
    # and the pile it finds under the hand may be taller than this one
    tip = rcp.tip_mid_world(robot, arm)
    up = np.array([tip[0], tip[1], max(tip[2] + 0.08, transit_bar_z + tip_bar_dz)])
    rcp.play_tip_cartesian(robot, ik, up, rcp.FINGER_OPEN_M, fps, 0.0, label="back off", freeze_wrist=True)
    return "ok"


def run_trial(robot, iks, fps: int, trial: Trial, rng: np.random.Generator) -> bool:
    ik = iks[trial.side]
    bar = trial.top_bar()
    trial.moved = bar
    rcp.set_target_body(f"bar_{bar}")
    try:
        bp = bar_pos(robot, bar)
        print(f"  pick: bar_{bar} (level {trial.level_of(bar)} of {trial.sizes[trial.target]}) "
              f"at ({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.3f}) with the {trial.side} arm -> centre")
        # no separate "open the gripper" step: the approach commands the open
        # width from its first tick, so the fingers open while the arm moves
        if not pick_bar(robot, ik, fps, trial, bar, rng):
            bad = others_disturbed(robot, trial, {bar})
            if bad:
                print(f"  fail: KNOCKED: {'; '.join(bad)}")
                trial.fail_stage = "knock-over"
            return False
    finally:
        rcp.set_target_body("cube")
        rcp.set_aim_target(rcp.AIM_TIP_ABOVE_CUBE_M, TABLE_TOP_Z)  # back to cube defaults
        rcp._AIM_OVERLAY["cube"] = None
        rcp._AIM_OVERLAY["goal_offset"] = None

    # settle, then judge
    for _ in range(int(0.4 * fps)):
        rcp.send_q(robot, ik, rcp.FINGER_OPEN_M)
        rcp.precise_sleep(1.0 / fps)
    p = bar_pos(robot, bar)
    d = float(np.linalg.norm(p[:2] - trial.place_xy))
    on_top = abs(float(p[2]) - bar_centre_z(trial.pile_n)) < 0.012  # flat on the pile (or the table)
    ok = d < PLACE_TOL and on_top
    print(f"  dropped bar_{bar}: {d * 100:.1f} cm from the centre, z={p[2]:.3f} "
          f"(pile of {trial.pile_n} expects {bar_centre_z(trial.pile_n):.3f}) -> {'OK' if ok else 'MISS'}")
    if not ok:
        trial.fail_stage = "place"
    bad = others_disturbed(robot, trial, {bar})
    if bad:
        print(f"  fail: KNOCKED: {'; '.join(bad)}")
        trial.fail_stage = "knock-over"
        ok = False
    return ok


STACK_KEEPOUT_M = 0.10  # a starting hand stays this far (xy) from every stack, unless well above it
# In deployment the next pick starts wherever the last drop ended, so most
# episodes begin from the previous episode's final pose; the rest from the cube
# picker's random starts, for variety.
CONTINUE_PROB = 0.75


def draw_start_poses(ik, other_ik, rng: np.random.Generator, tucked_prob: float):
    """The cube picker's start distribution (setup_start_pose), drawn WITHOUT
    touching the sim: three quarters of the time an arm starts from a jittered
    tuck, otherwise from a random pose in its workspace box; grippers start
    anywhere from closed to open (nearly closed when tucked)."""
    def one(ik_):
        tucked = rng.uniform() < tucked_prob
        if tucked:
            q = np.clip(rcp.jittered_tuck(ik_.arm, rng, ik_), ik_.lo, ik_.hi)
            g = float(rng.uniform(0.0, 0.006))
        else:
            q = rcp._random_start_q(ik_, ik_.arm, rng)
            g = float(rng.uniform(0.0, rcp.FINGER_OPEN_M))
        return q, g, tucked

    return one(ik), one(other_ik)


def hand_clear_of_stacks(ik, q: np.ndarray, tops: list[tuple[np.ndarray, float]]) -> bool:
    """Hand and fingertips of pose ``q`` outside every stack's keep-out, or well
    above it — evaluated in the planner's model, not the sim."""
    ik.set_q(q)
    pts = [ik.hand(), *rcp.finger_tips_from_data(ik.model, ik.data, ik.arm)]
    for pt in pts:
        for xy, top_z in tops:
            if float(np.linalg.norm(pt[:2] - xy)) < STACK_KEEPOUT_M and float(pt[2]) < top_z + 0.12:
                return False
    return True


def continue_from_here(robot, iks, trial: Trial, fps: int) -> bool:
    """Start this episode with both arms exactly where the last one left them,
    if that pose is clear of the new layout (the pile may have grown under the
    hand). Returns False if a fresh start is needed."""
    tops = [(np.array(xy), stack_top_z(sz)) for xy, sz in zip(trial.stack_xy, trial.sizes, strict=True)]
    tops.append((DROP_XY.copy(), stack_top_z(trial.pile_n)))
    for arm in rcp.ARMS:
        low = min(rcp.finger_lowest_z(robot, arm))
        for pt in (rcp.hand_pos_world(robot, arm), *rcp.finger_tips_world(robot, arm)):
            for xy, top_z in tops:
                if float(np.linalg.norm(pt[:2] - xy)) < STACK_KEEPOUT_M and low < top_z + 0.02:
                    print(f"  (cannot continue: {arm.side} hand over a stack at ({xy[0]:.2f},{xy[1]:.2f}), "
                          f"fingers {(low - top_z) * 100:+.1f} cm from its top — fresh start)")
                    return False
    for arm in rcp.ARMS:
        q = rcp._arm_q_real(robot, iks[arm.side])
        iks[arm.side].set_q(q)
        rcp._LAST_CMD[arm.side] = q.copy()
        rcp._OTHER_GRIP[arm.side] = rcp._finger_opening_m(robot, arm.side)
    ik = iks[trial.side]
    tip = ik.tip_mid()
    print(f"  {ik.arm.side} start CONTINUING from the last episode's end tip-mid=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})")
    rcp.settle_pose(robot, ik, rcp._finger_opening_m(robot, ik.arm.side), fps, hold_s=0.2)
    if others_disturbed(robot, trial, set()):
        print("  (cannot continue: settling disturbed a bar — fresh start)")
        trial.restore_bars()
        return False
    return True


def safe_start_pose(robot, iks, trial: Trial, rng: np.random.Generator, fps: int) -> bool:
    """Teleport both arms to random start poses, once. Candidates are drawn and
    checked in the planner's model first, so a hand that would start in or
    over a stack is never shown (the cube picker's random starts cover the far
    half of the table, exactly where the pads are; a hand teleported into a
    stack sends bars flying, one parked above a stack sweeps it on the idle
    arm's retreat). After a few rejected draws fall back to tucked starts,
    which are always clear."""
    import mujoco

    ik = iks[trial.side]
    other_ik = iks["left" if trial.side == "right" else "right"]
    tops = [(np.array(xy), stack_top_z(sz)) for xy, sz in zip(trial.stack_xy, trial.sizes, strict=True)]
    tops.append((DROP_XY.copy(), stack_top_z(trial.pile_n)))
    for attempt in range(12):
        prob = rcp.TUCKED_START_PROB if attempt < 8 else 1.0
        (q_a, g_a, tucked_a), (q_o, g_o, tucked_o) = draw_start_poses(ik, other_ik, rng, prob)
        if not (hand_clear_of_stacks(ik, q_a, tops) and hand_clear_of_stacks(other_ik, q_o, tops)):
            continue
        # apply, exactly as setup_start_pose does
        rcp._set_arm_qpos(robot, ik.arm.side, q_a)
        rcp._set_gripper_qpos(robot, ik.arm.side, g_a)
        rcp._set_arm_qpos(robot, other_ik.arm.side, q_o)
        rcp._set_gripper_qpos(robot, other_ik.arm.side, g_o)
        rcp.zero_sim_velocity(robot)
        mujoco.mj_forward(robot._model, robot._data)
        ik.set_q(q_a)
        rcp._LAST_CMD[ik.arm.side] = q_a.copy()
        rcp._LAST_CMD[other_ik.arm.side] = q_o.copy()
        rcp._OTHER_GRIP[other_ik.arm.side] = g_o
        tip = ik.tip_mid()
        print(f"  {ik.arm.side} start {'TUCKED (jittered)' if tucked_a else 'random'} "
              f"tip-mid=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}); "
              f"{other_ik.arm.side} starts {'TUCKED' if tucked_o else 'random'} (grip {g_o * 1000:.0f} mm)"
              + (f"  [{attempt} draw(s) rejected]" if attempt else ""))
        rcp.settle_pose(robot, ik, g_a, fps, hold_s=0.2)
        if others_disturbed(robot, trial, set()):
            print("  (start pose disturbed a bar — redrawing)")
            trial.restore_bars()
            continue
        return True
    trial.restore_bars()
    return False


def save_snapshot(robot, path: str) -> None:
    """Render the ego camera to a PNG (a look at the arranged scene)."""
    import mujoco
    from PIL import Image

    r = mujoco.Renderer(robot._model, 480, 640)
    r.update_scene(robot._data, camera="ego_camera")
    Image.fromarray(r.render()).save(path)
    r.close()
    print(f"  snapshot -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stacks", type=int, default=4,
                    help=f"number of pads/stacks on the arc (2-{MAX_STACKS}); even keeps the centreline clear")
    ap.add_argument("--arm-gain-scale", type=float, default=ARM_GAIN_SCALE,
                    help="servo stiffness multiplier (default 1.0 = the cube picker's gains, which the eval uses)")
    ap.add_argument("--model-path", default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"))
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--record", default=None, metavar="REPO_ID")
    ap.add_argument("--episodes", type=int, default=0,
                    help="With --record: keep going until this many SUCCESSFUL episodes are saved.")
    ap.add_argument("--cameras", choices=["chest", "all"], default="all")
    ap.add_argument("--snapshot", default=None, metavar="PNG", help="save the ego view of the first trial's scene")
    ap.add_argument("--debug", action="store_true",
                    help="verbose per-trial output; default is one 'episode N' line per saved episode")
    args = ap.parse_args()
    if not 2 <= args.stacks <= MAX_STACKS:
        ap.error(f"--stacks must be 2..{MAX_STACKS}")

    rng = np.random.default_rng(args.seed)
    rcp._AIM_OVERLAY_ENABLED = bool(args.debug)
    rcp.TUCK_TIP_TARGET = TUCK_TIP_TARGET  # before any tuck is solved (it is cached)
    rcp._APPROACH_GRIP_FN["fn"] = approach_grip
    rcp._APPROACH_SETTLE_S["s"] = APPROACH_SETTLE_S
    robot = rcp.make_robot(args.model_path, args.fps, viewer=not args.no_viewer,
                           cameras=args.cameras if args.record else "none",
                           arm_gain_scale=args.arm_gain_scale)
    hide_legacy_pads(robot)
    iks = {a.side: rcp.build_ik(robot, a) for a in rcp.ARMS}
    rcp.park_both_arms(robot, iks)
    rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)
    # the cube shares the scene: park it well clear of the bar warehouse
    rcp.set_cube_xy(robot, -0.90, -0.90)

    recorder = None
    if args.record:
        recorder = rcp.EpisodeRecorder(robot, args.record, args.fps, task="")
        rcp._RECORDER = recorder
        print(f"Recording to '{args.record}' (cameras={args.cameras}); "
              f"target {args.episodes or args.trials} successful episodes")

    if not args.debug:
        try:
            import datasets

            datasets.disable_progress_bars()
        except Exception:  # noqa: BLE001
            pass
        try:
            import av

            av.logging.set_level(av.logging.PANIC)
        except Exception:  # noqa: BLE001
            pass

    @contextlib.contextmanager
    def trial_output():
        # Quiet by default: swallow per-trial chatter, including C-level
        # encoder logs on stderr, unless --debug.
        if args.debug:
            yield
            return
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_err = os.dup(2)
        try:
            os.dup2(devnull, 2)
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            os.dup2(saved_err, 2)
            os.close(saved_err)
            os.close(devnull)

    target_eps = args.episodes if (args.record and args.episodes > 0) else 0
    max_trials = args.trials if not target_eps else max(args.trials, target_eps * 3)
    successes = 0
    t = 0
    per_side = {"left": [0, 0], "right": [0, 0]}
    fails: dict[str, int] = {}
    have_previous = False  # an episode has run, so "continue from here" has a here
    try:
        while t < max_trials:
            t += 1
            ok = False
            with trial_output():
                hdr = (f"episodes saved {successes}/{target_eps} (trial cap {max_trials})"
                       if target_eps else f"{t}/{max_trials}")
                print(f"\n=== Trial {t} — {hdr} ===")
                trial = None
                for _ in range(6):
                    cand = Trial(robot, rng, args.stacks)
                    if grasp_plannable(robot, iks[cand.side], cand, rng):
                        trial = cand
                        break
                    print("  (layout unplannable for the rule-chosen arm — resampling)")
                if trial is None:
                    print("  no plannable layout after 6 tries — skipping trial slot")
                    continue
                print(f"  {trial.describe()}")
                if args.snapshot and t == 1:
                    save_snapshot(robot, args.snapshot)
                cont = have_previous and rng.uniform() < CONTINUE_PROB and continue_from_here(robot, iks, trial, args.fps)
                if not cont and not safe_start_pose(robot, iks, trial, rng, args.fps):
                    print("  no clear start pose after several draws — skipping trial slot")
                    continue
                for a in rcp.ARMS:  # both arms retreat to their tuck when idle
                    rcp._RETREAT_TARGET[a.side] = rcp.tuck_q(iks[a.side])
                if recorder is not None:
                    recorder.task = trial.prompt
                    recorder.start()
                ok = run_trial(robot, iks, args.fps, trial, rng)
                have_previous = True
                per_side[trial.side][1] += 1
                if ok:
                    per_side[trial.side][0] += 1
                    if recorder is not None:
                        if recorder.save():
                            successes += 1
                            print(f"  episode {successes} saved")
                        else:
                            ok = False
                    else:
                        successes += 1
                    if ok:
                        print("  trial SUCCESS")
                else:
                    fails[trial.fail_stage or "?"] = fails.get(trial.fail_stage or "?", 0) + 1
                    if recorder is not None:
                        recorder.drop()
                    print(f"  trial FAIL ({trial.fail_stage})")
                # No park between episodes: the arms stay where they finished
                # and jump straight to the next randomised start (the snap to
                # a default pose and back looked like a glitch in the viewer).
            if ok and not args.debug:
                print(f"episode {successes}", flush=True)
            if target_eps and successes >= target_eps:
                break
    finally:
        if recorder is not None:
            recorder.finalize()
        print(f"\nDone: {successes}/{t} successful trials  "
              f"(left {per_side['left'][0]}/{per_side['left'][1]}, right {per_side['right'][0]}/{per_side['right'][1]})"
              + (f"  failures: {fails}" if fails else ""))
        robot.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        if rcp._RECORDER is not None:
            rcp._RECORDER.finalize()
        os._exit(0)
