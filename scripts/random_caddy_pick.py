#!/usr/bin/env python3
"""Caddy picking: stacks of brown chocolate bars on coloured pads; "get bar from blue pad".

Each trial:
  1. Arrange N stacks (default 4, 1-3 bars each) of identical brown square bars
     (50 x 50 x 25 mm) on an arc in front of the arms, each stack on a coloured
     pad. The pad colours are a fresh random draw from the palette every trial,
     so a colour says nothing about position and position nothing about colour:
     the policy has to find the pad the prompt names.
  2. The prompt names one pad: "get bar from blue pad".
  3. The arm is the one on the pad's side of the table — left arm for pads left
     of the robot, right arm for pads right of it. N is even by default so no
     pad sits on the centreline where that rule is ambiguous. (An odd N puts a
     slot there; it always goes to the LEFT arm.)
  4. The top bar of that stack is picked with the natural-motion machinery of
     the cube picker (aim, close in along the aim line, squeeze, lift; retries
     after a miss) and set down on the table directly in front of its pad,
     toward the robot.
  5. Success: exactly that bar rests in front of its pad and no other bar moved.

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
BAR_LEVEL_SHRINK = 0.04  # each bar up a stack is 4% smaller in footprint than the one below (see Trial)

# Grasp geometry for the aim machinery (same planner/executor as the cube
# picker; only the target description changes). The finger plates are ~7.2 cm
# tall centred on the "tip" point, so with the tips this far above the bar's
# centre the plate bottoms sit ~13 mm above whatever the bar rests on (the
# table, or the bar below it) and hold the bar's top ~12 mm. The arm sags
# under load; measured 2.5 cm at full reach with the cube picker's servo gains,
# which put the plates onto the bar BELOW the target (one-sided pinch, no
# lift). A 2.5 cm bar has no room for that, so this scene runs the position
# servos 3x stiffer (~0.5 cm sag; see make_robot). The eval must use the same
# --arm-gain-scale.
BAR_TIP_ABOVE_M = 0.038
BAR_PAD_CLEARANCE_M = 0.004
ARM_GAIN_SCALE = 3.0

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

# Where the bar goes: straight in front of its pad, toward the robot. Pad half
# 4.5 cm + bar half 2.5 cm + a 3 cm gap.
PLACE_IN_FRONT_M = 0.10
PLACE_TOL = 0.05  # bar counts as delivered within this radius of the spot

PAD_HALF_XY = 0.045
PAD_Z = TABLE_TOP_Z + 0.0015  # 3 mm slab, visual only
PAD_PARK = (-0.9, 0.0, -0.5)  # unused pads hide under the floor
WAREHOUSE = [(-0.55, -0.66 + 0.12 * i, 0.0125) for i in range(MAX_BARS)]

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
    return TABLE_TOP_Z + BAR_HALF[2] * (2 * level + 1)


# --- one trial ---------------------------------------------------------------


class Trial:
    """One arranged scene: N stacks on coloured pads, the prompt, the plan."""

    def __init__(self, robot, rng: np.random.Generator, n_stacks: int):
        import mujoco

        self.robot = robot
        self.n = n_stacks
        self.sizes = [int(rng.integers(1, MAX_PER_STACK + 1)) for _ in range(self.n)]

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
            m.geom_size[gid] = BAR_HALF
            m.geom_rgba[gid] = BAR_RGBA
        self.stack_bars: list[list[int]] = []
        bar_i = 0
        for s_idx, (x, y) in enumerate(self.stack_xy):
            bars = []
            for level in range(self.sizes[s_idx]):
                # Two near-identical square boxes face-to-face overflow MuJoCo's
                # box-box contact buffer (9 contacts, fatal): each bar up a stack
                # is a few percent smaller in footprint than the one below, so the
                # contact patch is always the upper bar's four corners.
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"bar_{bar_i}")
                shrink = 1.0 - BAR_LEVEL_SHRINK * level + float(rng.uniform(-0.005, 0.005))
                m.geom_size[gid] = BAR_HALF * np.array([shrink, shrink, 1.0 + float(rng.uniform(-0.01, 0.01))])
                jx, jy = rng.uniform(-0.0015, 0.0015, size=2)
                jyaw = float(rng.uniform(-0.05, 0.05))
                set_bar_pose(robot, bar_i, x + jx, y + jy, bar_centre_z(level),
                             yaw=self.stack_yaw[s_idx] + jyaw)
                bars.append(bar_i)
                bar_i += 1
            self.stack_bars.append(bars)
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
        th = self.stack_yaw[self.target]
        x, y = self.stack_xy[self.target]
        self.place_xy = np.array([x - PLACE_IN_FRONT_M * math.cos(th), y - PLACE_IN_FRONT_M * math.sin(th)])
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
        return int(round((float(bar_pos(self.robot, bar)[2]) - TABLE_TOP_Z - BAR_HALF[2]) / (2 * BAR_HALF[2])))

    def describe(self) -> str:
        cols = " ".join(f"{c[0]}:{s}" for c, s in zip(self.colours, self.sizes, strict=True))
        return (f"\"{self.prompt}\"  slot {self.target + 1}/{self.n} y={self.stack_xy[self.target][1]:+.2f} "
                f"-> {self.side} arm   [{cols}]")


BAR_OBSTACLE_RADIUS_M = 0.055  # keep fingertips this far from other bars' centres


def aim_target_for_bar(robot, trial: Trial, bar: int) -> None:
    """Point the aim machinery at ``bar`` where it is NOW: tip offset, the
    surface under it, approach along the slot's radial, every other bar an
    obstacle. Re-evaluated before each attempt, so a retry after a bar tumbled
    off its stack plans for where it landed."""
    level = max(0, trial.level_of(bar))
    support_z = TABLE_TOP_Z + 2 * BAR_HALF[2] * level
    obstacles = [
        (bar_pos(robot, i), BAR_OBSTACLE_RADIUS_M)
        for bars in trial.stack_bars for i in bars if i != bar
    ]
    rcp.set_aim_target(
        BAR_TIP_ABOVE_M, support_z, BAR_PAD_CLEARANCE_M,
        xy=bar_pos(robot, bar)[:2], azimuth=trial.stack_yaw[trial.target], obstacles=obstacles,
        # in_extra None: half a cube (= half a bar) so the pads centre on it.
        # squeeze None: the cube's squeeze; the bar is the cube's width.
    )


def grasp_plannable(robot, ik, trial: Trial, rng: np.random.Generator) -> bool:
    """Can the natural-motion planner reach the target's top bar from idle?"""
    bar = trial.top_bar()
    aim_target_for_bar(robot, trial, bar)
    q_idle = np.deg2rad(ik.arm.idle_deg)
    planned = rcp.plan_aim_at_cube(ik, q_idle, bar_pos(robot, bar), rng)
    if planned is None:
        return False
    return rcp.plan_via_lift(ik, q_idle, planned[0]) is not None


def others_disturbed(robot, trial: Trial, exclude: set[int]) -> list[str]:
    """Bars (outside ``exclude``) that strayed from their arranged pose."""
    bad = []
    for bars in trial.stack_bars:
        for i in bars:
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
        aim_target_for_bar(robot, trial, bar)
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
        if not rcp.grasp_and_lift(robot, ik, fps, rng, hold_s=0.0):  # straight into the carry
            trial.fail_stage = "grasp"
            continue
        trial.fail_stage = None
        return True
    print("  fail: out of attempts")
    return False


def carry_and_place(robot, ik, fps: int, target_xy: np.ndarray, grip_m: float, bar: int) -> bool:
    """Carry the held bar to the spot in front of its pad and set it down on the
    table: rise, translate at transit height, lower with the gripper's
    orientation held, release, retreat. The bar hangs wherever the pads caught
    it, so the tips are aimed at target + (tips - bar) and the BAR lands on the
    spot."""
    transit_z = TABLE_TOP_Z + 0.18
    arm = ik.arm

    def lost(stage: str) -> bool:
        d = float(np.linalg.norm(bar_pos(robot, bar) - rcp.tip_mid_world(robot, arm)))
        print(f"    [{stage}] bar-to-tips {d * 100:.1f} cm{'  <-- LOST' if d > 0.06 else ''}")
        return d > 0.06

    tip = rcp.tip_mid_world(robot, arm)
    hop = np.array([tip[0], tip[1], transit_z])
    if rcp.play_tip_cartesian(robot, ik, hop, grip_m, fps, 0.0, label="carry: rise", freeze_wrist=True) is None:
        return False
    if lost("after rise"):
        return False
    tip = rcp.tip_mid_world(robot, arm)
    offset = tip[:2] - bar_pos(robot, bar)[:2]
    over = np.array([target_xy[0] + offset[0], target_xy[1] + offset[1], transit_z])
    if rcp.play_tip_cartesian(robot, ik, over, grip_m, fps, 0.0, label="carry: over spot", lock_z=transit_z,
                              freeze_wrist=True, min_z=transit_z - 0.01, speed_mps=0.18) is None:
        return False
    if lost("over spot"):
        return False
    # lower with the gripper orientation held (freezing wrist angles re-pitches
    # the gripper as the shoulder/elbow descend)
    q_prev = rcp._cmd_seed(robot, arm.side)
    ik.set_q(q_prev)
    a = ik.rot()[:, 2]
    yaw_hold = math.atan2(float(a[1]), float(a[0]))
    pitch_hold = math.atan2(-float(a[2]), float(np.hypot(a[0], a[1])))
    rest_z = bar_centre_z(0)
    tip = rcp.tip_mid_world(robot, arm)
    bar_now = bar_pos(robot, bar)
    tip0 = ik.tip_mid().copy()
    tip_end = np.array([target_xy[0] + (tip[0] - bar_now[0]), target_xy[1] + (tip[1] - bar_now[1]),
                        rest_z + float(tip[2] - bar_now[2]) + 0.004])
    dist = float(np.linalg.norm(tip_end - tip0))
    n = max(2, int(max(0.4, dist / 0.12) * fps))
    print(f"  carry: lower onto the table ({n / fps:.1f}s, {dist * 100:.0f} cm)…")
    for k in range(n):
        u = (k + 1) / n
        s_u = u * u * (3.0 - 2.0 * u)
        tip_t = (1.0 - s_u) * tip0 + s_u * tip_end
        for _ in range(2):
            ik.step_tip_mid(tip_t, max_dq=math.radians(1.6), yaw=yaw_hold, pitch=pitch_hold, level=True)
        q = rcp._rate_limit_q(ik.q(), q_prev, math.radians(1.4))
        ik.set_q(q)
        rcp._hold_fingers(robot, ik, grip_m)
        rcp.precise_sleep(1.0 / fps)
        q_prev = q.copy()
        if float(bar_pos(robot, bar)[2]) <= rest_z + 0.002:
            break  # touched down
    print(f"  before release: bar {np.linalg.norm(bar_pos(robot, bar)[:2] - target_xy) * 100:.1f} cm off the spot, "
          f"z-above-rest {(bar_pos(robot, bar)[2] - rest_z) * 100:+.1f} cm")
    # Release, and do not move until the fingers are actually open.
    if rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.5) < 0.0:
        rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.5)
    tip = rcp.tip_mid_world(robot, arm)
    up = np.array([tip[0], tip[1], transit_z])
    rcp.play_tip_cartesian(robot, ik, up, rcp.FINGER_OPEN_M, fps, 0.0, label="carry: retreat", freeze_wrist=True)
    return True


def run_trial(robot, iks, fps: int, trial: Trial, rng: np.random.Generator) -> bool:
    ik = iks[trial.side]
    bar = trial.top_bar()
    trial.moved = bar
    rcp.set_target_body(f"bar_{bar}")
    try:
        bp = bar_pos(robot, bar)
        print(f"  pick: bar_{bar} (level {trial.level_of(bar)} of {trial.sizes[trial.target]}) "
              f"at ({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.3f}) with the {trial.side} arm "
              f"-> spot ({trial.place_xy[0]:.2f},{trial.place_xy[1]:.2f})")
        rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.0)
        if not pick_bar(robot, ik, fps, trial, bar, rng):
            bad = others_disturbed(robot, trial, {bar})
            if bad:
                print(f"  fail: KNOCKED: {'; '.join(bad)}")
                trial.fail_stage = "knock-over"
            return False
        if not carry_and_place(robot, ik, fps, trial.place_xy, rcp._AIM_LAST_HOLD["m"], bar):
            print("  fail: carry/place")
            trial.fail_stage = "carry"
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
    on_table = abs(float(p[2]) - bar_centre_z(0)) < 0.010
    ok = d < PLACE_TOL and on_table
    print(f"  placed bar_{bar}: {d * 100:.1f} cm from the spot, z={p[2]:.3f} -> {'OK' if ok else 'MISS'}")
    if not ok:
        trial.fail_stage = "place"
    bad = others_disturbed(robot, trial, {bar})
    if bad:
        print(f"  fail: KNOCKED: {'; '.join(bad)}")
        trial.fail_stage = "knock-over"
        ok = False
    return ok


STACK_KEEPOUT_M = 0.10  # a starting hand stays this far (xy) from every stack, unless well above it


def safe_start_pose(robot, iks, trial: Trial, rng: np.random.Generator, fps: int) -> bool:
    """The cube picker's start randomisation, re-drawn until neither arm starts
    in or over a stack. Its random starts cover the far half of the table —
    exactly where the pads are — and a hand teleported into a stack sends bars
    flying before the episode begins; a hand parked just above one sweeps it
    on the idle arm's retreat. After a few rejected draws fall back to tucked
    starts, which are always clear."""
    ik = iks[trial.side]
    tops = [(np.array(xy), bar_centre_z(sz - 1) + BAR_HALF[2]) for xy, sz in zip(trial.stack_xy, trial.sizes, strict=True)]
    prob0 = rcp.TUCKED_START_PROB
    try:
        for attempt in range(8):
            if attempt >= 5:
                rcp.TUCKED_START_PROB = 1.0
            trial.restore_bars()
            rcp.set_target_body(f"bar_{trial.top_bar()}")  # start-pose checks look at the target
            rcp.settle_pose(robot, ik, 0.0, fps, hold_s=0.3)
            rcp.setup_start_pose(robot, ik, rng, fps)
            rcp.set_target_body("cube")
            clear = True
            for arm in rcp.ARMS:
                hand = rcp.hand_pos_world(robot, arm)
                tips = rcp.finger_tips_world(robot, arm)
                for pt in (hand, *tips):
                    for xy, top_z in tops:
                        if float(np.linalg.norm(pt[:2] - xy)) < STACK_KEEPOUT_M and float(pt[2]) < top_z + 0.12:
                            clear = False
            if clear and not others_disturbed(robot, trial, set()):
                return True
            print("  (start pose in or over a stack — redrawing)")
        return False
    finally:
        rcp.TUCKED_START_PROB = prob0
        trial.restore_bars()


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
                    help="servo stiffness multiplier (1.0 = the cube picker's gains); pass the same value to the eval")
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
                if not safe_start_pose(robot, iks, trial, rng, args.fps):
                    print("  no clear start pose after several draws — skipping trial slot")
                    rcp.park_both_arms(robot, iks)
                    continue
                for a in rcp.ARMS:  # both arms retreat to their tuck when idle
                    rcp._RETREAT_TARGET[a.side] = rcp.tuck_q(iks[a.side])
                if recorder is not None:
                    recorder.task = trial.prompt
                    recorder.start()
                ok = run_trial(robot, iks, args.fps, trial, rng)
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
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks[trial.side], 0.0, args.fps, hold_s=0.15)
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
