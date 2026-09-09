#!/usr/bin/env python3
"""Caddy picking: 7 stacks of chocolate bars on an arc; "select N bars from stack K".

Each trial:
  1. Arrange 7 stacks (1-3 bars each) on an arc in front of the arms. Stack 1
     is the left-most, stack 7 the right-most; the slots are fixed, so a stack
     number always means the same place on the table (no placards needed).
     Bar colours are shuffled per trial so colour cannot stand in for number.
  2. Draw a request of one or more parts, e.g.
        "select 1 bar from stack 4"
        "select 2 bars from stack 1, 1 bar from stack 4 and 2 bars from stack 7"
     (a part never asks for more bars than its stack holds).
  3. Work through the parts in order. Each bar is picked from the top of its
     stack by the arm on that stack's side of the robot centre (left of centre
     -> left arm) and set down on a growing pile in the middle of the table.
  4. Success: every requested bar is on the pile, in order, and no other stack
     was disturbed (knocking a stack over is a failure).

Recording (--record) stores each successful trial as one episode whose task
string is the request.

Usage:
  MUJOCO_GL=egl python scripts/random_caddy_pick.py --trials 5 --debug
  MUJOCO_GL=egl python scripts/random_caddy_pick.py --no-viewer --parts-max 3 \\
      --record local/openarm_sim_caddy --episodes 300 --cameras chest
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
from chocolate_bars_sim import (  # noqa: E402
    BAR_HALF,
    FLAVORS,
    bar_pos,
    set_bar_color,
    set_bar_pose,
)

# Grasp geometry for a bar with the natural-motion (aim) machinery. The pad
# plates are ~7.2 cm tall centred on the "tip" point, so with the tips this far
# above the bar's centre the plate bottoms reach ~0.4 cm below it — holding the
# bar's top ~1.65 cm — while clearing whatever it rests on (the table, or the
# bar below in a stack) by BAR_PAD_CLEARANCE_M. Tight; hence the stiff servos.
BAR_TIP_ABOVE_M = 0.032
BAR_PAD_CLEARANCE_M = 0.008
# Bars leave ~1.2 cm under the pads; the default servo gains sag more than
# that at reach, so this scene runs the arms 3x stiffer (see make_robot).
ARM_GAIN_SCALE = 3.0

N_STACKS = 7
MAX_PER_STACK = 3
MAX_BARS = N_STACKS * MAX_PER_STACK  # 21 bar bodies in the scene
TABLE_TOP_Z = rcp.TABLE_TOP_Z

# The arc: centred on the robot base, stack 1 at the left end (+y). The radius
# puts the middle stack at the far edge of the table (table top ends at
# x = 0.555; a bar is 9 cm long) while staying inside reliable reach — at
# x >= 0.50 the extended arm's grip weakens and bars slip on the lift; the
# span keeps the end stacks inside the table's side edges (y = +-0.40).
ARC_RADIUS = 0.48
ARC_SPAN_DEG = 46.0          # stacks from +46 deg (left) to -46 deg (right)
ARC_RADIUS_JITTER = 0.015
ARC_ANGLE_JITTER_DEG = 1.5

# The pile in the middle of the table.
MIDDLE = np.array([0.31, 0.0])
MIDDLE_JITTER = 0.015
PLACE_TOL = 0.06             # bar counts as on the pile within this radius

WAREHOUSE = [(-0.55, -0.66 + 0.12 * i, 0.013) for i in range(MAX_BARS)]
PALETTE = FLAVORS + [("hazelnut", (0.62, 0.42, 0.22, 1))]


def plural(n: int) -> str:
    return "bar" if n == 1 else "bars"


def make_prompt(parts: list[tuple[int, int]]) -> str:
    """parts = [(stack_idx0, count), ...] -> 'select 2 bars from stack 1, 1 bar
    from stack 4 and 2 bars from stack 7'."""
    phrases = [f"{c} {plural(c)} from stack {s + 1}" for s, c in parts]
    if len(phrases) == 1:
        body = phrases[0]
    else:
        body = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return f"select {body}"


class Trial:
    """One arranged scene: 7 stacks on the arc, the request, the plan."""

    def __init__(self, robot, rng: np.random.Generator, parts_max: int, bars_max: int):
        import mujoco

        self.robot = robot
        self.sizes = [int(rng.integers(1, MAX_PER_STACK + 1)) for _ in range(N_STACKS)]

        # slots on the arc, left (+y) to right (-y), with a little jitter
        angles = np.deg2rad(np.linspace(ARC_SPAN_DEG, -ARC_SPAN_DEG, N_STACKS))
        self.stack_xy: list[tuple[float, float]] = []
        self.stack_yaw: list[float] = []
        for th in angles:
            th_j = float(th + np.deg2rad(rng.uniform(-ARC_ANGLE_JITTER_DEG, ARC_ANGLE_JITTER_DEG)))
            r_j = ARC_RADIUS + float(rng.uniform(-ARC_RADIUS_JITTER, ARC_RADIUS_JITTER))
            self.stack_xy.append((r_j * math.cos(th_j), r_j * math.sin(th_j)))
            self.stack_yaw.append(th_j)  # long axis points at the robot base

        colour_ids = rng.permutation(len(PALETTE))[:N_STACKS]
        self.flavors = [PALETTE[i] for i in colour_ids]

        for i in range(MAX_BARS):
            set_bar_pose(robot, i, *WAREHOUSE[i])
        self.stack_bars: list[list[int]] = []
        bar_i = 0
        for s_idx, (x, y) in enumerate(self.stack_xy):
            bars = []
            for level in range(self.sizes[s_idx]):
                z = TABLE_TOP_Z + BAR_HALF[2] * (2 * level + 1)
                set_bar_pose(robot, bar_i, x, y, z, yaw=self.stack_yaw[s_idx])
                set_bar_color(robot, bar_i, self.flavors[s_idx][1])
                bars.append(bar_i)
                bar_i += 1
            self.stack_bars.append(bars)
        rcp.zero_sim_velocity(robot)
        mujoco.mj_forward(robot._model, robot._data)
        self.initial_pos = {i: bar_pos(robot, i) for i in range(MAX_BARS)}

        # the request: 1..parts_max distinct stacks, ascending, counts within
        # each stack's height, total capped at bars_max
        n_parts = int(rng.integers(1, parts_max + 1))
        stacks = sorted(int(s) for s in rng.choice(N_STACKS, size=n_parts, replace=False))
        parts: list[tuple[int, int]] = []
        total = 0
        for s in stacks:
            c = int(rng.integers(1, self.sizes[s] + 1))
            c = min(c, bars_max - total)
            if c <= 0:
                break
            parts.append((s, c))
            total += c
        self.parts = parts
        self.prompt = make_prompt(parts)
        self.pile_xy = MIDDLE + rng.uniform(-MIDDLE_JITTER, MIDDLE_JITTER, size=2)
        self.fail_stage: str | None = None
        self.moved: list[int] = []  # bars already carried to the pile, in order

    def arm_for(self, stack_idx: int) -> str:
        """Side rule: left of the robot centre (y >= 0) -> left arm."""
        return "left" if self.stack_xy[stack_idx][1] >= 0.0 else "right"

    def top_bar(self, stack_idx: int) -> int:
        """Highest bar still IN the stack (a bar already carried to the pile
        would otherwise win — it sits higher than what's left)."""
        bars = [i for i in self.stack_bars[stack_idx] if i not in self.moved]
        return max(bars, key=lambda i: bar_pos(self.robot, i)[2])

    def picks(self) -> list[tuple[int, int]]:
        """(stack_idx, pick_number_within_part) in execution order."""
        out = []
        for s, c in self.parts:
            out += [(s, k) for k in range(c)]
        return out


def pile_tip_z(level: int) -> float:
    """Tip-mid height to set a bar down as pile level ``level`` (0 = table):
    the bar's resting centre plus the aim grasp's tip offset, plus slack."""
    centre = TABLE_TOP_Z + BAR_HALF[2] * (2 * level + 1)
    return centre + BAR_TIP_ABOVE_M + 0.006  # release a touch high; the bar drops the last bit


def bar_tilt_signed(robot, ik, bar: int) -> float:
    """Bar long-axis tilt (rad) from level, positive when the end away from the
    hand droops — the way a bar gripped off-centre hangs."""
    import mujoco

    bid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_BODY, f"bar_{bar}")
    u = robot._data.xmat[bid].reshape(3, 3)[:, 0]          # bar long axis
    a = robot._data.xmat[ik.body].reshape(3, 3)[:, 2]      # gripper approach axis
    if float(np.dot(u[:2], a[:2])) < 0.0:
        u = -u
    return math.atan2(-float(u[2]), float(np.hypot(u[0], u[1])))


def level_bar_in_hand(robot, ik, fps: int, bar: int, grip_m: float) -> None:
    """Pitch the wrist so the held bar is level before it is set down. Gripped
    a quarter of the way in, the far end hangs ~8 deg low; released like that
    it lands on one end and slides off the pile."""
    tilt = bar_tilt_signed(robot, ik, bar)
    if abs(tilt) < math.radians(2.0):
        return
    q0 = rcp._cmd_seed(robot, ik.arm.side)
    ik.set_q(q0)
    a = ik.rot()[:, 2]
    yaw = math.atan2(float(a[1]), float(a[0]))
    pitch = math.atan2(-float(a[2]), float(np.hypot(a[0], a[1])))
    tip = ik.tip_mid().copy()
    for _ in range(150):
        ik.step_tip_mid(tip, max_dq=math.radians(2.0), yaw=yaw, pitch=pitch - tilt, level=True)
    q1 = ik.q().copy()
    dq = float(np.max(np.abs(q1 - q0)))
    print(f"  level bar: tilt {math.degrees(tilt):+.0f}° -> pitching wrist by {math.degrees(-tilt):+.0f}°")
    rcp.play_joint_path(robot, ik, q0, q1, grip_m, fps, max(0.4, dq / math.radians(30.0)), "level bar",
                        abort_on_table=False)


def carry_and_place(
    robot, ik, fps: int, pile_xy: np.ndarray, level: int, grip_m: float, bar: int
) -> bool:
    """Transport the held bar to the pile and set it down on level ``level``.

    The grip is a quarter of the bar in from its end, so the fingertips are not
    over the bar's centre: aim the fingertips at pile + (tips - bar centre) so
    the BAR lands centred on the pile (the carry translates only)."""
    # carry high enough that the hanging bar (its bottom ~4.5 cm below the
    # tips) clears a pile that is already `level` bars tall
    transit_z = TABLE_TOP_Z + 0.18 + 2 * BAR_HALF[2] * level
    place_z = pile_tip_z(level)
    tip = rcp.tip_mid_world(robot, ik.arm)
    grip_offset = tip[:2] - bar_pos(robot, bar)[:2]
    target_xy = np.asarray(pile_xy, dtype=float) + grip_offset
    hop = np.array([tip[0], tip[1], transit_z])
    if rcp.play_tip_cartesian(robot, ik, hop, grip_m, fps, 0.0,
                              label="carry: rise", freeze_wrist=True) is None:
        return False
    over = np.array([target_xy[0], target_xy[1], transit_z])
    if rcp.play_tip_cartesian(robot, ik, over, grip_m, fps, 0.0,
                              label="carry: over pile", lock_z=transit_z,
                              freeze_wrist=True, min_z=transit_z - 0.01) is None:
        return False
    level_bar_in_hand(robot, ik, fps, bar, grip_m)
    # Lower closed-loop on the BAR's measured height (the tip-to-bar offset
    # varies with the grasp): descend at 6 cm/s until the bar is ~3 mm above its
    # resting height on the pile, keeping it centred and re-levelling on the way.
    rest_z = TABLE_TOP_Z + BAR_HALF[2] * (2 * level + 1)
    q_prev = rcp._cmd_seed(robot, ik.arm.side)
    ik.set_q(q_prev)
    wrist_hold = q_prev[4:7].copy()
    # Integrate the COMMANDED tip height (targets relative to the measured,
    # lagging tip never get ahead of the arm and the descent stalls).
    z_cmd = float(ik.tip_mid()[2])
    for k in range(int(5.0 * fps)):
        bz = float(bar_pos(robot, bar)[2])
        if bz <= rest_z + 0.003:
            break
        z_cmd -= min(0.06 / fps, max(0.0, bz - rest_z - 0.002))
        tip_t = ik.tip_mid().copy()
        tip_t[:2] = np.asarray(pile_xy, dtype=float) + (rcp.tip_mid_world(robot, ik.arm)[:2] - bar_pos(robot, bar)[:2])
        tip_t[2] = z_cmd
        for _ in range(3):
            ik.step_tip_mid(tip_t, max_dq=math.radians(1.6), freeze_wrist=wrist_hold)
        q = rcp._rate_limit_q(ik.q(), q_prev, math.radians(1.4))
        ik.set_q(q)
        rcp._hold_fingers(robot, ik, grip_m)
        rcp.precise_sleep(1.0 / fps)
        q_prev = q.copy()
        if k % fps == fps - 1 and abs(bar_tilt_signed(robot, ik, bar)) > math.radians(4.0):
            level_bar_in_hand(robot, ik, fps, bar, grip_m)
            q_prev = rcp._cmd_seed(robot, ik.arm.side)
            ik.set_q(q_prev)
            wrist_hold = q_prev[4:7].copy()
            z_cmd = float(ik.tip_mid()[2])
    level_bar_in_hand(robot, ik, fps, bar, grip_m)
    print(f"  before release: bar tilt {math.degrees(bar_tilt_signed(robot, ik, bar)):+.0f}°, "
          f"bar z-above-rest {(bar_pos(robot, bar)[2] - rest_z) * 100:+.1f} cm, "
          f"off-centre {np.linalg.norm(bar_pos(robot, bar)[:2] - pile_xy) * 100:.1f} cm")
    # Release, and do not move until the fingers are actually open: from the
    # squeeze to fully open takes ~0.7 s at the ramp rate, and retreating early
    # once flung a half-held bar off the table.
    if rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.5) < 0.0:
        print("  release: fingers still not open — waiting")
        rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.5)
    tip = rcp.tip_mid_world(robot, ik.arm)
    up = np.array([tip[0], tip[1], transit_z])
    rcp.play_tip_cartesian(robot, ik, up, rcp.FINGER_OPEN_M, fps, 0.0,
                           label="carry: retreat", freeze_wrist=True)
    return True


def stacks_disturbed(robot, trial: Trial, exclude: set[int]) -> list[str]:
    """Bars (outside ``exclude``) that strayed from their arranged pose."""
    bad = []
    for i in range(sum(trial.sizes)):
        if i in exclude:
            continue
        p0, p1 = trial.initial_pos[i], bar_pos(robot, i)
        dxy = float(np.linalg.norm(p1[:2] - p0[:2]))
        dz = abs(float(p1[2] - p0[2]))
        if dxy > 0.03 or dz > 0.012:
            bad.append(f"bar_{i} moved {dxy * 100:.1f}cm xy / {dz * 1000:.0f}mm z")
    return bad


BAR_OBSTACLE_RADIUS_M = 0.055  # keep fingertips this far from other bars' centres


def aim_target_for_bar(robot, trial: Trial, s_idx: int, level: int) -> None:
    """Point the aim machinery at the bar at ``level`` (0 = table) of stack
    ``s_idx``: tip offset, the surface under it, approach along the bar's
    length (radial), and every other bar as an obstacle."""
    x, y = trial.stack_xy[s_idx]
    support_z = TABLE_TOP_Z + 2 * BAR_HALF[2] * level
    own = set(trial.stack_bars[s_idx])
    # every other bar is an obstacle: the other stacks, and the bars already
    # on the pile (the pile sits on the approach line to the middle stacks,
    # which then have to be approached from above)
    obstacles = [
        (bar_pos(robot, i), BAR_OBSTACLE_RADIUS_M)
        for st, bars in enumerate(trial.stack_bars) for i in bars
        if st != s_idx or i in trial.moved
    ]
    rcp.set_aim_target(
        BAR_TIP_ABOVE_M, support_z, BAR_PAD_CLEARANCE_M,
        xy=np.array([x, y]), azimuth=trial.stack_yaw[s_idx], obstacles=obstacles,
        # Grip a quarter of the bar's length in from its near end (not the
        # middle): reaching to the middle put the hand over the next stack.
        in_extra_m=-0.5 * float(BAR_HALF[0]),
        # the plates only get around a 2.5 cm bar from a shallow approach
        max_pitch_deg=40.0,
        # a thin bar needs less squeeze than the cube; 18 mm crushed bars into
        # each other hard enough to overflow MuJoCo's contact buffer
        squeeze_m=0.013,
    )


def grasp_plannable(robot, ik, trial: Trial, s_idx: int, level: int, rng: np.random.Generator) -> bool:
    """Can the natural-motion planner reach this bar from the idle pose?"""
    aim_target_for_bar(robot, trial, s_idx, level)
    x, y = trial.stack_xy[s_idx]
    centre = np.array([x, y, TABLE_TOP_Z + BAR_HALF[2] * (2 * level + 1)])
    q_idle = np.deg2rad(ik.arm.idle_deg)
    planned = rcp.plan_aim_at_cube(ik, q_idle, centre, rng)
    if planned is None:
        return False
    return rcp.plan_via_lift(ik, q_idle, planned[0]) is not None


def retreat_to_tuck(robot, ik, fps: int, trial: Trial) -> bool:
    """Hand-over: take the finishing arm to its tuck along a path checked
    against the table and every bar (stacks and pile). Left to the idle-arm
    drift, a straight joint interpolation from above the pile to the tuck
    swept the forearm through the pile and neighbouring stacks."""
    q_now = rcp._cmd_seed(robot, ik.arm.side)
    q_tuck = np.deg2rad(ik.arm.tuck_deg)
    obstacles = [(bar_pos(robot, i), BAR_OBSTACLE_RADIUS_M) for bars in trial.stack_bars for i in bars]
    rcp.set_aim_target(rcp.AIM_TIP_ABOVE_CUBE_M, TABLE_TOP_Z, obstacles=obstacles)
    ik.set_q(q_now)
    waypoints = rcp.plan_via_lift(ik, q_now, q_tuck)
    if waypoints is None:
        print(f"  hand-over: no clear path for the {ik.arm.side} arm to its tuck — leaving it raised")
        return False
    q_prev = q_now
    for q_wp in waypoints:
        dq = float(np.max(np.abs(q_wp - q_prev)))
        rcp.play_joint_path(robot, ik, q_prev, q_wp, 0.0, fps, max(0.8, dq / math.radians(40.0)),
                            f"hand-over: {ik.arm.side} arm to tuck", abort_on_table=False)
        q_prev = q_wp
    return True


def run_trial(robot, iks, fps: int, trial: Trial, rng: np.random.Generator) -> bool:
    picks = trial.picks()
    print(f"  request: \"{trial.prompt}\"  sizes={trial.sizes}  picks={len(picks)}")
    moved = trial.moved
    prev_side: str | None = None

    def knocked(current: int | None = None) -> bool:
        bad = stacks_disturbed(robot, trial, set(moved) | ({current} if current is not None else set()))
        if bad:
            print(f"  fail: KNOCKED OVER stack(s): {'; '.join(bad)}")
            trial.fail_stage = "knock-over"
        return bool(bad)

    for n, (s_idx, _) in enumerate(picks):
        side = trial.arm_for(s_idx)
        ik = iks[side]
        if prev_side is not None and prev_side != side:
            retreat_to_tuck(robot, iks[prev_side], fps, trial)
        prev_side = side
        bar = trial.top_bar(s_idx)
        level = trial.stack_bars[s_idx].index(bar)
        rcp.set_target_body(f"bar_{bar}")
        aim_target_for_bar(robot, trial, s_idx, level)
        try:
            bp = bar_pos(robot, bar)
            rcp._AIM_OVERLAY["cube"] = bp.copy()  # green line to this bar (drawn only with --debug)
            print(f"  pick {n + 1}/{len(picks)}: stack {s_idx + 1} bar_{bar} (level {level}) with {side} arm "
                  f"at ({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.3f}) -> pile level {len(moved)}")
            # natural motion: aim at the bar, close in along the aim line, squeeze, lift
            rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.0)
            q_now = rcp._cmd_seed(robot, side)
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
                knocked(current=bar)
                trial.fail_stage = trial.fail_stage or "approach"
                return False
            if not rcp.grasp_and_lift(robot, ik, fps, rng, hold_s=0.0):  # straight into the carry
                knocked(current=bar)
                trial.fail_stage = trial.fail_stage or "grasp"
                return False
            hold = rcp._AIM_LAST_HOLD["m"]
            import mujoco as _mj
            _bid = _mj.mj_name2id(robot._model, _mj.mjtObj.mjOBJ_BODY, f"bar_{bar}")
            _R = robot._data.xmat[_bid].reshape(3, 3)
            print(f"  in hand: bar tilt {math.degrees(math.acos(min(1.0, abs(float(_R[2, 2]))))):.0f}° from level, "
                  f"grip offset {np.linalg.norm(rcp.tip_mid_world(robot, ik.arm)[:2] - bar_pos(robot, bar)[:2]) * 100:.1f} cm")
            if not carry_and_place(robot, ik, fps, trial.pile_xy, len(moved), hold, bar):
                print("  fail: carry/place")
                trial.fail_stage = "carry"
                return False
            moved.append(bar)
            # pile integrity after this pick: every earlier bar still where it was set down
            for lvl, b in enumerate(moved[:-1]):
                pb = bar_pos(robot, b)
                print(f"    pile check level {lvl}: bar_{b} {np.linalg.norm(pb[:2] - trial.pile_xy) * 100:.1f} cm off, z={pb[2]:.3f}")
            p_now = bar_pos(robot, bar)
            print(f"  placed bar_{bar}: {np.linalg.norm(p_now[:2] - trial.pile_xy) * 100:.1f} cm from pile centre, "
                  f"z={p_now[2]:.3f} (level {len(moved) - 1} expects {TABLE_TOP_Z + BAR_HALF[2] * (2 * len(moved) - 1):.3f})")
            if knocked():
                return False
        finally:
            rcp.set_target_body("cube")
            rcp.set_aim_target(rcp.AIM_TIP_ABOVE_CUBE_M, TABLE_TOP_Z)  # back to cube defaults
            rcp._AIM_OVERLAY["cube"] = None
            rcp._AIM_OVERLAY["goal"] = None

    # settle, then verify the pile: every moved bar near the middle, stacked in order
    ik = iks[trial.arm_for(picks[-1][0])]
    for _ in range(int(0.4 * fps)):
        rcp.send_q(robot, ik, rcp.FINGER_OPEN_M)
        rcp.precise_sleep(1.0 / fps)
    ok = True
    for level, bar in enumerate(moved):
        p = bar_pos(robot, bar)
        d = float(np.linalg.norm(p[:2] - trial.pile_xy))
        z_expect = TABLE_TOP_Z + BAR_HALF[2] * (2 * level + 1)
        stacked = abs(float(p[2]) - z_expect) < 0.012
        good = d < PLACE_TOL and stacked
        print(f"  pile level {level}: bar_{bar} dist={d * 100:.1f}cm z={p[2]:.3f} "
              f"(expect {z_expect:.3f}) {'OK' if good else 'MISS'}")
        ok &= good
    if knocked():
        ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts-max", type=int, default=3,
                    help="max number of 'N bars from stack K' parts per request (1 = simple form only)")
    ap.add_argument("--bars-max", type=int, default=5,
                    help="max total bars per request")
    ap.add_argument("--model-path",
                    default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"))
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--record", default=None, metavar="REPO_ID")
    ap.add_argument("--episodes", type=int, default=0,
                    help="With --record: keep going until this many SUCCESSFUL episodes are saved.")
    ap.add_argument("--cameras", choices=["chest", "all"], default="chest")
    ap.add_argument("--debug", action="store_true",
                    help="verbose per-trial output; default is one 'episode N' line per saved episode")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rcp._AIM_OVERLAY_ENABLED = bool(args.debug)
    # a 3.5 cm bar between fingers 10 cm apart: judge the straddle loosely
    rcp._STRADDLE_TOL["xy"] = 0.04
    rcp._STRADDLE_TOL["imbalance"] = 0.045
    robot = rcp.make_robot(args.model_path, args.fps, viewer=not args.no_viewer,
                           cameras=args.cameras if args.record else "none",
                           arm_gain_scale=ARM_GAIN_SCALE)
    iks = {a.side: rcp.build_ik(robot, a) for a in rcp.ARMS}
    rcp.park_both_arms(robot, iks)
    rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)
    rcp.set_cube_xy(robot, -0.55, 0.55)  # the cube shares the table otherwise

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
        # Quiet by default (see random_cube_pick): swallow per-trial chatter,
        # including C-level encoder logs on stderr, unless --debug.
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
                    cand = Trial(robot, rng, args.parts_max, args.bars_max)
                    # every requested pick must be plannable by the rule-chosen arm
                    plannable = True
                    for s_idx, c in cand.parts:
                        x, y = cand.stack_xy[s_idx]
                        for k in range(c):
                            level = cand.sizes[s_idx] - 1 - k
                            if not grasp_plannable(robot, iks[cand.arm_for(s_idx)], cand, s_idx, level, rng):
                                plannable = False
                    if plannable:
                        trial = cand
                        break
                    print("  (layout unplannable for the rule-chosen arms — resampling)")
                if trial is None:
                    print("  no plannable layout after 6 tries — skipping trial slot")
                    continue
                first_side = trial.arm_for(trial.parts[0][0])
                rcp.settle_pose(robot, iks[first_side], 0.0, args.fps, hold_s=0.3)
                rcp.setup_start_pose(robot, iks[first_side], rng, args.fps)
                # both arms retreat to their tuck when idle (an arm may hand over mid-request)
                for a in rcp.ARMS:
                    rcp._RETREAT_TARGET[a.side] = np.deg2rad(a.tuck_deg)
                if recorder is not None:
                    recorder.task = trial.prompt
                    recorder.start()
                ok = run_trial(robot, iks, args.fps, trial, rng)
                if ok:
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
                    if recorder is not None:
                        recorder.drop()
                    print("  trial FAIL")
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks[first_side], 0.0, args.fps, hold_s=0.15)
            if ok and not args.debug:
                print(f"episode {successes}", flush=True)
            if target_eps and successes >= target_eps:
                break
    finally:
        if recorder is not None:
            recorder.finalize()
        print(f"\nDone: {successes}/{t} successful trials")
        robot.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        if rcp._RECORDER is not None:
            rcp._RECORDER.finalize()
        os._exit(0)
