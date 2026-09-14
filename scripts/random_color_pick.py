#!/usr/bin/env python3
"""Colour sorting: a red or green cube, picked with the matching arm, put on the matching pad.

Each trial:
  1. Colour the cube red or green (50/50). Red means the LEFT arm and the red
     grey pad on the left; green means the RIGHT arm and the grey pad on the
     right. Pads are neutral grey — the CUBE's colour is the only cue. The colour is the only cue — cubes spawn across their
     arm's whole reach, including the shared middle band, so a green cube can
     sit left of centre and still calls for the right arm (cross-body). That
     is what stops the policy from learning "side -> arm" and ignoring colour.
  2. Natural-motion pick (aim, close in, squeeze, lift) with the recovery data
     of random_cube_pick --motion aim: retries after a miss, a share of episodes
     disturbed (cube nudged mid-approach, or dropped after the lift and
     re-picked).
  3. Carry the cube to its pad and set it down; if it misses the pad it is
     picked up and placed again.
  4. Success: the cube rests on the matching pad.

Recording (--record) stores each successful trial as one episode. The task
string is the same for every episode ("put the cube on the pad of its
colour") so the colour must be read from the image, not the prompt.

Usage:
  MUJOCO_GL=egl python scripts/random_color_pick.py --trials 5 --debug
  MUJOCO_GL=egl python scripts/random_color_pick.py --no-viewer \\
      --record local/openarm_color_sort_chest_300 --episodes 300 --cameras chest --seed 1
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

TABLE_TOP_Z = rcp.TABLE_TOP_Z
TASK = "put the cube on the pad of its colour"

# colour -> (rgba, arm, pad centre xy). Pads are visual-only geoms in the scene
# (pad_left / pad_right), matching these positions.
COLOURS = {
    "red": ((0.85, 0.20, 0.15, 1.0), "left", np.array([0.20, 0.30])),
    "green": ((0.15, 0.65, 0.20, 1.0), "right", np.array([0.20, -0.30])),
}
PAD_TOL = 0.06          # cube counts as on the pad within this of its centre
PAD_KEEPOUT = 0.11      # cubes never spawn this close to a pad
CARRY_TRANSIT_Z = TABLE_TOP_Z + 0.18
PLACE_MAX_ATTEMPTS = 3


def show_pads(robot, visible: bool = True) -> None:
    """The pads are invisible in the scene by default: pad_left is exactly the
    cube's red, so leaving them on made every 'pick up the red cube' task show
    two red objects and wrecked policies that had never seen one."""
    import mujoco

    for name in ("pad_left", "pad_right"):
        gid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        robot._model.geom_rgba[gid][3] = 1.0 if visible else 0.0


_COLOUR_BAG: list[str] = []


def next_colour(rng: np.random.Generator) -> str:
    """Draw red/green from a shuffled bag, so the two stay balanced.

    A plain coin flip shares the random stream with placement retries and IK
    seeding, so for a given seed it can land 9-1 — and the ratio changes
    whenever any unrelated code path changes how many draws it makes. A bag
    guarantees an even split whatever else consumes the stream.
    """
    if not _COLOUR_BAG:
        _COLOUR_BAG.extend(["red", "green"])
        rng.shuffle(_COLOUR_BAG)
    return _COLOUR_BAG.pop()


def set_cube_colour(robot, rgba) -> None:
    import mujoco

    m = robot._model
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "cube_mat")
    m.mat_rgba[mid] = rgba


CROSS_BODY_PROB = 0.4     # how often to try the far side of the table
CROSS_BODY_REACH_M = 0.15  # how far across the centreline an arm can actually fetch


def place_cube_for(robot, iks, rng: np.random.Generator, colour: str, max_tries: int = 40):
    """Drop the cube anywhere its colour's arm can actually plan a pick.

    Reachability is tested with the AIM planner (the machinery that does the
    pick), not the legacy grasp pose: the arm reaches noticeably further
    across the table than that pose suggests. Cross-body placements — a green
    cube on the LEFT, fetched by the right arm reaching over — are
    deliberately over-sampled, so which side a cube sits on says nothing about
    which arm will take it and the colour is the only usable cue.
    """
    _, side, _ = COLOURS[colour]
    ik = iks[side]
    own = (0.0, 0.32) if side == "left" else (-0.32, 0.0)
    # Only the strip just across the centreline is reachable cross-body — the
    # far half as a whole is not, so sampling it uniformly wasted nearly every
    # draw and cross-body picks ended up at 7-13% instead of the intended 40%.
    far = (-CROSS_BODY_REACH_M, 0.0) if side == "left" else (0.0, CROSS_BODY_REACH_M)
    for attempt in range(max_tries):
        # early attempts honour the cross-body draw; later ones fall back to
        # the arm's own side so a trial is never lost to an unreachable draw
        band = far if (rng.uniform() < CROSS_BODY_PROB and attempt < 3 * max_tries // 4) else own
        x = float(rng.uniform(*rcp.CUBE_X_RANGE))
        y = float(rng.uniform(*band))
        if any(np.hypot(x - pad[0], y - pad[1]) < PAD_KEEPOUT for _, _, pad in COLOURS.values()):
            continue
        yaw = float(rng.uniform(-0.6, 0.6))
        cube0 = rcp.set_cube_xy(robot, x, y, yaw=yaw)
        if rcp.plan_aim_at_cube(ik, np.deg2rad(ik.arm.idle_deg), rcp.cube_pos(robot), rng) is None:
            continue
        cross = (y > 0) != (side == "left")
        print(f"  {colour} cube @ xy=({x:.3f}, {y:.3f}) yaw={math.degrees(yaw):.0f}° -> {side} arm"
              f"{' (CROSS-BODY)' if cross else ''}, {side} pad")
        return cube0
    raise RuntimeError(f"could not place a {colour} cube reachable by the {side} arm")


def carry_to_pad(robot, ik, fps: int, pad_xy: np.ndarray, grip_m: float) -> bool:
    """Carry the held cube to the pad, holding the gripper's orientation, set it
    down, release (waiting for the fingers to actually open), retreat."""
    arm = ik.arm
    tip = rcp.tip_mid_world(robot, arm)
    hop = np.array([tip[0], tip[1], CARRY_TRANSIT_Z])
    if rcp.play_tip_cartesian(robot, ik, hop, grip_m, fps, 0.0,
                              label="carry: rise", freeze_wrist=True) is None:
        return False
    tip = rcp.tip_mid_world(robot, arm)
    offset = tip[:2] - rcp.cube_pos(robot)[:2]  # place by the CUBE's centre
    over = np.array([pad_xy[0] + offset[0], pad_xy[1] + offset[1], CARRY_TRANSIT_Z])
    if rcp.play_tip_cartesian(robot, ik, over, grip_m, fps, 0.0,
                              label="carry: over pad", lock_z=CARRY_TRANSIT_Z,
                              freeze_wrist=True, min_z=CARRY_TRANSIT_Z - 0.01,
                              speed_mps=0.18) is None:
        return False
    # lower with the gripper orientation held (freezing wrist angles re-pitches
    # the gripper as the shoulder/elbow descend)
    q_prev = rcp._cmd_seed(robot, arm.side)
    ik.set_q(q_prev)
    a = ik.rot()[:, 2]
    yaw_hold = math.atan2(float(a[1]), float(a[0]))
    pitch_hold = math.atan2(-float(a[2]), float(np.hypot(a[0], a[1])))
    tip = rcp.tip_mid_world(robot, arm)
    cube = rcp.cube_pos(robot)
    place_tip_z = rcp.CUBE_Z + float(tip[2] - cube[2]) + 0.004
    tip0 = ik.tip_mid().copy()
    tip_end = np.array([pad_xy[0] + (tip[0] - cube[0]), pad_xy[1] + (tip[1] - cube[1]), place_tip_z])
    dist = float(np.linalg.norm(tip_end - tip0))
    n = max(2, int(max(0.4, dist / 0.12) * fps))
    print(f"  carry: lower onto pad ({n / fps:.1f}s, {dist * 100:.0f} cm)…")
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
        if float(rcp.cube_pos(robot)[2]) <= rcp.CUBE_Z + 0.002:
            break  # touched down
    if rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.5) < 0.0:
        rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=1.5)
    tip = rcp.tip_mid_world(robot, arm)
    up = np.array([tip[0], tip[1], CARRY_TRANSIT_Z])
    rcp.play_tip_cartesian(robot, ik, up, rcp.FINGER_OPEN_M, fps, 0.0,
                           label="carry: retreat", freeze_wrist=True)
    return True


def on_pad(robot, pad_xy: np.ndarray) -> tuple[bool, float]:
    c = rcp.cube_pos(robot)
    d = float(np.linalg.norm(c[:2] - pad_xy))
    return (d < PAD_TOL and abs(float(c[2]) - rcp.CUBE_Z) < 0.012), d


def run_trial(robot, iks, fps: int, colour: str, cube0: np.ndarray, rng: np.random.Generator) -> bool:
    _, side, pad_xy = COLOURS[colour]
    ik = iks[side]
    for attempt in range(PLACE_MAX_ATTEMPTS):
        if attempt > 0:
            print(f"  RE-PLACE {attempt}: cube missed the pad — picking it up again")
        # pick (aim / approach / squeeze / lift, with retries and disturbances)
        if not rcp.run_aim_trial(robot, ik, fps, rcp.cube_pos(robot), rng, hold_s=0.0):
            return False
        if not carry_to_pad(robot, ik, fps, pad_xy, rcp._AIM_LAST_HOLD["m"]):
            print("  fail: carry")
            return False
        for _ in range(int(0.4 * fps)):  # settle
            rcp.send_q(robot, ik, rcp.FINGER_OPEN_M)
            rcp.precise_sleep(1.0 / fps)
        ok, d = on_pad(robot, pad_xy)
        print(f"  cube {d * 100:.1f} cm from the {side} pad centre — {'ON PAD' if ok else 'missed'}")
        if ok:
            return True
        if float(rcp.cube_pos(robot)[2]) < 0.2:
            print("  fail: cube fell off the table")
            return False
    print("  fail: could not get the cube onto the pad")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"))
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--record", default=None, metavar="REPO_ID")
    ap.add_argument("--episodes", type=int, default=0,
                    help="With --record: keep going until this many SUCCESSFUL episodes are saved.")
    ap.add_argument("--cameras", choices=["chest", "all"], default="chest")
    ap.add_argument("--debug", action="store_true",
                    help="verbose per-trial output and aim lines; default is one 'episode N' line per saved episode")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rcp._AIM_OVERLAY_ENABLED = bool(args.debug)
    robot = rcp.make_robot(args.model_path, args.fps, viewer=not args.no_viewer,
                           cameras=args.cameras if args.record else "none")
    show_pads(robot)
    iks = {a.side: rcp.build_ik(robot, a) for a in rcp.ARMS}
    rcp.park_both_arms(robot, iks)
    rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)

    recorder = None
    if args.record:
        recorder = rcp.EpisodeRecorder(robot, args.record, args.fps, task=TASK)
        rcp._RECORDER = recorder
        print(f"Recording to '{args.record}' (cameras={args.cameras}, task='{TASK}'); "
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
    used = {"red": 0, "green": 0}
    t = 0
    try:
        while t < max_trials:
            t += 1
            ok = False
            with trial_output():
                hdr = (f"episodes saved {successes}/{target_eps} (trial cap {max_trials})"
                       if target_eps else f"{t}/{max_trials}")
                print(f"\n=== Trial {t} — {hdr} ===")
                colour = next_colour(rng)
                rgba, side, _ = COLOURS[colour]
                set_cube_colour(robot, rgba)
                cube0 = place_cube_for(robot, iks, rng, colour)
                used[colour] += 1
                rcp.setup_start_pose(robot, iks[side], rng, args.fps)
                if recorder is not None:
                    recorder.start()
                ok = run_trial(robot, iks, args.fps, colour, cube0, rng)
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
                rcp.settle_pose(robot, iks[side], 0.0, args.fps, hold_s=0.15)
            if ok and not args.debug:
                print(f"episode {successes}", flush=True)
            if target_eps and successes >= target_eps:
                break
    finally:
        if recorder is not None:
            recorder.finalize()
        print(f"\nDone: {successes}/{t} successful trials (red={used['red']}, green={used['green']})")
        robot.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        if rcp._RECORDER is not None:
            rcp._RECORDER.finalize()
        os._exit(0)
