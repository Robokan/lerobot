#!/usr/bin/env python3
"""Language-conditioned chocolate-bar demos: n stacks in a row, ordinal prompts.

Each trial:
  1. Arrange n stacks (1-3 bars each) of chocolate bars in a row at the far
     edge of the table, left to right with spacing. Flavor colors are randomly
     assigned per trial so ordinal language cannot be shortcut via color.
  2. Generate a prompt like "place 1 of the left most bars in the middle of
     the table" or "place 2 of the 3rd to the left most bars in the middle of
     the table" (count 2 only when that stack holds >= 2 bars).
  3. The arm is chosen by which side of the robot's center the stack sits on
     (left of center -> left arm).
  4. Pick the stack's top bar, carry it, place it in the middle of the table;
     repeat for count 2 (second bar placed beside the first).

Recording (--record) stores each successful trial as one episode whose task
string is the generated prompt — the language-grounding dataset for VLAs.

Usage:
  MUJOCO_GL=egl python scripts/chocolate_bars_sim.py --trials 5 --no-viewer
  MUJOCO_GL=egl python scripts/chocolate_bars_sim.py --no-viewer \
      --record local/openarm_sim_chocbars --episodes 400 --cameras chest
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import random_cube_pick as rcp  # noqa: E402  (shared grasp machinery)

# Bars sit much lower than the 5 cm cube, which shrinks every clearance margin
# the approach machinery was tuned with — transit/hover higher for this scene.
rcp.TRANSIT_CLEARANCE = 0.15
rcp.HOVER_CLEARANCE = 0.11

MAX_BARS = 12
BAR_HALF = np.array([0.045, 0.0175, 0.00625])  # 90 x 35 x 12.5 mm
TABLE_TOP_Z = rcp.TABLE_TOP_Z

# Row of stacks at the far edge of the table (robot at x=0; far = large x).
ROW_X = 0.425
ROW_Y_SPAN = (-0.25, 0.25)
ROW_X_JITTER = 0.015
ROW_Y_JITTER = 0.010

# Where "the middle of the table" is; second bar goes beside the first.
MIDDLE = np.array([0.31, 0.0])
MIDDLE_JITTER = 0.02
SECOND_BAR_OFFSET_Y = 0.055
PLACE_TOL = 0.07  # bar counts as delivered within this radius of its target

# Flavor palette (name, rgba) — names reserved for future color-prompt work.
FLAVORS = [
    ("dark chocolate", (0.23, 0.13, 0.08, 1)),
    ("milk chocolate", (0.48, 0.29, 0.15, 1)),
    ("white chocolate", (0.93, 0.88, 0.76, 1)),
    ("ruby chocolate", (0.85, 0.45, 0.50, 1)),
    ("matcha", (0.55, 0.68, 0.40, 1)),
    ("caramel", (0.80, 0.55, 0.25, 1)),
]

WAREHOUSE = [(-0.55, -0.66 + 0.12 * i, 0.0065) for i in range(MAX_BARS)]


def paint_sign(robot, i: int, text: str) -> None:
    """Paint ``text`` (a digit) onto sign i's texture and refresh renderers."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    m = robot._model
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, f"sign_tex_{i}")
    adr, w, h, nc = m.tex_adr[tid], m.tex_width[tid], m.tex_height[tid], m.tex_nchannel[tid]
    img = Image.new("RGB", (w, h), (255, 255, 240))
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=52)
    except TypeError:
        font = ImageFont.load_default()
    bb = dr.textbbox((0, 0), text, font=font)
    dr.text(((w - bb[2] + bb[0]) / 2 - bb[0], (h - bb[3] + bb[1]) / 2 - bb[1]),
            text, fill=(10, 10, 10), font=font)
    arr = np.asarray(img, dtype=np.uint8)
    # cancel the +Z-face texture mapping (verified orientation: rot90 k=3)
    arr = np.rot90(arr, k=3)
    m.tex_data[adr : adr + w * h * nc] = np.ascontiguousarray(arr).reshape(-1)[: w * h * nc]


def refresh_camera_textures(robot) -> None:
    """Recreate camera renderers so repainted textures reach their GL contexts
    (mjr_uploadTexture needs each context current; recreation is simplest)."""
    import mujoco

    for cam in getattr(robot, "cameras", {}).values():
        r = getattr(cam, "_renderer", None)
        if r is None:
            continue
        h, w = r.height, r.width
        r.close()
        cam._renderer = mujoco.Renderer(robot._model, h, w)


def pose_sign(robot, i: int, x: float, y: float, z: float) -> None:
    """Float sign i (mocap body) at the given position, facing the robot."""
    import mujoco

    m, d = robot._model, robot._data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"sign_{i}")
    mid = m.body_mocapid[bid]
    d.mocap_pos[mid] = [x, y, z]
    # Placard lies FLAT on the table in front of its stack (+Z face up; the
    # digit bitmap is pre-rotated so it reads upright from the robot's view).
    d.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]


SIGN_PARK = [(-0.75, -0.66 + 0.12 * i, 0.02) for i in range(6)]
SIGN_LEAN_DEG = 35.0  # placard lean back from vertical
SIGN_SETBACK = 0.085  # placard sits this far in front of (toward robot) the stack


def ordinal_phrase(idx: int) -> str:
    """0 -> 'left most', 1 -> '2nd to the left most', ..."""
    if idx == 0:
        return "left most"
    n = idx + 1
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")
    return f"{n}{suffix} to the left most"


def make_prompt(count: int, stack_idx: int, rng: np.random.Generator) -> str:
    """Reference the number on the placard in front of the stack."""
    bars = "bar" if count == 1 else "bars"
    return f"place {count} {bars} from stack {stack_idx + 1} in the middle of the table"


def set_bar_pose(robot, i: int, x: float, y: float, z: float, yaw: float = 0.0) -> None:
    import mujoco

    m, d = robot._model, robot._data
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"bar_{i}_free")
    adr = m.jnt_qposadr[jid]
    dof = m.jnt_dofadr[jid]
    d.qpos[adr : adr + 3] = [x, y, z]
    d.qpos[adr + 3 : adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    d.qvel[dof : dof + 6] = 0.0


def set_bar_color(robot, i: int, rgba) -> None:
    import mujoco

    gid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_GEOM, f"bar_{i}")
    robot._model.geom_rgba[gid] = rgba


def bar_pos(robot, i: int) -> np.ndarray:
    import mujoco

    bid = mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_BODY, f"bar_{i}")
    return robot._data.xpos[bid].copy()


class Trial:
    """One arranged scene: n stacks, flavor colors, the prompt, the plan."""

    def __init__(self, robot, rng: np.random.Generator, n_range=(3, 6), force_count: int | None = None):
        import mujoco

        self.robot = robot
        n_lo, n_hi = n_range
        self.n = int(rng.integers(n_lo, n_hi + 1))

        # stack sizes such that total bars <= MAX_BARS
        while True:
            sizes = [int(rng.integers(1, 4)) for _ in range(self.n)]
            if sum(sizes) <= MAX_BARS:
                break
        self.sizes = sizes

        # evenly spaced row positions with jitter, left (y+) to right (y-)
        ys = np.linspace(ROW_Y_SPAN[1], ROW_Y_SPAN[0], self.n)
        self.stack_xy = []
        for y in ys:
            self.stack_xy.append((
                ROW_X + float(rng.uniform(-ROW_X_JITTER, ROW_X_JITTER)),
                float(y + rng.uniform(-ROW_Y_JITTER, ROW_Y_JITTER)),
            ))

        # random flavor per stack (no correlation with position across trials)
        flavor_ids = rng.permutation(len(FLAVORS))[: self.n]
        self.flavors = [FLAVORS[i] for i in flavor_ids]

        # park everything, then build stacks bottom-up
        for i in range(MAX_BARS):
            set_bar_pose(robot, i, *WAREHOUSE[i])
        self.stack_bars: list[list[int]] = []
        bar_i = 0
        for s_idx, (x, y) in enumerate(self.stack_xy):
            bars = []
            for level in range(self.sizes[s_idx]):
                z = TABLE_TOP_Z + BAR_HALF[2] * (2 * level + 1)
                set_bar_pose(robot, bar_i, x, y, z)
                set_bar_color(robot, bar_i, self.flavors[s_idx][1])
                bars.append(bar_i)
                bar_i += 1
            self.stack_bars.append(bars)
        # numbered flags above each stack (1 = left most); park the rest
        for i in range(6):
            pose_sign(robot, i, *SIGN_PARK[i])
        for s_idx, (x, y) in enumerate(self.stack_xy):
            pose_sign(robot, s_idx, x - SIGN_SETBACK, y, TABLE_TOP_Z + 0.004)

        rcp.zero_sim_velocity(robot)
        mujoco.mj_forward(robot._model, robot._data)

        # remember every bar's arranged pose: disturbing non-target bars
        # (knocking over a stack) fails the trial
        self.initial_pos = {i: bar_pos(robot, i) for i in range(MAX_BARS)}

        # choose target stack and count
        if force_count == 2:
            two_ok = [i for i, sz in enumerate(self.sizes) if sz >= 2]
            self.stack_idx = int(two_ok[rng.integers(0, len(two_ok))]) if two_ok else int(rng.integers(0, self.n))
        else:
            self.stack_idx = int(rng.integers(0, self.n))
        can_two = self.sizes[self.stack_idx] >= 2
        if force_count:
            self.count = force_count if (force_count == 1 or can_two) else 1
        else:
            self.count = 2 if (can_two and rng.uniform() < 0.5) else 1
        self.prompt = make_prompt(self.count, self.stack_idx, rng)

        # the user's arm rule: left of robot center (y>0) -> left arm.
        # Within +-3 cm of the centerline the rule is genuinely ambiguous, so
        # defer the choice to grasp-plan quality (resolved in main).
        y = self.stack_xy[self.stack_idx][1]
        self.arm_side = "left" if y > 0.03 else ("right" if y < -0.03 else None)
        self.arm_deferred = self.arm_side is None
        self.fail_stage: str | None = None

    def top_bar(self) -> int:
        """Highest remaining bar of the target stack (by live z)."""
        bars = self.stack_bars[self.stack_idx]
        return max(bars, key=lambda i: bar_pos(self.robot, i)[2])


def place_targets(count: int, rng: np.random.Generator) -> list[np.ndarray]:
    base = MIDDLE + rng.uniform(-MIDDLE_JITTER, MIDDLE_JITTER, size=2)
    if count == 1:
        return [base]
    return [base + np.array([0.0, +SECOND_BAR_OFFSET_Y / 2]),
            base + np.array([0.0, -SECOND_BAR_OFFSET_Y / 2])]


def carry_and_place(robot, ik, fps: int, target_xy: np.ndarray, grip_m: float) -> bool:
    """Transport the held bar to target_xy and set it down."""
    place_z_tip = TABLE_TOP_Z + 2 * BAR_HALF[2] + rcp.GRASP_CLEARANCE + 0.004
    transit_z = TABLE_TOP_Z + 0.16

    tip = rcp.tip_mid_world(robot, ik.arm)
    hop = np.array([tip[0], tip[1], transit_z])
    if rcp.play_tip_cartesian(robot, ik, hop, grip_m, fps, 0.0,
                              label="carry: rise", freeze_wrist=True) is None:
        return False
    over = np.array([target_xy[0], target_xy[1], transit_z])
    if rcp.play_tip_cartesian(robot, ik, over, grip_m, fps, 0.0,
                              label="carry: over middle", lock_z=transit_z,
                              freeze_wrist=True, min_z=transit_z - 0.01) is None:
        return False
    down = np.array([target_xy[0], target_xy[1], place_z_tip])
    if rcp.play_tip_cartesian(robot, ik, down, grip_m, fps, 0.0,
                              label="carry: lower", freeze_wrist=True,
                              min_z=place_z_tip - 0.005, speed_mps=0.12) is None:
        return False
    # release + retreat straight up
    rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=0.4)
    up = np.array([target_xy[0], target_xy[1], transit_z])
    rcp.play_tip_cartesian(robot, ik, up, rcp.FINGER_OPEN_M, fps, 0.0,
                           label="carry: retreat", freeze_wrist=True)
    return True


def resolve_center_arm(robot, iks, trial: Trial) -> str:
    """Stack sits on the centerline: pick the arm whose grasp pose plans better."""
    x, y = trial.stack_xy[trial.stack_idx]
    top_z = TABLE_TOP_Z + BAR_HALF[2] * (2 * trial.sizes[trial.stack_idx] - 1)
    grasp = rcp.clamp_tip_target(np.array([x, y, top_z + rcp.GRASP_CLEARANCE]))
    best, best_err = "right", 1e9
    for side, ik in iks.items():
        q = rcp.plan_q_to_tip_mid_robust(
            ik, np.deg2rad(ik.arm.idle_deg), grasp, yaw=0.0, pitch=rcp.GRASP_PITCH_RAD
        )
        if q is None:
            continue
        ik.set_q(q)
        err = float(np.linalg.norm(ik.tip_mid() - grasp))
        if err < best_err:
            best, best_err = side, err
    return best


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


def run_trial(robot, iks, fps: int, trial: Trial, rng: np.random.Generator) -> bool:
    ik = iks[trial.arm_side]
    print(f"  prompt: \"{trial.prompt}\"  (n={trial.n} sizes={trial.sizes} "
          f"stack@y={trial.stack_xy[trial.stack_idx][1]:+.2f} arm={trial.arm_side})")
    targets = place_targets(trial.count, rng)
    moved: list[int] = []

    def knocked(current: int | None = None) -> bool:
        bad = stacks_disturbed(robot, trial, set(moved) | ({current} if current is not None else set()))
        if bad:
            print(f"  fail: KNOCKED OVER stack(s): {'; '.join(bad)}")
            trial.fail_stage = "knock-over"
        return bool(bad)

    for k in range(trial.count):
        bar = trial.top_bar()
        rcp.set_target_body(f"bar_{bar}")
        try:
            bp = bar_pos(robot, bar)
            print(f"  pick {k + 1}/{trial.count}: bar_{bar} at "
                  f"({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.3f})")
            rcp.set_gripper(robot, ik, rcp.FINGER_OPEN_M, fps, hold_s=0.2)
            if rcp.approach_above_cube(robot, ik, bp, fps, 0.0) is None:
                knocked(current=bar)
                print("  fail: approach")
                trial.fail_stage = "approach"
                return False
            # Low targets (single-bar stacks) get a shallower wrist pitch so the
            # leading fingertip corner clears the table during the descent.
            low = bp[2] < TABLE_TOP_Z + 0.030
            pitch = rcp.GRASP_PITCH_RAD * (0.5 if low else 1.0)
            # On multi-bar stacks, grasp 4 mm higher so the pad bottoms stay
            # within the top bar instead of intruding into (and disturbing)
            # the bar beneath — knocking the stack is a failure.
            rcp.GRASP_CLEARANCE = 0.010 if low else 0.014
            if not rcp.pitch_tips_onto_cube(robot, ik, fps, 0.0, pitch_end=pitch):
                knocked(current=bar)
                print("  fail: could not place tips around bar")
                return False
            hold = rcp.set_gripper(robot, ik, rcp.GRASP_HOLD_M, fps, hold_s=3.0)
            if hold < 0.0:
                knocked(current=bar)
                print("  fail: never closed")
                return False
            # short straight lift with the bar, then carry
            tip = rcp.tip_mid_world(robot, ik.arm)
            lift = np.array([tip[0], tip[1], TABLE_TOP_Z + 0.16])
            if rcp.play_tip_cartesian(robot, ik, lift, hold, fps, 0.0,
                                      label="lift bar", freeze_wrist=True) is None:
                knocked(current=bar)
                print("  fail: lift")
                return False
            if float(bar_pos(robot, bar)[2]) < TABLE_TOP_Z + 0.05:
                knocked(current=bar)
                print("  fail: bar not lifted (slipped)")
                return False
            if not carry_and_place(robot, ik, fps, targets[k], hold):
                print("  fail: carry/place")
                return False
            moved.append(bar)
            if knocked():
                return False
        finally:
            rcp.set_target_body("cube")

    # settle, then verify every moved bar is at its target and on the table
    for _ in range(int(0.4 * fps)):
        rcp.send_q(robot, ik, rcp.FINGER_OPEN_M)
        rcp.precise_sleep(1.0 / fps)
    ok = True
    for bar, tgt in zip(moved, targets, strict=True):
        p = bar_pos(robot, bar)
        d = float(np.linalg.norm(p[:2] - tgt))
        on_table = TABLE_TOP_Z < p[2] < TABLE_TOP_Z + 0.08
        print(f"  bar_{bar}: dist-to-middle={d * 100:.1f}cm z={p[2]:.3f} "
              f"{'OK' if d < PLACE_TOL and on_table else 'MISS'}")
        ok &= d < PLACE_TOL and on_table

    if knocked():
        ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-min", type=int, default=3)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--model-path",
                    default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"))
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--record", default=None, metavar="REPO_ID")
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--cameras", choices=["chest", "all"], default="chest")
    ap.add_argument("--force-count", type=int, choices=[1, 2], default=None,
                    help="Force every prompt to ask for this many bars (demo aid).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    robot = rcp.make_robot(args.model_path, args.fps, viewer=not args.no_viewer,
                           cameras=args.cameras if args.record else "none")
    # The placard digits are static (1..6 left to right): paint them once, then
    # rebuild every GL context so the textures reach the GPU — cameras are
    # recreated, and the viewer (whose context has no public re-upload API) is
    # relaunched.
    for i in range(6):
        paint_sign(robot, i, str(i + 1))
    refresh_camera_textures(robot)
    if getattr(robot, "_viewer", None) is not None:
        import mujoco.viewer

        from lerobot.robots.mujoco_bi_openarm.viewer_keys import push_glfw_key

        robot._viewer.close()
        robot._viewer = mujoco.viewer.launch_passive(
            robot._model, robot._data,
            show_left_ui=False, show_right_ui=False, key_callback=push_glfw_key,
        )
    iks = {a.side: rcp.build_ik(robot, a) for a in rcp.ARMS}
    rcp.park_both_arms(robot, iks)
    rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)

    recorder = None
    if args.record:
        # task string is per-episode (the prompt), so start with a placeholder
        recorder = rcp.EpisodeRecorder(robot, args.record, args.fps, task="")
        rcp._RECORDER = recorder

    # park the cube out of the scene (it shares the table otherwise)
    rcp.set_cube_xy(robot, -0.90, -0.90)  # clear of the bar warehouse column

    target_eps = args.episodes if (args.record and args.episodes > 0) else 0
    max_trials = args.trials if not target_eps else max(args.trials, target_eps * 3)
    successes = 0
    t = 0
    try:
        while t < max_trials:
            t += 1
            hdr = (f"episodes saved {successes}/{target_eps} (trial cap {max_trials})"
                   if target_eps else f"{t}/{max_trials}")
            print(f"\n=== Trial {t} — {hdr} ===")
            trial = None
            for _ in range(6):
                cand = Trial(robot, rng, (args.n_min, args.n_max), force_count=args.force_count)
                if cand.arm_side is None:
                    cand.arm_side = resolve_center_arm(robot, iks, cand)
                # pre-check: the actual grasp pose must be plannable to <2 cm by
                # the chosen arm, else resample the layout (a data generator can
                # skip pathological geometry rather than burn a failed episode)
                x, y = cand.stack_xy[cand.stack_idx]
                top_z = TABLE_TOP_Z + BAR_HALF[2] * (2 * cand.sizes[cand.stack_idx] - 1)
                grasp = rcp.clamp_tip_target(np.array([x, y, top_z + rcp.GRASP_CLEARANCE]))
                ik_c = iks[cand.arm_side]
                q = rcp.plan_q_to_tip_mid_robust(
                    ik_c, np.deg2rad(ik_c.arm.idle_deg), grasp, yaw=0.0, pitch=rcp.GRASP_PITCH_RAD
                )
                if q is not None:
                    ik_c.set_q(q)
                    if float(np.linalg.norm(ik_c.tip_mid() - grasp)) < 0.02:
                        trial = cand
                        break
                print("  (layout unplannable for the rule-chosen arm — resampling)")
            if trial is None:
                print("  no plannable layout after 6 tries — skipping trial slot")
                continue
            # settle stacks briefly before anything moves
            rcp.settle_pose(robot, iks[trial.arm_side], 0.0, args.fps, hold_s=0.3)
            rcp.setup_start_pose(robot, iks[trial.arm_side], rng, args.fps)
            if recorder is not None:
                recorder.task = trial.prompt
                recorder.start()
            ok = run_trial(robot, iks, args.fps, trial, rng)
            if (not ok and trial.arm_deferred and trial.fail_stage == "approach"):
                # Centerline stack: the rule allows either arm, and nothing was
                # touched during the failed approach — retry with the other arm
                # as a fresh episode.
                other = "left" if trial.arm_side == "right" else "right"
                print(f"  centerline retry with {other} arm")
                if recorder is not None:
                    recorder.drop()
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks[other], 0.0, args.fps, hold_s=0.2)
                trial.arm_side = other
                trial.fail_stage = None
                rcp.setup_start_pose(robot, iks[other], rng, args.fps)
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
                    successes += 1
                print("  trial SUCCESS")
            else:
                if recorder is not None:
                    recorder.drop()
                print("  trial FAIL")
            rcp.park_both_arms(robot, iks)
            rcp.settle_pose(robot, iks[trial.arm_side], 0.0, args.fps, hold_s=0.15)
            if target_eps and successes >= target_eps:
                break
    finally:
        if recorder is not None:
            recorder.finalize()
        print(f"\nDone: {successes}/{t} successful trials")
        robot.disconnect()


if __name__ == "__main__":
    main()
