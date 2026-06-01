#!/usr/bin/env python
"""Replay a recorded episode from a LeRobot dataset on the physical BiOpenArm.

Plays the dataset's ``action`` column straight to the robot -- no policy, no
server. Useful for checking demo smoothness on the real arm (e.g. comparing
the raw ``openarm-chocolate-v4`` vs the smoothed ``openarm-chocolate-v4-smoothed``
set) before committing to a training run.

Dataset format note: this script runs in the *current* lerobot (codebase v3.0),
so point ``--repo-id`` at the v3.0 datasets (``local/openarm-chocolate-v4`` /
``local/openarm-chocolate-v4-smoothed``). The v2.1 copies built for openpi
training (``openarm-teleop-16dof-v4*``) will raise a BackwardCompatibilityError
here -- they hold the same recordings, just in the format openpi's pinned
lerobot reads.

The dataset stores actions in MODEL layout ([L8, R8], radians) -- the same
representation the FlashRT server emits -- so the conversion here mirrors
``scripts/run_flashrt_bridge.py:action_to_wire`` exactly (rad->deg on all 16
dims + arm-half swap to the follower's [R8, L8] wire layout) and reuses the
same ``ActionSafetyChecker`` used in deployment.

Before playing, the arms are ramped SLOWLY from their current pose to the
episode's first recorded pose (``--ramp-seconds``), so frame 0 is not a
violent single-tick jump (and does not trip the safety delta gate).

Episode scoping: ``LeRobotDataset(repo_id, episodes=[N])`` returns a view
scoped to that one episode -- ``num_frames`` and ``select_columns`` cover only
episode N (even though v3.0 datasets concatenate all episodes into one
parquet), so indexing 0..num_frames-1 walks exactly that episode in order.

Prereqs (each in its own shell):
  1. CAN buses up:  sudo bash scripts/bring_up_can.sh

Example:
    python scripts/replay_episode.py \
        --repo-id local/openarm-chocolate-v4-smoothed \
        --episode 0 \
        --left-can can3 --right-can can2 \
        --ramp-seconds 2.5
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.bi_openarm_follower import (
    BiOpenArmFollower,
    lift_arms_to_ready,
)
from lerobot.robots.bi_openarm_follower.action_safety import (
    ActionSafetyChecker,
    ActionSafetyConfig,
)
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import (
    BiOpenArmFollowerConfig,
)
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.rl.process import ProcessSignalHandler
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("replay_episode")


class _ReplayHalted(Exception):
    """Internal sentinel: a safety stop / operator shutdown halted the run.

    Used to unwind to the ``finally`` block (which disconnects the robot and
    disables torque) without logging a spurious fatal-error traceback.
    """


# The OpenArm follower's wire format is DEGREES for every motor (joints AND
# grippers -- Damiao motors use MotorNormMode.DEGREES); the dataset stores
# RADIANS for every dim. Same scale the checkpoint's angle_unit_processor uses
# (exclude_joints: []).
_RAD2DEG = 180.0 / np.pi


def _swap_halves(vec: np.ndarray) -> np.ndarray:
    """Swap the two equal halves of a 1-D 16-vector ([L8,R8] <-> [R8,L8]).

    Identical permutation to ``lerobot.processor.ArmSwapProcessorStep`` (and
    ``run_flashrt_bridge._swap_halves``) -- its own inverse.
    """
    half = vec.shape[-1] // 2
    out = vec.copy()
    out[:half] = vec[half:]
    out[half:] = vec[:half]
    return out


def action_to_wire(action_model: np.ndarray, action_keys: list[str]) -> dict[str, float]:
    """Dataset/model-layout action (rad, [L8,R8]) -> robot wire action_dict (deg, [R8,L8]).

    Mirrors ``run_flashrt_bridge.action_to_wire``: rad->deg on ALL 16 dims
    (grippers included), then swap halves back to the follower's wire layout.
    """
    a = np.asarray(action_model, dtype=np.float32).flatten()
    if a.shape[0] != len(action_keys):
        raise ValueError(
            f"dataset action dim {a.shape[0]} != robot action dim {len(action_keys)}"
        )
    a_wire = _swap_halves(a * _RAD2DEG)
    return {key: float(a_wire[i]) for i, key in enumerate(action_keys)}


def build_robot(args: argparse.Namespace) -> BiOpenArmFollower:
    """BiOpenArmFollower without cameras -- replay only needs joint state."""
    left_arm = OpenArmFollowerConfigBase(
        port=args.left_can,
        side="left",
        use_can_fd=False,
        can_bitrate=args.can_bitrate,
    )
    right_arm = OpenArmFollowerConfigBase(
        port=args.right_can,
        side="right",
        use_can_fd=False,
        can_bitrate=args.can_bitrate,
    )
    cfg = BiOpenArmFollowerConfig(
        id=args.robot_id,
        left_arm_config=left_arm,
        right_arm_config=right_arm,
        cameras={},
    )
    return BiOpenArmFollower(cfg)


def ramp_to_pose(
    robot: BiOpenArmFollower,
    action_keys: list[str],
    target_wire: dict[str, float],
    seconds: float,
    hz: float,
    safety_checker: "ActionSafetyChecker | None",
    shutdown_event,
) -> tuple[bool, dict[str, float] | None]:
    """Linearly interpolate from the current observed pose to ``target_wire``.

    Avoids a violent single-tick jump (and a frame-0 safety stop) when the
    robot's current pose differs from the episode's first recorded action.

    Gated by the same ``safety_checker`` and operator ``shutdown_event`` as the
    replay loop, so a bad command or a Ctrl-C during the approach halts the
    arms and holds the last good pose. Returns ``(ok, last_good_wire)`` --
    ``ok`` is False if a safety violation or shutdown interrupted the ramp.
    """
    cur = robot.get_observation()
    start = np.array([float(cur[k]) for k in action_keys], dtype=np.float64)
    tgt = np.array([target_wire[k] for k in action_keys], dtype=np.float64)
    n = max(1, int(round(seconds * hz)))
    max_move = float(np.max(np.abs(tgt - start)))
    logger.info(
        "Ramping to episode start pose over %.1fs (%d steps); max joint move %.1f deg",
        seconds, n, max_move,
    )
    interval = 1.0 / hz
    last_good_wire: dict[str, float] | None = None
    for i in range(1, n + 1):
        if shutdown_event.is_set():
            logger.warning("Shutdown requested during ramp; holding pose.")
            return False, last_good_wire
        alpha = i / n
        interp = (1.0 - alpha) * start + alpha * tgt
        action_wire = {k: float(interp[j]) for j, k in enumerate(action_keys)}
        if safety_checker is not None:
            err = safety_checker.check(action_wire, robot.get_observation())
            if err is not None:
                logger.error("%s", err)
                if last_good_wire is not None:
                    robot.send_action(last_good_wire)
                return False, last_good_wire
        robot.send_action(action_wire)
        last_good_wire = action_wire
        time.sleep(interval)
    return True, last_good_wire


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Dataset
    ap.add_argument("--repo-id", required=True, help="v3.0 dataset, e.g. local/openarm-chocolate-v4-smoothed")
    ap.add_argument("--root", default=None, help="Dataset root (default: $HF_LEROBOT_HOME/<repo-id>)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fps", type=float, default=None, help="Playback rate (default: dataset fps)")
    # Start-pose ramp
    ap.add_argument(
        "--ramp-seconds",
        type=float,
        default=2.5,
        help="Seconds to slowly move from the current pose to the episode's first frame.",
    )
    ap.add_argument(
        "--ramp-hz",
        type=float,
        default=50.0,
        help="Control rate used during the start-pose ramp.",
    )
    ap.add_argument(
        "--lift",
        action="store_true",
        help="Lift arms to the training-distribution ready pose before the ramp (optional).",
    )
    # Robot / CAN
    ap.add_argument("--robot-id", default="lumpa")
    ap.add_argument("--left-can", default="can3")
    ap.add_argument("--right-can", default="can2")
    ap.add_argument("--can-bitrate", type=int, default=1_000_000)
    # Action safety (mirrors run_flashrt_bridge / eval_with_real_robot defaults)
    _safety = ActionSafetyConfig()
    ap.add_argument("--no-action-safety", action="store_true")
    ap.add_argument(
        "--action-safety-max-joint-delta-deg",
        type=float,
        default=_safety.max_joint_delta_deg,
    )
    ap.add_argument(
        "--action-safety-max-gripper-delta-deg",
        type=float,
        default=None,
        help="None (default) disables the per-step delta gate for the gripper.",
    )
    ap.add_argument(
        "--action-safety-fast-joints",
        default="joint_7",
        help="Comma-separated motor-name substrings exempted from the per-step delta gate (fast distal joints).",
    )
    ap.add_argument(
        "--action-safety-max-fast-joint-delta-deg",
        type=float,
        default=None,
    )
    ap.add_argument("--action-safety-abs-joint-limit-deg", type=float, default=_safety.abs_joint_limit_deg)
    ap.add_argument("--action-safety-abs-gripper-limit-deg", type=float, default=_safety.abs_gripper_limit_deg)
    args = ap.parse_args()

    init_logging()

    # ----- Dataset (scoped to the single episode) -------------------------
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    fps = float(args.fps) if args.fps else float(dataset.fps)
    actions = dataset.select_columns(ACTION)
    n_frames = dataset.num_frames
    logger.info(
        "Loaded '%s' episode %d: %d frames @ %.0f fps (playing at %.0f fps)",
        args.repo_id, args.episode, n_frames, dataset.fps, fps,
    )
    if n_frames == 0:
        logger.error("Episode %d has 0 frames; nothing to replay.", args.episode)
        return

    def wire_at(idx: int) -> dict[str, float]:
        vec = np.asarray(actions[idx][ACTION], dtype=np.float32).flatten()
        return action_to_wire(vec, action_keys)

    # ----- Robot ----------------------------------------------------------
    robot = build_robot(args)
    logger.info("Connecting robot (left=%s right=%s)...", args.left_can, args.right_can)
    robot.connect()

    aborted = False
    played = 0
    try:
        action_keys = list(robot.action_features.keys())
        logger.info("Robot action keys (%d): %s", len(action_keys), action_keys)
        if len(action_keys) != 16:
            logger.warning(
                "Expected a 16-DOF bimanual layout; got %d. The arm-swap (8+8) "
                "and deg<->rad conversion assume that layout.",
                len(action_keys),
            )

        # ----- Action safety + operator e-stop ----------------------------
        # The same 3-gate hard-abort checker used in deployment, armed across
        # BOTH the ramp and the replay so any bad command halts the arms and
        # holds the last good pose. The signal handler gives a clean operator
        # e-stop: Ctrl-C / SIGTERM sets shutdown_event, the loops stop sending
        # and the finally-block disconnects (torque off).
        safety_checker = None
        if not args.no_action_safety:
            fast_joints = tuple(
                s.strip() for s in args.action_safety_fast_joints.split(",") if s.strip()
            )
            safety_checker = ActionSafetyChecker(
                ActionSafetyConfig(
                    enabled=True,
                    max_joint_delta_deg=args.action_safety_max_joint_delta_deg,
                    max_gripper_delta_deg=args.action_safety_max_gripper_delta_deg,
                    max_fast_joint_delta_deg=args.action_safety_max_fast_joint_delta_deg,
                    fast_joint_name_substrs=fast_joints,
                    abs_joint_limit_deg=args.action_safety_abs_joint_limit_deg,
                    abs_gripper_limit_deg=args.action_safety_abs_gripper_limit_deg,
                )
            )
            logger.info(
                "Action safety armed: joint_delta=%.2f deg/step, gripper_delta=%s, "
                "fast_joints=%s, abs_joint=%.1f deg, abs_gripper=%.1f deg",
                args.action_safety_max_joint_delta_deg,
                "disabled" if args.action_safety_max_gripper_delta_deg is None
                else f"{args.action_safety_max_gripper_delta_deg:.2f} deg/step",
                list(fast_joints) or "none",
                args.action_safety_abs_joint_limit_deg,
                args.action_safety_abs_gripper_limit_deg,
            )
        else:
            logger.warning("Action safety DISABLED (--no-action-safety).")

        signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
        shutdown_event = signal_handler.shutdown_event
        logger.info("Operator e-stop armed: Ctrl-C (SIGINT) / SIGTERM halts and holds pose.")

        last_good_wire: dict[str, float] | None = None

        # ----- Optional lift, then SLOW ramp to the episode start pose ----
        if args.lift:
            logger.info("Lifting arms to the training-distribution ready pose...")
            lift_arms_to_ready(robot, log_fn=logger.info)
        if args.ramp_seconds > 0:
            ok, last_good_wire = ramp_to_pose(
                robot, action_keys, wire_at(0), args.ramp_seconds, args.ramp_hz,
                safety_checker, shutdown_event,
            )
            if not ok:
                aborted = True
                raise _ReplayHalted("ramp interrupted by safety stop / shutdown")

        # ----- Replay loop ------------------------------------------------
        logger.info("Replaying episode %d (%d frames)...", args.episode, n_frames)
        interval = 1.0 / fps
        for idx in range(n_frames):
            if shutdown_event.is_set():
                logger.warning("Shutdown requested; stopping replay and holding pose.")
                if last_good_wire is not None:
                    robot.send_action(last_good_wire)
                aborted = True
                break

            t0 = time.perf_counter()
            action_wire = wire_at(idx)

            if safety_checker is not None:
                err = safety_checker.check(action_wire, robot.get_observation())
                if err is not None:
                    logger.error("%s", err)
                    if last_good_wire is not None:
                        robot.send_action(last_good_wire)
                    aborted = True
                    break

            robot.send_action(action_wire)
            last_good_wire = action_wire
            played += 1

            if idx % int(fps) == 0:
                logger.info("[replay] frame %d/%d (t=%.1fs)", idx, n_frames, idx / fps)

            sleep_s = interval - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except _ReplayHalted as halt:
        logger.warning("Replay halted: %s", halt)
        aborted = True
    except Exception:  # noqa: BLE001
        logger.exception("Fatal error during replay")
        aborted = True
    finally:
        logger.info("Shutting down (frames played=%d, aborted=%s)...", played, aborted)
        try:
            robot.disconnect()
            logger.info("Robot disconnected (torque disabled)")
        except Exception:  # noqa: BLE001
            logger.warning("robot.disconnect() failed", exc_info=True)

    logger.info("Replay finished.")


if __name__ == "__main__":
    main()
