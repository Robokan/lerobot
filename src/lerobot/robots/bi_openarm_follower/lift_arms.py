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

"""Move both OpenArm followers from "hanging at rest" to the "arms up, table-cleared"
ready pose before handing control to a policy.

Ported from SparkJAX's ``openpi_runner_node._raise_arms`` (see
``/home/evaughan/sparkpack/SparkJAX/sparkjax/sparkjax/teleop/openpi_runner_node.py``).
The spine waypoints are the symmetrized mean trajectory extracted from 11 real
teleop episodes (originally documented in ``scripts/generate_lift_episodes.py``
in that same repo). They sweep the arms outward to clear the table, then tuck
back in to a "ready" posture that matches the typical start state of the
chocolate-bars demos.

Differences from the SparkJAX version:
    * SparkJAX wrote ``LIFT_SPINE[0]`` once into a C++ unilateral-control FIFO
      and slept 3 s, relying on the controller's internal smooth-move ramp to
      drive the arms from their current pose to the start of the spine. lerobot
      drives the motors directly via PD set-points, so we read the current
      observation and interpolate from there to ``LIFT_SPINE[0]`` ourselves
      before streaming the spine.
    * SparkJAX values are radians (openpi-native). lerobot's
      :class:`OpenArmFollower` takes degrees, so we convert here.
    * Joint ordering matches the ``openarm-chocolate-v4`` dataset
      ``action.names``: ``[left_joint1..7, left_gripper, right_joint1..7,
      right_gripper]`` (16 values).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")
_LEFT_KEYS = tuple(f"left_{j}.pos" for j in _JOINT_NAMES)
_RIGHT_KEYS = tuple(f"right_{j}.pos" for j in _JOINT_NAMES)
_SPINE_KEY_ORDER: tuple[str, ...] = _LEFT_KEYS + _RIGHT_KEYS

# NOTE on the gripper column (index 7 = left_gripper, index 15 = right_gripper):
# the upstream SparkJAX waypoints have `gripper = 0.004 rad = +0.23 deg` for
# every spine row, just below the closed-stop. The lerobot calibration zero
# IS the closed stop (gripper range is configured `[-65, 0] deg` in
# OpenArmFollower), so commanding +0.23 deg ("past closed") makes the DM4310
# grip motor stall against its own mechanical stop, draw over-current, and
# fault-latch. After a fault-latch the motor IGNORES the handshake's
# CAN_CMD_ENABLE and the next lerobot-record startup fails with
# "ConnectionError: Handshake failed... ['gripper']" until you either send
# DISABLE-then-ENABLE on the bus (which the patched handshake now does
# automatically) or cycle 24V. We zero out the gripper column here so the
# spine commands a safely-closed grip throughout the lift, never overdriving
# the stop.
#
# Rows 0..8 are the *original* SparkJAX symmetrized-mean spine (extracted
# from 11 real teleop lift demonstrations). They sweep the arms outward to
# clear the table, then tuck back in to an "arms up, fully tucked" posture
# at row 8 = SparkJAX's _ARMS_UP target.
#
# Row 9 = READY_POSE_RAD (see constant block below). It is appended so that
# after the table-clearing lift completes the arms transition smoothly to
# the HIGH-cluster training-distribution start pose (filtered median across
# the 177 chocolate-task episodes whose joint_4 >= 40 deg). Without this
# tail, the policy sees an out-of-distribution arms-fully-tucked state
# (joint_4 ~ 88 deg, joint_7 ~ 80 deg) and its first action chunk drives
# the arms ~16-37 deg down/inward to return to the training distribution -
# which looks like "the arms dropped right after the lift" to an observer.
#
# Why HIGH (not MID): both clusters produce "hold-near-current" first
# chunks at runtime, but only HIGH is a visibly-lifted pose. The MID
# alternative (arms hanging slightly forward, j4 ~ 19 deg) was tried in
# 28-May runs and the lift was visually imperceptible (~17 deg max motion
# from zero). HIGH makes the lift visible AND was the most common start
# pose in the 177-episode filtered cluster. fps=50 (matching training,
# vs the old fps=30) is now the runtime default - so the cumulative
# downward drift that previously plagued HIGH (~1 deg/chunk over many
# chunks) should integrate at half the wall-clock rate.
LIFT_SPINE_RAD: np.ndarray = np.array(
    [
        # 0 % - start (arms hanging, symmetric)
        [0.266, -0.049, -0.064, 0.345, -0.007, 0.102, -0.023, 0.0,
         -0.266, 0.049, 0.064, 0.345, 0.007, -0.102, 0.023, 0.0],
        # 11 %
        [0.287, -0.049, -0.064, 0.364, -0.006, 0.102, -0.072, 0.0,
         -0.287, 0.049, 0.064, 0.364, 0.006, -0.102, 0.072, 0.0],
        # 22 %
        [0.559, -0.049, -0.064, 0.611, -0.006, 0.102, -0.228, 0.0,
         -0.559, 0.049, 0.064, 0.611, 0.006, -0.102, 0.228, 0.0],
        # 33 %
        [0.866, -0.049, -0.064, 1.022, -0.008, 0.102, -0.423, 0.0,
         -0.866, 0.049, 0.064, 1.022, 0.008, -0.102, 0.423, 0.0],
        # 44 %
        [1.052, -0.049, -0.064, 1.170, -0.011, 0.102, -0.713, 0.0,
         -1.052, 0.049, 0.064, 1.170, 0.011, -0.102, 0.713, 0.0],
        # 55 %
        [1.253, -0.049, -0.064, 1.377, -0.011, 0.102, -0.955, 0.0,
         -1.253, 0.049, 0.064, 1.377, 0.011, -0.102, 0.955, 0.0],
        # 66 %
        [1.359, -0.049, -0.064, 1.442, -0.011, 0.102, -1.179, 0.0,
         -1.359, 0.049, 0.064, 1.442, 0.011, -0.102, 1.179, 0.0],
        # 77 %
        [1.427, -0.050, -0.064, 1.530, -0.011, 0.102, -1.285, 0.0,
         -1.427, 0.050, 0.064, 1.530, 0.011, -0.102, 1.285, 0.0],
        # 88 % - SparkJAX's _ARMS_UP ("arms up, fully tucked"). The original
        # 9-waypoint spine ended here. Table is fully cleared by this point.
        [1.427, -0.050, -0.064, 1.530, -0.011, 0.102, -1.391, 0.0,
         -1.427, 0.050, 0.064, 1.530, 0.011, -0.102, 1.391, 0.0],
        # 100 % - small settle into HIGH-cluster training-distribution
        # ready pose. Tucks the shoulders inward (j1 +-82 -> +-75 deg),
        # unfolds the elbow a hair (j4 88 -> 79 deg), and relaxes the
        # wrist rotation (j7 +-80 -> +-73 deg). Net motion is ~9 deg over
        # the last segment (~190 ms at the default 1.7s duration), well
        # below SparkJAX's typical spine-segment speed. No table-collision
        # risk: the shoulders barely move (j2 stays under 8 deg) and the
        # arms remain elevated.
        [+1.3136, -0.1223, -0.2047, +1.3727, -0.0704, -0.1001, -1.2831, 0.0,
         -1.3083, +0.1223, +0.1997, +1.3727, +0.0708, +0.1001, +1.2690, 0.0],
    ],
    dtype=np.float64,
)

# READY_POSE_RAD == LIFT_SPINE_RAD[-1] (kept as a separate named constant for
# clarity in logs / debugging / external scripts that want to compare a
# captured pose against the policy's expected start pose).
#
# Values are the per-joint **median** of `observation.state[frame_index==0]`
# across the 177 chocolate-task episodes of openarm-chocolate-v4 whose
# joint_4 (elbow flex) starts at >= 40 deg (the "HIGH" cluster). The dataset
# has 282 episodes total but 105 of them start with arms hanging down at
# joint_4 ~ 19 deg (matches LIFT_SPINE[0] within 0.5 deg) - those are the
# MID cluster (operator started recording before lifting the arms).
# Including them would bias the median downward and put the policy ~30 deg
# below the true HIGH-cluster start on joints 1, 4, and 7.
#
# The bimodal distribution has a clean gap between 35 and 45 deg on
# joint_4_L (only 1 of 282 episodes falls in that 10-deg window), so
# the 40-deg threshold robustly separates the two clusters. If you
# re-fine-tune on a different dataset, regenerate this with:
#     mask = joint_4_left_start_deg >= 40
#     READY_POSE_RAD = np.median(states[mask, :], axis=0)
#     # then zero out the gripper columns (indices 7, 15) for safety.
#
# Per-joint medians of the filtered set (degrees) for reference:
#     joint_1: +-75.26   joint_2: -+ 7.01   joint_3: -+11.73   joint_4: +78.65
#     joint_5: -+ 4.03   joint_6: -+ 5.74   joint_7: -+73.52
# Compared to SPINE[-2] (SparkJAX's _ARMS_UP target), this is only
# +-7 deg on j1, -9 deg on j4, +-7 deg on j7 - a small, safe settle.
# Gripper columns are forced to 0.0 (closed) for the same hardware-safety
# reason that the rest of the spine has gripper=0.0 (see comment above).
READY_POSE_RAD: np.ndarray = LIFT_SPINE_RAD[-1].copy()

# Four-phase lift sequence (extends SparkJAX's user-visible behavior with
# a final "settle to training-distribution pose" tail on the spine):
#
#     Phase 0 (pre-zero, ``pre_zero_s``):
#         Slow soft-gain ramp from the current pose to the operator-calibrated
#         "default hanging" pose (``DEFAULT_BIMANUAL_START_POSE_DEG``).
#         Always runs first so the arms end at a known, symmetric posture
#         regardless of where they were left at the end of the previous
#         session (stuck mid-lift, hand-bumped, knocked, etc.).
#     Phase 1 (pre-ramp, ``pre_ramp_s``):
#         Soft-gain ramp from default-hanging -> ``LIFT_SPINE[0]``. Mirrors
#         SparkJAX's C++ ``Control::AdjustPosition`` smooth-move from the
#         arm's "rest" pose to the first spine waypoint (~2.2s + buffer).
#     Phase 2 (spine, ``spine_duration_s``):
#         Stiff-gain sweep through the 10 LIFT_SPINE waypoints:
#           - rows 0..7: SparkJAX's original table-clearing arc (arms outward
#             and up). DO NOT shortcut this part - the wide swing is the
#             only thing that keeps the elbows off the table on the way up.
#           - row  8 (SPINE[-2]): SparkJAX's _ARMS_UP target, arms fully
#             tucked up. The table is fully cleared by this point.
#           - row  9 (SPINE[-1] == READY_POSE_RAD): settle into the HIGH-
#             cluster training-distribution-center pose (filtered median of
#             the 177 chocolate-v4 episodes whose j4 >= 40 deg). Shoulders
#             barely move (j2 stays < 7 deg) so there is no table-collision
#             risk for this final tuck.
#     Phase 3 (hold, ``hold_s``):
#         Re-send ``LIFT_SPINE[-1]`` (= READY_POSE_RAD) so the arms settle
#         before the policy takes over. This is the pose the LoRA expects
#         to see as its first observation.
#
# Set ``pre_zero_s = 0`` (default-skipped) only if you've already manually
# positioned the arms or you know they're at the calibrated pose already.
DEFAULT_PRE_ZERO_S: float = 3.0
DEFAULT_PRE_RAMP_S: float = 3.0
# Bumped from SparkJAX's 1.5s to 1.7s after we appended READY_POSE_RAD as
# the spine's 10th waypoint. 1.7s / 9 segments ~ 189 ms/segment, matching
# SparkJAX's original 1.5s / 8 ~ 188 ms/segment pacing exactly, so the
# table-clearing arc retains its tested speed. The final READY tuck moves
# at most ~9 deg in 189 ms (~48 deg/s peak), far below DM4310 limits.
DEFAULT_LIFT_DURATION_S: float = 1.7
DEFAULT_LIFT_HZ: float = 50.0
DEFAULT_HOLD_S: float = 0.5

# Phase 0 target: the LeRobot-canonical OpenArm "zero pose" — arms hanging
# straight down with grippers closed. This is the SAME pose the user is
# instructed to put the arms in during the official calibration flow
# (see ``OpenArmFollower.calibrate``, which prompts:
#     "Position the arm in the following configuration:
#         - Arm hanging straight down
#         - Gripper closed
#      Press ENTER when ready..."
# and then writes that physical pose as the motor flash zero via
# ``bus.set_zero_position()``).  Because the calibration defines hanging-
# straight as encoder 0, every joint here is exactly 0.0 deg.
#
# Always running Phase 0 first means: regardless of where the arms were
# left at the end of the previous session (stuck mid-lift, hand-bumped,
# knocked, …), they end Phase 0 at the same physical posture as the
# calibration zero, which is by construction a safe, symmetric, known-good
# start state for the lift spine.
#
# If the arms ever stop physically matching this pose at all-zeros, the
# fix is to re-run LeRobot's calibration ("lerobot-calibrate --robot.type=
# bi_openarm_follower"), NOT to bake non-zero values in here. Overriding
# the target on a per-call basis is still supported via the
# ``default_start_pose_deg`` arg to :func:`lift_arms_to_ready`, and
# ``scripts/test_default_pose.py --capture`` can rewrite this dict in
# place from the live observation if you need a custom non-zero pose for
# a specific experiment.
DEFAULT_BIMANUAL_START_POSE_DEG: dict[str, float] = {
    "left_joint_1.pos":  +0.00,  "right_joint_1.pos":  +0.00,
    "left_joint_2.pos":  +0.00,  "right_joint_2.pos":  +0.00,
    "left_joint_3.pos":  +0.00,  "right_joint_3.pos":  +0.00,
    "left_joint_4.pos":  +0.00,  "right_joint_4.pos":  +0.00,
    "left_joint_5.pos":  +0.00,  "right_joint_5.pos":  +0.00,
    "left_joint_6.pos":  +0.00,  "right_joint_6.pos":  +0.00,
    "left_joint_7.pos":  +0.00,  "right_joint_7.pos":  +0.00,
    "left_gripper.pos":  +0.00,  "right_gripper.pos":  +0.00,
}

# Softer PD gains used during the pre-zero ramp (and optionally the pre-ramp
# to spine[0]). Ported verbatim from SparkJAX's
# ``openarm_teleop/src/controller/control.cpp::AdjustPosition`` which uses
# these gentle gains for the smooth-move from current pose to target before
# handing control to the leader/policy. Default stiff gains
# (kp=240, kd=5.0 on the four shoulder joints) yank the arms hard if the
# motors have drifted or the user is holding them; the softer ones produce
# a gentle compliant move at the cost of slower steady-state tracking
# (which is fine for a slow 6-second ramp).
SAFE_RAMP_KP: dict[str, float] = {
    "joint_1": 50.0,   # shoulder pan       (vs. 240 normal)
    "joint_2": 50.0,   # shoulder lift      (vs. 240 normal)
    "joint_3": 50.0,   # shoulder rotation  (vs. 240 normal)
    "joint_4": 50.0,   # elbow flex         (vs. 240 normal)
    "joint_5": 10.0,   # wrist roll         (vs.  24 normal)
    "joint_6": 10.0,   # wrist pitch        (vs.  31 normal)
    "joint_7": 10.0,   # wrist rotation     (vs.  25 normal)
    "gripper": 10.0,   # gripper            (vs.  25 normal)
}
SAFE_RAMP_KD: dict[str, float] = {
    "joint_1": 1.2,
    "joint_2": 1.2,
    "joint_3": 1.2,
    "joint_4": 1.2,
    "joint_5": 0.3,
    "joint_6": 0.2,
    "joint_7": 0.3,
    "gripper": 0.5,
}


def _interp_trajectory(
    start: np.ndarray, end: np.ndarray, duration_s: float, hz: float
) -> np.ndarray:
    """Linear interpolation from ``start`` to ``end`` at ``hz`` for ``duration_s``."""
    n = max(2, int(round(duration_s * hz)))
    t = np.linspace(0.0, 1.0, n)
    return start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]


def _read_current_pose_deg(robot: Any) -> np.ndarray:
    """Read the current 16-DOF pose from the bimanual observation, in degrees.

    Falls back to ``LIFT_SPINE[0]`` for any missing joint key with a warning.
    """
    obs = robot.get_observation()
    pose = np.empty(len(_SPINE_KEY_ORDER), dtype=np.float64)
    for i, key in enumerate(_SPINE_KEY_ORDER):
        v = obs.get(key)
        if v is None:
            fallback_deg = float(np.rad2deg(LIFT_SPINE_RAD[0, i]))
            logger.warning(
                "lift_arms: observation missing %s; falling back to spine[0]=%.2f deg",
                key, fallback_deg,
            )
            pose[i] = fallback_deg
        else:
            pose[i] = float(v)
    return pose


def _to_action_dict(pose_deg: np.ndarray) -> dict[str, float]:
    return {key: float(pose_deg[i]) for i, key in enumerate(_SPINE_KEY_ORDER)}


def _warn_if_outside_limits(robot: Any) -> None:
    """Log a warning if any LIFT_SPINE waypoint exceeds the follower's per-joint limits.

    This doesn't abort: ``OpenArmFollower.send_action`` already clips per-joint,
    so an over-range waypoint becomes a slightly trimmed end pose. We just want
    the user to know it's happening.
    """
    spine_deg = np.rad2deg(LIFT_SPINE_RAD)
    for side, keys, offset in (("left", _LEFT_KEYS, 0), ("right", _RIGHT_KEYS, 8)):
        arm = getattr(robot, f"{side}_arm", None)
        if arm is None:
            return
        limits = getattr(arm.config, "joint_limits", None)
        if not limits:
            return
        for j, joint in enumerate(_JOINT_NAMES):
            lo, hi = limits.get(joint, (-np.inf, np.inf))
            col = spine_deg[:, offset + j]
            if col.min() < lo or col.max() > hi:
                logger.warning(
                    "lift_arms: spine %s_%s in [%.2f, %.2f] deg exceeds limit [%.2f, %.2f]; "
                    "values will be clipped by send_action.",
                    side, joint, col.min(), col.max(), lo, hi,
                )


def _start_pose_to_array(pose_dict: dict[str, float]) -> np.ndarray:
    """Convert a dict keyed by ``{left,right}_{joint,gripper}.pos`` to the 16-DOF
    array in ``_SPINE_KEY_ORDER`` order. Missing keys default to 0.0 with a warning.
    """
    arr = np.zeros(len(_SPINE_KEY_ORDER), dtype=np.float64)
    for i, key in enumerate(_SPINE_KEY_ORDER):
        v = pose_dict.get(key)
        if v is None:
            logger.warning(
                "lift_arms: default_start_pose_deg missing key %s; using 0.0", key
            )
        else:
            arr[i] = float(v)
    return arr


def lift_arms_to_ready(
    robot: Any,
    *,
    pre_zero_s: float = DEFAULT_PRE_ZERO_S,
    pre_ramp_s: float = DEFAULT_PRE_RAMP_S,
    spine_duration_s: float = DEFAULT_LIFT_DURATION_S,
    hz: float = DEFAULT_LIFT_HZ,
    hold_s: float = DEFAULT_HOLD_S,
    default_start_pose_deg: dict[str, float] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Smoothly drive both OpenArm followers to the policy's training-
    distribution start pose.

    Four-phase motion (extends SparkJAX's ``_raise_arms`` with a leading
    "go to LeRobot-canonical zero pose first" safety phase and a trailing
    "settle into training-distribution-center pose" spine tail):

        0. **Go to zero pose** (``pre_zero_s``): linearly interpolate from
           the current pose to ``default_start_pose_deg`` (defaults to
           :data:`DEFAULT_BIMANUAL_START_POSE_DEG`, which is **all zeros**
           - the LeRobot-canonical OpenArm zero pose = arms hanging
           straight down, grippers closed, set by ``OpenArmFollower.calibrate``).
           Anchors the arms at a known-safe baseline regardless of where they
           ended up from a previous session (stuck mid-lift after a crash,
           hand-bumped, etc.). Skipped when ``pre_zero_s == 0``.
        1. **Pre-ramp** (``pre_ramp_s``): linearly interpolate from the current
           pose (re-read after phase 0) to ``LIFT_SPINE_RAD[0]`` at ``hz``.
           Avoids the PD controller yanking the arms when the spine's first
           set-point is far from the current pose.
        2. **Spine** (``spine_duration_s``): linearly interpolate through all
           10 spine waypoints at ``hz``. The first 9 are SparkJAX's table-
           clearing lift arc; the 10th (= :data:`READY_POSE_RAD`) is the
           training-distribution-center pose so the policy's first
           observation matches what it was fine-tuned on.
        3. **Hold** (``hold_s``): re-send ``LIFT_SPINE_RAD[-1]`` (= the ready
           pose) at ``hz`` so the arms settle before policy hand-off.

    Args:
        robot: A :class:`BiOpenArmFollower` (or any robot whose
            ``observation_features`` / ``action_features`` use the
            ``{left,right}_{joint_1..7,gripper}.pos`` naming and whose
            ``send_action`` takes degrees). Must be connected.
        pre_zero_s: Duration of the slow "go to LeRobot zero pose" phase
            before the spine. Always runs first when > 0. Set to 0 only if
            you have already manually positioned the arms at the zero pose.
        pre_ramp_s: Duration of the pre-ramp from the (post-pre-default)
            current pose to spine[0]. Set to 0 to skip.
        spine_duration_s: Duration of the spine sweep.
        hz: Command rate (Hz) for the smooth motion.
        hold_s: Duration to hold the final pose after the spine completes.
        default_start_pose_deg: Per-joint target pose (in degrees, keyed by
            ``{left,right}_{joint_1..7,gripper}.pos``) for phase 0. If
            ``None`` uses :data:`DEFAULT_BIMANUAL_START_POSE_DEG`. Pass ``{}``
            for the legacy all-zeros target.
        log_fn: Optional callable used to emit human-visible progress messages
            (defaults to :func:`logging.getLogger(__name__).info`).
    """
    if not robot.is_connected:
        raise RuntimeError("lift_arms_to_ready: robot must be connected before raising arms.")

    log = log_fn if log_fn is not None else logger.info

    _warn_if_outside_limits(robot)

    spine_deg = np.rad2deg(LIFT_SPINE_RAD)
    start_of_spine_deg = spine_deg[0].copy()
    end_of_spine_deg = spine_deg[-1].copy()
    pose_dict = DEFAULT_BIMANUAL_START_POSE_DEG if default_start_pose_deg is None else default_start_pose_deg
    default_start_deg = _start_pose_to_array(pose_dict)
    step_dt = 1.0 / hz

    # Phase 0 - slow ramp from current pose to the LeRobot-canonical OpenArm
    # zero pose (arms hanging straight, grippers closed; all joints = 0 deg
    # by definition of the calibration; see DEFAULT_BIMANUAL_START_POSE_DEG
    # comment block). Guarantees a known-safe start state regardless of
    # where the arms ended the last session (stuck mid-lift, knocked, etc.).
    # Uses the SOFT SAFE_RAMP_KP / SAFE_RAMP_KD gains so a perturbed /
    # hand-held arm transitions gently to the target instead of being
    # yanked by the normal stiff position gains. Mirrors SparkJAX's
    # ``Control::AdjustPosition``.
    if pre_zero_s > 0:
        current_deg = _read_current_pose_deg(robot)
        max_delta = float(np.max(np.abs(current_deg - default_start_deg)))
        log(
            f"lift_arms: slow-ramping {pre_zero_s:.2f}s from current pose to zero pose "
            f"(arms hanging, grippers closed) using SOFT gains "
            f"(kp_shoulder=50, kp_wrist=10) "
            f"(max delta = {max_delta:.1f} deg, peak speed ~{max_delta / pre_zero_s:.1f} deg/s)"
        )
        ramp = _interp_trajectory(current_deg, default_start_deg, pre_zero_s, hz)
        for pose in ramp:
            robot.send_action(_to_action_dict(pose), custom_kp=SAFE_RAMP_KP, custom_kd=SAFE_RAMP_KD)
            time.sleep(step_dt)

    # Phase 1 - pre-ramp from (post-zero) current pose to spine[0]. Still uses
    # the SOFT gains: at the end of phase 1 the arms reach spine[0] (still
    # hanging-ish, joint_1 ~15 deg), then phase 2 streams the spine and
    # smoothly transitions to the normal stiff gains for policy control.
    if pre_ramp_s > 0:
        current_deg = _read_current_pose_deg(robot)
        log(
            f"lift_arms: pre-ramping {pre_ramp_s:.2f}s from current pose to spine[0] "
            f"using SOFT gains "
            f"(max delta = {np.max(np.abs(current_deg - start_of_spine_deg)):.1f} deg)"
        )
        ramp = _interp_trajectory(current_deg, start_of_spine_deg, pre_ramp_s, hz)
        for pose in ramp:
            robot.send_action(_to_action_dict(pose), custom_kp=SAFE_RAMP_KP, custom_kd=SAFE_RAMP_KD)
            time.sleep(step_dt)

    # Phase 2 - stream the spine (table-clearing arc + settle to READY)
    n_frames = max(2, int(round(spine_duration_s * hz)))
    n_wp = len(spine_deg)
    t_wp = np.linspace(0.0, 1.0, n_wp)
    t_out = np.linspace(0.0, 1.0, n_frames)
    trajectory = np.empty((n_frames, spine_deg.shape[1]), dtype=np.float64)
    for j in range(spine_deg.shape[1]):
        trajectory[:, j] = np.interp(t_out, t_wp, spine_deg[:, j])
    log(
        f"lift_arms: streaming spine ({n_wp} waypoints -> {n_frames} frames over "
        f"{spine_duration_s:.2f}s @ {hz:.0f} Hz); final waypoint = READY_POSE_RAD "
        f"(HIGH-cluster median of 177 chocolate-task episodes; "
        f"max ~9 deg settle from SPINE[-2])"
    )
    for pose in trajectory:
        robot.send_action(_to_action_dict(pose))
        time.sleep(step_dt)

    # Phase 3 - hold READY_POSE_RAD (= spine[-1]) so the arms settle there
    # before the policy takes over. This is the pose the LoRA was fine-tuned
    # to see as its first observation, so the first action chunk should be
    # near-zero deltas instead of a large "return to training distribution"
    # correction.
    if hold_s > 0:
        n_hold = max(1, int(round(hold_s * hz)))
        log(
            f"lift_arms: holding READY_POSE_RAD for {hold_s:.2f}s ({n_hold} frames) "
            f"before policy hand-off"
        )
        action = _to_action_dict(end_of_spine_deg)
        for _ in range(n_hold):
            robot.send_action(action)
            time.sleep(step_dt)

    log("lift_arms: arms settled at READY_POSE_RAD (training-distribution-center pose)")


__all__ = [
    "LIFT_SPINE_RAD",
    "READY_POSE_RAD",
    "DEFAULT_PRE_ZERO_S",
    "DEFAULT_PRE_RAMP_S",
    "DEFAULT_LIFT_DURATION_S",
    "DEFAULT_LIFT_HZ",
    "DEFAULT_HOLD_S",
    "DEFAULT_BIMANUAL_START_POSE_DEG",
    "SAFE_RAMP_KP",
    "SAFE_RAMP_KD",
    "lift_arms_to_ready",
]
