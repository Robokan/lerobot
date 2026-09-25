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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

# Lightweight FK/IK-only MuJoCo model (the teleop solves IK on this; the robot
# owns a separate dynamic sim). Defaults to the same scene as the robot.
DEFAULT_MODEL_PATH = "~/sparkpack/openarm_mujoco/v1/scene.xml"


@TeleoperatorConfig.register_subclass("vr_mocap")
@dataclass
class VRMocapConfig(TeleoperatorConfig):
    """Configuration for the VR motion-capture teleoperator.

    Produces the same 16 right-first ``<side>_<motor>.pos`` (degrees) action
    keys as :class:`MujocoBiOpenArm`, computed by solving damped-least-squares
    IK from a per-hand end-effector target supplied by a pose source.
    """

    # IK model + solver params.
    model_path: str = DEFAULT_MODEL_PATH
    dls_lambda: float = 0.05
    max_iter: int = 10

    # Redundancy resolution. joint_weights is J1..J7 (shoulder -> wrist): a
    # larger value makes that joint cheaper to move, so wrist-heavy defaults let
    # the wrist absorb rotation before the elbow and shoulder are recruited.
    # limit_margin_deg is how far ahead of a joint limit the weight starts
    # tapering (the taper is what makes the handover smooth instead of a pop);
    # max_step_deg caps one IK iteration so a near-singular solve cannot jump.
    joint_weights: list[float] = field(
        default_factory=lambda: [0.15, 0.15, 0.15, 0.35, 1.0, 1.0, 1.0]
    )
    limit_margin_deg: float = 15.0
    max_step_deg: float = 6.0

    # Springs pulling each joint back toward the base pose; spring_gain scales
    # them all (0 disables). Graded as you would build it mechanically -- very
    # weak at the wrist, stiff at the shoulder -- so the wrist gives first when
    # you move out and the shoulder recovers first when you move back. Because
    # the base pose has the elbow bent 90 deg, a straight arm is the largest
    # displacement and so pulls back hardest.
    spring_weights: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 0.6, 0.02, 0.02, 0.02]
    )
    spring_gain: float = 0.15
    # Leashes. The commanded arm (the solver's own kinematic state) may lead
    # the ACTUAL sim arm by at most this much per joint: when the table stops
    # the arm the command used to run 190-208 deg away and the arm snapped
    # when contact released. Normal tracking lag is 4-5 deg.
    actual_leash_deg: float = 10.0
    # The pose target may lead the commanded tip by at most this much, so a
    # target that stalls (out of reach, joint limits) does not run away and
    # a reversed key answers within a second instead of after the backlog.
    target_leash_m: float = 0.04
    target_leash_deg: float = 20.0
    # Artificial range for the wrist roll (J5), degrees [lo, hi]. With the rest
    # position (0) as the lower limit, an L press finds the wrist already at
    # its stop in that direction, so the limit taper hands the turn to the
    # shoulder roll from the first press; J still has the wrist's full range
    # the other way, and an L after a J unwinds the wrist to 0 first, then
    # the shoulder. Model range is [-90, 90]; set that to restore it.
    wrist_limit_deg: list[float] = field(default_factory=lambda: [0.0, 90.0])
    # Mirror for the shoulder roll (J3): home (0) is its stop in the J direction,
    # so once J has brought the shoulder home it turns the wrist instead of
    # driving the shoulder on past home. Model range is [-90, 90].
    shoulder_roll_limit_deg: list[float] = field(default_factory=lambda: [-90.0, 0.0])
    # Wind-up hardening per joint (deg, 0 = off): a joint's willingness fades as
    # it winds away from rest, handing the motion to the next joint. Default
    # hardens only the wrist, over 45 deg = halfway through its travel.
    # Wrist wind-up hardening: the wrist's willingness to keep turning fades
    # over 45 deg from rest, so the shoulder joins a turn from about halfway
    # through the wrist's travel instead of at its 90 deg stop. Measured on
    # the real launch: shoulder onset at wrist 90 deg -> 44 deg. Wrist only.
    handover_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Bias toward joints that are coming home (part of the rest-spring feel). 0 = off.
    homing_boost: float = 6.0
    # Keyboard rotation keys turn the aligned joints ONE AT A TIME, each to its
    # limit: L goes shoulder-first then wrist, J goes wrist-first then shoulder.
    # Keyboard rotation keys pivot about the gripper TIP (the TCP); False = wrist pivot.
    rotate_about_tip: bool = True    # rotations pivot about the gripper tip (TCP); translations move it
    chain_rotation: bool = False   # OFF: back to basic IK (rotation keys pivot the wrist) until that is proven on screen
    # Unused by the sequential rule; kept so older command lines still parse.
    chain_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Elbow bend (deg) of the pose the springs pull toward. Purely a *desire*: the
    # arm boots with the elbow straight (as the real robot does) and is never
    # commanded here. The springs bend it toward this pose as soon as commanded
    # motion brings the hand somewhere the elbow has room to fold.
    rest_elbow_bend_deg: float = 90.0


    # Pose driver: "scripted" (headless deterministic motion, default),
    # "keyboard" (single-char terminal control), or "openxr" (Phase 2 VR).
    driver: str = "scripted"

    # Control / VR poll rate (Hz).
    vr_hz: int = 50

    # Scripted-driver trajectory shape (ignored by other drivers).
    scripted_amplitude: float = 0.06
    scripted_period_s: float = 6.0
