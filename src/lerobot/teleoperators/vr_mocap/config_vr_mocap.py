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
    # Damped-least-squares damping. 0.05 was under-damped near the reach
    # limit: with the elbow on its floor the commanded wrist reversed
    # direction on 169 of 180 consecutive ticks and the arm visibly shook
    # (J6/J7 reversing 33 and 51 times by more than half a degree). 0.15
    # removes it with no measured cost -- every translation moves the same
    # distance, the reach is unchanged and the roll ordering is intact --
    # and it makes sideways moves at full extension cleaner (a 14.6 cm pure
    # -y move where 0.05 gave 13.8 cm plus cross-axis error). 0.25 is too
    # much: the solver stops tracking (a 240-tick lift did not move at all).
    dls_lambda: float = 0.15
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
    # Stall rule (see VRMocap._leash_target): if the tip moved less than this
    # in a tick while the target is more than 4x this ahead, the target is
    # put back on the tip. Sized to the per-tick command (0.09 cm, 1.3 deg).
    stall_pos_m: float = 0.0003
    stall_deg: float = 0.25
    stall_ticks: int = 6           # consecutive stalled ticks before the target is put back (0 threshold = off)
    # Range for the wrist roll (J5), degrees [lo, hi]. The model's own range is
    # [-90, 90] and that is the default: L turns the wrist first and hands over
    # to the shoulder roll when the wrist runs out, the same way J does (the
    # shoulder is held while the wrist still has travel -- see VRMocap).
    # [0, 90] puts an artificial stop at the wrist's rest position, so an L
    # press finds it already stopped and the shoulder leads from the first
    # tick; that was the default until the wrist-first rule was in place.
    wrist_limit_deg: list[float] = field(default_factory=lambda: [-90.0, 90.0])
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
    # Keyboard j/l turn the gripper about its OWN axis (the original, default).
    # True = about the world vertical instead. They coincide with the arm
    # hanging. With the elbow bent the gripper-axis turn under the shoulder-
    # first pins is the case that does not yet return (measured: 37 deg short
    # of the target, J6 on its limit); the vertical turn only avoids it.
    # draw the commanded gripper pose (x/y/z triad) in the MuJoCo viewer from
    # the start -- the 'm' key does the same, but there is no keyboard in VR
    # How far the gripper travels per unit of hand travel (headset only).
    # 1.0 = your hand and the gripper move the same distance. Lower it when
    # your arm is longer than the robot's ~61 cm reach, so a full sweep of
    # yours maps inside the robot's workspace. Orientation is never scaled.
    # How far the gripper travels per unit of hand travel (headset only). One
    # scalar on the whole displacement, so x, y and z scale equally.
    # 1.4 -- verified in the headset on a 29 in arm. My arithmetic from the two
    # arm lengths said 0.84 and was wrong in practice; do not re-derive it.
    # HAND_SCALE on the launcher overrides.
    hand_scale: float = 1.4
    show_markers: bool = False
    yaw_about_vertical: bool = False
    chain_rotation: bool = False   # OFF: back to basic IK (rotation keys pivot the wrist) until that is proven on screen
    # Unused by the sequential rule; kept so older command lines still parse.
    chain_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Elbow bend (deg) of the pose the springs pull toward. Purely a *desire*: the
    # arm boots with the elbow straight (as the real robot does) and is never
    # commanded here. The springs bend it toward this pose as soon as commanded
    # motion brings the hand somewhere the elbow has room to fold.
    # The elbow bend the springs pull toward. 0 spends the arm's spare freedom
    # on staying extended, which is what a straight human arm should look like:
    # measured on a straight-arm raise, the elbow ends at 8 deg instead of 21.
    rest_elbow_bend_deg: float = 0.0


    # Pose driver: "scripted" (headless deterministic motion, default),
    # "keyboard" (single-char terminal control), or "openxr" (Phase 2 VR).
    driver: str = "scripted"

    # Control / VR poll rate (Hz).
    vr_hz: int = 50

    # Scripted-driver trajectory shape (ignored by other drivers).
    scripted_amplitude: float = 0.06
    scripted_period_s: float = 6.0
