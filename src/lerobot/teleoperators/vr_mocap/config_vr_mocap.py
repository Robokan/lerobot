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
    # Wind-up hardening per joint (deg, 0 = off): a joint's willingness fades as
    # it winds away from rest, handing the motion to the next joint. Default
    # hardens only the wrist, over 45 deg = halfway through its travel.
    handover_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 45.0, 45.0, 45.0])
    # Keyboard rotation keys as a spring chain: the requested rotation is split
    # across the joints in joint space by alignment / stiffness, so wrist AND
    # shoulder turn together from the first degree, the softer one more. A
    # joint at its limit in the needed direction has no compliance there.
    chain_rotation: bool = True
    # Stiffness J1..J7 for the OUTBOUND split (relative; lower = moves more).
    # Shoulder softer than the wrist, so the shoulder visibly turns with the
    # wrist and, with equal ranges, reaches its limit first; the wrist then
    # continues alone. The return order is fixed (wrist, elbow, shoulder) and
    # does not depend on these.
    chain_weights: list[float] = field(default_factory=lambda: [0.6, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0])

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
