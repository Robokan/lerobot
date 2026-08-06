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

    # Elbow bend (deg) for the IK model's starting/rest pose. Keep this equal to
    # the robot's base_elbow_bend_deg so both models sit in the same pose.
    base_elbow_bend_deg: float = 20.0

    # Pose driver: "scripted" (headless deterministic motion, default),
    # "keyboard" (single-char terminal control), or "openxr" (Phase 2 VR).
    driver: str = "scripted"

    # Control / VR poll rate (Hz).
    vr_hz: int = 50

    # Scripted-driver trajectory shape (ignored by other drivers).
    scripted_amplitude: float = 0.06
    scripted_period_s: float = 6.0
