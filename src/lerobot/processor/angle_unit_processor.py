# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Angular-unit conversion between the robot wire format and the model's
training units.

A VLA policy has no concept of physical units — joint angles are implicit in
the normalization statistics it was trained on. When the robot hardware speaks
a *different* angular unit than the dataset the checkpoint was trained on, the
live observation lands far outside the model's normalized range and the policy
degenerates (e.g. collapses to a near-"hold" action). The OpenArm follower
reads/writes **degrees** on the wire (Damiao motors are configured with
``MotorNormMode.DEGREES``), but openpi-native checkpoints (SparkJAX-recorded
data) are trained on **radians**.

This step rescales the joint dimensions of the state and/or action by a fixed
factor so the model always sees its training unit:

  * On the preprocessor (incoming): ``scale = π/180`` converts the
    degree-valued state (and, during training, action) into radians *before*
    normalization, so it matches radian-scale norm stats.
  * On the postprocessor (outgoing): ``scale = 180/π`` converts the
    radian-valued action the model produced back into degrees *after*
    un-normalization, so the robot receives a wire-correct command.

When the checkpoint's training unit already matches the wire unit (``scale ==
1.0``) the step is a no-op. Gripper dimensions are excluded by name
(``exclude_joints``) because the gripper command is not an angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

__all__ = ["AngleUnitProcessorStep", "DEG_TO_RAD", "RAD_TO_DEG"]

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def _build_joint_mask(
    feature_names: list[str] | None, exclude_joints: list[str] | None, dim: int
) -> list[bool]:
    """Build a per-dimension mask: True = convert (it's a joint angle), False = leave as-is.

    Mirrors ``RelativeActionsProcessorStep._build_mask`` so the joint/gripper
    split is identical across the pipeline. Dimensions whose feature name
    contains (or equals) any ``exclude_joints`` token are left untouched.
    """
    if not feature_names:
        return [True] * dim

    exclude_tokens = [str(name).lower() for name in (exclude_joints or []) if name]
    if not exclude_tokens:
        return [True] * dim

    mask: list[bool] = []
    for name in list(feature_names)[:dim]:
        joint_name = str(name).lower()
        is_excluded = any(tok == joint_name or tok in joint_name for tok in exclude_tokens)
        mask.append(not is_excluded)

    if len(mask) < dim:
        mask.extend([True] * (dim - len(mask)))

    return mask


def _scale_masked(tensor: torch.Tensor, scale: float, mask: list[bool]) -> torch.Tensor:
    """Multiply ``tensor[..., d]`` by ``scale`` where ``mask[d]`` is True, else by 1.0.

    Works for both (B, D) state and (B, T, D) / (B, D) action tensors since the
    multiplier broadcasts over the trailing dimension.
    """
    dims = min(len(mask), tensor.shape[-1])
    mask_t = torch.tensor(mask[:dims], dtype=torch.bool, device=tensor.device)
    mult = torch.where(
        mask_t,
        torch.full((dims,), float(scale), device=tensor.device, dtype=tensor.dtype),
        torch.ones(dims, device=tensor.device, dtype=tensor.dtype),
    )
    out = tensor.clone()
    out[..., :dims] = out[..., :dims] * mult
    return out


@ProcessorStepRegistry.register("angle_unit_processor")
@dataclass
class AngleUnitProcessorStep(ProcessorStep):
    """Rescale joint-angle dimensions of state/action by a fixed factor.

    Attributes:
        scale: Multiplicative factor applied to masked (joint) dims. ``π/180``
            for deg->rad (preprocessor), ``180/π`` for rad->deg
            (postprocessor), ``1.0`` for a no-op.
        feature_names: Action/state dimension names, used (with
            ``exclude_joints``) to build the joint mask. State and action share
            this layout on the OpenArm follower. If None, every dim is treated
            as a joint and converted.
        exclude_joints: Name substrings to skip (e.g. the gripper, which is not
            an angle).
        apply_to_observation: Convert ``observation.state`` (used on the
            preprocessor side).
        apply_to_action: Convert ``action`` (used on both sides — incoming for
            training-time relative computation, outgoing for the robot).
    """

    scale: float = 1.0
    feature_names: list[str] | None = None
    exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    apply_to_observation: bool = True
    apply_to_action: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Fast path: identity when the wire unit already matches the model unit.
        if self.scale == 1.0:
            return transition

        new_transition = transition.copy()

        if self.apply_to_observation:
            observation = new_transition.get(TransitionKey.OBSERVATION)
            if observation:
                state = observation.get(OBS_STATE)
                if state is not None:
                    mask = _build_joint_mask(self.feature_names, self.exclude_joints, state.shape[-1])
                    observation = dict(observation)
                    observation[OBS_STATE] = _scale_masked(state, self.scale, mask)
                    new_transition[TransitionKey.OBSERVATION] = observation

        if self.apply_to_action:
            action = new_transition.get(TransitionKey.ACTION)
            if action is not None:
                mask = _build_joint_mask(self.feature_names, self.exclude_joints, action.shape[-1])
                new_transition[TransitionKey.ACTION] = _scale_masked(action, self.scale, mask)

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "feature_names": self.feature_names,
            "exclude_joints": self.exclude_joints,
            "apply_to_observation": self.apply_to_observation,
            "apply_to_action": self.apply_to_action,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
