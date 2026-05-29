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

"""Swap the two arm halves of a bimanual state/action vector.

A VLA policy normalizes each state/action dimension positionally — dimension
*d* is normalized with the stats for *d*. So the physical meaning of each
dimension is fixed by the dataset the checkpoint was trained on. When the robot
streams its state in a DIFFERENT arm order than the training data, every joint
lands in the wrong normalization slot and the policy degenerates (coherent but
wrong motion, e.g. both arms drift up).

This happens specifically when running an **openpi/SparkJAX-trained** OpenArm
checkpoint on the **lerobot** ``BiOpenArmFollower``:

  * openpi/SparkJAX training data is packed **left-arm-first**:
    ``[L_j1..L_j7, L_grip, R_j1..R_j7, R_grip]``
  * lerobot ``BiOpenArmFollower.get_observation`` packs **right-arm-first**
    ``[R_j1..R_j7, R_grip, L_j1..L_j7, L_grip]`` (to match its teleoperator).

This step swaps the two equal halves of the 16-D vector so the model always
sees its training layout: deg/rad-agnostic, it is a pure permutation. The swap
is its **own inverse**, so the same step is used on the incoming observation
(robot -> model layout) and the outgoing action (model -> robot layout).

It is a **no-op unless explicitly enabled** (``enabled=False`` by default), so a
checkpoint that was both recorded and trained in lerobot — where the training
layout already matches the robot wire layout — is completely unaffected. Only
the openpi->lerobot conversion stamps ``enabled=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

__all__ = ["ArmSwapProcessorStep"]


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Swap the two equal halves of the last dimension.

    Works for (B, D) state and (B, T, D) / (B, D) action tensors. No-op (returns
    a clone) if the last dimension is not even, so a malformed/padded vector can
    never silently scramble.
    """
    dim = tensor.shape[-1]
    if dim < 2 or dim % 2 != 0:
        return tensor.clone()
    half = dim // 2
    out = tensor.clone()
    out[..., :half] = tensor[..., half:]
    out[..., half:] = tensor[..., :half]
    return out


@ProcessorStepRegistry.register("arm_swap_processor")
@dataclass
class ArmSwapProcessorStep(ProcessorStep):
    """Swap the two arm halves of ``observation.state`` and/or ``action``.

    Attributes:
        enabled: When False (default) the step is a pure pass-through — this is
            the native lerobot case where the training layout already matches
            the robot wire layout. Only openpi-converted checkpoints set True.
        apply_to_observation: Swap ``observation.state`` (preprocessor side,
            robot -> model layout).
        apply_to_action: Swap ``action`` (postprocessor side, model -> robot
            layout; also the preprocessor side during training so labels match).
    """

    enabled: bool = False
    apply_to_observation: bool = True
    apply_to_action: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = transition.copy()

        if self.apply_to_observation:
            observation = new_transition.get(TransitionKey.OBSERVATION)
            if observation:
                state = observation.get(OBS_STATE)
                if state is not None:
                    observation = dict(observation)
                    observation[OBS_STATE] = _swap_halves(state)
                    new_transition[TransitionKey.OBSERVATION] = observation

        if self.apply_to_action:
            action = new_transition.get(TransitionKey.ACTION)
            if action is not None:
                new_transition[TransitionKey.ACTION] = _swap_halves(action)

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "apply_to_observation": self.apply_to_observation,
            "apply_to_action": self.apply_to_action,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
