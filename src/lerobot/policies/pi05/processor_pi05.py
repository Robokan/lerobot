#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import (
    DEG_TO_RAD,
    RAD_TO_DEG,
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    AngleUnitProcessorStep,
    ArmSwapProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    # Angular-unit conversion between the robot wire format (the OpenArm
    # follower speaks degrees) and the model's training unit. There is no unit
    # concept anywhere else in the model — it's implicit in the norm stats — so
    # ``config.input_angle_unit`` records what those stats assume and we rescale
    # the joint dims accordingly. "radians" -> deg2rad on the way in, rad2deg on
    # the way out; "degrees" -> scale 1.0 (no-op). Gripper dims are excluded
    # (the gripper command is not an angle). See AngleUnitProcessorStep.
    angle_unit = getattr(config, "input_angle_unit", "degrees")
    angle_exclude = getattr(config, "angle_unit_exclude_joints", ["gripper"])
    angle_names = getattr(config, "action_feature_names", None)
    to_model_scale = DEG_TO_RAD if angle_unit == "radians" else 1.0
    from_model_scale = RAD_TO_DEG if angle_unit == "radians" else 1.0

    angle_to_model = AngleUnitProcessorStep(
        scale=to_model_scale,
        feature_names=angle_names,
        exclude_joints=list(angle_exclude),
        apply_to_observation=True,
        apply_to_action=True,
    )
    # Arm-order correction. openpi/SparkJAX checkpoints are trained left-arm-
    # first; the lerobot BiOpenArmFollower streams right-arm-first. When
    # ``config.swap_arm_halves`` is set the two 8-D arm blocks are swapped on
    # the incoming observation (robot -> model layout) and the outgoing action
    # (model -> robot layout). Default False = no-op, so a checkpoint recorded
    # AND trained in lerobot (layout already matches the wire) is untouched.
    swap_arms = getattr(config, "swap_arm_halves", False)
    arm_swap_to_model = ArmSwapProcessorStep(
        enabled=swap_arms, apply_to_observation=True, apply_to_action=True
    )
    arm_swap_from_model = ArmSwapProcessorStep(
        enabled=swap_arms, apply_to_observation=False, apply_to_action=True
    )

    angle_from_model = AngleUnitProcessorStep(
        scale=from_model_scale,
        feature_names=angle_names,
        exclude_joints=list(angle_exclude),
        apply_to_observation=False,
        apply_to_action=True,
    )

    # OpenPI order: raw → unit→model → relative → normalize → model →
    # unnormalize → absolute → unit→robot
    #
    # The deg→rad step runs BEFORE relative_step (so the cached anchor state and
    # the relative action deltas are both in radians) and BEFORE the normalizer
    # (so state/action match the radian-scale norm stats). The rad→deg step runs
    # AFTER absolute reconstruction so the absolute radian action is converted
    # to wire degrees as the very last numeric step before device transfer.
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # Reorder arms to the model's layout BEFORE the angle conversion, the
        # relative anchor, and normalization, so every downstream slot matches
        # the training stats. No-op unless swap_arm_halves is set.
        arm_swap_to_model,
        angle_to_model,
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        angle_from_model,
        # Reorder arms back to the robot's wire layout as the final numeric step
        # (after rad->deg) so the robot receives a wire-correct command. No-op
        # unless swap_arm_halves is set.
        arm_swap_from_model,
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
