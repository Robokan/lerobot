#!/usr/bin/env python

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

"""
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies on real robots.

This script demonstrates:
1. Creating a robot and policy (SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

For simulation environments, see eval_with_simulation.py

Usage:
    # Run RTC with Real robot with RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot without RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=false \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot with pi0.5 policy
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/pi05_check_rtc \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with bi_openarm_follower (dual-arm OpenArms) and pi0.5 policy
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=lerobot-data-collection/folding_final \
        --robot.type=bi_openarm_follower \
        --robot.cameras='{left_wrist: {type: opencv, index_or_path: "/dev/video4", width: 1280, height: 720, fps: 30}, base: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30}}' \
        --robot.left_arm_config.port=can0 \
        --robot.left_arm_config.side=left \
        --robot.left_arm_config.can_interface=socketcan \
        --robot.left_arm_config.disable_torque_on_disconnect=true \
        --robot.left_arm_config.max_relative_target=8.0 \
        --robot.right_arm_config.port=can1 \
        --robot.right_arm_config.side=right \
        --robot.right_arm_config.can_interface=socketcan \
        --robot.right_arm_config.disable_torque_on_disconnect=true \
        --robot.right_arm_config.max_relative_target=8.0 \
        --task="Fold the T-shirt properly" \
        --fps=30 \
        --duration=2000 \
        --interpolation_multiplier=3 \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --rtc.max_guidance_weight=5.0 \
        --rtc.prefix_attention_schedule=LINEAR \
        --device=cuda
"""

import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCConfig
from lerobot.processor import (
    AngleUnitProcessorStep,
    ArmSwapProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
)
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.processor.relative_action_processor import to_relative_actions
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    koch_follower,
    so_follower,
    unitree_g1,
)
from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, lift_arms_to_ready
from lerobot.robots.bi_openarm_follower.action_safety import ActionSafetyChecker, ActionSafetyConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SparkJAX-equivalent default thresholds (expressed in degrees, since the
# OpenArm follower's wire format is degrees). Used as the dataclass field
# defaults below so the CLI exposes them without duplicating the numbers.
_DEFAULT_SAFETY = ActionSafetyConfig()


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)
    interpolation_multiplier: int = 1  # Control rate multiplier (1=off, 2=2x, 3=3x)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 30

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Smoothly raise both arms to the training-distribution start pose before
    # handing control to the policy (BiOpenArmFollower only). Our openpi/SparkJAX
    # checkpoint was fine-tuned from an "arms up, table cleared" start pose, so
    # the policy expects to begin there. On by default; it only acts on a
    # BiOpenArmFollower (other robots log a skip), and you can disable it with
    # --lift_arms=false if the arms are already positioned.
    lift_arms: bool = field(
        default=True,
        metadata={"help": "Lift BiOpenArm followers to the training-distribution ready pose before policy hand-off"},
    )

    # ---- Action safety (BiOpenArmFollower only) -----------------------------
    # Hard-abort 3-gate checker (finite-value / absolute-envelope / per-step
    # delta) run on every action just before it reaches the motors. On a
    # violation the actor holds the last good pose and shuts the session down.
    # Critical for RTC: a bad chunk splice landing in the free region shows up
    # as a per-step delta spike, which gate 3 catches before the hardware does.
    # The thresholds are OpenArm/degree-specific, so the checker is only armed
    # for a BiOpenArmFollower (other robots log a skip).
    action_safety_enabled: bool = True
    action_safety_max_joint_delta_deg: float = _DEFAULT_SAFETY.max_joint_delta_deg
    # None (default) disables the per-step delta gate for the gripper: it is a
    # fast, human-teleoperated, near-binary actuator whose large per-step moves
    # are legitimate (60-80 deg/step open/close is normal) and cannot whip the
    # arm. The gripper is still bounded by the finite + absolute-envelope gates.
    action_safety_max_gripper_delta_deg: float | None = _DEFAULT_SAFETY.max_gripper_delta_deg
    action_safety_abs_joint_limit_deg: float = _DEFAULT_SAFETY.abs_joint_limit_deg
    action_safety_abs_gripper_limit_deg: float = _DEFAULT_SAFETY.abs_gripper_limit_deg

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")

        # Validate that robot configuration is provided
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def _reanchor_relative_rtc_prefix(
    prev_actions_absolute: Tensor,
    current_state: Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
    to_model_space_steps: tuple[Any, ...] = (),
) -> Tensor:
    """Convert absolute leftovers into model-space for relative-action RTC policies.

    When a policy uses relative actions, the RTC prefix (leftover actions from
    the previous chunk) is stored in absolute space. Before feeding it back to
    the policy we need to re-express it relative to the *current* robot state
    and then re-normalize.

    The leftover prefix and the anchor state arrive in the **robot wire space**
    (the arm order and angular unit the hardware speaks). For a checkpoint
    converted from openpi/SparkJAX the model's action space differs from the
    wire space (the two arm halves are swapped, and joints are in radians, not
    degrees). ``to_model_space_steps`` are the stamped preprocessor steps that
    run *before* the relative step (``ArmSwapProcessorStep``,
    ``AngleUnitProcessorStep``); re-applying them here lands the prefix in the
    exact same model action space the freshly predicted chunk lives in *before*
    we re-express it relative + normalize. These steps are no-ops for a native
    lerobot checkpoint (arm-swap disabled, angle scale == 1.0), so the original
    behaviour is unchanged for every other policy.
    """
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()

    # Map wire-space leftover + anchor into the model's action space (arm order
    # + angular unit) using the SAME stamped steps the preprocessor applies.
    if to_model_space_steps:
        transition = create_transition(observation={OBS_STATE: state}, action=action_cpu)
        for step in to_model_space_steps:
            transition = step(transition)
        state = transition[TransitionKey.OBSERVATION][OBS_STATE]
        action_cpu = transition[TransitionKey.ACTION]

    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)


def get_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        robot_observation_processor: Processor for raw robot observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        # Only keep .pos joints + camera streams if the policy was trained on positions,
        # not the full pos/vel/torque state the robot exposes.
        observation_features_hw = {
            key: value
            for key, value in robot.observation_features().items()
            if key.endswith(".pos") or isinstance(value, tuple)
        }

        dataset_features = hw_to_dataset_features(observation_features_hw, "observation")
        policy_device = policy.config.device

        # Load preprocessor and postprocessor from pretrained files
        # The stats are embedded in the processor .safetensors files
        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

        relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        # Stamped steps that map the robot wire space -> model space (arm order
        # + angular unit), in the order the preprocessor applies them *before*
        # the relative step. Used to re-anchor the RTC prefix for openpi-
        # converted checkpoints; empty/no-op for native lerobot checkpoints.
        relative_idx = next(
            (i for i, s in enumerate(preprocessor.steps) if s is relative_step),
            len(preprocessor.steps),
        )
        to_model_space_steps = tuple(
            s
            for s in preprocessor.steps[:relative_idx]
            if isinstance(s, (ArmSwapProcessorStep, AngleUnitProcessorStep))
        )
        if relative_step is not None:
            if relative_step.action_names is None:
                cfg_names = getattr(cfg.policy, "action_feature_names", None)
                if cfg_names:
                    relative_step.action_names = list(cfg_names)
                else:
                    relative_step.action_names = [
                        k for k in robot.robot.action_features if k.endswith(".pos")
                    ]
            logger.info("[GET_ACTIONS] Relative actions enabled: will re-anchor RTC prefix")

        def _prepare_obs(raw_obs):
            """Robot observation -> policy-ready feature dict (batched, on device)."""
            obs_processed = robot_observation_processor(raw_obs)
            feat = build_dataset_frame(dataset_features, obs_processed, prefix="observation")
            for name in feat:
                feat[name] = torch.from_numpy(feat[name])
                if "image" in name:
                    feat[name] = feat[name].type(torch.float32) / 255
                    feat[name] = feat[name].permute(2, 0, 1).contiguous()
                feat[name] = feat[name].unsqueeze(0).to(policy_device)
            feat["task"] = [cfg.task]  # Task should be a list, not a string!
            feat["robot_type"] = robot.robot.name if hasattr(robot.robot, "name") else ""
            return feat

        # Warm up the policy before any chunk is used. The first inference pays a
        # large cold-start cost (CUDA init / kernel compile, ~1 s here). If that
        # latency enters the tracker it permanently inflates the inference_delay
        # estimate (LatencyTracker.max() is an all-time max), so RTC builds every
        # subsequent chunk for a ~49-step delay while merge only discards the real
        # ~21 - the mismatch is exactly the violent chunk-seam jump we observed.
        # Running (and discarding) one inference here keeps the cold cost out of
        # the estimate so inference_delay reflects steady-state from the start.
        try:
            _ = policy.predict_action_chunk(
                preprocessor(_prepare_obs(robot.get_observation())),
                inference_delay=0,
                prev_chunk_left_over=None,
            )
            logger.info("[GET_ACTIONS] Warmup inference complete (cold-start cost excluded from latency)")
        except Exception as warmup_exc:  # noqa: BLE001 - warmup is best-effort
            logger.warning("[GET_ACTIONS] Warmup inference failed (continuing): %s", warmup_exc)

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                # Use a windowed percentile, NOT latency_tracker.max(): max() is a
                # monotonic all-time max, so a single cold/slow inference would pin
                # inference_delay high for the whole run and desync RTC's frozen
                # prefix from merge's actual discard (-> chunk-seam jumps).
                inference_latency = latency_tracker.percentile(0.95)
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs_with_policy_features = _prepare_obs(robot.get_observation())

                preproceseded_obs = preprocessor(obs_with_policy_features)

                # Re-anchor leftover actions for relative-action policies.
                # We need the *postprocessed* (absolute) leftover, not the original
                # (normalized/relative) one that get_left_over() returns.
                if (
                    prev_actions is not None
                    and relative_step is not None
                    and OBS_STATE in obs_with_policy_features
                ):
                    with action_queue.lock:
                        if action_queue.queue is not None:
                            prev_actions_abs = action_queue.queue[action_queue.last_index :].clone()
                        else:
                            prev_actions_abs = None
                    if prev_actions_abs is not None and prev_actions_abs.numel() > 0:
                        prev_actions = _reanchor_relative_rtc_prefix(
                            prev_actions_absolute=prev_actions_abs,
                            current_state=obs_with_policy_features[OBS_STATE],
                            relative_step=relative_step,
                            normalizer_step=normalizer_step,
                            policy_device=policy_device,
                            to_model_space_steps=to_model_space_steps,
                        )

                # Generate actions WITH RTC
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                # --- jerk diagnostics (degrees, robot wire space) ----------------
                # Separates the two possible jerk sources so we stop guessing:
                #   within_chunk = max |a[t+1]-a[t]| inside ONE freshly predicted
                #                  chunk  -> model roughness / training wobble.
                #   seam_gap     = |new_chunk[delay] - action robot is executing|
                #                  -> RTC/replace discontinuity at the splice.
                try:
                    pa = postprocessed_actions
                    diag = []
                    if pa.ndim == 2 and pa.shape[0] > 1:
                        step = (pa[1:] - pa[:-1]).abs()
                        per_dim = step.reshape(-1, step.shape[-1]).max(dim=0).values
                        wd = int(per_dim.argmax().item())
                        diag.append(f"within_chunk_max|d|={per_dim[wd].item():.1f}deg@dim{wd}")
                    cur_exec = action_queue.get_processed_left_over()
                    if cur_exec is not None and cur_exec.numel() > 0 and pa.ndim == 2:
                        d = min(int(new_delay), pa.shape[0] - 1)
                        seam = (pa[d] - cur_exec[0]).abs()
                        sd = int(seam.argmax().item())
                        diag.append(f"seam_gap@d{d}={seam[sd].item():.1f}deg@dim{sd}")
                    if diag:
                        logger.info("[GET_ACTIONS][diag] latency=%.0fms delay=%d | %s",
                                    new_latency * 1000, new_delay, " | ".join(diag))
                except Exception:  # noqa: BLE001 - diagnostics must never break control
                    pass

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_keys = [k for k in robot.action_features() if k.endswith(".pos")]

        # Arm the 3-gate action safety checker. The thresholds are OpenArm/
        # degree-specific, so it only runs for a BiOpenArmFollower. We seed its
        # delta baseline from one real observation (post-lift) so step 0 is gated
        # too; that observation is ignored by the checker after the first call.
        safety_checker = None
        seed_obs = None
        if cfg.action_safety_enabled:
            if isinstance(robot.robot, BiOpenArmFollower):
                safety_checker = ActionSafetyChecker(
                    ActionSafetyConfig(
                        enabled=True,
                        max_joint_delta_deg=cfg.action_safety_max_joint_delta_deg,
                        max_gripper_delta_deg=cfg.action_safety_max_gripper_delta_deg,
                        abs_joint_limit_deg=cfg.action_safety_abs_joint_limit_deg,
                        abs_gripper_limit_deg=cfg.action_safety_abs_gripper_limit_deg,
                    )
                )
                try:
                    seed_obs = robot.get_observation()
                except Exception as exc:  # noqa: BLE001 - seeding is best-effort
                    logger.warning("[ACTOR] could not seed safety checker from observation: %s", exc)
                gripper_delta = cfg.action_safety_max_gripper_delta_deg
                logger.info(
                    "[ACTOR] action safety armed (3-gate; joint delta %.2f deg/step, gripper delta %s)",
                    cfg.action_safety_max_joint_delta_deg,
                    "disabled" if gripper_delta is None
                    else f"{gripper_delta:.2f} deg/step",
                )
            else:
                logger.warning(
                    "[ACTOR] action_safety_enabled but robot is %s (not BiOpenArmFollower); "
                    "skipping (thresholds are OpenArm/degree-specific)",
                    type(robot.robot).__name__,
                )

        action_count = 0
        interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        action_interval = interpolator.get_control_interval(cfg.fps)

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            if interpolator.needs_new_action():
                new_action = action_queue.get()
                if new_action is not None:
                    # Gate the RAW policy action at the base (policy) fps so the
                    # per-step delta thresholds keep their 50 Hz meaning no matter
                    # what interpolation_multiplier is. Interpolated sub-steps are
                    # convex combinations of two consecutive gated actions, so they
                    # stay inside the same envelope / delta bounds by construction.
                    if safety_checker is not None:
                        raw = new_action.cpu()
                        raw_dict = {key: raw[i].item() for i, key in enumerate(action_keys)}
                        err = safety_checker.check(raw_dict, seed_obs)
                        if err is not None:
                            logger.error("[ACTOR] %s", err)
                            # Hold the last good pose, then shut the whole session
                            # down so the main thread disconnects (disables torque).
                            if safety_checker.last_action is not None:
                                robot.send_action(robot_action_processor((safety_checker.last_action, None)))
                            shutdown_event.set()
                            break
                    interpolator.add(new_action.cpu())

            action = interpolator.get()
            if action is not None:
                action = action.cpu()
                action_dict = {key: action[i].item() for i, key in enumerate(action_keys)}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)
                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _apply_torch_compile(policy, cfg: RTCDemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    # PI models handle their own compilation
    if policy.type == "pi05" or policy.type == "pi0":
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        # Disable CUDA graphs if requested (prevents tensor aliasing issues)
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration."""

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {cfg.device}")

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
    get_actions_thread = None
    actor_thread = None

    policy_class = get_policy_class(cfg.policy.type)

    # Load config and set compile_model for pi0/pi05 models
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type == "pi05" or cfg.policy.type == "pi0":
        config.compile_model = cfg.use_torch_compile

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_pretrained_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    # Turn on RTC
    policy.config.rtc_config = cfg.rtc

    # Init RTC processort, as by default if RTC disabled in the config
    # The processor won't be created
    policy.init_rtc_processor()

    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile to predict_action_chunk method if enabled
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # Optionally raise the arms to the training-distribution start pose before the
    # policy takes over (see RTCDemoConfig.lift_arms). Done before the control
    # threads start so the policy's first observation is already at the ready pose.
    if cfg.lift_arms:
        if isinstance(robot, BiOpenArmFollower):
            logger.info("Lifting arms to the training-distribution ready pose before policy hand-off")
            lift_arms_to_ready(robot, log_fn=logger.info)
        else:
            logger.warning(
                "lift_arms=True but robot is %s (not BiOpenArmFollower); skipping lift",
                type(robot).__name__,
            )

    robot_wrapper = RobotWrapper(robot)

    # Create robot observation processor
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    # Create action queue for communication between threads
    action_queue = ActionQueue(cfg.rtc)

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(policy, robot_wrapper, robot_observation_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    # Start action executor thread
    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()
    logger.info("Started actor thread")

    logger.info("Started stop by duration thread")

    # Main thread monitors for duration or shutdown
    logger.info(f"Running demo for {cfg.duration} seconds...")
    start_time = time.time()

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(10)

        # Log queue status periodically
        if int(time.time() - start_time) % 5 == 0:
            logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")

        if time.time() - start_time > cfg.duration:
            break

    logger.info("Demo duration reached or shutdown requested")

    # Signal shutdown
    shutdown_event.set()

    # Wait for threads to finish
    if get_actions_thread and get_actions_thread.is_alive():
        logger.info("Waiting for chunk requester thread to finish...")
        get_actions_thread.join()

    if actor_thread and actor_thread.is_alive():
        logger.info("Waiting for action executor thread to finish...")
        actor_thread.join()

    # Cleanup robot
    if robot:
        robot.disconnect()
        logger.info("Robot disconnected")

    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
