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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import CAN_CMD_REFRESH, CAN_PARAM_ID
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArmFollowerConfig,
)

logger = logging.getLogger(__name__)


def _allow_only_refresh(bus: DamiaoMotorsBus) -> None:
    """Make the bus raise instead of sending anything but a status refresh (dry run)."""
    send = bus.canbus.send

    def guarded(msg, *args, **kwargs):
        data = bytes(msg.data)
        if not (msg.arbitration_id == CAN_PARAM_ID and len(data) >= 3 and data[2] == CAN_CMD_REFRESH):
            raise RuntimeError(
                f"dry_run: blocked CAN frame id=0x{msg.arbitration_id:X} data={data.hex()} "
                "(only status refresh requests are allowed)"
            )
        return send(msg, *args, **kwargs)

    bus.canbus.send = guarded


class OpenArmFollower(Robot):
    """
    OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
    The arm uses Damiao motors in MIT control mode.
    """

    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        # Arm motors
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        if config.side is not None:
            if config.side == "left":
                config.joint_limits = LEFT_DEFAULT_JOINTS_LIMITS
            elif config.side == "right":
                config.joint_limits = RIGHT_DEFAULT_JOINTS_LIMITS
            else:
                raise ValueError(
                    "config.side must be either 'left', 'right' (for default values) or 'None' (for CLI values)"
                )
        else:
            logger.info(
                "Set config.side to either 'left' or 'right' to use pre-configured values for joint limits."
            )
        if config.joint_limits_override:
            config.joint_limits = {
                **config.joint_limits,
                **{k: tuple(v) for k, v in config.joint_limits_override.items()},
            }
        logger.info(f"Values used for joint limits: {config.joint_limits}.")
        if config.zero_offsets:
            logger.info(f"Zero offsets (deg, subtracted from readings): {config.zero_offsets}.")

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        We assume that at connection time, the arms are in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """

        if self.config.dry_run:
            # The handshake sends ENABLE, calibrate() can re-zero the motors and
            # configure()/enable_torque() write to them, so none of them run.
            logger.info(f"Connecting arm on {self.config.port} (DRY RUN: read-only, motors stay off)...")
            self.bus.connect(handshake=False)
            _allow_only_refresh(self.bus)
            for cam in self.cameras.values():
                cam.connect()
            logger.info(f"{self} connected (dry run).")
            return

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        self.bus.enable_torque()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms robot.

        The calibration procedure:
        1. Disable torque
        2. Ask user to position arms in hanging position with grippers closed
        3. Set this as zero position
        4. Record range of motion for each joint
        5. Save calibration
        """
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        # Step 1: Set zero position
        input(
            "\nCalibration: Set Zero Position)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # Set current position as zero for all motors
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° for safety by default for all joints")
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        # TODO(Steven, Pepijn): Slightly different from what it is happening in the leader
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get current observation from robot including position, velocity, and torque.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle
        instead of 3 separate reads.
        """
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        states = self.bus.sync_read_all_states()
        # Raw readings of this tick, reused by send_action's step cap instead of
        # a second full CAN read (which added latency to every command).
        self._obs_raw = {m: s.get("position", 0.0) for m, s in states.items()}
        self._obs_raw_t = time.perf_counter()

        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0) - self.config.zero_offsets.get(motor, 0.0)
            if self.config.use_velocity_and_torque:
                obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
                obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """
        Send action command to robot.

        The action magnitude may be clipped based on safety limits.

        Args:
            action: Dictionary with motor positions (e.g., "joint_1.pos", "joint_2.pos")
            custom_kp: Optional custom kp gains per motor (e.g., {"joint_1": 120.0, "joint_2": 150.0})
            custom_kd: Optional custom kd gains per motor (e.g., {"joint_1": 1.5, "joint_2": 2.0})

        Returns:
            The action actually sent (potentially clipped)
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Apply joint limit clipping to arm
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                goal_pos[motor_name] = clipped_position

        if self.config.dry_run:
            self._report_dry_run(goal_pos)

        # Limits above are in the true joint frame; the motors take raw readings.
        offsets = self.config.zero_offsets
        goal_pos = {m: p + offsets.get(m, 0.0) for m, p in goal_pos.items()}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        # (Not in a dry run: the arm never moves there, so the cap would clamp and
        # warn on every tick; _report_dry_run already showed the uncapped command.)
        requested = goal_pos
        if self.config.max_relative_target is not None and not self.config.dry_run:
            goal_pos = self._cap_step(goal_pos, self._present_raw())

        sent = {f"{m}.pos": p - offsets.get(m, 0.0) for m, p in goal_pos.items()}
        if self.config.trace_path:
            self._trace(requested, goal_pos)
        if self.config.dry_run:
            return sent

        # TODO(Steven, Pepijn): Refactor writing
        # Motor name to index mapping for gains
        motor_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }

        # Velocity feedforward: the MIT law is kp*(q_target - q) + kd*(v_target - v).
        # With v_target = 0 the damping brakes against every move, so a target that
        # steps each tick gives stop-go motion (the arm buzzed at the loop rate).
        # Feeding the target's own velocity lets kd smooth the motion instead.
        velocity_ff: dict[str, float] = {}
        now = time.perf_counter()
        prev = getattr(self, "_prev_goal_raw", None)
        if self.config.velocity_feedforward and prev is not None:
            dt = now - self._prev_goal_t
            if 0.0 < dt < 0.2:  # skip after a pause: a stale previous target is no guide
                vmax = self.config.velocity_feedforward_max
                velocity_ff = {
                    m: max(-vmax, min(vmax, (g - prev[m]) / dt)) for m, g in goal_pos.items() if m in prev
                }
        self._prev_goal_raw, self._prev_goal_t = dict(goal_pos), now

        # Use batch MIT control for arm (sends all commands, then collects responses)
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            idx = motor_index.get(motor_name, 0)
            # Use custom gains if provided, otherwise use config defaults
            if custom_kp is not None and motor_name in custom_kp:
                kp = custom_kp[motor_name]
            else:
                kp = (
                    self.config.position_kp[idx]
                    if isinstance(self.config.position_kp, list)
                    else self.config.position_kp
                )
            if custom_kd is not None and motor_name in custom_kd:
                kd = custom_kd[motor_name]
            else:
                kd = (
                    self.config.position_kd[idx]
                    if isinstance(self.config.position_kd, list)
                    else self.config.position_kd
                )
            commands[motor_name] = (kp, kd, position_degrees, velocity_ff.get(motor_name, 0.0), 0.0)

        self.bus._mit_control_batch(commands)

        return sent

    def _present_raw(self) -> dict[str, float]:
        """Raw motor positions: this tick's observation if fresh, else a new read."""
        if getattr(self, "_obs_raw", None) is not None and time.perf_counter() - self._obs_raw_t < 0.05:
            return self._obs_raw
        return self.bus.sync_read("Present_Position")

    def _cap_step(self, goal_raw: dict[str, float], present_raw: dict[str, float]) -> dict[str, float]:
        """max_relative_target per motor (deg per tick). Logs one summary a second, not every tick."""
        cap = self.config.max_relative_target
        if isinstance(cap, dict) and set(goal_raw) - set(cap):
            raise ValueError(f"max_relative_target has no cap for {sorted(set(goal_raw) - set(cap))}")
        out, hit = {}, {}
        for m, g in goal_raw.items():
            c = float(cap[m] if isinstance(cap, dict) else cap)
            p = present_raw[m]
            out[m] = min(max(g, p - c), p + c)
            if abs(out[m] - g) > 1e-4:
                hit[m] = g - out[m]
        self._cap_hits = getattr(self, "_cap_hits", 0) + (1 if hit else 0)
        now = time.perf_counter()
        if hit and now - getattr(self, "_cap_log_t", 0.0) >= 1.0:
            self._cap_log_t = now
            cells = " ".join(f"{m.replace('joint_', 'j')}{d:+.0f}" for m, d in hit.items())
            logger.warning(
                f"[{self.id}] step cap active on {self._cap_hits} ticks this second; "
                f"command beyond cap (deg): {cells}"
            )
            self._cap_hits = 0
        return out

    def _trace(self, requested_raw: dict[str, float], sent_raw: dict[str, float]) -> None:
        """One CSV row per tick, true frame: requested, sent (after the cap), actual."""
        offsets = self.config.zero_offsets
        if getattr(self, "_trace_fh", None) is None:
            self._trace_fh = open(self.config.trace_path, "w")  # noqa: SIM115 - held open across ticks, closed in disconnect()
            motors = list(self.bus.motors)
            self._trace_fh.write(
                ",".join(["t"] + [f"{k}_{m}" for k in ("req", "sent", "act") for m in motors]) + "\n"
            )
            self._trace_t0 = time.perf_counter()
        actual = {m: s["position"] for m, s in self.bus._last_known_states.items()}
        row = [f"{time.perf_counter() - self._trace_t0:.4f}"]
        for src in (requested_raw, sent_raw, actual):
            row += [f"{src[m] - offsets.get(m, 0.0):.3f}" if m in src else "" for m in self.bus.motors]
        self._trace_fh.write(",".join(row) + "\n")

    def _report_dry_run(self, goal_true: dict[str, float]) -> None:
        """Once a second, log how far the requested command is from the arm (true frame)."""
        now = time.perf_counter()
        if now - getattr(self, "_dry_report_t", 0.0) < 1.0:
            return
        self._dry_report_t = now
        offsets = self.config.zero_offsets
        actual = {m: s["position"] - offsets.get(m, 0.0) for m, s in self.bus._last_known_states.items()}
        diff = {m: goal_true[m] - actual[m] for m in goal_true if m in actual}
        worst = max(diff, key=lambda m: abs(diff[m]))
        cells = " ".join(f"{m.replace('joint_', 'j')}{d:+6.1f}" for m, d in diff.items())
        logger.info(f"[dry run {self.id}] command - actual (deg): {cells} | max {worst} {diff[worst]:+.1f}")

    @check_if_not_connected
    def disconnect(self):
        """Disconnect from robot."""

        # Disconnect CAN bus
        self.bus.disconnect(self.config.disable_torque_on_disconnect and not self.config.dry_run)
        if getattr(self, "_trace_fh", None) is not None:
            self._trace_fh.close()
            self._trace_fh = None

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
