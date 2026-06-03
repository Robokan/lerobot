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

"""VR motion-capture teleoperator for the simulated bimanual OpenArm.

Reproduces SparkJAX's "start VR" behaviour inside lerobot: a pose source
provides per-hand end-effector targets, which are solved to joint angles via
damped-least-squares IK on a lightweight FK/IK-only MuJoCo model, then emitted
as the same 16 right-first ``<side>_<motor>.pos`` (degrees) action keys the
:class:`MujocoBiOpenArm` robot consumes. Because lerobot uses identity
processors by default, ``teleop.action_features`` matches
``robot.action_features`` exactly, so the record loop drives the sim and the
recorded dataset shares the real chocolate-dataset schema.

IK lives here (not in the robot) because the recorded ``action`` must be joint
positions; IK therefore runs before the action is recorded, exactly as SparkJAX
does (arms just track joint targets).
"""

import logging
import os

import numpy as np

from lerobot.robots.mujoco_bi_openarm import gripper_m_to_deg
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_vr_mocap import VRMocapConfig

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
MOTOR_NAMES = ARM_JOINT_NAMES + ["gripper"]
SIDES = ["right", "left"]  # right-first, to match the robot

_RAD2DEG = 180.0 / np.pi


class VRMocap(Teleoperator):
    """VR mocap -> IK -> joint-position teleoperator (16 right-first *.pos deg)."""

    config_class = VRMocapConfig
    name = "vr_mocap"

    def __init__(self, config: VRMocapConfig):
        super().__init__(config)
        self.config = config

        self._model = None
        self._data = None
        self._ik = None
        self._source = None
        self._grip_m: dict[str, float] = {s: 0.0 for s in SIDES}
        self._connected = False

    @property
    def action_features(self) -> dict[str, type]:
        # Right first, then left — must equal MujocoBiOpenArm.action_features.
        features: dict[str, type] = {}
        for side in SIDES:
            for motor in MOTOR_NAMES:
                features[f"{side}_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import mujoco

        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        from .ik import IKSolver

        model_path = os.path.expanduser(self.config.model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MuJoCo IK model not found: {model_path}")
        logger.info("Loading IK MuJoCo model: %s", model_path)
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._ik = IKSolver(self._model, self._data, dls_lambda=self.config.dls_lambda)

        self._source = self._make_source()
        initial_ee = {s: self._ik.get_ee_pose(s) for s in SIDES}
        self._source.reset(initial_ee)
        self._source.start()

        self._connected = True
        logger.info("%s connected (driver=%s).", self, self.config.driver)

    def _make_source(self):
        driver = self.config.driver
        if driver == "scripted":
            from .pose_source import ScriptedPoseSource

            return ScriptedPoseSource(
                amplitude=self.config.scripted_amplitude,
                period_s=self.config.scripted_period_s,
                fps=float(self.config.vr_hz),
            )
        if driver == "keyboard":
            from .pose_source import KeyboardPoseSource

            return KeyboardPoseSource()
        if driver == "openxr":
            from .openxr_pose_source import OpenXRPoseSource

            return OpenXRPoseSource(vr_hz=self.config.vr_hz)
        raise ValueError(
            f"Unknown VRMocap driver '{driver}'. Expected 'scripted', 'keyboard', or 'openxr'."
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        ik = self._ik
        current_ee = {s: ik.get_ee_pose(s) for s in SIDES}
        targets = self._source.get_targets(current_ee)

        for side in SIDES:
            tgt = targets.get(side)
            if tgt is None or not tgt.active:
                continue  # hold: leave IK qpos (and thus joint output) unchanged
            ik.solve_ik(side, tgt.pos, tgt.quat, max_iter=self.config.max_iter)
            ik.set_finger(side, tgt.gripper_m)
            self._grip_m[side] = float(tgt.gripper_m)

        action: dict[str, float] = {}
        for side in SIDES:
            joints_rad = ik.joint_positions(side)
            for i, motor in enumerate(ARM_JOINT_NAMES):
                action[f"{side}_{motor}.pos"] = float(joints_rad[i]) * _RAD2DEG
            action[f"{side}_gripper.pos"] = gripper_m_to_deg(self._grip_m[side])
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # No haptics for the sim teleoperator.
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:  # noqa: BLE001
                logger.debug("pose source stop failed", exc_info=True)
        self._source = None
        self._ik = None
        self._data = None
        self._model = None
        self._connected = False
        logger.info("%s disconnected.", self)
