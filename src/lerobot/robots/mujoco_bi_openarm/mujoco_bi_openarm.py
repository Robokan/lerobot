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

"""MuJoCo-simulated bimanual OpenArm follower.

A drop-in stand-in for :class:`BiOpenArmFollower` that runs a MuJoCo dynamic sim
instead of the CAN hardware, so the standard ``lerobot-record`` /
``lerobot-teleoperate`` loop can drive simulated arms and record datasets with
the *exact same* 16-key, right-first, ``<side>_<motor>.pos`` (degrees) feature
contract as the real robot.

Control: the model's arm joints use direct-drive ``motor`` (torque) actuators,
so this robot applies a Python PD law each substep (``tau = kp*(target - q) -
kd*qdot``, clamped to the model force range). Finger actuators that are
position-servo type are commanded directly with the target opening (meters);
finger actuators that are torque type get the same PD law.

Layout/unit conversions mirror the real follower wire format:
* degrees <-> radians for all arm joints,
* gripper degrees <-> finger opening meters
  (``GRIPPER_OPEN_DEG`` -> ``FINGER_OPEN_M``, ``0 deg`` -> ``0 m``),
* right-first wire ordering: the dict keys carry the ``right_``/``left_`` prefix,
  so each key maps directly to its per-side MuJoCo joint (no half-vector swap
  needed — that is the dict-keyed equivalent of ``replay_episode._swap_halves``).
"""

import atexit
import logging
import math
import os
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.mujoco import MujocoCamera, MujocoCameraConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_mujoco_bi_openarm import MujocoBiOpenArmConfig

logger = logging.getLogger(__name__)

# Logical motor names per arm, in the order the real follower exposes them.
MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]

# Base pose: the model's all-zeros pose is a fully straight arm, which is a
# kinematic singularity -- the Jacobian loses rank there, so the IK has no
# authority in whole directions (measured: zero roll authority about the tool
# axis from every joint). It is also exactly on J4's lower stop, since the elbow
# range is 0..140 deg, so the elbow can only bend one way out of it.
#
# Pulling the wrist straight back from a straight arm physically requires bending
# the elbow first, and a solver sitting at the singularity cannot discover that.
# So both the sim robot and the teleop's IK model start here instead, and the
# rest-pose bias pulls back toward it.
BASE_ELBOW_BEND_DEG = 90.0

ARM_JOINT_NAMES = MOTOR_NAMES[:7]
# Right first, then left — matches BiOpenArmFollower / OpenArmMini ordering.
SIDES = ["right", "left"]

_RAD2DEG = 180.0 / np.pi
_DEG2RAD = np.pi / 180.0

# Gripper unit conversion (wire degrees <-> finger opening meters). The hardware
# follower opens the gripper to ~-165 deg for a full open and ~0 deg closed;
# the MuJoCo finger slide travels 0 .. 0.044 m. Linear map through the origin.
GRIPPER_OPEN_DEG = -165.0
FINGER_OPEN_M = 0.044


def apply_base_pose(mujoco, model, data, elbow_bend_deg: float = BASE_ELBOW_BEND_DEG) -> None:
    """Put both arms in the base pose: zeros, with the elbow bent off the stop.

    Called by the sim robot and by the VR teleoperator's IK model so the two stay
    in the same configuration and neither starts at the straight-arm singularity.
    """
    bend = math.radians(elbow_bend_deg)
    for side in SIDES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint4")
        if jid < 0:
            raise ValueError(f"Joint 'openarm_{side}_joint4' not found in MuJoCo model.")
        lo, hi = model.jnt_range[jid]
        data.qpos[model.jnt_qposadr[jid]] = float(np.clip(bend, lo, hi))
    mujoco.mj_forward(model, data)


def gripper_deg_to_m(deg: float) -> float:
    """Wire gripper degrees -> MuJoCo finger opening (meters), clamped to travel."""
    m = deg / GRIPPER_OPEN_DEG * FINGER_OPEN_M
    return float(np.clip(m, 0.0, FINGER_OPEN_M))


def gripper_m_to_deg(m: float) -> float:
    """MuJoCo finger opening (meters) -> wire gripper degrees."""
    return float(m / FINGER_OPEN_M * GRIPPER_OPEN_DEG)


class MujocoBiOpenArm(Robot):
    """Bimanual OpenArm follower simulated in MuJoCo."""

    config_class = MujocoBiOpenArmConfig
    name = "mujoco_bi_openarm"

    def __init__(self, config: MujocoBiOpenArmConfig):
        super().__init__(config)
        self.config = config

        self._model = None
        self._data = None
        self._substeps = 1
        self._viewer = None
        # Set when a viewer has been opened; used to skip the broken MuJoCo 3.9
        # aarch64 GL teardown that SIGSEGVs at interpreter exit (see disconnect).
        self._viewer_was_opened = False
        # Viewer fixed-camera cycle index (0 = free orbit). Advanced by keyboard `c`.
        self._viewer_cam_idx = 0

        # Per-(side, joint) actuator/qpos book-keeping, filled at connect().
        self._arm_ctrl: dict[tuple[str, str], dict[str, Any]] = {}
        # Per-(side) gripper: finger actuators + a representative finger qpos addr.
        self._gripper_ctrl: dict[str, dict[str, Any]] = {}

        # Build camera objects up front (NOT connected yet) so `len(self.cameras)`
        # is correct before connect(). lerobot-record sizes its async image-writer
        # thread pool from len(robot.cameras) *before* calling connect(); if this
        # were empty there, image writes would fall back to the synchronous main
        # thread and throttle the record loop. The shared live model/data is bound
        # in connect() via MujocoCamera.bind_shared_model().
        self.cameras: dict[str, MujocoCamera] = {}
        for key, cfg in config.cameras.items():
            if not isinstance(cfg, MujocoCameraConfig):
                raise TypeError(
                    f"MujocoBiOpenArm camera '{key}' must be a MujocoCameraConfig, got {type(cfg)}."
                )
            self.cameras[key] = MujocoCamera(cfg)
        self._connected = False

    # ------------------------------------------------------------------ features
    @property
    def _motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for side in SIDES:
            for motor in MOTOR_NAMES:
                features[f"{side}_{motor}.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.config.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ lifecycle
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import mujoco

        if "MUJOCO_GL" not in os.environ:
            # egl renders offscreen only, so a viewer window needs glx. glx also
            # serves the offscreen camera renders, so one backend covers both.
            os.environ["MUJOCO_GL"] = "glx" if self.config.viewer else "egl"

        model_path = str(Path(self.config.model_path).expanduser())
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
        logger.info("Loading MuJoCo model: %s", model_path)
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        if self.config.disable_collisions:
            self._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            logger.info("MuJoCo contacts disabled (disable_collisions=True).")
        mujoco.mj_forward(self._model, self._data)
        if self.config.start_elbow_bend_deg:
            apply_base_pose(mujoco, self._model, self._data, self.config.start_elbow_bend_deg)

        # Substeps so one send_action advances ~ 1/fps of sim time.
        if self.config.sim_substeps is not None:
            self._substeps = max(1, int(self.config.sim_substeps))
        else:
            self._substeps = max(1, round((1.0 / self.config.fps) / self._model.opt.timestep))
        logger.info(
            "Sim timestep=%.4fs, control fps=%d -> %d substeps/step",
            self._model.opt.timestep, self.config.fps, self._substeps,
        )

        self._build_index_maps(mujoco)
        self._build_cameras()

        if self.config.viewer:
            self._open_viewer()

        logger.info("%s connected.", self)
        self._connected = True

    def _open_viewer(self) -> None:
        """Open the passive viewer, framed on the arms like the SparkJAX rig."""
        import mujoco
        import mujoco.viewer

        from .viewer_keys import push_glfw_key

        self._viewer = mujoco.viewer.launch_passive(
            self._model,
            self._data,
            show_left_ui=False,
            show_right_ui=False,
            # Forward keys from the viewer window to keyboard teleop (stdin is
            # idle while the MuJoCo window has focus).
            key_callback=push_glfw_key,
        )
        cam = self._viewer.cam
        cam.azimuth = 0.0
        cam.elevation = -9.0
        cam.lookat[2] = 0.4
        cam.distance = 1.0
        # Draw camera frustums so ego (torso) + wrist mounts are visible in free
        # orbit; toggle with the viewer's usual camera-vis shortcut if needed.
        self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        self._viewer_was_opened = True
        self._viewer_cam_idx = 0
        logger.info(
            "MuJoCo viewer opened (close the window or Ctrl-C to stop). "
            "Press 'c' to cycle ego / right_wrist / left_wrist / free."
        )

    # Fixed cameras in the OpenArm scene, matching the VR headset toggles
    # (ego / right / left) plus the default free orbit view.
    _VIEWER_CAM_CYCLE = ("free", "ego_camera", "right_wrist_camera", "left_wrist_camera")

    def _apply_viewer_camera_cycles(self) -> None:
        """Honor pending `c` key presses: cycle the passive viewer camera."""
        if self._viewer is None:
            return
        from .viewer_keys import drain_camera_cycles

        n = drain_camera_cycles()
        if n <= 0:
            return
        import mujoco

        # One step per control tick even if key-repeat queued several `c`s.
        self._viewer_cam_idx = (self._viewer_cam_idx + 1) % len(self._VIEWER_CAM_CYCLE)
        name = self._VIEWER_CAM_CYCLE[self._viewer_cam_idx]
        if name == "free":
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            logger.info("Viewer camera -> free orbit")
            return
        cid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            logger.warning("Viewer camera '%s' not found in model", name)
            return
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._viewer.cam.fixedcamid = int(cid)
        logger.info("Viewer camera -> %s", name)

    def _build_index_maps(self, mujoco) -> None:
        """Resolve MuJoCo joint/actuator ids for the 16 logical DOF."""
        m = self._model

        def joint_qpos_adr(jname: str) -> tuple[int, int, int]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not found in MuJoCo model.")
            return int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]), int(jid)

        def act_id(aname: str) -> int:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise ValueError(f"Actuator '{aname}' not found in MuJoCo model.")
            return int(aid)

        affine = int(mujoco.mjtBias.mjBIAS_AFFINE)

        for side in SIDES:
            for i, motor in enumerate(ARM_JOINT_NAMES):
                qadr, dadr, jid = joint_qpos_adr(f"openarm_{side}_joint{i + 1}")
                aid = act_id(f"{side}_joint{i + 1}_ctrl")
                self._arm_ctrl[(side, motor)] = {
                    "qadr": qadr,
                    "dadr": dadr,
                    "aid": aid,
                    "kp": self.config.arm_kp[i],
                    "kd": self.config.arm_kd[i],
                    "frange": float(m.actuator_forcerange[aid][1]),
                    "qrange": (float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])),
                }

            fingers = []
            for fi in (1, 2):
                qadr, dadr, _ = joint_qpos_adr(f"openarm_{side}_finger_joint{fi}")
                aid = act_id(f"{side}_finger{fi}_ctrl")
                fingers.append({
                    "qadr": qadr,
                    "dadr": dadr,
                    "aid": aid,
                    "frange": float(m.actuator_forcerange[aid][1]),
                    "is_position": int(m.actuator_biastype[aid]) == affine,
                })
            self._gripper_ctrl[side] = {
                "fingers": fingers,
                "read_qadr": fingers[0]["qadr"],  # both fingers are tied by an equality constraint
            }

    def _build_cameras(self) -> None:
        # Bind the freshly loaded live model/data into the pre-built camera
        # objects, then connect them (creates each offscreen renderer).
        for cam in self.cameras.values():
            cam.bind_shared_model(self._model, self._data)
            cam.connect(warmup=True)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # Nothing to calibrate in sim.
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    # ------------------------------------------------------------------ I/O
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict[str, Any] = {}
        d = self._data
        for side in SIDES:
            for motor in ARM_JOINT_NAMES:
                qadr = self._arm_ctrl[(side, motor)]["qadr"]
                obs[f"{side}_{motor}.pos"] = float(d.qpos[qadr]) * _RAD2DEG
            m = float(d.qpos[self._gripper_ctrl[side]["read_qadr"]])
            obs[f"{side}_gripper.pos"] = gripper_m_to_deg(m)

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        import mujoco

        goal_deg = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}

        # Resolve per-actuator targets once (radians / meters).
        arm_targets: dict[tuple[str, str], float] = {}
        grip_targets: dict[str, float] = {}
        for side in SIDES:
            for motor in ARM_JOINT_NAMES:
                key = f"{side}_{motor}"
                if key in goal_deg:
                    info = self._arm_ctrl[(side, motor)]
                    tgt = goal_deg[key] * _DEG2RAD
                    arm_targets[(side, motor)] = float(np.clip(tgt, info["qrange"][0], info["qrange"][1]))
            gkey = f"{side}_gripper"
            if gkey in goal_deg:
                grip_targets[side] = gripper_deg_to_m(goal_deg[gkey])

        d = self._data
        from .viewer_keys import take_teleport

        if take_teleport():
            # h in teleop: set the joints to the commanded targets outright and
            # kill their velocity -- a jump to the default pose, not a swing
            for (side, motor), tgt in arm_targets.items():
                info = self._arm_ctrl[(side, motor)]
                d.qpos[info["qadr"]] = tgt
                d.qvel[info["dadr"]] = 0.0
            for side, tgt_m in grip_targets.items():
                for f in self._gripper_ctrl[side]["fingers"]:
                    if "qadr" in f:
                        d.qpos[f["qadr"]] = tgt_m
                        d.qvel[f["dadr"]] = 0.0
            mujoco.mj_forward(self._model, d)
        for _ in range(self._substeps):
            for (side, motor), tgt in arm_targets.items():
                info = self._arm_ctrl[(side, motor)]
                q = d.qpos[info["qadr"]]
                qd = d.qvel[info["dadr"]]
                tau = info["kp"] * (tgt - q) - info["kd"] * qd
                d.ctrl[info["aid"]] = float(np.clip(tau, -info["frange"], info["frange"]))
            for side, tgt_m in grip_targets.items():
                for f in self._gripper_ctrl[side]["fingers"]:
                    if f["is_position"]:
                        d.ctrl[f["aid"]] = tgt_m
                    else:
                        q = d.qpos[f["qadr"]]
                        qd = d.qvel[f["dadr"]]
                        tau = self.config.finger_kp * (tgt_m - q) - self.config.finger_kd * qd
                        d.ctrl[f["aid"]] = float(np.clip(tau, -f["frange"], f["frange"]))
            mujoco.mj_step(self._model, d)

        if self._viewer is not None:
            self._apply_viewer_camera_cycles()
            if self._viewer.is_running():
                self._viewer.sync()
            else:
                # Window closed by the user — drop the handle so we don't keep
                # syncing a dead viewer (can SIGSEGV on some MuJoCo builds).
                try:
                    self._viewer.close()
                except Exception:  # noqa: BLE001
                    logger.debug("viewer close after is_running=False failed", exc_info=True)
                self._viewer = None

        # Echo the joint commands actually applied (degrees), like the real robot.
        sent: dict[str, float] = {}
        for (side, motor), tgt in arm_targets.items():
            sent[f"{side}_{motor}.pos"] = tgt * _RAD2DEG
        for side, tgt_m in grip_targets.items():
            sent[f"{side}_gripper.pos"] = gripper_m_to_deg(tgt_m)
        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                logger.debug("camera disconnect failed", exc_info=True)
        if self._viewer is not None:
            try:
                if self._viewer.is_running():
                    self._viewer.close()
            except Exception:  # noqa: BLE001
                logger.debug("viewer close failed", exc_info=True)
            self._viewer = None
        # Keep the camera objects (re-bound on reconnect); just drop the sim.
        self._data = None
        self._model = None
        self._connected = False

        # MuJoCo 3.9.0 on this aarch64 box SIGSEGVs in GL teardown at interpreter
        # exit whenever a viewer was opened (reproducible with plain mujoco, no
        # lerobot). Apport then writes ~1GB crash dumps per run. Skip remaining
        # atexit/GL destructors with os._exit after a clean disconnect.
        if self._viewer_was_opened and os.environ.get("MUJOCO_SAFE_EXIT_AFTER_VIEWER", "0") == "1":
            atexit.register(os._exit, 0)
        logger.info("%s disconnected.", self)
