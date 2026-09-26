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


def apply_base_pose(mujoco, model, data, elbow_bend_deg: float = BASE_ELBOW_BEND_DEG,
                    pose_deg=None) -> None:
    """Put both arms in the base pose.

    ``pose_deg`` sets all seven joints (J1..J7, degrees) and wins outright. The
    left arm is mirrored, because its joint ranges are the mirror of the right's
    (J1 is -80..200 on the right and -200..80 on the left), so the same numbers
    describe the same physical posture on both arms. Out-of-range values are
    clipped to the joint's own limits. Without it only the elbow is set, as
    before, and the rest stay at zero.

    Called by the sim robot and by the VR teleoperator's IK model so the two stay
    in the same configuration and neither starts at the straight-arm singularity.
    """
    if pose_deg is not None:
        vals = [float(v) for v in pose_deg]
        if len(vals) != 7:
            raise ValueError(f"pose_deg needs 7 joint angles (J1..J7), got {len(vals)}")
        for i, v in enumerate(vals):
            ids = {sd: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{sd}_joint{i + 1}")
                   for sd in SIDES}
            if any(j < 0 for j in ids.values()):
                continue
            # Mirror the left arm ONLY on the joints the model itself mirrors
            # (J1 is -80..200 right and -200..80 left, so the same posture is
            # the negated angle). The elbow runs 0..140 on both, so negating it
            # would just clip to 0 and flatten the arm.
            rl, ll = model.jnt_range[ids["right"]], model.jnt_range[ids["left"]]
            mirrored = abs(ll[0] + rl[1]) < 1e-6 and abs(ll[1] + rl[0]) < 1e-6 and abs(rl[0] + rl[1]) > 1e-6
            for sd in SIDES:
                sign = -1.0 if (sd == "left" and mirrored) else 1.0
                lo, hi = model.jnt_range[ids[sd]]
                data.qpos[model.jnt_qposadr[ids[sd]]] = float(np.clip(math.radians(sign * v), lo, hi))
        mujoco.mj_forward(model, data)
        return
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
        self._arm_force: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._tcp_body_id: dict[str, int] = {}
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
        if not self.config.table_collisions:
            # The ARM passes through the table; everything else still rests on
            # it. Simply clearing the table's contact bits drops every bar on
            # the floor (measured: 39 cm in 0.6 s), which makes the scene
            # useless. Instead move the table to contact group 2 and add that
            # group to the free bodies (the bars and the cube), so table<->bar
            # still collides while table<->arm does not.
            tables, movers = [], 0
            for g in range(self._model.ngeom):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                body = self._model.geom_bodyid[g]
                if name.startswith("table"):
                    self._model.geom_contype[g] = 2
                    self._model.geom_conaffinity[g] = 2
                    tables.append(name)
                    continue
                free = any(
                    self._model.jnt_type[self._model.body_jntadr[body] + j] == mujoco.mjtJoint.mjJNT_FREE
                    for j in range(self._model.body_jntnum[body])
                )
                if free:
                    self._model.geom_contype[g] |= 2
                    self._model.geom_conaffinity[g] |= 2
                    movers += 1
            logger.info("arm passes through the table (%d table geoms, %d free-body geoms still collide with it)",
                        len(tables), movers)
        if self.config.arm_armature is not None:
            for side in ("left", "right"):
                for i, arm in enumerate(self.config.arm_armature[:7]):
                    jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i + 1}")
                    if jid >= 0:
                        self._model.dof_armature[self._model.jnt_dofadr[jid]] = float(arm)
            logger.info("arm joint armature set to %s", self.config.arm_armature)
        mujoco.mj_forward(self._model, self._data)
        if self.config.start_pose_deg is not None:
            apply_base_pose(mujoco, self._model, self._data, pose_deg=self.config.start_pose_deg)
            logger.info("start pose (J1..J7, deg, left mirrored): %s", list(self.config.start_pose_deg))
        elif self.config.start_elbow_bend_deg:
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

    def _update_contact_force(self, mujoco, arm_targets) -> None:
        """Estimate the force each arm is pressing with, in newtons at the gripper.

        There is no force sensor, so this is what a real arm would give you:
        how far each joint has been pushed off its commanded angle, times that
        joint's stiffness.

            tau_applied = kp * (target - q)          deflection x stiffness
            tau_needed  = M(q) qacc + C(q,qd) + g(q) what it takes to move the
                                                     arm through free space
            tau_ext     = tau_applied - tau_needed   what the world is pushing
                                                     back with

        Subtracting tau_needed is the gravity/motion compensation: hold the arm
        out in still air and the deflection is entirely the weight of the arm,
        which must read zero. Whatever is left is external. That joint-space
        residual is then resolved through the gripper Jacobian into a force at
        the tip, so the number is newtons of push and not a pile of torques.
        Low-passed because contact in a stiff sim is noisy tick to tick.
        """
        m, d = self._model, self._data
        if not self._arm_ctrl:
            return
        from .viewer_keys import set_arm_force

        # what it would take to be doing exactly this motion with no contact
        bias = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 1, bias)          # 1 = include qacc, so this covers
        bias += d.qfrc_passive * -1.0         # damping/friction the model applies
        for side in ("left", "right"):
            dofs, tau_ext = [], []
            for motor in ARM_JOINT_NAMES:
                key = (side, motor)
                if key not in self._arm_ctrl or key not in arm_targets:
                    return
                info = self._arm_ctrl[key]
                applied = info["kp"] * (arm_targets[key] - d.qpos[info["qadr"]])
                applied = float(np.clip(applied, -info["frange"], info["frange"]))
                dofs.append(info["dadr"])
                tau_ext.append(applied - float(bias[info["dadr"]]))
            tau_ext = np.asarray(tau_ext, float)
            if side not in self._tcp_body_id:
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_hand_tcp")
                self._tcp_body_id[side] = int(bid)
            body = self._tcp_body_id[side]
            if body < 0:
                continue
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, None, body)
            J = jacp[:, dofs]                  # 3 x 7, gripper tip
            # tau = J^T F  ->  F = (J J^T + lam I)^-1 J tau
            A = J @ J.T + (0.05 ** 2) * np.eye(3)
            f = np.linalg.solve(A, J @ tau_ext)
            mag = float(np.linalg.norm(f))
            prev = self._arm_force.get(side, 0.0)
            sm = prev + 0.25 * (mag - prev)     # ~4 tick time constant
            self._arm_force[side] = sm
            set_arm_force(side, sm)

    def _draw_target_markers(self, mujoco) -> None:
        """x (red) / y (green) / z (blue) triad at each commanded gripper pose."""
        from .viewer_keys import draw_markers_into

        scn = self._viewer.user_scn
        scn.ngeom = 0
        draw_markers_into(scn, mujoco, np)

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

        self._update_contact_force(mujoco, arm_targets)

        if self._viewer is not None:
            self._apply_viewer_camera_cycles()
            self._draw_target_markers(mujoco)
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
