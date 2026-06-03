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

"""MuJoCo Jacobian damped-least-squares IK + XR->robot transforms.

Ported (near verbatim) from SparkJAX ``scripts/test_vr_ik.py`` so the lerobot VR
teleoperator solves IK exactly the way the original "start VR" pipeline did:

* :class:`IKSolver` — per-arm damped-least-squares IK on the
  ``openarm_<side>_hand_tcp`` body using ``mj_jacBody`` + joint-limit clamping,
  plus ``set_finger`` for the gripper slide joints.
* XR->robot frame transforms (``xr_pos_to_robot``, ``xr_quat_to_robot``) and the
  quaternion helpers used by the OpenXR pose source (Phase 2).

``mujoco`` is imported lazily inside the solver so this module (and the pure
quaternion helpers below) stay import-safe in environments without the
``openarm-sim`` extra.
"""

import math

import numpy as np

# Body / joint names in the OpenArm MuJoCo model.
LEFT_TCP_BODY = "openarm_left_hand_tcp"
RIGHT_TCP_BODY = "openarm_right_hand_tcp"
LEFT_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]
LEFT_FINGER_JOINTS = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
RIGHT_FINGER_JOINTS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]

FINGER_OPEN_M = 0.044  # finger slide travel (0 = closed, 0.044 = fully open)

# XR -> Robot coordinate mapping quaternion (wxyz).
# XR: X=right, Y=up, Z=back.  Robot: X=forward, Y=left, Z=up.
_Q_XR_TO_ROBOT = np.array([0.5, 0.5, -0.5, -0.5])
_Q_XR_TO_ROBOT_INV = np.array([0.5, -0.5, 0.5, 0.5])


# --------------------------------------------------------------------------- #
# Quaternion helpers (pure numpy/math — no mujoco)
# --------------------------------------------------------------------------- #
def quat_mul(a, b):
    """Multiply two quaternions [w, x, y, z]."""
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def quat_inv(q):
    """Invert (conjugate of unit) quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mat_to_axis_angle(R):
    """Rotation matrix -> axis*angle (rotation vector)."""
    angle = math.acos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.zeros(3)
    return axis / n * angle


def axis_angle_to_quat(axis, angle):
    """Convert axis + angle to quaternion [w, x, y, z]."""
    if abs(angle) < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def xr_pos_to_robot(p):
    """OpenXR position -> robot frame position."""
    return np.array([-p[2], -p[0], p[1]])


# 90-degree rotation about robot Z to align the VR controller with the wrist.
_Q_GRIP_ROT = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), math.radians(90.0))


def xr_quat_to_robot(q_xyzw):
    """OpenXR quaternion (x,y,z,w) -> robot quaternion (w,x,y,z)."""
    qx, qy, qz, qw = q_xyzw
    q_wxyz = np.array([qw, qx, qy, qz])
    q_robot = quat_mul(quat_mul(_Q_XR_TO_ROBOT, q_wxyz), _Q_XR_TO_ROBOT_INV)
    return quat_mul(q_robot, _Q_GRIP_ROT)


# --------------------------------------------------------------------------- #
# IK solver
# --------------------------------------------------------------------------- #
class IKSolver:
    """Per-arm damped-least-squares IK on the OpenArm hand TCP bodies."""

    def __init__(self, model, data, dls_lambda: float = 0.05):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.dls_lambda = float(dls_lambda)

        self.left_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_TCP_BODY)
        self.right_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_TCP_BODY)

        self.joint_ids = {
            "left": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_JOINT_NAMES],
            "right": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_JOINT_NAMES],
        }
        self.qpos_idx = {
            side: [model.jnt_qposadr[j] for j in ids] for side, ids in self.joint_ids.items()
        }
        self.limits_low = {
            side: np.array([model.jnt_range[j][0] for j in ids]) for side, ids in self.joint_ids.items()
        }
        self.limits_high = {
            side: np.array([model.jnt_range[j][1] for j in ids]) for side, ids in self.joint_ids.items()
        }
        self.finger_qpos_idx = {
            "left": [
                model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in LEFT_FINGER_JOINTS
            ],
            "right": [
                model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in RIGHT_FINGER_JOINTS
            ],
        }

    def get_ee_pose(self, side):
        """Current TCP pose for ``side`` as (pos[3], quat[4] wxyz)."""
        mujoco = self._mujoco
        body_id = self.left_tcp_id if side == "left" else self.right_tcp_id
        pos = self.data.xpos[body_id].copy()
        mat = self.data.xmat[body_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return pos, quat

    def solve_ik(self, side, target_pos, target_quat, max_iter=20):
        """Iterate DLS IK toward (target_pos, target_quat); returns joint rad."""
        mujoco = self._mujoco
        body_id = self.left_tcp_id if side == "left" else self.right_tcp_id
        idx = self.qpos_idx[side]
        lo = self.limits_low[side]
        hi = self.limits_high[side]
        jids = self.joint_ids[side]

        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            cur_pos = self.data.xpos[body_id].copy()
            cur_mat = self.data.xmat[body_id].reshape(3, 3)
            pos_err = target_pos - cur_pos

            tgt_mat = np.zeros(9)
            mujoco.mju_quat2Mat(tgt_mat, target_quat)
            tgt_mat = tgt_mat.reshape(3, 3)
            ori_err = mat_to_axis_angle(tgt_mat @ cur_mat.T)

            dx = np.concatenate([pos_err, ori_err])
            if np.linalg.norm(dx) < 1e-4:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

            dof_idx = [self.model.jnt_dofadr[j] for j in jids]
            J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
            JJT = J @ J.T + self.dls_lambda**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, dx)

            for k, qi in enumerate(idx):
                self.data.qpos[qi] = np.clip(self.data.qpos[qi] + dq[k], lo[k], hi[k])

        mujoco.mj_forward(self.model, self.data)
        return np.array([self.data.qpos[i] for i in idx])

    def set_finger(self, side, val):
        """Set the finger slide joints for ``side`` (val in meters, clamped)."""
        val = float(np.clip(val, 0.0, FINGER_OPEN_M))
        for idx in self.finger_qpos_idx[side]:
            self.data.qpos[idx] = val

    def joint_positions(self, side):
        """Read the 7 arm joint angles (rad) for ``side`` in J1..J7 order."""
        return np.array([self.data.qpos[i] for i in self.qpos_idx[side]])
