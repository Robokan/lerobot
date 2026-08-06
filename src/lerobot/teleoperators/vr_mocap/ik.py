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

# Per-joint willingness to move, J1..J7 (shoulder -> wrist). The 6-DoF pose task
# is still solved either way; these only decide *which* joints do the work, so a
# wrist roll is absorbed distally instead of being smeared across the shoulder.
# Mirrors the arm's own stiffness split (shoulder/elbow kp=240, wrist kp=24-31).
DEFAULT_JOINT_WEIGHTS = (0.15, 0.15, 0.15, 0.35, 1.0, 1.0, 1.0)

# How far before a joint limit its weight starts tapering off. Handing the load
# over *before* saturation is what removes the pop: with a hard clip the joint
# contributes fully right up to the limit and then stops dead in one step.
DEFAULT_LIMIT_MARGIN_RAD = math.radians(15.0)

# A tapered joint never freezes completely, or it could not move back out of a
# limit it is already sitting against.
_WEIGHT_FLOOR = 0.02

# Per-joint cap on a single IK iteration. Near a singularity the damped solve
# can still ask for a large step; clamping keeps that from teleporting the arm.
DEFAULT_MAX_STEP_RAD = math.radians(6.0)

# Per-joint cap on one whole solve_ik call, i.e. per teleop control tick. The
# per-iteration cap above is not enough on its own: 20 iterations of 6 deg all
# pulling the same way is a 120 deg lurch inside a single tick. At 50 Hz, 3 deg
# per tick is 150 deg/s, which tracks a hand comfortably while staying smooth.
DEFAULT_MAX_DELTA_PER_CALL_RAD = math.radians(3.0)

# Largest task-space correction attempted in one iteration. DLS linearizes about
# the current pose, so a big error makes the step invalid, not just large --
# clamping it is what keeps a far-away target from producing a lurch.
DEFAULT_MAX_POS_ERR_M = 0.02
DEFAULT_MAX_ORI_ERR_RAD = math.radians(5.0)

# Pull back toward the rest pose, per joint J1..J7, applied in whatever freedom
# the pose task leaves over. Proximal joints want to go home; the wrist is left
# alone (0.0) wherever it happens to be wound to.
#
# Together with the wrist-heavy DEFAULT_JOINT_WEIGHTS this produces last-in
# first-out ordering, which is what makes the motion read as natural:
#   rotating out   the wrist is cheapest, so it leads; the shoulder is pinned
#                  home by this bias until the wrist taper forces it to help.
#   rotating back  the shoulder is the only joint being actively pulled home, so
#                  it unwinds first; the wrist stays wound until the shoulder is
#                  back, then gives up its rotation.
DEFAULT_REST_BIAS_WEIGHTS = (1.0, 1.0, 1.0, 0.7, 0.0, 0.0, 0.0)
DEFAULT_REST_GAIN = 0.25

# Exchange rate used to score position error against orientation error when
# deciding whether an iteration made progress: 1 rad of residual rotation counts
# for 5 cm of residual translation. Position therefore wins ties, so a rotation
# the arm cannot reach is abandoned rather than paid for with position.
_ORI_TO_POS_M_PER_RAD = 0.05

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


def _nullspace_projector(J, rcond=1e-6):
    """Orthogonal projector onto null(J): motions J cannot see.

    SVD rather than a damped inverse so the projection is exact -- a damped one
    leaves the primary task slightly coupled to whatever the nullspace does.
    """
    n = J.shape[1]
    _, s, Vt = np.linalg.svd(J, full_matrices=True)
    rank = int(np.sum(s > (s[0] * rcond))) if s.size and s[0] > 0 else 0
    Vn = Vt[rank:].T  # columns span the nullspace
    if Vn.size == 0:
        return np.zeros((n, n))
    return Vn @ Vn.T


def _clamp_norm(v, max_norm):
    """Scale ``v`` down so ``||v|| <= max_norm``, preserving its direction."""
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)


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

    def __init__(
        self,
        model,
        data,
        dls_lambda: float = 0.05,
        joint_weights=DEFAULT_JOINT_WEIGHTS,
        limit_margin_rad: float = DEFAULT_LIMIT_MARGIN_RAD,
        max_step_rad: float = DEFAULT_MAX_STEP_RAD,
        max_pos_err_m: float = DEFAULT_MAX_POS_ERR_M,
        max_ori_err_rad: float = DEFAULT_MAX_ORI_ERR_RAD,
        rest_bias_weights=DEFAULT_REST_BIAS_WEIGHTS,
        rest_gain: float = DEFAULT_REST_GAIN,
        ori_pos_tradeoff: float = _ORI_TO_POS_M_PER_RAD,
        max_delta_per_call_rad: float = DEFAULT_MAX_DELTA_PER_CALL_RAD,
    ):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.dls_lambda = float(dls_lambda)

        self.joint_weights = np.asarray(joint_weights, dtype=float)
        if self.joint_weights.shape != (7,):
            raise ValueError(f"joint_weights must have 7 entries (J1..J7), got {self.joint_weights.shape}")
        if np.any(self.joint_weights <= 0.0):
            raise ValueError(f"joint_weights must all be > 0, got {joint_weights}")
        self.limit_margin_rad = float(limit_margin_rad)
        self.max_step_rad = float(max_step_rad)
        self.max_pos_err_m = float(max_pos_err_m)
        self.max_ori_err_rad = float(max_ori_err_rad)
        self.rest_bias_weights = np.asarray(rest_bias_weights, dtype=float)
        if self.rest_bias_weights.shape != (7,):
            raise ValueError(
                f"rest_bias_weights must have 7 entries (J1..J7), got {self.rest_bias_weights.shape}"
            )
        self.rest_gain = float(rest_gain)
        self.ori_pos_tradeoff = float(ori_pos_tradeoff)
        self.max_delta_per_call_rad = float(max_delta_per_call_rad)
        # Rest pose per side, captured the first time that arm is solved (i.e. the
        # pose the arm is holding when teleop starts).
        self._rest_q: dict[str, np.ndarray] = {}

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

        # The pose this arm was holding when teleop began is "home" for the
        # rest-pose bias below.
        if side not in self._rest_q:
            self._rest_q[side] = np.array([self.data.qpos[qi] for qi in idx])
        q_rest = self._rest_q[side]
        q_start = np.array([self.data.qpos[qi] for qi in idx])

        tgt_mat = np.zeros(9)
        mujoco.mju_quat2Mat(tgt_mat, target_quat)
        tgt_mat = tgt_mat.reshape(3, 3)

        def pose_error():
            """(pos_err, ori_err, scalar score) at the current qpos."""
            mujoco.mj_forward(self.model, self.data)
            p = target_pos - self.data.xpos[body_id]
            o = mat_to_axis_angle(tgt_mat @ self.data.xmat[body_id].reshape(3, 3).T)
            score = float(np.linalg.norm(p)) + self.ori_pos_tradeoff * float(np.linalg.norm(o))
            return p.copy(), o, score

        for _ in range(max_iter):
            pos_err, ori_err, score = pose_error()
            if np.linalg.norm(np.concatenate([pos_err, ori_err])) < 1e-4:
                break

            # Keep each correction inside the range where the Jacobian
            # linearization still holds. A big raw error makes the step invalid,
            # not merely large, which is what produced the lurching.
            pos_err = _clamp_norm(pos_err, self.max_pos_err_m)
            ori_err = _clamp_norm(ori_err, self.max_ori_err_rad)

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

            dof_idx = [self.model.jnt_dofadr[j] for j in jids]
            Jp = jacp[:, dof_idx]
            Jr = jacr[:, dof_idx]
            q = np.array([self.data.qpos[qi] for qi in idx])

            # Two-pass weighted least-norm. Pass 1 finds which way each joint
            # wants to travel; pass 2 re-solves with joints that are heading into
            # a nearby limit down-weighted, so the load moves to the next joint
            # out (wrist -> elbow -> shoulder) smoothly instead of in one jump.
            dq = self._priority_solve(Jp, Jr, pos_err, ori_err, self.joint_weights, q, q_rest)
            weights = self.joint_weights * self._limit_taper(q, lo, hi, dq)
            dq = self._priority_solve(Jp, Jr, pos_err, ori_err, weights, q, q_rest)

            dq = np.clip(dq, -self.max_step_rad, self.max_step_rad)

            for k, qi in enumerate(idx):
                self.data.qpos[qi] = np.clip(q[k] + dq[k], lo[k], hi[k])

            # Only keep a step that actually improved the pose. Once the target
            # is out of reach -- wrist wound to its stop, say -- the linearized
            # step stops helping and the joints would otherwise thrash against
            # their limits (the source of the remaining jumps and drift). Undo
            # and stop instead: the arm rotates as far as it can and holds.
            if pose_error()[2] > score:
                for k, qi in enumerate(idx):
                    self.data.qpos[qi] = q[k]
                break

        # Rate-limit the tick as a whole, then settle the model on the result.
        lim = self.max_delta_per_call_rad
        for k, qi in enumerate(idx):
            self.data.qpos[qi] = np.clip(self.data.qpos[qi], q_start[k] - lim, q_start[k] + lim)
        mujoco.mj_forward(self.model, self.data)
        return np.array([self.data.qpos[i] for i in idx])

    def _weighted_dls(self, J, dx, weights, lam):
        """Weighted damped least squares: dq = W Jt (J W Jt + lam^2 I)^-1 dx.

        A larger weight makes that joint cheaper to move, so it absorbs more of
        the motion. Weighting the pseudo-inverse (rather than post-scaling dq)
        keeps the task solved -- only the redundant DoF usage changes.
        """
        W = np.diag(weights)
        A = J @ W @ J.T + lam**2 * np.eye(J.shape[0])
        return W @ J.T @ np.linalg.solve(A, dx)

    def _priority_solve(self, Jp, Jr, pos_err, ori_err, weights, q, q_rest):
        """Weighted 6-DoF solve, plus a rest-pose pull along the self-motion DoF.

        The pose is solved as one weighted least-squares problem: with the wrist
        weighted high the wrist absorbs rotation first, and the limit taper hands
        the load outward as it saturates.

        The rest-pose bias is projected into the nullspace of the *whole* 6-DoF
        task, not just position. That subspace is the arm's self-motion manifold
        (1-DoF for 7 joints against a 6-DoF task): the elbow-lift family of
        configurations that leave the hand pose untouched. Projecting it any
        wider lets the bias corrupt the pose it is supposed to preserve, which
        makes the iteration fight itself and stop converging.
        """
        J = np.vstack([Jp, Jr])
        dx = np.concatenate([pos_err, ori_err])
        dq = self._weighted_dls(J, dx, weights, self.dls_lambda)

        bias = -self.rest_gain * self.rest_bias_weights * (q - q_rest)
        return dq + _nullspace_projector(J) @ bias

    def _limit_taper(self, q, lo, hi, dq):
        """Scale factor per joint that fades out as it approaches a limit.

        Only the direction of travel matters: a joint pinned against its low
        limit is still free to move back up. Smoothstep rather than linear so
        the handover has no slope discontinuity (a corner there is visible as a
        twitch). Floored, never zero, so a joint can always escape a limit.
        """
        headroom = np.where(dq >= 0.0, hi - q, q - lo)
        t = np.clip(headroom / max(self.limit_margin_rad, 1e-9), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        return _WEIGHT_FLOOR + (1.0 - _WEIGHT_FLOOR) * t

    def set_finger(self, side, val):
        """Set the finger slide joints for ``side`` (val in meters, clamped)."""
        val = float(np.clip(val, 0.0, FINGER_OPEN_M))
        for idx in self.finger_qpos_idx[side]:
            self.data.qpos[idx] = val

    def joint_positions(self, side):
        """Read the 7 arm joint angles (rad) for ``side`` in J1..J7 order."""
        return np.array([self.data.qpos[i] for i in self.qpos_idx[side]])
