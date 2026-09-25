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
import os

import numpy as np

_IK_DEBUG = os.environ.get("IK_DEBUG", "") not in ("", "0")

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

# Hard floor on elbow bend, for numerical hygiene only -- not a reach limit. A
# dead-straight arm (J4 = 0) is an exact singularity: moving the hand along the
# arm's own length needs the elbow to bend, and a straight elbow has no
# instantaneous authority there, so pulling the hand back does nothing at all
# (measured: smallest singular value 0.00000, condition number 1.9e12, five dead
# control ticks). Three degrees of bend is visually straight but leaves the
# Jacobian usable (minSV 0.00466, condition 401). The springs below are what
# actually keep the arm away from here; this is just the backstop.
ELBOW_SINGULARITY_FLOOR_RAD = math.radians(3.0)

# Elbow bend of the rest pose the springs pull toward. Independent of wherever the
# arm actually starts: the real robot powers up straight, and the whole point is
# for it to *want* to come off that straight configuration.
DEFAULT_REST_ELBOW_BEND_RAD = math.radians(90.0)

# Index of the elbow within the J1..J7 vectors.
_ELBOW_IDX = 3

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

# Springs pulling each joint back toward the base pose, J1..J7. Unlike a
# nullspace-only bias these are allowed to trade against the pose task, which is
# what makes them behave like real springs: the arm extends when the target pulls
# it out, and relaxes back toward the base pose when the target stops pulling.
#
# Graded the way you would build it mechanically -- very weak springs at the
# wrist, stiff ones at the shoulder. That single gradient produces the whole
# ordering, with no separate sequencing logic:
#   moving out    the wrist has the least resistance, so it gives first; the
#                 shoulder only joins once the wrist has run out of travel.
#   moving back   the shoulder's stiffer spring dominates, so it recovers first,
#                 and the slack wrist unwinds afterwards.
# It also keeps the springs out of the way of fine positioning: the wrist, which
# does the precise work, barely droops.
#
# A plain linear spring gives the "harder the straighter it gets" feel for free --
# the base pose has the elbow at 90 deg, so a straight arm is the largest
# displacement and therefore the largest restoring force.
DEFAULT_SPRING_WEIGHTS = (1.0, 1.0, 1.0, 0.6, 0.02, 0.02, 0.02)
DEFAULT_SPRING_GAIN = 0.15

# What the spring displacement is "worth" when deciding whether an iteration made
# progress, in metres of equivalent pose error per rad^2 of displacement. Without
# this the progress guard would reject every spring step as a pose regression and
# undo it. At 0.02, one joint held 90 deg off base costs about 5 cm of pose error
# -- enough to relax the arm when the target is slack, light enough that a target
# the arm can actually reach still wins.
DEFAULT_SPRING_SCORE_M_PER_RAD2 = 0.02

# Position error at which the springs start being allowed to move the hand, and
# the span over which that ramps to full. Below the tolerance the target is being
# tracked and accuracy wins; past tolerance+span the target is unreachable and the
# springs take over to bring the arm back in.
DEFAULT_REACH_TOL_M = 0.03
DEFAULT_REACH_SPAN_M = 0.07

# Extra willingness given to a joint that is displaced from the base pose when the
# step being considered would bring it back. This is what makes the shoulder
# recover first: the springs alone cannot do it, because while the target is
# tracking they are confined to the nullspace and so cannot serve the hand motion
# at all. Scaled per joint by spring_weights, so the same stiff-shoulder/weak-wrist
# gradient decides the ordering. Being a weight and not a force, it changes which
# joints do the work without costing any steady-state accuracy.
# Wind-up hardening, J1..J7, degrees; 0 = off. A joint's willingness to take
# MORE of the task fades as it winds away from the rest pose, reaching the floor
# at this displacement, so the motion is handed to the next joint in the chain.
# This is the spring-chain feel the rest-pose springs were meant to give but
# cannot, because during a turn they are confined to the task nullspace: the
# light wrist twists first, and from about halfway through its travel the
# shoulder takes over progressively. Wrist joints have +-90 deg of travel, so
# 45 = halfway. Shoulder and elbow do not harden: they are the end of the chain.
DEFAULT_HANDOVER_DEG = (0.0, 0.0, 0.0, 0.0, 45.0, 45.0, 45.0)
DEFAULT_HOMING_BOOST = 6.0
DEFAULT_HOMING_SCALE_RAD = math.radians(45.0)

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


def quat_rotate(q, v):
    """Rotate vector ``v`` by unit quaternion ``q`` [w, x, y, z]."""
    # q v q^{-1} via the standard sandwich product.
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_inv(q))[1:]


# TCP body pose relative to ``openarm_<side>_hand`` in the MuJoCo model
# (``pos="0 0 0.08"``). Wrist/keyboard rotations pivot about the hand origin,
# not the TCP tip 8 cm out — holding the tip fixed made the wrist orbit it and
# the apparent center change with each rotation axis.
TCP_OFFSET_IN_HAND_M = np.array([0.0, 0.0, 0.08])


def hand_pos_from_tcp(tcp_pos, tcp_quat):
    """World hand origin from TCP pose (inverse of the fixed hand→TCP offset)."""
    return np.asarray(tcp_pos, dtype=float) - quat_rotate(tcp_quat, TCP_OFFSET_IN_HAND_M)


def tcp_pos_from_hand(hand_pos, tcp_quat):
    """World TCP position that keeps ``hand_pos`` fixed at the given orientation."""
    return np.asarray(hand_pos, dtype=float) + quat_rotate(tcp_quat, TCP_OFFSET_IN_HAND_M)


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
        spring_weights=DEFAULT_SPRING_WEIGHTS,
        spring_gain: float = DEFAULT_SPRING_GAIN,
        spring_score_m_per_rad2: float = DEFAULT_SPRING_SCORE_M_PER_RAD2,
        elbow_floor_rad: float = ELBOW_SINGULARITY_FLOOR_RAD,
        reach_tol_m: float = DEFAULT_REACH_TOL_M,
        reach_span_m: float = DEFAULT_REACH_SPAN_M,
        homing_boost: float = DEFAULT_HOMING_BOOST,
        handover_deg=DEFAULT_HANDOVER_DEG,
        homing_scale_rad: float = DEFAULT_HOMING_SCALE_RAD,
        rest_elbow_bend_rad: float = DEFAULT_REST_ELBOW_BEND_RAD,
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
        self.spring_weights = np.asarray(spring_weights, dtype=float)
        if self.spring_weights.shape != (7,):
            raise ValueError(
                f"spring_weights must have 7 entries (J1..J7), got {self.spring_weights.shape}"
            )
        self.spring_gain = float(spring_gain)
        self.handover_rad = np.radians(np.asarray(handover_deg, dtype=float))
        if self.handover_rad.shape != (7,):
            raise ValueError(f"handover_deg must have 7 entries (J1..J7), got {self.handover_rad.shape}")
        self.spring_score_m_per_rad2 = float(spring_score_m_per_rad2)
        self.reach_tol_m = float(reach_tol_m)
        self.reach_span_m = float(reach_span_m)
        self.homing_boost = float(homing_boost)
        self.homing_scale_rad = float(homing_scale_rad)
        self.ori_pos_tradeoff = float(ori_pos_tradeoff)
        self.max_delta_per_call_rad = float(max_delta_per_call_rad)
        self.rest_elbow_bend_rad = float(rest_elbow_bend_rad)

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
        # Keep the elbow off dead-straight (see ELBOW_SINGULARITY_FLOOR_RAD).
        for lows in self.limits_low.values():
            lows[_ELBOW_IDX] = max(float(lows[_ELBOW_IDX]), elbow_floor_rad)

        # Pose the springs pull toward. Deliberately a fixed, configured pose --
        # NOT wherever the arm happened to be when teleop started. The real robot
        # powers up with the elbow straight, which is a singularity, so capturing
        # the start pose would make the springs hold the arm in the one
        # configuration it most needs to leave.
        # Per-side mask (7 values, or None) limiting which joints may serve
        # ORIENTATION. The caller sets it for a gesture whose turn belongs to
        # named joints; without it the solver happily rolls the tool with the
        # elbow and wrist pitch once the intended joints are pinned, which
        # contorts the arm and does not retrace on the way back.
        self.ori_joint_mask: dict[str, np.ndarray | None] = {}
        self._rest_q: dict[str, np.ndarray] = {}
        for side in self.joint_ids:
            r = np.zeros(7)
            r[_ELBOW_IDX] = self.rest_elbow_bend_rad
            self._rest_q[side] = np.clip(r, self.limits_low[side], self.limits_high[side])
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

        q_rest = self._rest_q[side]
        # Enforce joint limits (incl. elbow singularity floor) before iterating.
        # A qpos below ``lo`` would be clipped on the first step, move XYZ, then
        # get reverted by the progress guard -- freezing the arm at an illegal
        # configuration (seen at the straight hang pose).
        for k, qi in enumerate(idx):
            self.data.qpos[qi] = np.clip(self.data.qpos[qi], lo[k], hi[k])
        q_start = np.array([self.data.qpos[qi] for qi in idx])

        tgt_mat = np.zeros(9)
        mujoco.mju_quat2Mat(tgt_mat, target_quat)
        tgt_mat = tgt_mat.reshape(3, 3)

        def pose_error():
            """(pos_err, ori_err, scalar score) at the current qpos.

            The score is what the progress guard minimizes, so it has to include
            the spring energy as well as the pose error -- otherwise every step
            that relaxes the arm toward the base pose reads as a pose regression
            and gets reverted.
            """
            mujoco.mj_forward(self.model, self.data)
            p = target_pos - self.data.xpos[body_id]
            o = mat_to_axis_angle(tgt_mat @ self.data.xmat[body_id].reshape(3, 3).T)
            disp = np.array([self.data.qpos[qi] for qi in idx]) - q_rest
            spring = self.spring_score_m_per_rad2 * float(np.sum(self.spring_weights * disp**2))
            score = (
                float(np.linalg.norm(p))
                + self.ori_pos_tradeoff * float(np.linalg.norm(o))
                + spring
            )
            return p.copy(), o, score

        for _ in range(max_iter):
            pos_err, ori_err, score = pose_error()
            # Converged only when the pose is reached AND the springs are satisfied.
            # Testing the pose alone would exit on the first iteration whenever the
            # hand already sits on its target -- which is the state the arm boots
            # in -- so the springs would never run and a straight arm would stay
            # straight forever.
            spring_disp = self.spring_weights * (
                np.array([self.data.qpos[qi] for qi in idx]) - q_rest
            )
            if (
                np.linalg.norm(np.concatenate([pos_err, ori_err])) < 1e-4
                and np.linalg.norm(spring_disp) < 1e-3
            ):
                break

            # How far out of reach the target is, 0 (tracking fine) .. 1 (hopeless).
            # Gates how much the springs may pull the hand off target below.
            reach_deficit = float(
                np.clip(
                    (np.linalg.norm(pos_err) - self.reach_tol_m) / max(self.reach_span_m, 1e-9),
                    0.0,
                    1.0,
                )
            )

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

            # Default: full 6-DoF priority solve (position + orientation + springs).
            # A hard "position only in null(Jr)" lock made most axes undriveable
            # from the hang pose (especially +z). Prefer orientation with a high
            # task weight instead, and reject a step only if it *worsens* wrist
            # attitude beyond a small tolerance — that still stops a twist-to-
            # reach cheat without freezing ordinary translation.
            #
            # When the TCP is already on target and only orientation remains
            # (pure wrist keys after the pose source rewrote the tip on a sphere
            # about the hand), put orientation in null(Jp) so we don't shove XYZ
            # to chase a twist.
            pos_n = float(np.linalg.norm(pos_err))
            ori_n = float(np.linalg.norm(ori_err))
            hold_pos = pos_n < 0.008 and ori_n > 1e-4
            if _IK_DEBUG:
                print(f"[ik] {side} pos_cm {pos_n * 100:.3f} ori_deg {math.degrees(ori_n):.3f} hold {int(hold_pos)}", flush=True)
            if hold_pos:
                ori_err = _clamp_norm(ori_err, min(self.max_ori_err_rad * 1.6, math.radians(8.0)))
                N = _nullspace_projector(Jp)
                dq_pos = self._weighted_dls(Jp, pos_err, np.ones(7), self.dls_lambda)
                # The bare joint_weights here were why a wrist turn ran to the
                # joint limit before the shoulder moved: nothing faded the wrist.
                w_ori = (self.joint_weights
                         * self._handover_taper(q, q_rest)
                         * self._limit_taper(q, lo, hi, np.sign(self._weighted_dls(Jr, ori_err, self.joint_weights, self.dls_lambda))))
                mask = self.ori_joint_mask.get(side)
                if mask is not None:
                    w_ori = w_ori * mask
                dq_ori = N @ self._weighted_dls(Jr, ori_err, w_ori, self.dls_lambda)
                dq = dq_pos + dq_ori
            else:
                # Orientation rows weighted up so translation prefers solutions
                # that keep the wrist attitude.
                ori_w = 8.0
                J = np.vstack([Jp, ori_w * Jr])
                dx = np.concatenate([pos_err, ori_w * ori_err])
                dq = self._weighted_dls(J, dx, self.joint_weights, self.dls_lambda)
                weights = (
                    self.joint_weights
                    * self._handover_taper(q, q_rest)
                    * self._limit_taper(q, lo, hi, dq)
                    * self._homing_boost(q, q_rest, dq)
                )
                dq = self._weighted_dls(J, dx, weights, self.dls_lambda)
                spring = -self.spring_gain * self.spring_weights * (q - q_rest)
                # Springs in the nullspace of the weighted task so they cannot
                # buy rest-pose relaxation by twisting the wrist.
                dq = dq + _nullspace_projector(J) @ spring

            dq = np.clip(dq, -self.max_step_rad, self.max_step_rad)

            for k, qi in enumerate(idx):
                self.data.qpos[qi] = np.clip(q[k] + dq[k], lo[k], hi[k])

            new_pos_err, new_ori_err, new_score = pose_error()
            if hold_pos:
                if float(np.linalg.norm(new_pos_err)) > pos_n + 0.003:
                    for k, qi in enumerate(idx):
                        self.data.qpos[qi] = q[k]
                    break
            else:
                # Soft no-twist: allow motion, but undo a step that makes the
                # wrist attitude clearly worse than before.
                if (float(np.linalg.norm(new_ori_err)) > ori_n + math.radians(2.0)
                        or new_score > score):
                    for k, qi in enumerate(idx):
                        self.data.qpos[qi] = q[k]
                    # Best effort instead of a freeze. With wrist joints on
                    # their limits no step can translate AND hold the attitude,
                    # and rejecting every step left the arm dead to w/s/a/d/r/f
                    # (measured: zero motion on all six until a rotation key
                    # took the wrist off its limits). So translate first and
                    # serve the attitude only in the nullspace of that; keep
                    # the step only if the tip actually got closer.
                    dq_p = self._weighted_dls(Jp, pos_err, weights, self.dls_lambda)
                    dq_r = _nullspace_projector(Jp) @ self._weighted_dls(Jr, ori_err, weights, self.dls_lambda)
                    dq = np.clip(dq_p + dq_r, -self.max_step_rad, self.max_step_rad)
                    for k, qi in enumerate(idx):
                        self.data.qpos[qi] = np.clip(q[k] + dq[k], lo[k], hi[k])
                    new_pos_err, _, _ = pose_error()
                    if _IK_DEBUG:
                        applied = np.array([self.data.qpos[qi] for qi in idx]) - q
                        print(f"[ikfb] {side} pos {pos_n*100:.2f}->{float(np.linalg.norm(new_pos_err))*100:.2f} cm  "
                              f"dq_p(deg) {np.round(np.degrees(dq_p), 2).tolist()}  applied {np.round(np.degrees(applied), 2).tolist()}  "
                              f"w {np.round(weights, 3).tolist()}  lo-hi width(deg) {np.round(np.degrees(hi - lo), 1).tolist()}", flush=True)
                    if float(np.linalg.norm(new_pos_err)) >= pos_n - 1e-5:
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

    def _priority_solve(self, Jp, Jr, pos_err, ori_err, weights, q, q_rest, reach_deficit):
        """Weighted 6-DoF solve plus base-pose springs.

        The pose is solved as one weighted least-squares problem: with the wrist
        weighted high the wrist absorbs rotation first, and the limit taper hands
        the load outward as it saturates.

        The springs are split by how well the target is being tracked, because a
        spring that is always free to move the hand costs accuracy everywhere
        (measured: 80 mm of steady-state droop at the stiffness the return
        ordering wants):

        * While the target is reachable the spring is confined to the nullspace of
          the full 6-DoF task -- the self-motion manifold -- so the arm relaxes
          toward the base pose *without* the hand drifting off target at all.
        * As the target moves out of reach (``reach_deficit`` -> 1) the spring is
          progressively allowed to act directly. There the hand cannot be put
          where it was asked anyway, so pulling the arm back in beats letting it
          hang at full extension against a singularity.
        """
        J = np.vstack([Jp, Jr])
        dx = np.concatenate([pos_err, ori_err])
        dq = self._weighted_dls(J, dx, weights, self.dls_lambda)

        spring = -self.spring_gain * self.spring_weights * (q - q_rest)
        free = _nullspace_projector(J) @ spring
        return dq + free + reach_deficit * (spring - free)

    def _homing_boost(self, q, q_rest, dq):
        """Per-joint weight multiplier favouring joints heading back to base pose.

        Only applies to joints whose step reduces their displacement, so it never
        encourages leaving the base pose -- it just makes coming back cheap, and
        cheapest for the joints with the stiffest springs.
        """
        disp = q - q_rest
        coming_home = (disp * dq) < 0.0
        mag = np.clip(np.abs(disp) / max(self.homing_scale_rad, 1e-9), 0.0, 1.0)
        return 1.0 + self.homing_boost * self.spring_weights * np.where(coming_home, mag, 0.0)

    def _handover_taper(self, q, q_rest):
        """Per-joint willingness that fades with displacement from the rest pose
        (smoothstep to the same floor as the limit taper). Joints with
        handover 0 never fade. Direction-agnostic: a wound-up wrist is equally
        reluctant to wind further either way; unwinding is served by the
        homing boost, which prefers it."""
        t = np.ones(7)
        active = self.handover_rad > 1e-9
        if np.any(active):
            frac = np.clip(np.abs(q - q_rest)[active] / self.handover_rad[active], 0.0, 1.0)
            keep = 1.0 - frac
            t[active] = keep * keep * (3.0 - 2.0 * keep)
        return _WEIGHT_FLOOR + (1.0 - _WEIGHT_FLOOR) * t

    def chain_step(self, side, axis_world, angle, stiffness=None, max_step_rad=None, ref_q=None):
        """Rotation about ``axis_world`` by signed ``angle`` (rad), applied in
        JOINT SPACE to the joints whose axes align with it, one at a time.

        ``ref_q`` is the DEFAULT pose (the arm at launch). The rule:
          * unwinding (a joint moving back toward default) goes first,
            wrist-first (distal to proximal), each only as far as default;
          * then winding (away from default) goes shoulder-first (proximal to
            distal), each to its joint limit;
          * the shoulder joints (J1-J3) wind only on the L key (negative
            angle): default is their stop in the J direction, so past default
            the wrist turns instead.
        The per-tick speed cap never passes motion to the next joint; only
        running out of travel does. The hand goes where this takes it and the
        caller re-targets IK on the result. Returns the per-joint step (rad)."""
        idx = self.qpos_idx[side]
        jids = self.joint_ids[side]
        lo, hi = self.limits_low[side], self.limits_high[side]
        q = np.array([self.data.qpos[qi] for qi in idx])
        ref = q.copy() if ref_q is None else np.asarray(ref_q, dtype=float)
        axis = np.asarray(axis_world, dtype=float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        c = np.array([float((self.data.xmat[self.model.jnt_bodyid[j]].reshape(3, 3)
                             @ self.model.jnt_axis[j]) @ axis) for j in jids])
        lim = self.max_delta_per_call_rad if max_step_rad is None else float(max_step_rad)
        eps = math.radians(0.2)
        dq = np.zeros(7)
        remaining = float(angle)

        def serve(i, room):
            """Give this joint as much of the remaining request as it has room
            for. Returns True if it was speed-capped (nothing passes on)."""
            nonlocal remaining
            step = remaining / c[i]
            if abs(step) > room:                       # out of travel: the rest passes on
                step = math.copysign(min(room, lim), step)
                dq[i] += step
                remaining -= c[i] * step
                return False
            dq[i] += math.copysign(min(abs(step), lim), step)
            remaining = 0.0
            return True

        # 1. unwinding, wrist first, each only back to default
        for i in (6, 5, 4, 3, 2, 1, 0):
            if abs(remaining) < 1e-9:
                break
            if abs(c[i]) < 0.05:
                continue
            step_dir = math.copysign(1.0, remaining / c[i])
            disp = q[i] - ref[i]
            if abs(disp) > eps and step_dir * disp < 0:        # moving toward default
                if serve(i, abs(disp)):
                    break
        # 2. winding, shoulder first, each to its limit; shoulder only on L
        for i in (0, 1, 2, 3, 4, 5, 6):
            if abs(remaining) < 1e-9:
                break
            if abs(c[i]) < 0.05:
                continue
            if i <= 2 and angle > 0:
                continue                                     # default is the shoulder's stop for J
            step_dir = math.copysign(1.0, remaining / c[i])
            room = (hi[i] - q[i] - dq[i]) if step_dir > 0 else (q[i] + dq[i] - lo[i])
            if room <= 1e-6:
                continue
            if serve(i, room):
                break
        q_new = np.clip(q + dq, lo, hi)
        for kk, qi in enumerate(idx):
            self.data.qpos[qi] = float(q_new[kk])
        self._mujoco.mj_forward(self.model, self.data)
        return q_new - q

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

    def joint_axis_world(self, side, k: int) -> np.ndarray:
        """Unit world-frame axis of arm joint ``k`` at the current pose."""
        j = self.joint_ids[side][k]
        a = self.data.xmat[self.model.jnt_bodyid[j]].reshape(3, 3) @ self.model.jnt_axis[j]
        return a / max(float(np.linalg.norm(a)), 1e-9)

    def tool_axis_alignment(self, side, k: int) -> float:
        """|cos| between arm joint ``k``'s world axis and the tool's own z axis.

        1 = the joint turns the tool about its own axis (it can serve a roll
        of the tool); 0 = it is perpendicular and cannot contribute at all.
        With the elbow bent the upper-arm roll leaves the tool axis, so a rule
        written for the straight arm has to know to stop waiting for it.
        """
        j = self.joint_ids[side][k]
        a = self.data.xmat[self.model.jnt_bodyid[j]].reshape(3, 3) @ self.model.jnt_axis[j]
        body_id = self.left_tcp_id if side == "left" else self.right_tcp_id
        z = self.data.xmat[body_id].reshape(3, 3)[:, 2]
        return abs(float(a @ z) / max(float(np.linalg.norm(a) * np.linalg.norm(z)), 1e-9))

    def rest_pose(self, side):
        """The pose the springs pull toward, for ``side`` (7 joint angles, rad)."""
        return self._rest_q[side].copy()

    def set_joint_positions(self, side, q):
        """Write the 7 arm joint angles for ``side`` (rad), clamped to limits."""
        q = np.clip(q, self.limits_low[side], self.limits_high[side])
        for k, qi in enumerate(self.qpos_idx[side]):
            self.data.qpos[qi] = float(q[k])
        self._mujoco.mj_forward(self.model, self.data)

    def joint_positions(self, side):
        """Read the 7 arm joint angles (rad) for ``side`` in J1..J7 order."""
        return np.array([self.data.qpos[i] for i in self.qpos_idx[side]])
