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
"""Enactic's OpenArm IK (``openarm_control``: a mink QP) driving our IK model.

``vr_mocap --teleop.ik_solver=enactic`` swaps our damped-least-squares solver
for the one Enactic's dora-openarm-webxr stack runs: a bounded frame task, a
null-space posture task, a kinetic-energy regulariser, a singularity-approach
limit and joint velocity limits with braking, all one QP (daqp). The
singularity limit is what keeps a reach past the arm's length from shaking.

The solver runs on OUR v1 OpenArm model (robot only, no scene objects: those
added free joints that made a solve 10x slower), not on Enactic's v2 model,
whose wrist joints J6/J7 are laid out differently from the v1 rig. Targets
are our TCP body poses, expressed relative to the ``arm_origin`` site as
Enactic's solver expects. The result is written back into our ``IKSolver``,
so FK, markers, homing and the joint action keep working unchanged.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import require_package

from .ik import LEFT_TCP_BODY, RIGHT_TCP_BODY, quat_inv, quat_mul

if TYPE_CHECKING:
    from .ik import IKSolver
    from .pose_source import HandTarget

SIDES = ("left", "right")
# Robot-only model next to the scene the IK loads (scene.xml includes it).
_ROBOT_ONLY_XML = "openarm_bimanual.xml"


class EnacticIK:
    """Per-tick bimanual solve with Enactic's ``openarm_control.Kinematics``."""

    def __init__(self, ik: IKSolver, model_path: str, fps: float, max_iters: int = 10,
                 limit_velocity: bool = True):
        require_package("openarm-control", extra="openarm-enactic-ik", import_name="openarm_control")
        import mujoco
        from openarm_control import IKParams, Kinematics
        from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S, ArmSetup
        from openarm_mujoco.v2 import JointResolver

        path = os.path.expanduser(model_path)
        robot_only = os.path.join(os.path.dirname(path), _ROBOT_ONLY_XML)
        if os.path.isfile(robot_only):
            path = robot_only
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        origin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin")
        if origin < 0:
            raise ValueError(f"ik_solver=enactic needs an 'arm_origin' site in {path}")
        frames = {}
        for side, body in (("left", LEFT_TCP_BODY), ("right", RIGHT_TCP_BODY)):
            frames[side] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if frames[side] < 0:
                raise ValueError(f"ik_solver=enactic: no body {body!r} in {path}")
        setup = ArmSetup(model, data, JointResolver(model), ["right", "left"], frames,
                         {"right": "body", "left": "body"}, origin, "site")
        vel = None
        if limit_velocity:
            vel = {f"openarm_{s}_joint{i + 1}": v
                   for s in SIDES for i, v in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)}
        # One solve per control tick covers the whole tick (dt = 1/fps), split
        # into max_iters substeps. Keep the substeps near their node's (500 Hz x
        # 5 = 0.8 ms): at 30 Hz x 5 the singularity limit overshoots past reach.
        self._kin = Kinematics(setup, IKParams(dt=1.0 / float(fps), max_iters=int(max_iters),
                                               velocity_limits=vel))
        self._ik = ik
        self._last: np.ndarray | None = None  # our IK joints right after the last solve
        self.model_path = path
        # Our IK world -> arm_origin, from OUR model (the frame targets arrive in).
        o = ik.site_pose("arm_origin")
        if o is None:
            raise ValueError("ik_solver=enactic needs an 'arm_origin' site in the IK model")
        self._o_pos, self._o_R = o
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(self._o_R, float).flatten())
        self._o_quat_inv = quat_inv(q)

    def _driver16(self) -> np.ndarray:
        """Our IK's current joints as Enactic's right[8]+left[8] (grippers 0)."""
        r = np.append(self._ik.joint_positions("right"), 0.0)
        l_ = np.append(self._ik.joint_positions("left"), 0.0)
        return np.concatenate([r, l_]).astype(np.float32)

    def _rel_pose(self, pos, quat) -> np.ndarray:
        p = self._o_R.T @ (np.asarray(pos, float) - self._o_pos)
        q = quat_mul(self._o_quat_inv, np.asarray(quat, float))
        return np.concatenate([p, q / np.linalg.norm(q)]).astype(np.float32)

    def solve(self, targets: dict[str, HandTarget | None]) -> bool:
        """One bimanual solve toward ``targets`` (our IK world frame); a side
        without a target holds its current pose. The joints land in our IK."""
        # Start from whatever our IK holds when something else moved it (homing,
        # the h key and first robot contact write joints there directly). Only
        # then: re-syncing every tick replaces the QP's own float64 state.
        now = self._driver16()
        if self._last is None or not np.array_equal(now, self._last):
            self._kin.sync(now)
        for side in SIDES:
            tgt = targets.get(side)
            if tgt is None:
                pos, quat = self._ik.get_ee_pose(side)
            else:
                pos, quat = tgt.pos, tgt.quat
            self._kin.set_target(side, self._rel_pose(pos, quat))
        result = self._kin.solve()
        if result is None:
            return False
        q = np.asarray(result, float)
        self._ik.set_joint_positions("right", q[0:7])
        self._ik.set_joint_positions("left", q[8:15])
        self._last = self._driver16()
        return True
