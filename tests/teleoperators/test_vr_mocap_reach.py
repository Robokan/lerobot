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
"""Reaching past the arm with the absolute ("neck") mapping: the real arms
shook there (the IK flipping tick to tick at the straight-elbow singularity).
Both fixes are checked: the reach clamp for our DLS solver, and Enactic's
mink QP solver (ik_solver=enactic)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lerobot.teleoperators.vr_mocap.config_vr_mocap import DEFAULT_MODEL_PATH
from lerobot.teleoperators.vr_mocap.pose_source import HandTarget


def _scene_path() -> Path:
    return Path(os.path.expanduser(os.environ.get("OPENARM_MUJOCO_SCENE", DEFAULT_MODEL_PATH)))


pytestmark = pytest.mark.skipif(not _scene_path().is_file(),
                                reason=f"OpenArm MuJoCo scene not found at {_scene_path()}")

BEYOND = np.array([0.80, -0.18, 0.55])  # 0.8 m ahead: ~15 cm past the right arm's reach
REACHABLE = np.array([0.55, -0.18, 0.55])


def _ik():
    import mujoco

    from lerobot.teleoperators.vr_mocap.ik import IKSolver

    model = mujoco.MjModel.from_xml_path(str(_scene_path()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ik = IKSolver(model, data)
    bent = np.radians([0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0])
    ik.set_joint_positions("right", bent)
    ik.set_joint_positions("left", bent * [-1, 1, 1, 1, 1, 1, 1])
    return ik


def _level_quat():
    import mujoco

    q = np.zeros(4)  # gripper ahead (TCP z = +x), wrist camera up (TCP x = +z)
    mujoco.mju_mat2Quat(q, np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float).flatten())
    return q


def _chatter_deg(history) -> float:
    """RMS tick-to-tick joint acceleration after settling, degrees."""
    h = np.degrees(np.asarray(history))[60:]
    return float(np.sqrt(np.mean(np.diff(h, 2, axis=0) ** 2)))


def _drive(ik, step, pos, ticks=150):
    rng = np.random.default_rng(0)
    hist = []
    for _ in range(ticks):
        tgt = HandTarget(pos=pos + rng.normal(0.0, 3e-4, 3), quat=_level_quat(), gripper_m=0.0, active=True)
        step(tgt)
        hist.append(ik.joint_positions("right"))
    return hist


def test_reach_clamp_leaves_reachable_targets_alone_and_pulls_far_ones_in():
    from lerobot.teleoperators.vr_mocap.ik import ReachLimiter

    ik = _ik()
    rl = ReachLimiter(ik, min_elbow_deg=10.0, knee_elbow_deg=30.0)
    knee, lim = rl.radii("right")
    assert 0.38 < knee < lim < 0.44  # wrist-centre radii, upper arm + forearm ~0.435 m
    q = _level_quat()
    np.testing.assert_allclose(rl.clamp("right", REACHABLE, q), REACHABLE, atol=1e-12)
    # However far the hand goes, the wrist stays inside the limit radius, and
    # the roll-off is monotone (further out never comes back in).
    radii = [rl.wrist_radius("right", rl.clamp("right", BEYOND + [dx, 0.0, 0.0], q), q)
             for dx in (0.0, 0.1, 0.5, 5.0)]
    assert all(knee < r <= lim + 1e-9 for r in radii), radii
    assert radii == sorted(radii)


def test_dls_past_reach_is_still_with_the_clamp():
    from lerobot.teleoperators.vr_mocap.ik import ReachLimiter

    ik = _ik()
    rl = ReachLimiter(ik)

    def step(tgt):
        tgt.pos = rl.clamp("right", tgt.pos, tgt.quat)
        ik.solve_ik("right", tgt.pos, tgt.quat, max_iter=10)

    hist = _drive(ik, step, BEYOND)
    assert _chatter_deg(hist) < 0.15, "arm shakes reaching past the clamp"
    assert np.degrees(ik.joint_positions("right")[3]) > 8.0  # held short of the singularity


def test_dls_past_reach_without_the_clamp_is_the_reported_shake():
    ik = _ik()
    hist = _drive(ik, lambda t: ik.solve_ik("right", t.pos, t.quat, max_iter=10), BEYOND)
    assert _chatter_deg(hist) > 0.5  # documents the bug the clamp fixes


@pytest.fixture
def enactic():
    pytest.importorskip("openarm_control")
    from lerobot.teleoperators.vr_mocap.enactic_ik import EnacticIK

    ik = _ik()
    return ik, EnacticIK(ik, str(_scene_path()), fps=30)


def test_enactic_ik_past_reach_is_still(enactic):
    ik, en = enactic
    hist = _drive(ik, lambda t: en.solve({"right": t, "left": None}), BEYOND)
    assert _chatter_deg(hist) < 0.15, "Enactic IK shakes past reach"
    assert np.degrees(ik.joint_positions("right")[3]) > 8.0


def test_enactic_ik_tracks_a_reachable_target_and_holds_the_other_arm(enactic):
    ik, en = enactic
    left_before = ik.get_ee_pose("left")[0].copy()
    for _ in range(60):
        en.solve({"right": HandTarget(pos=REACHABLE, quat=_level_quat(), gripper_m=0.0, active=True),
                  "left": None})
    np.testing.assert_allclose(ik.get_ee_pose("right")[0], REACHABLE, atol=5e-3)
    np.testing.assert_allclose(ik.get_ee_pose("left")[0], left_before, atol=2e-3)
