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
"""The "neck" (absolute, Enactic dora-openarm-webxr) hand mapping of the
OpenXR pose source, driven headless: the XR thread is never started, the test
writes the controller and neck poses it would publish."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from lerobot.teleoperators.vr_mocap.config_vr_mocap import DEFAULT_MODEL_PATH
from lerobot.teleoperators.vr_mocap.ik import TCP_OFFSET_IN_HAND_M, hand_pos_from_tcp
from lerobot.teleoperators.vr_mocap.openxr_pose_source import OpenXRPoseSource


def _scene_path() -> Path:
    return Path(os.path.expanduser(os.environ.get("OPENARM_MUJOCO_SCENE", DEFAULT_MODEL_PATH)))


ORIGIN = np.array([0.0, 0.0, 0.697])
NECK = np.array([0.3, 0.1, 1.55])  # operator's neck pivot, robot axes, anywhere in the room
OFFSET = np.array([-0.085, 0.0, -0.14])


def _source(blend_s: float) -> OpenXRPoseSource:
    src = OpenXRPoseSource()
    src.hand_mapping = "neck"
    src.arm_origin = (ORIGIN.copy(), np.eye(3))
    src.neck_frame_offset = OFFSET.copy()
    src.neck_blend_s = blend_s
    src._neck_pos = NECK.copy()
    src._tracking = True
    src._activation_id = 1
    for s in ("left", "right"):
        src._ctrl_valid[s] = True
    return src


def _ee(pos) -> dict:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    return {"left": (np.array([0.0, 0.15, 0.1]), q), "right": (np.asarray(pos, float), q)}


def test_neck_target_is_hand_relative_to_neck_off_arm_origin():
    src = _source(blend_s=0.0)
    hand = NECK + np.array([0.45, -0.2, -0.25])
    src._ctrl_pos["right"][:] = hand
    tgt = src.get_targets(_ee([0.0, -0.15, 0.1]))["right"]
    # The controller places the robot's WRIST (hand origin); the TCP target the
    # IK chases is the hand->TCP offset beyond it, along the gripper.
    np.testing.assert_allclose(hand_pos_from_tcp(tgt.pos, tgt.quat), ORIGIN + (hand - NECK) + OFFSET, atol=1e-9)
    # Where the operator stands does not matter, only hand relative to neck
    # (followed live here; by default the neck is held from X, see below).
    src.neck_follow, src.neck_smooth_hz = True, 0.0
    src._neck_pos = NECK + np.array([1.0, -2.0, 0.0])
    src._ctrl_pos["right"][:] = hand + np.array([1.0, -2.0, 0.0])
    tgt2 = src.get_targets(_ee([0.0, -0.15, 0.1]))["right"]
    np.testing.assert_allclose(tgt2.pos, tgt.pos, atol=1e-9)


def test_neck_mapping_glides_in_from_the_arm_after_x():
    src = _source(blend_s=0.2)
    ee = np.array([0.0, -0.15, 0.1])
    src._ctrl_pos["right"][:] = NECK + np.array([0.5, -0.2, -0.2])
    first = src.get_targets(_ee(ee))["right"]
    np.testing.assert_allclose(first.pos, ee, atol=1e-6)  # no lunge on X
    time.sleep(0.25)
    late = src.get_targets(_ee(ee))["right"]
    np.testing.assert_allclose(hand_pos_from_tcp(late.pos, late.quat),
                               ORIGIN + np.array([0.5, -0.2, -0.2]) + OFFSET, atol=1e-6)


def test_neck_hand_point_moves_which_part_of_the_hand_is_at_the_controller():
    src = _source(blend_s=0.0)
    src.neck_hand_point_m = float(TCP_OFFSET_IN_HAND_M[2])  # fingertips at the hand
    hand = NECK + np.array([0.45, -0.2, -0.25])
    src._ctrl_pos["right"][:] = hand
    tgt = src.get_targets(_ee([0.0, -0.15, 0.1]))["right"]
    np.testing.assert_allclose(tgt.pos, ORIGIN + (hand - NECK) + OFFSET, atol=1e-9)


def test_neck_mapping_falls_back_to_delta_without_a_neck_pose():
    src = _source(blend_s=0.0)
    src._neck_pos = None
    ee = np.array([0.0, -0.15, 0.1])
    src._ctrl_pos["right"][:] = np.array([0.2, -0.1, 1.2])
    src.get_targets(_ee(ee))  # captures the X reference
    src._ctrl_pos["right"][:] += np.array([0.1, 0.0, 0.0])
    tgt = src.get_targets(_ee(ee))["right"]
    np.testing.assert_allclose(tgt.pos, ee + np.array([0.1, 0.0, 0.0]), atol=1e-9)


@pytest.mark.skipif(not _scene_path().is_file(), reason=f"OpenArm MuJoCo scene not found at {_scene_path()}")
def test_full_reach_forward_straightens_the_elbow():
    """An operator's arm held straight out at shoulder height maps to a target
    at (or past) the robot's full reach, so the IK -- springs and all -- ends
    with the elbow nearly straight. The delta mapping from a chest-height X
    could not get there (the reported bug)."""
    import mujoco

    from lerobot.teleoperators.vr_mocap.ik import IKSolver

    model = mujoco.MjModel.from_xml_path(str(_scene_path()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ik = IKSolver(model, data)
    origin = ik.site_pose("arm_origin")
    assert origin is not None
    np.testing.assert_allclose(origin[0], ORIGIN, atol=1e-3)

    src = _source(blend_s=0.0)
    src.arm_origin = origin
    # Neck pivot ~18 cm above the shoulder line; a 62 cm arm straight ahead,
    # level with the shoulder, 18 cm right of the neck.
    hand = NECK + np.array([0.62, -0.18, -0.18])
    src._ctrl_pos["right"][:] = hand
    ik.set_joint_positions("right", np.radians([0.0, 0.0, -35.0, 90.0, 35.0, 0.0, 0.0]))  # bent, like home
    ee_pos, ee_quat = ik.get_ee_pose("right")
    target = src.get_targets({"left": ik.get_ee_pose("left"), "right": (ee_pos, ee_quat)})["right"]
    for _ in range(200):
        ik.solve_ik("right", target.pos, target.quat)
    elbow_deg = math.degrees(abs(ik.joint_positions("right")[3]))
    assert elbow_deg < 15.0, f"elbow still bent {elbow_deg:.1f} deg at full reach"


def test_head_motion_after_x_does_not_move_the_target():
    """Default: the neck pivot is taken on X and held, so looking around (or
    headset tracking noise) cannot shake the arms."""
    src = _source(blend_s=0.0)
    src._ctrl_pos["right"][:] = NECK + np.array([0.45, -0.2, -0.25])
    before = src.get_targets(_ee([0.0, -0.15, 0.1]))["right"].pos
    src._neck_pos = NECK + np.array([0.03, 0.05, -0.02])  # head turned / bobbed
    after = src.get_targets(_ee([0.0, -0.15, 0.1]))["right"].pos
    np.testing.assert_allclose(after, before, atol=1e-12)
    src._activation_id += 1  # next X re-anchors on the new neck position
    moved = src.get_targets(_ee([0.0, -0.15, 0.1]))["right"].pos
    np.testing.assert_allclose(moved - before, -np.array([0.03, 0.05, -0.02]), atol=1e-9)


def _level_forward_source():
    # Hands level, straight ahead at shoulder height (neck pivot ~15 cm above).
    # Much lower and farther out the arm physically cannot hold the gripper
    # level there and the IK, position first, leaves it pitched down.
    from lerobot.teleoperators.vr_mocap.ik import xr_quat_to_robot

    src = _source(blend_s=0.0)
    for s in ("left", "right"):
        src._ctrl_quat[s][:] = xr_quat_to_robot(np.array([0.0, 0.0, 0.0, 1.0]))  # level, pointing ahead
        src._grip_to_aim[s] = np.eye(3)
    src._ctrl_pos["left"][:] = NECK + np.array([0.35, 0.18, -0.15])
    src._ctrl_pos["right"][:] = NECK + np.array([0.35, -0.18, -0.15])
    return src


def test_level_controller_gives_level_gripper_camera_up_on_both_arms():
    from lerobot.teleoperators.vr_mocap.ik import quat_to_mat

    tgts = _level_forward_source().get_targets(_ee([0.0, -0.15, 0.1]))
    for side in ("left", "right"):
        R = quat_to_mat(tgts[side].quat)
        np.testing.assert_allclose(R[:, 2], [1, 0, 0], atol=1e-9)  # gripper points ahead
        np.testing.assert_allclose(R[:, 0], [0, 0, 1], atol=1e-9)  # camera side up
    # Same hand angle -> same gripper angle on both arms, whatever they were on X.
    np.testing.assert_allclose(tgts["left"].quat, tgts["right"].quat, atol=1e-9)


@pytest.mark.skipif(not _scene_path().is_file(), reason=f"OpenArm MuJoCo scene not found at {_scene_path()}")
def test_level_forward_orientation_is_reached_by_ik_with_the_camera_on_top():
    import mujoco

    from lerobot.teleoperators.vr_mocap.ik import IKSolver, quat_to_mat

    model = mujoco.MjModel.from_xml_path(str(_scene_path()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ik = IKSolver(model, data)
    src = _level_forward_source()  # IK from the default pose
    src.arm_origin = ik.site_pose("arm_origin")
    # The wrist camera is on the TCP's +x side (checked on the model, at rest).
    for side in ("left", "right"):
        pos, quat = ik.get_ee_pose(side)
        cam = data.cam_xpos[model.camera(f"{side}_wrist_camera").id]
        assert float(quat_to_mat(quat)[:, 0] @ (cam - pos)) > 0.02, f"{side} wrist camera not on TCP +x"
    tgts = src.get_targets({s: ik.get_ee_pose(s) for s in ("left", "right")})
    for side in ("left", "right"):
        for _ in range(300):
            ik.solve_ik(side, tgts[side].pos, tgts[side].quat)
        pos, quat = ik.get_ee_pose(side)
        R = quat_to_mat(quat)
        np.testing.assert_allclose(pos, tgts[side].pos, atol=2e-3)
        assert float(R[:, 2] @ [1, 0, 0]) > 0.99, f"{side} gripper not pointing ahead: {R[:, 2]}"
        assert float(R[:, 0] @ [0, 0, 1]) > 0.99, f"{side} wrist camera not on top: {R[:, 0]}"
