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

"""Regression tests for OpenArm VR/keyboard IK teleop feel.

Requires the OpenArm MuJoCo scene (default ``~/sparkpack/openarm_mujoco/v1/scene.xml``).
Skipped when the model or ``mujoco`` is unavailable.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from lerobot.teleoperators.vr_mocap.config_vr_mocap import DEFAULT_MODEL_PATH
from lerobot.teleoperators.vr_mocap.ik import (
    axis_angle_to_quat,
    hand_pos_from_tcp,
    quat_inv,
    quat_mul,
    tcp_pos_from_hand,
)
from lerobot.teleoperators.vr_mocap.pose_source import (
    MAX_POS_DELTA_PER_TICK_M,
    MAX_ROT_DELTA_PER_TICK_RAD,
    POS_STEP,
    ROT_STEP,
    KeyboardPoseSource,
)


def _scene_path() -> Path:
    return Path(os.path.expanduser(os.environ.get("OPENARM_MUJOCO_SCENE", DEFAULT_MODEL_PATH)))


pytestmark = pytest.mark.skipif(
    not _scene_path().is_file(),
    reason=f"OpenArm MuJoCo scene not found at {_scene_path()}",
)


def _quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    q_rel = quat_mul(q_a, quat_inv(q_b))
    w = float(np.clip(abs(q_rel[0]), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(w))


def _count_sign_flips(dq: np.ndarray, min_step_rad: float) -> int:
    """Count direction reversals on steps large enough to feel like a shake."""
    flips = 0
    for j in range(dq.shape[1]):
        s = np.sign(dq[:, j])
        s[np.abs(dq[:, j]) < min_step_rad] = 0
        flips += int(np.sum((s[1:] * s[:-1]) < 0))
    return flips


def _run_closed_loop_teleop(mujoco, *, disable_contacts: bool, n_ticks: int = 250):
    """Drive right-arm IK + PD through translate / rotate / translate.

    Returns joint history, EE history, target history, and contact counts.
    """
    from lerobot.robots.mujoco_bi_openarm.mujoco_bi_openarm import apply_base_pose
    from lerobot.teleoperators.vr_mocap.ik import IKSolver

    path = str(_scene_path())
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    if disable_contacts:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    else:
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    apply_base_pose(mujoco, model, data, 90.0)

    model_ik = mujoco.MjModel.from_xml_path(path)
    data_ik = mujoco.MjData(model_ik)
    model_ik.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    apply_base_pose(mujoco, model_ik, data_ik, 90.0)
    ik = IKSolver(model_ik, data_ik)

    kp = np.array([240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0])
    kd = np.array([5.0, 5.0, 3.0, 5.0, 0.3, 0.3, 0.3])
    aids, qadrs, dadrs = [], [], []
    fr = []
    for i in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i + 1}")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_joint{i + 1}_ctrl")
        aids.append(aid)
        qadrs.append(model.jnt_qposadr[jid])
        dadrs.append(model.jnt_dofadr[jid])
        fr.append(model.actuator_forcerange[aid])
    fr = np.asarray(fr)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_hand_tcp")
    substeps = max(1, round(0.02 / model.opt.timestep))

    side = "right"
    ik.set_joint_positions(side, np.array([data.qpos[a] for a in qadrs]))
    _p0, q0 = ik.get_ee_pose(side)
    tgt_p = data.xpos[body_id].copy()
    tgt_q = q0.copy()

    qs, ees, tgts, ncons = [], [], [], []
    for t in range(n_ticks):
        if t < 80:
            tgt_p = tgt_p + np.array([0.0025, 0.0, 0.0])
        elif t < 160:
            # Rotate about the hand/wrist origin (TCP tip sweeps on a sphere).
            hand = hand_pos_from_tcp(data.xpos[body_id], tgt_q)
            tgt_q = quat_mul(tgt_q, axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), 0.04))
            tgt_q = tgt_q / np.linalg.norm(tgt_q)
            tgt_p = tcp_pos_from_hand(hand, tgt_q)
        else:
            tgt_p = tgt_p + np.array([0.0, 0.0025, 0.0])

        ik.set_joint_positions(side, np.array([data.qpos[a] for a in qadrs]))
        q_cmd = ik.solve_ik(side, tgt_p.copy(), tgt_q.copy())
        for _ in range(substeps):
            q_now = np.array([data.qpos[a] for a in qadrs])
            qd = np.array([data.qvel[a] for a in dadrs])
            tau = np.clip(kp * (q_cmd - q_now) - kd * qd, fr[:, 0], fr[:, 1])
            for i, aid in enumerate(aids):
                data.ctrl[aid] = tau[i]
            mujoco.mj_step(model, data)

        qs.append(np.array([data.qpos[a] for a in qadrs]))
        ees.append(data.xpos[body_id].copy())
        tgts.append(tgt_p.copy())
        ncons.append(int(data.ncon))

    return {
        "q": np.asarray(qs),
        "ee": np.asarray(ees),
        "tgt": np.asarray(tgts),
        "ncon": np.asarray(ncons),
    }


@pytest.fixture
def ik_env():
    mujoco = pytest.importorskip("mujoco")
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    from lerobot.teleoperators.vr_mocap.ik import IKSolver

    model = mujoco.MjModel.from_xml_path(str(_scene_path()))
    data = mujoco.MjData(model)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    # Start with a bent elbow so the arm is away from the straight singularity.
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_joint4")
    data.qpos[model.jnt_qposadr[jid]] = math.radians(90.0)
    mujoco.mj_forward(model, data)
    solver = IKSolver(model, data)
    return mujoco, model, data, solver


def test_full_contacts_and_pickable_cube():
    """Arms, table, and cube all collide; cube rests on the table."""
    mujoco = pytest.importorskip("mujoco")
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    model = mujoco.MjModel.from_xml_path(str(_scene_path()))
    data = mujoco.MjData(model)
    assert not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))

    def _mask(name: str) -> tuple[int, int]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])

    assert _mask("openarm_right_link4_collision") == (1, 1)
    assert _mask("openarm_right_right_finger_collision") == (1, 1)
    assert _mask("cube") == (1, 1)
    assert _mask("table_top") == (1, 1)

    for _ in range(400):
        mujoco.mj_step(model, data)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert float(data.xpos[cube_id][2]) > 0.30

    from lerobot.robots.mujoco_bi_openarm import MujocoBiOpenArm
    from lerobot.robots.mujoco_bi_openarm.config_mujoco_bi_openarm import MujocoBiOpenArmConfig

    cfg = MujocoBiOpenArmConfig(
        model_path=str(_scene_path()),
        cameras={},
        viewer=False,
        disable_collisions=False,
    )
    robot = MujocoBiOpenArm(cfg)
    robot.connect(calibrate=False)
    try:
        assert not (robot._model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
    finally:
        robot.disconnect()


def test_in_place_pitch_keeps_hand_pivot_stationary(ik_env):
    """Wrist pitch pivots about the hand origin; that point must stay put."""
    _mujoco, _model, _data, ik = ik_env
    side = "right"
    p0, q0 = ik.get_ee_pose(side)
    hand0 = hand_pos_from_tcp(p0, q0)
    drifts: list[float] = []
    for _ in range(20):
        _p, q = ik.get_ee_pose(side)
        # Keyboard-style body-fixed pitch about the hand's local +Y.
        q_tgt = quat_mul(q, axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), ROT_STEP))
        q_tgt /= np.linalg.norm(q_tgt)
        # TCP tip is allowed to sweep; the hand/wrist pivot is held.
        p_tgt = tcp_pos_from_hand(hand0, q_tgt)
        ik.solve_ik(side, p_tgt, q_tgt)
        p_now, q_now = ik.get_ee_pose(side)
        drifts.append(float(np.linalg.norm(hand_pos_from_tcp(p_now, q_now) - hand0)))

    max_drift_m = max(drifts)
    achieved_deg = _quat_angle_deg(ik.get_ee_pose(side)[1], q0)
    assert max_drift_m < 0.03, f"hand pivot drifted {max_drift_m * 100:.1f} cm while pitching"
    assert achieved_deg > 15.0, f"pitch barely moved ({achieved_deg:.1f} deg)"


def test_xyz_step_is_smooth(ik_env):
    """A single small XYZ target step must not produce a joint pop."""
    _mujoco, _model, _data, ik = ik_env
    side = "right"
    p0, q0 = ik.get_ee_pose(side)
    q_before = ik.joint_positions(side).copy()
    target = p0.copy()
    target[0] += POS_STEP
    ik.solve_ik(side, target, q0.copy())
    q_after = ik.joint_positions(side)
    max_joint_delta_deg = float(np.max(np.abs(q_after - q_before)) * 180.0 / math.pi)
    # Per-call joint rate limit is 3 deg; a 5 mm step should stay well under that.
    assert max_joint_delta_deg <= 3.0 + 1e-3
    p_now, _ = ik.get_ee_pose(side)
    assert float(np.linalg.norm(p_now - p0)) < 0.02


def test_translate_keeps_wrist_orientation_reasonable(ik_env):
    """XYZ motion prefers keeping wrist attitude (soft — hard lock froze the arm)."""
    _mujoco, _model, _data, ik = ik_env
    side = "right"
    p0, q0 = ik.get_ee_pose(side)
    max_ori_deg = 0.0
    p = p0.copy()
    for _ in range(40):
        p = p + np.array([MAX_POS_DELTA_PER_TICK_M, 0.0, 0.0])
        ik.solve_ik(side, p.copy(), q0.copy())
        pe, qe = ik.get_ee_pose(side)
        max_ori_deg = max(max_ori_deg, _quat_angle_deg(qe, q0))
    # Unweighted 6-DoF used to swing ~20°+ mid-path; weighted ori should stay milder.
    assert max_ori_deg < 20.0, f"wrist twisted {max_ori_deg:.1f} deg during +x translate"
    assert float(np.linalg.norm(pe - p0)) > 0.05, "arm failed to translate in +x"


def test_unreachable_translate_settles_without_thrash(ik_env):
    """A far target that needs a big twist must settle, not buzz the joints."""
    _mujoco, _model, _data, ik = ik_env
    side = "right"
    p0, q0 = ik.get_ee_pose(side)
    target = p0 + np.array([0.0, 0.35, 0.0])
    for _ in range(60):
        ik.solve_ik(side, target.copy(), q0.copy())
    pe, qe = ik.get_ee_pose(side)
    moved = float(np.linalg.norm(pe - p0))
    assert moved < 0.35, f"unexpected full reach ({moved * 100:.1f} cm)"
    q_mid = ik.joint_positions(side).copy()
    ik.solve_ik(side, target.copy(), q0.copy())
    settle = float(np.max(np.abs(ik.joint_positions(side) - q_mid)) * 180.0 / math.pi)
    assert settle < 0.5, f"thrashing when stuck: {settle:.2f} deg/tick"
    del qe


def test_closed_loop_no_shake_or_lock():
    """IK+PD teleop must not shake, jam, or lock with contacts disabled."""
    mujoco = pytest.importorskip("mujoco")
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    traj = _run_closed_loop_teleop(mujoco, disable_contacts=True, n_ticks=250)
    q, ee, tgt, ncon = traj["q"], traj["ee"], traj["tgt"], traj["ncon"]

    assert int(np.max(ncon)) == 0, f"expected no contacts, got max ncon={int(np.max(ncon))}"

    dq = np.diff(q, axis=0)
    flips = _count_sign_flips(dq, min_step_rad=math.radians(0.15))
    max_acc_deg = float(np.max(np.abs(np.diff(q, n=2, axis=0))) * 180.0 / math.pi)

    # Healthy collision-free run is ~25–40 meaningful flips / ~2–3 deg accel.
    # Contact-jammed runs reverse far more often and spike acceleration.
    assert flips < 80, f"joint shaking: {flips} direction reversals"
    assert max_acc_deg < 6.0, f"joint jerk too high: {max_acc_deg:.2f} deg/tick^2"

    errs = np.linalg.norm(ee - tgt, axis=1)
    lock_windows = 0
    for i in range(0, len(ee) - 10):
        t_move = float(np.linalg.norm(tgt[i + 10] - tgt[i]))
        e_move = float(np.linalg.norm(ee[i + 10] - ee[i]))
        if t_move > 0.005 and e_move < 0.001 and float(np.mean(errs[i : i + 10])) > 0.02:
            lock_windows += 1
    assert lock_windows == 0, f"arm locked while target moved ({lock_windows} windows)"

    # +x phase should move when orientation-preserving nullspace allows it.
    # (Other axes may legitimately lock rather than twist the wrist.)
    translate_path = float(np.sum(np.linalg.norm(np.diff(ee[:80], axis=0), axis=1)))
    assert translate_path > 0.05, f"hand barely moved during +x ({translate_path * 100:.1f} cm)"


def test_hold_pose_does_not_oscillate(ik_env):
    """With a fixed target, joints must settle instead of buzzing."""
    _mujoco, _model, _data, ik = ik_env
    side = "right"
    p0, q0 = ik.get_ee_pose(side)
    qs = []
    for _ in range(40):
        ik.solve_ik(side, p0.copy(), q0.copy())
        qs.append(ik.joint_positions(side).copy())
    qs = np.asarray(qs)
    # After a short settle, peak-to-peak motion should be tiny.
    settled = qs[20:]
    ptp_deg = float(np.max(np.ptp(settled, axis=0)) * 180.0 / math.pi)
    assert ptp_deg < 0.5, f"hold-pose buzz: joint PTP {ptp_deg:.2f} deg"


def test_keyboard_caps_pos_flood_per_tick():
    """Many buffered translation keys must not jump the target more than the tick cap."""
    src = KeyboardPoseSource()
    p0 = np.array([0.3, -0.15, 0.2])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    ee = {"right": (p0.copy(), q0.copy()), "left": (p0.copy(), q0.copy())}
    src.reset(ee)
    with src._lock:
        src._queue.extend(["w"] * 20)
    targets = src.get_targets(ee)
    delta = float(np.linalg.norm(targets["right"].pos - p0))
    assert delta <= MAX_POS_DELTA_PER_TICK_M + 1e-9
    assert delta >= POS_STEP - 1e-9


def test_keyboard_caps_rot_flood_and_pivots_about_hand():
    """Buffered pitch keys apply one press-worth about the fixed hand pivot."""
    src = KeyboardPoseSource()
    p0 = np.array([0.3, -0.15, 0.2])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    # Achieved pose differs from the integrated target so snap is observable.
    act_pos = p0 + np.array([0.02, 0.0, 0.0])
    ee = {"right": (act_pos.copy(), q0.copy()), "left": (act_pos.copy(), q0.copy())}
    src.reset({"right": (p0.copy(), q0.copy()), "left": (p0.copy(), q0.copy())})
    with src._lock:
        src._queue.extend(["i"] * 10)
    targets = src.get_targets(ee)
    hand0 = hand_pos_from_tcp(act_pos, q0)
    hand_tgt = hand_pos_from_tcp(targets["right"].pos, targets["right"].quat)
    assert np.allclose(hand_tgt, hand0, atol=1e-9)
    # Tip must move off the old TCP (sphere about the hand), not stay glued to it.
    assert float(np.linalg.norm(targets["right"].pos - act_pos)) > 1e-4
    achieved = _quat_angle_deg(targets["right"].quat, q0)
    assert achieved == pytest.approx(math.degrees(MAX_ROT_DELTA_PER_TICK_RAD), abs=0.5)
    assert achieved == pytest.approx(math.degrees(ROT_STEP), abs=0.5)
