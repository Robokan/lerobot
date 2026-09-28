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

"""OpenArm follower zero offsets, limit overrides and dry run, on python-can's virtual bus."""

import pytest

pytest.importorskip("can")

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig  # noqa: E402

JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]


def make(tmp_path, channel, **kw):
    cfg = OpenArmFollowerConfig(
        id="t",
        calibration_dir=tmp_path,
        port=channel,
        side="left",
        can_interface="virtual",
        use_can_fd=False,
        **kw,
    )
    return OpenArmFollower(cfg)


def set_readings(robot, deg):
    for m in JOINTS:
        robot.bus._last_known_states[m]["position"] = deg.get(m, 0.0)


def test_dry_run_sends_only_refresh_and_offsets_round_trip(tmp_path):
    robot = make(
        tmp_path,
        "dry",
        dry_run=True,
        zero_offsets={"joint_1": -6.5, "joint_4": -9.8},
        joint_limits_override={"gripper": (-157.0, 5.0)},
    )
    sniffer = __import__("can").Bus(channel="dry", interface="virtual", receive_own_messages=False)
    robot.connect()  # would raise if the handshake/ENABLE/calibration ran: the guard blocks them
    assert robot.config.joint_limits["gripper"] == (-157.0, 5.0)
    assert robot.config.joint_limits["joint_1"] == (-75.0, 75.0)  # side default kept

    # Arm hanging straight down: raw readings equal the offsets -> true frame reads 0.
    robot.bus._batch_refresh = lambda motors: None
    set_readings(robot, {"joint_1": -6.5, "joint_4": -9.8})
    obs = robot.get_observation()
    assert obs["joint_1.pos"] == pytest.approx(0.0)
    assert obs["joint_4.pos"] == pytest.approx(0.0)

    # Commanding the current (true) pose, including elbow 0 at its limit, must not
    # move anything: the limit is applied in the true frame, not the raw one.
    sent = robot.send_action({f"{m}.pos": 0.0 for m in JOINTS} | {"gripper.pos": -150.0})
    assert sent["joint_4.pos"] == pytest.approx(0.0)
    assert sent["gripper.pos"] == pytest.approx(-150.0)  # inside the widened range

    frames = []
    while (msg := sniffer.recv(timeout=0.05)) is not None:
        frames.append(msg)
    assert frames == [], "dry run transmitted CAN frames"  # _batch_refresh stubbed; nothing else may go out

    with pytest.raises(RuntimeError, match="dry_run: blocked"):
        robot.bus.enable_torque()
    robot.disconnect()
    sniffer.shutdown()


def test_live_send_adds_offset_after_true_frame_clip(tmp_path):
    robot = make(tmp_path, "live", zero_offsets={"joint_4": -9.8})
    robot.bus.connect(handshake=False)
    robot.bus._is_connected = True
    captured = {}
    robot.bus._mit_control_batch = lambda cmds: captured.update(cmds)
    robot.send_action({"joint_4.pos": -3.0})  # below the 0 deg limit in the true frame
    kp, kd, raw_deg, *_ = captured["joint_4"]
    assert raw_deg == pytest.approx(0.0 + -9.8)  # clipped to 0 true, then offset to raw
    robot.bus.disconnect(False)


def test_step_cap_uses_this_ticks_reading_and_caps_per_motor(tmp_path):
    robot = make(tmp_path, "cap", max_relative_target={m: (10.0 if m == "gripper" else 2.0) for m in JOINTS})
    robot.bus.connect(handshake=False)
    robot.bus._is_connected = True
    robot.bus._batch_refresh = lambda motors: None
    set_readings(robot, {"joint_1": 5.0})
    robot.get_observation()  # caches this tick's raw readings
    robot.bus.sync_read = lambda *a, **k: pytest.fail("step cap re-read the bus")
    captured = {}
    robot.bus._mit_control_batch = lambda cmds: captured.update(cmds)
    sent = robot.send_action({"joint_1.pos": 50.0, "gripper.pos": -100.0})
    assert sent["joint_1.pos"] == pytest.approx(7.0)  # 5 + 2 deg cap
    assert sent["gripper.pos"] == pytest.approx(-10.0)  # 0 - 10 deg gripper cap
    assert captured["joint_1"][2] == pytest.approx(7.0)
    robot.bus.disconnect(False)


def test_position_read_ignores_late_replies_from_the_previous_tick(tmp_path):
    """A late MIT reply left in the queue must not be taken as the refresh answer."""
    import math

    import can

    from lerobot.motors.damiao.tables import MOTOR_LIMIT_PARAMS

    robot = make(tmp_path, "late")
    bus = robot.bus
    bus.connect(handshake=False)
    motor = "gripper"
    pmax = MOTOR_LIMIT_PARAMS[bus._motor_types[motor]][0]

    def reply(deg):
        q = bus._float_to_uint(math.radians(deg), -pmax, pmax, 16)
        return can.Message(
            arbitration_id=bus._get_motor_recv_id(motor),
            is_extended_id=False,
            data=[bus._get_motor_id(motor), q >> 8, q & 0xFF, 0x80, 0x08, 0x00, 25, 25],
        )

    motor_side = can.Bus(channel="late", interface="virtual")
    motor_side.send(reply(-40.0))  # last tick's MIT reply, arrived after its window closed
    send = bus.canbus.send

    def answer_refresh(msg, *a, **k):
        send(msg, *a, **k)
        motor_side.send(reply(-60.0))  # the motor answers the refresh with where it is NOW

    bus.canbus.send = answer_refresh
    bus._batch_refresh([motor])
    assert bus._last_known_states[motor]["position"] == pytest.approx(-60.0, abs=0.1)
    bus.disconnect(False)
    motor_side.shutdown()


def test_velocity_feedforward_sends_the_targets_own_speed(tmp_path):
    import time

    robot = make(tmp_path, "vff", velocity_feedforward=True)
    robot.bus.connect(handshake=False)
    robot.bus._is_connected = True
    sent = []
    robot.bus._mit_control_batch = lambda cmds: sent.append(dict(cmds))
    robot.send_action({"joint_1.pos": 0.0})
    time.sleep(0.05)
    robot.send_action({"joint_1.pos": 2.0})
    _, _, _, vel0, _ = sent[0]["joint_1"]
    _, _, _, vel1, _ = sent[1]["joint_1"]
    assert vel0 == 0.0  # no previous target yet
    assert 2.0 / 0.07 < vel1 < 2.0 / 0.045  # ~2 deg over ~50 ms -> ~40 deg/s
    robot.bus.disconnect(False)
