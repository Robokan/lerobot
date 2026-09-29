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

"""MitControlLoop against simulated Damiao motors on python-can's virtual bus."""

import math
import threading
import time

import numpy as np
import pytest

can = pytest.importorskip("can")

from lerobot.motors.damiao.tables import MOTOR_LIMIT_PARAMS  # noqa: E402
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig  # noqa: E402
from lerobot.robots.openarm_follower.control_loop import MitControlLoop  # noqa: E402


class FakeMotors(threading.Thread):
    """Answers every MIT command with a status frame at the commanded position (an ideal motor)."""

    def __init__(self, bus_obj, channel):
        super().__init__(daemon=True)
        self.b = bus_obj
        self.side = can.Bus(channel=channel, interface="virtual", receive_own_messages=False)
        self.by_id = {bus_obj._get_motor_id(m): m for m in bus_obj.motors}
        self.commanded = {m: [] for m in bus_obj.motors}  # (t, pos_deg, vel_deg_s)
        self.silent = set()
        self.running = True

    def run(self):
        while self.running:
            msg = self.side.recv(timeout=0.01)
            if msg is None or msg.arbitration_id not in self.by_id:
                continue
            m = self.by_id[msg.arbitration_id]
            if m in self.silent:
                continue
            pmax, vmax, _ = MOTOR_LIMIT_PARAMS[self.b._motor_types[m]]
            d = msg.data
            q = self.b._uint_to_float((d[0] << 8) | d[1], -pmax, pmax, 16)
            dq = self.b._uint_to_float((d[2] << 4) | (d[3] >> 4), -vmax, vmax, 12)
            self.commanded[m].append((time.perf_counter(), math.degrees(q), math.degrees(dq)))
            qu = self.b._float_to_uint(q, -pmax, pmax, 16)
            self.side.send(
                can.Message(
                    arbitration_id=self.b._get_motor_recv_id(m),
                    is_extended_id=False,
                    data=[0x10 | (msg.arbitration_id & 0x0F), qu >> 8, qu & 0xFF, 0x80, 0x08, 0x00, 25, 25],
                )
            )

    def stop(self):
        self.running = False
        self.join(1)
        self.side.shutdown()


@pytest.fixture
def rig(tmp_path):
    robot = OpenArmFollower(
        OpenArmFollowerConfig(
            id="t",
            calibration_dir=tmp_path,
            port="loop",
            side="right",
            can_interface="virtual",
            use_can_fd=False,
        )
    )
    bus = robot.bus
    bus.connect(handshake=False)
    motors = FakeMotors(bus, "loop")
    motors.start()
    gains = dict.fromkeys(bus.motors, (10.0, 0.5))
    loop = MitControlLoop(bus, gains, hz=250, stale_s=0.3, name="test")
    loop.start(dict.fromkeys(bus.motors, 0.0))
    yield loop, motors
    loop.stop()
    motors.stop()
    bus.disconnect(False)


def test_runs_near_its_rate_and_glides_between_targets_without_jumps(rig):
    loop, motors = rig
    t0 = time.perf_counter()
    for k in range(1, 16):  # a 30 Hz ramp: +3 deg per target
        loop.set_target({"joint_1": 3.0 * k})
        time.sleep(1 / 30)
    time.sleep(0.2)
    rate = loop.cycles / (time.perf_counter() - t0)
    assert rate > 150, f"loop ran at {rate:.0f} Hz"
    pos = np.array([p for _, p, _ in motors.commanded["joint_1"]])
    assert np.abs(np.diff(pos)).max() < 1.5  # small steps at ~250 Hz, not 3 deg jumps at 30 Hz
    assert pos[-1] == pytest.approx(45.0, abs=0.1)  # lands exactly on the last target
    vel = np.array([v for _, _, v in motors.commanded["joint_1"]])
    assert np.median(vel[vel > 1]) == pytest.approx(90.0, rel=0.25)  # 3 deg / 33 ms
    assert loop.states()["joint_1"]["position"] == pytest.approx(45.0, abs=0.1)


def test_holds_with_zero_velocity_when_targets_stop(rig):
    loop, motors = rig
    loop.set_target({"joint_2": 10.0})
    time.sleep(0.3)
    tail = motors.commanded["joint_2"][-10:]
    assert all(abs(p - 10.0) < 1e-2 and abs(v) < 1.0 for _, p, v in tail)


def test_a_silent_motor_stops_the_loop(rig):
    loop, motors = rig
    motors.silent.add("joint_7")
    time.sleep(0.6)
    assert isinstance(loop.error, ConnectionError) and "joint_7" in str(loop.error)


def test_follower_routes_actions_and_observations_through_the_loop(tmp_path):
    robot = OpenArmFollower(
        OpenArmFollowerConfig(
            id="t",
            calibration_dir=tmp_path,
            port="route",
            side="right",
            can_interface="virtual",
            use_can_fd=False,
            control_hz=250,
            zero_offsets={"joint_1": 2.0},
        )
    )
    bus = robot.bus
    bus.connect(handshake=False)
    motors = FakeMotors(bus, "route")
    motors.start()
    robot._loop = MitControlLoop(bus, dict.fromkeys(bus.motors, (10.0, 0.5)), 250, 0.3, "route")
    robot._loop.start(dict.fromkeys(bus.motors, 2.0))  # raw 2.0 = true 0.0 on joint_1
    bus._mit_control_batch = lambda cmds: pytest.fail("sent directly while the loop owns the bus")
    bus.sync_read_all_states = lambda *a, **k: pytest.fail("read the bus while the loop owns it")
    try:
        robot.send_action({"joint_1.pos": 5.0})
        time.sleep(0.2)
        obs = robot.get_observation()
        assert obs["joint_1.pos"] == pytest.approx(5.0, abs=0.1)  # true frame, via the loop's replies
        assert motors.commanded["joint_1"][-1][1] == pytest.approx(7.0, abs=0.1)  # raw = true + offset
    finally:
        robot._loop.stop()
        motors.stop()
        bus.disconnect(False)
