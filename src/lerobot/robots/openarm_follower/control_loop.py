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

"""High-rate MIT command loop for one OpenArm (Damiao motors on one CAN bus).

Damiao MIT mode is a bare impedance law, tau = kp*(q_ref - q) + kd*(v_ref - v)
+ tau_ff: there is no trajectory generation in the motor. Sending one command
per 30 Hz loop tick made each target a step the stiff kp jumped to every 33 ms,
which showed as a wobble on fast moves. Enactic's own controller (openarm_teleop)
runs its follower thread at 1 kHz with position AND velocity references.

This thread does the same at ``hz`` (default 250, what one 1 Mbit/s classic CAN
bus carries comfortably for 8 commands + 8 replies). The 30 Hz loop hands it
targets with :meth:`set_target`; each one starts a straight-line glide from the
position being commanded *now* to the new target, timed to land when the next
target is due (so ~one loop period of latency, and never a jump). The glide's
speed is the velocity reference. No new target -> it holds the last one with
zero velocity. The motors' replies to these commands are the state the loop
reports, so while it runs it is the only user of the bus.
"""

from __future__ import annotations

import logging
import threading
import time

import can

from lerobot.motors.damiao import DamiaoMotorsBus

logger = logging.getLogger(__name__)


class MitControlLoop:
    def __init__(
        self,
        bus: DamiaoMotorsBus,
        gains: dict[str, tuple[float, float]],
        hz: float,
        stale_s: float,
        name: str = "arm",
    ):
        self.bus = bus
        self.gains = gains
        self.hz = float(hz)
        self.stale_s = float(stale_s)
        self.name = name
        self.motors = list(gains)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self.error: Exception | None = None
        # trajectory segment: glide from p0 to p1, starting at t1, lasting dur
        self._p0: dict[str, float] = {}
        self._p1: dict[str, float] = {}
        self._t1 = 0.0
        self._dur = 1.0 / 30.0
        self._last_set = None
        self._state: dict[str, dict] = {}
        self._last_reply: dict[str, float] = {}
        self.cycles = 0
        self.max_period_s = 0.0

    # ------------------------------------------------------------------ API
    def start(self, initial_raw: dict[str, float]) -> None:
        """Begin holding ``initial_raw`` (the motors' positions right now)."""
        now = time.perf_counter()
        with self._lock:
            self._p0 = dict(initial_raw)
            self._p1 = dict(initial_raw)
            self._t1 = now
            self._state = {m: dict(self.bus._last_known_states[m]) for m in self.motors}
            self._last_reply = dict.fromkeys(self.motors, now)
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"mit-loop-{self.name}", daemon=True)
        self._thread.start()
        logger.info(f"{self.name}: MIT control loop at {self.hz:.0f} Hz")

    def set_target(self, raw: dict[str, float]) -> None:
        """New goal (raw motor degrees), typically at the teleop/policy rate."""
        now = time.perf_counter()
        with self._lock:
            here = self._reference(now)[0]
            # Glide over the interval the targets are arriving at (so it lands
            # as the next one is due); bounded, so a pause does not make a crawl
            # and a burst does not make a jump.
            gap = (now - self._last_set) if self._last_set is not None else 1.0 / 30.0
            self._dur = min(max(gap, 0.01), 0.1)
            self._p0 = here
            self._p1 = {m: float(raw.get(m, self._p1[m])) for m in self.motors}
            self._t1 = now
            self._last_set = now

    def states(self) -> dict[str, dict]:
        with self._lock:
            return {m: dict(s) for m, s in self._state.items()}

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ------------------------------------------------------------- internals
    def _reference(self, now: float) -> tuple[dict[str, float], dict[str, float]]:
        """Commanded position and velocity (deg, deg/s) at ``now``. Caller holds the lock."""
        s = (now - self._t1) / self._dur
        if s >= 1.0:
            return dict(self._p1), dict.fromkeys(self.motors, 0.0)
        s = max(s, 0.0)
        pos = {m: self._p0[m] + (self._p1[m] - self._p0[m]) * s for m in self.motors}
        vel = {m: (self._p1[m] - self._p0[m]) / self._dur for m in self.motors}
        return pos, vel

    def _run(self) -> None:
        bus = self.bus
        period = 1.0 / self.hz
        recv_to_motor = {bus._get_motor_recv_id(m): m for m in self.motors}
        frames = {m: (bus._get_motor_id(m), bus._motor_types[m]) for m in self.motors}
        next_t = time.perf_counter()
        prev = next_t
        try:
            while self._running:
                now = time.perf_counter()
                self.max_period_s = max(self.max_period_s, now - prev) if self.cycles else 0.0
                prev = now
                with self._lock:
                    pos, vel = self._reference(now)
                # late replies from the last cycle first, so they are not taken as this one's
                while (late := bus.canbus.recv(timeout=0)) is not None:
                    self._absorb(recv_to_motor.get(late.arbitration_id), late, now)
                for m in self.motors:
                    kp, kd = self.gains[m]
                    mid, mtype = frames[m]
                    data = bus._encode_mit_packet(mtype, kp, kd, pos[m], vel[m], 0.0)
                    bus.canbus.send(
                        can.Message(arbitration_id=mid, data=data, is_extended_id=False, is_fd=bus.use_can_fd)
                    )
                got = bus._recv_all_responses(list(recv_to_motor), timeout=min(0.6 * period, 0.003))
                now = time.perf_counter()
                for rid, msg in got.items():
                    self._absorb(recv_to_motor.get(rid), msg, now)
                silent = [m for m, t in self._last_reply.items() if now - t > self.stale_s]
                if silent:
                    raise ConnectionError(
                        f"{self.name}: {', '.join(silent)} stopped answering for over {self.stale_s:.1f} s "
                        "-- check that motor's cable"
                    )
                self.cycles += 1
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()  # fell behind: do not try to catch up in a burst
        except Exception as e:  # noqa: BLE001 - surfaced to the main loop via .error
            self.error = e
            logger.error(f"{self.name}: MIT control loop stopped: {e}")

    def _absorb(self, motor: str | None, msg: can.Message, now: float) -> None:
        if motor is None:
            return
        self.bus._process_response(motor, msg)
        with self._lock:
            self._state[motor] = dict(self.bus._last_known_states[motor])
            self._last_reply[motor] = now
