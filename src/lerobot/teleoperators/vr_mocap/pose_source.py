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

"""Pose sources for the VR motion-capture teleoperator.

A :class:`PoseSource` produces per-hand end-effector targets (in the robot
frame) that the :class:`VRMocap` teleoperator feeds to IK. This decouples the
*input device* from the IK/record plumbing: Phase 1 ships headless drivers
(:class:`KeyboardPoseSource`, :class:`ScriptedPoseSource`) and Phase 2 adds
``OpenXRPoseSource`` — swapping the driver requires no change to the robot or
the record loop.

Convention: each source seeds its per-hand targets from the robot's current TCP
poses (passed to :meth:`reset`) and then reports absolute target poses. For the
keyboard/scripted drivers the "delta-teleop reference capture" is therefore
implicit (the reset pose is the reference); the OpenXR driver performs the
controller-relative reference capture internally on each tracking toggle.

This module is import-safe without ``mujoco``/``pyopenxr`` (numpy + stdlib only;
the OpenXR backend lives in a separate module and imports its deps lazily).
"""

import abc
import logging
import sys
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .ik import FINGER_OPEN_M, axis_angle_to_quat, quat_mul

logger = logging.getLogger(__name__)

# Per-tick increments (match SparkJAX test_vr_ik.py keyboard mode).
POS_STEP = 0.005
ROT_STEP = 0.04
GRIP_STEP = 0.005

SIDES = ("left", "right")


@dataclass
class HandTarget:
    """Target end-effector pose for one hand, in the robot frame."""

    pos: np.ndarray  # (3,)
    quat: np.ndarray  # (4,) wxyz
    gripper_m: float  # finger opening in meters [0, FINGER_OPEN_M]
    active: bool  # whether IK should track this target this step


class PoseSource(abc.ABC):
    """Produces per-hand robot-frame EE targets for the VR teleoperator."""

    def start(self) -> None:  # noqa: B027 - optional hook
        """Begin producing poses (open device / input thread). Default no-op."""

    def stop(self) -> None:  # noqa: B027 - optional hook
        """Stop and release resources. Default no-op."""

    @abc.abstractmethod
    def reset(self, initial_ee: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Seed per-hand targets/references from current TCP poses ``{side: (pos, quat)}``."""

    @abc.abstractmethod
    def get_targets(
        self, current_ee: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, HandTarget]:
        """Return the latest per-hand targets given the robot's current TCP poses."""


# --------------------------------------------------------------------------- #
# Scripted driver (deterministic, fully headless)
# --------------------------------------------------------------------------- #
@dataclass
class ScriptedPoseSource(PoseSource):
    """Drives each hand on a small deterministic Lissajous loop around its start.

    Useful for headless end-to-end tests: the arms move smoothly and the gripper
    oscillates, so a recorded episode is non-trivial without any input device.
    """

    amplitude: float = 0.06  # meters
    period_s: float = 6.0
    fps: float = 30.0

    _t: int = field(default=0, init=False)
    _ref_pos: dict = field(default_factory=dict, init=False)
    _ref_quat: dict = field(default_factory=dict, init=False)

    def reset(self, initial_ee):
        self._t = 0
        self._ref_pos = {s: np.asarray(initial_ee[s][0], dtype=float).copy() for s in initial_ee}
        self._ref_quat = {s: np.asarray(initial_ee[s][1], dtype=float).copy() for s in initial_ee}

    def get_targets(self, current_ee):
        if not self._ref_pos:
            self.reset(current_ee)
        t_s = self._t / float(self.fps)
        self._t += 1
        w = 2.0 * np.pi / self.period_s
        targets: dict[str, HandTarget] = {}
        for k, side in enumerate(SIDES):
            if side not in self._ref_pos:
                continue
            phase = w * t_s + (k * np.pi)  # arms out of phase
            offset = np.array([
                self.amplitude * np.sin(phase),
                self.amplitude * 0.5 * np.sin(2.0 * phase),
                self.amplitude * 0.5 * (1.0 - np.cos(phase)),
            ])
            grip = 0.5 * (1.0 - np.cos(phase)) * FINGER_OPEN_M  # 0 .. open .. 0
            targets[side] = HandTarget(
                pos=self._ref_pos[side] + offset,
                quat=self._ref_quat[side].copy(),
                gripper_m=float(grip),
                active=True,
            )
        return targets


# --------------------------------------------------------------------------- #
# Keyboard driver (headless terminal, single-char stdin)
# --------------------------------------------------------------------------- #
class KeyboardPoseSource(PoseSource):
    """Drives the active hand's EE target from single-character terminal input.

    Runs without a GUI: a background thread reads stdin in cbreak mode. On a
    non-interactive stdin (piped/headless CI) it degrades gracefully to holding
    the reset pose. Keys (mirroring the SparkJAX keyboard test intent)::

        w / s   +x / -x          i / k   pitch +/- (rot Y)
        a / d   +y / -y          j / l   yaw   +/- (rot Z)
        r / f   +z / -z          u / o   roll  +/- (rot X)
        [ / ]   gripper open/close
        tab     switch active hand        space  reset targets to current pose
    """

    def __init__(self):
        self._pos: dict[str, np.ndarray] = {}
        self._quat: dict[str, np.ndarray] = {}
        self._grip: dict[str, float] = {s: 0.0 for s in SIDES}
        self._active_side = "right"
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        if not sys.stdin or not sys.stdin.isatty():
            logger.warning(
                "KeyboardPoseSource: stdin is not a TTY; running in hold mode "
                "(no keyboard input). Use driver='scripted' for headless motion."
            )
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info("KeyboardPoseSource active (active hand: %s).", self._active_side.upper())

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _reader_loop(self):
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                ch = sys.stdin.read(1)
                if ch:
                    with self._lock:
                        self._queue.append(ch)
        except Exception:  # noqa: BLE001
            logger.debug("KeyboardPoseSource reader loop ended", exc_info=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def reset(self, initial_ee):
        self._pos = {s: np.asarray(initial_ee[s][0], dtype=float).copy() for s in initial_ee}
        self._quat = {s: np.asarray(initial_ee[s][1], dtype=float).copy() for s in initial_ee}

    def _drain_keys(self) -> list[str]:
        with self._lock:
            keys = list(self._queue)
            self._queue.clear()
        return keys

    def get_targets(self, current_ee):
        if not self._pos:
            self.reset(current_ee)

        side = self._active_side
        pos = self._pos.setdefault(side, np.asarray(current_ee[side][0], dtype=float).copy())
        quat = self._quat.setdefault(side, np.asarray(current_ee[side][1], dtype=float).copy())

        for ch in self._drain_keys():
            if ch == "w":
                pos[0] += POS_STEP
            elif ch == "s":
                pos[0] -= POS_STEP
            elif ch == "a":
                pos[1] += POS_STEP
            elif ch == "d":
                pos[1] -= POS_STEP
            elif ch == "r":
                pos[2] += POS_STEP
            elif ch == "f":
                pos[2] -= POS_STEP
            elif ch == "i":
                quat[:] = quat_mul(axis_angle_to_quat(np.array([0, 1, 0]), ROT_STEP), quat)
            elif ch == "k":
                quat[:] = quat_mul(axis_angle_to_quat(np.array([0, 1, 0]), -ROT_STEP), quat)
            elif ch == "j":
                quat[:] = quat_mul(axis_angle_to_quat(np.array([0, 0, 1]), ROT_STEP), quat)
            elif ch == "l":
                quat[:] = quat_mul(axis_angle_to_quat(np.array([0, 0, 1]), -ROT_STEP), quat)
            elif ch == "u":
                quat[:] = quat_mul(axis_angle_to_quat(np.array([1, 0, 0]), ROT_STEP), quat)
            elif ch == "o":
                quat[:] = quat_mul(axis_angle_to_quat(np.array([1, 0, 0]), -ROT_STEP), quat)
            elif ch == "[":
                self._grip[side] = min(self._grip[side] + GRIP_STEP, FINGER_OPEN_M)
            elif ch == "]":
                self._grip[side] = max(self._grip[side] - GRIP_STEP, 0.0)
            elif ch in ("\t", ";"):
                self._active_side = "left" if side == "right" else "right"
                logger.info("Active hand: %s", self._active_side.upper())
            elif ch == " ":
                self.reset(current_ee)

        quat[:] = quat / np.linalg.norm(quat)

        targets: dict[str, HandTarget] = {}
        for s in SIDES:
            if s not in self._pos:
                continue
            targets[s] = HandTarget(
                pos=self._pos[s].copy(),
                quat=self._quat[s].copy(),
                gripper_m=self._grip[s],
                active=True,
            )
        return targets
