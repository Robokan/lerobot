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
import math
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .ik import (
    FINGER_OPEN_M,
    axis_angle_to_quat,
    hand_pos_from_tcp,
    quat_inv,
    quat_mul,
    quat_rotate,
    tcp_pos_from_hand,
)

logger = logging.getLogger(__name__)

# Per-keypress increments. SparkJAX used 0.04 rad/tick while a key was *held*
# at ~50 Hz; our keyboard driver applies one step per discrete press/repeat, so
# rotation needs a larger step or i/k/j/l/u/o feel almost still.
POS_STEP = 0.005
# A held movement key is a steady velocity: each repeat event arms the key for
# this many ticks and every tick applies 1/KEY_HOLD_TICKS of the step.
KEY_HOLD_TICKS = 6
_VELOCITY_KEYS = set("wsadrfikjluo[]")
ROT_STEP = 0.15  # ~8.6 deg per press
GRIP_STEP = 0.005

# Key-repeat / viewer flood can enqueue many identical chars between teleop
# ticks. Applying every one produces a 5–10 cm target jump and the IK rate
# limit then looks like a pop. Cap the net delta applied per get_targets call.
MAX_POS_DELTA_PER_TICK_M = 0.010  # 2 x POS_STEP
MAX_ROT_DELTA_PER_TICK_RAD = ROT_STEP  # one press-worth of body rotation
MAX_GRIP_DELTA_PER_TICK_M = GRIP_STEP

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
# How far an integrated target may run ahead of the pose the arm actually
# reached, before it is pulled back. Big enough that normal tracking lag is
# untouched, small enough that reversing a key responds immediately.
TARGET_LEASH_M = 0.05
# Orientation leash used to be 15 deg, which capped keyboard pitch as soon as
# IK lagged even slightly. Keep a looser cap so i/k/j/l/u/o can accumulate.
TARGET_LEASH_RAD = np.deg2rad(60.0)


def _leash_to_actual(pos, quat, actual):
    """Clamp a target pose to within the leash distance of the achieved pose."""
    act_pos, act_quat = actual
    act_pos = np.asarray(act_pos, dtype=float)
    act_quat = np.asarray(act_quat, dtype=float)

    delta = pos - act_pos
    dist = float(np.linalg.norm(delta))
    if dist > TARGET_LEASH_M:
        pos = act_pos + delta * (TARGET_LEASH_M / dist)

    # Relative rotation from achieved to target, shrunk if it exceeds the leash.
    q_rel = quat_mul(quat, quat_inv(act_quat))
    w = float(np.clip(abs(q_rel[0]), -1.0, 1.0))
    angle = 2.0 * float(np.arccos(w))
    if angle > TARGET_LEASH_RAD:
        axis = q_rel[1:] * (1.0 if q_rel[0] >= 0.0 else -1.0)
        n = float(np.linalg.norm(axis))
        if n > 1e-9:
            q_rel = axis_angle_to_quat(axis / n, TARGET_LEASH_RAD)
            quat = quat_mul(q_rel, act_quat)
    return pos, quat


class KeyboardPoseSource(PoseSource):
    """Drives the active hand's EE target from single-character keyboard input.

    Keys are taken from stdin (terminal focus) and, when the MuJoCo viewer is
    open, from the viewer window via :mod:`viewer_keys`. On a non-interactive
    stdin (piped/headless CI) with no viewer it degrades to holding the reset
    pose. Keys::

        w / s   +x / -x          i / k   pitch +/- (rot Y)
        a / d   +y / -y          j / l   yaw   +/- (rot Z)
        r / f   +z / -z          u / o   roll  +/- (rot X)
        [ / ]   gripper open/close
        tab     switch active hand        space  reset targets to current pose
        c       cycle viewer camera (ego / right_wrist / left_wrist / free)
        y / t   start / stop recording (when using lerobot-record)
        n / q   end episode early / quit recording
    """

    # When set (by VRMocap from config.chain_rotation), the rotation keys do
    # not rotate the IK target about a pinned wrist; they queue a world-frame
    # rotation request that the teleop splits across the joints as a spring
    # chain. The target then re-syncs to wherever the hand ended up.
    chain_rotation: bool = False
    # Rotation keys pivot about the gripper tip (TCP); translation keys move it.
    rotate_about_tip: bool = True

    def __init__(self):
        self._pos: dict[str, np.ndarray] = {}
        self._quat: dict[str, np.ndarray] = {}
        self._rot_request: dict[str, tuple[np.ndarray, float]] = {}
        self._home_request: set[str] = set()
        self._held: dict[str, int] = {}
        self._rot_anchor: dict[str, np.ndarray | None] = {}
        self._rot_anchor_ttl: dict[str, int] = {}
        self._grip: dict[str, float] = {s: 0.0 for s in SIDES}
        self._active_side = "right"
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    _HELP = """
============================================================
  Keyboard teleop  (active hand: {hand})
============================================================
  w / s     +x / -x (forward / back)
  a / d     +y / -y (left / right)
  r / f     +z / -z (up / down)
  i / k     pitch +/-
  j / l     yaw   +/-
  u / o     roll  +/-
  [ / ]     gripper open / close
  Tab       switch active hand
  Space     reset targets to current pose
  h         return the arm to the default (launch) pose
  m         show / hide x,y,z markers at the commanded gripper poses
  c         cycle viewer cam (ego / right / left / free)
  y / t     start / stop recording (record mode)
  n / q     end episode early / quit recording

  With VIEWER=1, focus the MuJoCo window for keys.
============================================================
""".strip()

    def start(self):
        print(self._HELP.format(hand=self._active_side.upper()), flush=True)
        if not sys.stdin or not sys.stdin.isatty():
            logger.warning(
                "KeyboardPoseSource: stdin is not a TTY; use the MuJoCo viewer "
                "window for keys (VIEWER=1), or driver='scripted' for headless."
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

    def resync_target(self, side, pos, quat):
        """After a chain step the hand has moved: put the target ON it, so IK
        holds the new pose instead of dragging the hand back a step."""
        self._pos[side] = np.asarray(pos, dtype=float).copy()
        self._quat[side] = np.asarray(quat, dtype=float).copy()

    def reset_grip(self, side, grip_m):
        self._grip[side] = float(grip_m)

    def take_home_request(self, side) -> bool:
        """True once if h was pressed for this hand since the last check."""
        if side in self._home_request:
            self._home_request.discard(side)
            return True
        return False

    def rotation_active(self, side) -> bool:
        """True while a rotation gesture (i/k/j/l/u/o held) is in progress for
        ``side`` -- the ticks the tip anchor is alive."""
        return self._rot_anchor_ttl.get(side, 0) > 0

    def take_rotation_request(self, side):
        """(axis_world, angle) queued by the rotation keys this tick, or None."""
        axis, ang = self._rot_request.pop(side, (None, 0.0))
        if axis is None or abs(ang) < 1e-12:
            return None
        ang = math.copysign(min(abs(ang), MAX_ROT_DELTA_PER_TICK_RAD), ang)
        return axis / max(float(np.linalg.norm(axis)), 1e-9), ang

    def _drain_keys(self) -> list[str]:
        with self._lock:
            keys = list(self._queue)
            self._queue.clear()
        # Keys typed into the MuJoCo viewer window (when VIEWER=1).
        try:
            from lerobot.robots.mujoco_bi_openarm.viewer_keys import drain_keys as drain_viewer_keys

            keys.extend(drain_viewer_keys())
        except Exception:  # noqa: BLE001
            pass
        return keys

    def get_targets(self, current_ee):
        if not self._pos:
            self.reset(current_ee)

        side = self._active_side
        pos = self._pos.setdefault(side, np.asarray(current_ee[side][0], dtype=float).copy())
        quat = self._quat.setdefault(side, np.asarray(current_ee[side][1], dtype=float).copy())

        # Keep the integrated target on a leash behind what the arm actually
        # achieved. Holding a key past the arm's reach would otherwise wind the
        # target far beyond it, and releasing/reversing would then do nothing
        # until the target walked all the way back -- a long dead zone that feels
        # like the controls have stopped responding.
        pos, quat = _leash_to_actual(pos, quat, current_ee[side])
        self._pos[side], self._quat[side] = pos, quat

        act_pos = np.asarray(current_ee[side][0], dtype=float)
        pos_before = pos.copy()
        grip_before = self._grip[side]
        # Accumulate requested body-fixed rotation as an axis-angle in the hand
        # frame, then apply once (capped) so a key-repeat flood cannot wind the
        # orientation target many presses ahead of IK in a single tick.
        rot_axis_angle = np.zeros(3)
        rot_synced = False
        # Pivot for i/k/j/l/u/o. The operator wants the gripper TIP (the TCP,
        # 8 cm out from the hand) to be the point that stays put while the
        # gripper turns, and the point the translation keys move. The older
        # wrist pivot is kept behind rotate_about_tip=False.
        pivot_hand = hand_pos_from_tcp(act_pos, np.asarray(current_ee[side][1], dtype=float))

        def _body_rot(local_axis: np.ndarray, angle: float, world_axis: np.ndarray | None = None) -> None:
            """Queue a body-fixed rotation about the hand / wrist origin.

            World-fixed pitch about +Y is singular at the hang pose and IK
            rejects it; body-fixed axes track. The hand pivot stays fixed and
            the TCP target is rewritten to match the new orientation.

            In chain mode a ``world_axis`` overrides the body axis: yaw is
            about VERTICAL, as the key help says ("rot Z"), not about the
            hand's own axis. With the arm hanging, the hand's axis points
            forward and the base yaw joint is perpendicular to it -- so a
            body-fixed yaw could only ever roll the forearm, and the shoulder
            never turned with the wrist.
            """
            nonlocal pos, quat, rot_synced
            if self.chain_rotation:
                if world_axis is not None:
                    axis_w = np.asarray(world_axis, dtype=float)
                else:
                    axis_w = quat_rotate(np.asarray(current_ee[side][1], dtype=float), local_axis)
                # keep the axis and a SIGNED angle: the sign is the key direction
                # (L negative, J positive) and decides the joint order
                prev_axis, prev_ang = self._rot_request.get(side, (axis_w, 0.0))
                self._rot_request[side] = (axis_w, prev_ang + angle)
                if not rot_synced:            # keep the target ON the hand, not ahead of it
                    quat = np.asarray(current_ee[side][1], dtype=float).copy()
                    pos = np.asarray(current_ee[side][0], dtype=float).copy()
                    self._pos[side], self._quat[side] = pos, quat
                    rot_synced = True
                return
            if not rot_synced:
                quat = np.asarray(current_ee[side][1], dtype=float).copy()
                pos = tcp_pos_from_hand(pivot_hand, quat)
                self._pos[side] = pos
                self._quat[side] = quat
                rot_synced = True
            rot_axis_angle[:3] += local_axis * angle

        keys = self._drain_keys()
        if keys and os.environ.get("VR_TELEOP_DEBUG"):
            print(f"[teleop] keys drained: {keys}", flush=True)
        ttl = self._rot_anchor_ttl.get(side, 0)
        if ttl > 0:
            self._rot_anchor_ttl[side] = ttl - 1
        elif self._rot_anchor.get(side) is not None:
            self._rot_anchor[side] = None          # gesture over: next rotation re-anchors
        # held-key velocity: (re)arm on each event, apply a fraction every tick
        for ch in keys:
            if ch in _VELOCITY_KEYS:
                self._held[ch] = KEY_HOLD_TICKS
        keys = [ch for ch in keys if ch not in _VELOCITY_KEYS]
        for ch, left in list(self._held.items()):
            if left <= 0:
                del self._held[ch]
                continue
            self._held[ch] = left - 1
            keys.append(ch)
        for ch in keys:
            if ch == "w":
                pos[0] += POS_STEP / KEY_HOLD_TICKS
            elif ch == "s":
                pos[0] -= POS_STEP / KEY_HOLD_TICKS
            elif ch == "a":
                pos[1] += POS_STEP / KEY_HOLD_TICKS
            elif ch == "d":
                pos[1] -= POS_STEP / KEY_HOLD_TICKS
            elif ch == "r":
                pos[2] += POS_STEP / KEY_HOLD_TICKS
            elif ch == "f":
                pos[2] -= POS_STEP / KEY_HOLD_TICKS
            elif ch == "i":
                _body_rot(np.array([0.0, 1.0, 0.0]), ROT_STEP / KEY_HOLD_TICKS)
            elif ch == "k":
                _body_rot(np.array([0.0, 1.0, 0.0]), -ROT_STEP / KEY_HOLD_TICKS)
            # world axis is -Z so that L turns the same way it did before the
            # yaw moved from the hand's axis to vertical (the operator's frame)
            elif ch == "m":
                try:
                    from lerobot.robots.mujoco_bi_openarm.viewer_keys import toggle_markers
                    print(f"[teleop] target markers {'ON' if toggle_markers() else 'off'}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
            elif ch == "h":
                self._home_request.update(SIDES)      # session reset: BOTH arms to default
            elif ch == "j":
                _body_rot(np.array([0.0, 0.0, 1.0]), ROT_STEP / KEY_HOLD_TICKS)
            elif ch == "l":
                _body_rot(np.array([0.0, 0.0, 1.0]), -ROT_STEP / KEY_HOLD_TICKS)
            elif ch == "u":
                _body_rot(np.array([1.0, 0.0, 0.0]), ROT_STEP / KEY_HOLD_TICKS)
            elif ch == "o":
                _body_rot(np.array([1.0, 0.0, 0.0]), -ROT_STEP / KEY_HOLD_TICKS)
            elif ch == "[":
                self._grip[side] = min(self._grip[side] + GRIP_STEP / KEY_HOLD_TICKS, FINGER_OPEN_M)
            elif ch == "]":
                self._grip[side] = max(self._grip[side] - GRIP_STEP / KEY_HOLD_TICKS, 0.0)
            elif ch in ("\t", ";"):
                self._active_side = "left" if side == "right" else "right"
                logger.info("Active hand: %s", self._active_side.upper())
            elif ch == "c":
                # Cycle the MuJoCo viewer through ego / wrist cams (VR X/A analog).
                try:
                    from lerobot.robots.mujoco_bi_openarm.viewer_keys import (
                        request_cycle_camera,
                    )

                    request_cycle_camera()
                except Exception:  # noqa: BLE001
                    logger.debug("viewer camera cycle request failed", exc_info=True)
            elif ch in ("y", "t", "n", "q"):
                # Forward recording controls to lerobot-record (viewer has focus).
                # Note: teleop `r` (+z) is NOT remapped — use Left arrow for re-record.
                try:
                    from lerobot.robots.mujoco_bi_openarm.viewer_keys import (
                        request_recording_control,
                    )

                    control = {"y": "y", "t": "t", "n": "right", "q": "esc"}[ch]
                    request_recording_control(control)
                except Exception:  # noqa: BLE001
                    logger.debug("recording control request failed", exc_info=True)
            elif ch == " ":
                self.reset(current_ee)
                pos = self._pos[side]
                quat = self._quat[side]
                pos_before = pos.copy()
                grip_before = self._grip[side]
                rot_axis_angle[:] = 0.0
                rot_synced = False

        # Clamp net translation / gripper so a held key cannot jump the target.
        if not rot_synced:
            delta = pos - pos_before
            dist = float(np.linalg.norm(delta))
            if dist > MAX_POS_DELTA_PER_TICK_M:
                pos[:] = pos_before + delta * (MAX_POS_DELTA_PER_TICK_M / dist)
        grip_delta = self._grip[side] - grip_before
        if abs(grip_delta) > MAX_GRIP_DELTA_PER_TICK_M:
            self._grip[side] = grip_before + math.copysign(MAX_GRIP_DELTA_PER_TICK_M, grip_delta)

        rot_angle = float(np.linalg.norm(rot_axis_angle))
        if rot_angle > 1e-12:
            if rot_angle > MAX_ROT_DELTA_PER_TICK_RAD:
                rot_axis_angle *= MAX_ROT_DELTA_PER_TICK_RAD / rot_angle
                rot_angle = MAX_ROT_DELTA_PER_TICK_RAD
            quat[:] = quat_mul(quat, axis_angle_to_quat(rot_axis_angle / rot_angle, rot_angle))
            quat[:] = quat / np.linalg.norm(quat)
            if self.rotate_about_tip:
                # The tip stays where the gesture STARTED, not wherever it has
                # drifted to: re-syncing to the actual tip each tick let 1.5 cm
                # of drift accumulate over a 33 deg turn. The anchor is taken on
                # the first rotation tick after a pause and held while keys
                # keep coming, so the solver pulls the tip back every tick.
                if self._rot_anchor.get(side) is None:
                    self._rot_anchor[side] = act_pos.copy()
                self._rot_anchor_ttl[side] = KEY_HOLD_TICKS + 2
                pos[:] = self._rot_anchor[side]
            else:
                # Keep the wrist/hand pivot fixed; tip follows on a sphere about it.
                pos[:] = tcp_pos_from_hand(pivot_hand, quat)
        else:
            quat[:] = quat / np.linalg.norm(quat)

        self._pos[side], self._quat[side] = pos, quat

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
