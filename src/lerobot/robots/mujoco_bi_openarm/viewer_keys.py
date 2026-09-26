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

"""Shared key inbox between the MuJoCo viewer window and keyboard teleop.

When ``VIEWER=1``, focus is usually on the MuJoCo window — so stdin-based
keyboard teleop sees nothing. The viewer pushes GLFW keycodes here; 
:class:`~lerobot.teleoperators.vr_mocap.pose_source.KeyboardPoseSource` drains
them each tick.
"""

from __future__ import annotations

import threading

import numpy as np
from collections import deque

# GLFW letter keycodes are uppercase ASCII (e.g. GLFW_KEY_E == 69 == ord('E')).
_GLFW_TAB = 258
_GLFW_SPACE = 32

_lock = threading.Lock()
_queue: deque[str] = deque(maxlen=256)
# How many viewer-camera cycle requests are pending (KeyboardPoseSource `c`).
_camera_cycles = 0
# Recording control keys from the MuJoCo viewer (y/t/n/q) for lerobot-record.
_recording_controls: deque[str] = deque(maxlen=32)


def glfw_key_to_char(keycode: int) -> str | None:
    """Map a MuJoCo/GLFW keycode to the teleop's single-character alphabet."""
    if keycode == _GLFW_TAB:
        return "\t"
    if keycode == _GLFW_SPACE:
        return " "
    if 65 <= keycode <= 90:  # A-Z
        return chr(keycode).lower()
    if 48 <= keycode <= 57:  # 0-9
        return chr(keycode)
    if keycode in (91, 93):  # [ ]
        return chr(keycode)
    if keycode in (59,):  # ;
        return chr(keycode)
    return None


def push_glfw_key(keycode: int) -> None:
    ch = glfw_key_to_char(int(keycode))
    if ch is None:
        return
    with _lock:
        _queue.append(ch)


def drain_keys() -> list[str]:
    with _lock:
        keys = list(_queue)
        _queue.clear()
    return keys


def request_cycle_camera() -> None:
    """Ask the MuJoCo viewer to advance to the next fixed camera (or free)."""
    global _camera_cycles
    with _lock:
        _camera_cycles += 1


def drain_camera_cycles() -> int:
    """Return and clear pending viewer-camera cycle requests."""
    global _camera_cycles
    with _lock:
        n = _camera_cycles
        _camera_cycles = 0
    return n


def request_recording_control(control: str) -> None:
    """Queue a recording control for :func:`lerobot_record.record_loop`.

    ``control`` is a logical name consumed by
    :func:`lerobot.utils.keyboard_input.apply_recording_control` (``y``, ``t``,
    ``right``, ``esc``, …).
    """
    with _lock:
        _recording_controls.append(control)


# Text shown INSIDE the headset, burned onto the camera frames the OpenXR view
# composites (never onto the frames the dataset records). The scene robot sets
# the prompt line at each new episode; the record loop sets the status line as
# recording starts and stops. A person in a headset cannot see the terminal,
# and "which pad?" and "am I recording?" are the two things they need to know.
_hud: dict[str, str] = {"prompt": "", "status": ""}


def set_hud_text(prompt: str | None = None, status: str | None = None) -> None:
    with _lock:
        if prompt is not None:
            _hud["prompt"] = prompt
        if status is not None:
            _hud["status"] = status


def get_hud_text() -> tuple[str, str]:
    with _lock:
        return _hud["prompt"], _hud["status"]


# Teleport request (teleop -> sim robot): on the next send_action the robot
# SETS its joints to the commanded targets instead of driving toward them, so
# an "h" home is a jump, not a swing across the table.
_teleport: dict[str, bool] = {"pending": False}


def request_teleport() -> None:
    with _lock:
        _teleport["pending"] = True


def take_teleport() -> bool:
    with _lock:
        v = _teleport["pending"]
        _teleport["pending"] = False
    return v


# Commanded-pose markers (teleop -> viewer): the IK target pose per hand, drawn
# as an x/y/z triad by the robot when the viewer syncs. m toggles them.
_markers: dict = {"enabled": False, "targets": {}}
_arm_force: dict = {"left": 0.0, "right": 0.0}


def set_target_marker(side: str, pos, quat) -> None:
    with _lock:
        _markers["targets"][side] = (np.asarray(pos, dtype=float).copy(), np.asarray(quat, dtype=float).copy())


def draw_markers_into(scn, mujoco, np) -> int:
    """Append the commanded-pose triads to an mjvScene; returns geoms added.

    Shared by the desktop viewer and the offscreen cameras. The viewer has its
    own user_scn, but the CAMERA renders are what the headset actually shows,
    and user_scn never reaches them -- markers drawn only there are invisible
    to anyone wearing the headset.
    """
    enabled, targets = get_markers()
    if not enabled:
        return 0
    added = 0
    for _side, (pos, quat) in targets.items():
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        for axis, rgba in ((0, (0.95, 0.2, 0.15, 0.95)),
                           (1, (0.2, 0.85, 0.2, 0.95)),
                           (2, (0.2, 0.4, 0.95, 0.95))):
            if scn.ngeom >= scn.maxgeom:
                return added
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array(rgba))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, pos, pos + 0.05 * R[:, axis])
            scn.ngeom += 1
            added += 1
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.008),
                                np.asarray(pos, dtype=float), np.eye(3).flatten(),
                                np.array([1.0, 1.0, 1.0, 0.9]))
            scn.ngeom += 1
            added += 1
    return added


def set_arm_force(side: str, newtons: float) -> None:
    """Publish the contact force estimate for one arm (N)."""
    with _lock:
        _arm_force[side] = float(newtons)


def get_arm_forces() -> dict:
    with _lock:
        return dict(_arm_force)


def set_markers(on: bool) -> None:
    """Turn the commanded-pose markers on or off explicitly (the 'm' key
    toggles them, but there is no keyboard in the headset)."""
    with _lock:
        _markers["enabled"] = bool(on)


def toggle_markers() -> bool:
    with _lock:
        _markers["enabled"] = not _markers["enabled"]
        return _markers["enabled"]


def get_markers():
    with _lock:
        return (_markers["enabled"], dict(_markers["targets"]))


def drain_recording_controls() -> list[str]:
    """Return and clear pending viewer recording-control requests."""
    with _lock:
        keys = list(_recording_controls)
        _recording_controls.clear()
    return keys
