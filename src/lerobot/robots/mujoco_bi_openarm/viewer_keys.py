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


def drain_recording_controls() -> list[str]:
    """Return and clear pending viewer recording-control requests."""
    with _lock:
        keys = list(_recording_controls)
        _recording_controls.clear()
    return keys
