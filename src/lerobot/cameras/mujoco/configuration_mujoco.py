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

from dataclasses import dataclass

from ..configs import CameraConfig

# NOTE: This module must stay import-safe without `mujoco` installed (it is only
# a dataclass), so Draccus can discover the "mujoco" camera choice even in
# environments that lack the `openarm-sim` extra. The actual MuJoCo import lives
# in ``mujoco_camera.py`` and is deferred to ``connect()``.


@CameraConfig.register_subclass("mujoco")
@dataclass
class MujocoCameraConfig(CameraConfig):
    """Configuration for a camera rendered from a MuJoCo simulation.

    A :class:`MujocoCamera` renders a named ``<camera>`` from a MuJoCo model.
    When created by :class:`~lerobot.robots.mujoco_bi_openarm.MujocoBiOpenArm`
    the camera shares the robot's live ``MjModel``/``MjData`` so frames reflect
    the current simulation state. When created standalone (e.g. via
    :func:`make_cameras_from_configs`) it loads its own static model from
    ``model_path`` and renders the default configuration.

    Args:
        mujoco_name: Name of the ``<camera>`` element in the MuJoCo XML to render
            (e.g. ``"front_camera"``). If empty, the model's default free camera
            (id ``-1``) is used.
        model_path: Optional path to a MuJoCo XML, used only in standalone mode.
        fps/width/height: Inherited; required when used inside a robot so the
            recorded dataset features have a concrete image shape.
    """

    mujoco_name: str = ""
    model_path: str | None = None
