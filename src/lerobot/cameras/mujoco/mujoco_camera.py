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

"""A :class:`~lerobot.cameras.camera.Camera` backed by a MuJoCo offscreen renderer.

The camera renders a named ``<camera>`` from a MuJoCo model into an RGB
``(H, W, 3)`` uint8 array, matching the contract every other lerobot camera
exposes. It is designed to be driven by :class:`MujocoBiOpenArm`, which shares
its live ``MjModel``/``MjData`` so the rendered frames track the simulation
(joint motion, gripper, contacts) exactly as the recorder reads joint state.

Headless rendering requires an OpenGL backend; set ``MUJOCO_GL=egl`` (preferred
on a server/Spark) or ``osmesa`` before importing this module. The launcher
``scripts/run_vr_sim.sh`` sets it for you.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..camera import Camera
from .configuration_mujoco import MujocoCameraConfig

logger = logging.getLogger(__name__)


class MujocoCamera(Camera):
    """Renders a named MuJoCo camera as an RGB frame.

    Two construction modes:

    * **Shared** (used by the robot): pass the robot's ``model`` and ``data`` so
      this camera renders the live sim. The camera does NOT own / step the sim;
      it only owns its offscreen :class:`mujoco.Renderer`.
    * **Standalone** (used by :func:`make_cameras_from_configs`): omit
      ``model``/``data``; the camera loads its own static model from
      ``config.model_path`` and renders the default configuration.
    """

    def __init__(
        self,
        config: MujocoCameraConfig,
        model: Any | None = None,
        data: Any | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.mujoco_name = config.mujoco_name

        # Shared sim handles (None in standalone mode -> loaded on connect()).
        self._model = model
        self._data = data
        self._owns_model = model is None

        self._renderer = None
        self._camera_id: int = -1
        self._connected = False

        if self.width is None or self.height is None:
            raise ValueError(
                "MujocoCameraConfig requires explicit `width` and `height` "
                "so the recorded dataset has a concrete image shape."
            )

    def __str__(self) -> str:
        return f"MujocoCamera({self.mujoco_name or 'free'}@{self.width}x{self.height})"

    def bind_shared_model(self, model: Any, data: Any) -> None:
        """Attach a robot's live ``MjModel``/``MjData`` before :meth:`connect`.

        Lets the owning robot construct camera objects in ``__init__`` (so
        ``len(robot.cameras)`` is correct before ``connect()`` — the record loop
        sizes its async image-writer thread pool from it) while still rendering
        the live shared sim. Must be called before ``connect()``.
        """
        if self._connected:
            raise RuntimeError("bind_shared_model() must be called before connect().")
        self._model = model
        self._data = data
        self._owns_model = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # MuJoCo cameras are defined in the model, not discovered from hardware.
        return []

    def connect(self, warmup: bool = True) -> None:
        if self._connected:
            raise RuntimeError(f"{self} is already connected.")

        import mujoco  # lazy: keep module import-safe without the openarm-sim extra

        if "MUJOCO_GL" not in os.environ:
            # Default to EGL for headless offscreen rendering. Override with
            # MUJOCO_GL=osmesa or =glfw if EGL is unavailable.
            os.environ["MUJOCO_GL"] = "egl"

        if self._owns_model:
            if not self.config.model_path:
                raise ValueError(
                    "Standalone MujocoCamera requires `model_path` in its config "
                    "(or attach it to a MujocoBiOpenArm robot which shares its model)."
                )
            model_path = str(Path(self.config.model_path).expanduser())
            self._model = mujoco.MjModel.from_xml_path(model_path)
            self._data = mujoco.MjData(self._model)
            mujoco.mj_forward(self._model, self._data)

        if self.mujoco_name:
            self._camera_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, self.mujoco_name
            )
            if self._camera_id < 0:
                raise ValueError(
                    f"Camera '{self.mujoco_name}' not found in the MuJoCo model. "
                    f"Available: {[mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self._model.ncam)]}"
                )
        else:
            self._camera_id = -1  # free/default camera

        self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
        self._connected = True

        if warmup:
            self.read()

        logger.debug("%s connected (MUJOCO_GL=%s).", self, os.environ.get("MUJOCO_GL"))

    def read(self) -> NDArray[Any]:
        """Render the current sim state from this camera as RGB ``(H, W, 3)`` uint8."""
        if not self._connected or self._renderer is None:
            raise RuntimeError(f"{self} is not connected.")
        self._renderer.update_scene(self._data, camera=self._camera_id)
        frame = self._renderer.render()
        # mujoco.Renderer already returns a contiguous uint8 RGB array; copy so
        # downstream async writers don't alias the renderer's internal buffer.
        return np.asarray(frame, dtype=np.uint8).copy()

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        # Rendering is synchronous and cheap; there is no background thread.
        return self.read()

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        return self.read()

    def disconnect(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001 - EGL teardown can be noisy at exit
                logger.debug("%s renderer.close() raised during disconnect", self, exc_info=True)
            self._renderer = None
        if self._owns_model:
            self._model = None
            self._data = None
        self._connected = False
