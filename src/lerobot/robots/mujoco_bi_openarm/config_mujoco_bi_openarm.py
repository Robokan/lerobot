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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.mujoco import MujocoCameraConfig

from ..config import RobotConfig

# Default MuJoCo scene shipped in the openarm_mujoco repo (arms + floor +
# front/side tracking cameras). Override via `--robot.model_path=...`.
DEFAULT_MODEL_PATH = "~/sparkpack/openarm_mujoco/v1/scene.xml"


def _default_cameras() -> dict[str, CameraConfig]:
    # Keys are the dataset feature names; `mujoco_name` is the <camera> element
    # in the model. The v1 scene only ships front/side fixed cameras (a richer
    # ego/wrist scene is a later milestone, per the plan's "out of scope").
    return {
        "front": MujocoCameraConfig(mujoco_name="front_camera", fps=50, width=640, height=480),
        "side": MujocoCameraConfig(mujoco_name="side_camera", fps=50, width=640, height=480),
    }


@RobotConfig.register_subclass("mujoco_bi_openarm")
@dataclass(kw_only=True)
class MujocoBiOpenArmConfig(RobotConfig):
    """Configuration for the MuJoCo-simulated bimanual OpenArm follower.

    Matches the :class:`~lerobot.robots.bi_openarm_follower.BiOpenArmFollower`
    feature contract (16 right-first ``<side>_<motor>.pos`` keys in degrees) so
    recordings share the schema of the real chocolate datasets. The robot owns a
    dynamic ``MjModel``/``MjData`` and drives the arms with a Python PD law on the
    torque actuators + position control on the finger actuators.
    """

    id: str | None = "mujoco_bi_openarm"

    # Path to the MuJoCo scene XML to load (dynamic sim model).
    model_path: str = DEFAULT_MODEL_PATH

    # Rendered cameras. Each MujocoCameraConfig.mujoco_name must name a <camera>
    # in the model; the dict key becomes the dataset image feature name.
    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)

    # Control rate (Hz). The number of mj_step substeps per send_action is
    # auto-derived as round((1/fps) / model.timestep) unless sim_substeps is set.
    fps: int = 50
    sim_substeps: int | None = None

    # PD gains for the 7 arm joints (J1..J7), applied as torque on the model's
    # direct-drive `motor` actuators (tau = kp*(target-q) - kd*qdot, clamped to
    # the model forcerange). Mirrors the real follower's MIT-control gains.
    arm_kp: list[float] = field(default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0])
    arm_kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 3.0, 5.0, 0.3, 0.3, 0.3])

    # Gains for finger joints that are driven as `motor` (torque) actuators in
    # the model (the left fingers). Position-type finger actuators (right
    # fingers) are commanded directly with the target opening in meters.
    finger_kp: float = 2000.0
    finger_kd: float = 50.0
