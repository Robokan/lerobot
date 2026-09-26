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
# ego/wrist cameras matching the real rig). Override via `--robot.model_path=...`.
DEFAULT_MODEL_PATH = "~/sparkpack/openarm_mujoco/v1/scene.xml"


def _default_cameras() -> dict[str, CameraConfig]:
    # Keys are the dataset feature names; `mujoco_name` is the <camera> element
    # in the model. Matches the real OpenArm ego + left/right wrist layout.
    return {
        "ego": MujocoCameraConfig(mujoco_name="ego_camera", fps=50, width=640, height=480),
        "left_wrist": MujocoCameraConfig(
            mujoco_name="left_wrist_camera", fps=50, width=640, height=480
        ),
        "right_wrist": MujocoCameraConfig(
            mujoco_name="right_wrist_camera", fps=50, width=640, height=480
        ),
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

    # Elbow bend (deg) the sim arms start at. Defaults to 0 -- straight -- to match
    # the real robot, which powers up with the elbow extended. The teleoperator's
    # springs are what bend it (see VRMocapConfig.rest_elbow_bend_deg); starting
    # bent here would hide whether that actually works.
    start_elbow_bend_deg: float = 0.0
    # Where both arms START, all seven joints (J1..J7, degrees). The left arm
    # is mirrored so the same numbers mean the same posture on both. This is
    # also the pose 'h' teleports back to, and the one the VR reference is
    # captured against. None = the old behaviour (zeros, elbow from
    # start_elbow_bend_deg).
    start_pose_deg: list[float] | None = None

    # Open an interactive on-screen MuJoCo viewer (mujoco.viewer.launch_passive)
    # so the sim can be watched while teleoperating. Off by default: recording
    # runs are headless, and the window costs a GL context plus frame time.
    # Needs a display; when enabled, MUJOCO_GL defaults to glx instead of egl
    # (egl is offscreen-only and cannot present a window).
    viewer: bool = False

    # Disable all MuJoCo contacts. Default False so arms collide with the table,
    # themselves, and the cube (for grasping). Set True only for unit tests /
    # kinematic debugging that need contacts off.
    disable_collisions: bool = False

    # Collide with the table (its top and legs). False leaves every OTHER
    # contact alone -- bars, the cube, the grippers, self-collision -- so you
    # can still pick things up while the arm passes through the table. Use it
    # to test arm motion without the table stopping the arm and holding the
    # command back. `disable_collisions` above kills ALL contacts instead.
    table_collisions: bool = True

    # PD gains for the 7 arm joints (J1..J7), applied as torque on the model's
    # direct-drive `motor` actuators (tau = kp*(target-q) - kd*qdot, clamped to
    # the model forcerange). Mirrors the real follower's MIT-control gains.
    arm_kp: list[float] = field(default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0])
    arm_kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 3.0, 5.0, 0.3, 0.3, 0.3])
    # Reflected rotor inertia added to each arm joint (kg m^2, J1..J7), a
    # sim-only stand-in for the geared motors the model leaves out (its joints
    # carry no armature). Without it the roll joints -- a slender link turning
    # about its own axis has almost no inertia -- sit at the stability edge of
    # the explicit 500 Hz PD above and ring: 150-300 direction reversals per
    # roll gesture with a perfectly smooth command. 0.005 (about a 10:1-geared
    # wrist motor's reflected inertia) brought that to zero on every joint with
    # unchanged end states; larger values gained nothing. None = model as is.
    arm_armature: list[float] | None = field(default_factory=lambda: [0.005] * 7)

    # Gains for finger joints that are driven as `motor` (torque) actuators in
    # the model (the left fingers). Position-type finger actuators (right
    # fingers) are commanded directly with the target opening in meters.
    # Left-arm fingers are torque actuators in the model; keep these stiff so a
    # closed gripper keeps squeezing when an object blocks the travel.
    finger_kp: float = 5000.0
    finger_kd: float = 80.0
