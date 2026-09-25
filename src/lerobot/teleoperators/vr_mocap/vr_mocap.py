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

"""VR motion-capture teleoperator for the simulated bimanual OpenArm.

Reproduces SparkJAX's "start VR" behaviour inside lerobot: a pose source
provides per-hand end-effector targets, which are solved to joint angles via
damped-least-squares IK on a lightweight FK/IK-only MuJoCo model, then emitted
as the same 16 right-first ``<side>_<motor>.pos`` (degrees) action keys the
:class:`MujocoBiOpenArm` robot consumes. Because lerobot uses identity
processors by default, ``teleop.action_features`` matches
``robot.action_features`` exactly, so the record loop drives the sim and the
recorded dataset shares the real chocolate-dataset schema.

IK lives here (not in the robot) because the recorded ``action`` must be joint
positions; IK therefore runs before the action is recorded, exactly as SparkJAX
does (arms just track joint targets).
"""

import logging
import math
import os

import numpy as np

from lerobot.robots.mujoco_bi_openarm import FINGER_OPEN_M, gripper_m_to_deg
from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_vr_mocap import VRMocapConfig

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
MOTOR_NAMES = ARM_JOINT_NAMES + ["gripper"]
SIDES = ["right", "left"]  # right-first, to match the robot

_RAD2DEG = 180.0 / np.pi


_HUD_FONT: dict = {}


def _hud_font(px: int):
    from PIL import ImageFont

    f = _HUD_FONT.get(px)
    if f is None:
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", px)
        except OSError:
            f = ImageFont.load_default()
        _HUD_FONT[px] = f
    return f


def _with_hud(frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Burn the HUD lines (episode prompt, recording status) onto COPIES of the
    frames bound for the headset. The arrays in ``frames`` are the robot's
    observation images — the same objects the record loop writes to the dataset
    — so they are never drawn on in place."""
    try:
        from lerobot.robots.mujoco_bi_openarm.viewer_keys import get_hud_text
    except Exception:  # noqa: BLE001
        return frames
    prompt, status = get_hud_text()
    if not prompt and not status:
        return frames
    from PIL import Image, ImageDraw

    out = {}
    for key, img in frames.items():
        h, w = img.shape[:2]
        pil = Image.fromarray(np.ascontiguousarray(img[..., :3]))
        draw = ImageDraw.Draw(pil, "RGBA")
        big, small = _hud_font(max(18, w // 24)), _hud_font(max(14, w // 30))
        pad = max(6, w // 80)
        y = pad
        if prompt:
            # Wrap on the " · " separators when the line is wider than the frame
            # (640 px at the ego camera): the pad colour must never be cut off.
            lines = [prompt]
            if draw.textlength(prompt, font=big) > w - 2 * pad and "  ·  " in prompt:
                parts = prompt.split("  ·  ")
                lines = [parts[0] + "  ·  " + parts[1], "  ·  ".join(parts[2:])] if len(parts) > 2 else parts
                if draw.textlength(lines[0], font=big) > w - 2 * pad:
                    lines = parts
            bottom = y
            for line in lines:
                box = draw.textbbox((pad, bottom), line, font=big)
                bottom = box[3] + pad // 2
            draw.rectangle((0, 0, w, bottom + pad // 2), fill=(0, 0, 0, 170))
            yy = y
            for line in lines:
                draw.text((pad, yy), line, font=big, fill=(255, 255, 255, 255))
                yy = draw.textbbox((pad, yy), line, font=big)[3] + pad // 2
            y = bottom + pad // 2
        if status:
            box = draw.textbbox((pad, y + pad // 2), status, font=small)
            rec = status.startswith("●")
            draw.rectangle((0, y, box[2] + pad, box[3] + pad), fill=(200, 40, 30, 200) if rec else (0, 0, 0, 150))
            draw.text((pad, y + pad // 2), status, font=small, fill=(255, 255, 255, 255))
        out[key] = np.asarray(pil)
    return out


class VRMocap(Teleoperator):
    """VR mocap -> IK -> joint-position teleoperator (16 right-first *.pos deg)."""

    config_class = VRMocapConfig
    name = "vr_mocap"

    def __init__(self, config: VRMocapConfig):
        super().__init__(config)
        self.config = config

        self._model = None
        self._data = None
        self._ik = None
        self._source = None
        self._grip_m: dict[str, float] = {s: 0.0 for s in SIDES}
        self._connected = False

    @property
    def action_features(self) -> dict[str, type]:
        # Right first, then left — must equal MujocoBiOpenArm.action_features.
        features: dict[str, type] = {}
        for side in SIDES:
            for motor in MOTOR_NAMES:
                features[f"{side}_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        # OpenXR consumes RGB observation images (ego / wrists) for the headset
        # view; other drivers ignore send_feedback.
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import mujoco

        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        from .ik import IKSolver

        model_path = os.path.expanduser(self.config.model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MuJoCo IK model not found: {model_path}")
        logger.info("Loading IK MuJoCo model: %s", model_path)
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._ik = IKSolver(
            self._model,
            self._data,
            dls_lambda=self.config.dls_lambda,
            joint_weights=self.config.joint_weights,
            limit_margin_rad=math.radians(self.config.limit_margin_deg),
            max_step_rad=math.radians(self.config.max_step_deg),
            spring_weights=self.config.spring_weights,
            spring_gain=self.config.spring_gain,
            handover_deg=self.config.handover_deg,
            homing_boost=self.config.homing_boost,
            rest_elbow_bend_rad=math.radians(self.config.rest_elbow_bend_deg),
        )

        # the DEFAULT pose for the rotation keys: where the arm is at launch
        # clipped to the solver's limits (the elbow is floored at 3 deg), so
        # "default" is a pose the arm can actually reach
        self._default_q = {s: np.clip(self._ik.joint_positions(s), self._ik.limits_low[s], self._ik.limits_high[s])
                           for s in SIDES}
        self._tick = 0
        self._debug_every = int(os.environ.get("VR_TELEOP_DEBUG", "0") or 0)
        self._source = self._make_source()
        if hasattr(self._source, "chain_rotation"):
            self._source.chain_rotation = bool(self.config.chain_rotation)
        if hasattr(self._source, "rotate_about_tip"):
            self._source.rotate_about_tip = bool(self.config.rotate_about_tip)
        self._source.reset({s: self._ik.get_ee_pose(s) for s in SIDES})
        self._source.start()

        self._connected = True
        logger.info("%s connected (driver=%s).", self, self.config.driver)

    def _make_source(self):
        driver = self.config.driver
        if driver == "scripted":
            from .pose_source import ScriptedPoseSource

            return ScriptedPoseSource(
                amplitude=self.config.scripted_amplitude,
                period_s=self.config.scripted_period_s,
                fps=float(self.config.vr_hz),
            )
        if driver == "keyboard":
            from .pose_source import KeyboardPoseSource

            return KeyboardPoseSource()
        if driver == "openxr":
            from .openxr_pose_source import OpenXRPoseSource

            return OpenXRPoseSource(vr_hz=self.config.vr_hz)
        raise ValueError(
            f"Unknown VRMocap driver '{driver}'. Expected 'scripted', 'keyboard', or 'openxr'."
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        ik = self._ik
        current_ee = {s: ik.get_ee_pose(s) for s in SIDES}
        targets = self._source.get_targets(current_ee)

        for side in SIDES:
            tgt = targets.get(side)
            if tgt is None or not tgt.active:
                continue  # hold: leave IK qpos (and thus joint output) unchanged
            take_home = getattr(self._source, "take_home_request", None)
            if callable(take_home) and take_home(side):
                # h: TELEPORT to the default pose -- the IK copy jumps, the
                # gripper opens, the target re-syncs, and the sim robot is told
                # to set its joints rather than swing there.
                from lerobot.robots.mujoco_bi_openarm import viewer_keys as _vk

                ik.set_joint_positions(side, self._default_q[side])
                ik.set_finger(side, FINGER_OPEN_M)
                self._grip_m[side] = float(FINGER_OPEN_M)
                resync = getattr(self._source, "resync_target", None)
                if callable(resync):
                    p, qt = ik.get_ee_pose(side)
                    resync(side, p, qt)
                reset_grip = getattr(self._source, "reset_grip", None)
                if callable(reset_grip):
                    reset_grip(side, FINGER_OPEN_M)
                _vk.request_teleport()
                continue
            take = getattr(self._source, "take_rotation_request", None)
            req = take(side) if callable(take) else None
            if req is not None:
                # rotation keys: sequential joint-space turn (L shoulder-first,
                # J wrist-first, each to its limit); IK skipped this tick so it
                # does not drag the hand back, then re-targeted on the result
                ik.chain_step(side, req[0], req[1], ref_q=self._default_q.get(side))
                resync = getattr(self._source, "resync_target", None)
                if callable(resync):
                    p, qt = ik.get_ee_pose(side)
                    resync(side, p, qt)
            else:
                ik.solve_ik(side, tgt.pos, tgt.quat, max_iter=self.config.max_iter)
            ik.set_finger(side, tgt.gripper_m)
            self._grip_m[side] = float(tgt.gripper_m)

        if self._debug_every and (self._tick % self._debug_every == 0):
            # VR_TELEOP_DEBUG=<n>: print the right arm's joints every n ticks so a
            # key press can be seen to move joints in the REAL teleop loop
            j = np.degrees(self._ik.joint_positions("right"))
            tip = np.asarray(self._ik.get_ee_pose("right")[0], float) * 100
            print(f"[teleop] tick {self._tick}  right J1..J7 = {np.round(j, 1).tolist()}  tip cm = {np.round(tip, 1).tolist()}", flush=True)
        self._tick += 1
        return self._joint_action()

    def _joint_action(self) -> RobotAction:
        """Current IK joint angles as the 16-key right-first wire action."""
        action: dict[str, float] = {}
        for side in SIDES:
            joints_rad = self._ik.joint_positions(side)
            for i, motor in enumerate(ARM_JOINT_NAMES):
                action[f"{side}_{motor}.pos"] = float(joints_rad[i]) * _RAD2DEG
            action[f"{side}_gripper.pos"] = gripper_m_to_deg(self._grip_m[side])
        return action

    def drain_recording_controls(self) -> list[str]:
        """Pending recording controls from the pose source (VR B/A/X buttons)."""
        drain = getattr(self._source, "drain_recording_controls", None)
        return drain() if callable(drain) else []

    def send_feedback(self, feedback: dict) -> None:
        """Forward robot observation images to the OpenXR headset view."""
        source = self._source
        if source is None or not hasattr(source, "update_camera_frames"):
            return
        frames = {
            key: value
            for key, value in feedback.items()
            if isinstance(value, np.ndarray) and getattr(value, "ndim", 0) == 3
        }
        if frames:
            source.update_camera_frames(_with_hud(frames))

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:  # noqa: BLE001
                logger.debug("pose source stop failed", exc_info=True)
        self._source = None
        self._ik = None
        self._data = None
        self._model = None
        self._connected = False
        logger.info("%s disconnected.", self)
