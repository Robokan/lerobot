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

# J3 (upper-arm roll) and J5 (forearm roll): the two joints a yaw gesture turns.
_ROLL_PAIR_MASK = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

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


def _vk_mark(side, pos, quat):
    try:
        from lerobot.robots.mujoco_bi_openarm.viewer_keys import set_target_marker
    except Exception:  # noqa: BLE001
        return
    set_target_marker(side, pos, quat)


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
        self._actual_q: dict = {}
        self._last_tgt: dict = {}
        self._rot_prev: dict = {}
        self._roll_side: dict = {}
        self._debug_every = int(os.environ.get("VR_TELEOP_DEBUG", "0") or 0)
        # artificial wrist-roll range (see config.wrist_limit_deg)
        lo_deg, hi_deg = self.config.wrist_limit_deg
        s_lo, s_hi = self.config.shoulder_roll_limit_deg
        # The model's own ranges. The artificial stops (wrist_limit_deg,
        # shoulder_roll_limit_deg) and the return-home pins are applied ONLY
        # while a rotation key is held (see get_action); translation and the
        # hold get the full ranges, otherwise a w/s/a/d move that takes the
        # shoulder roll off home left the wrist pinned and the shoulder frozen
        # -- a five-joint arm that locked up and chattered.
        self._limits_model = {s: (self._ik.limits_low[s].copy(), self._ik.limits_high[s].copy()) for s in SIDES}
        del lo_deg, hi_deg, s_lo, s_hi
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
        # commanded-pose markers for the viewer (m toggles them)
        for side, tgt in targets.items():
            if tgt is not None:
                _vk_mark(side, tgt.pos, tgt.quat)
                self._last_tgt[side] = np.asarray(tgt.pos, float).copy()

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

                ik.limits_low[side][:], ik.limits_high[side][:] = self._limits_model[side]
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
            # Return-home rule, by joint limits alone, applied EVERY tick. The
            # states the rotation keys walk through form one path, (shoulder
            # roll J3, wrist roll J5):  (0,+90) <-j- (0,0) -l-> (-90,0) -l->
            # (-90,-90). L: the shoulder to its stop first, then the wrist; J:
            # the wrist back to rest first, then the shoulder home, then the
            # wrist the other way. Off the path the same rules walk back onto
            # it. The pins stay on between gestures too: the two roll joints
            # share an axis when the arm is straight, so with them free the
            # shoulder spring traded its roll into the wrist after every turn
            # (-89/-61 drifted to -59/-90 in a second of settle), and with the
            # roll springs off instead, plain translation wandered the pair to
            # -85/+85. The freeze once blamed on the pins was the solver's
            # no-twist guard (see IKSolver.solve_ik best-effort step).
            model_lo, model_hi = self._limits_model[side]
            rot_active = getattr(self._source, "rotation_active", None)
            if True:
                qj = ik.joint_positions(side)
                q3, q5 = float(qj[2]), float(qj[4])
                d3, d5 = float(self._default_q[side][2]), float(self._default_q[side][4])
                tol = math.radians(2.0)
                eps = 1e-6
                hi_cfg = min(float(model_hi[4]), math.radians(self.config.wrist_limit_deg[1]))
                wlo_cfg = max(float(model_lo[4]), math.radians(self.config.wrist_limit_deg[0]))
                lo_cfg = max(float(model_lo[2]), math.radians(self.config.shoulder_roll_limit_deg[0]))
                shi_cfg = min(float(model_hi[2]), math.radians(self.config.shoulder_roll_limit_deg[1]))
                j3_home = abs(q3 - d3) < tol
                # The shoulder roll only shares the tool's axis while the arm
                # is straight. Bend the elbow and it cannot serve a yaw at all,
                # so "wrist waits for the shoulder to reach its stop" waits
                # forever -- at a 90 deg elbow that left L doing nothing.
                j3_useful = ik.tool_axis_alignment(side, 2) > 0.35
                j3_at_stop = abs(q3 - lo_cfg) < tol or not j3_useful
                # wrist, J side: free once the shoulder is home, else only unwind
                ik.limits_high[side][4] = max(hi_cfg, q5) + eps if j3_home else min(hi_cfg, max(q5, d5) + eps)
                # wrist, L side: the stop holds until the shoulder is at ITS
                # stop, then the model's range; below the stop only unwind
                ik.limits_low[side][4] = (min(float(model_lo[4]), q5) - eps if j3_at_stop
                                          else max(float(model_lo[4]), min(wlo_cfg, q5) - eps))
                # shoulder, L side: held while the wrist is above rest
                ik.limits_low[side][2] = q3 - eps if q5 > d5 + tol else min(lo_cfg, q3) - eps
                # shoulder, J side: held while the wrist is below rest; home is its stop
                ik.limits_high[side][2] = q3 + eps if q5 < d5 - tol else max(shi_cfg, q3) + eps

                # Detent at the default pose. ONE held gesture may not pass
                # THROUGH it: out and back within a press lands exactly where
                # the press started, and carrying on out the other side takes
                # a new press. Without this, returning ran straight through
                # home and out again -- j to the stop then l for the same time
                # came back to the wrist's 0 but left the shoulder at -18.
                # u = (J3-d3) + (J5-d5) is monotone along the whole path: 0 at
                # default, -180 at the L end, +90 at the J end, so "which side
                # of default" is just its sign.
                # A yaw gesture is the roll pair's turn and nobody else's.
                # Held past their limits the solver otherwise keeps rolling the
                # tool with J2/J4/J6/J7 -- measured: the elbow walked 49 -> 15
                # deg and J6 -18 -> 35 while the pair sat on its stop, and none
                # of it came back on the return.
                rot_axis = getattr(self._source, "rotation_axis", None)
                ik.ori_joint_mask[side] = (_ROLL_PAIR_MASK
                                           if callable(rot_axis) and rot_axis(side) == "yaw" else None)
                rot_on = bool(rot_active(side)) if callable(rot_active) else False
                if rot_on and not self._rot_prev.get(side):
                    u0 = (q3 - d3) + (q5 - d5)
                    self._roll_side[side] = 0.0 if abs(u0) < tol else math.copysign(1.0, u0)
                elif not rot_on:
                    self._roll_side[side] = 0.0
                self._rot_prev[side] = rot_on
                sgn = self._roll_side.get(side, 0.0)
                if sgn > 0:      # started on the J side: may not fall below default
                    ik.limits_low[side][2] = max(ik.limits_low[side][2], min(d3, q3) - eps)
                    ik.limits_low[side][4] = max(ik.limits_low[side][4], min(d5, q5) - eps)
                elif sgn < 0:    # started on the L side: may not rise above default
                    ik.limits_high[side][2] = min(ik.limits_high[side][2], max(d3, q3) + eps)
                    ik.limits_high[side][4] = min(ik.limits_high[side][4], max(d5, q5) + eps)
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
                # clip AFTER the solve too, so against an obstacle the command
                # sits steadily at the leash instead of stepping 3 deg past it
                # and being pulled back every tick (a sawtooth the PD chased)
                self._clip_to_actual(side)
                self._leash_target(side, tgt)
            ik.set_finger(side, tgt.gripper_m)
            self._grip_m[side] = float(tgt.gripper_m)

        if self._debug_every and (self._tick % self._debug_every == 0):
            # VR_TELEOP_DEBUG=<n>: print the right arm's joints every n ticks so a
            # key press can be seen to move joints in the REAL teleop loop
            j = np.degrees(self._ik.joint_positions("right"))
            tip = np.asarray(self._ik.get_ee_pose("right")[0], float) * 100
            tg = self._last_tgt.get("right")
            tg = "" if tg is None else f"  tgt cm = {np.round(np.asarray(tg, float) * 100, 1).tolist()}"
            print(f"[teleop] tick {self._tick}  right J1..J7 = {np.round(j, 1).tolist()}  tip cm = {np.round(tip, 1).tolist()}{tg}", flush=True)
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

    def _clip_to_actual(self, side) -> None:
        """Keep the commanded joints within actual_leash_deg of the last ACTUAL
        joints reported by the robot (no-op until the first feedback)."""
        a = self._actual_q.get(side)
        leash = math.radians(self.config.actual_leash_deg)
        if a is None or leash <= 0:
            return
        q = self._ik.joint_positions(side)
        qc = np.clip(q, a - leash, a + leash)
        if np.any(np.abs(qc - q) > 1e-9):
            self._ik.set_joint_positions(side, qc)

    def _leash_target(self, side, tgt) -> None:
        """Pull the pose target back to within target_leash_m / _deg of the
        commanded tip (see config). Only the source's stored target moves; a
        rotation gesture's tip anchor is untouched."""
        resync = getattr(self._source, "resync_target", None)
        if not callable(resync):
            return
        mujoco = self._ik._mujoco
        p_tip, q_tip = self._ik.get_ee_pose(side)
        p_tip = np.asarray(p_tip, float); q_tip = np.asarray(q_tip, float)
        new_p, new_q = np.asarray(tgt.pos, float).copy(), np.asarray(tgt.quat, float).copy()
        changed = False
        err = new_p - p_tip; d = float(np.linalg.norm(err))
        if d > self.config.target_leash_m:
            new_p = p_tip + err * (self.config.target_leash_m / d); changed = True
        v = np.zeros(3)
        mujoco.mju_subQuat(v, new_q, q_tip)          # rotation from the tip attitude to the target
        a = float(np.linalg.norm(v)); lim = math.radians(self.config.target_leash_deg)
        if a > lim:
            new_q = q_tip.copy()
            mujoco.mju_quatIntegrate(new_q, v * (lim / a), 1.0); changed = True
        if changed:
            resync(side, new_p, new_q)

    def send_feedback(self, feedback: dict) -> None:
        """Forward robot observation images to the OpenXR headset view."""
        if self._debug_every and (self._tick % self._debug_every == 0):
            # the ACTUAL sim joints next to the commanded ones, to see ringing
            act = [feedback.get(f"right_{m}.pos") for m in ARM_JOINT_NAMES]
            if all(a is not None for a in act):
                print(f"[teleop] tick {self._tick}  right ACTUAL J1..J7 = {np.round(act, 1).tolist()}", flush=True)
        # Leash the commanded arm to the actual one (see config.actual_leash_deg)
        leash = math.radians(self.config.actual_leash_deg)
        if leash > 0:
            for side in SIDES:
                act = [feedback.get(f"{side}_{m}.pos") for m in ARM_JOINT_NAMES]
                if all(a is not None for a in act):
                    first = side not in self._actual_q
                    self._actual_q[side] = np.radians(np.asarray(act, float))
                    if first:
                        # First contact with the robot: adopt ITS launch pose as
                        # the solver state, the h pose and the pose target. The
                        # solver's own model boots hanging straight; with
                        # --robot.start_elbow_bend_deg the robot does not, and
                        # the stale hanging-attitude target made the first key
                        # press swing the arm 21 cm up to point the hand down.
                        self._ik.limits_low[side][:], self._ik.limits_high[side][:] = self._limits_model[side]
                        self._ik.set_joint_positions(side, self._actual_q[side])
                        self._default_q[side] = self._ik.joint_positions(side).copy()
                        resync = getattr(self._source, "resync_target", None)
                        if callable(resync):
                            p0, q0 = self._ik.get_ee_pose(side)
                            resync(side, p0, q0)
                    else:
                        self._clip_to_actual(side)
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
