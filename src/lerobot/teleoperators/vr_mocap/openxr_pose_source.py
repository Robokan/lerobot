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

"""OpenXR pose source (Phase 2) for the VR motion-capture teleoperator.

Ported from SparkJAX ``scripts/test_vr_ik.py`` (``run_vr``): opens an OpenXR
session (via a hidden GLFW GL context), binds the controller grip pose / trigger
/ Y/X/A/B buttons across HTC Vive / Focus3 / Oculus Touch / simple interaction
profiles, performs head-based body-frame yaw calibration, and converts
controller poses into the robot frame. A background thread runs the XR frame
loop and publishes the latest per-hand robot-frame pose + trigger + tracking
toggle.

When MuJoCo (or real) camera frames are fed in via :meth:`update_camera_frames`,
the headset can show them as a composition quad with the same controls as
SparkJAX:

  X — toggle tracking
  Y — toggle passthrough ↔ active MuJoCo/robot camera
  thumbstick — push left = left wrist cam, push right = right wrist cam,
               centered = chest (ego) cam
  B — start recording          A — stop recording & save episode
  right grip squeeze — cancel (re-record) the current episode
  triggers — grippers

The controller-relative *reference capture* (delta-teleop) is performed in
:meth:`get_targets` (teleop thread) against the robot's current TCP pose, so the
IK model is never touched from the XR thread.

All VR deps (``pyopenxr``/``glfw``/``PyOpenGL``) are imported lazily in
:meth:`start`; this module stays import-safe without them. Test on the headset
machine — it cannot be exercised headless.
"""

from __future__ import annotations

import ctypes
import logging
import math
import threading
import time
from typing import Any

import numpy as np

from .ik import FINGER_OPEN_M, quat_inv, quat_mul, xr_pos_to_robot, xr_quat_to_robot
from .pose_source import SIDES, HandTarget, PoseSource

logger = logging.getLogger(__name__)

# Dataset / robot feature names → SparkJAX-style short names used by X/A toggles.
_CAM_ALIASES = {
    "ego": "ego",
    "left_wrist": "left",
    "right_wrist": "right",
    "left": "left",
    "right": "right",
}

_QUAD_VERT = """
#version 450 core
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID & 1) * 2.0, (gl_VertexID & 2) * 1.0);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
"""

_QUAD_FRAG = """
#version 450 core
in vec2 uv;
out vec4 frag;
uniform sampler2D tex;
void main() {
    frag = texture(tex, vec2(uv.x, 1.0 - uv.y));
}
"""


def _create_quad_program(GL):
    vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vs, _QUAD_VERT)
    GL.glCompileShader(vs)
    if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
        raise RuntimeError(f"Vertex shader error: {GL.glGetShaderInfoLog(vs)}")
    fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fs, _QUAD_FRAG)
    GL.glCompileShader(fs)
    if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
        raise RuntimeError(f"Fragment shader error: {GL.glGetShaderInfoLog(fs)}")
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        raise RuntimeError(f"Program link error: {GL.glGetProgramInfoLog(prog)}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog


def _rgb_to_rgba(frame: np.ndarray) -> np.ndarray:
    """Convert HxWx3 RGB uint8 to contiguous HxWx4 RGBA uint8."""
    if frame.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {frame.shape}")
    if frame.shape[2] == 4:
        return np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.shape[2] != 3:
        raise ValueError(f"Expected 3 or 4 channels, got {frame.shape[2]}")
    rgba = np.empty(frame.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = frame
    rgba[..., 3] = 255
    return rgba


class OpenXRPoseSource(PoseSource):
    """Real VR controller pose source using pyopenxr."""

    def __init__(self, vr_hz: int = 50):
        self.vr_hz = vr_hz
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

        # Latest controller state (robot frame), published by the XR thread.
        self._ctrl_pos = {s: np.zeros(3) for s in SIDES}
        self._ctrl_quat = {s: np.array([1.0, 0.0, 0.0, 0.0]) for s in SIDES}  # wxyz
        self._ctrl_valid = {s: False for s in SIDES}
        self._trigger = {s: 0.0 for s in SIDES}
        self._tracking = False
        self._activation_id = 0  # increments on each rising edge of tracking

        # Per-hand delta-teleop references (captured in get_targets).
        self._seen_activation = -1
        self._ref_ctrl_pos = {s: None for s in SIDES}
        self._ref_ctrl_quat = {s: None for s in SIDES}
        self._ref_ee_pos = {s: None for s in SIDES}
        self._ref_ee_quat = {s: None for s in SIDES}

        # Camera feed for the headset (fed from teleop.send_feedback / robot obs).
        # Keys are short names: ego / left / right. Values are RGBA uint8.
        self._cam_frames: dict[str, np.ndarray] = {}
        self._cam_seq = 0
        self._active_camera = "ego"
        self._force_passthrough = True  # SparkJAX default: passthrough until Y

        # Recording control requests (B=start, A=stop&save, X=discard) drained
        # by the record loop via drain_recording_controls().
        self._record_controls: list[str] = []
        # Left-grip squeeze edge state (tracking toggle) with hysteresis.
        self._squeeze_was_high = False

    def drain_recording_controls(self) -> list[str]:
        """Return and clear pending controller recording requests (B/A/X)."""
        with self._lock:
            controls = list(self._record_controls)
            self._record_controls.clear()
        return controls

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._xr_loop, daemon=True)
        self._thread.start()
        logger.info(
            "OpenXRPoseSource started; Y=tracking, B=passthrough/camera, "
            "X=ego/right, A=ego/left."
        )

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def reset(self, initial_ee):
        # References are captured lazily on the next tracking activation.
        with self._lock:
            for s in SIDES:
                self._ref_ctrl_pos[s] = None
                self._ref_ctrl_quat[s] = None
                self._ref_ee_pos[s] = None
                self._ref_ee_quat[s] = None
            self._seen_activation = self._activation_id

    def update_camera_frames(self, frames: dict[str, Any]) -> None:
        """Accept RGB(A) observation images keyed by dataset camera name.

        Called from the teleop thread via :meth:`VRMocap.send_feedback`. Only
        ``ego`` / ``left_wrist`` / ``right_wrist`` (and short aliases) are kept.
        """
        if not frames:
            return
        converted: dict[str, np.ndarray] = {}
        for key, value in frames.items():
            short = _CAM_ALIASES.get(key)
            if short is None or not isinstance(value, np.ndarray):
                continue
            try:
                converted[short] = _rgb_to_rgba(value)
            except ValueError:
                continue
        if not converted:
            return
        with self._lock:
            self._cam_frames.update(converted)
            self._cam_seq += 1

    # ------------------------------------------------------------------ targets
    def get_targets(self, current_ee):
        with self._lock:
            tracking = self._tracking
            activation_id = self._activation_id
            ctrl_pos = {s: self._ctrl_pos[s].copy() for s in SIDES}
            ctrl_quat = {s: self._ctrl_quat[s].copy() for s in SIDES}
            valid = dict(self._ctrl_valid)
            trigger = dict(self._trigger)

        # Rising edge of tracking -> drop references so they are recaptured.
        if activation_id != self._seen_activation:
            self._seen_activation = activation_id
            for s in SIDES:
                self._ref_ctrl_pos[s] = None

        targets: dict[str, HandTarget] = {}
        for side in SIDES:
            ee_pos, ee_quat = current_ee[side]
            if not tracking or not valid[side]:
                # Hold: report current pose, inactive.
                targets[side] = HandTarget(
                    pos=np.asarray(ee_pos).copy(),
                    quat=np.asarray(ee_quat).copy(),
                    gripper_m=(1.0 - trigger[side]) * FINGER_OPEN_M,
                    active=False,
                )
                continue

            if self._ref_ctrl_pos[side] is None:
                self._ref_ctrl_pos[side] = ctrl_pos[side].copy()
                self._ref_ctrl_quat[side] = ctrl_quat[side].copy()
                self._ref_ee_pos[side] = np.asarray(ee_pos).copy()
                self._ref_ee_quat[side] = np.asarray(ee_quat).copy()

            delta_pos = ctrl_pos[side] - self._ref_ctrl_pos[side]
            target_pos = self._ref_ee_pos[side] + delta_pos
            delta_quat = quat_mul(ctrl_quat[side], quat_inv(self._ref_ctrl_quat[side]))
            target_quat = quat_mul(delta_quat, self._ref_ee_quat[side])
            target_quat = target_quat / np.linalg.norm(target_quat)
            targets[side] = HandTarget(
                pos=target_pos,
                quat=target_quat,
                gripper_m=(1.0 - trigger[side]) * FINGER_OPEN_M,
                active=True,
            )
        return targets

    # ------------------------------------------------------------------ XR thread
    def _xr_loop(self):  # noqa: C901 - faithful port of the SparkJAX run_vr loop
        try:
            import glfw
            import xr
            from OpenGL import GL, GLX
        except Exception:  # noqa: BLE001
            logger.exception(
                "OpenXRPoseSource requires pyopenxr, glfw and PyOpenGL. Install them on the "
                "headset machine (e.g. `uv pip install pyopenxr glfw PyOpenGL`)."
            )
            self._running = False
            return

        if not glfw.init():
            logger.error("GLFW init failed")
            self._running = False
            return
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        window = glfw.create_window(4, 4, "lerobot-xr", None, None)
        if window is None:
            logger.error("GLFW window creation failed")
            glfw.terminate()
            self._running = False
            return
        glfw.make_context_current(window)
        glfw.swap_interval(0)

        extensions = xr.enumerate_instance_extension_properties()
        required = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        for ext in required:
            if ext not in extensions:
                logger.error("Required XR extension %s not available", ext)
                self._running = False
                return
        enabled_exts = list(required)
        for ext in ["XR_HTC_vive_focus3_controller_interaction"]:
            if ext in extensions:
                enabled_exts.append(ext)

        instance = xr.create_instance(xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(
                application_name="lerobot VR mocap",
                application_version=xr.Version(0, 1, 0),
                engine_name="pyopenxr",
                engine_version=xr.Version(0, 1, 0),
                api_version=xr.Version(1, 0, 0)),
            enabled_extension_names=enabled_exts))
        system_id = xr.get_system(instance, xr.SystemGetInfo(
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        pfn = ctypes.cast(
            xr.get_instance_proc_addr(instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR)
        gl_reqs = xr.GraphicsRequirementsOpenGLKHR()
        xr.check_result(xr.Result(pfn(instance, system_id, ctypes.byref(gl_reqs))))

        graphics_binding = xr.GraphicsBindingOpenGLXlibKHR(
            x_display=GLX.glXGetCurrentDisplay(),
            glx_drawable=GLX.glXGetCurrentDrawable(),
            glx_context=GLX.glXGetCurrentContext())
        session = xr.create_session(instance, xr.SessionCreateInfo(
            next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p),
            system_id=system_id))
        ref_space = xr.create_reference_space(session, xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.STAGE,
            pose_in_reference_space=xr.Posef()))

        # -- Actions ---------------------------------------------------------
        action_set = xr.create_action_set(instance, xr.ActionSetCreateInfo(
            action_set_name="vr_mocap", localized_action_set_name="VR Mocap", priority=0))
        hand_paths = [xr.string_to_path(instance, "/user/hand/left"),
                      xr.string_to_path(instance, "/user/hand/right")]
        grip_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="grip_pose", action_type=xr.ActionType.POSE_INPUT,
            localized_action_name="Grip Pose", subaction_paths=hand_paths))
        trigger_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="trigger", action_type=xr.ActionType.FLOAT_INPUT,
            localized_action_name="Trigger", subaction_paths=hand_paths))
        left_path = [xr.string_to_path(instance, "/user/hand/left")]
        right_path = [xr.string_to_path(instance, "/user/hand/right")]
        activate_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="activate", action_type=xr.ActionType.BOOLEAN_INPUT,
            localized_action_name="Activate Tracking", subaction_paths=left_path))
        x_button_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="x_button", action_type=xr.ActionType.BOOLEAN_INPUT,
            localized_action_name="X Button", subaction_paths=left_path))
        a_button_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="a_button", action_type=xr.ActionType.BOOLEAN_INPUT,
            localized_action_name="A Button", subaction_paths=right_path))
        b_button_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="b_button", action_type=xr.ActionType.BOOLEAN_INPUT,
            localized_action_name="B Button", subaction_paths=right_path))
        squeeze_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="squeeze", action_type=xr.ActionType.FLOAT_INPUT,
            localized_action_name="Grip Squeeze", subaction_paths=hand_paths))
        stick_x_action = xr.create_action(action_set, xr.ActionCreateInfo(
            action_name="stick_x", action_type=xr.ActionType.FLOAT_INPUT,
            localized_action_name="Thumbstick X", subaction_paths=hand_paths))

        for profile_path, extras in [
            ("/interaction_profiles/htc/vive_controller",
             {"trigger": "/input/trigger/value",
              "activate": "/user/hand/left/input/menu/click",
              "x_btn": "/user/hand/left/input/trackpad/click",
              "a_btn": "/user/hand/right/input/trackpad/click",
              "b_btn": None,
              "squeeze": "/input/squeeze/click",
              "stick_x": "/input/trackpad/x"}),
            ("/interaction_profiles/htc/vive_focus3_controller",
             {"trigger": "/input/trigger/value",
              "activate": "/user/hand/left/input/y/click",
              "x_btn": "/user/hand/left/input/x/click",
              "a_btn": "/user/hand/right/input/a/click",
              "b_btn": "/user/hand/right/input/b/click",
              "squeeze": "/input/squeeze/value",
              "stick_x": "/input/thumbstick/x"}),
            ("/interaction_profiles/oculus/touch_controller",
             {"trigger": "/input/trigger/value",
              "activate": "/user/hand/left/input/y/click",
              "x_btn": "/user/hand/left/input/x/click",
              "a_btn": "/user/hand/right/input/a/click",
              "b_btn": "/user/hand/right/input/b/click",
              "squeeze": "/input/squeeze/value",
              "stick_x": "/input/thumbstick/x"}),
            ("/interaction_profiles/khr/simple_controller",
             {"trigger": "/input/select/click",
              "activate": "/user/hand/left/input/menu/click",
              "x_btn": None, "a_btn": None, "b_btn": None,
              "squeeze": None, "stick_x": None}),
        ]:
            try:
                bindings = []
                for hand in ("/user/hand/left", "/user/hand/right"):
                    bindings.append(xr.ActionSuggestedBinding(
                        action=grip_action,
                        binding=xr.string_to_path(instance, f"{hand}/input/grip/pose")))
                    bindings.append(xr.ActionSuggestedBinding(
                        action=trigger_action,
                        binding=xr.string_to_path(instance, f"{hand}{extras['trigger']}")))
                    if extras.get("squeeze"):
                        bindings.append(xr.ActionSuggestedBinding(
                            action=squeeze_action,
                            binding=xr.string_to_path(instance, f"{hand}{extras['squeeze']}")))
                    if extras.get("stick_x"):
                        bindings.append(xr.ActionSuggestedBinding(
                            action=stick_x_action,
                            binding=xr.string_to_path(instance, f"{hand}{extras['stick_x']}")))
                bindings.append(xr.ActionSuggestedBinding(
                    action=activate_action,
                    binding=xr.string_to_path(instance, extras["activate"])))
                if extras.get("x_btn"):
                    bindings.append(xr.ActionSuggestedBinding(
                        action=x_button_action,
                        binding=xr.string_to_path(instance, extras["x_btn"])))
                if extras.get("a_btn"):
                    bindings.append(xr.ActionSuggestedBinding(
                        action=a_button_action,
                        binding=xr.string_to_path(instance, extras["a_btn"])))
                if extras.get("b_btn"):
                    bindings.append(xr.ActionSuggestedBinding(
                        action=b_button_action,
                        binding=xr.string_to_path(instance, extras["b_btn"])))
                xr.suggest_interaction_profile_bindings(instance,
                    xr.InteractionProfileSuggestedBinding(
                        interaction_profile=xr.string_to_path(instance, profile_path),
                        suggested_bindings=bindings))
                logger.info("Bound interaction profile: %s", profile_path)
            except xr.exception.XrException as e:
                logger.debug("Profile %s not bound: %s", profile_path, e)

        grip_spaces = [
            xr.create_action_space(session, xr.ActionSpaceCreateInfo(
                action=grip_action, subaction_path=hp, pose_in_action_space=xr.Posef()))
            for hp in hand_paths
        ]
        xr.attach_session_action_sets(session, xr.SessionActionSetsAttachInfo(action_sets=[action_set]))

        view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        formats = xr.enumerate_swapchain_formats(session)
        preferred = [GL.GL_SRGB8_ALPHA8, GL.GL_RGBA8]
        color_format = next((pf for pf in preferred if pf in formats), formats[0])

        # Passthrough fallback swapchain (1x1 transparent quad).
        pt_sc = xr.create_swapchain(session, xr.SwapchainCreateInfo(
            usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT,
            format=color_format, sample_count=1, width=1, height=1,
            face_count=1, array_size=1, mip_count=1))
        pt_images = xr.enumerate_swapchain_images(pt_sc, xr.SwapchainImageOpenGLKHR)
        pt_fbo = GL.glGenFramebuffers(1)

        # Camera quad resources — allocated lazily on the first frame we receive.
        cam_sc = None
        cam_images = None
        cam_fbo = None
        cam_gl_tex = None
        cam_w = cam_h = 0
        quad_prog = None
        quad_vao = None
        tex_loc = -1
        last_uploaded_seq = -1

        blend_modes = xr.enumerate_environment_blend_modes(instance, system_id, view_config_type)
        pt_blend_mode = (xr.EnvironmentBlendMode.ALPHA_BLEND
                         if xr.EnvironmentBlendMode.ALPHA_BLEND in blend_modes
                         else xr.EnvironmentBlendMode.OPAQUE)

        def _ensure_cam_resources(w: int, h: int) -> bool:
            nonlocal cam_sc, cam_images, cam_fbo, cam_gl_tex, cam_w, cam_h
            nonlocal quad_prog, quad_vao, tex_loc
            if cam_sc is not None and cam_w == w and cam_h == h:
                return True
            if cam_sc is not None:
                try:
                    xr.destroy_swapchain(cam_sc)
                except Exception:  # noqa: BLE001
                    pass
                cam_sc = None
            cam_sc = xr.create_swapchain(session, xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT,
                format=color_format, sample_count=1, width=w, height=h,
                face_count=1, array_size=1, mip_count=1))
            cam_images = xr.enumerate_swapchain_images(cam_sc, xr.SwapchainImageOpenGLKHR)
            if cam_fbo is None:
                cam_fbo = GL.glGenFramebuffers(1)
            if cam_gl_tex is None:
                cam_gl_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, cam_gl_tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
            )
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            if quad_prog is None:
                quad_prog = _create_quad_program(GL)
                tex_loc = GL.glGetUniformLocation(quad_prog, "tex")
                quad_vao = GL.glGenVertexArrays(1)
            cam_w, cam_h = w, h
            logger.info("Headset camera quad ready at %dx%d", w, h)
            return True

        def _submit_passthrough(frame_state):
            img_idx = xr.acquire_swapchain_image(pt_sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(pt_sc, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
            sc_tex = pt_images[img_idx].image
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, pt_fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, sc_tex, 0)
            GL.glViewport(0, 0, 1, 1)
            GL.glClearColor(0.0, 0.0, 0.0, 0.01)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            xr.release_swapchain_image(pt_sc, xr.SwapchainImageReleaseInfo())
            quad_layer = xr.CompositionLayerQuad(
                layer_flags=xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT,
                space=ref_space, eye_visibility=xr.EyeVisibility.BOTH,
                sub_image=xr.SwapchainSubImage(
                    swapchain=pt_sc,
                    image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(1, 1)),
                    image_array_index=0),
                pose=xr.Posef(orientation=xr.Quaternionf(0, 0, 0, 1),
                              position=xr.Vector3f(0, 1.0, -1.5)),
                size=xr.Extent2Df(5.0, 5.0))
            xr.end_frame(session, xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=pt_blend_mode,
                layers=[ctypes.byref(quad_layer)]))

        def _submit_camera_quad(frame_state):
            img_idx = xr.acquire_swapchain_image(cam_sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(cam_sc, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
            sc_tex = cam_images[img_idx].image
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cam_fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, sc_tex, 0
            )
            GL.glViewport(0, 0, cam_w, cam_h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glUseProgram(quad_prog)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, cam_gl_tex)
            GL.glUniform1i(tex_loc, 0)
            GL.glBindVertexArray(quad_vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            xr.release_swapchain_image(cam_sc, xr.SwapchainImageReleaseInfo())
            aspect = cam_w / max(cam_h, 1)
            panel_w = 2.5
            panel_h = panel_w / aspect
            quad_layer = xr.CompositionLayerQuad(
                layer_flags=xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT,
                space=ref_space,
                eye_visibility=xr.EyeVisibility.BOTH,
                sub_image=xr.SwapchainSubImage(
                    swapchain=cam_sc,
                    image_rect=xr.Rect2Di(
                        offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(cam_w, cam_h)
                    ),
                    image_array_index=0,
                ),
                pose=xr.Posef(
                    orientation=xr.Quaternionf(0, 0, 0, 1),
                    position=xr.Vector3f(0, 1.2, -2.0),
                ),
                size=xr.Extent2Df(panel_w, panel_h),
            )
            xr.end_frame(session, xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=pt_blend_mode,
                layers=[ctypes.byref(quad_layer)]))

        def _edge(action, path) -> bool:
            try:
                st = xr.get_action_state_boolean(
                    session, xr.ActionStateGetInfo(action=action, subaction_path=path)
                )
                return bool(st.is_active and st.current_state and st.changed_since_last_sync)
            except xr.exception.XrException:
                return False

        session_state = xr.SessionState.UNKNOWN
        session_running = False
        body_rotation = None
        body_quat_s2b = None

        try:
            while self._running:
                glfw.poll_events()
                while True:
                    try:
                        event_buf = xr.poll_event(instance)
                        if xr.StructureType(event_buf.type) == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                            ev = ctypes.cast(ctypes.byref(event_buf),
                                             ctypes.POINTER(xr.EventDataSessionStateChanged)).contents
                            session_state = xr.SessionState(ev.state)
                            if session_state == xr.SessionState.READY:
                                xr.begin_session(session, xr.SessionBeginInfo(view_config_type))
                                session_running = True
                            elif session_state in (xr.SessionState.STOPPING, xr.SessionState.LOSS_PENDING,
                                                    xr.SessionState.EXITING):
                                self._running = False
                                break
                    except xr.EventUnavailable:
                        break
                    except xr.exception.InstanceLostError:
                        self._running = False
                        break

                if not session_running or not self._running:
                    time.sleep(0.01)
                    continue

                frame_state = xr.wait_frame(session)
                xr.begin_frame(session)

                show_camera = False
                if session_state == xr.SessionState.FOCUSED:
                    try:
                        active_sets = (xr.ActiveActionSet * 1)(
                            xr.ActiveActionSet(action_set=action_set, subaction_path=xr.NULL_PATH))
                        xr.sync_actions(session, xr.ActionsSyncInfo(active_action_sets=active_sets))
                        pt = frame_state.predicted_display_time

                        # Head-based body-frame calibration (once per activation).
                        if self._tracking and body_rotation is None:
                            body_rotation, body_quat_s2b = self._calibrate_body(
                                xr, session, view_config_type, pt, ref_space)

                        for hand_idx, side in enumerate(["left", "right"]):
                            grip_loc = xr.locate_space(grip_spaces[hand_idx], ref_space, pt)
                            flags = grip_loc.location_flags
                            valid = bool(flags & xr.SpaceLocationFlags.POSITION_VALID_BIT
                                         and flags & xr.SpaceLocationFlags.ORIENTATION_VALID_BIT)
                            if valid:
                                p = grip_loc.pose.position
                                q = grip_loc.pose.orientation
                                bp = np.array([p.x, p.y, p.z])
                                bq = np.array([q.x, q.y, q.z, q.w])
                                if body_rotation is not None:
                                    bp = body_rotation @ bp
                                if body_quat_s2b is not None:
                                    qx, qy, qz, qw = bq
                                    q_body = quat_mul(body_quat_s2b, np.array([qw, qx, qy, qz]))
                                    bq = np.array([q_body[1], q_body[2], q_body[3], q_body[0]])
                                robot_pos = xr_pos_to_robot(bp)
                                robot_quat = xr_quat_to_robot(bq)
                                with self._lock:
                                    self._ctrl_pos[side][:] = robot_pos
                                    self._ctrl_quat[side][:] = robot_quat
                                    self._ctrl_valid[side] = True
                            else:
                                with self._lock:
                                    self._ctrl_valid[side] = False
                            try:
                                tr = xr.get_action_state_float(session, xr.ActionStateGetInfo(
                                    action=trigger_action, subaction_path=hand_paths[hand_idx]))
                                with self._lock:
                                    self._trigger[side] = tr.current_state if tr.is_active else 0.0
                            except xr.exception.XrException:
                                pass

                        # X toggles tracking.
                        if _edge(x_button_action, left_path[0]):
                            with self._lock:
                                self._tracking = not self._tracking
                                if self._tracking:
                                    self._activation_id += 1
                                    body_rotation = None
                                    body_quat_s2b = None
                                    tracking_now = True
                                else:
                                    tracking_now = False
                            logger.info(
                                "Tracking %s (X button)",
                                "ACTIVATED" if tracking_now else "PAUSED",
                            )

                        # Right grip squeeze cancels the current take
                        # (re-record). 0.7/0.3 hysteresis so an analog grip
                        # can't chatter the edge.
                        squeeze_cancel = False
                        try:
                            sq = xr.get_action_state_float(session, xr.ActionStateGetInfo(
                                action=squeeze_action, subaction_path=hand_paths[1]))
                            sq_val = sq.current_state if sq.is_active else 0.0
                            if not self._squeeze_was_high and sq_val > 0.7:
                                self._squeeze_was_high = True
                                squeeze_cancel = True
                            elif self._squeeze_was_high and sq_val < 0.3:
                                self._squeeze_was_high = False
                        except xr.exception.XrException:
                            pass

                        # B starts recording, A stops & saves, right squeeze
                        # discards the current take. Drained by the record loop.
                        control = None
                        if _edge(b_button_action, right_path[0]):
                            control = "y"
                        elif _edge(a_button_action, right_path[0]):
                            control = "t"
                        elif squeeze_cancel:
                            control = "left"
                        if control is not None:
                            with self._lock:
                                self._record_controls.append(control)

                        # Thumbstick x picks the headset camera: push left =
                        # left wrist, push right = right wrist, centered = chest
                        # (ego). Either stick works; the larger deflection wins.
                        # 0.6/0.3 hysteresis keeps the boundary from flickering.
                        stick = 0.0
                        for hp in hand_paths:
                            try:
                                st_x = xr.get_action_state_float(session, xr.ActionStateGetInfo(
                                    action=stick_x_action, subaction_path=hp))
                                if st_x.is_active and abs(st_x.current_state) > abs(stick):
                                    stick = st_x.current_state
                            except xr.exception.XrException:
                                pass
                        new_cam = None
                        if stick < -0.6:
                            new_cam = "left"
                        elif stick > 0.6:
                            new_cam = "right"
                        elif abs(stick) < 0.3:
                            new_cam = "ego"
                        if new_cam is not None and new_cam != self._active_camera:
                            with self._lock:
                                available = set(self._cam_frames)
                            if new_cam in available:
                                self._active_camera = new_cam
                                logger.info("Camera -> %s", self._active_camera.upper())
                            else:
                                logger.info(
                                    "Camera %s not available yet (have %s)",
                                    new_cam,
                                    sorted(available) or "none",
                                )

                        # Y toggles passthrough ↔ camera feed.
                        if _edge(activate_action, left_path[0]):
                            self._force_passthrough = not self._force_passthrough
                            if self._force_passthrough:
                                logger.info("View -> PASSTHROUGH (Y button)")
                            else:
                                logger.info(
                                    "View -> CAMERA [%s] (Y button)",
                                    self._active_camera.upper(),
                                )

                        # Upload latest active camera frame to GL texture.
                        with self._lock:
                            seq = self._cam_seq
                            rgba = self._cam_frames.get(self._active_camera)
                            if rgba is None and self._cam_frames:
                                # Fall back to any available frame.
                                fallback_name, rgba = next(iter(self._cam_frames.items()))
                                self._active_camera = fallback_name
                            force_pt = self._force_passthrough
                        if rgba is not None and not force_pt:
                            h, w = rgba.shape[:2]
                            if _ensure_cam_resources(w, h) and seq != last_uploaded_seq:
                                GL.glBindTexture(GL.GL_TEXTURE_2D, cam_gl_tex)
                                GL.glTexSubImage2D(
                                    GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba,
                                )
                                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                                last_uploaded_seq = seq
                            show_camera = cam_sc is not None
                        else:
                            show_camera = False
                    except xr.exception.SessionNotFocused:
                        show_camera = False

                if show_camera:
                    _submit_camera_quad(frame_state)
                else:
                    _submit_passthrough(frame_state)
        finally:
            try:
                if cam_sc is not None:
                    xr.destroy_swapchain(cam_sc)
                if cam_fbo is not None:
                    GL.glDeleteFramebuffers([int(cam_fbo)])
                if cam_gl_tex is not None:
                    GL.glDeleteTextures([int(cam_gl_tex)])
                if quad_vao is not None:
                    GL.glDeleteVertexArrays([int(quad_vao)])
                if quad_prog is not None:
                    GL.glDeleteProgram(int(quad_prog))
                GL.glDeleteFramebuffers([int(pt_fbo)])
                xr.destroy_swapchain(pt_sc)
                for sp in grip_spaces:
                    xr.destroy_space(sp)
                xr.destroy_action_set(action_set)
                xr.destroy_space(ref_space)
                xr.destroy_session(session)
                xr.destroy_instance(instance)
            except Exception:  # noqa: BLE001
                logger.debug("XR teardown raised", exc_info=True)
            try:
                glfw.destroy_window(window)
                glfw.terminate()
            except Exception:  # noqa: BLE001
                pass
            logger.info("OpenXRPoseSource thread exited.")

    @staticmethod
    def _calibrate_body(xr, session, view_config_type, predicted_time, ref_space):
        """Head-based body-frame yaw calibration (port of run_vr)."""
        try:
            _, head_views = xr.locate_views(session, xr.ViewLocateInfo(
                view_config_type, predicted_time, ref_space))
            if not head_views:
                return None, None
            hp = head_views[0].pose
            hq = np.array([hp.orientation.x, hp.orientation.y, hp.orientation.z, hp.orientation.w])
            R_head = np.array([
                [1 - 2 * (hq[1] ** 2 + hq[2] ** 2), 2 * (hq[0] * hq[1] - hq[3] * hq[2]),
                 2 * (hq[0] * hq[2] + hq[3] * hq[1])],
                [2 * (hq[0] * hq[1] + hq[3] * hq[2]), 1 - 2 * (hq[0] ** 2 + hq[2] ** 2),
                 2 * (hq[1] * hq[2] - hq[3] * hq[0])],
                [2 * (hq[0] * hq[2] - hq[3] * hq[1]), 2 * (hq[1] * hq[2] + hq[3] * hq[0]),
                 1 - 2 * (hq[0] ** 2 + hq[1] ** 2)]])
            hfwd = -R_head[:, 2]
            hfwd[1] = 0.0
            n = np.linalg.norm(hfwd)
            hfwd = np.array([0.0, 0.0, -1.0]) if n < 1e-6 else hfwd / n
            body_up = np.array([0.0, 1.0, 0.0])
            body_back = -hfwd
            body_right = np.cross(body_up, body_back)
            body_right /= np.linalg.norm(body_right)
            R_b2s = np.column_stack([body_right, body_up, body_back])
            theta = math.atan2(-hfwd[0], -hfwd[2])
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            logger.info("Body frame calibrated: theta=%.1f deg", math.degrees(theta))
            return R_b2s.T, np.array([c, 0.0, -s, 0.0])
        except Exception:  # noqa: BLE001
            logger.debug("body calibration failed", exc_info=True)
            return None, None
