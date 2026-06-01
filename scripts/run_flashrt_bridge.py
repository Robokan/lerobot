#!/usr/bin/env python
"""Synchronous lerobot <-> FlashRT bridge for the OpenArm chocolate task.

Drives the ``BiOpenArmFollower`` directly against a FlashRT (or openpi) server
that speaks the openpi websocket protocol. The server *is* the policy — this
process does no in-process model inference. The observation / action conversion
mirrors the converted checkpoint's stamped pre/post-processors
(``policy_preprocessor.json``): deg<->rad on ALL 16 dims (joints + grippers, the
``angle_unit_processor`` uses ``exclude_joints: []``) plus the arm-half swap
that maps lerobot's right-first wire layout to openpi's left-first model layout,
and the same 3-gate action safety checker used by the in-process RTC path.

Pipeline per control tick (target_hz, default 50):

    robot.get_observation()                      # wire: [R8, L8], all deg
      -> assemble 16-vec in action_features order
      -> swap halves          [R8,L8] -> [L8,R8] # model layout
      -> deg->rad on all 16 dims (joints + grippers)
      -> JPEG-encode ego/left_wrist/right_wrist
      -> {state, images{cam_high,cam_left_wrist,cam_right_wrist}, prompt}
      -> runner.next_action(obs)                  # AsyncChunkRunner (rtc | sync)
      <- action (16,)  model layout, rad
      -> rad->deg on all 16 dims (joints + grippers)
      -> swap halves          [L8,R8] -> [R8,L8] # back to wire layout
      -> ActionSafetyChecker.check(...)           # hard-abort on bad output
      -> robot.send_action(action_dict)

Two consumer paths, both on FlashRT's ``AsyncChunkRunner`` (the engine its
``ChunkedWebsocketClient`` wraps), built explicitly so the params are auditable
against lerobot's RTCConfig:

  * ``--mode rtc``  (default) — server-side RTC soft-guidance, the port of
    lerobot's ``RTCProcessor``. Params mirror ``run_chocolate_policy_rtc.sh``:
    ``execution_horizon=20``, ``schedule=linear``, full 50-step chunk,
    continuous replan. The lerobot run worked here modulo latency; FlashRT's
    lower delay keeps the splice inside the guided merge window.
  * ``--mode sync`` — full 50-step synchronous replan + 5-step seam blend
    (debug / A-B baseline).

Example::

    python scripts/run_flashrt_bridge.py \
        --server-host localhost --server-port 8011 \
        --left-can can3 --right-can can2 \
        --ego-cam /dev/video0 --left-wrist-cam /dev/video4 --right-wrist-cam /dev/video2 \
        --prompt "put the chocolate bars in the container" \
        --target-hz 50 --duration 120 --mode rtc
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# FlashRT (flash_rt.runtime.AsyncChunkRunner) + openpi websocket client live
# outside the lerobot tree; add them to the path so this script is runnable in
# the lerobot venv without installing either as a package.
sys.path.insert(0, "/home/evaughan/sparkpack/lerobot")
sys.path.insert(0, "/home/evaughan/sparkpack/FlashRT")
sys.path.insert(0, "/home/evaughan/sparkpack/openpi/packages/openpi-client/src")

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, lift_arms_to_ready
from lerobot.robots.bi_openarm_follower.action_safety import (
    ActionSafetyChecker,
    ActionSafetyConfig,
)
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import (
    BiOpenArmFollowerConfig,
)
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.rl.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flashrt_bridge")

# Degrees<->radians scale. The OpenArm follower's wire format is DEGREES for
# every motor (joints AND grippers — Damiao motors use MotorNormMode.DEGREES);
# the openpi/FlashRT server expects RADIANS for every dim. This matches the
# checkpoint's ``angle_unit_processor`` (policy_preprocessor.json), which lists
# all 16 features including ``left_gripper``/``right_gripper`` with
# ``exclude_joints: []`` and scale = pi/180 — i.e. the gripper is deg<->rad
# converted exactly like the joints. (diag_ab skipped the gripper only because
# its dataset state is already stored in radians; a LIVE robot is not.)
_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi

# Camera name (lerobot side) -> openpi server image key.
_CAM_TO_OPENPI_KEY = {
    "ego": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def _swap_halves(vec: np.ndarray) -> np.ndarray:
    """Swap the two equal halves of a 1-D 16-vector ([R8,L8] <-> [L8,R8]).

    Identical permutation to ``lerobot.processor.ArmSwapProcessorStep`` —
    its own inverse, so the same call maps obs (wire->model) and action
    (model->wire).
    """
    half = vec.shape[-1] // 2
    out = vec.copy()
    out[:half] = vec[half:]
    out[half:] = vec[:half]
    return out


def _as_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a camera frame (CHW or HWC, uint8 or float) to HWC uint8 RGB."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _encode_jpeg_rgb(img_hwc_uint8_rgb: np.ndarray) -> bytes:
    """Encode HWC RGB uint8 to JPEG bytes the way the openpi server expects.

    SparkJAX sends BGR cv2-encoded JPEG bytes; the server cv2-decodes (BGR)
    then runs cv2.cvtColor(BGR -> RGB). So RGB->BGR-flip then cv2.imencode
    means the server's BGR->RGB pass restores the original RGB. Mirrors
    ``scripts/diag_ab_openpi_vs_lerobot.py``.
    """
    bgr = cv2.cvtColor(img_hwc_uint8_rgb, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return jpg.tobytes()


def build_robot(args: argparse.Namespace) -> BiOpenArmFollower:
    """Construct the BiOpenArmFollower matching scripts/run_chocolate_policy.sh."""
    cameras = {
        "ego": OpenCVCameraConfig(
            index_or_path=Path(args.ego_cam),
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            fourcc="MJPG",
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=Path(args.left_wrist_cam),
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            fourcc="MJPG",
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=Path(args.right_wrist_cam),
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            fourcc="MJPG",
        ),
    }
    left_arm = OpenArmFollowerConfigBase(
        port=args.left_can,
        side="left",
        use_can_fd=False,
        can_bitrate=args.can_bitrate,
    )
    right_arm = OpenArmFollowerConfigBase(
        port=args.right_can,
        side="right",
        use_can_fd=False,
        can_bitrate=args.can_bitrate,
    )
    # Cameras are shared at the bimanual level (matches the CLI launcher, which
    # passes --robot.cameras at the top level rather than per-arm).
    cfg = BiOpenArmFollowerConfig(
        id=args.robot_id,
        left_arm_config=left_arm,
        right_arm_config=right_arm,
        cameras=cameras,
    )
    return BiOpenArmFollower(cfg)


def obs_to_openpi(
    raw_obs: dict,
    action_keys: list[str],
    prompt: str,
) -> tuple[dict, np.ndarray]:
    """Robot observation -> openpi server obs dict.

    Returns (openpi_obs, state_wire_deg) where state_wire_deg is the raw
    16-vec in wire layout/degrees (kept for logging / safety seeding).
    """
    # 1) assemble the 16-vec in the robot's action_features order ([R8, L8],
    #    all dims in degrees — joints AND grippers).
    state_wire = np.asarray(
        [float(raw_obs[k]) for k in action_keys], dtype=np.float32
    )

    # 2) swap halves -> model layout [L8, R8]; 3) deg->rad on ALL 16 dims
    #    (grippers included — see _DEG2RAD note / checkpoint angle_unit config).
    state_model = _swap_halves(state_wire) * _DEG2RAD

    # 4) JPEG-encode the three views the model was trained on.
    images = {}
    for cam, openpi_key in _CAM_TO_OPENPI_KEY.items():
        if cam not in raw_obs:
            raise KeyError(
                f"observation missing camera '{cam}'; got keys {list(raw_obs)}"
            )
        images[openpi_key] = _encode_jpeg_rgb(_as_hwc_uint8(raw_obs[cam]))

    openpi_obs = {
        "state": state_model.astype(np.float32),
        "images": images,
        "prompt": prompt,
    }
    return openpi_obs, state_wire


def action_to_wire(action_model: np.ndarray, action_keys: list[str]) -> dict[str, float]:
    """openpi model-layout action (rad joints) -> robot wire action_dict (deg)."""
    a = np.asarray(action_model, dtype=np.float32).flatten()
    if a.shape[0] != len(action_keys):
        raise ValueError(
            f"server returned action of dim {a.shape[0]} but robot expects "
            f"{len(action_keys)}"
        )
    # rad->deg on ALL 16 dims (grippers included), then swap back to wire
    # layout [R8, L8]. Mirrors the checkpoint's postprocessor angle_unit
    # (scale 180/pi, exclude_joints: []) + arm_swap.
    a_wire = _swap_halves(a * _RAD2DEG)
    return {key: float(a_wire[i]) for i, key in enumerate(action_keys)}


def build_runner(policy, args):
    """Build the chunk consumer for the chosen mode.

    Both modes are plain ``AsyncChunkRunner`` instances (the same engine
    FlashRT's ``ChunkedWebsocketClient`` wraps) configured explicitly so the
    RTC parameters are auditable against lerobot's RTCConfig rather than hidden
    behind the client's per-mode constants.

      * ``rtc``  — server-side RTC soft-guidance, the port of lerobot's
        ``RTCProcessor``. Continuous replan (``start_next_at=0`` ≈ lerobot's
        ``action_queue_size_to_get_new_actions=48``), splice at the measured
        inference delay ``d``, no client seam blend (the seam is removed inside
        the model by the prefix guidance), full 50-step chunk. Sends
        ``_rtc_prev_chunk`` + ``_rtc_inference_delay`` + ``_rtc_execution_horizon``
        + ``_rtc_schedule`` + ``_rtc_ref_state`` to the server every inference.
        ``_rtc_ref_state`` lets the server re-anchor the delta prefix to the
        current state (lerobot ``_reanchor_relative_rtc_prefix`` parity).
      * ``sync`` — play the whole 50-step chunk, then block on a fresh
        inference; 5-step client seam blend. Debug/A-B baseline.
    """
    from flash_rt.runtime import AsyncChunkRunner, CallablePolicyAdapter, RTCConfig

    # meta_keys lifts the server's normalised model-space chunk out of the
    # response so the runner can feed it back as `_rtc_prev_chunk` next call
    # (the RTC continuity signal). No-op if the server doesn't return it.
    adapter = CallablePolicyAdapter(
        fn=policy.infer,
        output_key="actions",
        meta_keys=("_rtc_chunk_model_space",),
    )
    d_seed = max(0, int(math.ceil((args.expected_latency_ms / 1000.0) * args.target_hz)))

    if args.mode == "rtc":
        cfg = RTCConfig(
            target_hz=args.target_hz,
            action_horizon=args.chunk_len,
            start_next_at=0,
            miss_policy="block",
            blend_steps=0,
            inference_delay_steps=d_seed,
            auto_inference_delay=True,
            enable_prefix_freeze=True,
            execution_horizon=args.rtc_execution_horizon,
            rtc_schedule=args.rtc_schedule,
            # Latency-miss guard: a server rebuild/jitter spike inflates the
            # per-call splice d (= ceil(latency*hz)); cap it to the guided
            # window so a spike degrades to a bounded, continuous splice
            # instead of whipping a proximal joint past the safety gate.
            max_splice_d_steps=args.rtc_execution_horizon,
            # Relative-action re-anchoring: the checkpoint outputs per-step
            # deltas (OpenArm v4 delta_action_mask), so the cached prefix is
            # anchored to the state at the inference that produced it. Tell
            # the runner the obs state key so it forwards that anchor state
            # (_rtc_ref_state); the server then re-expresses the prefix
            # relative to the current state before guidance — the missing
            # piece vs lerobot's _reanchor_relative_rtc_prefix.
            ref_state_key="state",
        )
    else:  # sync
        cfg = RTCConfig(
            target_hz=args.target_hz,
            action_horizon=args.chunk_len,
            start_next_at=args.chunk_len,
            miss_policy="block",
            blend_steps=5,
            inference_delay_steps=d_seed,
            auto_inference_delay=False,
            enable_prefix_freeze=False,
        )
    return AsyncChunkRunner(adapter, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Server
    ap.add_argument("--server-host", default="localhost")
    ap.add_argument("--server-port", type=int, default=8011)
    ap.add_argument(
        "--mode",
        choices=["rtc", "sync"],
        default="rtc",
        help=(
            "rtc = lerobot RTC soft-guidance (server-side prefix guidance, the validated "
            "path; only failed before due to inference latency). sync = full-chunk "
            "synchronous replan + seam blend (debug/A-B baseline)."
        ),
    )
    # RTC params — defaults match scripts/run_chocolate_policy_rtc.sh (the lerobot
    # RTC run that worked). execution_horizon MUST exceed the inference delay d
    # (~13 steps at 246 ms / 50 Hz) so the splice lands in the guided merge
    # window instead of the unguided free region — that latency overrun is
    # exactly what made the lerobot run seam/jerk.
    ap.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=20,
        help="RTC merge-window end (lerobot run used 20). Must be > inference delay d.",
    )
    ap.add_argument(
        "--rtc-schedule",
        choices=["linear", "exp"],
        default="linear",
        help="RTC prefix-attention schedule (lerobot run used LINEAR).",
    )
    ap.add_argument(
        "--expected-latency-ms",
        type=float,
        default=246.0,
        help="Seed latency for the first chunk splice (BF16 FlashRT steady-state ~246ms)",
    )
    ap.add_argument(
        "--chunk-len",
        type=int,
        default=50,
        help="Action chunk horizon. Fixed at 50 (checkpoint + server --chunk-size 50).",
    )
    # Robot / CAN
    ap.add_argument("--robot-id", default="lumpa")
    ap.add_argument("--left-can", default="can3")
    ap.add_argument("--right-can", default="can2")
    ap.add_argument("--can-bitrate", type=int, default=1_000_000)
    # Cameras
    ap.add_argument("--ego-cam", default="/dev/video0")
    ap.add_argument("--left-wrist-cam", default="/dev/video4")
    ap.add_argument("--right-wrist-cam", default="/dev/video2")
    ap.add_argument("--cam-fps", type=int, default=60)
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    # Control loop
    ap.add_argument("--target-hz", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--prompt", default="put the chocolate bars in the container")
    ap.add_argument(
        "--no-lift",
        action="store_true",
        help="Skip lifting arms to the training-distribution ready pose",
    )
    # Action safety (mirrors examples/rtc/eval_with_real_robot.py defaults)
    _safety = ActionSafetyConfig()
    ap.add_argument("--no-action-safety", action="store_true")
    ap.add_argument(
        "--action-safety-max-joint-delta-deg",
        type=float,
        default=_safety.max_joint_delta_deg,
    )
    ap.add_argument(
        "--action-safety-max-gripper-delta-deg",
        type=float,
        default=None,
        help="None (default) disables the per-step delta gate for the gripper",
    )
    ap.add_argument(
        "--action-safety-fast-joints",
        default="joint_7",
        help=(
            "Comma-separated motor-name substrings treated as fast distal joints "
            "(low-inertia, can snap quickly at a chunk seam without whipping the "
            "arm), exempted like the gripper from the standard per-step delta gate. "
            "Default 'joint_7' (the terminal wrist roll). Empty string disables."
        ),
    )
    ap.add_argument(
        "--action-safety-max-fast-joint-delta-deg",
        type=float,
        default=None,
        help=(
            "Per-step delta limit (deg) for the fast-joint class. None (default) "
            "disables the delta gate for them, like the gripper; they remain bounded "
            "by the finite + absolute-envelope gates and the follower's own clamp."
        ),
    )
    ap.add_argument(
        "--action-safety-abs-joint-limit-deg",
        type=float,
        default=_safety.abs_joint_limit_deg,
    )
    ap.add_argument(
        "--action-safety-abs-gripper-limit-deg",
        type=float,
        default=_safety.abs_gripper_limit_deg,
    )
    args = ap.parse_args()

    init_logging()

    # ----- FlashRT / openpi client ----------------------------------------
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    if args.mode == "rtc":
        d_seed = int(math.ceil((args.expected_latency_ms / 1000.0) * args.target_hz))
        logger.info(
            "Connecting to ws://%s:%d  mode=rtc (lerobot soft-guidance) "
            "exec_horizon=%d schedule=%s chunk=%d  seed d=%d steps (%.0f ms @ %.0f Hz)",
            args.server_host,
            args.server_port,
            args.rtc_execution_horizon,
            args.rtc_schedule,
            args.chunk_len,
            d_seed,
            args.expected_latency_ms,
            args.target_hz,
        )
        if d_seed >= args.rtc_execution_horizon:
            logger.warning(
                "inference delay d=%d >= execution_horizon=%d: the splice lands at/past "
                "the merge window so the seam is UNGUIDED (this is the latency overrun "
                "that jerked the lerobot run). Lower latency or raise --rtc-execution-horizon.",
                d_seed,
                args.rtc_execution_horizon,
            )
        # Parity note: lerobot's trusted run used max_guidance_weight=5.0.
        # The FlashRT server now defaults _rtc_max_gw to 5.0 (pipeline_rtx.py),
        # overridable via the FLASHRT_RTC_MAX_GW env on the server. The client
        # cannot set it per-call (the adapter only forwards execution_horizon +
        # schedule). A too-high ceiling over-pulls the chunk toward the prefix
        # in late denoising steps -> overshoot/oscillation; 5.0 matches the
        # config that was tuned on the robot.
        logger.info(
            "max_guidance_weight is server-side (pipeline default 5.0 = lerobot "
            "parity). Set FLASHRT_RTC_MAX_GW on the server to change it."
        )
    else:
        logger.info(
            "Connecting to ws://%s:%d  mode=sync (full-chunk replan + blend5) chunk=%d",
            args.server_host,
            args.server_port,
            args.chunk_len,
        )
    base_policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    client = build_runner(base_policy, args)

    # ----- Robot ----------------------------------------------------------
    robot = build_robot(args)
    logger.info("Connecting robot (left=%s right=%s)...", args.left_can, args.right_can)
    robot.connect()

    if not args.no_lift:
        logger.info("Lifting arms to the training-distribution ready pose...")
        lift_arms_to_ready(robot, log_fn=logger.info)

    action_keys = list(robot.action_features.keys())
    logger.info("Robot action keys (%d): %s", len(action_keys), action_keys)
    if len(action_keys) != 16:
        logger.warning(
            "Expected a 16-DOF bimanual action layout; got %d. The arm-swap "
            "(8+8) and deg<->rad conversion assume that layout.",
            len(action_keys),
        )

    # ----- Action safety --------------------------------------------------
    safety_checker = None
    if not args.no_action_safety:
        fast_joints = tuple(
            s.strip() for s in args.action_safety_fast_joints.split(",") if s.strip()
        )
        safety_cfg = ActionSafetyConfig(
            enabled=True,
            max_joint_delta_deg=args.action_safety_max_joint_delta_deg,
            max_gripper_delta_deg=args.action_safety_max_gripper_delta_deg,
            max_fast_joint_delta_deg=args.action_safety_max_fast_joint_delta_deg,
            fast_joint_name_substrs=fast_joints,
            abs_joint_limit_deg=args.action_safety_abs_joint_limit_deg,
            abs_gripper_limit_deg=args.action_safety_abs_gripper_limit_deg,
        )
        safety_checker = ActionSafetyChecker(safety_cfg)
        logger.info(
            "Action safety armed: joint_delta=%.2f deg/step, gripper_delta=%s, "
            "fast_joints=%s delta=%s, abs_joint=%.1f deg, abs_gripper=%.1f deg",
            safety_cfg.max_joint_delta_deg,
            "disabled"
            if safety_cfg.max_gripper_delta_deg is None
            else f"{safety_cfg.max_gripper_delta_deg:.2f} deg/step",
            list(safety_cfg.fast_joint_name_substrs) or "none",
            "disabled"
            if safety_cfg.max_fast_joint_delta_deg is None
            else f"{safety_cfg.max_fast_joint_delta_deg:.2f} deg/step",
            safety_cfg.abs_joint_limit_deg,
            safety_cfg.abs_gripper_limit_deg,
        )

    # ----- Control loop ---------------------------------------------------
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    control_interval = 1.0 / args.target_hz
    last_good_wire: dict[str, float] | None = None
    tick = 0
    aborted = False
    start = time.time()
    logger.info("Starting control loop: target_hz=%.1f, duration=%.0fs", args.target_hz, args.duration)

    try:
        while not shutdown_event.is_set() and (time.time() - start) < args.duration:
            t0 = time.perf_counter()

            raw_obs = robot.get_observation()
            openpi_obs, _state_wire = obs_to_openpi(raw_obs, action_keys, args.prompt)

            action_model = client.next_action(openpi_obs)
            action_wire = action_to_wire(action_model, action_keys)

            if safety_checker is not None:
                err = safety_checker.check(action_wire, raw_obs)
                if err is not None:
                    logger.error("%s", err)
                    # Hold the last good pose, then bail out.
                    if last_good_wire is not None:
                        robot.send_action(last_good_wire)
                    aborted = True
                    break

            robot.send_action(action_wire)
            last_good_wire = action_wire
            tick += 1

            if tick % int(args.target_hz) == 0:
                logger.info(
                    "[loop] tick=%d t=%.1fs queue/stats=%s",
                    tick,
                    time.time() - start,
                    getattr(client, "stats", ""),
                )

            dt = time.perf_counter() - t0
            sleep_s = control_interval - dt
            if sleep_s > 0:
                time.sleep(sleep_s)
    except Exception:  # noqa: BLE001
        logger.exception("Fatal error in control loop")
        aborted = True
    finally:
        logger.info("Shutting down (ticks executed=%d, aborted=%s)...", tick, aborted)
        try:
            client.close(wait=True)
        except Exception:  # noqa: BLE001
            logger.warning("client.close() failed", exc_info=True)
        try:
            robot.disconnect()
            logger.info("Robot disconnected (torque disabled)")
        except Exception:  # noqa: BLE001
            logger.warning("robot.disconnect() failed", exc_info=True)

    logger.info("FlashRT bridge finished.")


if __name__ == "__main__":
    main()
