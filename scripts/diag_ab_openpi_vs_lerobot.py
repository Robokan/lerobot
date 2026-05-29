"""A/B test: send the same observation to openpi PT server AND lerobot, compare predictions.

This is the definitive test for whether the lerobot port matches openpi numerically.

Strategy:
1. Load a frame from the training dataset (so images are at 224x224 already, matching
   what the openpi server expects and what lerobot trained on).
2. Build TWO obs dicts from the same data:
   - openpi format: state + JPEG-encoded images + prompt -> WebSocket call to server.
   - lerobot format: observation.state + observation.images.* uint8 HWC + task -> local
     policy.predict_action_chunk.
3. Compare the predicted 50-step action chunks side by side, in degrees and as motion deltas.

Note: both pipelines sample flow-matching noise internally. Differences here are dominated
by noise sampling unless we average enough samples. To get a deterministic comparison we
average over N noise draws on the lerobot side and compare to a single openpi sample.

Usage:
    python scripts/diag_ab_openpi_vs_lerobot.py [--frame IDX] [--avg N]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/evaughan/sparkpack/lerobot")
sys.path.insert(0, "/home/evaughan/sparkpack/openpi/packages/openpi-client/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from scripts.load_pi05_from_openpi import load_pi05_with_runtime_lora

POLICY_DIR = Path("outputs/pi05_chocolate_v4_from_openpi")
DATASET_ROOT = Path.home() / ".cache/huggingface/lerobot/local/openarm-chocolate-v4"
DATASET_REPO = "local/openarm-chocolate-v4"

OPENPI_HOST = "localhost"
OPENPI_PORT = 8002


def _img_chw_float_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert one image tensor [3,H,W] float[0,1] or uint8 -> HWC uint8."""
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _as_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a captured image (CHW or HWC, uint8 or float) to HWC uint8."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


# Gripper dims in the 16-D OpenArm state/action layout (left, right). These are
# NOT angles, so they are never deg<->rad converted.
_GRIPPER_DIMS = (7, 15)


def load_captured_frame(path: str):
    """Load a live-captured frame_NNN.npz dumped by lerobot_record.

    Returns (state_train_rad, state_le_wire, images_rgb, task) where:
      * state_le_wire is the EXACT state lerobot saw live (joints in degrees,
        gripper in native units) — feed this straight to the lerobot policy.
      * state_train_rad is that state converted to openpi/dataset units
        (joints deg->rad, gripper untouched) — feed this to the openpi server
        and use it as the comparison baseline.
    """
    data = np.load(path, allow_pickle=True)
    state_le = np.asarray(data["observation.state"], dtype=np.float32).flatten()
    joint_idx = [i for i in range(state_le.shape[-1]) if i not in _GRIPPER_DIMS]
    state_train_rad = state_le.copy()
    state_train_rad[joint_idx] = np.deg2rad(state_le[joint_idx])

    images_rgb = {}
    for cam in ("ego", "left_wrist", "right_wrist"):
        key = f"observation.images.{cam}"
        if key not in data:
            raise KeyError(f"captured frame missing {key}; keys={list(data.keys())}")
        images_rgb[cam] = _as_hwc_uint8(data[key])

    task = str(data["task"]) if "task" in data else "put the chocolate bars in the container"
    if not task:
        task = "put the chocolate bars in the container"
    return state_train_rad, state_le, images_rgb, task


def _encode_jpeg_rgb(img_hwc_uint8_rgb: np.ndarray) -> bytes:
    """Encode HWC RGB uint8 to JPEG bytes the way the openpi server expects.

    SparkJAX sends BGR cv2-encoded JPEG bytes; the server cv2-decodes (BGR) and then
    runs cv2.cvtColor(BGR -> RGB). So if we send RGB->BGR-flip then cv2.imencode,
    the server's BGR->RGB pass restores the original RGB. We mimic exactly that path.
    """
    bgr = cv2.cvtColor(img_hwc_uint8_rgb, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return jpg.tobytes()


def call_openpi(state: np.ndarray, images_rgb: dict[str, np.ndarray],
                prompt: str) -> np.ndarray:
    """One synchronous WebSocket round-trip to the openpi PT server."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    obs = {
        "state": state.astype(np.float32),
        "images": {
            "cam_high": _encode_jpeg_rgb(images_rgb["ego"]),
            "cam_left_wrist": _encode_jpeg_rgb(images_rgb["left_wrist"]),
            "cam_right_wrist": _encode_jpeg_rgb(images_rgb["right_wrist"]),
        },
        "prompt": prompt,
    }
    cli = WebsocketClientPolicy(host=OPENPI_HOST, port=OPENPI_PORT)
    t0 = time.time()
    result = cli.infer(obs)
    print(f"  [openpi]  round trip {1000*(time.time()-t0):.0f} ms, "
          f"keys={list(result.keys())}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    return actions  # shape [50, 16]


def call_lerobot(state: np.ndarray, images_rgb: dict[str, np.ndarray],
                 task: str, *, policy, preprocessor, postprocessor,
                 device) -> np.ndarray:
    """Run the same observation through the lerobot PI05 policy in-process.

    Calls predict_action `chunk_size` times against the same observation; the
    first call runs the model and caches the 50-step chunk, the rest pop from
    the queue. This mirrors the exact live-inference path used on the robot.
    """
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    obs = {
        "observation.state": state.astype(np.float32),
        "observation.images.ego": images_rgb["ego"],
        "observation.images.left_wrist": images_rgb["left_wrist"],
        "observation.images.right_wrist": images_rgb["right_wrist"],
    }
    chunk_len = policy.config.chunk_size
    chunk = []
    t0 = time.time()
    for i in range(chunk_len):
        a = predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type="bi_openarm_follower",
        )
        chunk.append(a.detach().cpu().numpy().flatten())
    print(f"  [lerobot] {chunk_len}-step chunk in "
          f"{1000*(time.time()-t0):.0f} ms total")
    return np.stack(chunk, axis=0).astype(np.float32)  # [50, 16]


def compare(chunk_op: np.ndarray, chunk_le: np.ndarray, state: np.ndarray,
            label_action: np.ndarray, label_chunk: np.ndarray | None) -> None:
    n = min(chunk_op.shape[0], chunk_le.shape[0])
    chunk_op = chunk_op[:n]
    chunk_le = chunk_le[:n]
    diff = chunk_op - chunk_le
    print("\n=== ACTION CHUNK COMPARISON (degrees) ===")
    print(f"openpi  : shape={chunk_op.shape} min/max/mean(deg) = "
          f"{np.rad2deg(chunk_op.min()):+.2f} / {np.rad2deg(chunk_op.max()):+.2f} / "
          f"{np.rad2deg(chunk_op.mean()):+.2f}")
    print(f"lerobot : shape={chunk_le.shape} min/max/mean(deg) = "
          f"{np.rad2deg(chunk_le.min()):+.2f} / {np.rad2deg(chunk_le.max()):+.2f} / "
          f"{np.rad2deg(chunk_le.mean()):+.2f}")
    print(f"|openpi - lerobot| (deg): mean={np.rad2deg(np.abs(diff)).mean():.3f} "
          f"max={np.rad2deg(np.abs(diff)).max():.3f}")

    motion_op = chunk_op - state[None, :]
    motion_le = chunk_le - state[None, :]
    op_max_motion = np.abs(np.rad2deg(motion_op)).max()
    le_max_motion = np.abs(np.rad2deg(motion_le)).max()
    print(f"\nMotion from current state (max |Δ| deg):  "
          f"openpi={op_max_motion:.2f}  lerobot={le_max_motion:.2f}")

    # Step-by-step first/middle/last for the 16 dims
    def fmt_step(label: str, action: np.ndarray) -> str:
        d = np.rad2deg(action)
        return (f"{label}: L=[{d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f},{d[3]:+.2f},"
                f"{d[4]:+.2f},{d[5]:+.2f},{d[6]:+.2f}] gripL={d[7]:+.3f} | "
                f"R=[{d[8]:+.2f},{d[9]:+.2f},{d[10]:+.2f},{d[11]:+.2f},"
                f"{d[12]:+.2f},{d[13]:+.2f},{d[14]:+.2f}] gripR={d[15]:+.3f}")

    print("\n--- step 0 ---")
    print(fmt_step("openpi ", chunk_op[0]))
    print(fmt_step("lerobot", chunk_le[0]))
    print("\n--- step 24 (middle) ---")
    print(fmt_step("openpi ", chunk_op[24]))
    print(fmt_step("lerobot", chunk_le[24]))
    print("\n--- step 49 (last) ---")
    print(fmt_step("openpi ", chunk_op[-1]))
    print(fmt_step("lerobot", chunk_le[-1]))

    if label_chunk is not None:
        print("\n--- ground truth chunk (training labels) ---")
        print(fmt_step("label0 ", label_chunk[0]))
        print(fmt_step("label24", label_chunk[min(24, len(label_chunk)-1)]))
        print(fmt_step("label49", label_chunk[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=43800,
                    help="Dataset frame index to test on (HIGH-pose active default)")
    ap.add_argument("--avg", type=int, default=1,
                    help="Avg over N samples on each side to absorb noise")
    ap.add_argument(
        "--upscale-to",
        default=None,
        help=(
            "If set as 'HxW' (e.g. '480x640'), upscale each 224x224 training "
            "image to this resolution (with letterbox black bars) and send "
            "THAT to both pipelines, forcing them to exercise their internal "
            "resize-with-pad path. This is the closest static repro of a live "
            "lerobot camera frame against a static openpi snapshot."
        ),
    )
    ap.add_argument(
        "--state-units",
        choices=["radians", "degrees"],
        default="radians",
        help=(
            "Units to use for the STATE input to lerobot (openpi server always "
            "gets radians since it's the dataset convention). 'degrees' simulates "
            "what bi_openarm_follower.get_observation() actually returns at live "
            "inference (Damiao motors are in degrees) without applying any "
            "rad->deg conversion at the robot boundary."
        ),
    )
    ap.add_argument(
        "--captured-frame",
        default=None,
        help=(
            "Path to a frame_NNN.npz captured live by lerobot_record "
            "(LEROBOT_DUMP_OBS_DIR=...). Uses the REAL live camera images and "
            "live (degree-wire) state instead of a training-dataset frame. This "
            "is the decisive test: feed the exact live observation to BOTH the "
            "openpi server and lerobot and see if they agree. If openpi also "
            "drifts up/holds on these images, the live camera frames are the "
            "problem; if openpi reaches for chocolate, the images are fine."
        ),
    )
    ap.add_argument(
        "--swap-rb",
        action="store_true",
        help=(
            "Swap R/B channels of the images before sending to the OPENPI side "
            "only. Use to test whether a BGR/RGB mismatch in the live capture "
            "is what breaks the model (lerobot's OpenCVCamera defaults to RGB; "
            "if the model was actually trained expecting the other order this "
            "flag flips it for the openpi reference)."
        ),
    )
    args = ap.parse_args()

    init_logging()

    # `state` is always in openpi/dataset units (radians joints) — it's what we
    # send to the openpi server and use as the comparison baseline. `forced_state_le`
    # (if set) is the exact wire-format state to feed lerobot.
    is_captured = args.captured_frame is not None
    forced_state_le: np.ndarray | None = None
    if is_captured:
        print(f"Loading CAPTURED LIVE frame from {args.captured_frame} ...")
        state, forced_state_le, images_rgb, task = load_captured_frame(args.captured_frame)
        label_action = state.copy()  # no ground-truth label for a live frame
        for cam in ("ego", "left_wrist", "right_wrist"):
            print(f"  cam {cam}: shape={images_rgb[cam].shape} "
                  f"dtype={images_rgb[cam].dtype} mean={images_rgb[cam].mean():.1f}")
    else:
        print(f"Loading dataset {DATASET_REPO} from {DATASET_ROOT} ...")
        ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)

        sample = ds[args.frame]
        state = sample["observation.state"].numpy().astype(np.float32)
        label_action = sample["action"].numpy().astype(np.float32)
        task = sample.get("task", "put the chocolate bars in the container")
        if not isinstance(task, str):
            task = "put the chocolate bars in the container"

        images_rgb = {}
        for cam in ("ego", "left_wrist", "right_wrist"):
            img_t = sample[f"observation.images.{cam}"]
            images_rgb[cam] = _img_chw_float_to_hwc_uint8(img_t)
            print(f"  cam {cam}: shape={images_rgb[cam].shape} "
                  f"dtype={images_rgb[cam].dtype} mean={images_rgb[cam].mean():.1f}")

    if args.upscale_to is not None:
        target_h, target_w = (int(x) for x in args.upscale_to.lower().split("x"))
        print(f"\nLetterbox-upscaling each 224x224 image to {target_h}x{target_w} ...")
        for cam, img in images_rgb.items():
            h0, w0 = img.shape[:2]
            ratio = min(target_w / w0, target_h / h0)
            new_w = int(w0 * ratio)
            new_h = int(h0 * ratio)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            pad_y = (target_h - new_h) // 2
            pad_x = (target_w - new_w) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            images_rgb[cam] = canvas
            print(
                f"  cam {cam}: {h0}x{w0} -> {target_h}x{target_w} "
                f"(content {new_h}x{new_w}, pad_y={pad_y}, pad_x={pad_x})"
            )

    print(f"\nFrame {args.frame}  task={task!r}")
    print(f"state(deg): {np.rad2deg(state).round(2)}")
    print(f"label action(deg): {np.rad2deg(label_action).round(2)}")
    print(f"label - state (deg, motion): {np.rad2deg(label_action - state).round(2)}")

    # Try to pull the ground-truth chunk if the dataset stores delta-action labels.
    label_chunk = None
    try:
        action_seq = sample["action_is_pad"]  # raise if not present
    except Exception:
        action_seq = None

    print("\nLoading lerobot policy ...")
    meta = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    policy, pre, post = load_pi05_with_runtime_lora(
        POLICY_DIR, ds_meta=meta, device="cuda")
    device = get_safe_torch_device("cuda")

    images_for_openpi = images_rgb
    if args.swap_rb:
        print("\n!! --swap-rb: flipping R/B channels for the OPENPI side only !!")
        images_for_openpi = {cam: np.ascontiguousarray(img[:, :, ::-1])
                             for cam, img in images_rgb.items()}

    print(f"\n=== Calling openpi server ({args.avg}x noise samples) ===")
    chunks_op = []
    for i in range(args.avg):
        c = call_openpi(state, images_for_openpi, task)
        chunks_op.append(c)
        print(f"   openpi  sample {i}: step0 gripL={c[0, 7]:+.3f} "
              f"step49 gripL={c[-1, 7]:+.3f}")
    chunk_op = np.mean(chunks_op, axis=0)
    std_op = np.std(chunks_op, axis=0)

    if is_captured:
        # The captured state is ALREADY the live wire format (joints in degrees,
        # gripper native) — feed it straight to lerobot, no synthetic conversion.
        state_le = forced_state_le
        print(
            f"\n!! Using CAPTURED live wire-format state for lerobot "
            f"(joints in degrees) !!\n"
            f"   state_rad (openpi units): {state.round(3)}\n"
            f"   state_wire (lerobot in): {state_le.round(3)}\n"
        )
    elif args.state_units == "degrees":
        # Simulate the live wire format: the OpenArm follower reports JOINT
        # angles in degrees but the gripper in its own (dataset-matching) units.
        # So convert only joint dims to degrees; leave gripper dims (7, 15) as
        # the raw radian-dataset value. Converting the gripper too would be a
        # 57x OOD corruption that has nothing to do with the real robot.
        joint_idx = [i for i in range(state.shape[-1]) if i not in (7, 15)]
        state_le = state.astype(np.float32).copy()
        state_le[joint_idx] = np.rad2deg(state[joint_idx])
        print(
            f"\n!! Sending lerobot the STATE IN DEGREES (joints only; gripper in "
            f"native units, simulating live robot) !!\n"
            f"   state_rad (training units): {state.round(3)}\n"
            f"   state_deg (live wire fmt):  {state_le.round(3)}\n"
        )
    else:
        state_le = state
    print(f"\n=== Calling lerobot ({args.avg}x noise samples, "
          f"state_units={args.state_units}) ===")
    chunks_le = []
    for i in range(args.avg):
        c = call_lerobot(state_le, images_rgb, task,
                          policy=policy, preprocessor=pre, postprocessor=post,
                          device=device)
        chunks_le.append(c)
        print(f"   lerobot sample {i}: step0 gripL={c[0, 7]:+.3f} "
              f"step49 gripL={c[-1, 7]:+.3f}")
    chunk_le = np.mean(chunks_le, axis=0)
    std_le = np.std(chunks_le, axis=0)

    # If the checkpoint declares radian training units, the processor pipeline
    # now emits the action in DEGREES (wire format) while openpi emits radians.
    # Convert lerobot's joint dims back to radians so the comparison below (all
    # in radians) is apples-to-apples. Gripper dims (7, 15) are excluded from
    # the angle conversion in the pipeline, so leave them untouched here too.
    angle_unit = getattr(policy.config, "input_angle_unit", "degrees")
    if angle_unit == "radians":
        joint_idx = [i for i in range(chunk_le.shape[-1]) if i not in (7, 15)]
        print(f"\n[unit] checkpoint input_angle_unit='radians' -> lerobot output is "
              f"in DEGREES; converting joint dims {joint_idx} back to radians for "
              f"comparison (grippers 7,15 left as-is).")
        chunk_le = chunk_le.copy()
        chunk_le[..., joint_idx] = np.deg2rad(chunk_le[..., joint_idx])
        std_le = std_le.copy()
        std_le[..., joint_idx] = np.deg2rad(std_le[..., joint_idx])

    compare(chunk_op, chunk_le, state, label_action, label_chunk)

    print("\n=== Gripper-specific summary (means and stds across noise samples) ===")
    for step in (0, 24, 49):
        print(
            f"step {step:>2d}: "
            f"openpi  gripL={chunk_op[step, 7]:+.3f}±{std_op[step, 7]:.3f}  "
            f"gripR={chunk_op[step, 15]:+.3f}±{std_op[step, 15]:.3f} | "
            f"lerobot gripL={chunk_le[step, 7]:+.3f}±{std_le[step, 7]:.3f}  "
            f"gripR={chunk_le[step, 15]:+.3f}±{std_le[step, 15]:.3f}"
        )


if __name__ == "__main__":
    main()
