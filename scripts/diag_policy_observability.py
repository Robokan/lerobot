"""Three-part diagnostic to answer: "Why is the chocolate policy not
attempting to grab the chocolate, and instead drifting downward?"

Suspected root causes (most likely first):
  (A) Camera index mapping is wrong - ``ego``/``left_wrist``/``right_wrist``
      may not be plugged into the USB hub ports the script assumes
      (/dev/video0/2/4). The policy then sees images that don't match
      what it was trained on, producing some random "fallback" motion.
  (B) Vision tower weights silently failed to load (the
      "Vision embedding key might need handling" warning at run start
      suggests this is at least *possible*). A blind model would produce
      generic motion regardless of what the cameras show.
  (C) The model itself is broken end-to-end (LoRA not applied correctly,
      norm stats wrong, etc.). LoRA was confirmed patched at runtime
      (252 modules), but a "blind run on training data" sanity check
      is still worth doing.

What this script does:
  1. Camera test
       Open /dev/video0, /dev/video2, /dev/video4 with OpenCV (matching
       what lerobot-record uses). Grab one frame from each and save as
       PNG under /tmp/diag_camera_<name>_<dev>.png. Print mean/min/max
       per channel and a short text "looks bright / dark / etc." so the
       user can visually verify the mapping by opening the PNGs.

  2. Vision tower sanity
       Load the policy from the same path the run uses. Walk to the
       SigLIP patch_embedding weights and report:
         - shape, dtype, finite-fraction, mean, std, abs-max
         - whether it looks like loaded pretrained weights (small std,
           non-zero mean) or like fresh random init (large std, mean~0).

  3. Tower test (run policy on a known training frame)
       Load the openarm-chocolate-v4_old dataset, pull the first frame
       of episode 0 (observation.state + 3 images), build a fake
       lerobot-record-style observation dict, and run the same
       predict_action() the live loop uses. Print:
         - the model's first chunk action (post all processors)
         - the training-label action at frame 0
         - the per-joint delta (model - label, in degrees)
       If the model's prediction is close to the training label, the
       model is fine and the live failure must be (A) cameras or (C2)
       state at inference time being out-of-distribution.

Run:
    source .venv/bin/activate
    python scripts/diag_policy_observability.py \\
        --policy-dir outputs/pi05_chocolate_v4_from_openpi \\
        --dataset-root /home/evaughan/.cache/huggingface/lerobot/local/openarm-chocolate-v4_old \\
        --ego-dev /dev/video0 \\
        --left-wrist-dev /dev/video4 \\
        --right-wrist-dev /dev/video2 \\
        --out-dir /tmp/diag_policy

Skip the camera test (e.g. when cameras are claimed by another process):
    python scripts/diag_policy_observability.py ... --skip-cameras
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _print_section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def _stat_summary(name: str, arr: np.ndarray) -> str:
    finite = np.isfinite(arr).mean()
    return (
        f"{name:30s} shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"finite={finite:.4f}  min={arr.min():+.4f} max={arr.max():+.4f} "
        f"mean={arr.mean():+.4f} std={arr.std():+.4f}"
    )


# Shared helper: override the device_processor steps' target device so
# tokens / images stay on the same device as the model. The saved JSON
# pipelines hard-code "cpu"; lerobot-record overrides via preprocessor_
# overrides, and we have to mirror that here for `predict_action` to work.
# DeviceProcessorStep caches `tensor_device` (a torch.device) in
# `__post_init__`, so we have to update BOTH `device` (str) and `tensor_device`
# (torch.device) - otherwise `_process_tensor` keeps moving things to cpu.
def _override_device_processors(preprocessor, postprocessor, device) -> None:
    import torch as _torch  # noqa: PLC0415
    dev_str = str(device) if hasattr(device, "type") else str(device)
    tdev = device if isinstance(device, _torch.device) else _torch.device(dev_str)
    for pipe in (preprocessor, postprocessor):
        for step in pipe.steps:
            if type(step).__name__ == "DeviceProcessorStep":
                step.device = tdev.type
                step.tensor_device = tdev
                step.non_blocking = "cuda" in str(step.device)


# ---------------------------------------------------------------------------
# Part 1: camera test
# ---------------------------------------------------------------------------
def _capture_one_frame(dev: str, label: str, out_dir: Path, settle_frames: int = 5):
    """Open a /dev/videoN with OpenCV, drop the first few frames (auto-
    exposure / white-balance settle), then capture and save one PNG.

    Returns the captured BGR ndarray (H, W, 3) or None on failure.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        print(f"[camera] OpenCV not available; can't open {dev}")
        return None

    print(f"[camera] {label}: opening {dev} ...")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[camera] {label}: FAILED to open {dev} (busy/missing?)")
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(settle_frames):
            cap.read()
            time.sleep(0.03)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[camera] {label}: read() failed on {dev}")
            return None
        dev_tag = dev.replace("/", "_").lstrip("_")
        out = out_dir / f"diag_camera_{label}_{dev_tag}.png"
        cv2.imwrite(str(out), frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        b, g, r = frame.mean(axis=(0, 1))  # OpenCV order
        brightness = float(rgb.mean())
        print(
            f"[camera] {label} ({dev}) -> {out}  "
            f"shape={frame.shape}  brightness={brightness:5.1f}  "
            f"BGR_mean=({b:5.1f},{g:5.1f},{r:5.1f})"
        )
        return frame
    finally:
        cap.release()


def run_camera_test(out_dir: Path, ego: str, left_wrist: str, right_wrist: str) -> None:
    _print_section(
        "PART 1: camera test\n"
        "(open the saved PNGs and visually confirm which physical view is "
        "ego/left_wrist/right_wrist. If they're swapped, the policy sees garbage.)"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, dev in [("ego", ego), ("left_wrist", left_wrist), ("right_wrist", right_wrist)]:
        _capture_one_frame(dev, label, out_dir)


# ---------------------------------------------------------------------------
# Part 2: vision tower sanity
# ---------------------------------------------------------------------------
def run_vision_tower_check(policy_dir: Path) -> "PI05Policy | None":  # type: ignore[name-defined]
    _print_section(
        "PART 2: vision tower sanity\n"
        "(load policy, inspect SigLIP patch_embedding weights; loaded "
        "pretrained weights should have small std (<~0.1) and shape "
        "(hidden, 3, 14, 14). Random init would have std=1.)"
    )
    print(f"[vision] loading policy from {policy_dir} (~30-60s, ~8GB on CPU first)...")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415

    policy = PI05Policy.from_pretrained(str(policy_dir))
    print(f"[vision] loaded. device={next(policy.parameters()).device}")

    pe_w = None
    pe_b = None
    for name, p in policy.named_parameters():
        if name.endswith("vision_tower.vision_model.embeddings.patch_embedding.weight"):
            pe_w = (name, p)
        elif name.endswith("vision_tower.vision_model.embeddings.patch_embedding.bias"):
            pe_b = (name, p)
    if pe_w is None:
        print("[vision] !!! patch_embedding.weight NOT FOUND in the model. Model is BLIND.")
    else:
        name, p = pe_w
        arr = p.detach().to(dtype=__import__("torch").float32).cpu().numpy()
        print(_stat_summary(f"patch_embedding.weight ({name.split('.')[-3]})", arr))
        # SigLIP base weights typically have std ~0.02-0.05. Random init (kaiming) would have std ~0.3.
        if arr.std() > 0.2:
            print("[vision] WARNING: std looks larger than trained SigLIP (random init?).")
        elif np.isfinite(arr).mean() < 0.99:
            print("[vision] WARNING: many non-finite values - load is corrupt.")
        else:
            print("[vision] OK: patch_embedding.weight stats are consistent with loaded SigLIP weights.")
    if pe_b is not None:
        name, p = pe_b
        arr = p.detach().to(dtype=__import__("torch").float32).cpu().numpy()
        print(_stat_summary(f"patch_embedding.bias", arr))

    return policy


# ---------------------------------------------------------------------------
# Part 3: tower test on a known training frame
# ---------------------------------------------------------------------------
def run_tower_test(policy_dir: Path, dataset_root: Path, policy=None) -> None:
    _print_section(
        "PART 3: tower test (run policy on a known training frame)\n"
        "(load first frame of training episode 0, run inference, compare "
        "predicted action to the recorded label. A working model should "
        "predict close to the label. A drifting/blind model will not.)"
    )
    import torch  # noqa: PLC0415

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.processor.pipeline import PolicyProcessorPipeline  # noqa: PLC0415
    from lerobot.processor.converters import (  # noqa: PLC0415
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415
    from lerobot.utils.device_utils import get_safe_torch_device  # noqa: PLC0415

    print(f"[tower] loading dataset {dataset_root.name} ...")
    ds = LeRobotDataset(repo_id=dataset_root.name, root=str(dataset_root))
    print(f"[tower] dataset: {ds.num_episodes} episodes, fps={ds.fps}, features={list(ds.features.keys())[:6]}...")

    if policy is None:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415

        print(f"[tower] loading policy from {policy_dir} ...")
        policy = PI05Policy.from_pretrained(str(policy_dir))

    # Move policy to GPU if available - matches what lerobot-record does
    device = get_safe_torch_device(policy.config.device)
    policy.to(device)
    policy.eval()
    print(f"[tower] policy on {device}, dtype={next(policy.parameters()).dtype}")

    # Build preprocessor/postprocessor from the saved config. We MUST set the
    # to_transition / to_output converters explicitly because the postprocessor
    # consumes Tensor inputs (the model's predicted action), not dict inputs.
    # See ``make_pre_post_processors`` in policies/factory.py for the canonical
    # wiring. Without this, the postprocessor crashes with
    # "EnvTransition must be a dictionary. Got Tensor".
    print(f"[tower] loading pre/postprocessor pipelines from {policy_dir} ...")
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    # Override the saved device_processor's target device to match the policy
    # (lerobot-record does this via preprocessor_overrides; the saved configs
    # hard-code "cpu" and would otherwise move tokens off cuda mid-pipeline).
    _override_device_processors(preprocessor, postprocessor, device)
    # Wire up the relative_step ref on AbsoluteActionsProcessorStep (lerobot
    # usually does this in factory.make_pre_post_processors; here we mirror
    # the same wiring manually).
    rel_step = None
    for step in preprocessor.steps:
        if type(step).__name__ == "RelativeActionsProcessorStep":
            rel_step = step
            break
    if rel_step is not None:
        for step in postprocessor.steps:
            if type(step).__name__ == "AbsoluteActionsProcessorStep":
                step.relative_step = rel_step

    # Pull the first frame of episode 0.
    print(f"[tower] fetching first frame of episode 0 ...")
    sample = ds[0]  # frame 0 of episode 0 (LeRobotDataset is flat-indexed)
    state = sample["observation.state"].detach().cpu().numpy()  # (16,)
    label_action = sample["action"].detach().cpu().numpy()  # (16,)
    task = sample.get("task", "put the chocolate bars in the container")
    print(f"[tower] frame 0 state    [rad]: {state}")
    print(f"[tower] frame 0 state    [deg]: {np.degrees(state)}")
    print(f"[tower] frame 0 label_action [rad]: {label_action}")
    print(f"[tower] frame 0 label_action [deg]: {np.degrees(label_action)}")
    print(f"[tower] frame 0 task: {task!r}")

    # Build lerobot-record-style observation dict (np arrays, channel-last uint8 images)
    obs: dict = {}
    for k, v in sample.items():
        if k.startswith("observation.images."):
            img = v
            # LeRobotDataset returns CHW float in [0,1]. Convert to HWC uint8.
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            obs[k] = img
            print(_stat_summary(f"obs[{k}]", img))
    obs["observation.state"] = state.astype(np.float32)

    # Run inference. Match predict_action's signature.
    print(f"[tower] running predict_action() ...")
    preprocessor.reset()
    postprocessor.reset()
    policy.reset()

    t0 = time.perf_counter()
    action = predict_action(
        observation=obs,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=policy.config.use_amp,
        task=task,
        robot_type="bi_openarm_follower",
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[tower] inference took {dt_ms:.0f} ms")

    # `action` is a (16,) torch tensor in radians, post-all-processors.
    pred = action.detach().cpu().numpy()
    print(f"[tower] predicted action [rad]: {pred}")
    print(f"[tower] predicted action [deg]: {np.degrees(pred)}")
    diff = pred - label_action
    print(f"[tower] (pred - label)    [deg]: {np.degrees(diff)}")
    print(f"[tower] |pred - label|.max [deg] = {np.degrees(np.abs(diff)).max():.3f}")
    print(f"[tower] |pred - label|.mean [deg] = {np.degrees(np.abs(diff)).mean():.3f}")

    # Diagnose
    if np.degrees(np.abs(diff)).mean() < 5.0:
        print(
            "[tower] OK: predicted action is close to recorded label "
            "(<5 deg mean). Model itself is fine on TRAINING data. "
            "The live failure must be from runtime observations (cameras, "
            "state distribution, or task)."
        )
    elif np.degrees(np.abs(diff)).mean() < 15.0:
        print(
            "[tower] MARGINAL: predicted action differs from label by 5-15 deg mean. "
            "Could be model is slightly off-distribution, or processors aren't "
            "wired the same as training."
        )
    else:
        print(
            "[tower] BAD: predicted action diverges >15 deg mean from training label. "
            "Model is broken even on training data - LoRA, norm stats, or the "
            "model itself is wrong."
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Part 4: cross-domain test
# Feed the model the LIVE camera frames we captured in Part 1 + a synthetic
# READY-pose state + the chocolate task, and see what it predicts. Then do
# the same with the equivalent TRAINING frame (an episode whose state is
# similar to READY) and see what it predicts. The difference tells us:
#   - if live cams produce sane output vs training cams in the same state,
#     the cameras are the problem (swap / orientation / FOV / lighting).
#   - if live cams also produce sane output, the issue is somewhere else.
# ---------------------------------------------------------------------------
def run_cross_domain_test(policy_dir: Path, dataset_root: Path, live_cam_dir: Path, policy=None) -> None:
    _print_section(
        "PART 4: cross-domain test\n"
        "(model on TRAINING episode that starts at HIGH cluster + same model on \n"
        " LIVE cameras + synthetic READY-pose state. Compare predicted actions.)"
    )
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    from lerobot.processor.pipeline import PolicyProcessorPipeline  # noqa: PLC0415
    from lerobot.processor.converters import (  # noqa: PLC0415
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415
    from lerobot.utils.device_utils import get_safe_torch_device  # noqa: PLC0415

    if policy is None:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415
        policy = PI05Policy.from_pretrained(str(policy_dir))

    device = get_safe_torch_device(policy.config.device)
    policy.to(device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir), config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir), config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    _override_device_processors(preprocessor, postprocessor, device)
    rel_step = None
    for step in preprocessor.steps:
        if type(step).__name__ == "RelativeActionsProcessorStep":
            rel_step = step
            break
    if rel_step is not None:
        for step in postprocessor.steps:
            if type(step).__name__ == "AbsoluteActionsProcessorStep":
                step.relative_step = rel_step

    task = "put the chocolate bars in the container"

    # ---- A: training frame whose state matches HIGH cluster (j4 ~ 75 deg)
    print("\n--- Test A: TRAINING episode with HIGH-cluster start state (j4 ~ 75 deg) ---")
    ds = LeRobotDataset(dataset_root.name, root=str(dataset_root))
    eps = ds.meta.episodes
    starts_idx = eps["dataset_from_index"]
    # find first episode whose left j4 starts >= 60 deg (HIGH cluster)
    chosen = None
    for i, f in enumerate(starts_idx):
        s = ds[int(f)]["observation.state"].detach().cpu().numpy()
        if np.degrees(s[3]) >= 60.0 and np.degrees(s[11]) >= 60.0:
            chosen = (i, int(f), s)
            print(f"  chose ep={i} (frame_idx={int(f)}) "
                  f"state[deg]={np.degrees(s).round(1)}")
            break
    if chosen is None:
        print("  no HIGH-cluster episode found in dataset")
        return
    ep_idx, frame_idx, train_state = chosen
    train_sample = ds[frame_idx]
    train_label_action = train_sample["action"].detach().cpu().numpy()

    def _to_uint8_hwc(img):
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    train_obs = {"observation.state": train_state.astype(np.float32)}
    for k, v in train_sample.items():
        if k.startswith("observation.images."):
            train_obs[k] = _to_uint8_hwc(v)
            print(_stat_summary(f"train_obs[{k}]", train_obs[k]))

    preprocessor.reset(); postprocessor.reset(); policy.reset()
    pred_train = predict_action(
        observation=train_obs, policy=policy, device=device,
        preprocessor=preprocessor, postprocessor=postprocessor,
        use_amp=policy.config.use_amp, task=task,
        robot_type="bi_openarm_follower",
    ).detach().cpu().numpy()
    diff_train = pred_train.flatten() - train_label_action
    print(f"  predicted [deg]:  {np.degrees(pred_train).round(2)}")
    print(f"  label     [deg]:  {np.degrees(train_label_action).round(2)}")
    print(f"  |pred-label| mean [deg] = {np.degrees(np.abs(diff_train)).mean():.3f}, max = {np.degrees(np.abs(diff_train)).max():.3f}")

    # ---- B: LIVE cameras + SAME training state (isolate camera contribution)
    print("\n--- Test B: LIVE cameras + same HIGH-cluster TRAINING state ---")
    live_obs = {"observation.state": train_state.astype(np.float32)}
    cam_files = {
        "observation.images.ego": "diag_camera_ego_dev_video0.png",
        "observation.images.left_wrist": "diag_camera_left_wrist_dev_video4.png",
        "observation.images.right_wrist": "diag_camera_right_wrist_dev_video2.png",
    }
    for k, fname in cam_files.items():
        fp = live_cam_dir / fname
        if not fp.exists():
            print(f"  !! missing {fp} - re-run with cameras enabled. Aborting Test B.")
            return
        bgr = cv2.imread(str(fp))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Resize to 224x224 like the training images (we'll let the model's
        # internal resize_with_pad_torch handle it from 480x640, but for
        # apples-to-apples we pre-resize the same way OpenCV does).
        rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        live_obs[k] = rgb_224
        print(_stat_summary(f"live_obs[{k}]", live_obs[k]))

    preprocessor.reset(); postprocessor.reset(); policy.reset()
    pred_live = predict_action(
        observation=live_obs, policy=policy, device=device,
        preprocessor=preprocessor, postprocessor=postprocessor,
        use_amp=policy.config.use_amp, task=task,
        robot_type="bi_openarm_follower",
    ).detach().cpu().numpy()
    diff_live = pred_live.flatten() - train_state  # delta from current state
    print(f"  predicted [deg]:  {np.degrees(pred_live).round(2)}")
    print(f"  state     [deg]:  {np.degrees(train_state).round(2)}")
    print(f"  (pred - state) [deg]: {np.degrees(diff_live).round(2)}")

    # Compare A and B predictions
    print("\n--- Comparison A vs B (training cams vs LIVE cams, SAME state) ---")
    diff_A_B = pred_live.flatten() - pred_train.flatten()
    print(f"  pred_train  [deg]: {np.degrees(pred_train).round(2)}")
    print(f"  pred_live   [deg]: {np.degrees(pred_live).round(2)}")
    print(f"  (live-train) [deg]: {np.degrees(diff_A_B).round(2)}")
    print(f"  |live-train|.max [deg] = {np.degrees(np.abs(diff_A_B)).max():.3f}, mean = {np.degrees(np.abs(diff_A_B)).mean():.3f}")
    if np.degrees(np.abs(diff_A_B)).max() > 5.0:
        print("  >> LIVE cameras produce significantly different action than TRAINING cameras with SAME state.")
        print("  >> Strong evidence the cameras are the OOD input. Check mapping/swap/orientation/lighting.")
    else:
        print("  >> LIVE cams produce similar action to TRAINING cams. Camera mapping is probably fine; OOD is elsewhere.")


# ---------------------------------------------------------------------------
# Part 5: full-chunk trajectory inspection at HIGH vs MID cluster state
# Reveals whether the model PLANS a downward trajectory or commands "hold".
# ---------------------------------------------------------------------------
def run_full_chunk_inspection(policy_dir: Path, live_cam_dir: Path, policy=None) -> None:
    _print_section(
        "PART 5: full-chunk trajectory inspection\n"
        "(at MID and HIGH cluster states, run the full 50-action chunk\n"
        " and print where each chunk position is heading. Reveals whether\n"
        " the model PLANS a downward trajectory or just commands 'hold'.)"
    )
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from lerobot.processor.pipeline import PolicyProcessorPipeline  # noqa: PLC0415
    from lerobot.processor.converters import (  # noqa: PLC0415
        batch_to_transition, policy_action_to_transition,
        transition_to_batch, transition_to_policy_action,
    )
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415
    from lerobot.utils.device_utils import get_safe_torch_device  # noqa: PLC0415

    if policy is None:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415
        policy = PI05Policy.from_pretrained(str(policy_dir))

    device = get_safe_torch_device(policy.config.device)
    policy.to(device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir), config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir), config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    _override_device_processors(preprocessor, postprocessor, device)
    rel_step = None
    for step in preprocessor.steps:
        if type(step).__name__ == "RelativeActionsProcessorStep":
            rel_step = step; break
    if rel_step is not None:
        for step in postprocessor.steps:
            if type(step).__name__ == "AbsoluteActionsProcessorStep":
                step.relative_step = rel_step

    def _load_cam(fname):
        bgr = cv2.imread(str(live_cam_dir / fname))
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    live_imgs = {
        "observation.images.ego": _load_cam("diag_camera_ego_dev_video0.png"),
        "observation.images.left_wrist": _load_cam("diag_camera_left_wrist_dev_video4.png"),
        "observation.images.right_wrist": _load_cam("diag_camera_right_wrist_dev_video2.png"),
    }
    if any(v is None for v in live_imgs.values()):
        print("  !! live camera PNGs missing - re-run with cameras enabled.")
        return

    # The two candidate ready poses, in rad.
    # HIGH = filtered median we currently use in lift_arms.py.
    # MID = unfiltered median over the bigger cluster (104/282 eps).
    HIGH = np.array([+1.3136, -0.1223, -0.2047, +1.3727, -0.0704, -0.1001, -1.2831, 0.0,
                     -1.3083, +0.1223, +0.1997, +1.3727, +0.0708, +0.1001, +1.2690, 0.0], dtype=np.float32)
    MID = np.array([+0.2716, -0.1141, -0.0731, +0.3397, +0.0217, -0.0368, -0.1150, +0.0261,
                    -0.2716, +0.1081, +0.0835, +0.3397, -0.0217, +0.0370, +0.1150, +0.0261], dtype=np.float32)

    joint_names = ["l_j1","l_j2","l_j3","l_j4","l_j5","l_j6","l_j7","l_grp",
                   "r_j1","r_j2","r_j3","r_j4","r_j5","r_j6","r_j7","r_grp"]
    task = "put the chocolate bars in the container"

    def _run_one_chunk(state, label):
        print(f"\n--- {label} ---")
        print(f"  state[deg]: " + "  ".join([f"{n}={np.degrees(v):+6.1f}" for n,v in zip(joint_names, state)]))
        preprocessor.reset(); postprocessor.reset(); policy.reset()
        # Cycle 50 calls. The first triggers fresh inference (chunk anchor latches
        # to current state). Subsequent calls just pop the queue. Crucially, we
        # KEEP the observation constant - in a real run the state would update,
        # but we want to see the model's PLANNED trajectory from this fixed start.
        actions = []
        for i in range(50):
            obs = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in live_imgs.items()}
            obs["observation.state"] = state.copy()
            act = predict_action(
                observation=obs, policy=policy, device=device,
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=policy.config.use_amp, task=task,
                robot_type="bi_openarm_follower",
            ).detach().cpu().numpy().flatten()
            actions.append(act)
        actions = np.array(actions)
        # Show the trajectory at key indices.
        for i in [0, 5, 10, 20, 30, 40, 49]:
            delta = np.degrees(actions[i] - state)
            print(f"  a[{i:2d}]-state [deg]: " + "  ".join([f"{n}={delta[j]:+6.1f}" for j,n in enumerate(joint_names)]))
        # Per-joint range of motion within the chunk.
        chunk_range = np.degrees(actions.max(axis=0) - actions.min(axis=0))
        print(f"  per-joint chunk MOTION RANGE [deg]: "
              + "  ".join([f"{n}={chunk_range[j]:+5.1f}" for j,n in enumerate(joint_names)]))
        print(f"  total |chunk motion| (mean across joints): {chunk_range.mean():.2f} deg")

    _run_one_chunk(HIGH, "HIGH cluster state (= current READY_POSE, 95/282 training eps)")
    _run_one_chunk(MID,  "MID cluster state (j4=20, 104/282 training eps - majority)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--ego-dev", default="/dev/video0")
    p.add_argument("--left-wrist-dev", default="/dev/video4")
    p.add_argument("--right-wrist-dev", default="/dev/video2")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/diag_policy"))
    p.add_argument("--skip-cameras", action="store_true",
                   help="skip the V4L2 camera capture (e.g. cameras are claimed)")
    p.add_argument("--skip-tower-test", action="store_true",
                   help="skip the model-on-training-frame test (e.g. dataset missing)")
    p.add_argument("--skip-cross-domain", action="store_true",
                   help="skip the cross-domain test (training-state + live-cam comparison)")
    p.add_argument("--skip-full-chunk", action="store_true",
                   help="skip the full-chunk trajectory inspection at HIGH vs MID state")
    args = p.parse_args()

    if not args.skip_cameras:
        try:
            run_camera_test(args.out_dir, args.ego_dev, args.left_wrist_dev, args.right_wrist_dev)
        except Exception:
            traceback.print_exc()

    policy = None
    try:
        policy = run_vision_tower_check(args.policy_dir)
    except Exception:
        traceback.print_exc()

    if not args.skip_tower_test:
        try:
            run_tower_test(args.policy_dir, args.dataset_root, policy=policy)
        except Exception:
            traceback.print_exc()

    if not args.skip_cross_domain:
        try:
            run_cross_domain_test(args.policy_dir, args.dataset_root, args.out_dir, policy=policy)
        except Exception:
            traceback.print_exc()

    if not args.skip_full_chunk:
        try:
            run_full_chunk_inspection(args.policy_dir, args.out_dir, policy=policy)
        except Exception:
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
