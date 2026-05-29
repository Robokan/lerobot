"""Test whether our lerobot port predicts the correct action on an ACTIVE
training frame (operator mid-task, not at episode-start hold). If yes, the
runtime is fine and the live failure is a live-env mismatch. If no, the
runtime has a bug that biases everything to 'hold'."""
from pathlib import Path
import numpy as np
import torch

import sys as _sys
_sys.path.insert(0, "/home/evaughan/sparkpack/lerobot")
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import init_logging
from scripts.load_pi05_from_openpi import load_pi05_with_runtime_lora
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device

init_logging()

POLICY_DIR = Path("outputs/pi05_chocolate_v4_from_openpi")
DATASET = Path.home() / ".cache/huggingface/lerobot/local/openarm-chocolate-v4"

print("Loading dataset...")
ds = LeRobotDataset("local/openarm-chocolate-v4", root=DATASET)

# Find training frames at HIGH pose with LARGE action-state delta (i.e., actively moving)
print("Searching for HIGH-pose active frames...")
JOINT_NAMES = ds.meta.features["action"]["names"][0]  # nested
print(f"action names: {JOINT_NAMES}")

# Look at BOTH-arms-HIGH frames (symmetric pose like our READY_POSE_RAD)
active_candidates = []
for i in (42700, 43000, 43800, 44000, 44500, 45000):
    try:
        s = ds[i]
    except Exception:
        continue
    state_deg = np.rad2deg(s["observation.state"].numpy())
    action_deg = np.rad2deg(s["action"].numpy())
    delta = action_deg - state_deg
    L_j1, L_j4 = state_deg[0], state_deg[3]
    R_j1, R_j4 = state_deg[8], state_deg[11]
    active_candidates.append((i, L_j1, L_j4, R_j1, R_j4, np.abs(delta).max(), np.abs(delta).mean()))

print(f"\n{len(active_candidates)} BOTH-HIGH (symmetric) candidate frames:")
for c in active_candidates:
    print(f"  frame {c[0]}: L=({c[1]:+.1f},{c[2]:+.1f}) R=({c[3]:+.1f},{c[4]:+.1f}) action_delta max={c[5]:.2f} mean={c[6]:.2f}")
if not active_candidates:
    import sys; sys.exit(1)

# Pick the first BOTH-HIGH frame
target_idx = active_candidates[0][0]
print(f"\n>>> Testing on frame {target_idx} <<<")
sample = ds[target_idx]
state = sample["observation.state"].numpy().astype(np.float32)
label_action = sample["action"].numpy().astype(np.float32)
task = sample.get("task", "put the chocolate bars in the container")
if not isinstance(task, str):
    task = "put the chocolate bars in the container"
print(f"Task: {task!r}")

# Get images and convert from float[0,1] CHW back to uint8 HWC (live cam format)
images = {}
for cam in ("ego", "left_wrist", "right_wrist"):
    img_t = sample[f"observation.images.{cam}"]
    img_np = img_t.permute(1, 2, 0).numpy() if img_t.shape[0] == 3 else img_t.numpy()
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    images[cam] = img_np
    print(f"  {cam}: shape={img_np.shape} dtype={img_np.dtype} mean={img_np.mean():.1f}")

print(f"\nState (deg):  {np.rad2deg(state).round(2)}")
print(f"Label action (deg): {np.rad2deg(label_action).round(2)}")
print(f"Label - state (deg): {np.rad2deg(label_action - state).round(2)}")

# Build observation dict (live-cam format)
obs = {"observation.state": state}
for cam, img in images.items():
    obs[f"observation.images.{cam}"] = img

# Load policy
print("\nLoading policy with runtime LoRA...")
meta = LeRobotDatasetMetadata("local/openarm-chocolate-v4", root=DATASET)
policy, preprocessor, postprocessor = load_pi05_with_runtime_lora(POLICY_DIR, ds_meta=meta, device="cuda")
policy.reset()
preprocessor.reset(); postprocessor.reset()
device = get_safe_torch_device("cuda")

# Run inference (same path as live)
print("Running inference...")
action_t = predict_action(
    observation=obs.copy(),
    policy=policy,
    device=device,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
    use_amp=policy.config.use_amp,
    task=task,
    robot_type="bi_openarm_follower",
)
pred = action_t.detach().cpu().numpy().flatten()
print(f"\nPredicted action (deg): {np.rad2deg(pred).round(2)}")
print(f"Label action (deg):     {np.rad2deg(label_action).round(2)}")
diff = pred - label_action
print(f"|pred - label| mean deg: {np.rad2deg(np.abs(diff)).mean():.3f}")
print(f"|pred - label| max  deg: {np.rad2deg(np.abs(diff)).max():.3f}")

# Compare action-delta-from-state
pred_motion = np.rad2deg(pred - state)
label_motion = np.rad2deg(label_action - state)
print(f"\nPredicted motion from state (deg): {pred_motion.round(2)}")
print(f"Label motion from state    (deg): {label_motion.round(2)}")
print(f"motion size: pred max={np.abs(pred_motion).max():.2f}  label max={np.abs(label_motion).max():.2f}")
print(f"motion size: pred mean={np.abs(pred_motion).mean():.2f}  label mean={np.abs(label_motion).mean():.2f}")
if np.abs(pred_motion).mean() < 0.5 < np.abs(label_motion).mean():
    print("\nBAD: model is 'holding' (small motion) while label commands real motion.")
    print("    → runtime IS broken for active HIGH-pose frames.")
elif np.rad2deg(np.abs(diff)).mean() < 5.0:
    print("\nGOOD: model matches label closely. Runtime is fine.")
    print("    → live failure is live-env mismatch, not the port.")
else:
    print("\nMARGINAL: model differs from label but both nonzero. Check direction.")
