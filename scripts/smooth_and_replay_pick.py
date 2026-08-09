#!/usr/bin/env python3
"""Clean a recorded OpenArm pick by removing pauses — keep the real path.

Straight joint-space shortcuts hit the table; this keeps the recorded geometry,
strips long stops, and reseeds motion at a steadier speed along that same path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ACTION_NAMES = [
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_joint_7.pos",
    "right_gripper.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_joint_7.pos",
    "left_gripper.pos",
]


def load_actions(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path)
    return np.stack(table.column("action").to_pylist()).astype(np.float64)


def trim_idle(action: np.ndarray, vel_thr: float = 0.05) -> np.ndarray:
    """Drop leading/trailing stillness (right arm + gripper)."""
    if len(action) < 3:
        return action
    vel = np.linalg.norm(np.diff(action[:, :8], axis=0), axis=1)
    moving = np.where(vel >= vel_thr)[0]
    if len(moving) == 0:
        return action
    start = int(moving[0])
    end = int(moving[-1]) + 1  # inclusive of last moving step's end frame
    return action[start : end + 1]


def compress_stops(action: np.ndarray, vel_thr: float = 0.06, max_still: int = 3) -> np.ndarray:
    """Drop long near-zero-velocity runs; keep the moving geometry intact."""
    if len(action) < 2:
        return action
    vel = np.linalg.norm(np.diff(action, axis=0), axis=1)
    keep = [0]
    still = 0
    for i, v in enumerate(vel, start=1):
        if v < vel_thr:
            still += 1
            if still <= max_still:
                keep.append(i)
        else:
            still = 0
            keep.append(i)
    return action[np.asarray(keep)]


def resample_constant_speed(action: np.ndarray, fps: int, speed_scale: float = 1.35) -> np.ndarray:
    """Resample along arc-length so motion is steadier — same path, less dawdling.

    ``speed_scale`` > 1 shortens the episode without cutting corners in joint space.
    """
    if len(action) < 2:
        return action.astype(np.float32)

    # Weight joints more than gripper so grasp timing still follows path fraction,
    # but bulk timing is driven by arm motion.
    weights = np.ones(action.shape[1], dtype=np.float64)
    weights[7] = 0.15
    weights[15] = 0.15
    delta = np.diff(action, axis=0) * weights
    seg = np.linalg.norm(delta, axis=1)
    # Tiny epsilon so truly-still samples don't collapse the parameterization oddly.
    seg = np.maximum(seg, 1e-6)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return action.astype(np.float32)

    # Original average speed in weighted units/frame; speed up a bit.
    avg = total / max(len(action) - 1, 1)
    step = avg * speed_scale
    n_out = max(int(np.ceil(total / step)) + 1, 2)
    samples = np.linspace(0.0, total, n_out)
    out = np.empty((n_out, action.shape[1]), dtype=np.float64)
    for j in range(action.shape[1]):
        out[:, j] = np.interp(samples, cum, action[:, j])
    return out.astype(np.float32)


def light_smooth(action: np.ndarray, window: int = 5) -> np.ndarray:
    """Small moving average on arm joints only (path-safe). Gripper unchanged."""
    if window < 3 or len(action) < window:
        return action
    out = action.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    for j in list(range(7)) + list(range(8, 15)):
        pad = window // 2
        x = np.pad(out[:, j], (pad, pad), mode="edge")
        out[:, j] = np.convolve(x, kernel, mode="valid")
    return out


def clean_trajectory(raw: np.ndarray, fps: int) -> np.ndarray:
    trimmed = trim_idle(raw)
    compressed = compress_stops(trimmed)
    resampled = resample_constant_speed(compressed, fps=fps, speed_scale=1.4)
    return light_smooth(resampled, window=5)


def replay(actions: np.ndarray, fps: int, model_path: str) -> None:
    from lerobot.robots.mujoco_bi_openarm import MujocoBiOpenArm, MujocoBiOpenArmConfig
    from lerobot.utils.robot_utils import precise_sleep

    robot = MujocoBiOpenArm(
        MujocoBiOpenArmConfig(
            viewer=True,
            cameras={},
            model_path=model_path,
            fps=fps,
        )
    )
    robot.connect(calibrate=False)
    print(f"Replaying {len(actions)} frames ({len(actions) / fps:.1f}s) — close viewer or Ctrl-C to stop")
    try:
        first = {n: float(actions[0, i]) for i, n in enumerate(ACTION_NAMES)}
        for _ in range(fps // 2):
            t0 = time.perf_counter()
            robot.send_action(first)
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))

        for row in actions:
            t0 = time.perf_counter()
            action = {n: float(row[i]) for i, n in enumerate(ACTION_NAMES)}
            robot.send_action(action)
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))

        last = {n: float(actions[-1, i]) for i, n in enumerate(ACTION_NAMES)}
        for _ in range(fps):
            t0 = time.perf_counter()
            robot.send_action(last)
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/lerobot/local/openarm-sim-vr_20260807_163416/data/chunk-000/file-000.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/openarm-sim-vr_pick_clean.npy",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--model-path",
        default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"),
    )
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()

    raw = load_actions(args.parquet)
    clean = clean_trajectory(raw, fps=args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, clean)
    print(
        f"Raw {len(raw)} frames ({len(raw) / args.fps:.1f}s) → "
        f"clean {len(clean)} frames ({len(clean) / args.fps:.1f}s) "
        f"[same path, pauses removed]\nSaved {args.out}"
    )
    if not args.no_replay:
        replay(clean, fps=args.fps, model_path=args.model_path)


if __name__ == "__main__":
    main()
