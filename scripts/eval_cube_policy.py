#!/usr/bin/env python3
"""Closed-loop evaluation of a trained policy on the randomized cube-pick task.

Runs a checkpoint in the same MuJoCo scene the training episodes came from:
same cube randomization, same dual-arm random starts — but the POLICY drives.
Scores pick success and breaks results down by cube side and by which arm the
policy actually committed, which directly exposes arm-selection confusion.

Usage:
  MUJOCO_GL=egl python scripts/eval_cube_policy.py \
    --policy outputs/groot_sim_cube/checkpoints/005000/pretrained_model \
    --dataset local/openarm_sim_cube_chest_100 \
    --cameras chest --trials 30 --seed 100 --no-viewer

  --policy zeros   runs a hold-still stub policy (harness self-test, no GPU).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import random_cube_pick as rcp  # noqa: E402  (scene + randomization helpers)

from lerobot.utils.robot_utils import precise_sleep  # noqa: E402


class ZerosPolicy:
    """Hold-current-pose stub used to self-test the harness without a checkpoint."""

    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        pass

    def act(self, obs: dict) -> dict:
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


class CheckpointPolicy:
    """A lerobot pretrained policy + its processor pipelines, driven per tick."""

    def __init__(self, path: str, dataset_repo_id: str, task: str, device: str = "cuda"):
        import torch

        from lerobot.common.control_utils import predict_action
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self._predict_action = predict_action
        self.torch = torch
        self.device = torch.device(device)
        self.task = task

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(self.device).eval()

        stats = LeRobotDatasetMetadata(dataset_repo_id).stats
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=path,
            dataset_stats=stats,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.action_keys: list[str] | None = None
        self.robot_type: str | None = None
        self.obs_features: dict | None = None

    def configure_for_robot(self, robot) -> None:
        from lerobot.utils.feature_utils import hw_to_dataset_features

        self.action_keys = list(robot.action_features.keys())
        self.robot_type = robot.name
        self.obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)

    def reset(self):
        self.policy.reset()
        self._queue: list[np.ndarray] = []

    def act(self, obs: dict) -> dict:
        # Chunked inference: relative-action policies (GR00T) must decode a
        # whole chunk against the observation it was generated from, so we
        # preprocess once, predict a chunk, postprocess the full chunk to
        # absolute actions, then feed them out one per tick.
        if not self._queue:
            from lerobot.policies.utils import prepare_observation_for_inference
            from lerobot.utils.feature_utils import build_dataset_frame

            frame = build_dataset_frame(self.obs_features, obs, prefix="observation")
            with self.torch.inference_mode():
                prepared = prepare_observation_for_inference(
                    frame, self.device, self.task, self.robot_type
                )
                preprocessed = self.pre(prepared)
                actions = self.policy.predict_action_chunk(preprocessed)
                processed = self.post(actions)
            chunk = processed.squeeze(0).cpu().numpy()
            n = getattr(self.policy.config, "n_action_steps", chunk.shape[0])
            self._queue = [chunk[i] for i in range(min(n, chunk.shape[0]))]
        vals = self._queue.pop(0).reshape(-1)
        return {k: float(v) for k, v in zip(self.action_keys, vals, strict=True)}


def run_trial(robot, iks, rng, policy, fps: int, time_limit_s: float) -> dict:
    cube0, arm = rcp.place_reachable_cube(robot, iks, rng)
    ik = iks[arm.side]
    rcp.setup_start_pose(robot, ik, rng, fps)
    policy.reset()

    q_start = {s: rcp._arm_q_real(robot, iks[s]) for s in ("left", "right")}
    travel = {"left": 0.0, "right": 0.0}
    prev_q = dict(q_start)

    n_max = int(time_limit_s * fps)
    held = 0
    success = False
    t_success = None
    for k in range(n_max):
        t0 = time.perf_counter()
        obs = robot.get_observation()
        action = policy.act(obs)
        robot.send_action(action)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

        for s in ("left", "right"):
            q = rcp._arm_q_real(robot, iks[s])
            travel[s] += float(np.abs(q - prev_q[s]).sum())
            prev_q[s] = q

        cz = float(rcp.cube_pos(robot)[2])
        if cz >= rcp.SUCCESS_CUBE_Z:
            held += 1
            if held >= fps // 2:  # held up for 0.5 s
                success = True
                t_success = (k + 1) / fps
                break
        else:
            held = 0
        if cz < 0.2:  # knocked off the table
            break

    committed = max(travel, key=travel.get) if max(travel.values()) > 0.5 else "none"
    return {
        "success": success,
        "t_success": t_success,
        "cube_y": float(cube0[1]),
        "intended_arm": arm.side,
        "committed_arm": committed,
        "final_cube_z": float(rcp.cube_pos(robot)[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="checkpoint dir, or 'zeros' for self-test")
    parser.add_argument("--dataset", default="local/openarm_sim_cube_chest_100",
                        help="training dataset (for normalization stats)")
    parser.add_argument("--cameras", choices=["chest", "all"], default="chest",
                        help="must match what the policy was trained on")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100,
                        help="use a seed NOT used for training data")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--time-limit", type=float, default=25.0)
    parser.add_argument("--task", default="pick up the red cube and lift it")
    parser.add_argument("--model-path", default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"))
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    robot = rcp.make_robot(args.model_path, args.fps, viewer=not args.no_viewer, cameras=args.cameras)
    iks = {a.side: rcp.build_ik(robot, a) for a in rcp.ARMS}
    rcp.park_both_arms(robot, iks)
    rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)

    if args.policy == "zeros":
        policy = ZerosPolicy(robot)
    else:
        policy = CheckpointPolicy(args.policy, args.dataset, args.task)
        policy.configure_for_robot(robot)

    rng = np.random.default_rng(args.seed)
    results = []
    try:
        for t in range(args.trials):
            r = run_trial(robot, iks, rng, policy, args.fps, args.time_limit)
            results.append(r)
            print(
                f"trial {t + 1:>3}/{args.trials}: {'SUCCESS' if r['success'] else 'fail   '} "
                f"cube_y={r['cube_y']:+.2f} intended={r['intended_arm']:5s} "
                f"committed={r['committed_arm']:5s} "
                f"{'t=%.1fs' % r['t_success'] if r['t_success'] else 'cube_z=%.3f' % r['final_cube_z']}"
            )
            rcp.park_both_arms(robot, iks)
            rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.15)
    finally:
        n = len(results)
        if n:
            ok = [r for r in results if r["success"]]
            print(f"\n==== {args.policy} ====")
            print(f"success: {len(ok)}/{n} ({100.0 * len(ok) / n:.0f}%)")
            for side, pred in (("left", lambda r: r["cube_y"] > 0), ("right", lambda r: r["cube_y"] <= 0)):
                grp = [r for r in results if pred(r)]
                if grp:
                    g_ok = sum(r["success"] for r in grp)
                    print(f"  cube on {side:5s} side: {g_ok}/{len(grp)}")
            wrong = sum(
                1 for r in results if r["committed_arm"] not in (r["intended_arm"], "none")
            )
            print(f"  arm-selection mismatches (committed != scripted choice): {wrong}/{n}")
            if ok:
                print(f"  mean time to success: {np.mean([r['t_success'] for r in ok]):.1f}s")
        robot.disconnect()


if __name__ == "__main__":
    main()
