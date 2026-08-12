#!/usr/bin/env python3
"""Replay recorded cube-pick episodes in the MuJoCo viewer.

Streams the recorded action columns back through the same sim the episodes
were captured in, at the recorded rate. Useful for eyeballing demonstration
quality (smoothness, approach paths, gripper timing).

Note: the cube's initial pose is not stored in the dataset, so a red cube is
placed at its default spot but will not match where the episode's cube was —
judge the ARM motion, not the grasp outcome. The recorded chest-camera video
(what the VLA actually sees, cube included) can be browsed with:
    lerobot-dataset-viz --repo-id local/openarm_new_sim_cube_chest_300 --episode-index 0

Usage:
    MUJOCO_GL=egl python scripts/replay_cube_episode.py \
        --dataset local/openarm_new_sim_cube_chest_300 --episodes 0 1 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import random_cube_pick as rcp  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="local/openarm_new_sim_cube_chest_300")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0],
                        help="episode indices to replay, in order")
    parser.add_argument("--fps", type=int, default=0,
                        help="playback rate; 0 = the dataset's recorded fps")
    parser.add_argument("--model-path",
                        default=str(Path.home() / "sparkpack/openarm_mujoco/v1/scene.xml"))
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless run (validation only)")
    args = parser.parse_args()

    ds = LeRobotDataset(args.dataset)
    fps = args.fps or int(ds.fps)
    action_names = ds.meta.features["action"]["names"]

    robot = rcp.make_robot(args.model_path, fps, viewer=not args.no_viewer, cameras="none")
    iks = {arm.side: rcp.build_ik(robot, arm) for arm in rcp.ARMS}

    for ep in args.episodes:
        start = int(ds.meta.episodes["dataset_from_index"][ep])
        stop = int(ds.meta.episodes["dataset_to_index"][ep])
        print(f"episode {ep}: {stop - start} frames @ {fps} fps")

        rcp.park_both_arms(robot, iks)
        rcp.settle_pose(robot, iks["right"], 0.0, fps, hold_s=0.2)

        # Jump the arms straight to the episode's first recorded pose so the
        # replay starts where the demonstration did (random start poses).
        first = ds[start]["action"].numpy().astype(float)
        robot.send_action({k: float(v) for k, v in zip(action_names, first, strict=True)})
        rcp.settle_pose(robot, iks["right"], 0.0, fps, hold_s=0.3)

        for i in range(start, stop):
            t0 = time.perf_counter()
            row = ds[i]["action"].numpy().astype(float)
            robot.send_action({k: float(v) for k, v in zip(action_names, row, strict=True)})
            precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        time.sleep(0.5)

    robot.disconnect()


if __name__ == "__main__":
    main()
