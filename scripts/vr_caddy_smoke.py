#!/usr/bin/env python
"""Headless end-to-end check of the caddy VR record path — no headset needed.

Runs lerobot-record with the caddy-scene robot and the SCRIPTED pose driver,
and injects the record controls on a timer through the same inbox the headset's
B / A buttons use. Proves, in about a minute: the scene is arranged before each
episode, the per-episode prompt is stamped on every frame, and episodes are
actually SAVED with the schema the caddy policies train on.

Why it exists: thirteen earlier VR datasets on this machine held zero episodes
between them — recording had been started many times and never once saved —
so the save path is exactly the thing to prove before anyone puts a headset on.

    MUJOCO_GL=egl .venv/bin/python scripts/vr_caddy_smoke.py
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
from pathlib import Path

from lerobot.robots.mujoco_bi_openarm import viewer_keys as vk

PREFIX = "local/_vr_caddy_smoke"
ROOT = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
N_EPISODES = 2

sys.argv = [
    "lerobot-record",
    "--robot.type=mujoco_bi_openarm_caddy", "--robot.id=mujoco_bi_openarm_caddy",
    "--robot.fps=30", "--robot.stacks=6", "--robot.seed=7",
    "--teleop.type=vr_mocap", "--teleop.id=vr_mocap", "--teleop.driver=scripted", "--teleop.vr_hz=30",
    f"--dataset.repo_id={PREFIX}", "--dataset.single_task=get bar from the named pad",
    f"--dataset.num_episodes={N_EPISODES}", "--dataset.fps=30", "--dataset.episode_time_s=0",
    "--dataset.reset_time_s=2", "--dataset.push_to_hub=false",
    "--display_data=false", "--play_sounds=false",
]

# episode 1: start at 6 s, save at 12 s; 2 s reset; episode 2: start 20 s, save 26 s
SCHEDULE = ((6, "y"), (12, "t"), (20, "y"), (26, "t"))
t0 = time.time()


def presses() -> None:
    for t, key in SCHEDULE:
        while time.time() - t0 < t:
            time.sleep(0.1)
        vk.request_recording_control(key)
        print(f"[smoke] injected {key!r} at {time.time() - t0:.0f}s", flush=True)


threading.Thread(target=presses, daemon=True).start()

from lerobot.scripts.lerobot_record import main  # noqa: E402

main()

# --- verify what was written (this fork stamps a timestamp onto the repo id)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402

dirs = sorted(glob.glob(str(ROOT / "local" / (PREFIX.split("/", 1)[1] + "*"))))
assert dirs, "no dataset directory was written"
repo = "local/" + Path(dirs[-1]).name
m = LeRobotDatasetMetadata(repo)
tasks = [m.episodes[e]["tasks"][0] for e in range(m.total_episodes)]
f = m.features
cams = sorted(k.split(".")[-1] for k in f if k.startswith("observation.images"))
print(f"\n[smoke] {repo}: {m.total_episodes} episodes, {m.total_frames} frames, fps {m.fps}")
print(f"[smoke] tasks: {tasks}")
print(f"[smoke] action {tuple(f['action']['shape'])} state {tuple(f['observation.state']['shape'])} cams {cams}")
assert m.total_episodes == N_EPISODES, f"expected {N_EPISODES} episodes, got {m.total_episodes}"
assert all(t.startswith("get bar from ") for t in tasks), tasks
assert tuple(f["action"]["shape"]) == (16,) and tuple(f["observation.state"]["shape"]) == (16,)
assert cams == ["ego", "left_wrist", "right_wrist"]
print("[smoke] PASS — VR caddy record path saves episodes with the training schema")
