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
import os
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


class _SmoothedPost:
    """Postprocessor wrapper: zero-phase 3-tap smoothing of chunk rows.

    Applied to the decoded absolute chunk (B, T, A) — including inside the RTC
    engine, which receives this wrapper as its postprocessor. Gripper columns
    pass through untouched (binary open/close must not be diluted). Endpoint
    rows are kept so chunk boundaries stay anchored. Delegates everything else
    (e.g. ``.steps`` introspection) to the wrapped pipeline.
    """

    def __init__(self, inner, action_keys: list[str]):
        self._inner = inner
        self._joint_idx = [i for i, k in enumerate(action_keys) if "gripper" not in k]

    def __call__(self, actions):
        out = self._inner(actions)
        if out.ndim == 3 and out.shape[1] >= 3:
            sm = out.clone()
            sm[:, 1:-1] = 0.25 * out[:, :-2] + 0.5 * out[:, 1:-1] + 0.25 * out[:, 2:]
            out = out.clone()
            out[:, :, self._joint_idx] = sm[:, :, self._joint_idx]
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


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
        self._trt_sock = None

    def connect_trt(self, socket_path: str) -> None:
        """Route the model call to a TRT inference server (lerobot pre/post
        processing stays local and bit-identical to training)."""
        import socket as _socket

        self._trt_sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self._trt_sock.connect(socket_path)
        print(f"  policy model calls -> TRT server at {socket_path}")

    def _remote_chunk(self, preprocessed: dict):
        import pickle
        import socket as _socket
        import struct

        msg = {}
        for k, v in preprocessed.items():
            if hasattr(v, "detach"):
                t = v.detach().to("cpu")
                if t.dtype == self.torch.bfloat16:
                    t = t.float()
                msg[k] = t.numpy()
        data = pickle.dumps(msg, protocol=4)
        self._trt_sock.sendall(struct.pack(">I", len(data)) + data)
        hdr = self._trt_sock.recv(4, _socket.MSG_WAITALL)
        (n,) = struct.unpack(">I", hdr)
        reply = pickle.loads(self._trt_sock.recv(n, _socket.MSG_WAITALL))
        if "error" in reply:
            raise RuntimeError(f"TRT server error: {reply['error']}")
        return self.torch.from_numpy(reply["action"]).to(self.device)

    def configure_for_robot(self, robot) -> None:
        from lerobot.utils.feature_utils import hw_to_dataset_features

        self.action_keys = list(robot.action_features.keys())
        self.robot_type = robot.name
        self.obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)

        # More Euler steps = better-integrated flow = less within-chunk wiggle.
        # Measured on the sim-cube checkpoint: 4 -> 16 steps cuts intra-chunk
        # direction reversals from ~50% to ~37% of steps. Eager path only; the
        # TRT server has its own GROOT_TRT_DENOISE_STEPS knob.
        steps = int(os.environ.get("GROOT_DENOISE_STEPS", "0"))
        if steps > 0:
            head = getattr(getattr(self.policy, "_groot_model", None), "action_head", None)
            if head is not None:
                head.num_inference_timesteps = steps
                print(f"  denoise steps -> {steps}")

        # Zero-phase smoothing of each predicted chunk (arm joints only): the
        # policy's raw chunks reverse direction on ~half their steps while the
        # training data reverses on ~4%; a centered 3-tap filter on the PLAN
        # adds no feedback lag (unlike filtering executed commands, which
        # costs enough tracking accuracy to break the grasp).
        if os.environ.get("GROOT_SMOOTH_CHUNK") == "1":
            self.post = _SmoothedPost(self.post, self.action_keys)
            print("  chunk smoothing ENABLED (centered 3-tap, arms only)")

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
                if self._trt_sock is not None:
                    gi = self.policy._filter_groot_inputs(preprocessed, include_action=False)
                    actions = self._remote_chunk(gi)
                else:
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


def build_rtc_engine(pol: CheckpointPolicy, robot, fps: int, horizon: int, task: str):
    """Construct lerobot's real RTC engine around our policy + processors."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference.factory import RTCInferenceConfig, create_inference_engine
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

    dataset_features = combine_feature_dicts(
        hw_to_dataset_features(robot.observation_features, "observation", True),
        hw_to_dataset_features(robot.action_features, "action", True),
    )
    engine = create_inference_engine(
        RTCInferenceConfig(rtc=RTCConfig(execution_horizon=horizon)),
        policy=pol.policy,
        preprocessor=pol.pre,
        postprocessor=pol.post,
        robot_wrapper=ThreadSafeRobot(robot),
        hw_features=pol.obs_features,
        dataset_features=dataset_features,
        ordered_action_keys=list(robot.action_features.keys()),
        task=task,
        fps=float(fps),
        device="cuda",
    )
    engine.start()
    return engine


def run_rtc_trial(robot, engine, pol, fps: int, time_limit_s: float = 30.0):
    """One episode under async RTC: continuous 30 Hz control, background chunks."""
    import time as _time

    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.robot_utils import precise_sleep

    import numpy as _np

    engine.reset()
    engine.resume()  # the RTC background thread starts paused
    t_end = _time.perf_counter() + time_limit_s
    held_since = None
    cmd = None  # slew-limited command state
    slew = 2.5  # deg per tick — spreads chunk-seam jumps (measured up to 28 deg)
    while _time.perf_counter() < t_end:
        t0 = _time.perf_counter()
        obs = robot.get_observation()
        engine.notify_observation(obs)
        frame = build_dataset_frame(pol.obs_features, obs, prefix="observation")
        a = engine.get_action(frame)
        if a is not None:
            target = a.detach().cpu().numpy().reshape(-1)
            if cmd is None:
                cmd = target.copy()
            else:
                cmd = cmd + _np.clip(target - cmd, -slew, slew)
            robot.send_action({k: float(v) for k, v in zip(pol.action_keys, cmd, strict=True)})
        elif cmd is not None:
            # queue priming/gap: hold the last command so the sim keeps stepping
            robot.send_action({k: float(v) for k, v in zip(pol.action_keys, cmd, strict=True)})
        z = float(rcp.cube_pos(robot)[2])
        if z >= rcp.SUCCESS_CUBE_Z:
            held_since = held_since or _time.perf_counter()
            if _time.perf_counter() - held_since > 0.5:
                return True, _time.perf_counter() - (t_end - time_limit_s)
        else:
            held_since = None
        precise_sleep(max(1.0 / fps - (_time.perf_counter() - t0), 0.0))
    return False, time_limit_s


def run_jit_trial(robot, pol, fps: int, time_limit_s: float = 30.0, slew: float = 2.5):
    """Just-in-time sequential chunking: execute each chunk to completion and
    compute the next one in a background thread during the current chunk's
    tail. New chunks start from (nearly) the state the old chunk actually
    reached, so seams are policy-consistent — the pattern NVIDIA's GR00T
    demos use, feasible here because TRT inference (~0.2 s) fits inside the
    chunk tail (~0.3 s)."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from lerobot.utils.robot_utils import precise_sleep

    pol.reset()
    pool = ThreadPoolExecutor(max_workers=1)
    prefetch_at = max(2, int(0.25 * fps) + 2)  # ticks-left threshold to prefetch

    # direct chunk computation without the act() side effects
    def chunk_from(obs) -> list:
        pol._queue = []
        first = pol.act(dict(obs))
        rows = [np.array([first[k] for k in pol.action_keys])]
        rows += [r.copy() for r in pol._queue]
        pol._queue = []
        return rows

    t_end = _time.perf_counter() + time_limit_s
    held_since = None
    queue: list = []
    future = None
    cmd = None
    while _time.perf_counter() < t_end:
        t0 = _time.perf_counter()
        obs = robot.get_observation()
        if len(queue) == prefetch_at and future is None:
            future = pool.submit(chunk_from, dict(obs))
        if not queue:
            if future is not None:
                queue = future.result()
                future = None
            else:
                queue = chunk_from(obs)  # first chunk of the episode (blocking)
        target = queue.pop(0)
        cmd = target.copy() if cmd is None else cmd + np.clip(target - cmd, -slew, slew)
        robot.send_action({k: float(v) for k, v in zip(pol.action_keys, cmd, strict=True)})
        z = float(rcp.cube_pos(robot)[2])
        if z >= rcp.SUCCESS_CUBE_Z:
            held_since = held_since or _time.perf_counter()
            if _time.perf_counter() - held_since > 0.5:
                pool.shutdown(wait=False)
                return True, _time.perf_counter() - (t_end - time_limit_s)
        else:
            held_since = None
        precise_sleep(max(1.0 / fps - (_time.perf_counter() - t0), 0.0))
    pool.shutdown(wait=False)
    return False, time_limit_s


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
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Drive trials through lerobot's async RTCInferenceEngine (background "
             "inference, continuous motion) instead of blocking sync chunks.",
    )
    parser.add_argument("--rtc-horizon", type=int, default=8)
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Just-in-time sequential chunking: full-chunk execution with the "
             "next chunk computed in the background during the tail. Smoothest "
             "async mode; needs TRT-fast inference.",
    )
    parser.add_argument(
        "--trt-socket",
        default=None,
        help="Unix socket of a sim_cube_trt_server; model calls go to TRT engines.",
    )
    parser.add_argument(
        "--smooth-chunk",
        action="store_true",
        help="zero-phase 3-tap smoothing of each predicted chunk (arm joints only); "
             "kills within-chunk dither without feedback lag",
    )
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=0,
        help="override flow-matching Euler steps (eager path; use GROOT_TRT_DENOISE_STEPS "
             "on the TRT server). 16 markedly reduces chunk wiggle vs the default 4.",
    )
    args = parser.parse_args()
    if args.smooth_chunk:
        os.environ["GROOT_SMOOTH_CHUNK"] = "1"
    if args.denoise_steps:
        os.environ["GROOT_DENOISE_STEPS"] = str(args.denoise_steps)

    robot = rcp.make_robot(args.model_path, args.fps, viewer=not args.no_viewer, cameras=args.cameras)
    iks = {a.side: rcp.build_ik(robot, a) for a in rcp.ARMS}
    rcp.park_both_arms(robot, iks)
    rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)

    if args.policy == "zeros":
        policy = ZerosPolicy(robot)
    else:
        policy = CheckpointPolicy(args.policy, args.dataset, args.task)
        policy.configure_for_robot(robot)
        if args.trt_socket:
            policy.connect_trt(args.trt_socket)

    rng = np.random.default_rng(args.seed)
    engine = None
    if args.rtc:
        engine = build_rtc_engine(policy, robot, args.fps, args.rtc_horizon, args.task)

    results = []
    try:
        for t in range(args.trials):
            if args.jit:
                cube0, arm = rcp.place_reachable_cube(robot, iks, rng)
                rcp.setup_start_pose(robot, iks[arm.side], rng, args.fps)
                ok, t_used = run_jit_trial(robot, policy, args.fps, args.time_limit)
                cz = float(rcp.cube_pos(robot)[2])
                r = {"success": ok, "t_success": t_used if ok else None,
                     "cube_y": float(cube0[1]), "intended_arm": arm.side,
                     "committed_arm": arm.side, "final_cube_z": cz}
                print(f"trial {t + 1:>3}/{args.trials}: "
                      f"{'SUCCESS' if ok else 'fail   '} cube_y={cube0[1]:+.2f} "
                      f"jit t={t_used:.1f}s cube_z={cz:.3f}")
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks[arm.side], 0.0, args.fps, hold_s=0.15)
            elif engine is not None:
                cube0, arm = rcp.place_reachable_cube(robot, iks, rng)
                rcp.setup_start_pose(robot, iks[arm.side], rng, args.fps)
                policy.reset()
                ok, t_used = run_rtc_trial(robot, engine, policy, args.fps, args.time_limit)
                cz = float(rcp.cube_pos(robot)[2])
                r = {"success": ok, "t_success": t_used if ok else None,
                     "cube_y": float(cube0[1]), "intended_arm": arm.side,
                     "committed_arm": arm.side, "final_cube_z": cz}
                print(f"trial {t + 1:>3}/{args.trials}: "
                      f"{'SUCCESS' if ok else 'fail   '} cube_y={cube0[1]:+.2f} "
                      f"rtc t={t_used:.1f}s cube_z={cz:.3f}")
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks[arm.side], 0.0, args.fps, hold_s=0.15)
            else:
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
        if engine is not None:
            engine.stop()
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
