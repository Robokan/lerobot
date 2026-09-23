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
        from lerobot.policies import get_policy_class, make_policy, make_pre_post_processors

        self._predict_action = predict_action
        self.torch = torch
        self.device = torch.device(device)
        self.task = task

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        meta = LeRobotDatasetMetadata(dataset_repo_id)
        # A LoRA checkpoint contains only adapter_model.safetensors; the base weights
        # live wherever adapter_config.json points. A bare from_pretrained looks for
        # model.safetensors and fails outright, so route adapters through make_policy,
        # which reads the adapter config, loads the base policy and applies the
        # adapter on top. Full fine-tunes keep the direct path they always had.
        if getattr(cfg, "use_peft", False) or (Path(path) / "adapter_config.json").is_file():
            cfg.use_peft = True
            self.policy = make_policy(cfg, ds_meta=meta)
        else:
            self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(self.device).eval()

        stats = meta.stats
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
        self.replan_every: int | None = None

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
            # Executing a whole chunk means committing n/fps seconds of motion
            # open-loop. Over the last centimetres of an approach that is long
            # enough for a small initial error to survive to the grasp, and no
            # camera is consulted in between. Keeping fewer rows re-plans against
            # a fresh observation; the discarded tail costs nothing, since every
            # chunk is decoded relative to the observation that produced it.
            if self.replan_every:
                n = min(n, self.replan_every)
            self._queue = [chunk[i] for i in range(min(n, chunk.shape[0]))]
        vals = self._queue.pop(0).reshape(-1)
        return {k: float(v) for k, v in zip(self.action_keys, vals, strict=True)}


# Set from --no-reset-key. With the viewer open, something was draining an "r"
# on the first tick of every trial: each one was abandoned before the arm moved,
# nothing was ever scored, and the loop ran forever while reporting "R pressed".
# Watching the policy run matters more than the abort shortcut, so this turns
# the shortcut off without turning the viewer off.
_RESET_KEY_ENABLED = {"on": True}


def reset_requested() -> bool:
    """True if R was pressed in the MuJoCo viewer window since the last check:
    abandon the current trial and set up a fresh cube and arm poses."""
    if not _RESET_KEY_ENABLED["on"]:
        return False
    try:
        from lerobot.robots.mujoco_bi_openarm.viewer_keys import drain_keys
    except Exception:  # noqa: BLE001
        return False
    keys = drain_keys()
    # With the viewer open this fired on the first tick of every trial, aborting
    # each one before the arm moved, while the caller reported it as "R pressed".
    # Twenty trials became an endless loop that scored nothing and looked exactly
    # like a policy that refuses to act. Say what was actually drained.
    if keys:
        print(f"  [viewer keys drained: {keys}]", flush=True)
    return "r" in keys


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
    held_demo = 0
    success = False
    lifted_demo = False  # the demonstrations' own success bar (aim generator): cube up >= 80% of its height
    t_success = None
    demo_z = rcp.CUBE_Z + 0.8 * rcp.AIM_LIFT_M
    reset = False
    for k in range(n_max):
        t0 = time.perf_counter()
        if reset_requested():
            reset = True
            break
        obs = robot.get_observation()
        action = policy.act(obs)
        robot.send_action(action)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

        for s in ("left", "right"):
            q = rcp._arm_q_real(robot, iks[s])
            travel[s] += float(np.abs(q - prev_q[s]).sum())
            prev_q[s] = q

        cz = float(rcp.cube_pos(robot)[2])
        if cz >= demo_z:
            held_demo += 1
            if held_demo >= fps // 2:
                lifted_demo = True
        else:
            held_demo = 0
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
        "reset": reset,
        "reset_reason": "R pressed in the viewer",
        "success": success,
        "lifted_demo": lifted_demo or success,
        "t_success": t_success,
        "cube_y": float(cube0[1]),
        "intended_arm": arm.side,
        "committed_arm": committed,
        "final_cube_z": float(rcp.cube_pos(robot)[2]),
    }


def run_color_trial(robot, iks, rng, policy, fps: int, time_limit_s: float) -> dict:
    """Colour-sorting task (random_color_pick): red cube -> left arm -> red pad,
    green -> right -> green pad. Success = cube resting on the matching pad for
    0.5 s. Same scene setup as the generator, policy drives."""
    import random_color_pick as rcol

    rcol.show_pads(robot)
    colour = "red" if rng.uniform() < 0.5 else "green"
    rgba, side, pad_xy = rcol.COLOURS[colour]
    other_pad = rcol.COLOURS["green" if colour == "red" else "red"][2]
    rcol.set_cube_colour(robot, rgba)
    cube0 = rcol.place_cube_for(robot, iks, rng, colour)
    rcp.setup_start_pose(robot, iks[side], rng, fps)
    policy.reset()

    q_start = {s_: rcp._arm_q_real(robot, iks[s_]) for s_ in ("left", "right")}
    travel = {"left": 0.0, "right": 0.0}
    prev_q = dict(q_start)
    held = 0
    success = False
    t_success = None
    reset = False
    for k in range(int(time_limit_s * fps)):
        t0 = time.perf_counter()
        if reset_requested():
            reset = True
            break
        obs = robot.get_observation()
        robot.send_action(policy.act(obs))
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        for s_ in ("left", "right"):
            q = rcp._arm_q_real(robot, iks[s_])
            travel[s_] += float(np.abs(q - prev_q[s_]).sum())
            prev_q[s_] = q
        ok, _ = rcol.on_pad(robot, pad_xy)
        if ok:
            held += 1
            if held >= fps // 2:
                success = True
                t_success = (k + 1) / fps
                break
        else:
            held = 0
        if float(rcp.cube_pos(robot)[2]) < 0.2:
            break
    committed = max(travel, key=travel.get) if max(travel.values()) > 0.5 else "none"
    wrong_pad, _ = rcol.on_pad(robot, other_pad)
    return {
        "reset": reset,
        "reset_reason": "R pressed in the viewer",
        "success": success,
        "lifted_demo": success,
        "t_success": t_success,
        "cube_y": float(cube0[1]),
        "colour": colour,
        "intended_arm": side,
        "committed_arm": committed,
        "wrong_pad": bool(wrong_pad),
        "final_cube_z": float(rcp.cube_pos(robot)[2]),
    }


def run_caddy_trial(robot, iks, rng, policy, fps: int, time_limit_s: float,
                    stacks: int = 6) -> dict:
    """Caddy picking (random_caddy_pick): stacks of identical brown bars on
    coloured pads, prompt names one pad, the bar goes on the pile at the centre.

    Success = the bar from the NAMED pad ends up on the pile and nothing else
    moved. Two ways to fail that a cube task does not have: the policy can take
    a bar from the wrong pad (the colour grounding failed) and it can knock a
    neighbouring stack over. Both are reported separately, because "took the
    right bar but fumbled it" and "took the wrong bar" need different fixes.
    """
    import random_caddy_pick as rc

    rc.QUIET = True   # the eval prints the prompt and the result, nothing else

    # The generator overrides these module globals in its main() before it
    # solves any pose. The eval has to do the same, or the scene it builds is
    # not the scene the policy was trained on: the idle arm would tuck to the
    # cube picker's deep park behind the table instead of the caddy task's rest
    # over the near edge, which changes every camera's view of the other arm.
    if rcp.TUCK_TIP_TARGET is not rc.TUCK_TIP_TARGET:
        rcp.TUCK_TIP_TARGET = rc.TUCK_TIP_TARGET
        rcp._TUCK_Q_CACHE.clear()          # it caches per side; the old target is in there
        rcp._APPROACH_GRIP_FN["fn"] = rc.approach_grip
        rcp._APPROACH_SETTLE_S["s"] = rc.APPROACH_SETTLE_S

    rc.hide_legacy_pads(robot)
    rcp.set_cube_xy(robot, -0.90, -0.90)
    trial = None
    for _ in range(8):
        cand = rc.Trial(robot, rng, stacks)
        if rc.grasp_plannable(robot, iks[cand.side], cand, rng):
            trial = cand
            break
    if trial is None:
        return {"reset": True, "success": None,
                "reset_reason": "no plannable grasp in 8 draws"}
    if not rc.safe_start_pose(robot, iks, trial, rng, fps):
        return {"reset": True, "success": None,
                "reset_reason": "no safe start pose in 12 draws"}
    for a in rcp.ARMS:
        rcp._RETREAT_TARGET[a.side] = rcp.tuck_q(iks[a.side])

    # The prompt is per-trial here, unlike the colour task's fixed string.
    policy.task = trial.prompt
    print(f"  {trial.prompt}", flush=True)
    policy.reset()

    # Drain the key queue HERE, not in the caller. The caller clears it before
    # setup, and setup then runs ~0.2 s of physics with viewer syncs — so any
    # key event delivered during setup (a release, an auto-repeat, or a press
    # made while the previous trial was still on screen) survives into tick 1
    # and aborts a trial nobody asked to abort. Clearing after setup closes that
    # window, so only a press made DURING the trial counts.
    reset_requested()

    target = trial.top_bar()
    watch = trial.arranged_bars()
    start = {i: rc.bar_pos(robot, i).copy() for i in watch}
    travel = {"left": 0.0, "right": 0.0}
    prev_q = {s_: rcp._arm_q_real(robot, iks[s_]) for s_ in ("left", "right")}
    lifted = False
    held = 0
    success = False
    t_success = None
    reset = False

    def on_pile(i: int) -> bool:
        p = rc.bar_pos(robot, i)
        return (float(np.linalg.norm(p[:2] - trial.place_xy)) < rc.PLACE_TOL
                and abs(float(p[2]) - rc.bar_centre_z(trial.pile_n)) < 0.012)

    for k in range(int(time_limit_s * fps)):
        t0 = time.perf_counter()
        if reset_requested():
            reset = True
            break
        obs = robot.get_observation()
        robot.send_action(policy.act(obs))
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        for s_ in ("left", "right"):
            q = rcp._arm_q_real(robot, iks[s_])
            travel[s_] += float(np.abs(q - prev_q[s_]).sum())
            prev_q[s_] = q
        if float(rc.bar_pos(robot, target)[2] - start[target][2]) > 0.04:
            lifted = True
        if on_pile(target):
            held += 1
            if held >= fps // 2:
                success = True
                t_success = (k + 1) / fps
                break
        else:
            held = 0

    # Which bars actually moved, and was the one it delivered the right one?
    moved = [i for i in watch
             if float(np.linalg.norm(rc.bar_pos(robot, i)[:2] - start[i][:2])) > 0.03]
    delivered = [i for i in watch if i not in trial.pile_bars and on_pile(i)]
    wrong_pad = bool(delivered) and target not in delivered
    disturbed = [i for i in moved if i != target and i not in delivered]
    return {
        "reset": reset,
        "reset_reason": "R pressed in the viewer",
        "success": bool(success and not disturbed),
        "lifted_demo": lifted,
        "t_success": t_success,
        "cube_y": float(trial.stack_xy[trial.target][1]),
        "colour": trial.colour,
        "intended_arm": trial.side,
        "committed_arm": max(travel, key=travel.get) if max(travel.values()) > 0.5 else "none",
        "wrong_pad": wrong_pad,
        "knocked": bool(disturbed),
        "pile_n": trial.pile_n,
        "final_cube_z": float(rc.bar_pos(robot, target)[2]),
    }


def build_rtc_engine(pol: CheckpointPolicy, robot, fps: int, horizon: int, task: str,
                     device: str = "cuda"):
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
        device=device,
    )
    engine.start()
    return engine


def run_color_rtc_trial(robot, iks, rng, engine, pol, fps: int, time_limit_s: float) -> dict:
    """Colour-sorting task under async RTC.

    Same scene, same success test and same result fields as run_color_trial —
    the difference is that actions come from the background inference engine
    instead of blocking the loop on a fresh chunk every n_action_steps ticks.
    The command slewing matches run_rtc_trial, gripper columns included (they
    must stay unslewed or the fingers close too late to catch the cube).
    """
    from lerobot.utils.feature_utils import build_dataset_frame

    import random_color_pick as rcol

    rcol.show_pads(robot)
    colour = "red" if rng.uniform() < 0.5 else "green"
    rgba, side, pad_xy = rcol.COLOURS[colour]
    other_pad = rcol.COLOURS["green" if colour == "red" else "red"][2]
    rcol.set_cube_colour(robot, rgba)
    cube0 = rcol.place_cube_for(robot, iks, rng, colour)
    rcp.setup_start_pose(robot, iks[side], rng, fps)
    pol.reset()
    engine.reset()
    # Publish the post-teleport observation BEFORE waking the thread so its
    # first chunk is anchored to this trial's start pose, not to whatever it
    # last saw (engine.reset() also drops the stale one; this just saves a tick).
    engine.notify_observation(robot.get_observation())
    engine.resume()  # the RTC background thread starts paused

    travel = {"left": 0.0, "right": 0.0}
    prev_q = {s_: rcp._arm_q_real(robot, iks[s_]) for s_ in ("left", "right")}
    success = False
    t_success = None
    reset = False
    held_since = None
    cmd = None  # slew-limited command state
    slew = 2.5  # deg per tick — spreads chunk-seam jumps
    grip_idx = np.array([i for i, k in enumerate(pol.action_keys) if "gripper" in k])
    t_start = time.perf_counter()
    t_end = t_start + time_limit_s
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        if reset_requested():
            reset = True
            break
        obs = robot.get_observation()
        engine.notify_observation(obs)
        frame = build_dataset_frame(pol.obs_features, obs, prefix="observation")
        a = engine.get_action(frame)
        if a is not None:
            target = a.detach().cpu().numpy().reshape(-1)
            if cmd is None:
                cmd = target.copy()
            else:
                cmd = cmd + np.clip(target - cmd, -slew, slew)
                cmd[grip_idx] = target[grip_idx]
        if cmd is not None:
            # queue priming/gap: resending the last command keeps the sim stepping
            robot.send_action({k: float(v) for k, v in zip(pol.action_keys, cmd, strict=True)})
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        for s_ in ("left", "right"):
            q = rcp._arm_q_real(robot, iks[s_])
            travel[s_] += float(np.abs(q - prev_q[s_]).sum())
            prev_q[s_] = q
        ok, _ = rcol.on_pad(robot, pad_xy)
        if ok:
            held_since = held_since or time.perf_counter()
            if time.perf_counter() - held_since > 0.5:
                success = True
                t_success = time.perf_counter() - t_start
                break
        else:
            held_since = None
        if float(rcp.cube_pos(robot)[2]) < 0.2:
            break
    committed = max(travel, key=travel.get) if max(travel.values()) > 0.5 else "none"
    wrong_pad, _ = rcol.on_pad(robot, other_pad)
    return {
        "reset": reset,
        "reset_reason": "R pressed in the viewer",
        "success": success,
        "lifted_demo": success,
        "t_success": t_success,
        "cube_y": float(cube0[1]),
        "colour": colour,
        "intended_arm": side,
        "committed_arm": committed,
        "wrong_pad": bool(wrong_pad),
        "final_cube_z": float(rcp.cube_pos(robot)[2]),
    }


def run_rtc_trial(robot, engine, pol, fps: int, time_limit_s: float = 30.0):
    """One episode under async RTC: continuous 30 Hz control, background chunks."""
    import time as _time

    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.robot_utils import precise_sleep

    import numpy as _np

    engine.reset()
    engine.notify_observation(robot.get_observation())  # anchor chunk 1 to THIS start pose
    engine.resume()  # the RTC background thread starts paused
    t_end = _time.perf_counter() + time_limit_s
    held_since = None
    cmd = None  # slew-limited command state
    slew = 2.5  # deg per tick — spreads chunk-seam jumps (measured up to 28 deg)
    # Slew must NOT touch the gripper: 0->44 at 2.5/tick takes 17 ticks
    # (0.6 s), long enough for a fast policy to start lifting before the
    # fingers close — the grasp silently misses.
    grip_idx = _np.array([i for i, k in enumerate(pol.action_keys) if "gripper" in k])
    while _time.perf_counter() < t_end:
        t0 = _time.perf_counter()
        if reset_requested():
            return None, _time.perf_counter() - (t_end - time_limit_s)
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
                cmd[grip_idx] = target[grip_idx]
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
    grip_idx = np.array([i for i, k in enumerate(pol.action_keys) if "gripper" in k])

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
    pops_since_submit = 0
    while _time.perf_counter() < t_end:
        t0 = _time.perf_counter()
        if reset_requested():
            pool.shutdown(wait=False)
            return None, _time.perf_counter() - (t_end - time_limit_s)
        obs = robot.get_observation()
        if len(queue) <= prefetch_at and future is None:
            future = pool.submit(chunk_from, dict(obs))
            pops_since_submit = 0
        if not queue:
            if future is not None:
                rows = future.result()
                # The chunk was conditioned on the observation captured at
                # submit time; the ticks executed since then are already in
                # the past. Without this skip the new chunk re-commands the
                # arm back along the path it just travelled — a visible
                # forward/back/forward sweep at every chunk boundary.
                skip = min(pops_since_submit, len(rows) - 1)
                queue = rows[skip:]
                future = None
            else:
                queue = chunk_from(obs)  # first chunk of the episode (blocking)
        target = queue.pop(0)
        pops_since_submit += 1
        if cmd is None:
            cmd = target.copy()
        else:
            cmd = cmd + np.clip(target - cmd, -slew, slew)
            cmd[grip_idx] = target[grip_idx]  # gripper unslewed (see run_rtc_trial)
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
    parser.add_argument("--arm-gain-scale", type=float, default=1.0,
                        help="servo stiffness multiplier; match the generator that recorded the dataset (all use 1.0)")
    parser.add_argument("--cameras", choices=["chest", "all"], default="chest",
                        help="must match what the policy was trained on")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--no-reset-key", action="store_true",
        help="ignore R from the viewer window. Use it when trials are being "
             "abandoned that you did not abandon: the viewer stays open and you "
             "can watch, but the abort shortcut is off.",
    )
    parser.add_argument("--seed", type=int, default=100,
                        help="use a seed NOT used for training data")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--time-limit", type=float, default=25.0)
    parser.add_argument("--task", default=None,
                        help="task prompt given to the policy (default: the generator's string for --task-mode)")
    parser.add_argument("--task-mode", choices=["lift", "color", "caddy"], default="lift",
                        help="lift: pick up the cube (random_cube_pick); color: red/green cube onto "
                             "its pad (random_color_pick); caddy: bar from the named coloured pad "
                             "onto the pile (random_caddy_pick)")
    parser.add_argument("--stacks", type=int, default=6,
                        help="caddy mode: pads on the arc; match the recorded data")
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
        "--tucked-prob", type=float, default=None,
        help="how often each arm starts tucked (0 = never, 0.75 = current default). Checkpoints "
             "trained before tucked starts existed see them as out-of-distribution.",
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
    parser.add_argument(
        "--replan-every", type=int, default=0, metavar="N",
        help="re-plan after N executed actions instead of the full chunk (16). "
             "Shortens open-loop execution during the approach; costs one extra "
             "inference per N steps.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="where the eager policy lives. With --trt-socket the engines already hold "
             "the weights and the eager model is never called, so 'cpu' keeps ~12 GB out "
             "of VRAM — required to run the TRT server and this script on one 24 GB card.",
    )
    args = parser.parse_args()
    if args.no_reset_key:
        _RESET_KEY_ENABLED["on"] = False
    if args.tucked_prob is not None:
        rcp.TUCKED_START_PROB = float(args.tucked_prob)
    if args.task is None:
        if args.task_mode == "caddy":
            args.task = "get bar from red pad"  # placeholder; set per trial from the prompt
        elif args.task_mode == "color":
            import random_color_pick as rcol

            args.task = rcol.TASK
        else:
            args.task = "pick up the red cube and lift it"
    if args.task_mode in ("color", "caddy") and args.jit:
        parser.error("--task-mode color is implemented for the synchronous and --rtc paths only")
    if args.trt_socket:
        # The RTC engine calls policy.predict_action_chunk directly, which
        # routes to TRT via this env var — the connect_trt socket only covers
        # the sync act() path. Without it, RTC silently runs eager inference
        # (~800 ms/chunk here), every merge discards the whole 16-row chunk
        # (real_delay >= chunk length) and the arm never receives an action.
        os.environ["GROOT_TRT_SOCKET"] = args.trt_socket
    if args.smooth_chunk:
        os.environ["GROOT_SMOOTH_CHUNK"] = "1"
    if args.denoise_steps:
        os.environ["GROOT_DENOISE_STEPS"] = str(args.denoise_steps)

    robot = rcp.make_robot(args.model_path, args.fps, viewer=not args.no_viewer, cameras=args.cameras,
                           arm_gain_scale=args.arm_gain_scale)
    iks = {a.side: rcp.build_ik(robot, a) for a in rcp.ARMS}
    rcp.park_both_arms(robot, iks)
    rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.2)

    if args.policy == "zeros":
        policy = ZerosPolicy(robot)
    else:
        print(f"loading policy from {args.policy} ...", flush=True)
        _t_load = time.time()
        policy = CheckpointPolicy(args.policy, args.dataset, args.task, device=args.device)
        print(f"policy ready in {time.time() - _t_load:.0f}s", flush=True)
        if args.replan_every:
            policy.replan_every = args.replan_every
            print(f"  re-planning every {args.replan_every} actions "
                  f"({args.replan_every / args.fps * 1000:.0f} ms of open-loop motion)")
        policy.configure_for_robot(robot)
        if args.trt_socket and not args.rtc:
            # Not under --rtc: the TRT server serves one client at a time, and
            # the RTC engine opens its own connection from the inference thread
            # (GROOT_TRT_SOCKET, see below). Connecting here as well would get
            # that idle socket accepted first, leaving the engine's connect()
            # sitting unaccepted in the backlog — its first chunk then blocks
            # forever on recv, no chunk ever lands, and the arm never moves.
            # The sync act() and --jit paths do need this socket.
            policy.connect_trt(args.trt_socket)

    rng = np.random.default_rng(args.seed)
    # The scene seed above does not touch the policy: GR00T's flow-matching
    # head draws its initial action noise from torch's global RNG
    # (groot_n1_7.py, torch.randn in get_action), so two runs of the same
    # checkpoint on the same 30 scenes disagreed by 24 points on one colour
    # subgroup. Seed torch too, so --seed reproduces the whole run and an A/B
    # between checkpoints differs only in the checkpoint.
    import torch as _torch

    _torch.manual_seed(args.seed)
    engine = None
    if args.rtc:
        engine = build_rtc_engine(policy, robot, args.fps, args.rtc_horizon, args.task,
                                  device=args.device)

    results = []
    try:
        t = 0
        while t < args.trials:
            reset_requested()  # clear stale key presses from the previous trial
            if args.jit:
                cube0, arm = rcp.place_reachable_cube(robot, iks, rng)
                rcp.setup_start_pose(robot, iks[arm.side], rng, args.fps)
                ok, t_used = run_jit_trial(robot, policy, args.fps, args.time_limit)
                cz = float(rcp.cube_pos(robot)[2])
                r = {"success": ok, "t_success": t_used if ok else None,
                     "cube_y": float(cube0[1]), "intended_arm": arm.side,
                     "committed_arm": arm.side, "final_cube_z": cz}
                if ok is not None:
                    print(f"trial {t + 1:>3}/{args.trials}: "
                          f"{'SUCCESS' if ok else 'fail   '} cube_y={cube0[1]:+.2f} "
                          f"jit t={t_used:.1f}s cube_z={cz:.3f}")
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks[arm.side], 0.0, args.fps, hold_s=0.15)
            elif engine is not None and args.task_mode == "color":
                r = run_color_rtc_trial(robot, iks, rng, engine, policy, args.fps, args.time_limit)
            elif engine is not None:
                cube0, arm = rcp.place_reachable_cube(robot, iks, rng)
                rcp.setup_start_pose(robot, iks[arm.side], rng, args.fps)
                policy.reset()
                ok, t_used = run_rtc_trial(robot, engine, policy, args.fps, args.time_limit)
                cz = float(rcp.cube_pos(robot)[2])
                r = {"success": ok, "t_success": t_used if ok else None,
                     "cube_y": float(cube0[1]), "intended_arm": arm.side,
                     "committed_arm": arm.side, "final_cube_z": cz}
                if ok is not None:
                    print(f"trial {t + 1:>3}/{args.trials}: "
                          f"{'SUCCESS' if ok else 'fail   '} cube_y={cube0[1]:+.2f} "
                          f"rtc t={t_used:.1f}s cube_z={cz:.3f}")
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks[arm.side], 0.0, args.fps, hold_s=0.15)
            elif args.task_mode == "caddy":
                r = run_caddy_trial(robot, iks, rng, policy, args.fps, args.time_limit, args.stacks)
            elif args.task_mode == "color":
                r = run_color_trial(robot, iks, rng, policy, args.fps, args.time_limit)
            else:
                r = run_trial(robot, iks, rng, policy, args.fps, args.time_limit)
            if r.get("reset") or r["success"] is None:
                # Not necessarily R: an unplannable grasp and an unsafe start
                # pose come back the same way, and calling all three "R pressed"
                # hid a harness bug behind a user action nobody performed.
                print(f"  trial abandoned ({r.get('reset_reason', 'cause not recorded')})"
                      " — not counted")
                rcp.park_both_arms(robot, iks)
                rcp.settle_pose(robot, iks["right"], 0.0, args.fps, hold_s=0.15)
                continue
            t += 1
            results.append(r)
            print(
                f"trial {t:>3}/{args.trials}: {'SUCCESS' if r['success'] else 'fail   '} "
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
            demo_ok = sum(1 for r in results if r.get("lifted_demo"))
            print(
                f"  lifted >= {0.8 * rcp.AIM_LIFT_M * 100:.0f} cm (the demonstrations' own success bar): "
                f"{demo_ok}/{n}"
            )
            for side, pred in (("left", lambda r: r["cube_y"] > 0), ("right", lambda r: r["cube_y"] <= 0)):
                grp = [r for r in results if pred(r)]
                if grp:
                    g_ok = sum(r["success"] for r in grp)
                    print(f"  cube on {side:5s} side: {g_ok}/{len(grp)}")
            wrong = sum(
                1 for r in results if r["committed_arm"] not in (r["intended_arm"], "none")
            )
            print(f"  arm-selection mismatches (committed != scripted choice): {wrong}/{n}")
            if args.task_mode == "caddy":
                print(f"  took a bar from the WRONG pad: {sum(1 for r in results if r.get('wrong_pad'))}/{n}")
                print(f"  knocked another bar over:      {sum(1 for r in results if r.get('knocked'))}/{n}")
                for colour in sorted({r.get("colour") for r in results if r.get("colour")}):
                    grp = [r for r in results if r.get("colour") == colour]
                    print(f"  {colour:7s}: {sum(r['success'] for r in grp)}/{len(grp)}")
                for lo, hi, label in ((0, 0, "empty"), (1, 2, "1-2 bars"), (3, 9, "3+ bars")):
                    grp = [r for r in results if lo <= r.get("pile_n", -1) <= hi]
                    if grp:
                        print(f"  pile {label:8s}: {sum(r['success'] for r in grp)}/{len(grp)}")
            elif args.task_mode == "color":
                for colour in ("red", "green"):
                    grp = [r for r in results if r.get("colour") == colour]
                    if grp:
                        print(f"  {colour:5s} cubes: {sum(r['success'] for r in grp)}/{len(grp)}")
                print(f"  cube ended on the WRONG pad: {sum(1 for r in results if r.get('wrong_pad'))}/{n}")
            if ok:
                print(f"  mean time to success: {np.mean([r['t_success'] for r in ok]):.1f}s")
        robot.disconnect()


if __name__ == "__main__":
    main()
