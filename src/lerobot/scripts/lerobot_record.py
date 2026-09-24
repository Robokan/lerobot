# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

To stream the data to Foxglove instead of Rerun, add ``--display_mode=foxglove`` (then connect the
Foxglove app to ``ws://127.0.0.1:8765``; override the port with ``--display_port=<port>``).

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.rgb_encoder.vcodec=h264 \\
    --dataset.rgb_encoder.preset=fast \\
    --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.mujoco import MujocoCameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    mujoco_bi_openarm,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
    vr_mocap,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import apply_recording_control, init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


def _drain_viewer_recording_controls(events: dict, teleop=None) -> None:
    """Apply recording controls from the MuJoCo viewer keys and, when the
    teleoperator surfaces them (VR controller B/A/X), from the headset."""
    try:
        from lerobot.robots.mujoco_bi_openarm.viewer_keys import drain_recording_controls
    except Exception:  # noqa: BLE001
        pass
    else:
        for control in drain_recording_controls():
            apply_recording_control(control, events)
    drain = getattr(teleop, "drain_recording_controls", None)
    if callable(drain):
        for control in drain():
            apply_recording_control(control, events)
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Visualization backend used when display_data is True: "rerun" or "foxglove".
    display_mode: str = "rerun"
    # For "rerun": IP of a remote server to send to. For "foxglove": interface to bind the WebSocket
    # server to (127.0.0.1 for local only, 0.0.0.0 for all interfaces).
    display_ip: str | None = None
    # For "rerun": port of the remote server. For "foxglove": port to bind the WebSocket server to.
    display_port: int | None = None
    # Whether to display compressed (JPEG) images instead of raw frames
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps
    # None / <=0 means no wall-clock limit (Y/T-gated capture runs until T/n/q).
    if control_time_s is None or control_time_s <= 0:
        control_time_s = float("inf")

    # When a dataset is attached, frames are gated on events["recording_active"]
    # (Y starts, T/n stops). Reset loops pass dataset=None and use a wall clock
    # from loop entry like before.
    gated = dataset is not None
    no_action_count = 0
    timestamp = 0.0
    start_episode_t: float | None = None if gated else time.perf_counter()
    was_recording = False
    last_slow_warn_t = 0.0
    slow_warn_interval_s = 2.0
    has_episode_time_limit = control_time_s != float("inf")

    while True:
        start_loop_t = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop *before* draining Y/T: KeyboardPoseSource queues
        # viewer recording keys inside get_action/send_feedback.
        if isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            # Most teleops no-op; OpenXR vr_mocap uses this for the headset camera feed.
            teleop.send_feedback(obs)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            _drain_viewer_recording_controls(events, teleop)
            if events["exit_early"] or events["stop_recording"]:
                events["exit_early"] = False
                events["recording_active"] = False
                break
            continue

        _drain_viewer_recording_controls(events, teleop)

        if events["exit_early"]:
            events["exit_early"] = False
            events["recording_active"] = False
            break
        if events["stop_recording"]:
            events["recording_active"] = False
            break

        if gated:
            if events.get("recording_active"):
                if not was_recording:
                    start_episode_t = time.perf_counter()
                    was_recording = True
                    ep_idx = dataset.num_episodes
                    limit_msg = (
                        f"Optional max duration: {control_time_s}s."
                        if has_episode_time_limit
                        else "No time limit — press T to stop and save."
                    )
                    print(
                        f"\n>>> RECORDING STARTED — episode {ep_idx} "
                        f"(repo_id={dataset.repo_id})\n"
                        f"    {limit_msg}\n",
                        flush=True,
                    )
                    log_say("Recording started", play_sounds=False)
                    logging.info(
                        "Recording started: episode %s → %s",
                        ep_idx,
                        dataset.root,
                    )
                assert start_episode_t is not None
                timestamp = time.perf_counter() - start_episode_t
                if has_episode_time_limit and timestamp >= control_time_s:
                    events["recording_active"] = False
                    print(
                        f"\n>>> RECORDING STOPPED — episode time limit "
                        f"({control_time_s}s) reached; saving…\n",
                        flush=True,
                    )
                    logging.info("Episode time limit reached — saving.")
                    break
            else:
                was_recording = False
        else:
            assert start_episode_t is not None
            timestamp = time.perf_counter() - start_episode_t
            if timestamp >= control_time_s:
                break

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset only while recording is armed (Y…T).
        if dataset is not None and events.get("recording_active"):
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            # A robot that arranges its own scene (mujoco_bi_openarm_caddy) publishes
            # the prompt for the current layout; it overrides the run-wide task.
            frame = {**observation_frame, **action_frame,
                     "task": getattr(robot, "current_task", None) or single_task}
            dataset.add_frame(frame)

        if display_data:
            log_visualization_data(
                display_mode,
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            now = time.perf_counter()
            if now - last_slow_warn_t >= slow_warn_interval_s:
                last_slow_warn_t = now
                logging.warning(
                    "Record loop is running slower (%.1f Hz) than the target FPS (%s Hz). "
                    "Dataset timestamps assume %s Hz — lower --dataset.fps (e.g. 30) or use "
                    "CAMERAS=0 / close the viewer if you need a steadier rate. "
                    "Common causes: camera renders, viewer GL, disk image writes.",
                    1 / dt_s,
                    fps,
                    fps,
                )

        precise_sleep(max(sleep_time_s, 0.0))


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        # Connect the teleoperator before the robot so the robot isn't left idle (and possibly
        # tripping a firmware watchdog) during teleop init. Matches lerobot_teleoperate.py.
        if teleop is not None:
            teleop.connect()
        robot.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.rgb_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            first_pass = True     # connect() already arranged the first scene
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                events["recording_active"] = False
                events["exit_early"] = False
                ep_idx = dataset.num_episodes
                print(
                    f"\n=== Episode {ep_idx} ready "
                    f"(will save as episode-{ep_idx:06d} under {dataset.root}) ===\n"
                    f"    repo_id: {dataset.repo_id}\n"
                    f"    Press Y to START recording, T to STOP and save.\n"
                    f"    (n=end early, Left arrow=re-record, q=quit; "
                    f"letter r is teleop +z, not re-record)\n",
                    flush=True,
                )
                # Robots that lay out their own scene get a fresh one per episode.
                # Not on the first: connect() already arranged it.
                new_episode = getattr(robot, "new_episode", None)
                if callable(new_episode) and not first_pass:
                    new_episode()          # also after a re-record: the failed attempt disturbed the table
                first_pass = False
                task_hint = getattr(robot, "current_task", None)
                log_say(f"Ready for episode {ep_idx}. Press Y to start."
                        + (f"  Task: {task_hint}" if task_hint else ""), cfg.play_sounds)
                logging.info(
                    "Press Y to start recording episode %s (%s), T to stop and save.",
                    ep_idx,
                    dataset.repo_id,
                )
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    events["recording_active"] = False

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    events["recording_active"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Only save when the episode buffer has frames (Y…T / timeout).
                if not dataset.has_pending_frames():
                    print(
                        "\n>>> No frames captured — nothing saved. Press Y to start recording.\n",
                        flush=True,
                    )
                    logging.warning("No frames captured for this episode — skipping save.")
                    if events["stop_recording"]:
                        break
                    continue

                ep_idx = dataset.num_episodes
                n_frames = (
                    dataset.writer.episode_buffer["size"]
                    if dataset.writer is not None and dataset.writer.episode_buffer is not None
                    else 0
                )
                print(
                    f"\n>>> RECORDING STOPPED — saving episode {ep_idx} "
                    f"({n_frames} frames)…\n",
                    flush=True,
                )
                # Sequential video encode: ProcessPool after MuJoCo viewer/GL init
                # deadlocks on this box (fork + OpenGL), looking like a freeze on Map:.
                print(
                    "\n>>> Encoding videos (sequential — may take a bit for long episodes)…\n",
                    flush=True,
                )
                dataset.save_episode(parallel_encoding=False)
                recorded_episodes += 1
                print(
                    f"\n>>> SAVED episode {ep_idx} as:\n"
                    f"    repo_id : {dataset.repo_id}\n"
                    f"    path    : {dataset.root}\n"
                    f"    episode : episode-{ep_idx:06d}  ({n_frames} frames)\n"
                    f"    total episodes in dataset: {dataset.num_episodes}\n",
                    flush=True,
                )
                logging.info(
                    "Saved episode %s (%s frames) to %s (repo_id=%s)",
                    ep_idx,
                    n_frames,
                    dataset.root,
                    dataset.repo_id,
                )
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        # Persist any unsaved frames before viewer disconnect / os._exit.
        if dataset is not None and dataset.has_pending_frames():
            try:
                ep_idx = dataset.num_episodes
                n_frames = dataset.writer.episode_buffer["size"]
                print(
                    f"\n>>> Flushing unsaved episode {ep_idx} ({n_frames} frames) on exit…\n",
                    flush=True,
                )
                dataset.save_episode(parallel_encoding=False)
                print(
                    f"\n>>> SAVED episode {ep_idx} as:\n"
                    f"    repo_id : {dataset.repo_id}\n"
                    f"    path    : {dataset.root}\n"
                    f"    episode : episode-{ep_idx:06d}  ({n_frames} frames)\n",
                    flush=True,
                )
            except Exception:  # noqa: BLE001
                logging.exception("Failed to flush pending episode on exit")

        if dataset:
            dataset.finalize()

        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)

        # Disconnect robot last: VIEWER=1 registers atexit(os._exit) which skips
        # remaining cleanup if we disconnect earlier.
        if robot.is_connected:
            robot.disconnect()
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
