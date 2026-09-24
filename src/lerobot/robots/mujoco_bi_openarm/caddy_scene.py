#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""The simulated OpenArm with the caddy-picking scene arranged around it, for
human (VR) demonstrations.

Why this exists: 900 planner-scripted episodes of this task trained three
policies that all scored 0-1/20 closed loop, while 100 human teleop episodes on
the real robot produced occasional successes. The planner's demonstrations are
too perfect to learn from — no drift, no recovery, and paths threaded through
margins a policy cannot hold. So the demonstrations will come from a person.

This robot is :class:`MujocoBiOpenArm` plus one thing: before every recorded
episode it lays out the same scene the scripted generator used — N stacks of
bars on coloured pads on an arc, a pile of already-fetched bars at the centre —
draws a target pad, and publishes the prompt (``"get bar from <colour> pad"``)
as ``current_task`` so the record loop stamps every frame with it. The layout
code is the generator's own (:mod:`scripts.random_caddy_pick.Trial`), so a
human dataset and a scripted one are drawn from the identical distribution and
can be trained on together or compared directly.

The arms are NOT repositioned between episodes: the operator is holding them.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.robots.config import RobotConfig

from .config_mujoco_bi_openarm import MujocoBiOpenArmConfig
from .mujoco_bi_openarm import MujocoBiOpenArm

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("mujoco_bi_openarm_caddy")
@dataclass
class MujocoBiOpenArmCaddyConfig(MujocoBiOpenArmConfig):
    """MujocoBiOpenArm + the caddy scene re-arranged before every episode."""

    # Number of coloured pads / stacks on the arc (the scripted datasets used 6).
    stacks: int = 6
    # Seed for the layout draws. Every episode advances the same stream, so one
    # seed gives one reproducible sequence of scenes.
    seed: int = 0


class MujocoBiOpenArmCaddy(MujocoBiOpenArm):
    config_class = MujocoBiOpenArmCaddyConfig
    name = "mujoco_bi_openarm_caddy"

    def __init__(self, config: MujocoBiOpenArmCaddyConfig):
        super().__init__(config)
        self.config: MujocoBiOpenArmCaddyConfig = config
        self._rng = np.random.default_rng(config.seed)
        self._rc = None            # scripts.random_caddy_pick, imported lazily
        self._rcp = None           # scripts.random_cube_pick
        self.trial = None
        self.current_task: str | None = None
        self.episodes_arranged = 0

    # The scene code lives in scripts/, not in the package (it is the data
    # generator, and it is on the list to move into a project repo of its own).
    # Import it lazily from the checkout this file sits in, so lerobot-record
    # can construct this robot without any change to sys.path on the caller's
    # side.
    def _scene_modules(self):
        if self._rc is None:
            scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            import random_caddy_pick as rc  # noqa: E402
            import random_cube_pick as rcp  # noqa: E402

            self._rc, self._rcp = rc, rcp
        return self._rc, self._rcp

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        rc, rcp = self._scene_modules()
        rc.hide_legacy_pads(self)
        rcp.set_cube_xy(self, -0.90, -0.90)   # the cube shares the scene; park it out of the way
        self.new_episode()

    def new_episode(self) -> str:
        """Lay out a fresh scene and return its prompt. Called by the record
        loop before every episode (and on re-record), and once at connect."""
        rc, _ = self._scene_modules()
        # Trial() draws the SIDE from a shuffled two-element bag, so the arms
        # alternate evenly however many scenes are drawn; and it re-parks every
        # bar before placing the new stacks, so nothing from the last episode
        # survives on the table.
        self.trial = rc.Trial(self, self._rng, self.config.stacks)
        self.current_task = self.trial.prompt
        self.episodes_arranged += 1
        logger.info("caddy scene %d: %s", self.episodes_arranged, self.trial.describe())
        print(f"\n=== EPISODE {self.episodes_arranged}:  \"{self.current_task}\"  "
              f"({self.trial.side} arm) ===\n", flush=True)
        return self.current_task
