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

"""Force a known-good encoder zero by physically re-homing both OpenArm
followers to their calibration default pose ("arms hanging straight down,
grippers closed") before any motion.

Why this is necessary:
    The lerobot OpenArm calibration file
    (``~/.cache/huggingface/lerobot/calibration/robots/openarm_follower/*.json``)
    stores **only joint limit ranges and `homing_offset=0`** — there is no
    absolute reference pose recorded. Every ``OpenArmFollower.connect()``
    silently calls ``bus.set_zero_position()``, which captures whatever the
    arms physically happen to be at the moment of connect as the new encoder
    zero. If the arms came in twisted (left over from a previous bad session,
    operator-bumped, etc.), that twisted pose becomes "zero", and commanding
    (0, 0, ..., 0) won't move them — the lift spine then runs from a wrong
    baseline and the policy sees state vectors offset from training.

    SparkJAX sidesteps this because its C++ ``unilateral_control`` binary
    runs ``AdjustPosition`` with default-zero target + soft gains, which
    will compliantly pull the arms toward zero IF the encoder zero already
    matches the calibration zero. But the encoder zero only matches the
    calibration zero if the arms were physically at hanging-down + closed
    when ``set_zero_position()`` ran — exactly what this routine guarantees.

What this routine does:
    1. Disables torque on both arms (arms go limp; user can move freely by hand).
    2. Prompts the operator to physically position the arms in the default
       pose and press ENTER.
    3. Re-runs ``bus.set_zero_position()`` on both arms — captures the now-
       correct physical pose as the encoder zero.
    4. Re-enables torque (the PD loop now holds the arms at the just-captured
       zero, which is where they physically are, so no yank).

Call this BEFORE :func:`lift_arms_to_ready` to ensure phase 0's "ramp to
zero" actually corresponds to the calibration default pose.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REHOME_PROMPT: str = (
    "\n"
    "============================================================\n"
    "  RE-HOME ARMS\n"
    "  Torque is OFF on both arms — they will be limp.\n"
    "  Physically position both arms in the default pose:\n"
    "    - Arm hanging straight down\n"
    "    - Wrist neutral (NOT twisted left/right)\n"
    "    - Gripper closed\n"
    "  Press ENTER when both arms are in position...\n"
    "============================================================"
)


def rehome_at_default_pose(
    robot: Any,
    *,
    prompt: str = DEFAULT_REHOME_PROMPT,
    settle_s: float = 0.5,
    interactive: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Disable torque, prompt the operator to re-position both arms, then
    capture the new physical pose as encoder zero and re-enable torque.

    Args:
        robot: A :class:`BiOpenArmFollower` (must expose ``left_arm`` and
            ``right_arm`` attributes, each with a ``bus`` exposing
            ``disable_torque()``, ``enable_torque()``, and
            ``set_zero_position()``). Must be connected.
        prompt: Text shown to the operator before they press ENTER. Override
            with an empty string to skip the prompt and just do the
            torque-off → set-zero → torque-on dance (useful for scripted
            flows that prompted elsewhere).
        settle_s: Seconds to sleep after re-enabling torque, to let the PD
            loop settle on the new zero before the caller starts streaming
            commands.
        interactive: If False, skip the ``input(...)`` call. Useful when
            running this from non-interactive contexts where the operator
            cannot type. In non-interactive mode the routine still does the
            disable/set-zero/enable sequence — useful when you have *some*
            other mechanism guaranteeing the arms are at the default pose.
        log_fn: Optional callable for human-visible progress messages
            (defaults to :func:`logging.getLogger(__name__).info`).
    """
    if not robot.is_connected:
        raise RuntimeError("rehome_at_default_pose: robot must be connected first.")
    if not hasattr(robot, "left_arm") or not hasattr(robot, "right_arm"):
        raise TypeError(
            f"rehome_at_default_pose expects a bimanual robot with left_arm + right_arm; "
            f"got {type(robot).__name__}"
        )

    log = log_fn if log_fn is not None else logger.info

    log("re-home: disabling torque on both arms (they will go limp)")
    robot.left_arm.bus.disable_torque()
    robot.right_arm.bus.disable_torque()

    if interactive and prompt:
        try:
            input(prompt)
        except EOFError:
            # Non-interactive stdin (e.g. redirected) — fall through.
            log("re-home: stdin not a tty, skipping prompt (assuming arms are positioned)")

    log("re-home: capturing current physical pose as the new encoder zero")
    robot.left_arm.bus.set_zero_position()
    robot.right_arm.bus.set_zero_position()

    log("re-home: re-enabling torque")
    robot.left_arm.bus.enable_torque()
    robot.right_arm.bus.enable_torque()

    if settle_s > 0:
        time.sleep(settle_s)

    log("re-home: done — encoder zero now matches the physical default pose")


__all__ = [
    "DEFAULT_REHOME_PROMPT",
    "rehome_at_default_pose",
]
