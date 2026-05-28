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

"""Hard-abort safety gate for policy actions sent to the OpenArm follower(s).

Ported from SparkJAX's ``openpi_runner_node._check_action_safety`` (see
``/home/evaughan/sparkpack/SparkJAX/sparkjax/sparkjax/teleop/openpi_runner_node.py``).

Three independent gates, any one fails -> abort the policy session:

  1. **Finite-value gate.** Rejects NaN / +Inf / -Inf produced by a miscalibrated
     FP8 pipeline, a malformed server message, or an upstream divide-by-zero.
     This *must* come first because ``nan > x`` is False, so the delta gate
     silently passes NaN.
  2. **Absolute-envelope gate.** Rejects values outside a wide sanity envelope
     (default ±200 deg per joint, ±300 deg per gripper). Catches unit
     confusions (rad-vs-deg) and runaway integrators that survive the delta
     gate by drifting slowly. Intentionally *wider* than the follower's actual
     joint limits — the follower's own clamp is the authoritative kinematic
     bound; this is a sanity gate.
  3. **Per-step delta gate.** Rejects steps larger than the configured
     ``max_joint_delta_deg`` / ``max_gripper_delta_deg``. Defaults match
     SparkJAX's ``0.5 rad/step`` (≈ 28.6°) for arm joints and ``1.0/step``
     (≈ 57.3°) for grippers at 50 Hz — already aggressive. The primary
     protection against chunk-swap discontinuities (e.g. RTC splice landing in
     the free region) and bad diffusion outputs.

Units: this module works in **degrees** to match the OpenArm follower's
wire-side interface (``Motor(..., MotorNormMode.DEGREES)`` in
``OpenArmFollower.__init__``). SparkJAX's underlying numbers are in radians
(openpi-native); the defaults here are the rad→deg conversion of the SparkJAX
defaults.

On the very first call the checker seeds ``_last_action`` from the observed
follower joint state (passed via ``observation``) so step 0 is also gated.
On any violation, the checker becomes "aborted" and every subsequent
:py:meth:`ActionSafetyChecker.check` returns the same error string — so the
caller can hold the last good pose and bail out without re-arming until the
operator explicitly resets the checker.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Rad→deg conversions of SparkJAX's hard-abort thresholds (defined as
# ``_DEFAULT_MAX_JOINT_DELTA = 0.5`` and ``_DEFAULT_MAX_GRIP_DELTA = 1.0`` in
# ``openpi_runner_node.py``). At 50 Hz, 28.65 deg/step = ~1432 deg/s on a joint,
# which is already very aggressive — the threshold is "is this physically
# implausible" not "is this slightly jerky".
_RAD_TO_DEG = 180.0 / math.pi
_SPARKJAX_DEFAULT_MAX_JOINT_DELTA_DEG = 0.5 * _RAD_TO_DEG   # 28.647...
_SPARKJAX_DEFAULT_MAX_GRIP_DELTA_DEG = 1.0 * _RAD_TO_DEG    # 57.295...

# Wide absolute sanity envelope. The OpenArm's actual joint limits are much
# tighter (e.g. ±75 deg on joint_1, [-40, 40] on joint_6); those are enforced
# by ``OpenArmFollower.send_action`` via ``config.joint_limits``. This envelope
# catches a different failure mode: garbage values (NaN-survived integrators,
# rad-vs-deg confusion, etc.) that the kinematic clamp would happily truncate
# silently. ±200 deg is impossible for any real action but well-defined for
# floats.
_DEFAULT_ABS_JOINT_LIMIT_DEG = 200.0    # SparkJAX uses ±180° (i.e. ±π rad)
_DEFAULT_ABS_GRIPPER_LIMIT_DEG = 300.0   # SparkJAX uses ±3.5 in "gripper units"

_DEFAULT_GRIPPER_NAME_SUBSTR = "gripper"


class ActionSafetyViolation(RuntimeError):
    """Raised by the caller when an :class:`ActionSafetyChecker` reports a violation.

    Carrying it as a dedicated exception lets the outer record loop catch it
    specifically (vs. unrelated runtime errors) for telemetry, while still
    propagating up to the ``try/finally`` that disconnects the robot and
    disables torque.
    """


@dataclass
class ActionSafetyConfig:
    """Tunables for :class:`ActionSafetyChecker`.

    Defaults reproduce SparkJAX's behavior expressed in degrees (since the
    OpenArm follower's wire format is degrees, not radians).
    """

    # Master switch; when False, :py:meth:`ActionSafetyChecker.check` always returns None.
    enabled: bool = True
    # Per-step delta thresholds. Any motor whose key contains
    # ``gripper_name_substr`` uses ``max_gripper_delta_deg``; everything else
    # uses ``max_joint_delta_deg``.
    max_joint_delta_deg: float = _SPARKJAX_DEFAULT_MAX_JOINT_DELTA_DEG
    max_gripper_delta_deg: float = _SPARKJAX_DEFAULT_MAX_GRIP_DELTA_DEG
    # Absolute-envelope thresholds (|value| <= limit).
    abs_joint_limit_deg: float = _DEFAULT_ABS_JOINT_LIMIT_DEG
    abs_gripper_limit_deg: float = _DEFAULT_ABS_GRIPPER_LIMIT_DEG
    # Substring used to flag a motor as a gripper (e.g. "gripper" matches
    # "left_gripper.pos" and "right_gripper.pos").
    gripper_name_substr: str = _DEFAULT_GRIPPER_NAME_SUBSTR


class ActionSafetyChecker:
    """Three-gate safety guard for ``RobotAction`` dicts.

    Usage::

        checker = ActionSafetyChecker(ActionSafetyConfig())
        ...
        err = checker.check(action_dict, observation_dict)
        if err is not None:
            # hold last good pose, then surface the abort
            if checker.last_action is not None:
                robot.send_action(checker.last_action)
            raise ActionSafetyViolation(err)

    Thread-safe: no, single-threaded use only.
    """

    def __init__(self, cfg: ActionSafetyConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ActionSafetyConfig()
        self.last_action: dict[str, float] | None = None
        self._aborted: bool = False
        self._abort_reason: str | None = None

    @property
    def aborted(self) -> bool:
        return self._aborted

    def reset(self) -> None:
        """Clear last-action seed and abort state. Use between sessions."""
        self.last_action = None
        self._aborted = False
        self._abort_reason = None

    def _is_gripper(self, key: str) -> bool:
        return self.cfg.gripper_name_substr in key

    def _seed_from_observation(self, action: dict[str, float], obs: dict[str, Any] | None) -> None:
        """Seed ``last_action`` from the observed joint positions for the same keys.

        Mirrors SparkJAX's behavior of seeding the delta-gate baseline from the
        actual follower state so step 0 is also bounded.
        """
        if obs is None:
            self.last_action = dict(action)
            return
        seed: dict[str, float] = {}
        for k in action:
            v = obs.get(k)
            if v is None:
                seed[k] = float(action[k])
            else:
                try:
                    seed[k] = float(v)
                except (TypeError, ValueError):
                    seed[k] = float(action[k])
        self.last_action = seed

    def check(
        self, action: dict[str, float], observation: dict[str, Any] | None = None
    ) -> str | None:
        """Run all three gates against ``action``. Returns None if safe, else the abort reason.

        Once a violation occurs, the checker is "aborted" and subsequent calls
        keep returning the original reason without further evaluation. Call
        :py:meth:`reset` to re-arm (only after an operator-visible decision).
        """
        if not self.cfg.enabled:
            return None
        if self._aborted:
            return self._abort_reason

        if not isinstance(action, dict):
            return self._abort(
                f"SAFETY STOP: action is not a dict (got {type(action).__name__})"
            )
        if not action:
            return self._abort("SAFETY STOP: action dict is empty")

        # Gate 1: finite-value
        for k, v in action.items():
            try:
                vf = float(v)
            except (TypeError, ValueError):
                return self._abort(f"SAFETY STOP: non-numeric action value at {k}: {v!r}")
            if not math.isfinite(vf):
                return self._abort(f"SAFETY STOP: non-finite action at {k} (value={vf})")

        # Gate 2: absolute envelope
        for k, v in action.items():
            vf = float(v)
            limit = (
                self.cfg.abs_gripper_limit_deg
                if self._is_gripper(k)
                else self.cfg.abs_joint_limit_deg
            )
            if abs(vf) > limit:
                return self._abort(
                    f"SAFETY STOP: {k} absolute value {vf:.3f} deg exceeds envelope "
                    f"±{limit:.3f} deg"
                )

        # Gate 3: per-step delta (after seeding on the very first call)
        if self.last_action is None:
            self._seed_from_observation(action, observation)
            return None

        for k, v in action.items():
            prev = self.last_action.get(k)
            if prev is None:
                # New key showed up mid-session — just register it without gating.
                continue
            delta = abs(float(v) - float(prev))
            limit = (
                self.cfg.max_gripper_delta_deg
                if self._is_gripper(k)
                else self.cfg.max_joint_delta_deg
            )
            if delta > limit:
                return self._abort(
                    f"SAFETY STOP: {k} jumped {delta:.3f} deg/step "
                    f"(limit {limit:.3f} deg/step). Prev={prev:.3f}, New={float(v):.3f}"
                )

        # All gates passed -> update baseline
        self.last_action = dict(action)
        return None

    def _abort(self, reason: str) -> str:
        self._aborted = True
        self._abort_reason = reason
        logger.error(reason)
        return reason


__all__ = [
    "ActionSafetyConfig",
    "ActionSafetyChecker",
    "ActionSafetyViolation",
]
