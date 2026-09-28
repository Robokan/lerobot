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

"""One-euro filter for controller poses (Casiez et al., CHI 2012).

A low-pass whose cutoff rises with speed: a hand held still is smoothed hard
(the tracking jitter that made the lightly damped wrist motors ring), a hand
moving fast is barely filtered, so there is little lag where it would be felt.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt / (dt + tau)


class OneEuroPose:
    """Filters a position (m) and a unit quaternion (w, x, y, z) sampled at irregular times."""

    def __init__(
        self,
        min_cutoff_hz: float = 1.0,
        beta_pos: float = 5.0,
        beta_rot: float = 0.5,
        d_cutoff_hz: float = 1.0,
    ):
        self.min_cutoff_hz = min_cutoff_hz
        self.beta_pos = beta_pos
        self.beta_rot = beta_rot
        self.d_cutoff_hz = d_cutoff_hz
        self.reset()

    def reset(self) -> None:
        """Forget the history: the next sample passes through unfiltered."""
        self._t = None
        self._p = self._q = None
        self._v = 0.0  # filtered linear speed, m/s
        self._w = 0.0  # filtered angular speed, rad/s

    def __call__(self, pos, quat, t: float) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(pos, dtype=float)
        quat = np.asarray(quat, dtype=float)
        quat = quat / np.linalg.norm(quat)
        if self._t is None or t <= self._t:
            self._t, self._p, self._q = t, pos.copy(), quat.copy()
            return pos, quat
        dt = t - self._t
        self._t = t

        # position
        speed = float(np.linalg.norm(pos - self._p)) / dt
        self._v += _alpha(self.d_cutoff_hz, dt) * (speed - self._v)
        a = _alpha(self.min_cutoff_hz + self.beta_pos * self._v, dt)
        self._p = self._p + a * (pos - self._p)

        # orientation: same rule on the angle, applied as a slerp
        if float(np.dot(quat, self._q)) < 0.0:
            quat = -quat  # same rotation, nearer hemisphere
        ang = 2.0 * math.acos(min(1.0, abs(float(np.dot(quat, self._q)))))
        self._w += _alpha(self.d_cutoff_hz, dt) * (ang / dt - self._w)
        a = _alpha(self.min_cutoff_hz + self.beta_rot * self._w, dt)
        self._q = _slerp(self._q, quat, a)
        return self._p.copy(), self._q.copy()


def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    q = (math.sin((1.0 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)
