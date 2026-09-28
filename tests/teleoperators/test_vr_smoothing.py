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

import numpy as np

from lerobot.teleoperators.vr_mocap.smoothing import OneEuroPose

Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def test_still_hand_jitter_is_attenuated():
    rng = np.random.default_rng(0)
    f = OneEuroPose()
    out = [f(rng.normal(0, 0.002, 3), Q0, i / 90)[0] for i in range(900)]  # 2 mm jitter at 90 Hz
    assert np.std(np.array(out[90:]), axis=0).max() < 0.0007  # cut to about a third


def test_fast_motion_passes_with_little_lag():
    f = OneEuroPose()
    lag = []
    for i in range(180):  # 1 m/s sweep
        x = np.array([i / 90, 0.0, 0.0])
        p, _ = f(x, Q0, i / 90)
        lag.append(x[0] - p[0])
    assert max(lag[60:]) < 0.02  # under 2 cm behind at 1 m/s


def test_reset_passes_the_next_pose_through():
    f = OneEuroPose()
    for i in range(30):
        f(np.zeros(3), Q0, i / 90)
    f.reset()
    p, q = f(np.array([0.5, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]), 1.0)
    assert np.allclose(p, [0.5, 0, 0]) and np.allclose(q, [0, 1, 0, 0])
