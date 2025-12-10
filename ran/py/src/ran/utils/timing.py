# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Timing utilities."""

import logging
import time
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


def _time_function(
    func: Callable, func_inputs: dict, num_repeats: int = 10
) -> tuple[float, float, float]:
    """Time a function over several repeats and return mean time."""
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        _ = func(**func_inputs)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.min(times), np.max(times)  # type: ignore[return-value]


def log_time(
    f1: Callable,
    f2: Callable,
    f1_inputs: dict,
    f2_inputs: dict | None = None,
    repeat: int = 5,
) -> None:
    """
    Time two functions and print the time taken.

    Inputs:
    - f1: first function to time
    - f2: second function to time
    - f1_inputs: inputs for the first function
    - f2_inputs: inputs for the second function. If None, f2_inputs = f1_inputs.
    - repeat: number of times to repeat timing
    """
    if f2_inputs is None:
        f2_inputs = f1_inputs
    t1, t1_min, t1_max = _time_function(f1, f1_inputs, repeat)
    logger.info(f"f1 ({f1.__name__}) time: {t1:.3f} seconds (min: {t1_min:.3f}, max: {t1_max:.3f})")

    t2, t2_min, t2_max = _time_function(f2, f2_inputs, repeat)
    logger.info(f"f2 ({f2.__name__}) time: {t2:.3f} seconds (min: {t2_min:.3f}, max: {t2_max:.3f})")

    logger.info(f"Difference: {abs(t1 - t2) * 1e3:.1f} ms")

    percent = ((t1 - t2) / t2) * 100 if t2 != 0 else 0
    logger.info(f"Speed up: {(t1 / t2):.2f}x ({percent:.1f}% faster)")
