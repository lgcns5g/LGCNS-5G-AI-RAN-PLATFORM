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

"""Helper for comparing arrays."""

import logging
from typing import Any

import numpy as np

# note: "Any" avoids type errors when using untyped objects from H5 dictionaries
logger = logging.getLogger(__name__)


def max_abs_diff(
    x: Any,  # noqa: ANN401
    y: Any,  # noqa: ANN401
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """Compare two arrays and optionally print the differences.

    Args:
        x: First array to compare.
        y: Second array to compare.
        eps: Small value to avoid division by zero.

    Returns
    -------
        max_abs: Maximum absolute difference.
        mean_abs: Mean absolute difference.
        max_rel_pct: Maximum relative percentage difference.
        mean_rel_pct: Mean relative percentage difference.
    """
    # Flatten, ignore all-zero denom to avoid inf
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    abs_diff = np.abs(x - y)
    denom = np.maximum(np.abs(y), eps)

    max_abs = float(abs_diff.max(initial=0.0))
    mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
    max_rel_pct = float((abs_diff / denom).max(initial=0.0) * 100.0)
    mean_rel_pct = float((abs_diff / denom).mean() * 100.0) if abs_diff.size else 0.0

    logger.info("\t max abs: %s", max_abs)
    logger.info("\t mean abs: %s", mean_abs)
    logger.info("\t max rel: %.4f %%", max_rel_pct)
    logger.info("\t mean rel: %.4f %%", mean_rel_pct)

    return (max_abs, mean_abs, max_rel_pct, mean_rel_pct)
