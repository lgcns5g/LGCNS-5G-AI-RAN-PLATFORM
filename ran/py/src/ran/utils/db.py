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

"""Decibel utility: db(x) = 10 * log10(x).

Accepts scalars or array-like inputs and returns float64 NumPy arrays.
"""

import numpy as np
from numpy.typing import ArrayLike

from ran.types import FloatArrayNP, FloatNP


def db(x: ArrayLike) -> FloatArrayNP:
    """Compute 10*log10(x) elementwise.

    Parameters
    ----------
    x : array-like
        Linear-scale input. Must be positive to avoid -inf results.

    Returns
    -------
    np.ndarray
        Decibel-scale output as float64.
    """
    arr = np.asarray(x, dtype=FloatNP)
    return 10.0 * np.log10(arr)


__all__ = ["db"]
