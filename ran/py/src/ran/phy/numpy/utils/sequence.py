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

"""PHY sequence utilities: generation of Gold sequence."""

import numpy as np

from ran.types import IntArrayNP, IntNP


def gold_sequence(c_init: int, n: int) -> IntArrayNP:
    """
    Build a Gold sequence of length `n` using the same logic as the MATLAB version.

    Inputs:
    - c_init: initial seed to Gold sequence (integer)
    - n: length of desired Gold sequence

    Output:
    - c: NumPy array of shape (n,) with 0/1 entries
    """
    nc = 1600  # Offset for modulo-2 sum of the two M-sequences

    length_needed = n + nc + 31
    x1: IntArrayNP = np.zeros(length_needed + 1, dtype=IntNP)
    x2: IntArrayNP = np.zeros(length_needed + 1, dtype=IntNP)

    x1[1] = 1

    bits_str = np.binary_repr(int(c_init) % (1 << 31), width=31)
    x2_init = np.fromiter((int(ch) for ch in bits_str[::-1]), dtype=IntNP)
    x2[1 : 31 + 1] = x2_init

    span = n + nc - 31
    for idx in range(1, span + 1):
        x1[idx + 31] = (x1[idx + 3] + x1[idx]) & 1
        x2[idx + 31] = (x2[idx + 3] + x2[idx + 2] + x2[idx + 1] + x2[idx]) & 1

    c: IntArrayNP = np.empty(n, dtype=IntNP)
    for idx in range(1, n + 1):
        c[idx - 1] = (x1[idx + nc] + x2[idx + nc]) & 1

    return c


__all__ = ["gold_sequence"]
