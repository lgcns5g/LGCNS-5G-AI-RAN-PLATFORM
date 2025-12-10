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

"""PHY modulation and demodulation utilities."""

import numpy as np

from ran.types import ComplexArrayNP, IntArrayNP, IntNP


def qpsk_map(
    c: IntArrayNP | list[int],
    m: int,
) -> ComplexArrayNP:
    """
    Map 2*m binary chips into QPSK symbols with unit average power.

    Args:
        c: Gold sequence chips; length must equal 2*m. Accepts list/array-like.
        m: number of QPSK symbols to produce

    Returns
    -------
        Complex array of shape (m,) with dtype complex128
    """
    c_arr = np.asarray(c)
    if c_arr.size != 2 * m:
        msg = f"Gold sequence length {c_arr.size} != 2*m={2 * m}"
        raise ValueError(msg)
    bits = c_arr.astype(IntNP, copy=False)
    pairs = bits.reshape(-1, 2)
    real_part = 1 - 2 * pairs[:, 0]
    imag_part = 1 - 2 * pairs[:, 1]
    return (real_part + 1j * imag_part) / np.sqrt(2.0)


__all__ = ["qpsk_map"]
