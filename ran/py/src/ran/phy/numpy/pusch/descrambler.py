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

"""
Descramble PUSCH data LLR sequence using a Gold sequence mask.

This mirrors the MATLAB call in detPusch.m:
    [~, LLR_descr] = descramble_bits(LLRseq, N_id, n_rnti)

Rules:
- NumPy only
- snake_case names
- precise type hints with built-in containers

The descrambling sequence uses a Gold sequence c of length len(LLRseq)
with seed aligned to the common PUSCH data scrambling initialization.
"""

from ran.phy.numpy.utils import gold_sequence
from ran.types import FloatArrayNP, FloatNP


def descramble_bits(
    llrseq: FloatArrayNP,
    n_id: int,
    n_rnti: int,
) -> FloatArrayNP:
    """
    Descramble the soft-bit sequence using a Gold sequence mask.

    Args:
        llrseq: shape (N,) float64, concatenated LLRs across symbols
        n_id: cell/slot ID used in seed
        n_rnti: RNTI used in seed

    Returns
    -------
        llr_descr: shape (N,) float64, descrambled LLRs
    """
    # Seed for PUSCH data scrambling
    c_init = ((n_rnti << 15) + n_id) % (1 << 31)

    c = gold_sequence(c_init, llrseq.size)
    mask = 1 - 2 * c.astype(FloatNP)  # 0->+1, 1->-1
    return llrseq * mask


__all__ = ["descramble_bits"]
