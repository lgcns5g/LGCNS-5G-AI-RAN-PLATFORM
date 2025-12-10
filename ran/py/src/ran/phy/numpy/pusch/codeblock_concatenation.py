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

"""Block desegmentation (NumPy translation of block_desegment.m)."""

import numpy as np

from ran.phy.numpy.pusch.crc_decoder import crc_decode
from ran.types import FloatArrayNP, FloatNP, IntArrayNP, IntNP


def codeblock_concatenation(
    tb_cbs_est: FloatArrayNP,
    c: int,
    k_prime: int,
) -> tuple[FloatArrayNP, IntArrayNP]:
    """
    Remove filler and concatenate codeblocks (per-CB CRC24B removal only when C>1).

    Args
    ----
    tb_cbs_est : FloatArrayNP
        Estimated codeblock bits after LDPC decoding, shape (k', C) or larger.
        Each column corresponds to a codeblock, and rows correspond to bits.
    c : int
        Number of codeblocks in the transport block.
    k_prime : int
        Number of bits per codeblock after filler removal (including CRC24B if C > 1).
        Must satisfy 1 <= k_prime <= tb_cbs_est.shape[0].

    Returns
    -------
    tb_crc_est_vec : FloatArrayNP
        Concatenated transport block bits after CRC removal (if C > 1). (column-major!)
        Shape is ((k_prime-24)*C,) if C > 1, or (k_prime,) if C == 1.
    cb_err : IntArrayNP
        Per-codeblock CRC error flags. Shape is (C,) if C > 1, or (1,) if C == 1.
    """
    # Validate k' and trim filler (rows k'..end)
    if k_prime <= 0 or k_prime > tb_cbs_est.shape[0]:
        raise ValueError(f"k_prime must in [1, {tb_cbs_est.shape[0]}]. Got {k_prime}.")
    tb_cbs_est = tb_cbs_est[:k_prime, :c]  # (k', C)

    # Prepare outputs with precise types for both branches
    tb_crc_est: FloatArrayNP
    cb_err: IntArrayNP

    if c == 1:
        # No per-CB CRC in single-CB case—pass through
        cb_err = np.zeros(1, dtype=IntNP)
        tb_crc_est = tb_cbs_est  # passthrough
    else:
        # Remove CRC24B from each CB (k' includes 24-bit CB CRC)
        out_rows = k_prime - 24
        if out_rows <= 0:
            raise ValueError(f"k_prime ({k_prime}) must be >= 24 to remove CRC24B. Got {k_prime}.")
        tb_crc_est = np.empty((out_rows, c), dtype=FloatNP, order="F")
        cb_err = np.empty((c,), dtype=IntNP)

        # Per-column CRC decode
        for ci in range(c):
            x_wo_crc, err = crc_decode(tb_cbs_est[:, ci], "24B")
            tb_crc_est[:, ci] = x_wo_crc  # exact fit (out_rows,)
            cb_err[ci] = err

    # Concatenate column-major (MATLAB style)
    tb_crc_est_vec = tb_crc_est.ravel(order="F")
    return tb_crc_est_vec, cb_err


__all__ = ["codeblock_concatenation"]
