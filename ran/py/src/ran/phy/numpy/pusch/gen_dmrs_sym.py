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
NumPy implementation of MATLAB-style PUSCH DMRS generation utilities.

This module mirrors the logic used in the MATLAB pipeline for building
frequency-domain DMRS sequences and their underlying Gold sequences.
It is designed to be compatible with the channel estimation code that
consumes these sequences (see `_apply_chest_ls_main.py`).

Key functions:
- gen_dmrs_sym: produce per-symbol, per-SCID DMRS resources
"""

import numpy as np

from ran.phy.numpy.utils import gold_sequence, qpsk_map
from ran.types import ComplexArrayNP, ComplexNP, IntArrayNP, IntNP
from ran.constants import N_SYM_PER_SLOT


def gen_dmrs_sym(
    slot_number: int,
    n_f: int,
    n_dmrs_id: int,
    sym_idx_dmrs: IntArrayNP | None = None,
    *,
    n_t: int = 14,
) -> tuple[ComplexArrayNP, IntArrayNP]:
    """Generate DMRS symbols and scrambling sequences.

    Parameters
    ----------
    slot_number : int
        Integer slot number
    n_f : int
        Length of Gold sequence per port (must be even)
    n_dmrs_id : int
        DMRS identity (integer)
    sym_idx_dmrs : IntArrayNP | None, optional
        0-based indices of DMRS symbols to generate.
        Alternatively, provide n_t (number of OFDM symbols) to generate indices [0, 1, ..., n_t-1].
        Exactly one of sym_idx_dmrs or n_t must be provided.
    n_t : int, default=14
        Number of OFDM symbols

    Returns
    -------
    r_dmrs : ComplexArrayNP
        Complex array of shape (n_f//2, n_sym, 2), with 2 = number of SCIDs
    scr_seq : IntArrayNP
        Integer array of shape (n_f, n_sym, 2)
    """
    if (n_f % 2) != 0:
        msg = "n_f must be even to form complex DMRS from Gold sequence"
        raise ValueError(msg)

    if sym_idx_dmrs is None:
        sym_idx_dmrs = np.arange(n_t, dtype=IntNP)

    t_idx_vec: IntArrayNP = sym_idx_dmrs.astype(IntNP) + 1

    n_sym = t_idx_vec.size

    r_dmrs: ComplexArrayNP = np.empty((n_f // 2, n_sym, 2), dtype=ComplexNP)
    scr_seq: IntArrayNP = np.empty((n_f, n_sym, 2), dtype=IntNP)

    # n_scid_vec: 1D array containing the two possible scrambling identities (SCID)
    n_scid_vec = np.array([0, 1], dtype=IntNP)

    # Compute c_init for all (t_idx, n_scid) pairs: shape (n_t, 2)
    c_init_mat = (
        (1 << 17) * (slot_number * N_SYM_PER_SLOT + t_idx_vec[:, None]) * (2 * n_dmrs_id + 1)
        + 2 * n_dmrs_id
        + n_scid_vec[None, :]
    ) % (1 << 31)

    # For each (t_idx, n_scid), generate Gold sequence and QPSK map
    for scid_idx in range(2):  # [0, 1]
        for t_idx in range(n_sym):
            c_init = c_init_mat[t_idx, scid_idx]
            c = gold_sequence(c_init, n_f)
            scr_seq[:, t_idx, scid_idx] = c
            r_dmrs[:, t_idx, scid_idx] = qpsk_map(c, n_f // 2)

    return r_dmrs, scr_seq
