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
Soft demapper (NumPy translation of softDemapper.m_bits legacy path).

Implements max-log LLRs per layer using PAM decomposition for QAM.
Handles TdiMode selection of N0 from Ree and symbol-range expansion
when numDmrsCdmGrpsNoData == 1.
"""

from typing import TYPE_CHECKING

import numpy as np

from ran.constants import LLR_CLAMP_ABS
from ran.types import (
    ComplexArrayNP,
    ComplexNP,
    FloatArrayNP,
    FloatNP,
    IntNP,
)

if TYPE_CHECKING:
    from ran.types import (
        IntArrayNP,
    )


def _simplified_llr_mapping(
    i_axis: FloatArrayNP,
    q_axis: FloatArrayNP,
    qam_bits: int,
) -> FloatArrayNP:
    """Piecewise simplified LLR mapping (pre-variance scaling).

    Args:
        i_axis: real axis values shaped (L, F, T)
        q_axis: imag axis values shaped (L, F, T) (ignored for BPSK)
        qam_bits: number of bits per QAM symbol

    Returns
    -------
        LLR tensor shaped (qam_bits, L, F, T), scaled by the constellation factor only.

    Axis convention
    ---------------
    - L: layers (UE streams), size = nl
    - F: frequency bins/subcarriers in allocated band, size = n_f_alloc (e.g., 12*nPrb)
    - T: time symbols (OFDM symbols in the demapped interval), size = n_sym
    """
    n_l, n_f_alloc, n_t = i_axis.shape
    llr = np.zeros((qam_bits, n_l, n_f_alloc, n_t), dtype=FloatNP)

    if qam_bits == 8:  # 256QAM  # noqa: PLR2004
        a = FloatNP(1.0 / np.sqrt(170.0))
        llr[0] = i_axis
        llr[2] = -np.abs(i_axis) + 8.0 * a
        llr[4] = -np.abs(np.abs(i_axis) - 8.0 * a) + 4.0 * a
        llr[6] = -np.abs(np.abs(np.abs(i_axis) - 8.0 * a) - 4.0 * a) + 2.0 * a
        llr[1] = q_axis
        llr[3] = -np.abs(q_axis) + 8.0 * a
        llr[5] = -np.abs(np.abs(q_axis) - 8.0 * a) + 4.0 * a
        llr[7] = -np.abs(np.abs(np.abs(q_axis) - 8.0 * a) - 4.0 * a) + 2.0 * a
        llr *= 2.0 * a
    elif qam_bits == 6:  # 64QAM  # noqa: PLR2004
        a = FloatNP(1.0 / np.sqrt(42.0))
        llr[0] = i_axis
        llr[2] = -np.abs(i_axis) + 4.0 * a
        llr[4] = -np.abs(np.abs(i_axis) - 4.0 * a) + 2.0 * a
        llr[1] = q_axis
        llr[3] = -np.abs(q_axis) + 4.0 * a
        llr[5] = -np.abs(np.abs(q_axis) - 4.0 * a) + 2.0 * a
        llr *= 2.0 * a
    elif qam_bits == 4:  # 16QAM  # noqa: PLR2004
        a = FloatNP(1.0 / np.sqrt(10.0))
        llr[0] = i_axis
        llr[2] = -np.abs(i_axis) + 2.0 * a
        llr[1] = q_axis
        llr[3] = -np.abs(q_axis) + 2.0 * a
        llr *= 2.0 * a
    elif qam_bits == 2:  # QPSK  # noqa: PLR2004
        a = FloatNP(1.0 / np.sqrt(2.0))
        llr[0] = i_axis
        llr[1] = q_axis
        llr *= 2.0 * a
    elif qam_bits == 1:  # BPSK
        a = FloatNP(1.0)
        llr[0] = i_axis
        llr *= 2.0 * a
    else:
        msg = "qam_bits must be one of {1, 2, 4, 6, 8}"
        raise ValueError(msg)

    return llr


def soft_demapper(x: ComplexArrayNP, ree: FloatArrayNP, qam_bits: int) -> FloatArrayNP:
    """
    Simplified cuPHY-style soft demapper (vectorized), 100%-matching tables.

    Args:
        x: estimated symbols on allocated tones, shape (n_f_alloc, n_sym, n_layers)
        ree: noise variance per layer/tone (optional sym axis),
                   shape (n_layers, n_f_alloc) or (n_layers, n_f_alloc, 1)
        qam_bits: QAM modulation order (number of bits per symbol). One of {1, 2, 4, 6, 8}.

    Returns
    -------
        LLR tensor of shape (qam_bits, nl, n_f_alloc, n_sym) = (qam_bits, L, F, T)
    """
    n_f_alloc = x.shape[0]

    # Gather N0 per layer and tone and drop optional sym axis
    n0 = ree
    if n0.ndim == 3:  # noqa: PLR2004
        n0 = n0[..., 0]
    # pam variance per (L, F)
    pam_var = np.maximum(n0, np.finfo(FloatNP).tiny) / 2.0

    # BPSK: Undo pi/2 rotation per frequency index position
    if qam_bits == 1:
        k: IntArrayNP = np.arange(n_f_alloc, dtype=IntNP)
        phase_even = np.exp(-1j * np.pi / 4.0)
        phase_odd = np.exp(-1j * 3.0 * np.pi / 4.0)
        phase = np.where((k & 1) == 1, phase_odd, phase_even).astype(ComplexNP)
        x = x * phase[:, None, None]

    # Real/imag axes in (L, F, T)
    x_transposed = x.transpose(2, 0, 1)
    i_axis = np.real(x_transposed)
    q_axis = np.imag(x_transposed) if qam_bits > 1 else np.zeros_like(i_axis)

    # Piecewise simplified mapping per QAM size (pre-variance scaling)
    llr = _simplified_llr_mapping(i_axis=i_axis, q_axis=q_axis, qam_bits=qam_bits)

    # Scale by 1 / PAM variance (equals 2 / N0) & clamp to LLR_CLAMP_ABS
    return np.clip(llr / pam_var[None, :, :, None], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)


__all__ = ["soft_demapper"]
