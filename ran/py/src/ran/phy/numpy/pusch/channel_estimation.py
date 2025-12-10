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

"""NumPy-based PUSCH LS and delay-domain channel estimation (5G NR).

Implements efficient, test-aligned block routines for:
- channel_est_ls: Minimal LS on DMRS tones only (shape: 6*n_prb x nl x n_ant)
- channel_est_dd: LS + delay-domain truncation + interpolation
  (shape: 12*n_prb x nl x n_ant, allocation slice only)

Indexing convention:
- start_prb, sym_idx_dmrs, port_idx: 0-based
- xtf: (n_f, n_t, n_ant)
- r_dmrs: (n_f, n_t, 2)
- sym_idx_dmrs: (n_dmrs_sym,)
"""

import numpy as np

from ran.types import ComplexArrayNP


def channel_est_ls(x_dmrs: ComplexArrayNP, y_dmrs: ComplexArrayNP) -> ComplexArrayNP:
    """
    Core LS on precomputed PRB-band DMRS slices.

    Args
    ----
        x_dmrs: (12*n_prb, n_sym, nl)
        y_dmrs: (12*n_prb, n_sym, nl, n_ant)

    Returns
    -------
        h_compact : (6*n_prb, nl, n_ant)
            Compact LS per DMRS tone (averaged across DMRS symbols).
    """
    # Matched filter, keep even bins, average across DMRS symbols
    y_mf = np.conj(x_dmrs)[..., None] * y_dmrs  # (12*n_prb, n_sym, nl, n_ant)
    return y_mf[::2, :, :, :].mean(axis=1)  # (6*n_prb, nl, n_ant)


def _delay_domain_truncation(
    h_compact: ComplexArrayNP,
    trunc_ratio: float = 0.05,
) -> ComplexArrayNP:
    """
    IFFT/FFT refinement on compact LS tones WITH truncation.

    Args
    ----
        h_compact: (lc, nl, n_ant)
            Compact LS channel estimates with subcarriers on the first axis.
        trunc_ratio: Fraction (0.0 to 1.0) of the delay-domain taps to keep (e.g., 0.05 keeps 5%).

    Returns
    -------
        h_ref: (lc, nl, n_ant)
            Refined compact LS tones (after truncation in delay domain).
    """
    lc = h_compact.shape[0]
    trunc_len = max(1, round(trunc_ratio * lc))
    # Go to delay domain
    h_delay = np.fft.ifft(h_compact, n=lc, axis=0)
    # Zero out taps beyond trunc_len
    h_delay[trunc_len:, ...] = 0.0
    # Back to frequency domain
    return np.fft.fft(h_delay, n=lc, axis=0)


def _interpolate(
    h_compact: ComplexArrayNP,  # (lc, nl, n_ant)
) -> ComplexArrayNP:
    """
    Expand compact LS tones (every 2 subcarriers) to the allocation slice only.

    Args
    ----
        h_compact: (lc, nl, n_ant)
            Compact LS channel estimates with subcarriers on the first axis.

    Returns
    -------
        h_alloc: (2*lc, nl, n_ant)
            Allocation slice with each compact bin repeated into a 2-bin block.
            Caller is responsible for assigning this slice into the full grid.
    """
    # Repeat along subcarrier axis (first axis)
    return np.repeat(h_compact, 2, axis=0)


def channel_est_dd(x_dmrs: ComplexArrayNP, y_dmrs: ComplexArrayNP) -> ComplexArrayNP:
    """
    Delay-domain truncation + interpolation on precomputed DMRS slices.

    Args
    ----
        x_dmrs: (12*n_prb, n_sym, nl)
            Transmitted DMRS reference symbols, post scrambling & OCC (subcarrier, symbol, layer).

        y_dmrs: (12*n_prb, n_sym, nl, n_ant)
            Received demapped DMRS symbols (subcarrier, symbol, layer, antenna).

    Returns
    -------
        h_alloc : (12*n_prb, nl, n_ant, n_dmrs_sym)

    Steps:
      1) Compact LS on DMRS tones from slices.
      2) IFFT/FFT with delay-domain truncation.
      3) Interpolate to allocation (repeat along frequency).

      Note: n_dmrs_sym is the number of DMRS symbols per slot, currently only supporting 1.
    """
    h_compact = channel_est_ls(x_dmrs=x_dmrs, y_dmrs=y_dmrs)
    h_compact_ref = _delay_domain_truncation(h_compact, trunc_ratio=0.05)
    return _interpolate(h_compact_ref)[..., None]


__all__ = [
    "channel_est_dd",
    "channel_est_ls",
]
