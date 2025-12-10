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
DMRS helper utilities shared by LS and covariance paths.

This module centralizes small, well-tested building blocks to construct the
"transmitted DMRS at REs" reference and the indices/patterns needed by both
Least-Squares (LS) channel estimation and covariance estimation.

Design goals
------------
- Single source of truth for DMRS indices, OCC patterns, and per-layer grid
  selection rules so LS and covariance remain in lockstep.
- Broadcast-friendly outputs to avoid unnecessary ``np.tile`` allocations.
- Explicit shapes in docstrings for easy integration and review.
"""

import numpy as np

from ran.phy.numpy.utils import focc_dmrs, tocc_dmrs
from ran.types import ComplexArrayNP, IntArrayNP, IntNP


def dmrs_even_rows_from_lc(lc: int) -> IntArrayNP:
    """Return even-row indices for DMRS tones given lc=6*n_prb."""
    return 2 * np.arange(lc, dtype=IntNP)


def dmrs_focc_pattern(nf: int, n_sym: int) -> ComplexArrayNP:
    """Frequency-domain OCC pattern tiled to (lc, n_sym)."""
    return focc_dmrs(nf)[:, None].repeat(n_sym, axis=1)


def dmrs_tocc_pattern(nf: int, n_sym: int) -> ComplexArrayNP:
    """Time-domain OCC pattern tiled to (lc, n_sym)."""
    return tocc_dmrs(n_sym)[None, :].repeat(nf, axis=0)


def parse_port_cfg(port_idx: IntArrayNP) -> tuple[IntArrayNP, IntArrayNP, IntArrayNP]:
    """Parse per-layer DMRS config bitfields from ``port_idx``.

    Parameters
    ----------
    port_idx : ndarray of int64, shape (nl,)
        Per-layer bitfields with the following layout (LS convention):
          - bit0: fOCC enable (0/1)
          - bit1: grid select (0 even, 1 odd) - determines DMRS grid offset
          - bit2: tOCC enable (0/1)

    Returns
    -------
    focc_cfg : ndarray of int64, shape (nl,)
    grid_cfg : ndarray of int64, shape (nl,)
    tocc_cfg : ndarray of int64, shape (nl,)
    """
    focc_cfg = (port_idx & 0b001).astype(IntNP)
    grid_cfg = ((port_idx & 0b010) >> 1).astype(IntNP)
    tocc_cfg = ((port_idx & 0b100) >> 2).astype(IntNP)
    return focc_cfg, grid_cfg, tocc_cfg


def layer_grid_offsets(port_idx: IntArrayNP, grid_table: IntArrayNP) -> IntArrayNP:
    """Return per-layer grid frequency offset ``delta`` from ``grid_table``.

    Parameters
    ----------
    port_idx : ndarray of int64, shape (nl,)
        0-based layer bitfields (index into ``grid_table``).
    grid_table : ndarray of int64, shape (>= max(port_idx)+1,)
        Lookup table mapping port index to a grid offset (typically 0 or 1).

    Returns
    -------
    deltas : ndarray of int64, shape (nl,)
        Per-layer offsets to be added to PRB-band indices.
    """
    return grid_table[port_idx.astype(IntNP)]


def build_layer_freq_indices(
    freq_idx_dmrs_local: IntArrayNP,
    port_idx: IntArrayNP,
    grid_table: IntArrayNP,
) -> IntArrayNP:
    """Build per-layer PRB-band DMRS row indices with grid offsets.

    Parameters
    ----------
    freq_idx_dmrs_local : ndarray of int64, shape (lc,)
        Local PRB-band DMRS rows (even subcarriers only).
    port_idx : ndarray of int64, shape (nl,)
        0-based per-layer bitfields (index into ``grid_table``).
    grid_table : ndarray of int64, shape (>= max(port_idx)+1,)
        Lookup table mapping port index to a grid offset (0/1).

    Returns
    -------
    layer_freq : ndarray of int64, shape (lc, nl)
        Per-layer indices inside the PRB band after applying per-layer grid
        offsets. To get absolute subcarrier rows, add ``base = 12*start_prb``.
    """
    deltas = layer_grid_offsets(port_idx=port_idx, grid_table=grid_table)
    return (freq_idx_dmrs_local[:, None] + deltas[None, :]).astype(IntNP)


def embed_dmrs_ul(
    r_dmrs: ComplexArrayNP,
    nl: int,
    port_idx: IntArrayNP,
    vec_scid: IntArrayNP,
    energy: float,
) -> ComplexArrayNP:
    """Construct PRB-band DMRS slice for UL (allocation-local embedding).

    Parameters
    ----------
    r_dmrs : ndarray of complex128, shape (12*n_prb, n_dmrs_sym, 2)
        DMRS references already trimmed to band rows and DMRS symbols.
    nl : int
        Number of layers.
    port_idx : ndarray of int64, shape (nl,)
        Per-layer bitfields (bit0: fOCC, bit1: grid select, bit2: tOCC).
    vec_scid : ndarray of int64, shape (nl,)
        Per-layer SCID selector in {0, 1}.
    energy : float
        Energy scaling factor to apply to the embedded DMRS.

    Returns
    -------
    x_dmrs : ndarray of complex128, shape (12*n_prb, n_sym, nl)
        PRB-band DMRS slice. Only DMRS REs are non-zero; other tones in the
        PRB band are zero.
    """
    lc, n_sym = r_dmrs.shape[:2]
    prb_len = 2 * lc

    # Initialize PRB-band grid (freq x dmrs_sym x layer)
    x_dmrs = np.zeros((prb_len, n_sym, nl), dtype=r_dmrs.dtype)

    # Indices and OCC patterns (broadcast-friendly)
    freq_idx_dmrs_local = dmrs_even_rows_from_lc(lc)
    focc = dmrs_focc_pattern(lc, n_sym)
    tocc = dmrs_tocc_pattern(lc, n_sym)

    # Energy scaling
    r_scaled = r_dmrs * np.sqrt(energy)

    # Per-layer enables and grid offsets from bitfields
    focc_cfg, grid_cfg, tocc_cfg = parse_port_cfg(port_idx)

    for layer in range(nl):
        scid = vec_scid[layer]
        r_sel = r_scaled[..., scid]
        r_l = focc * r_sel if focc_cfg[layer] != 0 else r_sel
        r_l = tocc * r_l if tocc_cfg[layer] != 0 else r_l
        freq_ix = (freq_idx_dmrs_local + grid_cfg[layer]).astype(IntNP)
        x_dmrs[freq_ix, :, layer] = r_l

    return x_dmrs


def extract_raw_dmrs_type_1(
    xtf_band_dmrs: ComplexArrayNP,
    nl: int,
    port_idx: IntArrayNP,
) -> ComplexArrayNP:
    """Extract received Type-1 DMRS REs only (no processing, no combining).

    Parameters
    ----------
    xtf_band_dmrs : ndarray of complex128, shape (n_f, n_dmrs_sym, n_ant)
        PRB-band received TF grid DMRS-only slice (already trimmed along time).
    nl : int
        Number of layers.
    port_idx : ndarray of int64, shape (nl,)
        Per-layer bitfields (bit1 selects grid: 0 even, 1 odd).

    Returns
    -------
    y_dmrs : ndarray of complex128, shape (n_f, n_dmrs_sym, nl, n_ant)
        PRB-band slice with only DMRS REs populated per layer; others are zero.

    Note: Assumes even number of frequency bins (i.e. subcarriers, or n_f).
    """
    n_f, n_dmrs_sym, n_ant = xtf_band_dmrs.shape

    if n_f <= 0 or n_dmrs_sym == 0:
        raise ValueError(f"Invalid input shape: {xtf_band_dmrs.shape}")

    # Even-bin local rows for compact DMRS positions inside PRB band
    freq_idx_dmrs: IntArrayNP = 2 * np.arange(n_f // 2, dtype=IntNP)

    # Per-layer grid select (0 even, 1 odd)
    _, grid_cfg, _ = parse_port_cfg(port_idx)

    # Initialize PRB-band container
    y_dmrs = np.zeros((n_f, n_dmrs_sym, nl, n_ant), dtype=xtf_band_dmrs.dtype)

    # Fill per layer at the proper grid (adds 0 or 1 to even rows)
    for layer in range(nl):
        freq_ix = (freq_idx_dmrs + grid_cfg[layer]).astype(IntNP)
        y_dmrs[freq_ix, :, layer, :] = xtf_band_dmrs[freq_ix, :, :]

    return y_dmrs


__all__ = [
    "build_layer_freq_indices",
    "embed_dmrs_ul",
    "extract_raw_dmrs_type_1",
    "layer_grid_offsets",
    "parse_port_cfg",
]
