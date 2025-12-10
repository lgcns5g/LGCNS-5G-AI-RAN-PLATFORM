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

"""Unit tests for slice-based LS channel estimation.

Targets `ran.phy.numpy.pusch.channel_estimation.channel_est_ls` using allocation-local
DMRS slices built via `embed_dmrs_ul` and raw DMRS extraction via
`extract_raw_dmrs_type_1`.
"""

from typing import TYPE_CHECKING

import numpy as np

from ran.phy.numpy.pusch import channel_est_ls, embed_dmrs_ul
from ran.phy.numpy.pusch.dmrs_utils import extract_raw_dmrs_type_1
from ran.types import ComplexNP, IntNP

if TYPE_CHECKING:
    from ran.types import ComplexArrayNP, IntArrayNP


def test_ls_single_prb_unit_symbols() -> None:
    """LS on a single-PRB allocation with unit DMRS and unit received symbols."""
    n_prb = 1
    n_dmrs_sym = 2
    n_ant = 1
    nl = 1

    # Base grid, no OCC; SCID=0
    port_idx = np.array([0], dtype=IntNP)
    vec_scid = np.array([0], dtype=IntNP)

    # Build PRB-band DMRS slice X (tx reference embedded locally)
    r_dmrs_band: ComplexArrayNP = np.ones((6 * n_prb, n_dmrs_sym, 2), dtype=ComplexNP)
    x_dmrs = embed_dmrs_ul(
        r_dmrs=r_dmrs_band,
        nl=nl,
        port_idx=port_idx,
        vec_scid=vec_scid,
        energy=1.0,
    )

    # Build raw received DMRS slice Y from unit grid on DMRS symbols
    xtf_band_dmrs: ComplexArrayNP = np.ones((12 * n_prb, n_dmrs_sym, n_ant), dtype=ComplexNP)
    y_dmrs = extract_raw_dmrs_type_1(
        xtf_band_dmrs=xtf_band_dmrs,
        nl=nl,
        port_idx=port_idx,
    )

    h_compact = channel_est_ls(x_dmrs=x_dmrs, y_dmrs=y_dmrs)

    assert h_compact.shape == (6 * n_prb, nl, n_ant)
    assert h_compact.dtype == ComplexNP
    np.testing.assert_allclose(h_compact, 1.0 + 0.0j, rtol=0, atol=0)


def test_ls_multi_prb_unit_symbols() -> None:
    """LS on a 5-PRB allocation with unit DMRS and received symbols returns ones."""
    n_prb = 5
    n_dmrs_sym = 2
    n_ant = 1
    nl = 1

    port_idx = np.array([0], dtype=IntNP)
    vec_scid = np.array([0], dtype=IntNP)

    r_dmrs_band: ComplexArrayNP = np.ones((6 * n_prb, n_dmrs_sym, 2), dtype=ComplexNP)
    x_dmrs = embed_dmrs_ul(
        r_dmrs=r_dmrs_band,
        nl=nl,
        port_idx=port_idx,
        vec_scid=vec_scid,
        energy=1.0,
    )

    xtf_band_dmrs: ComplexArrayNP = np.ones((12 * n_prb, n_dmrs_sym, n_ant), dtype=ComplexNP)
    y_dmrs = extract_raw_dmrs_type_1(
        xtf_band_dmrs=xtf_band_dmrs,
        nl=nl,
        port_idx=port_idx,
    )

    h_compact = channel_est_ls(x_dmrs=x_dmrs, y_dmrs=y_dmrs)

    assert h_compact.shape == (6 * n_prb, nl, n_ant)
    np.testing.assert_allclose(h_compact, 1.0 + 0.0j, rtol=0, atol=0)


def test_ls_even_row_selection() -> None:
    """LS compacts even-row DMRS bins as expected."""
    n_prb = 1
    n_dmrs_sym = 2
    n_ant = 1
    nl = 1

    # grid = 0 (even-row DMRS)
    port_idx = np.array([0], dtype=IntNP)
    vec_scid = np.array([0], dtype=IntNP)

    # Build X (transmitted signal) with unit DMRS embedded at even rows
    r_dmrs_band: ComplexArrayNP = np.ones((6 * n_prb, n_dmrs_sym, 2), dtype=ComplexNP)
    x_dmrs = embed_dmrs_ul(
        r_dmrs=r_dmrs_band,
        nl=nl,
        port_idx=port_idx,
        vec_scid=vec_scid,
        energy=1.0,
    )

    # Build Y by placing value 2.0 on even-row DMRS only
    xtf_band_dmrs: ComplexArrayNP = np.zeros((12 * n_prb, n_dmrs_sym, n_ant), dtype=ComplexNP)
    dmrs_re_indices: IntArrayNP = 2 * np.arange(6, dtype=IntNP)  # even rows for grid=0
    xtf_band_dmrs[dmrs_re_indices, :, 0] = 2.0 + 0.0j
    y_dmrs = extract_raw_dmrs_type_1(
        xtf_band_dmrs=xtf_band_dmrs,
        nl=nl,
        port_idx=port_idx,
    )

    h_compact = channel_est_ls(x_dmrs=x_dmrs, y_dmrs=y_dmrs)

    np.testing.assert_allclose(h_compact, 2.0 + 0.0j, rtol=0, atol=0)
