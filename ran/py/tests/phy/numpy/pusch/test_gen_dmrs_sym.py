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

"""Unit tests for NumPy PUSCH DMRS utilities in `ran.phy.numpy.pusch.gen_dmrs_sym`.

These tests mirror the reference tests in `tests/phy_ref/pusch/test_gen_dmrs_sym.py`
but target the NumPy implementation under `ran.phy.numpy.pusch`.
"""

from typing import TYPE_CHECKING

import numpy as np

from ran.constants import SC_PER_PRB
from ran.phy.numpy.pusch import gen_dmrs_sym
from ran.types import ComplexNP, IntNP

if TYPE_CHECKING:
    from ran.types import IntArrayNP


class TestGenDmrsSymNP:
    """Tests for the gen_dmrs_sym function (NumPy version)."""

    def test_basic_generation_shapes_and_types(self) -> None:
        """Test dtype and shape of DMRS symbols and scrambling sequences."""
        slot_number = 0
        n_f = SC_PER_PRB * 3
        n_t = 14
        n_id = 0
        sym_idx_dmrs: IntArrayNP = np.arange(n_t, dtype=IntNP)

        r_dmrs, scr = gen_dmrs_sym(
            slot_number=slot_number, n_f=n_f, n_dmrs_id=n_id, sym_idx_dmrs=sym_idx_dmrs
        )
        assert r_dmrs.shape == (n_f // 2, n_t, 2)
        assert scr.shape == (n_f, n_t, 2)
        assert r_dmrs.dtype == ComplexNP
        assert scr.dtype == IntNP

    def test_scrambling_is_binary(self) -> None:
        """Test that the scrambling sequence is binary."""
        sym_idx_dmrs: IntArrayNP = np.arange(7, dtype=IntNP)
        _, scr = gen_dmrs_sym(slot_number=1, n_f=12 * 2, n_dmrs_id=17, sym_idx_dmrs=sym_idx_dmrs)
        assert np.all(np.isin(scr, [0, 1]))

    def test_qpsk_magnitude(self) -> None:
        """Test that the DMRS symbols have the correct magnitude (unit modulus)."""
        sym_idx_dmrs: IntArrayNP = np.arange(2, dtype=IntNP)
        r_dmrs, _ = gen_dmrs_sym(slot_number=2, n_f=12 * 1, n_dmrs_id=3, sym_idx_dmrs=sym_idx_dmrs)
        magnitudes = np.abs(r_dmrs)
        np.testing.assert_allclose(magnitudes, 1.0, rtol=1e-10)

    def test_different_scid_columns_differ(self) -> None:
        """Test that the DMRS symbols differ for different SCIDs."""
        n_f = SC_PER_PRB * 4
        n_t = 4
        sym_idx_dmrs: IntArrayNP = np.arange(n_t, dtype=IntNP)
        _, scr = gen_dmrs_sym(slot_number=3, n_f=n_f, n_dmrs_id=5, sym_idx_dmrs=sym_idx_dmrs)
        equal_mask = scr[:, :, 0] == scr[:, :, 1]
        assert np.any(~equal_mask)  # not all positions should be equal
