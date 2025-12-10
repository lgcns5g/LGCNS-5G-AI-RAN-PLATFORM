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

"""Tests for derate_match (NumPy implementation)."""

import numpy as np

from ran.constants import LLR_CLAMP_ABS, SC_PER_PRB
from ran.phy.numpy.pusch.derate_matcher import derate_match
from ran.types import FloatNP, IntNP


class TestDerateMatch:
    """Unit tests for derate_match function."""

    def test_shapes_types_and_clamping_with_filler(self) -> None:
        """Output shapes, dtype, clamping, and filler region behavior."""
        # Parameters (BGN=2 for smaller tables), small Zc to keep arrays small
        bgn = 2
        zc = 2
        c = 2
        nl = 1
        qam_bits = 4  # 16QAM
        k = 40
        f = 8
        k_prime = 20  # ensure filler region exists
        rv_idx = 0
        nref = 0  # use full n = Zc * 50
        g = SC_PER_PRB  # total bits across CBs

        # E allocation => e = [4, 8] (since nl*qam_bits=4, c=2, g=12)
        n_total = g
        # Create llr_descr as (N_total, 1) Fortran layout source (force clamp)
        llr_descr: np.ndarray = np.full((n_total, 1), 2.0 * LLR_CLAMP_ABS, dtype=FloatNP)

        derate_cbs, nv_parity, idx, sizes = derate_match(
            llr_descr=llr_descr,
            bgn=bgn,
            c=c,
            qam_bits=qam_bits,
            k=k,
            f=f,
            k_prime=k_prime,
            zc=zc,
            nl=nl,
            rv_idx=rv_idx,
            nref=nref,
            g=g,
        )

        # Shapes and dtypes
        n = zc * 50
        assert derate_cbs.shape == (n, c)
        assert idx.shape == (n, c)
        assert sizes.shape == (c,)
        assert derate_cbs.dtype == FloatNP
        assert idx.dtype == IntNP
        assert sizes.dtype == IntNP

        # Clamping
        assert np.all(derate_cbs <= LLR_CLAMP_ABS)
        assert np.all(derate_cbs >= -LLR_CLAMP_ABS)

        # Filler region [k' - 2Zc, k - 2Zc)
        start_fill = k_prime - 2 * zc
        end_fill = k - 2 * zc
        if end_fill > start_fill:
            fill_slice = derate_cbs[start_fill:end_fill, :]
            assert fill_slice.size > 0
            np.testing.assert_allclose(fill_slice, LLR_CLAMP_ABS, rtol=0, atol=0)

        # nv_parity bounds for BGN=2
        min_nv_parity = 4
        max_nv_parity = 42
        assert min_nv_parity <= nv_parity <= max_nv_parity

    def test_energy_conservation_no_filler_no_clamp(self) -> None:
        """Without clamp/filler, sum of placed LLRs equals sum of inputs per CB."""
        bgn = 2
        zc = 2
        c = 2
        nl = 1
        qam_bits = 4
        k = 40
        f = 8
        k_prime = k  # no filler
        rv_idx = 0
        nref = 0
        g = SC_PER_PRB  # e = (4,8)

        # Build llr_descr sequential values below clamp to avoid saturation
        n_total = g
        llr_descr: np.ndarray = (np.arange(1, n_total + 1, dtype=FloatNP)[:, None]) / LLR_CLAMP_ABS

        derate_cbs, _, _, _ = derate_match(
            llr_descr=llr_descr,
            bgn=bgn,
            c=c,
            qam_bits=qam_bits,
            k=k,
            f=f,
            k_prime=k_prime,
            zc=zc,
            nl=nl,
            rv_idx=rv_idx,
            nref=nref,
            g=g,
        )

        # e allocation
        e = (4, 8)
        # Sum of non-zero entries per column equals sum of corresponding inputs
        # Column 0 uses first 4 numbers: 1..4
        # Column 1 uses next 8 numbers: 5..12
        col0_sum_in = float(np.sum(llr_descr[: e[0], 0]))
        col1_sum_in = float(np.sum(llr_descr[e[0] : e[0] + e[1], 0]))
        col0_sum_out = float(np.sum(derate_cbs[:, 0]))
        col1_sum_out = float(np.sum(derate_cbs[:, 1]))
        np.testing.assert_allclose(col0_sum_out, col0_sum_in, rtol=0, atol=1e-12)
        np.testing.assert_allclose(col1_sum_out, col1_sum_in, rtol=0, atol=1e-12)

    def test_indices_within_expected_range(self) -> None:
        """Returned indices must lie within [1+offset, n_cb+offset]."""
        bgn = 2
        zc = 2
        c = 2
        nl = 1
        qam_bits = 4
        k = 40
        f = 8
        k_prime = k
        rv_idx = 0
        nref = 0
        g = 12
        n_total = g
        llr_descr: np.ndarray = np.ones((n_total, 1), dtype=FloatNP)

        _, _, idx, _ = derate_match(
            llr_descr=llr_descr,
            bgn=bgn,
            c=c,
            qam_bits=qam_bits,
            k=k,
            f=f,
            k_prime=k_prime,
            zc=zc,
            nl=nl,
            rv_idx=rv_idx,
            nref=nref,
            g=g,
        )

        offset = 2 * zc - 1
        n_cb = zc * 50
        # Consider only first sum(e) rows (others are zero)
        e = (4, 8)
        used0 = idx[: e[0], 0]
        used1 = idx[: e[1], 1]
        assert np.all(used0 >= 1 + offset)
        assert np.all(used0 <= n_cb + offset)
        assert np.all(used1 >= 1 + offset)
        assert np.all(used1 <= n_cb + offset)

    def test_rv_variants_bg2_change_k0(self) -> None:
        """RV indices should alter distribution (k0) across Ncb."""
        bgn = 2
        zc = 2
        c = 1
        nl = 1
        qam_bits = 4
        k = 40
        f = 8
        k_prime = k
        nref = 0
        g = 8  # e = [8]
        llr_descr: np.ndarray = np.arange(1, g + 1, dtype=FloatNP)[:, None]
        idx_sets = []
        rv_list = (0, 1, 2, 3)
        for rv in rv_list:
            _, _, idx, sizes = derate_match(
                llr_descr=llr_descr,
                bgn=bgn,
                c=c,
                qam_bits=qam_bits,
                k=k,
                f=f,
                k_prime=k_prime,
                zc=zc,
                nl=nl,
                rv_idx=rv,
                nref=nref,
                g=g,
            )
            idx_sets.append(idx[: sizes[0], 0].copy())
        # RV variants produce different starting indices
        assert len({int(s[0]) for s in idx_sets}) == len(rv_list)

    def test_clamp_thresholds_extreme_llrs(self) -> None:
        """Clamp thresholds apply for extreme LLRs."""
        bgn = 2
        zc = 2
        c = 1
        nl = 1
        qam_bits = 4
        k = 40
        f = 8
        k_prime = k
        nref = 0
        g = 8
        llr_descr: np.ndarray = np.array(
            [1e20, -1e20, 1e6, -1e6, 5000, -5000, 10001, -10001],
            dtype=FloatNP,
        )[:, None]
        derate_cbs, _, _, _ = derate_match(
            llr_descr=llr_descr,
            bgn=bgn,
            c=c,
            qam_bits=qam_bits,
            k=k,
            f=f,
            k_prime=k_prime,
            zc=zc,
            nl=nl,
            rv_idx=0,
            nref=nref,
            g=g,
        )
        assert np.all(derate_cbs <= LLR_CLAMP_ABS)
        assert np.all(derate_cbs >= -LLR_CLAMP_ABS)
