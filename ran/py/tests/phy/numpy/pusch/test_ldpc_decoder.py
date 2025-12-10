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

"""Tests for LDPC decode (NumPy implementation)."""

from typing import TYPE_CHECKING

import numpy as np

from ran.phy.numpy.pusch import ldpc_decode
from ran.types import FloatNP, IntNP

if TYPE_CHECKING:
    from ran.types import FloatArrayNP


def test_shapes_and_all_ones_for_zero_llrs_bg2() -> None:
    """BG2: zero LLRs decode to hard ones; verify shapes and values."""
    zc = 4
    c = 2
    bgn = 2
    i_ls = 1
    # Per implementation: nV=52 for BG2 → rows = zc*(nV-2)
    n_v = 52
    rows = zc * (n_v - 2)
    derate_cbs: FloatArrayNP = np.zeros((rows, c), dtype=FloatNP)
    nv_parity = 10  # any valid index into normalization table
    max_itr = 5

    tb_out, num_itr = ldpc_decode(
        derate_cbs=derate_cbs,
        nv_parity=nv_parity,
        zc=zc,
        c=c,
        bgn=bgn,
        i_ls=i_ls,
        max_num_itr_cbs=max_itr,
    )

    # BG2: nV_sym = 10 → output shape (10*Zc, C)
    assert tb_out.shape == (10 * zc, c)
    # With zero APPs, all hard bits are 1
    np.testing.assert_array_equal(tb_out, np.ones_like(tb_out))
    # Iterations used == max_itr
    np.testing.assert_array_equal(num_itr, max_itr * np.ones((c,), dtype=IntNP))


def test_single_codeblock_scalar_vs_vector_max_iters() -> None:
    """C=1 returns length-1 num_itr; C>1 follows per-CB limits."""
    zc = 4
    c = 1
    bgn = 2
    i_ls = 1
    n_v = 52
    rows = zc * (n_v - 2)
    derate_cbs: FloatArrayNP = np.zeros((rows, c), dtype=FloatNP)
    nv_parity = 8
    max_itr = 3
    tb_out, num_itr = ldpc_decode(
        derate_cbs=derate_cbs,
        nv_parity=nv_parity,
        zc=zc,
        c=c,
        bgn=bgn,
        i_ls=i_ls,
        max_num_itr_cbs=max_itr,
    )
    assert tb_out.shape == (10 * zc, c)
    np.testing.assert_array_equal(tb_out, np.ones_like(tb_out))
    np.testing.assert_array_equal(num_itr, np.array([max_itr], dtype=IntNP))

    # C = 3, vector of max iterations
    c = 3
    derate_cbs2: FloatArrayNP = np.zeros((rows, c), dtype=FloatNP)
    max_itrs = 5
    _, num_itr2 = ldpc_decode(
        derate_cbs=derate_cbs2,
        nv_parity=nv_parity,
        zc=zc,
        c=c,
        bgn=bgn,
        i_ls=i_ls,
        max_num_itr_cbs=max_itrs,
    )
    np.testing.assert_array_equal(num_itr2, max_itrs)


def test_bg1_shape_and_zero_llrs() -> None:
    """BG1 path: output shape (22*Zc, C) and all ones for zero LLRs."""
    zc = 4
    c = 1
    bgn = 1
    i_ls = 1
    n_v = 68
    rows = zc * (n_v - 2)
    derate_cbs: FloatArrayNP = np.zeros((rows, c), dtype=FloatNP)
    nv_parity = 12
    max_itr = 4

    tb_out, num_itr = ldpc_decode(
        derate_cbs=derate_cbs,
        nv_parity=nv_parity,
        zc=zc,
        c=c,
        bgn=bgn,
        i_ls=i_ls,
        max_num_itr_cbs=max_itr,
    )

    assert tb_out.shape == (22 * zc, c)
    np.testing.assert_array_equal(tb_out, np.ones_like(tb_out))
    np.testing.assert_array_equal(num_itr, np.array([max_itr], dtype=IntNP))


def test_invalid_rows_raises() -> None:
    """Rows must equal Zc*(nV-2); otherwise reshape should fail."""
    zc = 4
    c = 1
    bgn = 2
    i_ls = 1
    n_v = 52
    rows = zc * (n_v - 2) - 1  # invalid
    derate_cbs: FloatArrayNP = np.zeros((rows, c), dtype=FloatNP)
    nv_parity = 10
    with np.testing.assert_raises(ValueError):
        _ = ldpc_decode(
            derate_cbs=derate_cbs,
            nv_parity=nv_parity,
            zc=zc,
            c=c,
            bgn=bgn,
            i_ls=i_ls,
            max_num_itr_cbs=2,
        )


def test_determinism() -> None:
    """Same inputs must produce identical outputs across runs."""
    zc = 4
    c = 2
    bgn = 2
    i_ls = 1
    n_v = 52
    rows = zc * (n_v - 2)
    rng = np.random.default_rng(123)
    derate_cbs: FloatArrayNP = rng.standard_normal((rows, c)).astype(FloatNP)
    nv_parity = 9

    out1 = ldpc_decode(
        derate_cbs=derate_cbs,
        nv_parity=nv_parity,
        zc=zc,
        c=c,
        bgn=bgn,
        i_ls=i_ls,
        max_num_itr_cbs=3,
    )
    out2 = ldpc_decode(
        derate_cbs=derate_cbs,
        nv_parity=nv_parity,
        zc=zc,
        c=c,
        bgn=bgn,
        i_ls=i_ls,
        max_num_itr_cbs=3,
    )
    for a, b in zip(out1, out2, strict=True):
        np.testing.assert_array_equal(a, b)
