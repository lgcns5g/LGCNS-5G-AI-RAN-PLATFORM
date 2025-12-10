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

"""Unit tests for `ran.phy.numpy.pusch.post_eq_noisevar_sinr` (NumPy implementation)."""

import numpy as np

from ran.phy.numpy.pusch import post_eq_noisevar_sinr
from ran.types import FloatNP, IntNP


def test_shapes_and_basic_fullslot_single_prb() -> None:
    """3D Ree input, single PRB allocation, both layers to same UE."""
    nl, n_f, n_sym = 2, 12, 1

    # Constant Ree per layer across all tones/symbols
    ree: np.ndarray = np.zeros((nl, n_f, n_sym), dtype=FloatNP)
    ree[0, :, 0] = 2.0
    ree[1, :, 0] = 8.0

    layer2ue = np.array([0, 0], dtype=IntNP)  # both layers belong to UE0
    n_ue = 1

    nv_db, sinr_db = post_eq_noisevar_sinr(
        ree=ree,
        layer2ue=layer2ue,
        n_ue=n_ue,
    )

    # Expected: average over layers and allocated tones of 1/Ree
    alloc_ree = np.stack([ree[0, :, 0], ree[1, :, 0]], axis=0)
    exp_avg_lin = np.mean(1.0 / alloc_ree)  # scalar over layers and tones
    exp_nv_db = -10.0 * np.log10(exp_avg_lin)
    exp_sinr_db = -exp_nv_db

    assert nv_db.shape == (n_sym, n_ue)
    assert sinr_db.shape == (n_sym, n_ue)
    np.testing.assert_allclose(nv_db[0, 0], exp_nv_db, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sinr_db[0, 0], exp_sinr_db, rtol=0, atol=1e-12)


def test_two_ues_layer_mapping() -> None:
    """Different layer->UE mappings should populate separate UE columns."""
    nl, n_f, n_sym = 2, 12, 1
    ree: np.ndarray = np.zeros((nl, n_f, n_sym), dtype=FloatNP)
    ree[0, :, 0] = 4.0  # UE0
    ree[1, :, 0] = 16.0  # UE1
    layer2ue = np.array([0, 1], dtype=IntNP)
    n_ue = 2

    nv_db, sinr_db = post_eq_noisevar_sinr(
        ree=ree,
        layer2ue=layer2ue,
        n_ue=n_ue,
    )

    # For constant Ree per layer, nv_db = 10*log10(Ree) per UE
    exp0 = 10.0 * np.log10(4.0)
    exp1 = 10.0 * np.log10(16.0)
    np.testing.assert_allclose(nv_db[0, 0], exp0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(nv_db[0, 1], exp1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sinr_db[0, 0], -exp0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sinr_db[0, 1], -exp1, rtol=0, atol=1e-12)


def test_multiple_symbols_positions_independent() -> None:
    """Multiple symbols handled independently (no cumulative averaging)."""
    nl, n_f, n_sym = 1, 24, 3
    ree_vals = np.array([2.0, 6.0, 12.0], dtype=FloatNP)
    ree: np.ndarray = np.zeros((nl, n_f, n_sym), dtype=FloatNP)
    for t in range(n_sym):
        ree[0, :, t] = ree_vals[t]

    layer2ue = np.array([0], dtype=IntNP)
    n_ue = 1

    nv_db, sinr_db = post_eq_noisevar_sinr(
        ree=ree,
        layer2ue=layer2ue,
        n_ue=n_ue,
    )

    # For constant Ree per symbol, nv_db = 10*log10(Ree) per symbol
    expected_nv = 10.0 * np.log10(ree_vals)
    expected_sn = -expected_nv
    np.testing.assert_allclose(nv_db[:, 0], expected_nv, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sinr_db[:, 0], expected_sn, rtol=0, atol=1e-12)


def test_empty_allocation_returns_zeros() -> None:
    """If n_prb=0, outputs should be zeros with correct shape."""
    nl, n_f, n_sym = 1, 0, 2
    ree: np.ndarray = np.ones((nl, n_f, n_sym), dtype=FloatNP)
    layer2ue = np.array([0], dtype=IntNP)
    nv, sn = post_eq_noisevar_sinr(
        ree,
        layer2ue=layer2ue,
        n_ue=1,
    )
    assert nv.shape == (n_sym, 1)
    assert sn.shape == (n_sym, 1)
    np.testing.assert_allclose(nv, 0.0)
    np.testing.assert_allclose(sn, 0.0)
