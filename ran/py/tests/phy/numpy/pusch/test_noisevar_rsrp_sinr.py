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

"""Unit tests for noise variance, RSRP, and SINR helpers (NumPy versions).

Covers the functions in `ran.phy.numpy.pusch.noisevar_rsrp_sinr`:
- noise_variance_db: full-slot averaging with +0.5 dB bias
- rsrp_db: accumulation over freq/ant/layers per UE, full-slot average
- sinr_db: broadcast subtraction per position
- noise_rsrp_sinr_db: end-to-end wrapper
"""

import numpy as np

from ran.phy.numpy.pusch.noisevar_rsrp_sinr import (
    noise_rsrp_sinr_db,
    noise_variance_db,
    rsrp_db,
    sinr_db,
)
from ran.types import ComplexNP, FloatNP, IntNP
from ran.utils import db


def test_noise_variance_full_slot_average_and_bias() -> None:
    """noise_variance_db returns full-slot mean (per pos) plus 0.5 dB bias."""
    # tmp_noise_var shape: (n_prb, n_pos)
    tmp_noise_var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=FloatNP)
    out_db = noise_variance_db(tmp_noise_var)

    # Per-position means along PRB axis: [ (1+3)/2, (2+4)/2 ] = [2, 3] + 0.5 dB
    expected_lin: np.ndarray = np.array([2.0, 3.0], dtype=FloatNP)
    expected_db = db(expected_lin) + 0.5
    np.testing.assert_allclose(out_db, expected_db, rtol=0, atol=1e-12)


def test_rsrp_full_slot_two_ues_one_layer_each() -> None:
    """rsrp_db aggregates per UE and full-slot averages across positions."""
    n_f, nl, n_ant, n_pos = 4, 2, 1, 2
    # H ones everywhere
    h_est_save: np.ndarray = np.ones((n_f, nl, n_ant, n_pos), dtype=ComplexNP)
    # Map layers to separate UEs
    layer2ue = np.array([0, 1], dtype=IntNP)
    # number of UEs derived from layer2ue (2)

    out_db = rsrp_db(h_est_save, layer2ue, n_ue=2)

    # Each UE has one layer; |H|^2 sum = n_f * n_ant = 4
    # Normalize by (n_f * n_ant) => linear = 1.0 for all positions
    # Full-slot average replicates same value to both positions
    expected_lin: np.ndarray = np.full((n_pos, 2), 1.0, dtype=FloatNP)
    expected_db = db(expected_lin)
    np.testing.assert_allclose(out_db, expected_db, rtol=0, atol=1e-12)


def test_rsrp_full_slot_average_across_positions() -> None:
    """rsrp_db outputs the mean across positions, repeated at all positions."""
    n_f, nl, n_ant, n_pos = 4, 2, 1, 2
    h_est_save: np.ndarray = np.ones((n_f, nl, n_ant, n_pos), dtype=ComplexNP)
    # Increase amplitude at pos1 by 2 → power x4 for both layers
    h_est_save[..., 1] *= 2.0
    layer2ue = np.array([0, 1], dtype=IntNP)
    # number of UEs derived from layer2ue (2)

    out_db = rsrp_db(h_est_save, layer2ue, n_ue=2)

    # UE0 and UE1 identical behavior with 1 layer each
    # pos0 lin = 4/(4*1) = 1; pos1 lin = 16/(4*1) = 4
    # full-slot mean per UE: (1 + 4)/2 = 2.5, replicated across positions
    expected_lin = np.array([[2.5, 2.5], [2.5, 2.5]], dtype=FloatNP)
    expected_db = db(expected_lin)
    np.testing.assert_allclose(out_db, expected_db, rtol=0, atol=1e-12)


def test_sinr_basic_broadcast_subtraction() -> None:
    """sinr_db subtracts noise from RSRP with broadcasting over UEs."""
    rsrp_vals = np.array([[15.0, 16.0], [23.0, 24.0]], dtype=FloatNP)
    noise_vals = np.array([10.0, 20.0], dtype=FloatNP)
    out = sinr_db(rsrp_vals, noise_vals[:, None])
    expected = np.array([[5.0, 6.0], [3.0, 4.0]], dtype=FloatNP)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_noise_rsrp_sinr_db_end_to_end() -> None:
    """End-to-end wrapper returns consistent shapes and values."""
    # Noise: 2 PRBs, 2 positions
    tmp_noise_var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=FloatNP)

    # Channel: 4 tones (1 PRB assumed in RSRP normalization),
    #          2 layers->2 UEs, 1 ant, 2 positions
    n_f, nl, n_ant, n_pos = 4, 2, 1, 2
    h_est_save: np.ndarray = np.ones((n_f, nl, n_ant, n_pos), dtype=ComplexNP)
    # Scale pos1 by 2 → power x4
    h_est_save[..., 1] *= 2.0
    layer2ue = np.array([0, 1], dtype=IntNP)
    # number of UEs derived from layer2ue (2)

    noise_db, rsrp_db_val, sinr_db_val = noise_rsrp_sinr_db(
        mean_noise_var=tmp_noise_var,
        h_est=h_est_save,
        layer2ue=layer2ue,
        n_ue=2,
    )

    # Expected noise dB: per-position mean over PRBs + 0.5 dB
    expected_noise_lin = np.array([2.0, 3.0], dtype=FloatNP)
    expected_noise_db = db(expected_noise_lin) + 0.5
    np.testing.assert_allclose(noise_db, expected_noise_db, atol=1e-12)

    # Expected RSRP dB (full-slot average replicated)
    # pos0 lin = 1, pos1 lin = 4 → mean = 2.5 per UE, replicated across positions
    expected_rsrp_lin = np.array([[2.5, 2.5], [2.5, 2.5]], dtype=FloatNP)
    expected_rsrp_db = db(expected_rsrp_lin)
    np.testing.assert_allclose(rsrp_db_val, expected_rsrp_db, atol=1e-12)

    # Obtain SINR as RSRP - noise
    expected_sinr_db = expected_rsrp_db - expected_noise_db
    np.testing.assert_allclose(sinr_db_val, expected_sinr_db, atol=1e-12)
