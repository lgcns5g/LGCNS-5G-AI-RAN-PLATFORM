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

"""Unit tests for `ran.phy.numpy.pusch.measure_rssi` (NumPy implementation)."""

import numpy as np

from ran.constants import SC_PER_PRB
from ran.phy.numpy.pusch import measure_rssi
from ran.types import ComplexNP, FloatNP


def test_basic_shapes_and_dtypes() -> None:
    """Basic shapes and dtypes for output values."""
    n_f_alloc, n_dmrs, n_ant = 24, 2, 2
    # Use tiny non-zero amplitude to avoid log10(0) warnings in db()
    xtf_band_dmrs: np.ndarray = np.full((n_f_alloc, n_dmrs, n_ant), 1e-12 + 0.0j, dtype=ComplexNP)

    rssi_db, rssi_reported_db = measure_rssi(xtf_band_dmrs=xtf_band_dmrs)

    assert rssi_db.shape == (n_dmrs, n_ant)
    assert rssi_db.dtype == FloatNP
    assert isinstance(rssi_reported_db, float)


def test_selection_and_power_sums() -> None:
    """Power sums over PRB-band slice and aggregation are correct."""
    n_f_alloc, n_dmrs, n_ant = 12, 2, 2
    xtf_band_dmrs: np.ndarray = np.zeros((n_f_alloc, n_dmrs, n_ant), dtype=ComplexNP)
    # Symbol 0 amplitudes: ant0=2, ant1=3
    xtf_band_dmrs[:, 0, 0] = 2.0 + 0.0j
    xtf_band_dmrs[:, 0, 1] = 3.0 + 0.0j
    # Symbol 1 amplitudes: ant0=4, ant1=1
    xtf_band_dmrs[:, 1, 0] = 4.0 + 0.0j
    xtf_band_dmrs[:, 1, 1] = 1.0 + 0.0j

    rssi_db, rssi_reported = measure_rssi(xtf_band_dmrs=xtf_band_dmrs)

    # Per-symbol per-antenna power sums: sum |A|^2 over 12 tones
    p_s0_ant0 = SC_PER_PRB * (2.0**2)  # 48
    p_s0_ant1 = SC_PER_PRB * (3.0**2)  # 108
    p_s1_ant0 = SC_PER_PRB * (4.0**2)  # 192
    p_s1_ant1 = SC_PER_PRB * (1.0**2)  # 12

    # Expected RSSI per symbol per antenna
    rssi_db_s0_ant0 = 10.0 * np.log10(p_s0_ant0)
    rssi_db_s0_ant1 = 10.0 * np.log10(p_s0_ant1)
    rssi_db_s1_ant0 = 10.0 * np.log10(p_s1_ant0)
    rssi_db_s1_ant1 = 10.0 * np.log10(p_s1_ant1)

    atol = 1e-12

    # Test RSSI per symbol per antenna
    np.testing.assert_allclose(rssi_db[0, 0], rssi_db_s0_ant0, rtol=0, atol=atol)
    np.testing.assert_allclose(rssi_db[0, 1], rssi_db_s0_ant1, rtol=0, atol=atol)
    np.testing.assert_allclose(rssi_db[1, 0], rssi_db_s1_ant0, rtol=0, atol=atol)
    np.testing.assert_allclose(rssi_db[1, 1], rssi_db_s1_ant1, rtol=0, atol=atol)

    # Reported RSSI (all symbs): Avg across symbols per antenna & sum over antennas
    mean_ant0 = (p_s0_ant0 + p_s1_ant0) / 2.0  # 120
    mean_ant1 = (p_s0_ant1 + p_s1_ant1) / 2.0  # 60
    exp_reported = 10.0 * np.log10(mean_ant0 + mean_ant1)  # 10log10(180)
    np.testing.assert_allclose(rssi_reported, exp_reported, rtol=0, atol=atol)


def test_multiple_prbs_and_allocation_bounds() -> None:
    """Multiple PRBs summed in PRB slice; single symbol equals reported."""
    n_f_alloc, n_dmrs, n_ant = 24, 1, 1
    xtf_band_dmrs: np.ndarray = np.zeros((n_f_alloc, n_dmrs, n_ant), dtype=ComplexNP)
    xtf_band_dmrs[:, 0, 0] = 2.0 + 0.0j

    rssi_db, rssi_reported_db = measure_rssi(xtf_band_dmrs=xtf_band_dmrs)

    p = 24 * (2.0**2)
    exp_db = 10.0 * np.log10(p)
    np.testing.assert_allclose(rssi_db[0, 0], exp_db, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rssi_reported_db, exp_db, rtol=0, atol=1e-12)


def test_multi_prbs_multi_ant_with_phase() -> None:
    """RSSI is power-only; phase rotations must not change results."""
    n_f_alloc, n_dmrs, n_ant = 24, 2, 2
    xtf_band_dmrs: np.ndarray = np.zeros((n_f_alloc, n_dmrs, n_ant), dtype=ComplexNP)
    amp = np.array([2.0, 3.0])
    phase = np.array([0.3, -1.1])
    for a in range(n_ant):
        xtf_band_dmrs[:, :, a] = amp[a] * np.exp(1j * phase[a])
    rssi_db, rssi_reported = measure_rssi(xtf_band_dmrs=xtf_band_dmrs)
    p0 = 24 * (amp[0] ** 2)
    p1 = 24 * (amp[1] ** 2)
    exp0 = 10.0 * np.log10(p0)
    exp1 = 10.0 * np.log10(p1)
    np.testing.assert_allclose(rssi_db[:, 0], exp0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rssi_db[:, 1], exp1, rtol=0, atol=1e-12)
    exp_sum = 10.0 * np.log10(p0 + p1)
    np.testing.assert_allclose(rssi_reported, exp_sum, rtol=0, atol=1e-12)
