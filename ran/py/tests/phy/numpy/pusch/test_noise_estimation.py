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

"""Unit tests for noise estimation helpers (NumPy implementations).

Covers:
- embed_dmrs_ul
- estimate_r_tilde
- estimate_noise_covariance
"""

from typing import TYPE_CHECKING, Any

import numpy as np

import ran.phy.numpy.pusch.noise_estimation as ne
from ran.phy.numpy.pusch.noise_estimation import (
    estimate_covariance,
    estimate_noise_covariance,
    estimate_r_tilde,
    ncov_shrinkage,
)
from ran.phy.numpy.pusch.dmrs_utils import embed_dmrs_ul
from ran.types import ComplexNP, FloatNP, IntNP

if TYPE_CHECKING:
    from ran.types import IntArrayNP


def _make_basic_inputs(n_prb: int = 1, nl: int = 1) -> dict[str, Any]:
    """Make basic inputs for covariance estimation (0-based indices)."""
    # One DMRS symbol at index 0 (0-based)
    sym_idx_dmrs = np.array([0], dtype=IntNP)
    # r_dmrs indexed as r_dmrs[scram_idx, sym_idx, scid]
    # scram_idx in [0..6*n_prb-1] → make first dim 6*n_prb
    r_dmrs: np.ndarray = np.ones((6 * n_prb, int(sym_idx_dmrs.max()) + 1, 2), dtype=ComplexNP)
    n_scid = 0
    energy = 1.0
    start_prb = 0  # 0-based
    port_idx = np.array([0], dtype=IntNP)  # 0-based bitfield

    return {
        "r_dmrs": r_dmrs,
        "nl": nl,
        "port_idx": port_idx,
        "vec_scid": np.full((nl,), n_scid, dtype=IntNP),
        "sym_idx_dmrs": sym_idx_dmrs,
        "energy": energy,
        "n_prb": n_prb,
        "start_prb": start_prb,
    }


def test_embed_dmrs_ul_basic_no_precoding() -> None:
    """Test embed_dmrs_ul with no precoding and 0-based indices."""
    args = _make_basic_inputs(n_prb=1, nl=1)
    x_dmrs = embed_dmrs_ul(
        r_dmrs=args["r_dmrs"],
        nl=args["nl"],
        port_idx=args["port_idx"],
        vec_scid=args["vec_scid"],
        energy=args["energy"],
    )

    # Expected DMRS REs are at even subcarriers within the PRB
    freq_idx_dmrs = np.array([0, 2, 4, 6, 8, 10], dtype=IntNP)
    sym0 = int(args["sym_idx_dmrs"][0])  # already 0-based

    # Values should be 1+0j at antenna 0 (layer 0)
    vals = x_dmrs[freq_idx_dmrs, sym0, 0]
    assert np.all(vals == 1.0 + 0.0j)
    # Non-DMRS tones should remain 0
    non_dmrs_mask: np.ndarray = np.ones(12, dtype=bool)
    non_dmrs_mask[freq_idx_dmrs] = False
    assert np.all(x_dmrs[non_dmrs_mask, sym0, 0] == 0.0 + 0.0j)


def test_estimate_r_tilde_zero_when_xtf_equals_dmrs_and_h_is_one() -> None:
    """estimate_r_tilde returns zero when xtf equals dmrs and H is ones."""
    args = _make_basic_inputs(n_prb=1, nl=1)
    x_dmrs = embed_dmrs_ul(
        r_dmrs=args["r_dmrs"],
        nl=args["nl"],
        port_idx=args["port_idx"],
        vec_scid=args["vec_scid"],
        energy=args["energy"],
    )
    # Use the DMRS-only grid as received signal slice
    xtf_band_dmrs = x_dmrs.copy()
    # h_est_save shape (n_f, nl, n_ant, n_pos) with ones
    n_f = xtf_band_dmrs.shape[0]
    h_est_save = np.ones((n_f, 1, 1, 1), dtype=ComplexNP)

    r_tilde = estimate_r_tilde(
        xtf_band_dmrs=xtf_band_dmrs,
        x_dmrs=x_dmrs,
        h_est_band_dmrs=h_est_save,
    )

    assert r_tilde.shape == (12, 1, 1)
    assert np.allclose(r_tilde, 0.0 + 0.0j)


def test_estimate_r_tilde_single_position_shapes() -> None:
    """estimate_r_tilde shapes for single DMRS position (4D H)."""
    args = _make_basic_inputs(n_prb=1, nl=1)
    x_dmrs = embed_dmrs_ul(
        r_dmrs=args["r_dmrs"],
        nl=args["nl"],
        port_idx=args["port_idx"],
        vec_scid=args["vec_scid"],
        energy=args["energy"],
    )
    xtf_band_dmrs = x_dmrs.copy()
    n_f = xtf_band_dmrs.shape[0]
    # Provide 4D h_est_save with one position
    h_est_save = np.ones((n_f, 1, 1, 1), dtype=ComplexNP)

    r_tilde = estimate_r_tilde(
        xtf_band_dmrs=xtf_band_dmrs,
        x_dmrs=x_dmrs,
        h_est_band_dmrs=h_est_save,
    )
    assert r_tilde.shape == (12, 1, 1)
    assert np.allclose(r_tilde, 0.0 + 0.0j)


def test_estimate_noise_covariance_regularizer_only_zero_r_tilde() -> None:
    """estimate_noise_covariance with zero r_tilde yields regularizer on diagonal."""
    n_prb = 1
    n_ant = 2
    n_pos = 1
    # r_tilde zeros → nCov should be just the regularizer on the diagonal
    r_tilde: np.ndarray = np.zeros((12 * n_prb, n_pos, n_ant), dtype=ComplexNP)
    reg = 0.01

    n_cov, mean_noise_var = estimate_noise_covariance(
        r_tilde=r_tilde,
        rww_regularizer_val=reg,
    )

    assert n_cov.shape == (n_ant, n_ant, n_prb, n_pos)
    # Diagonals equal to the regularizer; off-diagonals zero
    for ii in range(n_prb):
        for pp in range(n_pos):
            mat = n_cov[:, :, ii, pp]
            assert np.allclose(np.diag(mat).real, reg)
            off_diag = mat.copy()
            np.fill_diagonal(off_diag, 0.0)
            assert np.allclose(off_diag, 0.0 + 0.0j)

    # mean_noise_var is average diagonal magnitude per antenna per (n_prb, n_pos)
    assert np.allclose(mean_noise_var, reg)


def test_ncov_shrinkage_matches_rblw_on_mean() -> None:
    """Shrinkage equals RBLW applied to the per-PRB mean across positions."""
    n_ant = 2
    n_prb = 1
    n_pos = 2
    # Build two different covariances for the same PRB
    cov_pos0 = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=ComplexNP)
    cov_pos1 = np.array([[4.0, -0.2], [-0.2, 3.0]], dtype=ComplexNP)
    n_cov: np.ndarray = np.empty((n_ant, n_ant, n_prb, n_pos), dtype=ComplexNP)
    n_cov[:, :, 0, 0] = cov_pos0
    n_cov[:, :, 0, 1] = cov_pos1
    # Different sample counts per position; summed for shrinkage
    out = ncov_shrinkage(n_cov=n_cov)

    # Compute expected using the same RBLW formula on the mean across positions
    r_mean = n_cov.mean(axis=3)  # (n_ant, n_ant, n_prb)
    # With the new API, shrinkage uses constant sample count PRB_SC * n_pos
    t_total: IntArrayNP = np.full((n_prb,), ne.PRB_SC * n_pos, dtype=IntNP)
    r_mean_prb = np.transpose(r_mean, (2, 0, 1))  # (n_prb, n_ant, n_ant)

    # RBLW shrinkage (vectorized for n_prb)
    tr_r = np.real(np.trace(r_mean_prb, axis1=1, axis2=2))
    rr = r_mean_prb @ r_mean_prb
    tr_rr = np.real(np.trace(rr, axis1=1, axis2=2))
    denom_core = tr_rr - (tr_r * tr_r) / n_ant
    rho = np.zeros_like(denom_core, dtype=FloatNP)
    good = (denom_core != 0.0) & np.isfinite(denom_core)
    num = ((t_total - 2.0) / t_total) * tr_rr + (tr_r * tr_r)
    den = (t_total + 2.0) * denom_core
    rho[good] = np.clip(num[good] / den[good], 0.0, 1.0)
    target_scale = tr_r / n_ant
    eye = np.eye(n_ant, dtype=ComplexNP)
    expected_prb = (1.0 - rho)[:, None, None] * r_mean_prb + rho[:, None, None] * target_scale[
        :, None, None
    ] * eye
    expected = np.transpose(expected_prb, (1, 2, 0))[:, :, :, None]
    expected = np.repeat(expected, n_pos, axis=3)

    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_estimate_covariance_end_to_end_zero_residual_regularizer_only() -> None:
    """End-to-end: zero residual yields reg-only diagonal (with tiny eps)."""
    args = _make_basic_inputs(n_prb=1, nl=1)
    # Make xtf equal to the DMRS-only grid so r_tilde -> 0 with H=1
    x_dmrs = embed_dmrs_ul(
        r_dmrs=args["r_dmrs"],
        nl=args["nl"],
        port_idx=args["port_idx"],
        vec_scid=args["vec_scid"],
        energy=args["energy"],
    )
    xtf_band_dmrs = x_dmrs.copy()
    n_f = xtf_band_dmrs.shape[0]
    n_ant = 1
    n_pos = 1
    h_est_save = np.ones((n_f, 1, n_ant, n_pos), dtype=ComplexNP)
    reg = 0.01

    n_cov, mean_noise_var = estimate_covariance(
        xtf_band_dmrs=xtf_band_dmrs,
        x_dmrs=x_dmrs,
        h_est_band_dmrs=h_est_save,
        rww_regularizer_val=reg,
    )

    assert n_cov.shape == (n_ant, n_ant, args["n_prb"], n_pos)
    # Expected diagonal = reg + tiny_eps / (PRB_SC * n_pos)
    expected_diag = reg + ne.TINY_EPS / (ne.PRB_SC * n_pos)
    assert np.allclose(np.diag(n_cov[:, :, 0, 0]).real, expected_diag)
    # off-diagonals are zero
    off_diag = n_cov[:, :, 0, 0].copy()
    np.fill_diagonal(off_diag, 0.0)
    assert np.allclose(off_diag, 0.0 + 0.0j)
    # mean_noise_var equals expected diagonal value
    assert np.allclose(mean_noise_var, expected_diag)
