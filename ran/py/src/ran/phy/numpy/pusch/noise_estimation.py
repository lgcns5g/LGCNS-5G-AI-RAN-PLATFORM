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
NumPy translations of MATLAB UL DMRS embedding and covariance estimation helpers.

Implements three components used in detPusch.m:
- estimate_r_tilde: compute residual r_tilde by subtracting desired DMRS
- estimate_noise_covariance: build per-PRB noise covariance from r_tilde

Rules followed:
- NumPy only
- Inputs/outputs mirror MATLAB semantics; indices remain 0-based where MATLAB
  uses 0-based symbol indices. Shapes follow the MATLAB comments.
- SimCtrl globals are modeled as module-level constants in CAPS and used in
  conditional branches, ready to be overridden later.
"""

import numpy as np

from ran.types import (
    ComplexArrayNP,
    ComplexNP,
    FloatArrayNP,
    FloatNP,
    IntNP,
)

# Small module-level constants
PRB_SC: int = 12
TINY_EPS: float = 1e-10


def estimate_r_tilde(
    xtf_band_dmrs: ComplexArrayNP,
    x_dmrs: ComplexArrayNP,
    h_est_band_dmrs: ComplexArrayNP,
) -> ComplexArrayNP:
    """Compute r_tilde from already-sliced PRB-band and DMRS-symbol inputs.

    Args:
        xtf_band_dmrs: Received TF grid, shape (n_prb*12, n_t_dmrs, n_ant)
        x_dmrs: DMRS-only TF grid, shape (n_prb*12, n_t_dmrs, nl)
        h_est_band_dmrs: Estimated channel, shape (n_prb*12, nl, n_ant, n_t_dmrs)

    Returns
    -------
        r_tilde: shape (n_prb*12, n_t_dmrs, n_ant)
    """
    # Subtract denorm * sum_over_layers( x_dmrs * H ) for all dims at once
    contrib = np.einsum("fsl,flas->fsa", x_dmrs, h_est_band_dmrs, optimize=True)
    return xtf_band_dmrs - contrib


def estimate_noise_covariance(
    r_tilde: ComplexArrayNP,
    rww_regularizer_val: float,
) -> tuple[ComplexArrayNP, FloatArrayNP]:
    """Compute per-PRB noise covariance from r_tilde without sym_idx input.

    Args
    ----
        r_tilde: residuals, shape (n_prb*12, n_pos, n_ant)
        rww_regularizer_val: regularization value for noise covariance

    Returns
    -------
        n_cov: (n_ant, n_ant, n_prb, n_pos)
        mean_noise_var: (n_prb, n_pos)
    """
    n_sc, n_pos, n_ant = r_tilde.shape
    n_prb = n_sc // PRB_SC

    # Reshape to (n_prb, PRB_SC(=12), n_pos, n_ant)
    y = r_tilde.reshape(n_prb, PRB_SC, n_pos, n_ant)

    # Sum over tones (t) directly into (a,b,n,p)
    # y[n,t,p,a] * y*[n,t,p,b] -> out[a,b,n,p]
    n_cov = np.einsum("ntpa,ntpb->abnp", y, y.conj(), optimize=True)

    # Normalize and RWW regularize (broadcast eye over n,p)
    denom = PRB_SC * n_pos
    eye = np.eye(n_ant, dtype=ComplexNP)[..., None, None]  # (a,b,1,1)
    n_cov = n_cov + TINY_EPS * eye
    n_cov = (n_cov / denom) + (rww_regularizer_val * eye)

    # tmp_noise_var: mean abs(diagonal) per (n_prb, n_pos)
    diags = np.abs(np.diagonal(n_cov, axis1=0, axis2=1))  # (n_prb, n_pos, n_ant)
    mean_noise_var = diags.mean(axis=-1)  # mean over antennas -> (n_prb, n_pos)

    return n_cov, mean_noise_var


def _n_cov_shrinkage(r_in: ComplexArrayNP, t_samples: IntNP) -> ComplexArrayNP:
    """Vectorized RBLW shrinkage for a stack of covariance matrices.

    Args:
        r_in: Covariance matrices, shape (..., n_ant, n_ant)
        t_samples: Sample count

    Returns
    -------
        Shrunk covariance matrices with same shape as r_in
    """
    n_ant = r_in.shape[-1]

    # Core traces
    tr_r = np.real(np.trace(r_in, axis1=-2, axis2=-1))
    rr = r_in @ r_in
    tr_rr = np.real(np.trace(rr, axis1=-2, axis2=-1))
    denom_core = tr_rr - (tr_r * tr_r) / n_ant

    # RBLW pieces
    num = ((t_samples - 2.0) / t_samples) * tr_rr + (tr_r * tr_r)
    den = (t_samples + 2.0) * denom_core

    # rho with guards
    rho = np.zeros_like(den, dtype=FloatNP)
    good = (den != 0.0) & np.isfinite(den)
    rho[good] = np.clip(num[good] / den[good], 0.0, 1.0)

    rho = rho[..., None, None]
    target_scale = tr_r[..., None, None] / n_ant  # (n_ant, 1, 1)
    eye = np.eye(n_ant, dtype=ComplexNP)  # (n_ant, n_ant)
    return (1.0 - rho) * r_in + rho * target_scale * eye


def ncov_shrinkage(n_cov: ComplexArrayNP) -> ComplexArrayNP:
    """Apply optional shrinkage to ``n_cov`` without mutating the input.

    Args:
        n_cov: Noise covariance, shape (n_ant, n_ant, n_prb, n_pos)

    Returns
    -------
        Covariance array. Shape (n_ant, n_ant, n_prb, n_pos)
    """
    a, a1, n_prb, n_pos = n_cov.shape
    if a != a1:
        msg = f"n_cov must have square antenna dimensions, got shape {n_cov.shape}"
        raise ValueError(msg)

    # Mean over positions -> (a, a, prb)
    r_mean = n_cov.mean(axis=3)

    # Put matrices on the last two axes expected by _n_cov_shrinkage: (prb, a, a)
    r_mean_prb = np.moveaxis(r_mean, -1, 0)

    # Shrink to (prb, a, a)
    r_shrunk_prb = _n_cov_shrinkage(r_mean_prb, IntNP(PRB_SC * n_pos))

    # Back to (a, a, prb, 1)
    r_shrunk = np.moveaxis(r_shrunk_prb, 0, -1)[..., None]

    # Broadcast to (a, a, prb, pos) without copying
    return np.broadcast_to(r_shrunk, (a, a, n_prb, n_pos))


def estimate_covariance(
    xtf_band_dmrs: ComplexArrayNP,
    x_dmrs: ComplexArrayNP,
    h_est_band_dmrs: ComplexArrayNP,
    rww_regularizer_val: float,
) -> tuple[ComplexArrayNP, FloatArrayNP]:
    """Top-level covariance pipeline from slices without sym_idx argument.

    Args:
        xtf_band_dmrs: Received TF grid, shape (n_prb*12, n_t_dmrs, n_ant)
        x_dmrs: DMRS-only TF grid, shape (n_prb*12, n_t_dmrs, nl)
        h_est_band_dmrs: Estimated channel, shape (n_prb*12, nl, n_ant, n_t_dmrs)
        rww_regularizer_val: Regularization value for noise covariance.

    Returns
    -------
        n_cov: (n_ant, n_ant, n_prb, n_pos)
        mean_noise_var: (n_prb, n_pos)

    Note:
    Infers ``n_pos`` from input shapes and computes noise covariance.
    """
    r_tilde = estimate_r_tilde(
        xtf_band_dmrs=xtf_band_dmrs,
        x_dmrs=x_dmrs,
        h_est_band_dmrs=h_est_band_dmrs,
    )

    n_cov, mean_noise_var = estimate_noise_covariance(
        r_tilde=r_tilde,
        rww_regularizer_val=rww_regularizer_val,
    )

    n_cov = ncov_shrinkage(n_cov=n_cov)
    return n_cov, mean_noise_var


__all__ = [
    "estimate_covariance",
    "estimate_noise_covariance",
    "estimate_r_tilde",
    "ncov_shrinkage",
]
