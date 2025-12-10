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

"""JAX implementation of interference+noise covariance estimation (optimized PHY).

Implements DMRS-only covariance estimation using a cleaner approach than the
reference implementation. Works directly on DMRS subcarriers without
embedding/extraction confusion.
"""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ran.phy.jax.utils import complex_mul


def estimate_covariance(
    y_dmrs__ri_dsym_rxant_dsc: Array,
    x_dmrs__ri_port_dsym_dsc: Array,
    h_est__ri_port_rxant_dsc: Array,
    n_prb: int,
    rww_regularizer_val: float,
    n_dmrs_sc_per_prb: int,
    energy: float,
) -> tuple[Array, Array]:
    """Estimate interference+noise covariance using DMRS subcarriers only.

    Computes covariance from residuals:
    residual = y_dmrs - sum_port(sqrt(energy) * x_dmrs_port * H_port)
    where H_port is the normalized channel estimate (H = H_true after matched filter normalization)
    Computes per-PRB covariance: R_ww = (1/N) * sum(residual * residual^H)

    Args:
        y_dmrs__ri_dsym_rxant_dsc: Raw received DMRS with stacked real/imag,
            shape (2, n_sym, n_rxant, n_sc)
        x_dmrs__ri_port_dsym_dsc: Transmitted DMRS with OCC and stacked real/imag,
            shape (2, n_port, n_sym, n_sc)
        h_est__ri_port_rxant_dsc: Channel estimates with stacked real/imag,
            shape (2, n_port, n_rxant, n_sc)
        n_prb: Number of PRBs in the allocation
        rww_regularizer_val: Regularization value for numerical stability
        n_dmrs_sc_per_prb: Number of DMRS subcarriers per PRB (typically 6)
        energy: Energy scaling factor for DMRS transmission (typically 1.0 or 2.0)

    Returns:
        n_cov__ri_rxant_rxant_prb: Noise covariance with stacked real/imag,
            shape (2, n_rxant, n_rxant, n_prb)
        mean_noise_var__prb: Mean noise variance per PRB (n_prb,)
    """
    n_dmrs_sym = y_dmrs__ri_dsym_rxant_dsc.shape[1]
    n_rxant = y_dmrs__ri_dsym_rxant_dsc.shape[2]

    # ------------------------------------------------------------------------
    # Compute desired signal component: sum_port(x_dmrs_port * H_port)
    # ------------------------------------------------------------------------

    # Expand H to have DMRS symbol dimension: (2, n_port, 1, n_rxant, n_sc)
    h__ri_port_dsym_rxant_dsc = jnp.expand_dims(h_est__ri_port_rxant_dsc, axis=2)

    # Expand x_dmrs to have Rx antenna dimension: (2, n_port, n_sym, 1, n_sc)
    x__ri_port_dsym_rxant_dsc = jnp.expand_dims(x_dmrs__ri_port_dsym_dsc, axis=3)

    # Scale x_dmrs by sqrt(energy) to reconstruct transmitted signal power
    # Since H is normalized (H = H_true), we need:
    # desired = H * x * sqrt(energy) = H_true * x * sqrt(energy)
    x__ri_port_dsym_rxant_dsc = x__ri_port_dsym_rxant_dsc * jnp.sqrt(energy)

    # Complex multiplication: x[port,sym,rxant,sc] * H[port,sym,rxant,sc]
    # Shape: (2, n_port, n_sym, n_rxant, n_sc)
    desired__ri_port_dsym_rxant_dsc = complex_mul(
        x__ri_port_dsym_rxant_dsc,
        h__ri_port_dsym_rxant_dsc,
    )

    # Sum over DMRS ports (axis 1): (2, n_sym, n_rxant, n_sc)
    desired__ri_dsym_rxant_dsc = jnp.sum(desired__ri_port_dsym_rxant_dsc, axis=1)

    # ------------------------------------------------------------------------
    # Compute residuals: residual = y_dmrs - desired
    # ------------------------------------------------------------------------

    residual__ri_dsym_rxant_dsc = y_dmrs__ri_dsym_rxant_dsc - desired__ri_dsym_rxant_dsc

    # Normalize by sqrt(energy) to account for DMRS energy scaling
    # residual__ri_dsym_rxant_dsc = residual__ri_dsym_rxant_dsc / 2.0

    # ------------------------------------------------------------------------
    # Reshape to per-PRB structure
    # ------------------------------------------------------------------------

    residual__ri_sym_rxant_prb_dmrssc = residual__ri_dsym_rxant_dsc.reshape(
        2, n_dmrs_sym, n_rxant, n_prb, n_dmrs_sc_per_prb
    )

    # ------------------------------------------------------------------------
    # Compute interference+noise covariance matrix per PRB
    # ------------------------------------------------------------------------

    # Compute R_ww = (1/N) * sum_sample(residual * residual^H) for each PRB
    # Contract over DMRS symbols (n_dmrs_sym) and DMRS subcarriers (n_dmrs_sc_per_prb)
    # residual * conj(residual) = (a + bi) * (a - bi) = (aa + bb) + (ba - ab)i
    # Result: (2, n_rxant, n_rxant, n_prb)

    # Extract real and imag components for einsum
    residual_real = residual__ri_sym_rxant_prb_dmrssc[0]
    residual_imag = residual__ri_sym_rxant_prb_dmrssc[1]

    # Real part: real*real + imag*imag
    cov_real__rxant_rxant_prb = jnp.einsum(
        "sapc,sbpc->abp", residual_real, residual_real, optimize=True
    )
    cov_real__rxant_rxant_prb += jnp.einsum(
        "sapc,sbpc->abp", residual_imag, residual_imag, optimize=True
    )

    # Imag part: imag*real - real*imag
    cov_imag__rxant_rxant_prb = jnp.einsum(
        "sapc,sbpc->abp", residual_imag, residual_real, optimize=True
    )
    cov_imag__rxant_rxant_prb -= jnp.einsum(
        "sapc,sbpc->abp", residual_real, residual_imag, optimize=True
    )

    # Normalize by number of samples (complex samples, not real+imag separately)
    n_sample = n_dmrs_sym * n_dmrs_sc_per_prb
    cov_real__rxant_rxant_prb = cov_real__rxant_rxant_prb / n_sample
    cov_imag__rxant_rxant_prb = cov_imag__rxant_rxant_prb / n_sample

    # Add regularization (diagonal, real part only)
    regularizer__rxant_rxant = rww_regularizer_val * jnp.eye(n_rxant, dtype=jnp.float16)
    cov_real__rxant_rxant_prb += regularizer__rxant_rxant[:, :, None]

    # Stack into single tensor: (2, n_rxant, n_rxant, n_prb)
    n_cov__ri_rxant_rxant_prb = jnp.stack(
        [cov_real__rxant_rxant_prb, cov_imag__rxant_rxant_prb], axis=0
    )

    # ------------------------------------------------------------------------
    # Compute mean interference+noise variance (diagonal elements, magnitude)
    # ------------------------------------------------------------------------

    # Extract diagonal elements (complex): real and imag parts
    # Shape: (n_rxant, n_prb)
    diag_real__rxant_prb = jnp.einsum("iik->ik", cov_real__rxant_rxant_prb)
    diag_imag__rxant_prb = jnp.einsum("iik->ik", cov_imag__rxant_rxant_prb)

    # Compute magnitude: |diag| = sqrt(real² + imag²)
    # Shape: (n_rxant, n_prb)
    diag_mag__rxant_prb = jnp.sqrt(diag_real__rxant_prb**2 + diag_imag__rxant_prb**2)

    # Mean over antennas and PRBs
    # Shape: scalar (averaged over all PRBs and antennas)
    mean_interference_noise_var = jnp.mean(diag_mag__rxant_prb)

    # Broadcast to per-PRB shape (all PRBs get the same averaged value)
    mean_interference_noise_var__prb = jnp.full(
        (n_prb,), mean_interference_noise_var, dtype=diag_mag__rxant_prb.dtype
    )

    return n_cov__ri_rxant_rxant_prb, mean_interference_noise_var__prb


def apply_shrinkage(
    n_cov__ri_rxant_rxant_prb: Array,
    n_samples_per_prb: int,
) -> Array:
    """Apply RBLW shrinkage to covariance matrices.

    Shrinks each PRB's covariance toward scaled identity:
    R_shrunk = (1-rho)*R + rho*(tr(R)/N_ant)*I

    Args:
        n_cov__ri_rxant_rxant_prb: Covariance with stacked real/imag, shape (2, n_ant, n_ant, n_prb)
        n_samples_per_prb: Number of samples used per PRB (6 * n_dmrs_syms)

    Returns:
        n_cov_shrunk__ri: Shrunk covariance with stacked real/imag, shape (2, n_ant, n_ant, n_prb)
    """
    n_ant = n_cov__ri_rxant_rxant_prb.shape[1]

    # Transpose for easier computation: (2, n_prb, n_ant, n_ant)
    r__ri = jnp.transpose(n_cov__ri_rxant_rxant_prb, (0, 3, 1, 2))
    r_real = r__ri[0]  # (n_prb, n_ant, n_ant)
    r_imag = r__ri[1]  # (n_prb, n_ant, n_ant)

    # Compute traces (only real part contributes to trace)
    tr_r = jnp.trace(r_real, axis1=1, axis2=2)  # (n_prb,)

    # R * R (complex matrix multiplication)
    # (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    rr_real = jnp.einsum("pij,pjk->pik", r_real, r_real, optimize=True)
    rr_real -= jnp.einsum("pij,pjk->pik", r_imag, r_imag, optimize=True)
    rr_imag = jnp.einsum("pij,pjk->pik", r_real, r_imag, optimize=True)
    rr_imag += jnp.einsum("pij,pjk->pik", r_imag, r_real, optimize=True)

    tr_rr = jnp.trace(rr_real, axis1=1, axis2=2)  # (n_prb,)

    # RBLW shrinkage parameter
    t_eff = jnp.maximum(jnp.float16(n_samples_per_prb), 1.0)
    denom_core = tr_rr - (tr_r * tr_r) / n_ant

    num = ((t_eff - 2.0) / t_eff) * tr_rr + (tr_r * tr_r)
    den = (t_eff + 2.0) * denom_core

    # Safe division (avoid isfinite for TensorRT compatibility)
    good = den != 0.0
    rho = jnp.where(good, jnp.clip(num / den, 0.0, 1.0), 0.0)  # (n_prb,)

    # Target: scaled identity
    target_scale = tr_r / n_ant  # (n_prb,)
    eye = jnp.eye(n_ant, dtype=jnp.float16)

    # Shrinkage: (1-rho)*R + rho*target*I
    rho_expanded = rho[:, None, None]  # (n_prb, 1, 1)
    target_expanded = target_scale[:, None, None]  # (n_prb, 1, 1)

    r_shrunk_real = (1.0 - rho_expanded) * r_real + rho_expanded * target_expanded * eye
    r_shrunk_imag = (1.0 - rho_expanded) * r_imag  # Imaginary part only scales

    # Stack and transpose back to (2, n_ant, n_ant, n_prb)
    r_shrunk__ri = jnp.stack([r_shrunk_real, r_shrunk_imag], axis=0)
    n_cov_shrunk__ri = jnp.transpose(r_shrunk__ri, (0, 2, 3, 1))

    return n_cov_shrunk__ri


__all__ = [
    "estimate_covariance",
    "apply_shrinkage",
]
