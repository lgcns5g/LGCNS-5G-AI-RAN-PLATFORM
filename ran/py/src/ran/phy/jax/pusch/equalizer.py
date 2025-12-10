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

"""PUSCH equalizer implementation.

This module provides:
- MMSE-IRC equalizer weight derivation
- Equalizer application to received signals
- Post-equalization noise variance and SINR computation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ran.trt_plugins.cholesky_factor_inv import cholesky_inv_4x4


# =============================================================================
# MMSE-IRC Equalizer Weights
# =============================================================================


def get_mmse_irc_weights(
    h_est__ri_port_rxant_sc: jnp.ndarray,
    n_cov__ri_rxant_rxant_prb: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Derive MMSE-IRC equalizer weights using efficient Cholesky factorization.

    Computes equalizer weights using the MMSE-IRC formula:
        W = (H^H R^{-1} H + I)^{-1} H^H R^{-1}

    where R is the noise+interference covariance matrix.

    Computational Strategy (Efficient Factorization):
    --------------------------------------------------
    Instead of directly computing R^{-1}, we use Cholesky decomposition:
        R = L L^H  =>  R^{-1} = L^{-H} L^{-1}

    Then define N = L^{-1} H, which allows us to factor:
        1. H^H R^{-1} H = H^H L^{-H} L^{-1} H = (L^{-1} H)^H (L^{-1} H) = N^H N
        2. H^H R^{-1} = (L^{-1} H)^H L^{-1} = N^H L^{-1}

    So the full formula becomes:
        W = (N^H N + I)^{-1} N^H L^{-1}
            └─── G ──┘      └── T ──┘

    Benefits:
        - G is only (n_port x n_port) - much smaller than (n_rxant x n_rxant)
        - N is computed once and reused for both G and T
        - Never need to compute full R^{-1}
        - More numerically stable than direct inversion

    Args:
        h_est__ri_port_rxant_sc: Channel estimates with stacked real/imag,
            shape (2, n_port, n_rxant, n_sc)
        n_cov__ri_rxant_rxant_prb: Interference+noise covariance with stacked real/imag,
            shape (2, n_rxant, n_rxant, n_prb)

    Returns:
        w__ri_port_rxant_sc: Equalizer weights with stacked real/imag, shape (2, n_port, n_rxant, n_sc)
        ree__port_sc: Error metric, shape (n_port, n_sc)
    """
    # Get dimensions
    n_port = h_est__ri_port_rxant_sc.shape[1]
    n_rxant = h_est__ri_port_rxant_sc.shape[2]
    n_sc = h_est__ri_port_rxant_sc.shape[3]

    # Extract real and imag for computation
    h_est_real__port_rxant_sc = h_est__ri_port_rxant_sc[0]
    h_est_imag__port_rxant_sc = h_est__ri_port_rxant_sc[1]
    n_cov_real__rxant_rxant_prb = n_cov__ri_rxant_rxant_prb[0]
    n_cov_imag__rxant_rxant_prb = n_cov__ri_rxant_rxant_prb[1]

    # Identity matrix for ports
    eye_port_real__port_port = jnp.eye(n_port, dtype=jnp.float16)

    # ------------------------------------------------------------------------
    # Cholesky: L^{-1} from Cholesky inverse, then expand to subcarrier grid
    # ------------------------------------------------------------------------

    # Make diagonal real for numerical stability (covariance diagonal must be real)
    # Do this on original (n_rxant, n_rxant, n_prb) format for cleaner indexing
    rxant_idx__rxant = jnp.arange(n_rxant)
    n_cov_imag__rxant_rxant_prb = n_cov_imag__rxant_rxant_prb.at[
        rxant_idx__rxant, rxant_idx__rxant, :
    ].set(0.0)

    # Transpose to (n_prb, n_rxant, n_rxant) for Cholesky (expects batch in dim 0)
    n_cov_real__prb_rxant_rxant = jnp.einsum("abc->cab", n_cov_real__rxant_rxant_prb, optimize=True)
    n_cov_imag__prb_rxant_rxant = jnp.einsum("abc->cab", n_cov_imag__rxant_rxant_prb, optimize=True)

    if n_rxant == 4:
        l_inv_real__prb_rxant_rxant, l_inv_imag__prb_rxant_rxant = cholesky_inv_4x4(
            n_cov_real__prb_rxant_rxant.astype(jnp.float32),
            n_cov_imag__prb_rxant_rxant.astype(jnp.float32),
        )
    else:
        raise ValueError(f"Unsupported matrix size: {n_rxant}")

    # Expand L^{-1} to per-subcarrier grid: (n_prb, n_rxant, n_rxant) -> (n_sc, n_rxant, n_rxant)
    # This duplicates each PRB's covariance 12 times (minimal memory cost for 4×4 or 8×8 matrices)
    prb_idx__sc = jnp.arange(n_sc) // 12
    l_inv_real__sc_rxant_rxant = l_inv_real__prb_rxant_rxant[prb_idx__sc]
    l_inv_imag__sc_rxant_rxant = l_inv_imag__prb_rxant_rxant[prb_idx__sc]

    # Transpose H to batch over subcarriers: (n_port, n_rxant, n_sc) -> (n_sc, n_port, n_rxant)
    h_real__sc_port_rxant = h_est_real__port_rxant_sc.transpose(2, 0, 1)
    h_imag__sc_port_rxant = h_est_imag__port_rxant_sc.transpose(2, 0, 1)

    # ------------------------------------------------------------------------
    # Compute N = L^{-1} H (batched over subcarriers)
    # ------------------------------------------------------------------------
    # Compute N once and use it for both G = N^H N + I and T = N^H L^{-1}

    # Complex matmul per subcarrier: (n_sc, n_rxant, n_rxant) @ (n_sc, n_port, n_rxant)
    # Result: (n_sc, n_rxant, n_port)
    n_real__sc_rxant_port = jnp.einsum(
        "sab,scb->sac", l_inv_real__sc_rxant_rxant, h_real__sc_port_rxant, optimize=True
    ) - jnp.einsum("sab,scb->sac", l_inv_imag__sc_rxant_rxant, h_imag__sc_port_rxant, optimize=True)
    n_imag__sc_rxant_port = jnp.einsum(
        "sab,scb->sac", l_inv_real__sc_rxant_rxant, h_imag__sc_port_rxant, optimize=True
    ) + jnp.einsum("sab,scb->sac", l_inv_imag__sc_rxant_rxant, h_real__sc_port_rxant, optimize=True)

    # ------------------------------------------------------------------------
    # Compute G = N^H N + I (batched over subcarriers)
    # ------------------------------------------------------------------------
    # Replaces H^H R^{-1} H from the original formula
    # Since N = L^{-1} H, we have: H^H R^{-1} H = N^H N

    # N^H @ N: conjugate transpose means swap rxant/port and negate imaginary
    # Contract over rxant dimension (a), result: (n_sc, n_port, n_port)
    g_real__sc_port_port = (
        jnp.einsum("sac,sak->sck", n_real__sc_rxant_port, n_real__sc_rxant_port, optimize=True)
        + jnp.einsum("sac,sak->sck", n_imag__sc_rxant_port, n_imag__sc_rxant_port, optimize=True)
        + eye_port_real__port_port[None, :, :]
    )
    g_imag__sc_port_port = jnp.einsum(
        "sac,sak->sck", n_real__sc_rxant_port, n_imag__sc_rxant_port, optimize=True
    ) - jnp.einsum("sac,sak->sck", n_imag__sc_rxant_port, n_real__sc_rxant_port, optimize=True)

    # ------------------------------------------------------------------------
    # Compute G^{-1} (batched over subcarriers)
    # ------------------------------------------------------------------------
    # Invert the small (n_port × n_port) matrix G per subcarrier
    # For n_port=1, this is just scalar complex inversion

    if n_port == 1:
        # For 1×1: 1/(a + ib) = a/(a² + b²) - ib/(a² + b²)
        denom__sc_port_port = (
            g_real__sc_port_port * g_real__sc_port_port
            + g_imag__sc_port_port * g_imag__sc_port_port
        )
        # Clamp denominator to prevent division by zero when G is degenerate
        min_denom = 1e-10
        denom_clamped__sc_port_port = jnp.maximum(denom__sc_port_port, min_denom)
        g_inv_real__sc_port_port = g_real__sc_port_port / denom_clamped__sc_port_port
        g_inv_imag__sc_port_port = -g_imag__sc_port_port / denom_clamped__sc_port_port
    else:
        # Raise not implemented error
        error_msg = f"Unsupported matrix size: {n_port}"
        error_msg += "\nSupported matrix sizes: 1"
        raise NotImplementedError(error_msg)

    # ------------------------------------------------------------------------
    # Compute T = N^H L^{-1} (batched over subcarriers)
    # ------------------------------------------------------------------------
    # This replaces the term H^H R^{-1} from the original formula
    # Since N = L^{-1} H, we have: H^H R^{-1} = H^H L^{-H} L^{-1} = N^H L^{-1}
    # We reuse N (computed in Step 1) here!

    # Conjugate transpose N: (n_sc, n_rxant, n_port) -> (n_sc, n_port, n_rxant)
    # Conjugate means we negate the imaginary part
    n_herm_real__sc_port_rxant = jnp.einsum("sac->sca", n_real__sc_rxant_port, optimize=True)
    n_herm_imag__sc_port_rxant = jnp.einsum("sac->sca", -n_imag__sc_rxant_port, optimize=True)

    # Complex matmul: T = N^H @ L^{-1}
    # (n_sc, n_port, n_rxant) @ (n_sc, n_rxant, n_rxant) -> (n_sc, n_port, n_rxant)
    # Contract over first rxant dimension (a)
    t_real__sc_port_rxant = jnp.einsum(
        "sca,sab->scb", n_herm_real__sc_port_rxant, l_inv_real__sc_rxant_rxant, optimize=True
    ) - jnp.einsum(
        "sca,sab->scb", n_herm_imag__sc_port_rxant, l_inv_imag__sc_rxant_rxant, optimize=True
    )
    t_imag__sc_port_rxant = jnp.einsum(
        "sca,sab->scb", n_herm_real__sc_port_rxant, l_inv_imag__sc_rxant_rxant, optimize=True
    ) + jnp.einsum(
        "sca,sab->scb", n_herm_imag__sc_port_rxant, l_inv_real__sc_rxant_rxant, optimize=True
    )

    # ------------------------------------------------------------------------
    # Compute W = G^{-1} T (batched over subcarriers)
    # ------------------------------------------------------------------------
    # Combine the inverted small matrix G^{-1} with T
    # This gives us the equalizer weights: W = (H^H R^{-1} H + I)^{-1} H^H R^{-1}

    # G_inv @ T: (n_sc, n_port, n_port) @ (n_sc, n_port, n_rxant)
    # Contract over second port dimension (k), result: (n_sc, n_port, n_rxant)
    w_real__sc_port_rxant = jnp.einsum(
        "sck,ska->sca", g_inv_real__sc_port_port, t_real__sc_port_rxant, optimize=True
    ) - jnp.einsum("sck,ska->sca", g_inv_imag__sc_port_port, t_imag__sc_port_rxant, optimize=True)
    w_imag__sc_port_rxant = jnp.einsum(
        "sck,ska->sca", g_inv_real__sc_port_port, t_imag__sc_port_rxant, optimize=True
    ) + jnp.einsum("sck,ska->sca", g_inv_imag__sc_port_port, t_real__sc_port_rxant, optimize=True)

    # Transpose back to (n_port, n_rxant, n_sc) output format
    w_real__port_rxant_sc = w_real__sc_port_rxant.transpose(1, 2, 0).astype(jnp.float16)
    w_imag__port_rxant_sc = w_imag__sc_port_rxant.transpose(1, 2, 0).astype(jnp.float16)

    # ------------------------------------------------------------------------
    # Compute error metrics and apply lambda scaling
    # ------------------------------------------------------------------------
    # Extract diagonal of G^{-1} to compute reliability metric (REE)
    # and lambda scaling factor for the weights

    # Extract diagonal of G^{-1}: (n_sc, n_port, n_port) -> (n_sc, n_port)
    diag_g_inv__sc_port = jnp.diagonal(g_inv_real__sc_port_port, axis1=1, axis2=2)

    # Lambda scaling factor (post-equalization SNR adjustment)
    # Clamp diagonal to prevent division by zero when channel is very weak.
    # In such cases G ≈ I, so diag(G^{-1}) ≈ 1, making denominator -> 0
    max_diag_g_inv = 1.0 - 1e-7  # Upper bound for numerical stability
    diag_g_inv_clamped__sc_port = jnp.minimum(diag_g_inv__sc_port, max_diag_g_inv)
    lambda_vec__sc_port = 1.0 / (1.0 - diag_g_inv_clamped__sc_port)

    # Reliability metric (effective noise enhancement)
    min_ree = 1.0 / 10000.0
    ree__sc_port = jnp.maximum(min_ree, lambda_vec__sc_port * diag_g_inv_clamped__sc_port)

    # Transpose to (n_port, n_sc) output format
    lambda_vec__port_sc = lambda_vec__sc_port.T.astype(jnp.float16)
    ree__port_sc = ree__sc_port.T.astype(jnp.float16)

    # Apply lambda scaling to equalizer weights
    w_real__port_rxant_sc = w_real__port_rxant_sc * lambda_vec__port_sc[:, None, :]
    w_imag__port_rxant_sc = w_imag__port_rxant_sc * lambda_vec__port_sc[:, None, :]

    # Stack results into stacked real/imag format
    w__ri_port_rxant_sc = jnp.stack([w_real__port_rxant_sc, w_imag__port_rxant_sc], axis=0)

    return w__ri_port_rxant_sc, ree__port_sc


# =============================================================================
# Apply Equalizer
# =============================================================================


def apply_equalizer(
    xtf__ri_sym_rxant_sc: jnp.ndarray,
    w__ri_port_rxant_sc: jnp.ndarray,
) -> jnp.ndarray:
    """Apply equalizer weights to received signal for all OFDM symbols.

    Performs: x_est = W * y for each symbol, where:
    - W are the equalizer weights (one per subcarrier)
    - y is the received signal

    This applies equalization to ALL symbols in the time-frequency grid.
    The caller can extract specific symbols (e.g., data symbols) afterwards.

    Args:
        xtf__ri_sym_rxant_sc: Received signal with stacked real/imag,
            shape (2, n_sym, n_rxant, n_sc)
        w__ri_port_rxant_sc: Equalizer weights with stacked real/imag,
            shape (2, n_port, n_rxant, n_sc)

    Returns:
        x_est__ri_port_sym_sc: Equalized symbols with stacked real/imag,
            shape (2, n_port, n_sym, n_sc)
    """
    # Extract real and imag components
    xtf_real__sym_rxant_sc = xtf__ri_sym_rxant_sc[0]
    xtf_imag__sym_rxant_sc = xtf__ri_sym_rxant_sc[1]
    w_real__port_rxant_sc = w__ri_port_rxant_sc[0]
    w_imag__port_rxant_sc = w__ri_port_rxant_sc[1]

    # Apply weights: W @ y (no conjugation)
    x_est_real__port_sym_sc = jnp.einsum(
        "pas,das->pds", w_real__port_rxant_sc, xtf_real__sym_rxant_sc, optimize=True
    ) - jnp.einsum("pas,das->pds", w_imag__port_rxant_sc, xtf_imag__sym_rxant_sc, optimize=True)
    x_est_imag__port_sym_sc = jnp.einsum(
        "pas,das->pds", w_real__port_rxant_sc, xtf_imag__sym_rxant_sc, optimize=True
    ) + jnp.einsum("pas,das->pds", w_imag__port_rxant_sc, xtf_real__sym_rxant_sc, optimize=True)

    # Stack results
    x_est__ri_port_sym_sc = jnp.stack([x_est_real__port_sym_sc, x_est_imag__port_sym_sc], axis=0)

    return x_est__ri_port_sym_sc


# =============================================================================
# Post-Equalization Metrics
# =============================================================================


def db(x: Array) -> Array:
    """Compute 10*log10(x) elementwise.

    Args:
        x: Linear-scale input (must be positive to avoid -inf results)

    Returns:
        Decibel-scale output as float32
    """
    return 10.0 * jnp.log10(x)


def post_eq_noisevar_sinr(
    ree__layer_sym_freq: Array,
    start_prb: int,
    n_prb: int,
    layer2ue: Array | tuple,
    n_ue: int,
) -> tuple[Array, Array]:
    """Compute post-equalization noise variance and SINR per UE and symbol (dB).

    Computes post-equalization metrics by:
    1. Extracting allocated frequency resources (PRB allocation)
    2. Computing 1/Ree and averaging over allocated subcarriers
    3. Aggregating layers to UEs using one-hot encoding
    4. Converting to dB scale

    Args:
        ree__layer_sym_freq: Equalization error metric, shape (n_layer, n_sym, n_freq)
        start_prb: Starting PRB index (0-based)
        n_prb: Number of allocated PRBs
        layer2ue: Mapping from layer index to UE index (0-based), tuple or array
        n_ue: Number of UEs

    Returns:
        Tuple containing:
        - post_eq_noise_var_db: Post-eq noise variance in dB, shape (n_ue,)
        - post_eq_sinr_db: Post-eq SINR in dB, shape (n_ue,)

    Note:
        The equalization error Ree represents noise variance.
        SINR_dB = 10*log10(SNR) = 10*log10(1/Ree)
        noise_var_dB = 10*log10(Ree)
        Therefore: SINR_dB = -noise_var_dB
        Values are averaged over all symbols.
    """
    # Extract allocated subcarriers using array indexing (JIT-compatible)
    alloc_base = 12 * start_prb
    alloc_sc_idxs = alloc_base + jnp.arange(12 * n_prb, dtype=jnp.int32)
    ree__layer_sym_allocfreq = ree__layer_sym_freq[:, :, alloc_sc_idxs]

    # Clamp Ree to avoid division by zero and compute SNR = 1/Ree
    # (Ree represents noise variance, so 1/Ree is the signal-to-noise ratio)
    ree_clamped__layer_sym_allocfreq = jnp.maximum(ree__layer_sym_allocfreq, 1e-10)
    snr__layer_sym_allocfreq = 1.0 / ree_clamped__layer_sym_allocfreq

    # Average over allocated subcarriers: (n_layer, n_sym)
    snr_mean__layer_sym = jnp.mean(snr__layer_sym_allocfreq, axis=2)

    # Aggregate layers -> UEs using one-hot encoding and einsum
    # Create one-hot encoding: (n_layer, n_ue)
    one_hot__layer_ue = jax.nn.one_hot(layer2ue, n_ue, dtype=jnp.float32)

    # Compute layer counts per UE for averaging
    layer_counts__ue = jnp.sum(one_hot__layer_ue, axis=0)  # (n_ue,)

    # Aggregate: (n_layer, n_ue).T @ (n_layer, n_sym) -> (n_ue, n_sym)
    snr_sum__ue_sym = jnp.einsum("lu,ls->us", one_hot__layer_ue, snr_mean__layer_sym)

    # Average by dividing by layer counts (avoid div by zero)
    layer_counts_clamped__ue = jnp.maximum(layer_counts__ue, 1.0)
    snr_mean__ue_sym = snr_sum__ue_sym / layer_counts_clamped__ue[:, None]

    # Convert to dB scale
    # noise_var_dB = 10*log10(Ree) = -10*log10(1/Ree) = -10*log10(SNR)
    # SINR_dB = 10*log10(SNR) = -noise_var_dB
    snr_clamped__ue_sym = jnp.maximum(snr_mean__ue_sym, 1e-10)
    post_eq_sinr_db__ue_sym = db(snr_clamped__ue_sym)
    post_eq_noise_var_db__ue_sym = -post_eq_sinr_db__ue_sym

    # Average over symbols: (n_ue, n_sym) -> (n_ue,)
    post_eq_noise_var_db__ue = jnp.mean(post_eq_noise_var_db__ue_sym, axis=1)
    post_eq_sinr_db__ue = jnp.mean(post_eq_sinr_db__ue_sym, axis=1)

    return post_eq_noise_var_db__ue, post_eq_sinr_db__ue


# =============================================================================
# High-Level Equalizer Function
# =============================================================================


def equalizer(
    xtf__ri_sym_rxant_sc: Array,
    h_interp__ri_port_rxant_sc: Array,
    n_cov__ri_rxant_rxant_prb: Array,
    data_sym_idxs: tuple,
    layer2ue: tuple,
    n_ue: jnp.int32,
    start_prb: jnp.int32,
    n_prb: jnp.int32,
) -> tuple[Array, Array, Array, Array]:
    """Compute and apply MMSE-IRC equalizer with post-equalization metrics.

    This function derives MMSE-IRC equalizer weights from the channel estimates
    and covariance matrix, applies the weights to the received signal, extracts
    data symbols (excluding DMRS symbols), and computes post-equalization metrics.

    Args:
        xtf__ri_sym_rxant_sc: Received resource grid (2, n_sym, n_rxant, n_sc)
        h_interp__ri_port_rxant_sc: Interpolated channel estimates (2, n_port, n_rxant, n_sc)
        n_cov__ri_rxant_rxant_prb: Noise covariance matrix (2, n_rxant, n_rxant, n_prb)
        data_sym_idxs: Data symbol indices as tuple
        layer2ue: Mapping from layer index to UE index as tuple
        n_ue: Number of UEs
        start_prb: 0-based starting PRB index
        n_prb: Number of PRBs in the allocation

    Returns:
        x_est__ri_port_datasym_sc: Equalized data symbols (2, n_port, n_datasym, n_sc)
        ree__port_sc: Post-equalization noise variance (n_port, n_sc)
        post_eq_noise_var_db__ue: Post-eq noise variance per UE (n_ue,)
        post_eq_sinr_db__ue: Post-eq SINR per UE (n_ue,)
    """
    # ---------------------------------------------------------------
    # Equalizer
    # ---------------------------------------------------------------

    # Derive MMSE-IRC equalizer weights
    w__ri_port_rxant_sc, ree__port_sc = get_mmse_irc_weights(
        h_est__ri_port_rxant_sc=h_interp__ri_port_rxant_sc,
        n_cov__ri_rxant_rxant_prb=n_cov__ri_rxant_rxant_prb,
    )

    # Apply MMSE-IRC equalizer weights to received signal
    x_est__ri_port_sym_sc = apply_equalizer(
        xtf__ri_sym_rxant_sc=xtf__ri_sym_rxant_sc,
        w__ri_port_rxant_sc=w__ri_port_rxant_sc,
    )

    # Extract data symbols
    x_est__ri_port_datasym_sc = x_est__ri_port_sym_sc[:, :, data_sym_idxs, :]

    # ---------------------------------------------------------------
    # Post-equalization noise variance and SINR
    # ---------------------------------------------------------------

    # Expand ree to data symbol dimension
    n_datasym = len(data_sym_idxs)
    ree__port_datasym_sc = jnp.broadcast_to(
        ree__port_sc[:, None, :], (ree__port_sc.shape[0], n_datasym, ree__port_sc.shape[1])
    )

    # Compute post-equalization metrics
    post_eq_noise_var_db__ue, post_eq_sinr_db__ue = post_eq_noisevar_sinr(
        ree__layer_sym_freq=ree__port_datasym_sc,
        start_prb=start_prb,
        n_prb=n_prb,
        layer2ue=layer2ue,
        n_ue=n_ue,
    )

    return x_est__ri_port_datasym_sc, ree__port_sc, post_eq_noise_var_db__ue, post_eq_sinr_db__ue


__all__ = [
    "equalizer",
    "apply_equalizer",
    "get_mmse_irc_weights",
    "post_eq_noisevar_sinr",
]
