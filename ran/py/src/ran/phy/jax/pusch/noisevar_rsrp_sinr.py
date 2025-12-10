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

"""Noise variance, RSRP and SINR calculators (JAX-optimized version).

Optimized JAX implementation of noise variance, RSRP, and SINR calculations
for PUSCH receiver processing. This module provides TensorRT-compatible
functions that work with stacked real/imag tensors.

Functions:
- noise_variance_db: Compute noise variance per DMRS position (dB)
- rsrp_db: Compute RSRP per DMRS position and UE (dB)
- sinr_db: Compute SINR per DMRS position and UE (dB)
- noise_rsrp_sinr_db: Compute all three metrics in one call
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def db(x: Array) -> Array:
    """Compute 10*log10(x) elementwise.

    Args:
        x: Linear-scale input (must be positive to avoid -inf results)

    Returns:
        Decibel-scale output as float32
    """
    return 10.0 * jnp.log10(x)


def noise_variance_db(
    noise_var__dsym_prb: Array,
) -> Array:
    """Compute noise variance (dB) as global mean replicated per position.

    Full-slot processing: computes a single global mean of the noise variance
    across all DMRS symbols and PRBs, then replicates it for each position.

    Args:
        noise_var__dsym_prb: Linear noise variances (n_dsym, n_prb)

    Returns:
        Noise variance in dB, shape (n_dsym,) - same value repeated

    Note:
        The +0.5 offset matches the MATLAB reference implementation.
    """
    n_dsym = noise_var__dsym_prb.shape[0]

    # Mean noise var over all PRBs and DMRS symbols
    global_mean = jnp.mean(noise_var__dsym_prb)
    global_mean_clamped = jnp.maximum(global_mean, 1e-10)  # Floor at -100 dB
    result = db(global_mean_clamped) + jnp.float16(0.5)

    # Replicate for each DMRS position to match reference shape (n_dsym,)
    return jnp.full(n_dsym, result, dtype=jnp.float16)


def rsrp_db(
    h_est__ri_port_dsym_rxant_dsc: Array,
    layer2ue__layer: Array | tuple,
    n_prb: int,
    n_ue: int,
    energy: float,
) -> Array:
    """Compute RSRP per DMRS symbol and UE (dB).

    Computes Reference Signal Received Power (RSRP) by:
    1. Computing |H|^2 = real^2 + imag^2 for all channel estimates
    2. Summing across DMRS subcarriers and antennas for each port
    3. Aggregating ports (layers) belonging to the same UE
    4. Normalizing by (n_prb * 12 subcarriers/PRB * n_rxant)
    5. Scaling by energy to account for DMRS transmission power
    6. Taking full-slot average across DMRS symbols
    7. Converting to dB scale

    Args:
        h_est__ri_port_dsym_rxant_dsc: Channel estimates with stacked real/imag,
            shape (2, n_port, n_dsym, n_rxant, n_dsc)
        layer2ue__layer: Mapping from layer index to UE index (0-based), shape (n_layer,)
        n_prb: Number of PRBs for normalization
        n_ue: Number of UEs
        energy: DMRS energy scaling factor (typically 1.0 or 2.0)

    Returns:
        RSRP in dB, shape (n_dsym, n_ue)

    Note:
        The channel estimates H are normalized (divided by sqrt(energy) in matched filter),
        so RSRP = energy * |H|² represents the actual received power with DMRS energy scaling.
        The normalization factor (n_prb * 12 * n_rxant) represents the total number of
        resource elements and antennas over which power is measured.
    """
    n_dsym = h_est__ri_port_dsym_rxant_dsc.shape[2]
    n_rxant = h_est__ri_port_dsym_rxant_dsc.shape[3]

    # Compute |H|^2 = real^2 + imag^2
    h_power__port_dsym_rxant_dsc = (
        h_est__ri_port_dsym_rxant_dsc[0] ** 2 + h_est__ri_port_dsym_rxant_dsc[1] ** 2
    )

    # Sum across DMRS subcarriers and antennas -> (n_port, n_dsym)
    h_power__port_dsym = jnp.sum(
        h_power__port_dsym_rxant_dsc,
        axis=(2, 3),  # Sum over rxant and dsc dimensions
    )

    # Create one-hot encoding: (n_port, n_ue)
    one_hot__port_ue = jax.nn.one_hot(layer2ue__layer, n_ue, dtype=jnp.float16)
    # Aggregate: (n_port, n_ue).T @ (n_port, n_dsym) -> (n_ue, n_dsym)
    rsrp_lin__ue_dsym = jnp.einsum("pu,pd->ud", one_hot__port_ue, h_power__port_dsym)

    # Normalize by (n_prb * 12 * n_rxant) and scale by energy
    # Since H is normalized (divided by sqrt(energy)), |H|² needs to be scaled by energy
    # to represent the actual received power: RSRP = energy * |H|²
    rsrp_lin__ue_dsym /= n_prb * 12 * n_rxant
    rsrp_lin__ue_dsym *= energy

    # Full-slot average: compute mean over DMRS symbols and replicate
    mean_over_dsym__ue = jnp.mean(rsrp_lin__ue_dsym, axis=1, keepdims=True)
    rsrp_lin__ue_dsym = jnp.repeat(mean_over_dsym__ue, repeats=n_dsym, axis=1)

    # Convert to dB, clamping to avoid log of zero
    rsrp_clamped__ue_dsym = jnp.maximum(rsrp_lin__ue_dsym, 1e-10)  # Floor at -100 dB
    rsrp_db__ue_dsym = db(rsrp_clamped__ue_dsym)

    # Transpose to match reference shape (n_dsym, n_ue)
    return rsrp_db__ue_dsym.T


def sinr_db(
    rsrp_db__dsym_ue: Array,
    noise_var_db__dsym: Array,
) -> Array:
    """Compute SINR per DMRS symbol and UE (dB).

    SINR (Signal-to-Interference-plus-Noise Ratio) is computed as the difference
    between RSRP and noise variance in dB scale:
        SINR_dB = RSRP_dB - NoiseVar_dB

    Args:
        rsrp_db__dsym_ue: RSRP in dB, shape (n_dsym, n_ue)
        noise_var_db__dsym: Noise variance in dB, shape (n_dsym,)

    Returns:
        SINR in dB, shape (n_dsym, n_ue)

    Note:
        Broadcasting subtracts noise_var_db (n_dsym,) from each UE column.
    """
    # Reshape noise_var_db to (n_dsym, 1) for explicit broadcasting across UEs
    return rsrp_db__dsym_ue - noise_var_db__dsym[:, None]


def noise_rsrp_sinr_db(
    noise_var__dsym_prb: Array,
    h_est__ri_port_dsym_rxant_dsc: Array,
    layer2ue__layer: Array | tuple,
    n_prb: int,
    n_ue: int,
    energy: float,
) -> tuple[Array, Array, Array]:
    """Compute noise variance, RSRP, and SINR per DMRS symbol and UE (all in dB).

    This function computes all three signal quality metrics in one call:
    - Noise variance: Measure of interference and thermal noise (global mean replicated)
    - RSRP: Reference Signal Received Power (signal strength)
    - SINR: Signal-to-Interference-plus-Noise Ratio (signal quality)

    Args:
        noise_var__dsym_prb: Linear noise variances, shape (n_dsym, n_prb)
        h_est__ri_port_dsym_rxant_dsc: Channel estimates with stacked real/imag,
            shape (2, n_port, n_dsym, n_rxant, n_dsc)
        layer2ue__layer: Mapping from layer index to UE index (0-based),
            shape (n_layer,)
        n_prb: Number of PRBs
        n_ue: Number of UEs
        energy: DMRS energy scaling factor (typically 1.0 or 2.0)

    Returns:
        Tuple containing:
        - noise_var_db__dsym: Noise variance in dB, shape (n_dsym,)
        - rsrp_db__dsym_ue: RSRP in dB, shape (n_dsym, n_ue)
        - sinr_db__dsym_ue: SINR in dB, shape (n_dsym, n_ue)

    Example:
        >>> noise_var = jnp.ones((2, 273), dtype=jnp.float16) * 0.01
        >>> h_est = jnp.stack([jnp.randn(1, 2, 4, 3276), jnp.randn(1, 2, 4, 3276)], axis=0)
        >>> layer2ue = jnp.array([0], dtype=jnp.int32)
        >>> noise_db, rsrp, sinr = noise_rsrp_sinr_db(
        ...     noise_var, h_est, layer2ue, 273, 1
        ... )
        >>> noise_db.shape
        (2,)
        >>> rsrp.shape
        (2, 1)
        >>> sinr.shape
        (2, 1)
    """
    # Compute noise variance in dB, shape (n_dsym,)
    noise_var_db__dsym = noise_variance_db(noise_var__dsym_prb)

    # Compute RSRP in dB, shape (n_dsym, n_ue)
    rsrp_db__dsym_ue = rsrp_db(
        h_est__ri_port_dsym_rxant_dsc,
        layer2ue__layer,
        n_prb,
        n_ue,
        energy,
    )

    # Compute SINR in dB, shape (n_dsym, n_ue)
    sinr_db__dsym_ue = sinr_db(rsrp_db__dsym_ue, noise_var_db__dsym)

    return noise_var_db__dsym, rsrp_db__dsym_ue, sinr_db__dsym_ue


__all__ = [
    "noise_rsrp_sinr_db",
    "noise_variance_db",
    "rsrp_db",
    "sinr_db",
]
