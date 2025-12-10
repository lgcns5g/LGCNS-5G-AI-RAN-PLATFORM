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

"""PUSCH channel estimator implementation."""

from dataclasses import dataclass, fields
from typing import TypeAlias

import numpy as np
from jax import Array
from jax import numpy as jnp

from ran.trt_plugins.dmrs import (
    dmrs_3276,
    extract_raw_dmrs_type1,
    gen_transmitted_dmrs_with_occ,
)

from ran.phy.jax.pusch.ai_tukey_filter import ai_tukey_filter, AITukeyFilterConfig
from ran.phy.jax.pusch.free_energy_filter import free_energy_filter, FreeEnergyFilterConfig
from ran.phy.jax.pusch.identity_filter import identity_filter, IdentityFilterConfig
from ran.phy.jax.pusch.weighted_threshold_filter import (
    weighted_threshold_filter,
    WeightedThresholdFilterConfig,
)
from ran.phy.jax.pusch.noise_estimation import apply_shrinkage, estimate_covariance
from ran.phy.jax.pusch.noisevar_rsrp_sinr import noise_rsrp_sinr_db
from ran.phy.jax.utils import complex_mul_conj

# Type alias for channel filter configurations
ChannelFilterConfig: TypeAlias = (
    AITukeyFilterConfig
    | FreeEnergyFilterConfig
    | IdentityFilterConfig
    | WeightedThresholdFilterConfig
)

# -----------------------------------------------------------------------------
# Data classes for static and dynamic inputs
# -----------------------------------------------------------------------------


@dataclass
class ChannelEstimatorDynamicInputs:
    """Dynamic inputs passed at runtime (not compiled as constants).

    Attributes
    ----------
    xtf__ri_sym_rxant_sc : np.ndarray
        Received resource grid as float32 array with shape
        (2, n_sym, n_rxant, n_sc) with real/imaginary components stacked in dim0.
    """

    xtf__ri_sym_rxant_sc: np.ndarray

    def to_tuple(self) -> tuple:
        """Convert to ordered tuple for positional function call.

        Returns
        -------
        tuple
            Ordered tuple of field values.
        """
        return tuple(getattr(self, field.name) for field in fields(self))

    def __str__(self) -> str:
        """Return information about inputs."""
        lines = [f"\nChannel Estimator Dynamic inputs ({len(fields(self))} total):"]
        for i, field in enumerate(fields(self)):
            value = getattr(self, field.name)
            if hasattr(value, "shape"):
                shape_str = f"shape={value.shape}"
            else:
                shape_str = f"value={value}"
            dtype_str = ""
            if hasattr(value, "dtype"):
                dtype_str = f", dtype={value.dtype}"
            lines.append(f"  [{i}] {field.name}: {shape_str}{dtype_str}")
        return "\n".join(lines)


@dataclass
class ChannelEstimatorStaticInputs:
    """Static inputs compiled as constants (not passed at runtime).

    Attributes
    ----------
    slot_number : int
        Slot number for DMRS generation.
    n_dmrs_id : int
        DMRS identity parameter.
    rww_regularizer_val : float
        Regularization value for covariance matrix.
    start_prb : int
        Starting PRB index (0-based).
    scids : tuple[int, ...]
        SCID selection per layer (0 or 1) as tuple of integers.
    apply_cov_shrinkage : bool
        Whether to apply RBLW shrinkage to covariance.
    channel_filter_method : str
        Channel filter method: 'free_energy_filter', 'ai_tukey_filter',
        'weighted_threshold_filter', or 'identity_filter'.
    dmrs_sym_idxs : tuple[int, ...]
        DMRS symbol indices as tuple of integers.
    dmrs_port_nums : tuple[int, ...]
        Per-layer DMRS port bitfield as tuple of integers.
    layer2ue : tuple[int, ...]
        Mapping from layer index to UE index as tuple of integers.
    n_prb : int
        Number of PRBs in the allocation.
    n_ue : int
        Number of UEs.
    n_f : int
        Number of subcarriers in the full resource grid.
    n_t : int
        Number of OFDM symbols per slot.
    energy : float
        Energy scaling factor for DMRS transmission.
    channel_filter_config : ChannelFilterConfig | None
        Configuration for channel filter. Required when channel_filter_method
        is 'ai_tukey_filter'.
    """

    slot_number: int
    n_dmrs_id: int
    rww_regularizer_val: float
    start_prb: int
    scids: tuple[int, ...]
    apply_cov_shrinkage: bool
    channel_filter_method: str
    dmrs_sym_idxs: tuple[int, ...]
    dmrs_port_nums: tuple[int, ...]
    layer2ue: tuple[int, ...]
    n_prb: int
    n_ue: int
    n_f: int
    n_t: int
    energy: float
    channel_filter_config: ChannelFilterConfig | None = None

    def to_tuple(self) -> tuple:
        """Convert to ordered tuple for positional function call.

        Returns
        -------
        tuple
            Ordered tuple of field values.
        """
        return tuple(getattr(self, field.name) for field in fields(self))

    def __str__(self) -> str:
        """Return information about inputs."""
        total = len(fields(self))
        lines = [f"\nChannel Estimator Static inputs ({total} total, compiled as constants):"]
        for i, field in enumerate(fields(self)):
            value = getattr(self, field.name)
            if hasattr(value, "shape"):
                shape_str = f"shape={value.shape}"
            else:
                shape_str = f"value={value}"
            dtype_str = ""
            if hasattr(value, "dtype"):
                dtype_str = f", dtype={value.dtype}"
            lines.append(f"  [{i}] {field.name}: {shape_str}{dtype_str}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Data classes for outputs
# -----------------------------------------------------------------------------


@dataclass
class ChannelEstimatorOutputs:
    """Channel estimator outputs from the channel_estimator function.

    Attributes
    ----------
    h_interp__ri_port_rxant_sc : np.ndarray
        Interpolated channel estimates with shape (2, n_port, n_rxant, n_sc).
    n_cov__ri_rxant_rxant_prb : np.ndarray
        Noise covariance matrix with shape (2, n_rxant, n_rxant, n_prb).
    h_est__ri_port_dsym_rxant_dsc : np.ndarray
        Channel estimates per DMRS symbol with shape
        (2, n_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    noise_var_db : np.ndarray
        Noise variance in dB with shape (1,).
    rsrp_db__ue_dsym : np.ndarray
        RSRP per UE and DMRS symbol with shape (n_ue, n_dmrs_syms).
    sinr_db__ue_dsym : np.ndarray
        SINR per UE and DMRS symbol with shape (n_ue, n_dmrs_syms).
    """

    h_interp__ri_port_rxant_sc: np.ndarray
    n_cov__ri_rxant_rxant_prb: np.ndarray
    h_est__ri_port_dsym_rxant_dsc: np.ndarray
    noise_var_db: np.ndarray
    rsrp_db__ue_dsym: np.ndarray
    sinr_db__ue_dsym: np.ndarray

    @classmethod
    def from_tuple(cls, outputs: tuple) -> "ChannelEstimatorOutputs":
        """Create ChannelEstimatorOutputs from channel_estimator output tuple.

        Parameters
        ----------
        outputs : tuple
            Tuple of 6 arrays returned by channel_estimator function.

        Returns
        -------
        ChannelEstimatorOutputs
            Instance with named fields.
        """
        return cls(
            h_interp__ri_port_rxant_sc=outputs[0],
            n_cov__ri_rxant_rxant_prb=outputs[1],
            h_est__ri_port_dsym_rxant_dsc=outputs[2],
            noise_var_db=outputs[3],
            rsrp_db__ue_dsym=outputs[4],
            sinr_db__ue_dsym=outputs[5],
        )

    def to_tuple(self) -> tuple:
        """Convert to ordered tuple.

        Returns
        -------
        tuple
            Ordered tuple of field values.
        """
        return tuple(getattr(self, field.name) for field in fields(self))

    def __str__(self) -> str:
        """Return information about outputs."""
        lines = [f"\nChannel Estimator Outputs ({len(fields(self))} total):"]
        for i, field in enumerate(fields(self)):
            value = getattr(self, field.name)
            if hasattr(value, "shape"):
                shape_str = f"shape={value.shape}"
            else:
                shape_str = f"value={value}"
            dtype_str = ""
            if hasattr(value, "dtype"):
                dtype_str = f", dtype={value.dtype}"
            lines.append(f"  [{i}] {field.name}: {shape_str}{dtype_str}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Helper functions for channel estimation
# -----------------------------------------------------------------------------


def dmrs_matched_filter(
    y_dmrs__ri_dsym_rxant_dsc: Array,
    x_dmrs__ri_port_dsym_dsc: Array,
    energy: float,
) -> Array:
    """Apply matched filter to DMRS: H = y * conj(x_dmrs) / sqrt(energy).

    Given raw received signal y and known transmitted reference x_dmrs,
    compute channel estimates via matched filtering. The received signal
    (without DMRS port dimension) is matched with each transmitted DMRS
    port reference to obtain per-port channel estimates. The result is
    normalized by sqrt(energy) to account for DMRS energy scaling.

    Parameters
    ----------
    y_dmrs__ri_dsym_rxant_dsc : Array
        Raw received signal at DMRS positions with shape
        (2, n_dmrs_syms, n_rxant, n_dmrs_sc).
    x_dmrs__ri_port_dsym_dsc : Array
        Transmitted DMRS reference per port with shape
        (2, n_port, n_dmrs_syms, n_dmrs_sc).
    energy : float
        DMRS energy scaling factor (typically 1.0 or 2.0).

    Returns
    -------
    Array
        Channel estimates per port and symbol with shape
        (2, n_port, n_dmrs_syms, n_rxant, n_dmrs_sc).

    Notes
    -----
    Tensor axes have the following meaning:
    - ri: Real/imaginary part.
    - port: DMRS port index (0 to n_port-1).
    - dsym: DMRS symbol index (0 to n_dmrs_syms-1).
    - rxant: Antenna index (0 to n_rxant-1).
    - dsc: DMRS subcarrier index (0 to n_dmrs_sc-1).

    The output is normalized by sqrt(energy) so downstream processing
    doesn't need to know about DMRS energy scaling.
    """
    # Add DMRS port dimension to y:
    # (2, n_dmrs_syms, n_rxant, n_dmrs_sc) -> (2, 1, n_dmrs_syms, n_rxant, n_dmrs_sc)
    y__ri_port_dsym_rxant_dsc = y_dmrs__ri_dsym_rxant_dsc[:, None, :, :, :]

    # Add Rx antenna dimension to x:
    # (2, n_port, n_dmrs_syms, n_dmrs_sc) -> (2, n_port, n_dmrs_syms, 1, n_dmrs_sc)
    x__ri_port_dsym_rxant_dsc = x_dmrs__ri_port_dsym_dsc[:, :, :, None, :]

    # Matched filter: y * conj(x) / sqrt(energy)
    # Shapes: (2, 1, n_sym, n_ant, n_sc) * (2, n_port, n_sym, 1, n_sc)
    # -> (2, n_port, n_sym, n_ant, n_sc)
    # Note: x has unit energy (|x|² = 1.0), but transmitted with sqrt(energy) scaling
    h__ri_port_dsym_rxant_dsc = complex_mul_conj(
        y__ri_port_dsym_rxant_dsc,
        x__ri_port_dsym_rxant_dsc,
    )

    # Normalize by sqrt(energy) to get true channel estimate
    h__ri_port_dsym_rxant_dsc = h__ri_port_dsym_rxant_dsc / jnp.sqrt(energy)

    return h__ri_port_dsym_rxant_dsc


def frequency_interpolation_dmrs_type1(
    h__ri_port_rxant_sc: Array,
    cdm_group_id: int,
) -> Array:
    """Interpolate channel estimates from DMRS subcarriers to all subcarriers.

    Uses linear interpolation between DMRS subcarrier estimates to fill subcarriers.

    For example:
    - DMRS are on even subcarriers: 0, 2, 4, 6, ...
    - Data are on odd subcarriers: 1, 3, 5, 7, ...
    - Interpolation: H[1] = (H[0] + H[2]) / 2, H[3] = (H[2] + H[4]) / 2, etc.

    Args:
        h__ri_port_rxant_sc: Channel estimates at DMRS positions with stacked real/imag,
            shape (2, n_port, n_rxant, n_dmrs_sc) where n_dmrs_sc = 6 * n_prb
        cdm_group_id: CDM group ID (0 or 1) indicating DMRS subcarrier offset
    Returns:
        h_interp__ri_port_rxant_sc: Interpolated channel estimates,
            shape (2, n_port, n_rxant, n_sc) where n_sc = 12 * n_prb
    """

    # Determine the number of CDM groups and the offset for the even/odd subcarriers
    if cdm_group_id == 0:
        offset = 0
    elif cdm_group_id == 1:
        offset = 1
    else:
        error_msg = f"Invalid CDM group ID: {cdm_group_id}"
        raise ValueError(error_msg)

    n_dmrs_sc = h__ri_port_rxant_sc.shape[-1]
    n_sc = 2 * n_dmrs_sc

    # Create output array with double the subcarriers (last dimension)
    # Shape: (2, n_port, n_rxant, n_sc)
    shape = h__ri_port_rxant_sc.shape[:-1] + (n_sc,)
    h_interp__ri_port_rxant_sc = jnp.zeros(shape, dtype=h__ri_port_rxant_sc.dtype)

    # Fill DMRS subcarriers based on CDM group offset
    # CDM group 0 (offset=0): Fill even subcarriers (0, 2, 4, ...)
    # CDM group 1 (offset=1): Fill odd subcarriers (1, 3, 5, ...)
    h_interp__ri_port_rxant_sc = h_interp__ri_port_rxant_sc.at[..., offset::2].set(
        h__ri_port_rxant_sc
    )

    # Interpolate the opposite subcarriers (data subcarriers)
    # CDM group 0: interpolate odd subcarriers (1, 3, 5, ...)
    # CDM group 1: interpolate even subcarriers (0, 2, 4, ...)
    data_offset = 1 - offset

    # Linear interpolation: SC[i] = avg(SC[i-1], SC[i+1])
    # Works for both real (index 0) and imag (index 1) components
    if data_offset == 1:
        # Interpolate odd positions (1, 3, 5, ..., n_sc-3)
        # Note: Using explicit indices to avoid negative indexing which creates dynamic slices
        h_interp__ri_port_rxant_sc = h_interp__ri_port_rxant_sc.at[..., 1 : n_sc - 1 : 2].set(
            (h__ri_port_rxant_sc[..., : n_dmrs_sc - 1] + h__ri_port_rxant_sc[..., 1:n_dmrs_sc]) / 2
        )
        # Handle last odd subcarrier (extrapolate from last DMRS)
        # Use explicit n_sc-1 instead of -1 to avoid dynamic slice
        h_interp__ri_port_rxant_sc = h_interp__ri_port_rxant_sc.at[..., n_sc - 1].set(
            h__ri_port_rxant_sc[..., n_dmrs_sc - 1]
        )
    else:
        # Interpolate even positions (0, 2, 4, ..., n_sc-2)
        # First even position (0) extrapolates from first DMRS
        h_interp__ri_port_rxant_sc = h_interp__ri_port_rxant_sc.at[..., 0].set(
            h__ri_port_rxant_sc[..., 0]
        )
        # Middle even positions interpolate between adjacent DMRS
        # Use explicit indices to avoid negative indexing
        h_interp__ri_port_rxant_sc = h_interp__ri_port_rxant_sc.at[..., 2 : n_sc - 2 : 2].set(
            (h__ri_port_rxant_sc[..., : n_dmrs_sc - 1] + h__ri_port_rxant_sc[..., 1:n_dmrs_sc]) / 2
        )

    return h_interp__ri_port_rxant_sc


# -----------------------------------------------------------------------------
# Channel estimator function
# -----------------------------------------------------------------------------


def channel_estimator(
    xtf__ri_sym_rxant_sc: Array,
    slot_number: jnp.int32,
    n_dmrs_id: jnp.int32,
    rww_regularizer_val: jnp.float32,
    start_prb: jnp.int32,
    scids: tuple[int, ...],
    apply_cov_shrinkage: bool,
    channel_filter_method: str,
    dmrs_sym_idxs: tuple[int, ...],
    dmrs_port_nums: tuple[int, ...],
    layer2ue: tuple[int, ...],
    n_prb: jnp.int32,
    n_ue: jnp.int32,
    n_f: jnp.int32,
    n_t: jnp.int32,
    energy: jnp.float32,
    channel_filter_config: ChannelFilterConfig | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Perform DMRS-based channel estimation and covariance matrix estimation.

    This function handles DMRS generation, extraction, matched filtering,
    channel estimation with FFT-based denoising, frequency interpolation,
    interference+noise covariance estimation, and pre-equalization signal
    quality metrics.

    Parameters
    ----------
    xtf__ri_sym_rxant_sc : Array
        Received resource grid with shape (2, n_sym, n_rxant, n_sc).
    slot_number : jnp.int32
        Slot number (compile-time constant).
    n_dmrs_id : jnp.int32
        DMRS identity (compile-time constant).
    rww_regularizer_val : jnp.float32
        Regularization value for covariance matrix (compile-time constant).
    start_prb : jnp.int32
        0-based starting PRB index for the allocation (compile-time constant).
    scids : tuple[int, ...]
        SCID selection (0 or 1) per layer as tuple (compile-time constant).
    apply_cov_shrinkage : bool
        Whether to apply RBLW shrinkage to covariance (compile-time constant).
    channel_filter_method : str
        Channel filter method: 'free_energy_filter', 'ai_tukey_filter',
        'weighted_threshold_filter', or 'identity_filter' (compile-time constant).
    dmrs_sym_idxs : tuple[int, ...]
        DMRS symbol indices as tuple (compile-time constant).
    dmrs_port_nums : tuple[int, ...]
        Per-layer DMRS port bitfield as tuple (compile-time constant).
    layer2ue : tuple[int, ...]
        Mapping from layer index to UE index as tuple (compile-time constant).
    n_prb : jnp.int32
        Number of PRBs in the allocation (compile-time constant).
    n_ue : jnp.int32
        Number of UEs (compile-time constant).
    n_f : jnp.int32
        Number of subcarriers in the full resource grid (compile-time constant).
    n_t : jnp.int32
        Number of OFDM symbols per slot (compile-time constant).
    energy : jnp.float32
        Energy scaling factor for DMRS transmission (compile-time constant).
    channel_filter_config : ChannelFilterConfig | None, optional
        Configuration for channel filter. Required when channel_filter_method
        is 'ai_tukey_filter'.

    Returns
    -------
    h_interp__ri_port_rxant_sc : Array
        Interpolated channel estimates with shape (2, n_port, n_rxant, n_sc).
    n_cov__ri_rxant_rxant_prb : Array
        Noise covariance matrix with shape (2, n_rxant, n_rxant, n_prb).
    h_est__ri_port_dsym_rxant_dsc : Array
        Channel estimates per DMRS symbol with shape
        (2, n_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    noise_var_db : Array
        Noise variance in dB with shape (1,).
    rsrp_db__ue_dsym : Array
        RSRP per UE and DMRS symbol with shape (n_ue, n_dmrs_syms).
    sinr_db__ue_dsym : Array
        SINR per UE and DMRS symbol with shape (n_ue, n_dmrs_syms).

    Raises
    ------
    ValueError
        If channel_filter_config is None when channel_filter_method is
        'ai_tukey_filter', or if channel_filter_method is invalid.
    TypeError
        If channel_filter_config is not AITukeyFilterConfig when
        channel_filter_method is 'ai_tukey_filter'.
    """
    # ---------------------------------------------------------------
    # DMRS Generation and extraction
    # ---------------------------------------------------------------

    # Extract raw received signal on the DMRS resource elements
    y_dmrs__ri_dsym_rxant_dsc, n_dmrs_sc_per_prb, dmrs_port_grid_cfgs = extract_raw_dmrs_type1(
        xtf__ri_sym_rxant_sc=xtf__ri_sym_rxant_sc,
        dmrs_sym_idxs=jnp.asarray(dmrs_sym_idxs),
        n_prb=n_prb,
        start_prb=start_prb,
        dmrs_port_nums=dmrs_port_nums,
    )

    # Generate DMRS
    r_dmrs__ri_sym_cdm_dsc, _ = dmrs_3276(
        slot_number=slot_number,
        n_dmrs_id=n_dmrs_id,
    )
    # Convert to float16
    r_dmrs__ri_sym_cdm_dsc = jnp.array(r_dmrs__ri_sym_cdm_dsc, dtype=jnp.float16)

    # Compute n_dmrs_sc from static n_prb (DMRS Type 1: 6 subcarriers per PRB)
    n_dmrs_sc = 6 * n_prb

    # Generate the transmitted DMRS reference signal with OCC
    x_dmrs__ri_port_dsym_dsc = gen_transmitted_dmrs_with_occ(
        r_dmrs__ri_sym_cdm_dsc=r_dmrs__ri_sym_cdm_dsc,
        dmrs_port_nums=dmrs_port_nums,
        scids=scids,
        dmrs_sym_idxs=dmrs_sym_idxs,
        n_dmrs_sc=n_dmrs_sc,
    )

    # ---------------------------------------------------------------
    # Channel estimation
    # ---------------------------------------------------------------

    # Matched filter
    h__ri_port_dsym_rxant_dsc = dmrs_matched_filter(
        y_dmrs__ri_dsym_rxant_dsc=y_dmrs__ri_dsym_rxant_dsc,
        x_dmrs__ri_port_dsym_dsc=x_dmrs__ri_port_dsym_dsc,
        energy=energy,
    )

    # Filter channel (n_dmrs_sc already computed above from static n_prb)
    if channel_filter_method == "free_energy_filter":
        # Type narrow: free_energy_filter expects FreeEnergyFilterConfig or None
        free_energy_config = (
            channel_filter_config
            if isinstance(channel_filter_config, FreeEnergyFilterConfig)
            or channel_filter_config is None
            else None
        )
        h_est__ri_port_dsym_rxant_dsc = free_energy_filter(
            h__ri_port_dsym_rxant_dsc,
            n_dmrs_sc=n_dmrs_sc,
            config=free_energy_config,
        )
    elif channel_filter_method == "ai_tukey_filter":
        if channel_filter_config is None or not isinstance(
            channel_filter_config, AITukeyFilterConfig
        ):
            raise ValueError(
                "channel_filter_config must be AITukeyFilterConfig for 'ai_tukey_filter'"
            )
        h_est__ri_port_dsym_rxant_dsc = ai_tukey_filter(
            h__ri_port_dsym_rxant_dsc, channel_filter_config, n_dmrs_sc=n_dmrs_sc
        )
    elif channel_filter_method == "weighted_threshold_filter":
        # Type narrow: weighted_threshold_filter expects WeightedThresholdFilterConfig or None
        weighted_threshold_config = (
            channel_filter_config
            if isinstance(channel_filter_config, WeightedThresholdFilterConfig)
            or channel_filter_config is None
            else None
        )
        h_est__ri_port_dsym_rxant_dsc = weighted_threshold_filter(
            h__ri_port_dsym_rxant_dsc,
            n_dmrs_sc=n_dmrs_sc,
            config=weighted_threshold_config,
        )
    elif channel_filter_method == "identity_filter":
        # Type narrow: identity_filter expects IdentityFilterConfig or None
        identity_config = (
            channel_filter_config
            if isinstance(channel_filter_config, IdentityFilterConfig)
            or channel_filter_config is None
            else None
        )
        h_est__ri_port_dsym_rxant_dsc = identity_filter(
            h__ri_port_dsym_rxant_dsc,
            n_dmrs_sc=n_dmrs_sc,
            config=identity_config,
        )
    else:
        raise ValueError(f"Invalid channel filter method: {channel_filter_method}")

    # Average across DMRS symbols
    h_est__ri_port_rxant_dsc = jnp.mean(h_est__ri_port_dsym_rxant_dsc, axis=2)

    # Interpolate channel estimates to all subcarriers
    if all(g == dmrs_port_grid_cfgs[0] for g in dmrs_port_grid_cfgs):
        cdm_group_id = dmrs_port_grid_cfgs[0]
        h_interp__ri_port_rxant_sc = frequency_interpolation_dmrs_type1(
            h__ri_port_rxant_sc=h_est__ri_port_rxant_dsc,
            cdm_group_id=cdm_group_id,
        )
    else:
        # Nothing to interpolate
        h_interp__ri_port_rxant_sc = h_est__ri_port_rxant_dsc

    # ---------------------------------------------------------------
    # Covariance estimation
    # ---------------------------------------------------------------

    # Estimate interference+noise covariance
    n_cov__ri_rxant_rxant_prb, mean_noise_var__prb = estimate_covariance(
        y_dmrs__ri_dsym_rxant_dsc=y_dmrs__ri_dsym_rxant_dsc,
        x_dmrs__ri_port_dsym_dsc=x_dmrs__ri_port_dsym_dsc,
        h_est__ri_port_rxant_dsc=h_est__ri_port_rxant_dsc,
        n_prb=n_prb,
        rww_regularizer_val=rww_regularizer_val,
        n_dmrs_sc_per_prb=n_dmrs_sc_per_prb,
        energy=energy,
    )

    # Optionally apply shrinkage to covariance estimate
    if apply_cov_shrinkage:
        n_samples_per_prb = 6 * len(dmrs_sym_idxs)
        n_cov__ri_rxant_rxant_prb = apply_shrinkage(
            n_cov__ri_rxant_rxant_prb,
            n_samples_per_prb,
        )

    # ---------------------------------------------------------------
    # Noise variance, RSRP, and SINR computation
    # ---------------------------------------------------------------

    # Expand mean_noise_var from (n_prb,) to (n_dsym, n_prb)
    noise_var__dsym_prb = mean_noise_var__prb[None, :]

    # Compute noise variance, RSRP, and SINR
    noise_var_db__dsym, rsrp_db__dsym_ue, sinr_db__dsym_ue = noise_rsrp_sinr_db(
        noise_var__dsym_prb=noise_var__dsym_prb,
        h_est__ri_port_dsym_rxant_dsc=h_est__ri_port_dsym_rxant_dsc,
        layer2ue__layer=layer2ue,
        n_prb=n_prb,
        n_ue=n_ue,
        energy=energy,
    )

    # Convert to single-rank tensor
    noise_var_db = noise_var_db__dsym[0:1]

    # Transpose to match output format: (n_ue, n_dsym)
    rsrp_db__ue_dsym = rsrp_db__dsym_ue.T
    sinr_db__ue_dsym = sinr_db__dsym_ue.T

    return (
        h_interp__ri_port_rxant_sc,
        n_cov__ri_rxant_rxant_prb,
        h_est__ri_port_dsym_rxant_dsc,
        noise_var_db,
        rsrp_db__ue_dsym,
        sinr_db__ue_dsym,
    )
