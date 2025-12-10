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

"""Optimized PUSCH Rx implementation using JAX and MLIR-TensorRT."""

from dataclasses import dataclass, fields

import numpy as np
from jax import Array
from jax import numpy as jnp

from ran.phy.jax.pusch.channel_estimation import (
    ChannelFilterConfig,
    channel_estimator,
)
from ran.phy.jax.pusch.equalizer import equalizer
from ran.phy.jax.pusch.soft_demapper import soft_demapper

# -----------------------------------------------------------------------------
# Data classes for static and dynamic inputs
# -----------------------------------------------------------------------------


@dataclass
class PuschInnerRxDynamicInputs:
    """Dynamic inputs passed at runtime to TensorRT.

    Attributes
    ----------
    xtf__rxant_sym_sc_ri : np.ndarray
        Received resource grid.
    """

    xtf__rxant_sym_sc_ri: np.ndarray

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
        lines = [f"\nDynamic inputs ({len(fields(self))} total):"]
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
class PuschInnerRxStaticInputs:
    """Static inputs compiled as constants in TensorRT engine.

    These inputs are not passed at runtime.

    Attributes
    ----------
    slot_number : int
        Slot number.
    n_dmrs_id : int
        DMRS identity.
    rww_regularizer_val : float
        Regularization value for covariance matrix.
    start_prb : int
        Starting PRB index (0-based).
    nl_offset : int
        Layer offset for multi-layer processing.
    scids : tuple
        SCID selection per layer.
    apply_cov_shrinkage : bool
        Whether to apply RBLW shrinkage to covariance.
    channel_filter_method : str
        Channel filter method: 'free_energy_filter' or 'ai_tukey_filter'.
    qam_bits : int
        QAM modulation order (bits per symbol).
    dmrs_sym_idxs : tuple
        DMRS symbol indices.
    data_sym_idxs : tuple
        Data symbol indices.
    dmrs_port_nums : tuple
        Per-layer DMRS port bitfield.
    layer2ue : tuple
        Mapping from layer index to UE index.
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
    channel_filter_config : ChannelFilterConfig | None, optional
        Configuration for channel filter.
    """

    slot_number: int
    n_dmrs_id: int
    rww_regularizer_val: float
    start_prb: int
    nl_offset: int
    scids: tuple
    apply_cov_shrinkage: bool
    channel_filter_method: str
    qam_bits: int
    dmrs_sym_idxs: tuple
    data_sym_idxs: tuple
    dmrs_port_nums: tuple
    layer2ue: tuple
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
        lines = [f"\nStatic inputs ({total} total, compiled as constants):"]
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
class PuschInnerRxOutputs:
    """PUSCH Inner Receiver outputs from the pusch_inner_receiver function.

    Attributes
    ----------
    llr__time_allocfreq_layer_qambit : np.ndarray
        Log-likelihood ratios with shape (n_datasym, n_allocsc, n_layer, qam_bits).
    post_eq_noise_var_db__ue_sym : np.ndarray
        Post-equalization noise variance per UE with shape (n_ue,).
    post_eq_sinr_db__ue_sym : np.ndarray
        Post-equalization SINR per UE with shape (n_ue,).
    """

    llr__time_allocfreq_layer_qambit: np.ndarray
    post_eq_noise_var_db__ue_sym: np.ndarray
    post_eq_sinr_db__ue_sym: np.ndarray

    @classmethod
    def from_tuple(cls, outputs: tuple) -> "PuschInnerRxOutputs":
        """Create PuschInnerRxOutputs from tuple.

        Parameters
        ----------
        outputs : tuple
            Tuple returned by pusch_inner_receiver function.

        Returns
        -------
        PuschInnerRxOutputs
            Instance with named fields.
        """
        return cls(
            llr__time_allocfreq_layer_qambit=outputs[0],
            post_eq_noise_var_db__ue_sym=outputs[1],
            post_eq_sinr_db__ue_sym=outputs[2],
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
        """Return debug information about outputs."""
        lines = [f"\nOutputs ({len(fields(self))} total):"]
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
# PUSCH Inner Receiver function
# -----------------------------------------------------------------------------


def pusch_inner_rx(
    # Dynamic inputs (runtime)
    xtf__rxant_sym_sc_ri: Array,
    # Static inputs (compile-time)
    slot_number: jnp.int32,
    n_dmrs_id: jnp.int32,
    rww_regularizer_val: jnp.float32,
    start_prb: jnp.int32,
    nl_offset: jnp.int32,
    scids: tuple,
    apply_cov_shrinkage: bool,
    channel_filter_method: str,
    qam_bits: jnp.int32,
    dmrs_sym_idxs: tuple,
    data_sym_idxs: tuple,
    dmrs_port_nums: tuple,
    layer2ue: tuple,
    n_prb: jnp.int32,
    n_ue: jnp.int32,
    n_f: jnp.int32,
    n_t: jnp.int32,
    energy: jnp.float32,
    channel_filter_config: ChannelFilterConfig | None = None,
) -> tuple[Array, Array, Array]:
    """PUSCH Inner Receiver function.

    The PUSCH inner receiver performs the following steps:

    1. DMRS-based channel estimation and covariance estimation
    2. MMSE-IRC equalization
    3. Soft demapping and LLR generation

    The function returns LLRs and post-equalization noise variance
    and SINR estimates.

    The function can be compiled with MLIR-TensorRT to a single
    TensorRT engine for inclusion in higher-performance C++ pipelines.

    The function has both dynamic and static arguments (static
    arguments are fixed at compile time).

    Parameters
    ----------
    xtf__rxant_sym_sc_ri : Array
        Received resource grid with shape (n_rxant, n_sym, n_sc, 2).
    slot_number : jnp.int32
        Slot number (compile-time constant).
    n_dmrs_id : jnp.int32
        DMRS identity (compile-time constant).
    rww_regularizer_val : jnp.float32
        Regularization value for covariance matrix (compile-time constant).
    start_prb : jnp.int32
        0-based starting PRB index for the allocation (compile-time constant).
    nl_offset : jnp.int32
        Layer offset for multi-layer processing (compile-time constant).
    scids : tuple
        SCID selection (0 or 1) per layer as tuple (compile-time constant).
    apply_cov_shrinkage : bool
        Whether to apply RBLW shrinkage to covariance (compile-time constant).
    channel_filter_method : str
        Channel filter method: 'free_energy_filter' or 'ai_tukey_filter' (compile-time
        constant).
    qam_bits : jnp.int32
        QAM modulation order (bits per symbol): 1, 2, 4, 6, or 8 (compile-time constant).
    dmrs_sym_idxs : tuple
        DMRS symbol indices as tuple (compile-time constant).
    data_sym_idxs : tuple
        Data symbol indices as tuple (compile-time constant).
    dmrs_port_nums : tuple
        Per-layer DMRS port bitfield as tuple (compile-time constant).
    layer2ue : tuple
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
        is 'ai_tukey_filter' (compile-time constant).

    Returns
    -------
    llr__time_allocfreq_layer_qambit : Array
        LLRs with shape (n_datasym, n_allocsc, n_layer, qam_bits).
    post_eq_noise_var_db__ue : Array
        Post-equalization noise variance per UE with shape (n_ue,).
    post_eq_sinr_db__ue : Array
        Post-equalization SINR per UE with shape (n_ue,).
    """
    # Convert input format
    # C++ pipeline emits (column-major, real/imag interleaved).
    # Transpose for row-major layout needed by TensorRT (row-major, real/imag stacked)
    xtf__ri_sym_rxant_sc = jnp.einsum("abcd->dbac", xtf__rxant_sym_sc_ri)

    # Channel estimation
    (
        h_interp__ri_port_rxant_sc,
        n_cov__ri_rxant_rxant_prb,
        _h_est__ri_port_dsym_rxant_dsc,  # Unused in main pipeline
        _noise_var_db,
        _rsrp_db__ue_dsym,
        _sinr_db__ue_dsym,
    ) = channel_estimator(
        xtf__ri_sym_rxant_sc=xtf__ri_sym_rxant_sc,
        slot_number=slot_number,
        n_dmrs_id=n_dmrs_id,
        rww_regularizer_val=rww_regularizer_val,
        start_prb=start_prb,
        scids=scids,
        apply_cov_shrinkage=apply_cov_shrinkage,
        channel_filter_method=channel_filter_method,
        dmrs_sym_idxs=dmrs_sym_idxs,
        dmrs_port_nums=dmrs_port_nums,
        layer2ue=layer2ue,
        n_prb=n_prb,
        n_ue=n_ue,
        n_f=n_f,
        n_t=n_t,
        energy=energy,
        channel_filter_config=channel_filter_config,
    )

    # MIMO equalizer
    (x_est__ri_port_datasym_sc, ree__port_sc, post_eq_noise_var_db__ue, post_eq_sinr_db__ue) = (
        equalizer(
            xtf__ri_sym_rxant_sc=xtf__ri_sym_rxant_sc,
            h_interp__ri_port_rxant_sc=h_interp__ri_port_rxant_sc,
            n_cov__ri_rxant_rxant_prb=n_cov__ri_rxant_rxant_prb,
            data_sym_idxs=data_sym_idxs,
            layer2ue=layer2ue,
            n_ue=n_ue,
            start_prb=start_prb,
            n_prb=n_prb,
        )
    )

    # Soft demapping
    llr__qambit_layer_allocfreq_time = soft_demapper(
        x_est__ri_port_datasym_sc=x_est__ri_port_datasym_sc,
        ree__port_sc=ree__port_sc,
        start_prb=start_prb,
        nl_offset=nl_offset,
        qam_bits=qam_bits,
        n_prb=n_prb,
    )

    # Output LLR format conversion.
    # C++ pipeline ingests column-major interleaved format (time, freq, layer, qam_bits)
    # Also cast to float16 to match C++ pipeline.
    llr__time_allocfreq_layer_qambit = jnp.copy(
        jnp.einsum("abcd->dcba", llr__qambit_layer_allocfreq_time)
    ).astype(jnp.float16)

    return (
        llr__time_allocfreq_layer_qambit,
        post_eq_noise_var_db__ue,
        post_eq_sinr_db__ue,
    )
