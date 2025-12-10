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

"""JAX-optimized soft demapper for PUSCH Rx pipeline.

Implements simplified cuPHY-style piecewise LLR mapping for QAM constellations.
Based on the cleaner implementation from ran.phy.numpy.pusch.soft_demapper.
Optimized for JIT compilation and TensorRT execution.
"""

from jax import Array, numpy as jnp


# =============================================================================
# Internal LLR Mapping
# =============================================================================


def _simplified_llr_mapping(
    i_axis__layer_allocsc_sym: Array,
    q_axis__layer_allocsc_sym: Array,
    qam_bits: int,
) -> Array:
    """Piecewise simplified LLR mapping (pre-variance scaling).

    Args:
        i_axis__layer_allocsc_sym: real axis values (n_layer, n_allocfreq, n_time)
        q_axis__layer_allocsc_sym: imag axis values (n_layer, n_allocfreq, n_time) (ignored for BPSK)
        qam_bits: number of bits per QAM symbol

    Returns:
        LLR tensor (8, n_layer, n_allocfreq, n_time), scaled by constellation factor only.
        Output is always 8 bits (max for 256QAM) with unused bits zero-filled.

    Axis convention:
        - n_layer: UE streams/layers
        - n_allocfreq: frequency bins/subcarriers in allocated band (12 * n_prb)
        - n_time: OFDM symbols in the demapped interval
    """
    n_layer, n_allocfreq, n_time = i_axis__layer_allocsc_sym.shape

    # Always output 8 bits for consistent shape (256QAM max)
    llr = jnp.zeros((8, n_layer, n_allocfreq, n_time), dtype=jnp.float16)

    # Compute all QAM mappings (JAX will optimize unused branches)
    # 256QAM (8 bits)
    a_256 = jnp.float16(1.0 / jnp.sqrt(170.0))
    llr_256 = llr.at[0].set(i_axis__layer_allocsc_sym)
    llr_256 = llr_256.at[2].set(-jnp.abs(i_axis__layer_allocsc_sym) + 8.0 * a_256)
    llr_256 = llr_256.at[4].set(
        -jnp.abs(jnp.abs(i_axis__layer_allocsc_sym) - 8.0 * a_256) + 4.0 * a_256
    )
    llr_256 = llr_256.at[6].set(
        -jnp.abs(jnp.abs(jnp.abs(i_axis__layer_allocsc_sym) - 8.0 * a_256) - 4.0 * a_256)
        + 2.0 * a_256
    )
    llr_256 = llr_256.at[1].set(q_axis__layer_allocsc_sym)
    llr_256 = llr_256.at[3].set(-jnp.abs(q_axis__layer_allocsc_sym) + 8.0 * a_256)
    llr_256 = llr_256.at[5].set(
        -jnp.abs(jnp.abs(q_axis__layer_allocsc_sym) - 8.0 * a_256) + 4.0 * a_256
    )
    llr_256 = llr_256.at[7].set(
        -jnp.abs(jnp.abs(jnp.abs(q_axis__layer_allocsc_sym) - 8.0 * a_256) - 4.0 * a_256)
        + 2.0 * a_256
    )
    llr_256 = llr_256 * (2.0 * a_256)

    # 64QAM (6 bits)
    a_64 = jnp.float16(1.0 / jnp.sqrt(42.0))
    llr_64 = llr.at[0].set(i_axis__layer_allocsc_sym)
    llr_64 = llr_64.at[2].set(-jnp.abs(i_axis__layer_allocsc_sym) + 4.0 * a_64)
    llr_64 = llr_64.at[4].set(
        -jnp.abs(jnp.abs(i_axis__layer_allocsc_sym) - 4.0 * a_64) + 2.0 * a_64
    )
    llr_64 = llr_64.at[1].set(q_axis__layer_allocsc_sym)
    llr_64 = llr_64.at[3].set(-jnp.abs(q_axis__layer_allocsc_sym) + 4.0 * a_64)
    llr_64 = llr_64.at[5].set(
        -jnp.abs(jnp.abs(q_axis__layer_allocsc_sym) - 4.0 * a_64) + 2.0 * a_64
    )
    llr_64 = llr_64 * (2.0 * a_64)

    # 16QAM (4 bits)
    a_16 = jnp.float16(1.0 / jnp.sqrt(10.0))
    llr_16 = llr.at[0].set(i_axis__layer_allocsc_sym)
    llr_16 = llr_16.at[2].set(-jnp.abs(i_axis__layer_allocsc_sym) + 2.0 * a_16)
    llr_16 = llr_16.at[1].set(q_axis__layer_allocsc_sym)
    llr_16 = llr_16.at[3].set(-jnp.abs(q_axis__layer_allocsc_sym) + 2.0 * a_16)
    llr_16 = llr_16 * (2.0 * a_16)

    # QPSK (2 bits)
    a_qpsk = jnp.float16(1.0 / jnp.sqrt(2.0))
    llr_qpsk = llr.at[0].set(i_axis__layer_allocsc_sym)
    llr_qpsk = llr_qpsk.at[1].set(q_axis__layer_allocsc_sym)
    llr_qpsk = llr_qpsk * (2.0 * a_qpsk)

    # BPSK (1 bit)
    a_bpsk = jnp.float16(1.0)
    llr_bpsk = llr.at[0].set(i_axis__layer_allocsc_sym * (2.0 * a_bpsk))

    # Select appropriate LLR based on qam_bits using jnp.select
    llr = jnp.select(
        [qam_bits == 8, qam_bits == 6, qam_bits == 4, qam_bits == 2],
        [llr_256, llr_64, llr_16, llr_qpsk],
        default=llr_bpsk,
    )

    return llr


# =============================================================================
# Format Conversion
# =============================================================================


def convert_llr_to_reference_format(
    llr__ri_pambit_layer_allocfreq_time: Array,
    qam_bits: int,
) -> Array:
    """Convert LLR from JAX TensorRT format to standard reference format.

    Converts from separated real/imag axes to interleaved bit format matching
    standard 5G NR test vectors and reference implementations.

    Args:
        llr__ri_pambit_layer_allocfreq_time: JAX format LLR tensor
            shape (2, max_pam_bits, n_layer, n_allocfreq, n_time)
            - Dim 0: [0]=real axis, [1]=imag axis
            - Dim 1: PAM bit index (0-3)
        qam_bits: QAM modulation order (bits per symbol): 1, 2, 4, 6, or 8

    Returns:
        llr__qambit_layer_allocfreq_time: Reference format LLR tensor
            shape (qam_bits, n_layer, n_allocfreq, n_time)
            - Bits interleaved: [real0, imag0, real1, imag1, ...]
    """
    llr_real = llr__ri_pambit_layer_allocfreq_time[0]  # (4, n_layer, n_allocfreq, n_time)
    llr_imag = llr__ri_pambit_layer_allocfreq_time[1]  # (4, n_layer, n_allocfreq, n_time)

    n_layer = llr_real.shape[1]
    n_allocfreq = llr_real.shape[2]
    n_time = llr_real.shape[3]

    # Create output with correct qam_bits size
    llr_ref = jnp.zeros((qam_bits, n_layer, n_allocfreq, n_time), dtype=jnp.float16)

    # Interleave real and imaginary bits
    # For BPSK (qam_bits=1): only bit 0 (real)
    # For QPSK (qam_bits=2): bit 0 (real), bit 1 (imag)
    # For 16QAM (qam_bits=4): bits 0,2 (real), bits 1,3 (imag)
    # For 64QAM (qam_bits=6): bits 0,2,4 (real), bits 1,3,5 (imag)
    # For 256QAM (qam_bits=8): bits 0,2,4,6 (real), bits 1,3,5,7 (imag)

    pam_bits = qam_bits // 2 if qam_bits > 1 else 1

    if qam_bits == 1:
        # BPSK: only real axis, first PAM bit
        llr_ref = llr_ref.at[0].set(llr_real[0])
    else:
        # QAM: interleave real/imag
        for i in range(pam_bits):
            llr_ref = llr_ref.at[2 * i].set(llr_real[i])
            llr_ref = llr_ref.at[2 * i + 1].set(llr_imag[i])

    return llr_ref


# =============================================================================
# Main Soft Demapper Function
# =============================================================================


def compute_llrs(
    x_est__ri_sc_sym_layer: Array,
    ree__layer_sc_sym: Array,
    n_prb: int,
    start_prb: int,
    nl_offset: int,
    qam_bits: int,
    output_reference_format: bool = False,
) -> Array:
    """JAX-optimized soft demapper with simplified piecewise LLR mapping.

    Computes log-likelihood ratios (LLRs) for received symbols using
    simplified cuPHY-style piecewise mapping. Supports BPSK, QPSK, 16QAM, 64QAM, and 256QAM.

    Args:
        x_est__ri_sc_sym_layer: Estimated symbols with stacked real/imag,
            shape (2, n_freq, n_time, n_layer)
        ree__layer_sc_sym: Noise variance (n_layer, n_freq, n_time)
        n_prb: Number of PRBs in the allocation
        start_prb: Starting PRB index (0-based)
        nl_offset: Layer offset for multi-layer processing
        qam_bits: QAM modulation order (bits per symbol): 1, 2, 4, 6, or 8
        output_reference_format: If True, output in standard reference format
            (qam_bits, n_layer, n_allocfreq, n_time) with interleaved bits.
            If False, output in TensorRT format (2, max_pam_bits, n_layer, n_allocfreq, n_time).
            Default: False for backward compatibility and TensorRT optimization.

    Returns:
        llr__ri_pambit_layer_allocfreq_time: LLR tensor
            If output_reference_format=False (default):
                shape (2, max_pam_bits, n_layer, n_allocfreq, n_time)
                - First dimension: 0=real axis LLRs, 1=imag axis LLRs
                - Second dimension: PAM bit index (max 4 for 256QAM)
                - Output is padded to max_pam_bits=4; unused bits are zero-filled
            If output_reference_format=True:
                shape (qam_bits, n_layer, n_allocfreq, n_time)
                - Bits interleaved: [real0, imag0, real1, imag1, ...]

    Notes:
        - Default TensorRT format is optimized for GPU pipeline compatibility
        - Reference format matches standard 5G NR test vectors
        - BPSK handling includes pi/2 rotation compensation
        - Cleaner implementation than legacy PAM decomposition approach
    """
    n_time = x_est__ri_sc_sym_layer.shape[2]
    n_layer = x_est__ri_sc_sym_layer.shape[3]

    # Extract real and imag components
    x_est_real__sc_sym_layer = x_est__ri_sc_sym_layer[0]
    x_est_imag__sc_sym_layer = x_est__ri_sc_sym_layer[1]

    # Frequency allocation indices (0-based): subcarriers in the allocation
    freq_idx__allocfreq = 12 * start_prb + jnp.arange(12 * n_prb, dtype=jnp.int32)

    # Validate nl_offset parameter
    if nl_offset != 0:
        msg = f"nl_offset must be 0, got {nl_offset}. Multi-layer offset not yet implemented."
        raise ValueError(msg)

    # Layer selection indices
    layer_idx__layer = jnp.arange(n_layer, dtype=jnp.int32)

    # Gather symbols over allocated subcarriers and select layers
    x_real__allocsc_sym_layer = x_est_real__sc_sym_layer[freq_idx__allocfreq, :n_time, :]
    x_real__allocsc_sym_layer = x_real__allocsc_sym_layer[:, :, layer_idx__layer]

    x_imag__allocsc_sym_layer = x_est_imag__sc_sym_layer[freq_idx__allocfreq, :n_time, :]
    x_imag__allocsc_sym_layer = x_imag__allocsc_sym_layer[:, :, layer_idx__layer]

    # Gather noise variance per layer and subcarrier (drop optional sym axis)
    n0__layer_allocfreq_time = ree__layer_sc_sym[layer_idx__layer, :, :]
    n0__layer_allocfreq = n0__layer_allocfreq_time[:, freq_idx__allocfreq, 0]

    # PAM variance per (n_layer, n_allocfreq)
    pam_var__layer_allocsc = jnp.maximum(n0__layer_allocfreq, 1e-6) / 2.0  # Larger minimum

    # Transpose to (n_layer, n_allocfreq, n_time)
    x_real__layer_allocsc_sym = x_real__allocsc_sym_layer.transpose(2, 0, 1)
    x_imag__layer_allocsc_sym = x_imag__allocsc_sym_layer.transpose(2, 0, 1)

    # BPSK: Undo pi/2 rotation per frequency index position
    # Use absolute subcarrier indices (not local indices) for phase alternation
    k__allocsc = freq_idx__allocfreq
    # phase_even = exp(-i*pi/4) = (1-i)/sqrt(2)
    # phase_odd = exp(-i*3*pi/4) = (-1-i)/sqrt(2)
    sqrt2_inv = jnp.float16(1.0 / jnp.sqrt(2.0))
    phase_real_even = sqrt2_inv
    phase_imag_even = -sqrt2_inv
    phase_real_odd = -sqrt2_inv
    phase_imag_odd = -sqrt2_inv

    # Select phase based on subcarrier index parity
    is_odd__allocsc = (k__allocsc & 1) == 1
    phase_real__allocsc = jnp.where(is_odd__allocsc, phase_real_odd, phase_real_even)
    phase_imag__allocsc = jnp.where(is_odd__allocsc, phase_imag_odd, phase_imag_even)

    # Apply rotation: (x_r + i*x_i) * (p_r + i*p_i) = (x_r*p_r - x_i*p_i) + i*(x_r*p_i + x_i*p_r)
    x_rot_real__layer_allocsc_sym = (
        x_real__layer_allocsc_sym * phase_real__allocsc[None, :, None]
        - x_imag__layer_allocsc_sym * phase_imag__allocsc[None, :, None]
    )
    x_rot_imag__layer_allocsc_sym = (
        x_real__layer_allocsc_sym * phase_imag__allocsc[None, :, None]
        + x_imag__layer_allocsc_sym * phase_real__allocsc[None, :, None]
    )

    # Apply rotation only for BPSK
    i_axis__layer_allocsc_sym = jnp.where(
        qam_bits == 1, x_rot_real__layer_allocsc_sym, x_real__layer_allocsc_sym
    )
    q_axis__layer_allocsc_sym = jnp.where(
        qam_bits == 1, x_rot_imag__layer_allocsc_sym, x_imag__layer_allocsc_sym
    )

    # Piecewise simplified mapping per QAM size (pre-variance scaling)
    llr__qambit_layer_allocsc_sym = _simplified_llr_mapping(
        i_axis__layer_allocsc_sym=i_axis__layer_allocsc_sym,
        q_axis__layer_allocsc_sym=q_axis__layer_allocsc_sym,
        qam_bits=qam_bits,
    )

    # Scale by 1 / PAM variance (equals 2 / N0)
    llr__qambit_layer_allocsc_sym = (
        llr__qambit_layer_allocsc_sym / pam_var__layer_allocsc[None, :, :, None]
    )

    # Separate into real and imag axis LLRs for TensorRT compatibility
    # For BPSK (qam_bits=1): all LLR is real axis
    # For QPSK (qam_bits=2): bit 0 is real, bit 1 is imag
    # For 16QAM (qam_bits=4): bits 0,2 are real, bits 1,3 are imag
    # For 64QAM (qam_bits=6): bits 0,2,4 are real, bits 1,3,5 are imag
    # For 256QAM (qam_bits=8): bits 0,2,4,6 are real, bits 1,3,5,7 are imag

    # Extract even indices (real axis) and odd indices (imag axis)
    # llr__qambit_layer_allocsc_sym has shape (8, n_layer, n_allocfreq, n_time)
    # Extract: llr[0, 2, 4, 6] -> real, llr[1, 3, 5, 7] -> imag
    llr_real__pambit_layer_allocfreq_time = llr__qambit_layer_allocsc_sym[
        0::2
    ]  # Shape: (4, n_layer, n_allocfreq, n_time)
    llr_imag__pambit_layer_allocfreq_time = llr__qambit_layer_allocsc_sym[
        1::2
    ]  # Shape: (4, n_layer, n_allocfreq, n_time)

    # Stack real/imag: (2, max_pam_bits, n_layer, n_allocfreq, n_time)
    llr__ri_pambit_layer_allocfreq_time = jnp.stack(
        [llr_real__pambit_layer_allocfreq_time, llr_imag__pambit_layer_allocfreq_time], axis=0
    )

    # Optionally convert to reference format for test vector comparison
    if output_reference_format:
        return convert_llr_to_reference_format(llr__ri_pambit_layer_allocfreq_time, qam_bits)

    return llr__ri_pambit_layer_allocfreq_time


# =============================================================================
# High-Level Soft Demapper Function
# =============================================================================


def soft_demapper(
    x_est__ri_port_datasym_sc: Array,
    ree__port_sc: Array,
    start_prb: jnp.int32,
    nl_offset: jnp.int32,
    qam_bits: jnp.int32,
    n_prb: jnp.int32,
) -> Array:
    """High-level soft demapper interface.

    This function performs soft demapping on the equalized symbols to produce
    log-likelihood ratios (LLRs) in reference format.

    Args:
        x_est__ri_port_datasym_sc: Equalized symbols (2, n_port, n_datasym, n_sc)
        ree__port_sc: Post-equalization noise variance (n_port, n_sc)
        start_prb: Starting PRB index (0-based)
        nl_offset: Layer offset for multi-layer processing
        qam_bits: QAM modulation order (bits per symbol)
        n_prb: Number of PRBs in the allocation

    Returns:
        llr__qambit_layer_allocfreq_time: LLR tensor in reference format
            shape (qam_bits, n_layer, n_allocfreq, n_time)
    """
    # Transpose equalized symbols to match demapper input shape
    x_est__ri_sc_datasym_port = jnp.einsum("abcd->adcb", x_est__ri_port_datasym_sc)

    # Add time dimension to ree: (n_port, n_sc) -> (n_port, n_sc, 1)
    ree__layer_sc_sym = ree__port_sc[:, :, None]

    # Ensure proper types for static arguments
    n_prb_static = int(n_prb)
    start_prb_static = int(start_prb)
    nl_offset_static = int(nl_offset)
    qam_bits_static = int(qam_bits)

    # Soft demapper - outputs in reference format
    llr__qambit_layer_allocfreq_time = compute_llrs(
        x_est__ri_sc_sym_layer=x_est__ri_sc_datasym_port,
        ree__layer_sc_sym=ree__layer_sc_sym,
        n_prb=n_prb_static,
        start_prb=start_prb_static,
        nl_offset=nl_offset_static,
        qam_bits=qam_bits_static,
        output_reference_format=True,
    )

    return llr__qambit_layer_allocfreq_time


__all__ = ["soft_demapper", "compute_llrs", "convert_llr_to_reference_format"]
