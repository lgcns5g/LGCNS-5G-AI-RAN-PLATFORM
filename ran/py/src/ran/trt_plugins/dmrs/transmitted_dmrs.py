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

import jax
from jax import Array, numpy as jnp


def gen_transmitted_dmrs_with_occ(
    r_dmrs__ri_sym_cdm_dsc: Array,
    dmrs_port_nums: Array | tuple,
    scids: tuple,
    dmrs_sym_idxs: tuple,
    n_dmrs_sc: int,
) -> Array:
    """Compute transmitted DMRS reference signal.

    Compute transmitted DMRS symbols per layer including frequency
    and time orthogonal cover code (OCC) patterns.

    Parameters
    ----------
    r_dmrs__ri_sym_cdm_dsc : Array
        Reference DMRS with shape (2, n_sym, n_cdm, n_dmrs_sc).
    dmrs_port_nums : Array | tuple
        Per-layer DMRS port bitfield (length n_port).
    scids : tuple
        SCID selection per layer (0 or 1) as tuple with length n_port.
    dmrs_sym_idxs : tuple
        DMRS symbol indices (e.g., [2, 7, 11]) as tuple.
    n_dmrs_sc : int
        Number of DMRS subcarriers (static, compile-time constant).

    Returns
    -------
    Array
        Transmitted DMRS reference with shape (2, n_port, n_dmrs_syms, n_dmrs_sc).

    Notes
    -----
    Tensor axes have the following meaning:
    - ri: Separately stacked real/imaginary (size 2).
    - port: Port index (0 to n_port-1).
    - sym: DMRS symbol index (0 to n_dmrs_syms-1).
    - sc: Full resource grid subcarrier index.
    """

    # Compute number of DMRS symbols and ports
    n_dmrs_sym = len(dmrs_sym_idxs)
    n_port = len(dmrs_port_nums)

    # Generate frequency and time OCC patterns
    focc_pattern__ri_dsc = _gen_focc(n_dmrs_sc)[None, :]  # shape: (1, dsc)
    tocc_pattern__ri_dsym = _gen_tocc(n_dmrs_sym)[None, :]  # shape: (1, sym)

    # Initialize output for DMRS symbols (2, n_port, n_dmrs_sym, n_dmrs_sc)
    x_dmrs__ri_port_sym_sc = jnp.zeros(
        (2, n_port, n_dmrs_sym, n_dmrs_sc),
        dtype=jnp.float16,
    )

    # Build transmitted DMRS for each symbol and layer
    for sym_idx, sym in enumerate(dmrs_sym_idxs):
        for pidx, port in enumerate(dmrs_port_nums):
            # Get base DMRS sequence
            scid = scids[pidx]
            dmrs_seq__ri_sc = r_dmrs__ri_sym_cdm_dsc[:, sym, scid, :]

            # Apply frequency OCC if needed
            focc_cfg = _get_focc_cfg(dmrs_port_num=port)
            dmrs_seq__ri_sc = jax.lax.cond(
                focc_cfg != 0,
                lambda x: focc_pattern__ri_dsc * x,
                lambda x: x,
                dmrs_seq__ri_sc,
            )

            # Apply time OCC if needed
            tocc_cfg = _get_tocc_cfg(dmrs_port_num=port)
            tocc_val = tocc_pattern__ri_dsym[:, sym_idx]
            dmrs_seq__ri_sc = jax.lax.cond(
                tocc_cfg != 0,
                lambda x, v=tocc_val: v * x,
                lambda x, _v=tocc_val: x,
                dmrs_seq__ri_sc,
            )

            # Store transmitted DMRS for this symbol and layer
            x_dmrs__ri_port_sym_sc = x_dmrs__ri_port_sym_sc.at[:, pidx, sym_idx, :].set(
                dmrs_seq__ri_sc.astype(jnp.float16)
            )

    return x_dmrs__ri_port_sym_sc


def _get_focc_cfg(dmrs_port_num: Array) -> Array:
    """Get fOCC configuration from port indices.

    Parameters
    ----------
    dmrs_port_num : Array
        DMRS port number.

    Returns
    -------
    Array
        Frequency OCC configuration (0 or 1).
    """
    return ((dmrs_port_num & 0b001) >> 0).astype(jnp.int32)


def _gen_focc(n_dmrs_sc: int) -> Array:
    """Generate fOCC pattern as real values: +1, -1, +1, -1, ...

    Parameters
    ----------
    n_dmrs_sc : int
        Number of DMRS subcarriers (typically 6 * n_prb).

    Returns
    -------
    Array
        Real vector of shape (n_dmrs_sc,) with alternating +/-1 values.
    """
    indices = jnp.arange(n_dmrs_sc, dtype=jnp.int32)
    return jnp.power(-1, indices).astype(jnp.float16)


def _get_tocc_cfg(dmrs_port_num: Array) -> Array:
    """Get tOCC configuration from port indices.

    Parameters
    ----------
    dmrs_port_num : Array
        DMRS port number.

    Returns
    -------
    Array
        Time OCC configuration (0 or 1).
    """
    return ((dmrs_port_num & 0b100) >> 2).astype(jnp.int32)


def _gen_tocc(n_dmrs_sc: int) -> Array:
    """Generate tOCC pattern as real values: +1, -1, +1, -1, ...

    Parameters
    ----------
    n_dmrs_sc : int
        Number of DMRS symbols in the slot/group.

    Returns
    -------
    Array
        Real vector of shape (n_dmrs_sc,) with alternating +/-1 values.
    """
    indices = jnp.arange(n_dmrs_sc, dtype=jnp.int32)
    return jnp.power(-1, indices).astype(jnp.float16)


def apply_dmrs_to_channel(
    H__sc_sym_rxant: Array,
    x_dmrs__port_sym_dsc: Array,
    dmrs_sc_idxs: Array,
    dmrs_idx: Array | tuple,
    energy: float,
) -> Array:
    """Apply DMRS transmission to clean channel.

    Transmits DMRS symbols through the channel: y = h * x * sqrt(energy)

    Parameters
    ----------
    H__sc_sym_rxant : Array
        Clean channel with shape (n_sc, n_sym, n_rxant), complex-valued.
    x_dmrs__port_sym_dsc : Array
        Transmitted DMRS with shape (n_port, n_dmrs_sym, n_dmrs_sc),
        complex-valued (dtype complex64/complex128).
    dmrs_sc_idxs : Array
        DMRS subcarrier indices with shape (n_dmrs_sc,).
    dmrs_idx : Array | tuple
        DMRS symbol indices with shape (n_dmrs_sym,).
    energy : float
        DMRS energy scaling factor.

    Returns
    -------
    Array
        Channel with DMRS applied, shape (n_sc, n_sym, n_rxant), complex-valued.

    Notes
    -----
    This function expects complex-valued DMRS input. The output from
    gen_transmitted_dmrs_with_occ() has shape (2, n_port, n_dmrs_syms, n_dmrs_sc)
    in stacked real/imaginary format and requires conversion to complex format
    before passing to this function. Use: x_dmrs = x_dmrs[0] + 1j * x_dmrs[1]
    """
    # Start with copy of full channel (preserves non-DMRS elements)
    H_dmrs__sc_sym_rxant = H__sc_sym_rxant.copy()

    # Zero out DMRS positions before accumulation
    num_rxant = H__sc_sym_rxant.shape[2]
    nl = x_dmrs__port_sym_dsc.shape[0]  # number of ports
    H_dmrs__sc_sym_rxant = H_dmrs__sc_sym_rxant.at[
        dmrs_sc_idxs[:, None, None],
        dmrs_idx[None, :, None],  # type: ignore[call-overload]
        jnp.arange(num_rxant)[None, None, :],
    ].set(0)

    # Accumulate DMRS contributions at DMRS positions (over all ports)
    energy_sqrt = jnp.sqrt(energy)
    for sym_idx in range(len(dmrs_idx)):
        sym = dmrs_idx[sym_idx]
        for port_idx_iter in range(nl):
            for rxant in range(num_rxant):
                # Multiply channel by transmitted DMRS: y = h * x * sqrt(energy)
                contrib = (
                    H__sc_sym_rxant[dmrs_sc_idxs, sym, rxant]
                    * x_dmrs__port_sym_dsc[port_idx_iter, sym_idx, :]
                    * energy_sqrt
                )
                H_dmrs__sc_sym_rxant = H_dmrs__sc_sym_rxant.at[dmrs_sc_idxs, sym, rxant].add(
                    contrib
                )

    return H_dmrs__sc_sym_rxant
