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

"""Extract raw received signal at DMRS positions."""

from jax import Array, numpy as jnp


def _get_freq_grid_cfg(port_nums: tuple) -> tuple:
    """Get grid configuration from port indices.

    3GPP TR 38.211, 6.4.1.1.3 Precoding and mapping to physical resources.

    The grid configuration is determined by the port index:
    - Port 1000 -> 0
    - Port 1001 -> 0
    - Port 1010 -> 1
    - Port 1011 -> 1

    Parameters
    ----------
    port_nums : tuple
        Port numbers as a tuple of integers.

    Returns
    -------
    tuple
        Grid configuration as a tuple of integers.
    """
    return tuple((p & 0b010) >> 1 for p in port_nums)


def extract_raw_dmrs_type1(
    xtf__ri_sym_rxant_sc: Array,
    dmrs_sym_idxs: Array,
    n_prb: int,
    start_prb: int,
    dmrs_port_nums: tuple,
) -> tuple[Array, int, tuple]:
    """Extract raw received signal at DMRS positions.

    Extract DMRS resource elements from the appropriate frequency grids
    (even and/or odd), as determined from the DMRS port numbers.

    Parameters
    ----------
    xtf__ri_sym_rxant_sc : Array
        Raw received channel with shape (2, n_t, n_ant, n_f).
    dmrs_sym_idxs : Array
        DMRS symbol indices (e.g., [2, 7, 11]).
    n_prb : int
        Number of PRBs in the allocation.
    start_prb : int
        0-based starting PRB index.
    dmrs_port_nums : tuple
        Per-layer DMRS port numbers as static tuple (length n_port).

    Returns
    -------
    y_dmrs__ri_dsym_rxant_dsc : Array
        Raw DMRS with shape (2, n_dmrs_syms, n_ant, n_dmrs_sc).
    n_dmrs_sc_per_prb : int
        Number of DMRS subcarriers per PRB (6 or 12).
    dmrs_port_grid_cfgs : tuple
        Grid configuration for each port.

    Notes
    -----
    Tensor axes have the following meaning:
    - ri: Separately stacked real/imaginary (size 2).
    - sym: DMRS symbol index (0 to n_dmrs_syms-1).
    - rxant: Rx antenna index (0 to n_ant-1).
    - sc: Full resource grid subcarrier index.
    - n_dmrs_sc: Number of extracted DMRS subcarriers (6*n_prb for single
      CDM group, 12*n_prb if both CDM groups).
    - CDM group 0 (ports 1000/1001) uses even subcarriers (offset 0, 2, 4, ...).
    - CDM group 1 (ports 1010/1011) uses odd subcarriers (offset 1, 3, 5, ...).
    - When both groups present, all subcarriers are extracted (CDM groups
      interleaved).
    """
    n_dmrs_sc = 6 * n_prb
    dmrs_base = 12 * start_prb

    # Determine unique CDM groups from DMRS port numbers:
    # 0 (even subcarriers) and
    # 1 (odd subcarriers)
    dmrs_port_grid_cfgs = _get_freq_grid_cfg(dmrs_port_nums)

    if all(g == dmrs_port_grid_cfgs[0] for g in dmrs_port_grid_cfgs):
        # All ports use the same CDM group, so we have a single CDM group (extract only
        # those even or odd subcarriers).
        cdm_group_id = dmrs_port_grid_cfgs[0]
        dmrs_sc_idxs = dmrs_base + 2 * jnp.arange(n_dmrs_sc, dtype=jnp.int32) + cdm_group_id
        y_dmrs__ri_dsym_rxant_dsc = xtf__ri_sym_rxant_sc[:, dmrs_sym_idxs, :, :][
            :, :, :, dmrs_sc_idxs
        ]
        n_dmrs_sc_per_prb = 6
    else:
        # Both CDM groups present, so we extract all DMRS subcarriers
        dmrs_sc_idxs = dmrs_base + jnp.arange(2 * n_dmrs_sc, dtype=jnp.int32)
        y_dmrs__ri_dsym_rxant_dsc = xtf__ri_sym_rxant_sc[:, dmrs_sym_idxs, :, :][
            :, :, :, dmrs_sc_idxs
        ]
        n_dmrs_sc_per_prb = 12

    return y_dmrs__ri_dsym_rxant_dsc, n_dmrs_sc_per_prb, dmrs_port_grid_cfgs


__all__ = [
    "extract_raw_dmrs_type1",
]
