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

"""LDPC decode (NumPy translation of LDPC_decode.m and subroutines)."""

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from ran.phy.numpy.pusch._ldpc_tanner_tables import LDPC_TANNER_TABLES
from ran.types import FloatArrayNP, FloatNP, IntArrayNP, IntNP


def ldpc_decode(
    derate_cbs: FloatArrayNP,
    nv_parity: int,
    zc: int,
    c: int,
    bgn: int,
    i_ls: int,
    max_num_itr_cbs: int,
) -> tuple[FloatArrayNP, IntArrayNP]:
    """Run layered min-sum LDPC decoding per 3GPP TS 38.212.

    Args:
        derate_cbs: LLRs after de-rate match shaped (N, C), where C is the
            number of codeblocks and N depends on `zc` and the graph size.
        nv_parity: int, Number of parity variable nodes inferred by de-rate matching.
        zc: int, Lifting size (Zc) of the LDPC base graph.
        c: int, Number of codeblocks in the transport block.
        bgn: int, Base graph number (1 or 2).
        i_ls: Layer-shift index to select the parity-check permutation table.
        max_num_itr_cbs: int, Maximum iterations per codeblock

    Returns
    -------
    tb_out: FloatArrayNP, Decoded systematic bits as float64 in shape
        (nV_sym*zc, C), matched to MATLAB's column-major layout.
    num_itr: IntArrayNP, Iterations used per codeblock, shape (C,).
    """
    # Build Tanner parameters
    tanner_par = _load_tanner(bgn, i_ls, zc)

    # Add puncturing and reshape into (zc, n_v, c)
    n_v = cast("int", tanner_par["nV"])  # total variable nodes
    llr_aug = np.vstack(
        [
            np.zeros((2 * zc, c), dtype=FloatNP),
            derate_cbs[: zc * (n_v - 2), :],
        ]
    )
    llr_reshaped = llr_aug.reshape(zc, n_v, c, order="F")

    # Decode each codeblock
    tb_cbs_est = np.zeros_like(llr_reshaped)
    num_itr: IntArrayNP = np.zeros((c,), dtype=IntNP)

    # Normalization alpha
    alpha = _set_ldpc_normalization(nv_parity, bgn)

    for c_idx in range(c):
        tb_cbs_est_c, itr_c = _msa_layering(
            llr_reshaped[:, :, c_idx], zc, alpha, max_num_itr_cbs, tanner_par
        )
        tb_cbs_est[:, :, c_idx] = tb_cbs_est_c
        num_itr[c_idx] = itr_c

    # Match MATLAB output shape: keep only systematic nodes and reshape
    n_v_sym = cast("int", tanner_par["nV_sym"])  # number of systematic nodes
    tb_sys = tb_cbs_est[:, :n_v_sym, :]
    # reshape to (n_v_sym*zc, c) in column-major order
    tb_out = tb_sys.reshape(zc * n_v_sym, c, order="F")

    return tb_out, num_itr


def _load_tanner(bgn: int, i_ls: int, zc: int) -> dict[str, object]:
    """Load Tanner graph tables for the requested base graph and layer shift.

    Args:
        bgn: Base graph number (1 or 2).
        i_ls: Layer-shift index selecting the row permutations.
        zc: Lifting size used to modulo-reduce neighbor shifts.

    Returns
    -------
    dict
        Dictionary including:
        - 'nC': number of check nodes (rows) in base graph
        - 'nV': total variable nodes (columns) after lifting
        - 'nV_sym': number of systematic variable nodes
        - 'numNeighbors': array of length nC with degrees per check node
        - 'NeighborIdx': list of length nC, each int64 array of neighbor col
          indices (1-based)
        - 'NeighborShift': list of length nC, each int64 array of cyclic shifts
    """
    table = LDPC_TANNER_TABLES
    bgn_prefix = f"BG{bgn}_"
    n_c, n_v, n_v_sym = (46, 68, 22) if bgn == 1 else (42, 52, 10)

    neighbor_indices = cast("Sequence[IntArrayNP]", table[f"{bgn_prefix}NeighborIndices"])
    num_neighbors_raw = cast("Sequence[int]", table[f"{bgn_prefix}numNeighbors"])
    num_neighbors_arr = np.asarray(num_neighbors_raw, dtype=IntNP).ravel()
    ls_prefix = f"{bgn_prefix}NeighborPermutations_LS"

    neighbor_shift = cast("Sequence[IntArrayNP]", table[f"{ls_prefix}{i_ls}"])

    # Mod shifts by zc
    neighbor_shift_mod = []
    neighbor_idx_list = []
    for c_idx in range(n_c):
        n_neighbors = int(num_neighbors_arr[c_idx])
        idx_row = neighbor_indices[c_idx].ravel()[:n_neighbors]
        sh_row = neighbor_shift[c_idx].ravel()[:n_neighbors]
        sh_row = np.mod(sh_row, zc)
        neighbor_idx_list.append(idx_row.astype(IntNP))
        neighbor_shift_mod.append(sh_row.astype(IntNP))

    return {
        "nC": n_c,
        "nV": n_v,
        "nV_sym": n_v_sym,
        "numNeighbors": num_neighbors_arr,
        "NeighborIdx": neighbor_idx_list,
        "NeighborShift": neighbor_shift_mod,
    }


def _set_ldpc_normalization(nv_parity: int, bgn: int) -> float:
    """Return min-sum normalization factor alpha.

    Uses a truncated table derived from MATLAB's LDPC_decode.m. The values are
    indexed by the number of parity nodes and depend on the base graph.

    Args:
        nv_parity: Number of parity variable nodes.
        bgn: Base graph number (1 or 2).

    Returns
    -------
    float
        Normalization factor alpha in [0, 1].
    """
    if bgn == 1:
        table = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.79,
            0.77,
            0.75,
            0.73,
            0.75,
            0.70,
            0.67,
            0.68,
            0.67,
            0.67,
            0.68,
            0.66,
            0.65,
            0.66,
            0.64,
            0.65,
            0.65,
            0.65,
            0.65,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.67,
            0.66,
            0.65,
            0.64,
            0.63,
            0.63,
            0.63,
            0.63,
            0.63,
            0.62,
            0.63,
            0.63,
            0.64,
            0.63,
            0.63,
            0.63,
            0.62,
            0.63,
        ]
    else:
        table = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.86,
            0.84,
            0.80,
            0.77,
            0.75,
            0.75,
            0.74,
            0.74,
            0.74,
            0.73,
            0.73,
            0.73,
            0.73,
            0.72,
            0.70,
            0.71,
            0.71,
            0.71,
            0.71,
            0.70,
            0.69,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.68,
            0.67,
            0.67,
            0.68,
            0.69,
            0.69,
            0.69,
            0.69,
            0.69,
            0.69,
        ]
    idx = int(np.clip(nv_parity, 0, len(table) - 1))
    return table[idx]


def _msa_layering(
    llr: FloatArrayNP,
    zc: int,
    alpha: float,
    max_itr: int,
    tanner_par: Mapping[str, object],
) -> tuple[FloatArrayNP, int]:
    """Layered min-sum iterations.

    Args:
        llr: Input LLRs shaped (zc, nV).
        zc: Lifting size.
        alpha: Normalization factor.
        max_itr: Maximum number of iterations.
        tanner_par: Graph description from _load_tanner().

    Returns
    -------
    tb_cbs_est: Hard decisions as float64 (0/1) shaped (zc, nV).
    num_itr: Number of iterations executed.
    """
    # Cache Tanner arrays locally (no per-iteration np.asarray)
    n_c = cast("int", tanner_par["nC"])  # number of check nodes
    num_neighbors = np.asarray(tanner_par["numNeighbors"], dtype=IntNP)
    neighbor_idx = cast("Sequence[IntArrayNP]", tanner_par["NeighborIdx"])
    neighbor_shift = cast("Sequence[IntArrayNP]", tanner_par["NeighborShift"])

    app = np.array(llr, dtype=FloatNP, copy=True)
    max_deg = int(np.max(num_neighbors))
    c2v: FloatArrayNP = np.zeros((zc, n_c, max_deg), dtype=FloatNP)

    # Precompute roll indices and column grid once per iteration context
    rows_base: IntArrayNP = np.arange(zc, dtype=IntNP)[:, None]
    cols_grid = np.broadcast_to(np.arange(max_deg, dtype=IntNP)[None, :], (zc, max_deg))
    roll_idx_v2c: list[IntArrayNP] = []
    roll_idx_c2v: list[IntArrayNP] = []
    for row in range(n_c):
        deg_row = int(num_neighbors[row])
        sh = neighbor_shift[row].ravel()[:deg_row].astype(IntNP)
        # v2c uses np.roll(x, -shift)
        roll_idx_v2c.append((rows_base + sh[None, :]) % zc)
        # c2v uses np.roll(x, +shift)
        roll_idx_c2v.append((rows_base - sh[None, :]) % zc)

    for _ in range(max_itr):
        for c_idx in range(n_c):
            v2c = _compute_v2c(
                c_idx,
                app,
                c2v,
                zc,
                neighbor_idx,
                num_neighbors,
                roll_idx_v2c,
                cols_grid,
            )
            cc2v = _compute_cc2v(c_idx, v2c, zc, num_neighbors)
            _update_c2v(
                c_idx,
                app,
                c2v,
                v2c,
                cc2v,
                alpha,
                neighbor_idx,
                num_neighbors,
                roll_idx_c2v,
                cols_grid,
            )

    # Hard decision
    tb_cbs_est: FloatArrayNP = (app <= 0).astype(FloatNP)
    return tb_cbs_est, max_itr


def _compute_v2c(
    c_idx: int,
    app: FloatArrayNP,
    c2v: FloatArrayNP,
    zc: int,
    neighbor_idx: Sequence[IntArrayNP],
    num_neighbors: IntArrayNP,
    roll_idx_v2c: Sequence[IntArrayNP],
    cols_grid: IntArrayNP,
) -> FloatArrayNP:
    """Compute variable-to-check messages for one check node row.

    Args:
        c_idx: Check node index (row index).
        app: A posteriori LLRs shaped (zc, nV).
        c2v: Check-to-variable messages shaped (zc, nC, max_deg).
        zc: Lifting size.
        neighbor_idx: Variable node indices for each check node.
        num_neighbors: Number of neighbors per check node.

    Returns
    -------
    v2c: Variable-to-check messages shaped (zc, deg, 2): absolute LLR and sign (+1/-1).
    """
    n_neighbors = int(num_neighbors[c_idx])
    v_idx_row = neighbor_idx[c_idx][:n_neighbors]

    # Gather all neighbor columns and subtract existing messages
    col_idx = (v_idx_row.astype(IntNP) - 1).ravel()

    app_sub = app[:, col_idx]  # (zc, deg)
    c2v_sub = c2v[:, c_idx, :n_neighbors]  # (zc, deg)
    diff = app_sub - c2v_sub  # (zc, deg)

    # Column-wise cyclic shift using precomputed indices
    roll_idx = roll_idx_v2c[c_idx][:, :n_neighbors]
    cols = cols_grid[:, :n_neighbors]
    vec = diff[roll_idx, cols]

    v2c_abs = np.abs(vec)
    v2c_sgn = 1.0 - 2.0 * (vec < 0)
    v2c_out: FloatArrayNP = np.empty((zc, n_neighbors, 2), dtype=FloatNP)
    v2c_out[:, :, 0] = v2c_abs
    v2c_out[:, :, 1] = v2c_sgn
    return v2c_out


def _compute_cc2v(
    c_idx: int,
    v2c: FloatArrayNP,
    zc: int,
    num_neighbors: IntArrayNP,
) -> FloatArrayNP:
    """Compute per-row min1/min2 and sign product from v2c messages.

    Args:
        c_idx: Check node index (row index).
        v2c: Variable-to-check messages shaped (zc, deg, 2): absolute LLR and sign.
        zc: Lifting size.
        num_neighbors: Number of neighbors per check node.

    Returns
    -------
    cc2v: Array shaped (zc, 4): [min1, min2, sign_product, argmin_index].
    """
    n_neighbors = int(num_neighbors[c_idx])
    # v2c1: (zc, deg), v2c2: (zc, deg)
    v2c1 = v2c[:, :n_neighbors, 0]
    v2c2 = v2c[:, :n_neighbors, 1]

    # min1/min2 via partition; works row-wise
    # partition returns a view where the smallest is at index 0, second smallest at 1
    part = np.partition(v2c1, 1, axis=1)
    min1 = part[:, 0]
    min2 = part[:, 1]
    min1_idx = np.argmin(v2c1, axis=1) + 1  # 1-based to match current code

    sgn_prb = np.prod(v2c2, axis=1)

    cc2v: FloatArrayNP = np.empty((zc, 4), dtype=FloatNP)
    cc2v[:, 0] = min1
    cc2v[:, 1] = min2
    cc2v[:, 2] = sgn_prb
    cc2v[:, 3] = min1_idx
    return cc2v


def _update_c2v(  # noqa: PLR0913
    c_idx: int,
    app: FloatArrayNP,
    c2v: FloatArrayNP,
    v2c: FloatArrayNP,
    cc2v: FloatArrayNP,
    alpha: float,
    neighbor_idx: Sequence[IntArrayNP],
    num_neighbors: IntArrayNP,
    roll_idx_c2v: Sequence[IntArrayNP],
    cols_grid: IntArrayNP,
) -> None:
    """Update check-to-variable messages and a posteriori LLRs for one row.

    Args:
        c_idx: Check node index (row index).
        app: A posteriori LLRs shaped (zc, n_v).
        c2v: Check-to-variable messages shaped (zc, n_c, max_degree).
        v2c: Variable-to-check messages shaped (zc, deg, 2): absolute LLR and sign.
        cc2v: Check computation results shaped (zc, 4):
            [min1, min2, sign_product, argmin_index].
        alpha: Normalization factor for min-sum algorithm.
        neighbor_idx: List of neighbor variable node indices per check node (1-based).
        num_neighbors: Number of neighbors per check node.
    """
    deg = int(num_neighbors[c_idx])
    v_idx_row = neighbor_idx[c_idx][:deg]

    # Row-wise scalars
    min1 = cc2v[:, 0]
    min2 = cc2v[:, 1]
    sgn_pr = cc2v[:, 2]
    argmin = cc2v[:, 3].astype(IntNP)  # 1-based

    # Old messages slice for all neighbors
    old_msg = c2v[:, c_idx, :deg].copy()

    # Choose min1/min2 per neighbor via broadcasting
    i_idx: IntArrayNP = np.arange(1, deg + 1, dtype=IntNP)[None, :]
    use_min1 = argmin[:, None] != i_idx
    c2v_abs_mat = np.where(use_min1, min1[:, None], min2[:, None])
    c2v_sgn_mat = sgn_pr[:, None] * v2c[:, :deg, 1]

    new_msg_var = alpha * (c2v_abs_mat * c2v_sgn_mat)

    # Apply column-wise cyclic shift using precomputed indices
    roll_idx = roll_idx_c2v[c_idx][:, :deg]
    cols = cols_grid[:, :deg]
    new_msg_rolled = new_msg_var[roll_idx, cols]

    # Compute delta before overwriting and scatter-add to app
    delta = (new_msg_rolled - old_msg).reshape(-1)
    zc_rows = app.shape[0]
    row_idx: IntArrayNP = np.repeat(np.arange(zc_rows, dtype=IntNP), deg)
    col_idx = np.tile((v_idx_row.astype(IntNP) - 1).ravel(), zc_rows)
    np.add.at(app, (row_idx, col_idx), delta)

    # Finally write updated messages
    c2v[:, c_idx, :deg] = new_msg_rolled


__all__ = ["ldpc_decode"]
