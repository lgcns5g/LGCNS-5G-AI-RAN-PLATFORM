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
Derate match (NumPy translation of derate_match.m).

Implements TS 38.212 Section 5.4.2 de-rate matching steps using the
MATLAB logic provided in derate_match.m including LBMR, RV offsets,
bit deinterleaving, bit selection undo, filler insertion, and clamping.
"""

import numpy as np

from ran.constants import LLR_CLAMP_ABS
from ran.types import FloatArrayNP, FloatNP, IntArrayNP, IntNP


def derate_match(  # noqa: PLR0913, PLR0915
    llr_descr: FloatArrayNP,
    bgn: int,
    c: int,
    qam_bits: int,
    k: int,
    f: int,
    k_prime: int,
    zc: int,
    nl: int,
    rv_idx: int,
    nref: int,
    g: int,
) -> tuple[
    FloatArrayNP,
    int,
    IntArrayNP,
    IntArrayNP,
]:
    """
    De-rate match LLRs into codeblocks.

    Implements the de-rate matching steps defined in 3GPP TS 38.212 §5.4.2
    using LBMR (limited buffer mother rate), RV offsets, bit deinterleaving,
    bit-selection undo, filler insertion, and clamping.

    Args:
        llr_descr:
            (N?, C) float64. Descrambled LLRs arranged per codeblock (CB).
            The first dimension must contain at least max(E_r) per CB; only
            the first E_r entries in each column are consumed (where E_r is
            computed internally from G, C, Q_m, and N_layers).
        bgn:
            int. LDPC base graph number (1 or 2).
        c:
            int. Number of codeblocks (CBs) after transport-block segmentation.
        qam_bits:
            int. Modulation order Q_m in bits per symbol
            (1=BPSK, 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM).
        k:
            int. Codeblock length AFTER filler insertion, i.e., the expanded
            CB length used by the mother LDPC code before parity section
            (K). By definition K = K' + f. Indices in this function are
            expressed relative to K (e.g., the filler region is
            [K'-2Z_c, K-2Z_c)).
        f:
            int. Number of filler bits in the CB. These positions are not
            transmitted and are reinserted with clamped LLRs during de-rate
            matching. By definition f = K - K'.
        k_prime:
            int. Codeblock length BEFORE filler insertion (K'). This includes
            the information bits for the CB plus the CB-CRC, but excludes
            filler bits.
        zc:
            int. LDPC lifting size Z_c (submatrix dimension used to expand the
            base graph). It sets the mother code lengths:
            N = Z_c * (66 for BG1, 50 for BG2).
        nl:
            int. Number of layers (N_layers) used for transmission.
        rv_idx:
            int. Redundancy version index (0..3) used to compute the circular
            buffer read-out offset k0 for HARQ de-rate matching.
        nref:
            int. Optional cap on the per-CB circular buffer length (N_ref).
            If > 0, n_cb = min(N, N_ref); otherwise n_cb = N. This mirrors
            the spec's LBMR behavior when resources limit the read-out span.
        g:
            int. Total number of coded bits G allocated to this codeword
            across all layers for the scheduled PDSCH/PUSCH allocation. G is
            used to compute the per-CB bit budgets E_r.

    Returns
    -------
        derate_cbs:
            (N, C) float64. LLRs per CB after de-rate matching, including
            filler re-insertion and clamping, where N = Z_c * (66 or 50).
        nv_parity:
            int. Number of parity variable nodes inferred for the read-out,
            clamped to [4, 46] for BG1 or [4, 42] for BG2 in this
            implementation.
        derate_cbs_indices:
            (N, C) int64. One-based indices (with final offset applied) of
            the mother-code positions that each consumed LLR contributed to,
            per CB.
        derate_cbs_sizes:
            (C,) int64. The number of consumed bits E_r per CB column.

    Notes
    -----
    • Mother code length: N = Z_c * (66 for BG1, 50 for BG2).
    • K (k) and K' (k_prime) differ by the filler count f = K - K'.
      Filler positions are not transmitted and are reconstructed by
      assigning clamped LLRs in [K'-2Z_c, K-2Z_c).
    • G determines the per-CB E_r through the standard split across C CBs,
      modulation order Q_m, and N_layers.
    """
    # Compute per-codeblock bit budgets E_r (vector "e"): how many coded bits
    # are available/consumed for each CB r after accounting for modulation and
    # layers. TS 38.212 splits total coded bits G across C CBs such that most
    # CBs get floor(G/C') and the remainder get ceil(G/C'), where C' = C * Q_m * N_layers
    # expressed in bits per symbol across layers.
    nl_qam = qam_bits * nl  # Q_m * N_layers (bits per symbol across all layers)
    # Denominator C' = C * (Q_m * N_layers)
    denom = c * nl_qam
    # floor(G / C') scaled back to bits (multiple of Q_m * N_layers)
    e_floor = nl_qam * (g // denom)
    # ceil(G / C') using integer math trick: ceil(a/b) = (a + b - 1) // b
    e_ceil = nl_qam * ((g + denom - 1) // denom)
    # Determine how many CBs receive e_ceil instead of e_floor. The spec assigns
    # the larger share to the last (mod(G/(Q_m*N_layers), C)) CBs. We compute the
    # remainder and a threshold so that r <= threshold -> e_floor else e_ceil.
    rem = (g // nl_qam) % c  # remainder CB count when distributing symbols
    threshold = c - rem - 1  # index cutoff between floor and ceil allocations
    r_idx: IntArrayNP = np.arange(c, dtype=IntNP)  # CB index r = 0..C-1
    # e[r] holds E_r, i.e., number of coded bits consumed for CB r
    e = np.where(r_idx <= threshold, e_floor, e_ceil).astype(IntNP, copy=False)

    # Limited Buffer Mother Rate (LBMR): the circular buffer length used during
    # read-out. Mother-code length N depends on base graph (BG1: 66*Z_c, BG2: 50*Z_c).
    # Optionally cap by N_ref (if > 0) per spec when resource-limited.
    n = zc * (66 if bgn == 1 else 50)  # Mother-code length N
    n_cb = min(n, nref) if nref > 0 else n  # Effective circular buffer length

    # Redundancy Version (RV) offset k0: start position in the circular buffer.
    # Per 38.212, k0 = floor((v * N_cb) / (Z_c * D)) * Z_c with v from a small
    # table that depends on BG and RV, and D in {66, 50} (BG1/BG2). We pre-store
    # the numerators (v values) in a tuple and compute k0 branchlessly.
    denom = 66 if bgn == 1 else 50  # D (BG-dependent constant)
    numerators = (0, 17, 33, 56) if bgn == 1 else (0, 13, 25, 43)  # v per RV
    if bgn not in (1, 2):
        msg = "BGN is not supported"
        raise ValueError(msg)
    if rv_idx < 0 or rv_idx >= len(numerators):
        msg = "rv is not supported"
        raise ValueError(msg)
    num = numerators[rv_idx]
    # k0 is a multiple of Z_c; for v=0 the spec defines k0 = 0.
    k0 = 0 if num == 0 else ((num * n_cb) // (denom * zc)) * zc

    # Parity section sizing: infer how many parity variable-node groups were read.
    # max_llr_per_cb: theoretical max read length per CB (bounded by N)
    # max_parity_nodes: BG-dependent cap on parity groups
    # nllr_per_cb: total consumed LLRs per CB considering k0 offset (circular)
    # nsym_llr_per_cb: systematic portion size = K' - 2*Z_c (excludes filler)
    # npar_llr_per_cb: remaining LLRs attributed to parity section
    max_llr_per_cb = zc * (66 if bgn == 1 else 50)
    max_parity_nodes = 46 if bgn == 1 else 42
    nllr_per_cb = min(g // c + k0, max_llr_per_cb)
    nsym_llr_per_cb = k - f - 2 * zc
    npar_llr_per_cb = nllr_per_cb - nsym_llr_per_cb

    # Compute number of parity variable nodes (nv_parity). Each node accounts for
    # a Z_c-sized group in the parity portion. Clamp to [4, max_parity_nodes] to
    # mirror MATLAB and hardware behavior.
    nv_parity = max(4, min((npar_llr_per_cb + zc - 1) // zc, max_parity_nodes))

    # Outputs
    # derate_cbs: accumulated LLRs in the mother-code domain for each CB (shape N x C)
    derate_cbs: FloatArrayNP = np.zeros((n, c), dtype=FloatNP)
    # derate_cbs_indices: for traceability, the 1-based mother-code indices that
    # each consumed LLR mapped to (per CB), after final offset.
    derate_cbs_indices: IntArrayNP = np.zeros((n, c), dtype=IntNP)
    # derate_cbs_sizes: the number of consumed bits E_r for each CB (length C)
    derate_cbs_sizes: IntArrayNP = np.zeros((c,), dtype=IntNP)

    # Process each codeblock
    current_bit = 0  # cursor into flattened LLR stream across CBs
    llr_descr_vec = llr_descr.ravel(order="F")  # column-major flatten (per-CB contiguous)

    # Build acceptance mask in the circular buffer domain (1-based indexing as in spec):
    # accept positions outside the filler region [K'-2Z_c, K-2Z_c). This inverts the
    # bit-selection performed during rate-matching.
    idx1b_base: IntArrayNP = np.arange(1, n_cb + 1, dtype=IntNP)
    accept_mask_base = (idx1b_base <= (k_prime - 2 * zc)) | (idx1b_base > (k - 2 * zc))

    # Apply RV offset: rotate mask by -k0 so that index 1 corresponds to the read-out start
    accept_mask_cycle = np.roll(accept_mask_base, -k0)
    # j_accept_cycle: 0-based indices within one n_cb-long cycle that are accepted
    j_accept_cycle = np.nonzero(accept_mask_cycle)[0].astype(IntNP)
    num_accept_per_cycle = j_accept_cycle.size  # accepted positions per cycle
    if num_accept_per_cycle == 0:
        msg = "No acceptable positions found in cycle; check parameters"
        raise ValueError(msg)

    for ci in range(c):
        e_c = e[ci]  # E_r for CB r=ci (number of bits to consume for this CB)
        # Slice this CB's LLRs: [current_bit, current_bit + E_r)
        llr_c = llr_descr_vec[current_bit : current_bit + e_c]
        # Undo bit interleaving across Q_m: reshape to (E_r/Q_m, Q_m) then transpose
        # to restore bit order per symbol, finally flatten back.
        llr_c_mat = np.reshape(llr_c, (qam_bits, e_c // qam_bits), order="F").T
        llr_c = llr_c_mat.ravel(order="F")

        # Generate the accepted index sequence for exactly E_r positions by
        # repeating a full acceptance cycle and truncating to length E_r.
        reps = (e_c + num_accept_per_cycle - 1) // num_accept_per_cycle
        j_seq = np.tile(j_accept_cycle, reps)[:e_c]
        idx_seq = (k0 + j_seq) % n_cb  # 0-based indices in circular buffer domain
        idx1b_seq = idx_seq + 1  # 1-based version for output traceability

        # Accumulate LLRs into mother-code positions. Multiple reads can map to the
        # same index due to circularity; use bincount to sum contributions.
        add_sums = np.bincount(idx_seq, weights=llr_c, minlength=n_cb)
        derate_cbs[:, ci] += add_sums
        # Record the mapping indices and E_r for this CB
        derate_cbs_indices[:e_c, ci] = idx1b_seq
        derate_cbs_sizes[ci] = e_c
        current_bit += e_c

    # Reinsert filler positions in mother-code domain: [K'-2Z_c, K-2Z_c). These were
    # not transmitted; set them to saturated LLRs to reflect certainty placeholders.
    start_fill = k_prime - 2 * zc
    end_fill = k - 2 * zc
    if end_fill > start_fill:
        derate_cbs[start_fill:end_fill, :] = LLR_CLAMP_ABS

    # Saturate LLRs to match hardware/Matlab clamping behavior
    np.clip(derate_cbs, -LLR_CLAMP_ABS, LLR_CLAMP_ABS, out=derate_cbs)

    # Convert indices to the C/Matlab offset convention (add 2*Z_c - 1)
    derate_cbs_indices = derate_cbs_indices + (2 * zc - 1)

    return derate_cbs, nv_parity, derate_cbs_indices, derate_cbs_sizes


__all__ = ["derate_match"]
