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

"""Reference outer rx for PUSCH Rx pipeline.

This module provides outer rx blocks including descrambling,
de-rate matching, LDPC decoding, codeblock concatenation, and CRC decoding.
These are used for end-to-end verification of the PUSCH Rx pipeline.
"""

from dataclasses import dataclass, fields

import numpy as np

# 5G PHY reference model outer rx blocks
from ran.phy.numpy.pusch.descrambler import descramble_bits
from ran.phy.numpy.pusch.derate_matcher import derate_match
from ran.phy.numpy.pusch.ldpc_decoder import ldpc_decode
from ran.phy.numpy.pusch.codeblock_concatenation import codeblock_concatenation
from ran.phy.numpy.pusch.crc_decoder import crc_decode


@dataclass
class OuterRxParams:
    """Parameters needed for outer rx (from test vector)."""

    # Descrambling
    n_id: int
    n_rnti: int

    # De-rate matching
    bgn: int
    c: int
    k: int
    f: int
    k_prime: int
    zc: int
    nl: int
    rv_idx: int
    nref: int
    g: int

    # LDPC decoding
    i_ls: int
    max_num_itr_cbs: int

    # CRC decoding
    crc_name: str

    def print_debug(self) -> None:
        """Print debug information about outer rx parameters."""
        print(f"Outer rx parameters ({len(fields(self))} total):")
        for i, field in enumerate(fields(self)):
            value = getattr(self, field.name)
            print(f"  [{i}] {field.name}: {value}")


@dataclass
class OuterRxOutputs:
    """Outer receiver outputs."""

    tb_llr_descr: np.ndarray  # Descrambled LLRs
    derate_cbs: np.ndarray  # De-rate matched codeblocks
    nv_parity: int  # Number of parity bits
    tb_cbs_est: np.ndarray  # Estimated codeblocks
    num_itr: np.ndarray  # LDPC iterations per codeblock
    tb_crc_est: np.ndarray  # Transport block with CRC
    cb_err: np.ndarray  # Codeblock errors
    tb_est: np.ndarray  # Decoded transport block
    tb_err: int  # Transport block error (0 = success, 1 = failure)

    def print_debug(self) -> None:
        """Print debug information about outer rx outputs."""
        print(f"\nOuter rx outputs ({len(fields(self))} total):")
        for i, field in enumerate(fields(self)):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                shape_str = f"shape={value.shape}, dtype={value.dtype}"
            else:
                shape_str = f"value={value}"
            print(f"  [{i}] {field.name}: {shape_str}")


def pusch_outer_rx(
    llr__time_allocfreq_layer_qambit: np.ndarray,
    outer_rx_params: OuterRxParams,
    qam_bits: int,
) -> OuterRxOutputs:
    """Apply outer rx blocks to LLRs from optimized PUSCH Rx.

    This function takes the LLR output from the JAX optimized pusch inner rx and
    applies the reference outer rx blocks to decode the transport block.

    Args:
        llr__time_allocfreq_layer_qambit: LLRs in column-major format
            (n_datasym, n_allocsc, n_layer, qam_bits)
        outer_rx_params: Backend processing parameters from test vector
        qam_bits: QAM modulation order (bits per symbol)

    Returns:
        OuterRxOutputs: Decoded transport block and intermediate outputs
    """

    # ---------------------------------------------------------------
    # Block 9: Descramble bits
    # ---------------------------------------------------------------
    # Convert LLRs from column-major to row-major format and flatten
    # Input:  llr__time_allocfreq_layer_qambit (n_datasym, n_allocsc, n_layer, qam_bits)
    # Output: llr__qambit_layer_allocfreq_time (qam_bits, n_layer, n_allocfreq, n_time)
    llr__qambit_layer_allocfreq_time = np.transpose(llr__time_allocfreq_layer_qambit, (3, 2, 1, 0))

    # Flatten to 1D sequence (transpose first to match reference order)
    llr_seq = llr__qambit_layer_allocfreq_time.transpose(3, 2, 1, 0).ravel()

    # Descramble the LLRs
    tb_llr_descr = descramble_bits(
        llrseq=llr_seq,
        n_id=outer_rx_params.n_id,
        n_rnti=outer_rx_params.n_rnti,
    )

    # ---------------------------------------------------------------
    # Block 10: Derate match
    # ---------------------------------------------------------------
    derate_cbs, nv_parity, _, _ = derate_match(
        llr_descr=tb_llr_descr,
        bgn=outer_rx_params.bgn,
        c=outer_rx_params.c,
        qam_bits=qam_bits,
        k=outer_rx_params.k,
        f=outer_rx_params.f,
        k_prime=outer_rx_params.k_prime,
        zc=outer_rx_params.zc,
        nl=outer_rx_params.nl,
        rv_idx=outer_rx_params.rv_idx,
        nref=outer_rx_params.nref,
        g=outer_rx_params.g,
    )

    # ---------------------------------------------------------------
    # Block 11: LDPC decode
    # ---------------------------------------------------------------
    tb_cbs_est, num_itr = ldpc_decode(
        derate_cbs=derate_cbs,
        nv_parity=nv_parity,
        zc=outer_rx_params.zc,
        c=outer_rx_params.c,
        bgn=outer_rx_params.bgn,
        i_ls=outer_rx_params.i_ls,
        max_num_itr_cbs=outer_rx_params.max_num_itr_cbs,
    )

    # ---------------------------------------------------------------
    # Block 12: Codeblock concatenation
    # ---------------------------------------------------------------
    tb_crc_est, cb_err = codeblock_concatenation(
        tb_cbs_est=tb_cbs_est,
        c=outer_rx_params.c,
        k_prime=outer_rx_params.k_prime,
    )

    # ---------------------------------------------------------------
    # Block 13: CRC decode
    # ---------------------------------------------------------------
    tb_est, tb_err = crc_decode(
        tb_crc_est=tb_crc_est,
        crc_name=outer_rx_params.crc_name,
    )

    return OuterRxOutputs(
        tb_llr_descr=tb_llr_descr,
        derate_cbs=derate_cbs,
        nv_parity=nv_parity,
        tb_cbs_est=tb_cbs_est,
        num_itr=num_itr,
        tb_crc_est=tb_crc_est,
        cb_err=cb_err,
        tb_est=tb_est,
        tb_err=tb_err,
    )
