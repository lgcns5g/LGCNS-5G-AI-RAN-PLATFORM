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

"""End-to-end PUSCH reference receiver pipeline.

Provides a single function `pusch_rx` that wires the main reference blocks in
order, mapping outputs to subsequent inputs. This mirrors the flow used in the
TV-based script `tests/test_vectors.py` (Blocks 1→3, 4.1→4.4, 6.1→6.2, 7, 8, 9,
10, 12), but as a callable library function.

Notes
-----
- Arguments are explicit and use snake_case. Indexing differences between the
  MATLAB TV shapes and this pipeline (e.g., list of `h_est` per posDmrs) are
  resolved internally with the common single-position case handled by taking
  the first element.
- Callers are expected to provide values consistent with the reference blocks.
  This function does not perform parameter validation beyond basic shape use.
"""

from typing import Any

from ran.constants.constants import SC_PER_PRB
from ran.phy.numpy.pusch.channel_estimation import channel_est_dd
from ran.phy.numpy.pusch.codeblock_concatenation import codeblock_concatenation
from ran.phy.numpy.pusch.noise_estimation import estimate_covariance
from ran.phy.numpy.pusch.crc_decoder import crc_decode
from ran.phy.numpy.pusch.derate_matcher import derate_match
from ran.phy.numpy.pusch.descrambler import descramble_bits
from ran.phy.numpy.pusch.dmrs_utils import embed_dmrs_ul, extract_raw_dmrs_type_1
from ran.phy.numpy.pusch.equalizer import equalize
from ran.phy.numpy.pusch.gen_dmrs_sym import gen_dmrs_sym
from ran.phy.numpy.pusch.ldpc_decoder import ldpc_decode
from ran.phy.numpy.pusch.measure_rssi import measure_rssi
from ran.phy.numpy.pusch.noisevar_rsrp_sinr import noise_rsrp_sinr_db
from ran.phy.numpy.pusch.post_eq_noisevar_sinr import post_eq_noisevar_sinr
from ran.phy.numpy.pusch.soft_demapper import soft_demapper
from ran.utils._map_keys import trim_to_signature


def pusch_rx(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the reference PUSCH Rx pipeline end-to-end from a unified dict.

    - Inputs are auto-trimmed per block; only required kwargs are passed.
    - Outputs from prior blocks are injected via small overrides.
    """

    def _call(fn: Any, overrides: dict | None = None) -> Any:  # noqa: ANN401
        kwargs = trim_to_signature(fn, inputs)
        if overrides:
            kwargs.update(overrides)
        return fn(**kwargs)

    # Block 1: DMRS (generate only requested DMRS symbols)
    r_dmrs, _ = _call(gen_dmrs_sym)

    # Split RX grid into DMRS and DATA symbol indices
    rx_grid = inputs["xtf"]  # (nf, n_t, n_ant)
    n_f_start = SC_PER_PRB * inputs["start_prb"]
    n_f_end = n_f_start + SC_PER_PRB * inputs["n_prb"]
    freq_slice = slice(n_f_start, n_f_end)
    dmrs_sym = rx_grid[freq_slice, inputs["sym_idx_dmrs"], :]  # (nf, n_dmrs_sym, n_ant)
    data_sym = rx_grid[freq_slice, inputs["sym_idx_data"], :]  # (nf, n_data_sym, n_ant)

    # Block 2A: embed DMRS (PRB-band slice)
    x_dmrs = embed_dmrs_ul(  # TX side
        r_dmrs=r_dmrs,
        nl=inputs["nl"],
        port_idx=inputs["port_idx"],
        vec_scid=inputs["vec_scid"],
        energy=inputs["energy"],
    )  # (12*n_prb, n_dmrs_sym, nl)

    # Block 2B: extract received DMRS REs from RX grid
    y_dmrs = extract_raw_dmrs_type_1(  # RX side
        xtf_band_dmrs=dmrs_sym,
        nl=inputs["nl"],
        port_idx=inputs["port_idx"],
    )  # (nf, n_dmrs_sym, nl, n_ant)

    # Block 2-3: LS channel estimation (+ DD-trunc) + interpolation (+ dmrs sym axis)
    h_est = _call(channel_est_dd, {"x_dmrs": x_dmrs / 2, "y_dmrs": y_dmrs})
    # Divide by 2 to normalize DMRS power: channel estimation expects unit-power DMRS
    # but gen_dmrs_sym generates DMRS with 2x power amplification

    # Block 4: Covariance estimation (slice-based API)
    n_cov, mean_noise_var = estimate_covariance(
        xtf_band_dmrs=dmrs_sym,
        x_dmrs=x_dmrs,
        h_est_band_dmrs=h_est,
        rww_regularizer_val=inputs["rww_regularizer_val"],
    )

    # Block 5: Noise variance, RSRP and SINR estimation
    noise_db, rsrp_db, sinr_db = _call(
        noise_rsrp_sinr_db, {"mean_noise_var": mean_noise_var, "h_est": h_est}
    )

    # Block 6: Equalizer
    x_est, ree = _call(equalize, {"h_est": h_est, "noise_intf_cov": n_cov, "xtf_data": data_sym})

    # Block 7: post-EQ (slice frequency allocation before calling)
    post_noise_db, post_sinr_db = _call(post_eq_noisevar_sinr, {"ree": ree})

    # Block 8: soft demapper (pre-slice frequency allocation)
    llr_demap = _call(soft_demapper, {"x": x_est, "ree": ree})

    # Block 9: descramble bits
    # llr_seq = llr_demap.ravel(order="F")  # noqa: ERA001
    llr_seq = llr_demap.transpose(3, 2, 1, 0).ravel()
    tb_llr_descr = _call(descramble_bits, {"llrseq": llr_seq})

    # Block 10: Derate match
    derate_cbs, nv_parity, derate_cbs_idxs, derate_cbs_sizes = _call(
        derate_match, {"llr_descr": tb_llr_descr}
    )
    # Block 11: LDPC decode
    tb_cbs_est, num_itr = _call(ldpc_decode, {"derate_cbs": derate_cbs, "nv_parity": nv_parity})

    # Block 12: Codeblock concatenation
    tb_crc_est_vec, cb_err = _call(codeblock_concatenation, {"tb_cbs_est": tb_cbs_est})

    # Block 13: CRC decode
    tb_est, tb_err = _call(crc_decode, {"tb_crc_est": tb_crc_est_vec})

    # Block 14: DMRS RSSI (use PRB-band DMRS slice)
    dmrs_rssi_db, dmrs_rssi_reported_db = measure_rssi(xtf_band_dmrs=dmrs_sym)

    return {
        # Block 1: DMRS
        "r_dmrs": r_dmrs,
        # Block 2: embed DMRS (PRB-band slice) and extract received DMRS REs from RX grid
        # "XtfDmrs": x_dmrs,  # tv["XtfDmrs"][:, 2:3, :]  # noqa: ERA001
        # Block 2-3: LS channel estimation(+ DD-trunc) + interpolation
        "H_est": h_est,
        # Block 4: Covariance estimation
        "nCov": n_cov,
        "tmp_noiseVar": mean_noise_var,
        # Block 5: Noise variance, RSRP and SINR estimation
        "noiseVardB": noise_db,
        "rsrpdB": rsrp_db,
        "sinrdB": sinr_db,
        # Block 6: Equalizer
        "Ree": ree,
        "X_est": x_est,
        # Block 7: Post-EQ noise variance and SINR estimation
        "postEqNoiseVardB": post_noise_db,
        "postEqSinrdB": post_sinr_db,
        # Block 8: Soft demapper
        "LLR_demap": llr_demap,
        # Block 9: Descramble bits
        "LLR_descr": tb_llr_descr,
        # Block 10: Derate match
        "derateCbs": derate_cbs,
        "nV_parity": nv_parity,
        "derateCbsIndices": derate_cbs_idxs,
        "derateCbsIndicesSizes": derate_cbs_sizes,
        # Block 11: LDPC decode
        "TbCbs_est": tb_cbs_est,
        "numItr": num_itr,
        # Block 12: Codeblock concatenation
        "TbCrc_est": tb_crc_est_vec,
        "cb_err": cb_err,
        # Block 13: CRC decode
        "Tb_est": tb_est,
        "Tb_err": tb_err,
        # Block 14: DMRS RSSI
        "dmrsRssiDb": dmrs_rssi_db,
        "dmrsRssiReportedDb": dmrs_rssi_reported_db,
    }


def pusch_inner_rx(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the reference PUSCH Rx pipeline end-to-end from a unified dict.

    - Inputs are auto-trimmed per block; only required kwargs are passed.
    - Outputs from prior blocks are injected via small overrides.
    """
    # Split RX grid into DMRS and DATA symbol indices
    rx_grid = inputs["xtf"]  # (nf, n_t, n_ant)
    n_f_start = SC_PER_PRB * inputs["start_prb"]
    n_f_end = n_f_start + SC_PER_PRB * inputs["n_prb"]
    freq_slice = slice(n_f_start, n_f_end)
    dmrs_sym = rx_grid[freq_slice, inputs["sym_idx_dmrs"], :]  # (nf, n_dmrs_sym, n_ant)
    data_sym = rx_grid[freq_slice, inputs["sym_idx_data"], :]  # (nf, n_data_sym, n_ant)

    n_f = rx_grid.shape[0]

    # Block 1: DMRS (generate only requested DMRS symbols)
    r_dmrs, _ = gen_dmrs_sym(
        slot_number=inputs["slot_number"],
        n_f=n_f,
        n_dmrs_id=inputs["n_dmrs_id"],
        sym_idx_dmrs=inputs["sym_idx_dmrs"],
    )

    # Block 2: embed DMRS (PRB-band slice) and extract received DMRS REs from RX grid
    x_dmrs = embed_dmrs_ul(
        r_dmrs=r_dmrs,
        nl=inputs["nl"],
        port_idx=inputs["port_idx"],
        vec_scid=inputs["vec_scid"],
        energy=inputs["energy"],
    )  # (12*n_prb, n_dmrs_sym, nl)

    y_dmrs = extract_raw_dmrs_type_1(
        xtf_band_dmrs=dmrs_sym,
        nl=inputs["nl"],
        port_idx=inputs["port_idx"],
    )  # (nf, n_dmrs_sym, nl, n_ant)

    # Block 2-3: LS channel estimation (+ DD-trunc) + interpolation (+ dmrs sym axis)
    h_est = channel_est_dd(x_dmrs=x_dmrs / 2, y_dmrs=y_dmrs)
    # Divide by 2 to normalize DMRS power: channel estimation expects unit-power DMRS
    # but gen_dmrs_sym generates DMRS with 2x power amplification

    # Block 4: Covariance estimation (slice-based API)
    n_cov, mean_noise_var = estimate_covariance(
        xtf_band_dmrs=dmrs_sym,
        x_dmrs=x_dmrs,
        h_est_band_dmrs=h_est,
        rww_regularizer_val=inputs["rww_regularizer_val"],
    )

    # Block 5: Noise variance, RSRP and SINR estimation
    noise_db, rsrp_db, sinr_db = noise_rsrp_sinr_db(
        mean_noise_var=mean_noise_var,
        h_est=h_est,
        layer2ue=inputs["layer2ue"],
        n_ue=inputs["n_ue"],
    )

    # Block 6: Equalizer
    x_est, ree = equalize(
        h_est=h_est,
        noise_intf_cov=n_cov,
        xtf_data=data_sym,
    )

    # Block 7: post-EQ
    post_noise_db, post_sinr_db = post_eq_noisevar_sinr(
        ree=ree,
        layer2ue=inputs["layer2ue"],
        n_ue=inputs["n_ue"],
    )

    # Block 8: soft demapper
    llr_demap = soft_demapper(
        x=x_est,
        ree=ree,
        qam_bits=inputs["qam_bits"],
    )  # (qam_bits, nl, nf_alloc, n_sym)

    # Block 14: DMRS RSSI (use PRB-band DMRS slice)
    dmrs_rssi_db, dmrs_rssi_reported_db = measure_rssi(xtf_band_dmrs=dmrs_sym)

    return {
        # Block 1: DMRS
        "r_dmrs": r_dmrs,
        # Block 2: embed DMRS (PRB-band slice) and extract received DMRS REs from RX grid
        "Xdmrs": x_dmrs,
        # Block 2-3: LS channel estimation (+ DD-trunc) + interpolation (+ dmrs sym axis)
        "H_est": h_est,
        # Block 4: Covariance estimation
        "nCov": n_cov,
        "tmp_noiseVar": mean_noise_var,
        # Block 5: Noise variance, RSRP and SINR estimation
        "noiseVardB": noise_db,
        "rsrpdB": rsrp_db,
        "sinrdB": sinr_db,
        # Block 6: Equalizer
        "X_est": x_est,
        "Ree": ree,
        # Block 7: Post-EQ noise variance and SINR estimation
        "postEqNoiseVardB": post_noise_db,
        "postEqSinrdB": post_sinr_db,
        # Block 8: Soft demapper
        "LLR_demap": llr_demap,
        # Block 14: Measure RSSI
        "dmrsRssiDb": dmrs_rssi_db,
        "dmrsRssiReportedDb": dmrs_rssi_reported_db,
    }
