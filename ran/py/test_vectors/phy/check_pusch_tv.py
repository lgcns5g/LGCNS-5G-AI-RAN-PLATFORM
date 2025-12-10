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

"""Test-vector tests for PUSCH refactored pipeline."""

from collections.abc import Callable
from typing import Any

import numpy as np

from ran.constants import LLR_CLAMP_ABS, SC_PER_PRB
from ran.phy.numpy.pusch.dmrs_utils import embed_dmrs_ul, extract_raw_dmrs_type_1
from ran.utils._max_abs_diff import max_abs_diff
from test_vectors._tv_utils import LOGGER, TvLoader

# Key map
from test_vectors.phy.key_map import (
    build_cb_concat_kwargs,
    build_cov_kwargs,
    build_crc_kwargs,
    build_derate_kwargs,
    build_descramble_kwargs,
    build_dmrs_kwargs,
    build_dmrs_rssi_kwargs,
    build_embed_dmrs_kwargs,
    build_eq_apply_kwargs,
    build_eq_derive_kwargs,
    build_eq_equalizer_kwargs,
    build_ldpc_kwargs,
    build_ls_kwargs,
    build_ncov_kwargs,
    build_noise_rsrp_sinr_kwargs,
    build_noisevar_kwargs,
    build_post_eq_kwargs,
    build_pusch_inner_rx_kwargs,
    build_pusch_rx_kwargs,
    build_r_tilde_kwargs,
    build_rsrp_kwargs,
    build_shrink_kwargs,
    build_sinr_kwargs,
    build_softdemap_kwargs,
)


def tv_block1_dmrs(tv_name: str, dmrs_func: Callable) -> None:
    """Test 1: DMRS generation."""
    tv = TvLoader.load(tv_name)
    dmrs_kwargs = build_dmrs_kwargs(tv)
    r_dmrs, _ = dmrs_func(**dmrs_kwargs)
    tv_r_dmrs = tv["r_dmrs"][:, dmrs_kwargs["sym_idx_dmrs"], :]
    max_abs_diff(r_dmrs, tv_r_dmrs)
    np.testing.assert_allclose(r_dmrs, tv_r_dmrs)


def tv_block2_ls(tv_name: str, ls_func: Callable) -> None:
    """Test 2: Least Squares Channel Estimation.

    Build DMRS slices locally (PRB band) and call the core LS routine that
    operates on slices to avoid coupling test logic to embedding/extraction
    helpers inside the implementation.
    """
    # imports moved to top-level

    tv = TvLoader.load(tv_name)

    # Build kwargs
    ls_kwargs = build_ls_kwargs(tv)

    # Precompute slices (band-trim and DMRS-only time for both r_dmrs and xtf)
    n_f_start = SC_PER_PRB * ls_kwargs["start_prb"]
    n_f_end = n_f_start + SC_PER_PRB * ls_kwargs["n_prb"]
    r_dmrs_band = ls_kwargs["r_dmrs"][:, ls_kwargs["sym_idx_dmrs"], :]
    x_dmrs = embed_dmrs_ul(
        r_dmrs=r_dmrs_band,
        nl=ls_kwargs["nl"],
        port_idx=ls_kwargs["port_idx"],
        vec_scid=ls_kwargs["vec_scid"],
        energy=2,
    )
    y_dmrs = extract_raw_dmrs_type_1(
        xtf_band_dmrs=ls_kwargs["xtf"][n_f_start:n_f_end, ls_kwargs["sym_idx_dmrs"], :],
        nl=ls_kwargs["nl"],
        port_idx=ls_kwargs["port_idx"],
    )

    # Core LS on slices, then undo RX scaling to match TV reference
    h_ls_est = ls_func(x_dmrs=x_dmrs, y_dmrs=y_dmrs) / np.sqrt(2)

    LOGGER.debug("H_LS_est diffs:")
    max_abs_diff(h_ls_est, tv["H_LS_est"][::2])
    np.testing.assert_allclose(h_ls_est, tv["H_LS_est"][::2], err_msg="H_LS_est")


def tv_block3_ls_dd(tv_name: str, ls_dd_func: Callable) -> None:
    """Test 3: Least Squares Channel Estimation (w/ DD-trunc & interpolation).

    Build DMRS slices in the TV and, if the provided function accepts
    slices (x_dmrs, y_dmrs), route to that core; otherwise, call the original
    signature for backwards compatibility.
    """
    tv = TvLoader.load(tv_name)
    ls_dd_kwargs = build_ls_kwargs(tv)

    # Precompute slices (band-trim and DMRS-only time for both r_dmrs and xtf)
    n_f_start = SC_PER_PRB * ls_dd_kwargs["start_prb"]
    n_f_end = n_f_start + SC_PER_PRB * ls_dd_kwargs["n_prb"]
    r_dmrs_band = ls_dd_kwargs["r_dmrs"][:, ls_dd_kwargs["sym_idx_dmrs"], :]
    x_dmrs = embed_dmrs_ul(
        r_dmrs=r_dmrs_band,
        nl=ls_dd_kwargs["nl"],
        port_idx=ls_dd_kwargs["port_idx"],
        vec_scid=ls_dd_kwargs["vec_scid"],
        energy=2.0,
    )
    y_dmrs = extract_raw_dmrs_type_1(
        xtf_band_dmrs=ls_dd_kwargs["xtf"][n_f_start:n_f_end, ls_dd_kwargs["sym_idx_dmrs"], :],
        nl=ls_dd_kwargs["nl"],
        port_idx=ls_dd_kwargs["port_idx"],
    )

    # Prefer slice-based core if function matches that signature; else fallback
    h_ls_dd_est = ls_dd_func(x_dmrs=x_dmrs / 2, y_dmrs=y_dmrs)

    h_ls_dd_est = h_ls_dd_est[..., 0].swapaxes(0, 2)
    LOGGER.debug("H_est diffs:")
    max_abs_diff(h_ls_dd_est, tv["H_est"])
    np.testing.assert_allclose(h_ls_dd_est, tv["H_est"], err_msg="H_est", rtol=2e-2)


def tv_block4_cov(tv_name: str, cov_from_slices_func: Callable) -> None:
    """Test 4 (slice-based): Covariance Estimation using PRB-band DMRS slices.

    Builds `x_dmrs`, `xtf_band_dmrs`, and `h_est_band_dmrs` using DMRS helpers,
    then calls the slice-based covariance path.
    """
    tv = TvLoader.load(tv_name)

    # Build kwargs using existing key-map helpers
    embed_kwargs = build_embed_dmrs_kwargs(tv)
    cov_kwargs = build_cov_kwargs(tv)

    # Build PRB-band DMRS slice
    embed_kwargs["r_dmrs"] = embed_kwargs["r_dmrs"][:, embed_kwargs["sym_idx_dmrs"], :]
    embed_kwargs.pop("sym_idx_dmrs")
    embed_kwargs.pop("n_prb")
    embed_kwargs.pop("start_prb")
    x_dmrs = embed_dmrs_ul(**embed_kwargs)

    # Create PRB-band + DMRS-symbol slices for xtf and h_est
    xtf = cov_kwargs.pop("xtf")
    h_est = cov_kwargs.pop("h_est")
    sym_idx_dmrs = cov_kwargs.pop("sym_idx_dmrs")
    n_prb = cov_kwargs.pop("n_prb")
    start_prb = cov_kwargs.pop("start_prb")

    n_f_start, n_f_end = SC_PER_PRB * start_prb, SC_PER_PRB * (start_prb + n_prb)

    freq_slice = slice(n_f_start, n_f_end)

    # Call slice-based covariance (explicit sym_idx path)
    n_cov, mean_noise_var = cov_from_slices_func(
        xtf_band_dmrs=xtf[freq_slice, sym_idx_dmrs, :],
        x_dmrs=x_dmrs[freq_slice, :, :],
        h_est_band_dmrs=h_est[freq_slice, :, :, :],
        rww_regularizer_val=cov_kwargs["rww_regularizer_val"],
    )

    # Compare with TVs
    max_abs_diff(n_cov[..., 0], tv["nCov"])
    np.testing.assert_allclose(n_cov[..., 0], tv["nCov"], err_msg="nCov")
    max_abs_diff(mean_noise_var, tv["tmp_noiseVar"])
    np.testing.assert_allclose(mean_noise_var, tv["tmp_noiseVar"], err_msg="tmp_noiseVar")


def tv_block4a_embed_dmrs(tv_name: str, embed_func: Callable) -> None:
    """Test 4a: Embed DMRS."""
    tv = TvLoader.load(tv_name)
    embed_kwargs = build_embed_dmrs_kwargs(tv)
    embed_kwargs["r_dmrs"] = embed_kwargs["r_dmrs"][:, embed_kwargs["sym_idx_dmrs"], :]
    start_prb = embed_kwargs.pop("start_prb")
    n_prb = embed_kwargs.pop("n_prb")
    sym_idx_dmrs = embed_kwargs.pop("sym_idx_dmrs")
    x_dmrs = embed_func(**embed_kwargs)
    # Compare only the PRB-band DMRS rows from the TV full-grid
    nf_start = SC_PER_PRB * start_prb
    nf_end = nf_start + SC_PER_PRB * n_prb
    tv_dmrs = tv["XtfDmrs"]
    tv_band = tv_dmrs[nf_start:nf_end, sym_idx_dmrs, :]
    # IMPLICIT: Returns slice with layer axis instead of antenna; layer==antenna here
    tv_band = tv_band[..., 0][..., None]
    max_abs_diff(x_dmrs, tv_band)  # Equivalent to tv["XtfDmrs"][:, 2:3, :]
    np.testing.assert_allclose(x_dmrs, tv_band)


def tv_block4b_r_tilde(tv_name: str, rtilde_func: Callable) -> None:
    """Test 4b: Estimate R_tilde from PRB-band DMRS slices."""
    tv = TvLoader.load(tv_name)
    rtilde_kwargs = build_r_tilde_kwargs(tv)
    xtf = rtilde_kwargs.pop("xtf")
    h_est = rtilde_kwargs.pop("h_est")
    sym_idx_dmrs = rtilde_kwargs.pop("sym_idx_dmrs")
    n_prb = rtilde_kwargs.pop("n_prb")
    start_prb = rtilde_kwargs.pop("start_prb")

    n_f_start = SC_PER_PRB * start_prb
    n_f_end = n_f_start + SC_PER_PRB * n_prb

    r_tilde = rtilde_func(
        xtf_band_dmrs=xtf[n_f_start:n_f_end, sym_idx_dmrs, :],  # (n_f, n_pos, n_ant)
        x_dmrs=rtilde_kwargs["x_dmrs"][n_f_start:n_f_end, sym_idx_dmrs, :],  # (n_f, n_pos, nl)
        h_est_band_dmrs=h_est[n_f_start:n_f_end, :, :, :],  # (n_f, nl, n_ant, n_pos)
    )
    max_abs_diff(r_tilde, tv["r_tilde"])
    np.testing.assert_allclose(r_tilde, tv["r_tilde"])


def tv_block4c_ncov(tv_name: str, ncov_func: Callable) -> None:
    """Test 4c: Estimate Noise Covariance."""
    tv = TvLoader.load(tv_name)
    ncov_kwargs = build_ncov_kwargs(tv)
    _ = ncov_kwargs.pop("sym_idx_dmrs")  # no longer needed
    n_cov, mean_noise_var = ncov_func(**ncov_kwargs)
    max_abs_diff(n_cov[..., 0], tv["nCov_before_shrinkage"])
    np.testing.assert_allclose(
        n_cov[..., 0], tv["nCov_before_shrinkage"], err_msg="nCov_before_shrinkage", rtol=1e-6
    )
    max_abs_diff(mean_noise_var, tv["tmp_noiseVar"])
    np.testing.assert_allclose(
        mean_noise_var, tv["tmp_noiseVar"], err_msg="tmp_noiseVar", rtol=1e-6
    )


def tv_block4d_shrink(tv_name: str, shrink_func: Callable) -> None:
    """Test 4d: Shrink Noise Covariance."""
    tv = TvLoader.load(tv_name)
    shrink_kwargs = build_shrink_kwargs(tv)
    n_cov_shrunk = shrink_func(**shrink_kwargs)
    max_abs_diff(n_cov_shrunk[..., 0], tv["nCov"])
    np.testing.assert_allclose(n_cov_shrunk[..., 0], tv["nCov"])


def tv_block5a_noisevar(tv_name: str, noisevar_func: Callable) -> None:
    """Test 5a: Noise variance estimation."""
    tv = TvLoader.load(tv_name)
    noisevar_kwargs = build_noisevar_kwargs(tv)
    noise_db = noisevar_func(**noisevar_kwargs)
    LOGGER.debug("noiseVardB diffs:")
    max_abs_diff(noise_db, tv["noiseVardB"])
    np.testing.assert_allclose(noise_db, tv["noiseVardB"][0])


def tv_block5b_rsrp(tv_name: str, rsrp_func: Callable) -> None:
    """Test 5b: RSRP estimation."""
    tv = TvLoader.load(tv_name)
    rsrp_kwargs = build_rsrp_kwargs(tv)
    rsrp_db_val = rsrp_func(**rsrp_kwargs)
    LOGGER.debug("rsrpdB diffs:")
    max_abs_diff(rsrp_db_val, tv["rsrpdB"])
    np.testing.assert_allclose(rsrp_db_val, tv["rsrpdB"])


def tv_block5c_sinr(tv_name: str, sinr_func: Callable) -> None:
    """Test 5c: SINR estimation."""
    tv = TvLoader.load(tv_name)
    sinr_kwargs = build_sinr_kwargs(tv)
    sinr_db_val = sinr_func(**sinr_kwargs)
    LOGGER.debug("sinrdB diffs:")
    max_abs_diff(sinr_db_val, tv["sinrdB"])
    np.testing.assert_allclose(sinr_db_val, tv["sinrdB"])


def tv_block5_noise_rsrp_sinr(tv_name: str, noise_rsrp_sinr_db_func: Callable) -> None:
    """Test 5: Noise variance, RSRP and SINR estimation."""
    tv = TvLoader.load(tv_name)
    noisevar_kwargs = build_noise_rsrp_sinr_kwargs(tv)
    noise_db, rsrp_db_val, sinr_db_val = noise_rsrp_sinr_db_func(**noisevar_kwargs)
    LOGGER.debug("noiseVardB diffs:")
    max_abs_diff(noise_db, tv["noiseVardB"])
    np.testing.assert_allclose(noise_db, tv["noiseVardB"][0])
    LOGGER.debug("rsrpdB diffs:")
    max_abs_diff(rsrp_db_val, tv["rsrpdB"])
    np.testing.assert_allclose(rsrp_db_val, tv["rsrpdB"])
    LOGGER.debug("sinrdB diffs:")
    max_abs_diff(sinr_db_val, tv["sinrdB"])
    np.testing.assert_allclose(sinr_db_val, tv["sinrdB"])


def tv_block6a_equalizer_derive(tv_name: str, derive_eq_func: Callable) -> None:
    """Test 6a: MIMO equalizer derive (implementation path)."""
    tv = TvLoader.load(tv_name)
    derive_kwargs = build_eq_derive_kwargs(tv)
    w, ree = derive_eq_func(**derive_kwargs)
    LOGGER.debug("W diffs:")
    max_abs_diff(w, tv["W"])
    LOGGER.debug("Ree diffs:")
    max_abs_diff(ree, tv["Ree"])
    np.testing.assert_allclose(w, tv["W"], err_msg="W")
    np.testing.assert_allclose(ree[..., 0], tv["Ree"], err_msg="Ree")


def tv_block6b_equalizer_apply(tv_name: str, apply_eq_func: Callable) -> None:
    """Test 6b: MIMO equalizer apply (implementation path)."""
    tv = TvLoader.load(tv_name)
    apply_kwargs = build_eq_apply_kwargs(tv)
    # Adapt to new API: apply_equalizer(xtf_data, w)
    xtf = apply_kwargs.pop("xtf")
    sym_idx_data = apply_kwargs.pop("sym_idx_data")
    apply_kwargs["xtf_data"] = xtf[:, sym_idx_data, :]
    x_est = apply_eq_func(**apply_kwargs)
    LOGGER.debug("X_est diffs:")
    max_abs_diff(x_est, tv["X_est"])
    np.testing.assert_allclose(x_est[..., 0], tv["X_est"])


def tv_block6_equalizer(tv_name: str, equalize_func: Callable) -> None:
    """Test 6c: Equalizer block (derive + apply)."""
    tv = TvLoader.load(tv_name)
    equalize_kwargs = build_eq_equalizer_kwargs(tv)
    # Adapt to new API: equalize(..., xtf_data=xtf[:, sym_idx_data, :])
    xtf = equalize_kwargs.pop("xtf")
    sym_idx_data = equalize_kwargs.pop("sym_idx_data")
    equalize_kwargs["xtf_data"] = xtf[:, sym_idx_data, :]
    x_est, ree = equalize_func(**equalize_kwargs)
    LOGGER.debug("X_est diffs:")
    max_abs_diff(x_est, tv["X_est"])
    np.testing.assert_allclose(x_est[..., 0], tv["X_est"])
    LOGGER.debug("Ree diffs:")
    max_abs_diff(ree, tv["Ree"])
    np.testing.assert_allclose(ree[..., 0], tv["Ree"])


def tv_block7_post_eq(tv_name: str, post_eq_func: Callable) -> None:
    """Test 7: Post-EQ noise variance and SINR."""
    tv = TvLoader.load(tv_name)
    post_kwargs = build_post_eq_kwargs(tv)
    # Adapt to new API: post_eq_noisevar_sinr(ree, layer2ue, n_ue)
    ree = post_kwargs.pop("ree")
    start_prb = post_kwargs.pop("start_prb")
    n_prb = post_kwargs.pop("n_prb")
    n_f_start = SC_PER_PRB * start_prb
    n_f_end = n_f_start + SC_PER_PRB * n_prb
    post_kwargs["ree"] = ree[:, n_f_start:n_f_end, :]
    post_noise_db, post_sinr_db = post_eq_func(**post_kwargs)
    LOGGER.debug("postEqNoiseVardB diffs:")
    max_abs_diff(post_noise_db, tv["postEqNoiseVardB"])
    LOGGER.debug("postEqSinrdB diffs:")
    max_abs_diff(post_sinr_db, tv["postEqSinrdB"])
    np.testing.assert_allclose(post_noise_db, tv["postEqNoiseVardB"], err_msg="postEqNoiseVardB")
    np.testing.assert_allclose(post_sinr_db, tv["postEqSinrdB"], err_msg="postEqSinrdB")


def tv_block8_soft_demapper(tv_name: str, soft_demap_func: Callable) -> None:
    """Test 8: Soft demapper."""
    tv = TvLoader.load(tv_name)
    sd_kwargs = build_softdemap_kwargs(tv)
    # Adapt to new API: soft_demapper(x, ree, qam_bits)
    x_est = sd_kwargs.pop("x_est")
    ree = sd_kwargs.pop("ree")
    start_prb = sd_kwargs.pop("start_prb")
    n_prb = sd_kwargs.pop("n_prb")
    n_f_start = SC_PER_PRB * start_prb
    n_f_end = n_f_start + SC_PER_PRB * n_prb
    sd_kwargs["x"] = x_est[n_f_start:n_f_end, :, :]
    sd_kwargs["ree"] = ree[:, n_f_start:n_f_end, :]
    llr_demap = soft_demap_func(**sd_kwargs)
    LOGGER.debug("LLRseq diffs:")
    llrseq = np.ravel(llr_demap, order="F")[..., None]
    llrseq_tv = np.clip(tv["LLRseq"], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)
    llr_demap_tv = np.clip(tv["LLR_demap"], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)
    max_abs_diff(llrseq, llrseq_tv)
    LOGGER.debug("LLR_demap diffs:")
    max_abs_diff(llr_demap, llr_demap_tv)
    np.testing.assert_allclose(llrseq, llrseq_tv, err_msg="LLRseq")
    np.testing.assert_allclose(llr_demap, llr_demap_tv, err_msg="LLR_demap")


def tv_block9_descramble(tv_name: str, descramble_func: Callable) -> None:
    """Test 9: Descramble bits."""
    tv = TvLoader.load(tv_name)
    descr_kwargs = build_descramble_kwargs(tv)
    llr_descr = descramble_func(**descr_kwargs)
    LOGGER.debug("LLR_descr diffs:")
    max_abs_diff(llr_descr, tv["LLR_descr"])
    np.testing.assert_allclose(llr_descr, tv["LLR_descr"][..., 0], err_msg="LLR_descr")


def tv_block10_derate(tv_name: str, derate_func: Callable) -> None:
    """Test 10a: Derate."""
    tv = TvLoader.load(tv_name)
    dm_kwargs = build_derate_kwargs(tv)
    derate_cbs, nv_parity, derate_cbs_idxs, derate_cbs_sizes = derate_func(**dm_kwargs)
    LOGGER.debug("derateCbs diffs:")
    derate_cbs_tv = np.clip(tv["derateCbs"], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)
    max_abs_diff(derate_cbs, derate_cbs_tv)
    LOGGER.debug("nV_parity diffs:")
    max_abs_diff(nv_parity, tv["nV_parity"])
    LOGGER.debug("derateCbsIndices diffs:")
    max_abs_diff(derate_cbs_idxs, tv["derateCbsIndices"])
    LOGGER.debug("derateCbsIndicesSizes diffs:")
    max_abs_diff(derate_cbs_sizes, tv["derateCbsIndicesSizes"])
    np.testing.assert_allclose(derate_cbs, derate_cbs_tv, err_msg="derateCbs")
    np.testing.assert_allclose(nv_parity, tv["nV_parity"], err_msg="nV_parity")
    np.testing.assert_allclose(derate_cbs_idxs, tv["derateCbsIndices"], err_msg="derateCbsIndices")
    np.testing.assert_allclose(
        derate_cbs_sizes,
        tv["derateCbsIndicesSizes"][..., 0],
        err_msg="derateCbsIndicesSizes",
    )


def tv_block11_ldpc_decode(tv_name: str, ldpc_decode_func: Callable) -> None:
    """Test 10b: LDPC."""
    tv = TvLoader.load(tv_name)
    ldpc_kwargs = build_ldpc_kwargs(tv)
    tb_cbs_est, num_itr = ldpc_decode_func(**ldpc_kwargs)
    LOGGER.debug("TbCbs_est diffs:")
    max_abs_diff(tb_cbs_est, tv["TbCbs_est"])
    LOGGER.debug("numItr diffs:")
    max_abs_diff(num_itr, tv["numItr"])
    np.testing.assert_allclose(tb_cbs_est, tv["TbCbs_est"], err_msg="TbCbs_est")
    np.testing.assert_allclose(num_itr, tv["numItr"][:, 0], err_msg="numItr")


def tv_block12_cb_concat(tv_name: str, concat_func: Callable) -> None:
    """Test 10c: CB concatenation."""
    tv = TvLoader.load(tv_name)
    concat_kwargs = build_cb_concat_kwargs(tv)
    tb_crc_est_vec, cb_err = concat_func(**concat_kwargs)
    LOGGER.debug("TbCrc_est diffs:")
    max_abs_diff(tb_crc_est_vec, tv["TbCrc_est"])
    LOGGER.debug("cbErr diffs:")
    max_abs_diff(cb_err, tv["cbErr"])
    np.testing.assert_allclose(tb_crc_est_vec, tv["TbCrc_est"][..., 0], err_msg="TbCrc_est")
    np.testing.assert_allclose(cb_err, tv["cbErr"][:, 0], err_msg="cbErr")


def tv_block13_crc_decode(tv_name: str, crc_decode_func: Callable) -> None:
    """Test 10d: CRC."""
    tv = TvLoader.load(tv_name)
    crc_kwargs = build_crc_kwargs(tv)
    tb_est, tb_err = crc_decode_func(**crc_kwargs)
    LOGGER.debug("Tb_est diffs:")
    max_abs_diff(tb_est, tv["Tb_est"])
    np.testing.assert_allclose(tb_est, tv["Tb_est"], err_msg="Tb_est")
    LOGGER.debug("tbErr diffs:")
    max_abs_diff(tb_err, tv["tbErr"])
    np.testing.assert_allclose(tb_err, tv["tbErr"][:, 0], err_msg="tbErr")


def tv_block14_dmrs_rssi(tv_name: str, rssi_func: Callable) -> None:
    """Test 12: DMRS RSSI."""
    tv = TvLoader.load(tv_name)
    dmrs_rssi_kwargs = build_dmrs_rssi_kwargs(tv)

    # Adapt to new API: measure_rssi(xtf_band_dmrs)
    xtf = dmrs_rssi_kwargs.pop("xtf")
    sym_idx_dmrs = dmrs_rssi_kwargs.pop("sym_idx_dmrs")
    start_prb = dmrs_rssi_kwargs.pop("start_prb")
    n_prb = dmrs_rssi_kwargs.pop("n_prb")
    n_f_start = SC_PER_PRB * start_prb
    n_f_end = n_f_start + SC_PER_PRB * n_prb
    dmrs_rssi_kwargs["xtf_band_dmrs"] = xtf[n_f_start:n_f_end, sym_idx_dmrs, :]

    dmrs_rssi_db, dmrs_rssi_rep_db = rssi_func(**dmrs_rssi_kwargs)
    LOGGER.debug("dmrsRssiDb diffs:")
    max_abs_diff(dmrs_rssi_db, tv["dmrsRssiDb"])
    LOGGER.debug("dmrsRssiReportedDb diffs:")
    max_abs_diff(dmrs_rssi_rep_db, tv["dmrsRssiReportedDb"])
    rtol = 1e-6
    np.testing.assert_allclose(dmrs_rssi_db, tv["dmrsRssiDb"], err_msg="dmrsRssiDb", rtol=rtol)
    np.testing.assert_allclose(
        dmrs_rssi_rep_db,
        tv["dmrsRssiReportedDb"],
        err_msg="dmrsRssiReportedDb",
        rtol=rtol,
    )


def tv_pusch_rx(tv_name: str, pusch_rx_func: Callable) -> None:
    """End-to-end test for unified reference PUSCH pipeline function.

    Uses the same TV as other block tests and compares a few key outputs.
    """

    def check(
        key: str,
        rtol: float | None = None,
        atol: float | None = None,
        out: Any | None = None,  # noqa: ANN401
        out_tv: Any | None = None,  # noqa: ANN401
    ) -> None:
        """Check helper to log, diff, and assert for a given output key.

        Args:
            key: The key of the output to check.
            rtol: The relative tolerance for the comparison.
            atol: The absolute tolerance for the comparison.
            out: The output to compare to the TV.
            out_tv: The TV output to compare to the output.
        """
        o = outputs[key] if out is None else out
        t = tv[key] if out_tv is None else out_tv
        LOGGER.debug(f"{key} diffs (e2e):")
        max_abs_diff(o, t)
        if atol is not None:
            np.testing.assert_allclose(o, t, err_msg=key, atol=atol)
        else:
            np.testing.assert_allclose(o, t, err_msg=key, rtol=rtol if rtol is not None else 1e-7)

    tv = TvLoader.load(tv_name)
    inputs = build_pusch_rx_kwargs(tv)
    outputs = pusch_rx_func(inputs)  # do not unpack for full receiver

    # Block 1: DMRS
    check("r_dmrs", out_tv=tv["r_dmrs"][:, inputs["sym_idx_dmrs"], :])

    # Block 2+3: LS + DD-trunc channel estimation
    check("H_est", rtol=2e-2, out=outputs["H_est"][..., 0].swapaxes(0, 2))

    # Block 4: Covariance estimation
    check("nCov", atol=3e-5, out=outputs["nCov"][..., 0])
    check("tmp_noiseVar", rtol=3e-1)

    # Block 5: Noise variance, RSRP and SINR estimation
    check("noiseVardB", rtol=3e-3, out=outputs["noiseVardB"][..., None])
    check("rsrpdB", atol=6e-3)
    check("sinrdB", atol=1e-1)

    # Block 6: Equalizer
    check("X_est", rtol=2e-2, out=outputs["X_est"][..., 0])
    check("Ree", out=outputs["Ree"][..., 0])

    # Block 7: Post-EQ noise variance and SINR
    check("postEqNoiseVardB")
    check("postEqSinrdB")

    # Block 8: Soft demapper
    llr_demap_tv = np.clip(tv["LLR_demap"], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)
    check("LLR_demap", rtol=2e-1, out_tv=llr_demap_tv)

    # Block 9: Descramble bits
    llr_descr_tv = np.clip(tv["LLR_descr"], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)
    check("LLR_descr", rtol=2e-1, out=outputs["LLR_descr"][..., None], out_tv=llr_descr_tv)

    # Block 10: Derate Match
    derate_cbs_tv = np.clip(tv["derateCbs"], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)
    check("derateCbs", rtol=2e-1, out_tv=derate_cbs_tv)
    check("nV_parity", rtol=2e-1)
    check("derateCbsIndices")
    check("derateCbsIndicesSizes", out=outputs["derateCbsIndicesSizes"][..., None])

    # Block 11: LDPC
    check("TbCbs_est")
    check("numItr", out=outputs["numItr"][..., None])

    # Block 12: CB Concatenation
    check("TbCrc_est", out=outputs["TbCrc_est"][..., None])
    check("Tb_est", out=outputs["Tb_est"][..., None])

    # Block 14: DMRS RSSI
    check("dmrsRssiDb", rtol=1e-6)
    check("dmrsRssiReportedDb", rtol=1e-6)


def tv_pusch_inner_rx(
    tv_name: str,
    pusch_inner_rx_func: Callable,
    *,
    unpack: bool = False,
) -> None:
    """End-to-end test for unified reference PUSCH pipeline function.

    Uses the same TV as other block tests and compares a few key outputs.
    """

    def check(
        key: str,
        rtol: float | None = None,
        atol: float | None = None,
        out: Any | None = None,  # noqa: ANN401
        out_tv: Any | None = None,  # noqa: ANN401
    ) -> None:
        """Check helper to log, diff, and assert for a given output key."""
        o = outputs[key] if out is None else out
        t = tv[key] if out_tv is None else out_tv
        LOGGER.debug(f"{key} diffs (e2e):")
        max_abs_diff(o, t)
        if atol is not None:
            np.testing.assert_allclose(o, t, err_msg=key, atol=atol)
        else:
            np.testing.assert_allclose(o, t, err_msg=key, rtol=rtol if rtol is not None else 1e-7)

    tv = TvLoader.load(tv_name)
    inputs = build_pusch_inner_rx_kwargs(tv)
    outputs = pusch_inner_rx_func(**inputs) if unpack else pusch_inner_rx_func(inputs)

    # Block 1: DMRS
    check("r_dmrs", out_tv=tv["r_dmrs"][:, inputs["sym_idx_dmrs"], :])

    # Block 2+3: LS + DD-trunc channel estimation
    check("H_est", rtol=2e-2, out=outputs["H_est"][..., 0].swapaxes(0, 2))

    # Block 4: Covariance estimation
    check("nCov", atol=3e-5, out=outputs["nCov"][..., 0])
    check("tmp_noiseVar", rtol=3e-1)

    # Block 5: Noise variance, RSRP and SINR estimation
    check("noiseVardB", rtol=3e-3, out=outputs["noiseVardB"][..., None])
    check("rsrpdB", atol=6e-3)
    check("sinrdB", atol=1e-1)

    # Block 6: Equalizer
    check("X_est", rtol=2e-2, out=outputs["X_est"][..., 0])
    check("Ree", out=outputs["Ree"][..., 0])

    # Block 7: Post-EQ noise variance and SINR
    check("postEqNoiseVardB")
    check("postEqSinrdB")

    # Block 8: Soft demapper
    llr_demap_tv = np.clip(tv["LLR_demap"], -LLR_CLAMP_ABS, LLR_CLAMP_ABS)
    check("LLR_demap", rtol=2e-1, out_tv=llr_demap_tv)

    # Block 14: DMRS RSSI
    check("dmrsRssiDb", rtol=1e-6)
    check("dmrsRssiReportedDb", rtol=1e-6)
