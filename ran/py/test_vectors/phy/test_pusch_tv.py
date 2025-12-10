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

from ran.phy.numpy.pusch import (
    apply_equalizer,
    channel_est_dd,
    channel_est_ls,
    codeblock_concatenation,
    crc_decode,
    derate_match,
    derive_equalizer,
    descramble_bits,
    embed_dmrs_ul,
    equalize,
    estimate_covariance,
    estimate_noise_covariance,
    estimate_r_tilde,
    gen_dmrs_sym,
    ldpc_decode,
    measure_rssi,
    ncov_shrinkage,
    noise_rsrp_sinr_db,
    noise_variance_db,
    post_eq_noisevar_sinr,
    pusch_rx,
    rsrp_db,
    sinr_db,
    soft_demapper,
)
from test_vectors.phy.check_pusch_tv import (
    tv_block1_dmrs,
    tv_block2_ls,
    tv_block3_ls_dd,
    tv_block4_cov,
    tv_block4a_embed_dmrs,
    tv_block4b_r_tilde,
    tv_block4c_ncov,
    tv_block4d_shrink,
    tv_block5_noise_rsrp_sinr,
    tv_block5a_noisevar,
    tv_block5b_rsrp,
    tv_block5c_sinr,
    tv_block6_equalizer,
    tv_block6a_equalizer_derive,
    tv_block6b_equalizer_apply,
    tv_block7_post_eq,
    tv_block8_soft_demapper,
    tv_block9_descramble,
    tv_block10_derate,
    tv_block11_ldpc_decode,
    tv_block12_cb_concat,
    tv_block13_crc_decode,
    tv_block14_dmrs_rssi,
    tv_pusch_rx,
)

DEFAULT_TV_NAME = "TVnr_7201_cuPhyMax.h5"


def test_block1_dmrs() -> None:
    """Test 1: DMRS generation."""
    tv_block1_dmrs(DEFAULT_TV_NAME, gen_dmrs_sym)


def test_block2_ls() -> None:
    """Test 2: Least Squares Channel Estimation."""
    tv_block2_ls(DEFAULT_TV_NAME, channel_est_ls)


def test_block3_ls_dd() -> None:
    """Test 3: Delay-domain Least Squares Channel Estimation."""
    tv_block3_ls_dd(DEFAULT_TV_NAME, channel_est_dd)


def test_block4_cov() -> None:
    """Test 4: Covariance Estimation (E2E and components)."""
    tv_block4_cov(DEFAULT_TV_NAME, estimate_covariance)


def test_block4a_embed_dmrs() -> None:
    """Test 4a: Embed DMRS."""
    tv_block4a_embed_dmrs(DEFAULT_TV_NAME, embed_dmrs_ul)


def test_block4b_r_tilde() -> None:
    """Test 4b: Estimate R_tilde."""
    tv_block4b_r_tilde(DEFAULT_TV_NAME, estimate_r_tilde)


def test_block4c_ncov() -> None:
    """Test 4c: Estimate Noise Covariance."""
    tv_block4c_ncov(DEFAULT_TV_NAME, estimate_noise_covariance)


def test_block4d_shrink() -> None:
    """Test 4d: Shrink Noise Covariance."""
    tv_block4d_shrink(DEFAULT_TV_NAME, ncov_shrinkage)


def test_block5a_noisevar() -> None:
    """Test 5a: Noise variance estimation."""
    tv_block5a_noisevar(DEFAULT_TV_NAME, noise_variance_db)


def test_block5b_rsrp() -> None:
    """Test 5b: RSRP estimation."""
    tv_block5b_rsrp(DEFAULT_TV_NAME, rsrp_db)


def test_block5c_sinr() -> None:
    """Test 5c: SINR estimation."""
    tv_block5c_sinr(DEFAULT_TV_NAME, sinr_db)


def test_block5_noise_rsrp_sinr() -> None:
    """Test 5: Noise variance, RSRP and SINR estimation."""
    tv_block5_noise_rsrp_sinr(DEFAULT_TV_NAME, noise_rsrp_sinr_db)


def test_block6a_equalizer_derive() -> None:
    """Test 6a: MIMO equalizer derive (implementation path)."""
    tv_block6a_equalizer_derive(DEFAULT_TV_NAME, derive_equalizer)


def test_block6b_equalizer_apply() -> None:
    """Test 6b: MIMO equalizer apply (implementation path)."""
    tv_block6b_equalizer_apply(DEFAULT_TV_NAME, apply_equalizer)


def test_block6_equalizer() -> None:
    """Test 6c: Equalizer block (derive + apply)."""
    tv_block6_equalizer(DEFAULT_TV_NAME, equalize)


def test_block7_post_eq() -> None:
    """Test 7: Post-EQ noise variance and SINR."""
    tv_block7_post_eq(DEFAULT_TV_NAME, post_eq_noisevar_sinr)


def test_block8_soft_demapper() -> None:
    """Test 8: Soft demapper."""
    tv_block8_soft_demapper(DEFAULT_TV_NAME, soft_demapper)


def test_block9_descramble() -> None:
    """Test 9: Descramble bits."""
    tv_block9_descramble(DEFAULT_TV_NAME, descramble_bits)


def test_block10_derate() -> None:
    """Test 10a: Derate."""
    tv_block10_derate(DEFAULT_TV_NAME, derate_match)


def test_block11_ldpc_decode() -> None:
    """Test 10b: LDPC."""
    tv_block11_ldpc_decode(DEFAULT_TV_NAME, ldpc_decode)


def test_block12_cb_concat() -> None:
    """Test 10c: CB concatenation."""
    tv_block12_cb_concat(DEFAULT_TV_NAME, codeblock_concatenation)


def test_block13_crc_decode() -> None:
    """Test 10d: CRC."""
    tv_block13_crc_decode(DEFAULT_TV_NAME, crc_decode)


def test_block14_dmrs_rssi() -> None:
    """Test 12: DMRS RSSI."""
    tv_block14_dmrs_rssi(DEFAULT_TV_NAME, measure_rssi)


def test_pusch_rx() -> None:
    """End-to-end test for unified reference PUSCH pipeline function.

    Uses the same TV as other block tests and compares a few key outputs.
    """
    tv_pusch_rx(DEFAULT_TV_NAME, pusch_rx)
