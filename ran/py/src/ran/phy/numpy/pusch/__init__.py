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

"""PUSCH operations package."""

from ran.phy.numpy.utils import awgn

from .channel_estimation import (
    channel_est_dd,
    channel_est_ls,
)
from .codeblock_concatenation import codeblock_concatenation
from .noise_estimation import (
    estimate_covariance,
    estimate_noise_covariance,
    estimate_r_tilde,
    ncov_shrinkage,
)
from .crc_decoder import crc_decode
from .derate_matcher import derate_match
from .descrambler import descramble_bits
from .dmrs_utils import embed_dmrs_ul, extract_raw_dmrs_type_1
from .equalizer import (
    apply_equalizer,  # intermediate block
    derive_equalizer,  # intermediate block
    equalize,
)
from .gen_dmrs_sym import gen_dmrs_sym
from .ldpc_decoder import ldpc_decode
from .measure_rssi import measure_rssi
from .noisevar_rsrp_sinr import (
    noise_rsrp_sinr_db,
    noise_variance_db,  # intermediate block
    rsrp_db,  # intermediate block
    sinr_db,  # intermediate block
)
from .outer_receiver import OuterRxOutputs, OuterRxParams, pusch_outer_rx
from .post_eq_noisevar_sinr import post_eq_noisevar_sinr
from .pusch_rx import pusch_rx, pusch_inner_rx
from .soft_demapper import soft_demapper

__all__ = [
    "awgn",
    "apply_equalizer",
    "channel_est_dd",
    "channel_est_ls",
    "codeblock_concatenation",
    "crc_decode",
    "derate_match",
    "derive_equalizer",
    "descramble_bits",
    "embed_dmrs_ul",
    "equalize",
    "estimate_covariance",
    "estimate_noise_covariance",
    "estimate_r_tilde",
    "extract_raw_dmrs_type_1",
    "gen_dmrs_sym",
    "ldpc_decode",
    "measure_rssi",
    "ncov_shrinkage",
    "noise_rsrp_sinr_db",
    "noise_variance_db",
    "post_eq_noisevar_sinr",
    "pusch_rx",
    "pusch_inner_rx",
    "rsrp_db",
    "sinr_db",
    "soft_demapper",
    "OuterRxOutputs",
    "OuterRxParams",
    "pusch_outer_rx",
]
