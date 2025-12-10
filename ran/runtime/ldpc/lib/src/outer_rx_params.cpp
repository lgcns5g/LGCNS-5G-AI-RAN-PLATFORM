/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuphy.h>
#include <driver_types.h>
#include <quill/LogMacros.h>
#include <range/v3/iterator/basic_iterator.hpp>
#include <range/v3/iterator/unreachable_sentinel.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/facade.hpp>
#include <range/v3/view/view.hpp>
#include <range/v3/view/zip.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda_runtime.h>

#include "ldpc/derate_match_params.hpp"
#include "ldpc/ldpc_log.hpp"
#include "ldpc/ldpc_params.hpp"
#include "ldpc/outer_rx_params.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/buffer.hpp"
#include "memory/device_allocators.hpp"
#include "ran_common.hpp"
#include "utils/error_macros.hpp"

namespace rv = ranges::views;

namespace ran::ldpc {

SingleTbPuschOuterRxParams::SingleTbPuschOuterRxParams(
        const LdpcParams &ldpc_params, const std::optional<DerateMatchParams> &de_rm_params)
        : de_rm_params_(de_rm_params.value_or(DerateMatchParams{})), ldpc_params_(ldpc_params) {}

const DerateMatchParams &SingleTbPuschOuterRxParams::de_rm_params() const { return de_rm_params_; }

void SingleTbPuschOuterRxParams::to_per_tb_params(PerTbParams &tb_params) const {
    const LdpcParams &ldpc = this->ldpc_params();

    tb_params.ndi = static_cast<std::uint32_t>(this->de_rm_params().ndi);
    tb_params.rv = static_cast<std::uint32_t>(ldpc.redundancy_version());
    tb_params.Qm = static_cast<std::uint32_t>(this->de_rm_params().mod_order);
    tb_params.bg = static_cast<std::uint32_t>(ldpc.base_graph());
    tb_params.Nl = this->de_rm_params().num_layers;
    tb_params.num_CBs = ldpc.num_code_blocks();
    tb_params.Zc = ldpc.lifting_size();
    tb_params.N = ldpc.circular_buffer_size();
    tb_params.Ncb = ldpc.circular_buffer_size();
    tb_params.Ncb_padded = ldpc.circular_buffer_size_padded();
    tb_params.G = ldpc.rate_matching_length();
    tb_params.K = ldpc.num_code_block_info_bits();
    tb_params.F = ldpc.num_filler_bits();
    tb_params.cinit = this->de_rm_params().scrambling_init;
    static constexpr std::uint32_t BITS_PER_BYTE = 8;
    tb_params.nDataBytes = ldpc.transport_block_size() / BITS_PER_BYTE;
    tb_params.nZpBitsPerCb = tb_params.K + ldpc.num_parity_nodes() * ldpc.lifting_size();

    tb_params.firstCodeBlockIndex = 0;
    tb_params.encodedSize = tb_params.G;

    if (this->de_rm_params().num_layers > ran::common::MAX_NUM_LAYERS_PER_UE_GRP) {
        const std::string error_message = std::format(
                "Number of layers must be less than or equal to {}",
                ran::common::MAX_NUM_LAYERS_PER_UE_GRP);
        RT_LOGC_ERROR(ran::ldpc::LdpcComponent::OuterRxParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    for (std::size_t j = 0; j < this->de_rm_params().num_layers; ++j) {
        gsl_lite::at(tb_params.layer_map_array, j) = this->de_rm_params().layer_map[j];
    }

    tb_params.userGroupIndex = this->de_rm_params().user_group_idx;
    tb_params.nBBULayers = this->de_rm_params().num_ue_grp_layers;
    tb_params.startLLR = 0; // startLLR is not used in current implementation

    // Additional SingleTbPuschOuterRxParams fields
    tb_params.enableTfPrcd = static_cast<std::uint8_t>(this->enable_tf_prcd());
    tb_params.nDmrsCdmGrpsNoData = this->de_rm_params().n_dmrs_cdm_grps_no_data;

    // Leave all UCI, mapping, and runtime pointers zero-initialized.
    tb_params.isDataPresent = 1; // Only PUSCH for now.
    tb_params.tbSize = ldpc.transport_block_size();
    tb_params.debug_d_derateCbsIndices = nullptr;
}

PuschOuterRxParams::PuschOuterRxParams(
        const std::vector<SingleTbPuschOuterRxParams> &pusch_outer_rx_params,
        const std::vector<std::uint16_t> &sch_user_idxs)
        : pusch_outer_rx_params_(pusch_outer_rx_params), sch_user_idxs_(sch_user_idxs) {

    if (pusch_outer_rx_params_.size() != sch_user_idxs_.size()) {
        const std::string error_message = std::format(
                "Transport block parameters and user indices must have the "
                "same size: {} != {}",
                pusch_outer_rx_params_.size(),
                sch_user_idxs_.size());
        RT_LOGC_ERROR(ran::ldpc::LdpcComponent::OuterRxParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    // Validate layer mapping
    const auto invalid_params = std::find_if(
            pusch_outer_rx_params_.begin(),
            pusch_outer_rx_params_.end(),
            [](const auto &tb_params) {
                return tb_params.de_rm_params().layer_map.size() !=
                       tb_params.de_rm_params().num_layers;
            });

    if (invalid_params != pusch_outer_rx_params_.end()) {
        const std::string error_message = std::format(
                "Layer map size and number of layers must be the same: {} != {}",
                invalid_params->de_rm_params().layer_map.size(),
                invalid_params->de_rm_params().num_layers);
        RT_LOGC_ERROR(ran::ldpc::LdpcComponent::OuterRxParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    tb_params_cpu_ = framework::memory::Buffer<PerTbParams, framework::memory::PinnedAlloc>(
            pusch_outer_rx_params_.size());
    tb_params_gpu_ = framework::memory::Buffer<PerTbParams, framework::memory::DeviceAlloc>(
            pusch_outer_rx_params_.size());
    to_per_tb_params();
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void PuschOuterRxParams::copy_tb_params_to_gpu(cudaStream_t stream) {
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
            tb_params_gpu_.addr(),
            tb_params_cpu_.addr(),
            tb_params_cpu_.size() * sizeof(PerTbParams),
            cudaMemcpyHostToDevice,
            stream));
}

void PuschOuterRxParams::to_per_tb_params() {
    for (auto &&[i, tb_params] : pusch_outer_rx_params_ | rv::enumerate) {
        tb_params.to_per_tb_params(tb_params_cpu_[i]);
    }
}

} // namespace ran::ldpc
