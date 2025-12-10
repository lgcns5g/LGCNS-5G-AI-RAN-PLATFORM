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
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <quill/LogMacros.h>

#include "ldpc/ldpc_log.hpp"
#include "ldpc/ldpc_params.hpp"
#include "log/rt_log_macros.hpp"

namespace ran::ldpc {

LdpcParams::LdpcParams(
        const std::uint32_t transport_block_size_in,
        const float code_rate_in,
        const std::optional<std::uint32_t> rate_matching_length_in,
        const std::optional<std::uint8_t> redundancy_version_in)
        : transport_block_size_(transport_block_size_in), code_rate_(code_rate_in),
          rate_matching_length_(rate_matching_length_in.value_or(0)),
          redundancy_version_(redundancy_version_in.value_or(0)) {

    // Validate input parameters
    if (transport_block_size_ == 0) {
        const std::string error_message = "Transport block size must be greater than 0";
        RT_LOGC_ERROR(LdpcComponent::LdpcParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    if (code_rate_ <= 0.0F || code_rate_ > 1.0F) {
        const std::string error_message = "Code rate must be between 0.0 and 1.0";
        RT_LOGC_ERROR(LdpcComponent::LdpcParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }

    if (redundancy_version_ > 3) {
        const std::string error_message = "Redundancy version must be 0-3";
        RT_LOGC_ERROR(LdpcComponent::LdpcParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }
    // Compute all derived parameters
    compute_derived_parameters();

    // Derive number of parity nodes
    if (rate_matching_length_in.has_value() && redundancy_version_in.has_value()) {
        compute_num_parity_nodes();
    }
}

void LdpcParams::compute_derived_parameters() {

    // Derive base graph
    base_graph_ = get_base_graph(transport_block_size_, code_rate_);

    // Derive number of code blocks
    num_code_blocks_ = get_num_code_blocks(transport_block_size_, base_graph_);

    // Derive number of info nodes
    num_info_nodes_ = get_num_info_nodes(transport_block_size_, base_graph_);

    // Derive K'
    k_prime_ = get_k_prime(transport_block_size_, num_code_blocks_);

    // Derive lifting size
    lifting_size_ = get_lifting_size(
            transport_block_size_, base_graph_, num_code_blocks_, num_info_nodes_, k_prime_);

    // Derive number of code block info bits, K in 38.212
    num_code_block_info_bits_ = get_num_code_block_info_bits(base_graph_, lifting_size_);

    // Derive number of filler bits
    num_filler_bits_ = num_code_block_info_bits_ - k_prime_;

    // Derive circular buffer size
    circular_buffer_size_ = base_graph_ == 1 ? lifting_size_ * UNPUNCTURED_VAR_NODES_BG1
                                             : lifting_size_ * UNPUNCTURED_VAR_NODES_BG2;
    circular_buffer_size_padded_ =
            (circular_buffer_size_ + 2 * lifting_size_ + PADDING_ALIGNMENT - 1) /
            PADDING_ALIGNMENT * PADDING_ALIGNMENT;
}

std::uint8_t LdpcParams::get_base_graph(const std::uint32_t tb_size, const float code_rate) {
    const std::uint8_t bg =
            ((tb_size <= TB_SIZE_BG_THRESHOLD1) || (code_rate <= CODE_RATE_BG_THRESHOLD_1) ||
             ((tb_size <= TB_SIZE_BG_THRESHOLD2) && (code_rate <= CODE_RATE_BG_THRESHOLD_2)))
                    ? 2
                    : 1;
    return bg;
}

std::uint8_t LdpcParams::get_tb_crc_size(const std::uint32_t tb_size) {
    return (tb_size > TB_SIZE_THRESHOLD) ? TB_CRC_SIZE_LARGE : TB_CRC_SIZE_SMALL;
}

std::uint32_t LdpcParams::get_tb_size_with_crc(const std::uint32_t tb_size) {
    return tb_size + get_tb_crc_size(tb_size);
}

std::uint32_t
LdpcParams::get_num_code_blocks(const std::uint32_t tb_size, const std::uint8_t base_graph) {

    if (base_graph != 1 && base_graph != 2) {
        const std::string error_message = "Base graph must be 1 or 2";
        RT_LOGC_ERROR(LdpcComponent::LdpcParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }
    std::uint32_t code_blocks{}; // C in 38.212

    const std::uint32_t max_code_block_size =
            (base_graph == 1) ? MAX_CODE_BLOCK_SIZE_BG1 : MAX_CODE_BLOCK_SIZE_BG2; // Kcb in 38.212
    const std::uint32_t tb_size_with_crc = get_tb_size_with_crc(tb_size);          // B in 38.212
    if (tb_size_with_crc <= max_code_block_size) {
        code_blocks = 1;
    } else {
        code_blocks = static_cast<std::uint32_t>(std::ceil(
                static_cast<double>(tb_size_with_crc) /
                static_cast<double>(max_code_block_size - CB_CRC_SIZE)));
    }
    return code_blocks;
}

std::uint32_t
LdpcParams::get_num_info_nodes(const std::uint32_t tb_size, const std::uint8_t base_graph) {
    const std::uint32_t tb_size_with_crc = get_tb_size_with_crc(tb_size);
    std::uint32_t kb{}; // TS 38.212 section 5.2.2
    if (base_graph == 1) {
        kb = INFO_NODES_BG1;
    } else if (base_graph == 2) {
        if (tb_size_with_crc > TB_SIZE_THRESHOLD_KB10) {
            kb = INFO_NODES_BG2_MAX;
        } else if (tb_size_with_crc > TB_SIZE_THRESHOLD_KB9) {
            kb = INFO_NODES_BG2_MEDIUM;
        } else if (tb_size_with_crc > TB_SIZE_THRESHOLD_KB8) {
            kb = INFO_NODES_BG2_SMALL;
        } else {
            kb = INFO_NODES_BG2_SMALLEST;
        }
    } else {
        const std::string error_message = "Base graph must be 1 or 2";
        RT_LOGC_ERROR(LdpcComponent::LdpcParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }
    return kb;
}

std::uint32_t
LdpcParams::get_k_prime(const std::uint32_t tb_size, const std::uint32_t num_code_blocks) {
    const std::uint32_t tb_size_with_crc = get_tb_size_with_crc(tb_size);

    // Derive B' and K' from 38.212 section 5.2.2
    std::uint32_t b_prime{};
    if (num_code_blocks == 1) {
        b_prime = tb_size_with_crc;
    } else {
        b_prime = tb_size_with_crc + num_code_blocks * CB_CRC_SIZE;
    }
    return b_prime / num_code_blocks;
}

std::uint32_t LdpcParams::get_lifting_size(
        const std::uint32_t tb_size,
        const std::uint8_t base_graph,
        std::optional<std::uint32_t> num_code_blocks,
        std::optional<std::uint32_t> num_info_nodes,
        std::optional<std::uint32_t> k_prime) {
    if (!num_code_blocks.has_value()) {
        num_code_blocks = get_num_code_blocks(tb_size, base_graph);
    }

    if (!num_info_nodes.has_value()) {
        num_info_nodes = get_num_info_nodes(tb_size, base_graph);
    }

    if (!k_prime.has_value()) {
        k_prime = get_k_prime(tb_size, num_code_blocks.value());
    }

    static constexpr auto Z = std::to_array<std::uint32_t>(
            {2,   4,  8,  16, 32,  64,  128, 256, 3,  6,  12,  24,  48, 96, 192, 384, 5,
             10,  20, 40, 80, 160, 320, 7,   14,  28, 56, 112, 224, 9,  18, 36,  72,  144,
             288, 11, 22, 44, 88,  176, 352, 13,  26, 52, 104, 208, 15, 30, 60,  120, 240});

    // Find smallest Zc such that Zc * Kb >= k_prime:
    std::uint32_t zc = MAX_LIFTING_SIZE; // max possible
    for (const auto &z_val : Z) {
        const std::uint32_t temp = z_val * num_info_nodes.value();
        if ((temp >= k_prime.value()) && (z_val < zc)) {
            zc = z_val;
        }
    }
    return zc;
}

std::uint32_t LdpcParams::get_num_code_block_info_bits(
        const std::uint8_t base_graph, const std::uint32_t lifting_size) {
    if (base_graph == 1) {
        return INFO_NODES_BG1 * lifting_size;
    } else if (base_graph == 2) {
        return INFO_NODES_BG2_MAX * lifting_size;
    } else {
        const std::string error_message = "Base graph must be 1 or 2";
        RT_LOGC_ERROR(LdpcComponent::LdpcParams, "{}", error_message);
        throw std::invalid_argument(error_message);
    }
}

void LdpcParams::compute_num_parity_nodes() {

    // RV factors lookup tables (RV is already validated in constructor to be 0-3)
    static constexpr std::array<std::uint32_t, 4> RV_FACTORS_BG1 = {
            0, RV1_START_POS_FACTOR_BG1, RV2_START_POS_FACTOR_BG1, RV3_START_POS_FACTOR_BG1};
    static constexpr std::array<std::uint32_t, 4> RV_FACTORS_BG2 = {
            0, RV1_START_POS_FACTOR_BG2, RV2_START_POS_FACTOR_BG2, RV3_START_POS_FACTOR_BG2};

    // Calculate k0 based on redundancy version (rv) and base graph (bg)
    // redundancy version is already validated in constructor to be 0-3
    const auto &rv_factors = (base_graph_ == 1) ? RV_FACTORS_BG1 : RV_FACTORS_BG2;
    const auto unpunctured_var_nodes =
            (base_graph_ == 1) ? UNPUNCTURED_VAR_NODES_BG1 : UNPUNCTURED_VAR_NODES_BG2;
    const std::uint32_t k0 = (rv_factors.at(redundancy_version_) * circular_buffer_size_ /
                              (unpunctured_var_nodes * lifting_size_)) *
                             lifting_size_;

    const std::uint32_t kd = num_code_block_info_bits_ - num_filler_bits_ - 2 * lifting_size_;
    const std::uint32_t max_parity_nodes =
            base_graph_ == 1 ? MAX_PARITY_NODES_BG1 : MAX_PARITY_NODES_BG2;
    const std::uint32_t circ_buffer_for_parity = std::min<std::uint32_t>(
            rate_matching_length_ / num_code_blocks_ + k0, circular_buffer_size_);
    if (lbrm_enabled_ == 0U) {
        num_parity_nodes_ = ((circ_buffer_for_parity - kd + lifting_size_ - 1) / lifting_size_);
    } else {
        num_parity_nodes_ = ((circ_buffer_for_parity - kd) / lifting_size_);
    }

    num_parity_nodes_ = std::max<std::uint32_t>(
            MIN_PARITY_NODES, std::min<std::uint32_t>(max_parity_nodes, num_parity_nodes_.value()));
}

std::uint32_t LdpcParams::num_parity_nodes() const { return num_parity_nodes_.value_or(0); }

} // namespace ran::ldpc
