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

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ldpc/ldpc_params.hpp"

namespace {

using ran::ldpc::LdpcParams;

// Allow magic numbers in this file to check output against.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Test constructor with basic valid parameters and verify field population
 */
TEST(LdpcParamsTest, ConstructorBasicFieldPopulation) {
    const std::uint32_t tb_size = 1000;
    const float code_rate = 0.5F;
    const std::uint32_t rm_length = 2000;
    const std::uint8_t rv = 0;

    const LdpcParams params(tb_size, code_rate, rm_length, rv);

    // Check input parameters are stored correctly
    EXPECT_EQ(params.transport_block_size(), tb_size);
    EXPECT_FLOAT_EQ(params.code_rate(), code_rate);
    EXPECT_EQ(params.rate_matching_length(), rm_length);
    EXPECT_EQ(params.redundancy_version(), rv);
    EXPECT_EQ(params.lbrm_enabled(), 0);

    // Check that derived parameters are populated (non-zero)
    EXPECT_GT(params.base_graph(), 0);
    EXPECT_LE(params.base_graph(), 2);
    EXPECT_GT(params.num_code_blocks(), 0);
    EXPECT_GT(params.num_info_nodes(), 0);
    EXPECT_GT(params.k_prime(), 0);
    EXPECT_GT(params.lifting_size(), 0);
    EXPECT_GT(params.num_code_block_info_bits(), 0);
    EXPECT_GT(params.circular_buffer_size(), 0);
    EXPECT_GT(params.circular_buffer_size_padded(), 0);

    // Check that parity nodes are computed when optional parameters are provided
    EXPECT_GT(params.num_parity_nodes(), 0);
}

/**
 * Test constructor with optional parameters not provided
 */
TEST(LdpcParamsTest, ConstructorOptionalParametersNotProvided) {
    const std::uint32_t tb_size = 500;
    const float code_rate = 0.3F;

    const LdpcParams params(tb_size, code_rate, std::nullopt, std::nullopt);

    // Check input parameters
    EXPECT_EQ(params.transport_block_size(), tb_size);
    EXPECT_FLOAT_EQ(params.code_rate(), code_rate);
    EXPECT_EQ(params.rate_matching_length(), 0); // Default value
    EXPECT_EQ(params.redundancy_version(), 0);   // Default value

    // Check that basic derived parameters are still computed
    EXPECT_GT(params.base_graph(), 0);
    EXPECT_LE(params.base_graph(), 2);
    EXPECT_GT(params.num_code_blocks(), 0);
    EXPECT_GT(params.num_info_nodes(), 0);
    EXPECT_GT(params.k_prime(), 0);
    EXPECT_GT(params.lifting_size(), 0);
    EXPECT_GT(params.num_code_block_info_bits(), 0);
}

/**
 * Test input validation - zero transport block size
 */
TEST(LdpcParamsTest, ValidationZeroTransportBlockSize) {
    EXPECT_THROW(
            { const LdpcParams params(0, 0.5F, std::nullopt, std::nullopt); },
            std::invalid_argument);
}

/**
 * Test input validation - invalid code rate (too low)
 */
TEST(LdpcParamsTest, ValidationInvalidCodeRateTooLow) {
    EXPECT_THROW(
            { const LdpcParams params(1000, 0.0F, std::nullopt, std::nullopt); },
            std::invalid_argument);
}

/**
 * Test input validation - invalid code rate (too high)
 */
TEST(LdpcParamsTest, ValidationInvalidCodeRateTooHigh) {
    EXPECT_THROW(
            { const LdpcParams params(1000, 1.5F, std::nullopt, std::nullopt); },
            std::invalid_argument);
}

/**
 * Test input validation - invalid redundancy version
 */
TEST(LdpcParamsTest, ValidationInvalidRedundancyVersion) {
    EXPECT_THROW(
            {
                const LdpcParams params(1000, 0.5F, 2000, 4); // RV must be 0-3
            },
            std::invalid_argument);
}

/**
 * Test base graph selection for different scenarios
 */
TEST(LdpcParamsTest, BaseGraphSelection) {
    // Small transport block should use base graph 2
    {
        const LdpcParams params(200, 0.5F, std::nullopt, std::nullopt);
        EXPECT_EQ(params.base_graph(), 2);
    }

    // Large transport block with high code rate should use base graph 1
    {
        const LdpcParams params(5000, 0.8F, std::nullopt, std::nullopt);
        EXPECT_EQ(params.base_graph(), 1);
    }

    // Low code rate should use base graph 2
    {
        const LdpcParams params(1000, 0.2F, std::nullopt, std::nullopt);
        EXPECT_EQ(params.base_graph(), 2);
    }
}

/**
 * Test different redundancy versions with field population
 */
TEST(LdpcParamsTest, RedundancyVersions) {
    const std::uint32_t tb_size = 1000;
    const float code_rate = 0.5F;
    const std::uint32_t rm_length = 2000;

    for (std::uint8_t rv = 0; rv <= 3; ++rv) {
        const LdpcParams params(tb_size, code_rate, rm_length, rv);

        EXPECT_EQ(params.redundancy_version(), rv);
        EXPECT_GT(params.num_parity_nodes(), 0);

        // Ensure parity nodes are within valid bounds
        const std::uint32_t max_parity = (params.base_graph() == 1)
                                                 ? LdpcParams::MAX_PARITY_NODES_BG1
                                                 : LdpcParams::MAX_PARITY_NODES_BG2;
        EXPECT_GE(params.num_parity_nodes(), LdpcParams::MIN_PARITY_NODES);
        EXPECT_LE(params.num_parity_nodes(), max_parity);
    }
}

/**
 * Test static method: get_base_graph
 */
TEST(LdpcParamsTest, StaticMethodGetBaseGraph) {
    // Small TB should return BG2
    EXPECT_EQ(LdpcParams::get_base_graph(200, 0.5F), 2);

    // Large TB with high code rate should return BG1
    EXPECT_EQ(LdpcParams::get_base_graph(5000, 0.8F), 1);

    // Low code rate should return BG2
    EXPECT_EQ(LdpcParams::get_base_graph(1000, 0.2F), 2);
}

/**
 * Test static method: get_tb_crc_size
 */
TEST(LdpcParamsTest, StaticMethodGetTbCrcSize) {
    // Small TB should use 16-bit CRC
    EXPECT_EQ(LdpcParams::get_tb_crc_size(1000), LdpcParams::TB_CRC_SIZE_SMALL);

    // Large TB should use 24-bit CRC
    EXPECT_EQ(LdpcParams::get_tb_crc_size(5000), LdpcParams::TB_CRC_SIZE_LARGE);
}

/**
 * Test static method: get_num_code_blocks
 */
TEST(LdpcParamsTest, StaticMethodGetNumCodeBlocks) {
    // Valid base graphs
    EXPECT_GT(LdpcParams::get_num_code_blocks(1000, 1), 0);
    EXPECT_GT(LdpcParams::get_num_code_blocks(1000, 2), 0);

    // Invalid base graph should throw
    EXPECT_THROW({ LdpcParams::get_num_code_blocks(1000, 3); }, std::invalid_argument);
}

/**
 * Test static method: get_num_info_nodes
 */
TEST(LdpcParamsTest, StaticMethodGetNumInfoNodes) {
    // Base graph 1 should return fixed value
    EXPECT_EQ(LdpcParams::get_num_info_nodes(1000, 1), LdpcParams::INFO_NODES_BG1);

    // Base graph 2 should return variable value based on TB size
    const std::uint32_t info_nodes_bg2{LdpcParams::get_num_info_nodes(1000, 2)};
    EXPECT_GT(info_nodes_bg2, 0);
    EXPECT_LE(info_nodes_bg2, LdpcParams::INFO_NODES_BG2_MAX);

    // Invalid base graph should throw
    EXPECT_THROW({ LdpcParams::get_num_info_nodes(1000, 0); }, std::invalid_argument);
}

/**
 * Test static method: get_lifting_size
 */
TEST(LdpcParamsTest, StaticMethodGetLiftingSize) {
    const std::uint32_t lifting_size{
            LdpcParams::get_lifting_size(1000, 1, std::nullopt, std::nullopt, std::nullopt)};
    EXPECT_GT(lifting_size, 0);
    EXPECT_LE(lifting_size, LdpcParams::MAX_LIFTING_SIZE);

    // Invalid base graph should throw
    EXPECT_THROW(
            { LdpcParams::get_lifting_size(1000, 3, std::nullopt, std::nullopt, std::nullopt); },
            std::invalid_argument);
}

/**
 * Test static method: get_num_code_block_info_bits
 */
TEST(LdpcParamsTest, StaticMethodGetNumCodeBlockInfoBits) {
    const std::uint32_t lifting_size = 32;

    // Base graph 1
    EXPECT_EQ(
            LdpcParams::get_num_code_block_info_bits(1, lifting_size),
            LdpcParams::INFO_NODES_BG1 * lifting_size);

    // Base graph 2
    EXPECT_EQ(
            LdpcParams::get_num_code_block_info_bits(2, lifting_size),
            LdpcParams::INFO_NODES_BG2_MAX * lifting_size);

    // Invalid base graph should throw
    EXPECT_THROW(
            { LdpcParams::get_num_code_block_info_bits(3, lifting_size); }, std::invalid_argument);
}

/**
 * Test filler bits calculation
 */
TEST(LdpcParamsTest, FillerBitsCalculation) {
    const LdpcParams params(1000, 0.5F, std::nullopt, std::nullopt);

    // Filler bits should be the difference between code block info bits and
    // k_prime
    EXPECT_EQ(params.num_filler_bits(), params.num_code_block_info_bits() - params.k_prime());
}

/**
 * Test circular buffer size calculation
 */
TEST(LdpcParamsTest, CircularBufferSizeCalculation) {
    const LdpcParams params(1000, 0.5F, std::nullopt, std::nullopt);

    // Circular buffer size should be calculated based on base graph
    const std::uint32_t expected_cb_size{
            (params.base_graph() == 1)
                    ? params.lifting_size() * LdpcParams::UNPUNCTURED_VAR_NODES_BG1
                    : params.lifting_size() * LdpcParams::UNPUNCTURED_VAR_NODES_BG2};

    EXPECT_EQ(params.circular_buffer_size(), expected_cb_size);

    // Padded circular buffer size should be greater than or equal to regular size
    EXPECT_GE(params.circular_buffer_size_padded(), params.circular_buffer_size());
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
