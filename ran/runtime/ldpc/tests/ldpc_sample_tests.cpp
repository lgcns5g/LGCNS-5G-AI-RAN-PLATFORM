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

/**
 * @file ldpc_sample_tests.cpp
 * @brief Sample tests for LDPC library documentation
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "ldpc/crc_decoder_module.hpp"
#include "ldpc/ldpc_decoder_module.hpp"
#include "ldpc/ldpc_derate_match_module.hpp"
#include "ldpc/ldpc_params.hpp"
#include "pipeline/types.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

TEST(LdpcSampleTests, BasicLdpcDecoderSetup) {
    // example-begin decoder-setup-1
    // Configure LDPC decoder with static parameters
    const ran::ldpc::LdpcDecoderModule::StaticParams decoder_params{
            .clamp_value = 20.0F,
            .max_num_iterations = 20,
            .max_num_cbs_per_tb = 152,
            .max_num_tbs = 1,
            .normalization_factor = 0.125F,
            .max_iterations_method = ran::ldpc::LdpcMaxIterationsMethod::Fixed,
            .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

    // Create decoder module instance
    const auto decoder =
            std::make_unique<ran::ldpc::LdpcDecoderModule>("ldpc_decoder", decoder_params);
    // example-end decoder-setup-1

    // Verify module identity
    EXPECT_EQ(decoder->get_type_id(), "ldpc_decoder_module");
    EXPECT_EQ(decoder->get_instance_id(), "ldpc_decoder");
}

TEST(LdpcSampleTests, BasicDerateMatchSetup) {
    // example-begin derate-match-setup-1
    // Configure LDPC derate matching module
    const ran::ldpc::LdpcDerateMatchModule::StaticParams derate_params{
            .enable_scrambling = true,
            .max_num_tbs = 1,
            .max_num_cbs_per_tb = 152,
            .max_num_rm_llrs_per_cb = 27000,
            .max_num_ue_grps = 1};

    // Create derate matching module
    const auto derate_match =
            std::make_unique<ran::ldpc::LdpcDerateMatchModule>("ldpc_derate_match", derate_params);
    // example-end derate-match-setup-1

    // Verify module configuration
    EXPECT_EQ(derate_match->get_type_id(), "ldpc_derate_match_module");
    EXPECT_EQ(derate_match->get_instance_id(), "ldpc_derate_match");
}

TEST(LdpcSampleTests, BasicCrcDecoderSetup) {
    // example-begin crc-decoder-setup-1
    // Configure CRC decoder module
    const ran::ldpc::CrcDecoderModule::StaticParams crc_params{
            .reverse_bytes = true, .max_num_cbs_per_tb = 152, .max_num_tbs = 1};

    // Create CRC decoder module
    const auto crc_decoder =
            std::make_unique<ran::ldpc::CrcDecoderModule>("crc_decoder", crc_params);
    // example-end crc-decoder-setup-1

    // Verify module identity
    EXPECT_EQ(crc_decoder->get_type_id(), "crc_decoder_module");
    EXPECT_EQ(crc_decoder->get_instance_id(), "crc_decoder");
}

TEST(LdpcSampleTests, ModulePortInspection) {
    // example-begin module-ports-1
    const ran::ldpc::LdpcDecoderModule::StaticParams params{
            .clamp_value = 20.0F,
            .max_num_iterations = 20,
            .max_num_cbs_per_tb = 152,
            .max_num_tbs = 1,
            .normalization_factor = 0.125F,
            .max_iterations_method = ran::ldpc::LdpcMaxIterationsMethod::Fixed,
            .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

    const auto decoder = std::make_unique<ran::ldpc::LdpcDecoderModule>("decoder", params);

    // Query module input and output ports
    const auto input_ports = decoder->get_input_port_names();
    const auto output_ports = decoder->get_output_port_names();
    // example-end module-ports-1

    // Decoder has one input port for LLRs
    EXPECT_EQ(input_ports.size(), 1);
    EXPECT_EQ(input_ports[0], "llrs");

    // Decoder has one output port for decoded bits
    EXPECT_EQ(output_ports.size(), 1);
    EXPECT_EQ(output_ports[0], "decoded_bits");
}

TEST(LdpcSampleTests, MemoryRequirements) {
    // example-begin memory-requirements-1
    const ran::ldpc::LdpcDerateMatchModule::StaticParams params{
            .enable_scrambling = true,
            .max_num_tbs = 1,
            .max_num_cbs_per_tb = 152,
            .max_num_rm_llrs_per_cb = 27000,
            .max_num_ue_grps = 1};

    const auto module = std::make_unique<ran::ldpc::LdpcDerateMatchModule>("derate_match", params);

    // Query memory requirements before allocation
    const auto requirements = module->get_requirements();
    // example-end memory-requirements-1

    // Requirements include device tensor memory
    EXPECT_GT(requirements.device_tensor_bytes, 0U);
}

TEST(LdpcSampleTests, DecoderStaticParams) {
    // example-begin decoder-params-1
    // Configure decoder with custom parameters
    const ran::ldpc::LdpcDecoderModule::StaticParams custom_params{
            .clamp_value = 15.0F,          // Custom LLR clamping value
            .max_num_iterations = 10,      // Reduce max iterations for performance
            .max_num_cbs_per_tb = 100,     // Maximum code blocks per transport block
            .max_num_tbs = 4,              // Support multiple transport blocks
            .normalization_factor = 0.15F, // Custom normalization
            .max_iterations_method = ran::ldpc::LdpcMaxIterationsMethod::Fixed,
            .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

    const auto decoder =
            std::make_unique<ran::ldpc::LdpcDecoderModule>("custom_decoder", custom_params);
    // example-end decoder-params-1

    EXPECT_EQ(decoder->get_instance_id(), "custom_decoder");
}

TEST(LdpcSampleTests, CrcModulePorts) {
    // example-begin crc-ports-1
    const ran::ldpc::CrcDecoderModule::StaticParams params{
            .reverse_bytes = true, .max_num_cbs_per_tb = 152, .max_num_tbs = 1};

    const auto crc = std::make_unique<ran::ldpc::CrcDecoderModule>("crc", params);

    // CRC decoder has one input port for decoded bits
    const auto inputs = crc->get_input_port_names();

    // CRC decoder has three output ports
    const auto outputs = crc->get_output_port_names();
    // example-end crc-ports-1

    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(inputs[0], "decoded_bits");
    EXPECT_EQ(outputs.size(), 3);
    EXPECT_EQ(outputs[0], "cb_crcs");
    EXPECT_EQ(outputs[1], "tb_crcs");
    EXPECT_EQ(outputs[2], "tb_payloads");
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
