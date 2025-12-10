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
 * @file pusch_sample_tests.cpp
 * @brief Sample tests for PUSCH library documentation
 */

#include <any>
#include <memory>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include "ldpc/crc_decoder_module.hpp"
#include "ldpc/ldpc_decoder_module.hpp"
#include "ldpc/ldpc_derate_match_module.hpp"
#include "ldpc/ldpc_params.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/types.hpp"
#include "pusch/inner_rx_module.hpp"
#include "pusch/pusch_module_factories.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "ran_common.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

namespace pipeline = framework::pipeline;
using ran::common::PhyParams;
using ran::ldpc::CrcDecoderModule;
using ran::ldpc::LdpcDecoderModule;
using ran::ldpc::LdpcDerateMatchModule;
using ran::pusch::InnerRxModule;
using ran::pusch::PuschModuleFactory;
using ran::pusch::PuschPipeline;

TEST(PuschSampleTests, CreatePuschModuleFactory) {
    // example-begin create-factory-1
    // Create a module factory for PUSCH pipeline
    auto factory = std::make_unique<PuschModuleFactory>();

    // Check supported module types
    const bool supports_inner_rx = factory->supports_module_type("inner_rx_module");
    const bool supports_ldpc = factory->supports_module_type("ldpc_decoder_module");
    // example-end create-factory-1

    EXPECT_TRUE(supports_inner_rx);
    EXPECT_TRUE(supports_ldpc);
}

TEST(PuschSampleTests, CreateInnerRxModule) {
    // example-begin create-inner-rx-1
    // Configure physical layer parameters
    PhyParams phy_params{};
    phy_params.num_rx_ant = 4;
    phy_params.num_prb = 273;
    phy_params.bandwidth = 100;

    // Configure inner_rx module parameters
    const InnerRxModule::StaticParams params{
            .phy_params = phy_params, .execution_mode = pipeline::ExecutionMode::Stream};

    // Create the inner_rx module
    auto module = std::make_unique<InnerRxModule>("inner_rx_0", params);
    // example-end create-inner-rx-1

    // Verify module configuration
    EXPECT_EQ(module->get_type_id(), "inner_rx_module");
    EXPECT_EQ(module->get_instance_id(), "inner_rx_0");
}

TEST(PuschSampleTests, InspectModulePorts) {
    // example-begin inspect-ports-1
    // Create inner_rx module
    PhyParams phy_params{};
    phy_params.num_rx_ant = 4;
    phy_params.num_prb = 273;

    const InnerRxModule::StaticParams params{
            .phy_params = phy_params, .execution_mode = pipeline::ExecutionMode::Stream};

    auto module = std::make_unique<InnerRxModule>("inner_rx_0", params);

    // Inspect module ports
    const auto input_ports = module->get_input_port_names();
    const auto output_ports = module->get_output_port_names();
    // example-end inspect-ports-1

    EXPECT_FALSE(input_ports.empty());
    EXPECT_FALSE(output_ports.empty());
}

TEST(PuschSampleTests, CreatePipelineSpec) {
    // example-begin create-pipeline-spec-1
    // Create pipeline specification
    pipeline::PipelineSpec spec;
    spec.pipeline_name = "PuschPipeline";
    spec.execution_mode = pipeline::ExecutionMode::Stream;

    // Configure physical layer parameters
    PhyParams phy_params{};
    phy_params.num_rx_ant = 4;
    phy_params.num_prb = 273;

    // Add inner receiver (inner_rx) module to specification
    const InnerRxModule::StaticParams inner_rx_params{
            .phy_params = phy_params, .execution_mode = spec.execution_mode};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "inner_rx_module",
            .instance_id = "inner_rx_0",
            .init_params = std::any(inner_rx_params)});

    // Add outer receiver modules (derate match, LDPC decoder, CRC decoder)
    const LdpcDerateMatchModule::StaticParams derate_params{
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_rm_llrs_per_cb = ran::ldpc::MAX_NUM_RM_LLRS_PER_CB,
            .max_num_ue_grps = ran::common::MAX_NUM_UE_GRPS};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "ldpc_derate_match_module",
            .instance_id = "ldpc_derate_match_0",
            .init_params = std::any(derate_params)});

    const LdpcDecoderModule::StaticParams decoder_params{
            .clamp_value = ran::ldpc::LDPC_CLAMP_VALUE,
            .max_num_iterations = ran::ldpc::LDPC_MAX_ITERATIONS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .normalization_factor = ran::ldpc::LDPC_NORMALIZATION_FACTOR,
            .max_iterations_method = ran::ldpc::LdpcMaxIterationsMethod::Fixed,
            .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "ldpc_decoder_module",
            .instance_id = "ldpc_decoder_0",
            .init_params = std::any(decoder_params)});

    const CrcDecoderModule::StaticParams crc_params{
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "crc_decoder_module",
            .instance_id = "crc_decoder_0",
            .init_params = std::any(crc_params)});
    // example-end create-pipeline-spec-1

    // Verify specification
    EXPECT_EQ(spec.pipeline_name, "PuschPipeline");
    EXPECT_EQ(spec.modules.size(), 4);
}

TEST(PuschSampleTests, PipelineSetup) {
    // Setup pipeline (not shown in doc)
    auto factory = std::make_unique<PuschModuleFactory>();

    PhyParams phy_params{};
    phy_params.num_rx_ant = 4;
    phy_params.num_prb = 273;

    pipeline::PipelineSpec spec;
    spec.pipeline_name = "PuschPipeline";
    spec.execution_mode = pipeline::ExecutionMode::Stream;

    const InnerRxModule::StaticParams inner_rx_params{
            .phy_params = phy_params, .execution_mode = spec.execution_mode};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "inner_rx_module",
            .instance_id = "inner_rx_0",
            .init_params = std::any(inner_rx_params)});

    const LdpcDerateMatchModule::StaticParams derate_params{
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_rm_llrs_per_cb = ran::ldpc::MAX_NUM_RM_LLRS_PER_CB,
            .max_num_ue_grps = ran::common::MAX_NUM_UE_GRPS};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "ldpc_derate_match_module",
            .instance_id = "ldpc_derate_match_0",
            .init_params = std::any(derate_params)});

    const LdpcDecoderModule::StaticParams decoder_params{
            .clamp_value = ran::ldpc::LDPC_CLAMP_VALUE,
            .max_num_iterations = ran::ldpc::LDPC_MAX_ITERATIONS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .normalization_factor = ran::ldpc::LDPC_NORMALIZATION_FACTOR,
            .max_iterations_method = ran::ldpc::LdpcMaxIterationsMethod::Fixed,
            .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "ldpc_decoder_module",
            .instance_id = "ldpc_decoder_0",
            .init_params = std::any(decoder_params)});

    const CrcDecoderModule::StaticParams crc_params{
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "crc_decoder_module",
            .instance_id = "crc_decoder_0",
            .init_params = std::any(crc_params)});

    auto pipeline = std::make_unique<PuschPipeline>(
            "test_pipeline", gsl_lite::not_null<pipeline::IModuleFactory *>(factory.get()), spec);

    // example-begin pipeline-setup-1
    // Setup allocates memory and initializes all modules
    pipeline->setup();
    // example-end pipeline-setup-1

    EXPECT_EQ(pipeline->get_pipeline_id(), "test_pipeline");
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
