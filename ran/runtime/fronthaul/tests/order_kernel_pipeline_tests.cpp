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

// Basic instantiation and API validation tests for OrderKernel pipeline
// These tests exercise the factory pattern, module/pipeline creation, and API contracts
// WITHOUT requiring DOCA hardware or actual kernel execution

#include <any>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gdrapi.h>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include "fronthaul/order_kernel_factories.hpp"
#include "fronthaul/order_kernel_module.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/gdrcopy_buffer.hpp"
#include "net/doca_types.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace ran::fronthaul::tests {

// Namespace aliases for cleaner code
namespace pipeline = framework::pipeline;
namespace memory = framework::memory;
namespace tensor = framework::tensor;
namespace net = framework::net;

namespace {

/**
 * Dummy DOCA RX queue parameters for testing
 */
const net::DocaRxQParams *create_dummy_doca_params() {
    // Static dummy object - safe to take address, but fields are uninitialized
    // This is only for API contract testing, not actual DOCA operations
    static const net::DocaRxQParams dummy_params{};
    return &dummy_params;
}

} // namespace

/**
 * Test fixture for OrderKernel pipeline tests
 *
 * Provides:
 * - Factory instantiation
 * - GDRCopy handle management
 * - CUDA stream management
 */
class OrderKernelPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GDRCopy handle (throws if driver unavailable)
        gdr_handle_ = memory::make_unique_gdr_handle();

        // Create DOCA params for testing
        doca_params_ = create_dummy_doca_params();

        // Create factories
        module_factory_ = std::make_unique<OrderKernelModuleFactory>();
        pipeline_factory_ = std::make_unique<OrderKernelPipelineFactory>();

        // Set DOCA params in pipeline factory
        pipeline_factory_->set_doca_params(doca_params_);

        RT_LOG_DEBUG("OrderKernelPipelineTest: Factories initialized");
    }

    void TearDown() override {
        // Destroy factories
        pipeline_factory_.reset();
        module_factory_.reset();

        // gdr_handle_ automatically cleaned up by unique_ptr destructor

        RT_LOG_DEBUG("OrderKernelPipelineTest: Cleanup complete");
    }

    memory::UniqueGdrHandle gdr_handle_{nullptr};              //!< GDRCopy handle (RAII-managed)
    const net::DocaRxQParams *doca_params_{nullptr};           //!< Dummy DOCA params
    std::unique_ptr<OrderKernelModuleFactory> module_factory_; //!< Module factory
    std::unique_ptr<OrderKernelPipelineFactory> pipeline_factory_; //!< Pipeline factory
};

// ============================================================================
// Factory Tests
// ============================================================================

/** Test factory instantiation */
TEST_F(OrderKernelPipelineTest, FactoryInstantiation) {
    // Factories created in SetUp()
    ASSERT_NE(module_factory_, nullptr);
    ASSERT_NE(pipeline_factory_, nullptr);
}

/** Test module factory supported types */
TEST_F(OrderKernelPipelineTest, ModuleFactorySupportedTypes) {
    // Check supported type
    EXPECT_TRUE(module_factory_->supports_module_type("order_kernel_module"));

    // Check unsupported types
    EXPECT_FALSE(module_factory_->supports_module_type("unknown_module"));
    EXPECT_FALSE(module_factory_->supports_module_type(""));
}

/** Test pipeline factory supported types */
TEST_F(OrderKernelPipelineTest, PipelineFactorySupportedTypes) {
    // Check supported type
    EXPECT_TRUE(pipeline_factory_->is_pipeline_type_supported("order_kernel_pipeline"));

    // Check unsupported types
    EXPECT_FALSE(pipeline_factory_->is_pipeline_type_supported("unknown_pipeline"));
    EXPECT_FALSE(pipeline_factory_->is_pipeline_type_supported(""));

    // Check get_supported_pipeline_types
    const auto supported_types = pipeline_factory_->get_supported_pipeline_types();
    ASSERT_EQ(supported_types.size(), 1);
    EXPECT_EQ(supported_types[0], "order_kernel_pipeline");
}

// ============================================================================
// Module Creation Tests
// ============================================================================

/** Create module in stream execution mode */
TEST_F(OrderKernelPipelineTest, ModuleCreation_StreamMode) {

    // Create module via factory
    OrderKernelModule::StaticParams params;
    params.execution_mode = pipeline::ExecutionMode::Stream;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;

    const std::any params_any = params;
    auto module =
            module_factory_->create_module("order_kernel_module", "test_module_stream", params_any);

    ASSERT_NE(module, nullptr);
    EXPECT_EQ(module->get_type_id(), "order_kernel_module");
    EXPECT_EQ(module->get_instance_id(), "test_module_stream");
}

/** Create module in graph execution mode */
TEST_F(OrderKernelPipelineTest, ModuleCreation_GraphMode) {

    // Create module via factory
    OrderKernelModule::StaticParams params;
    params.execution_mode = pipeline::ExecutionMode::Graph;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;

    const std::any params_any = params;
    auto module =
            module_factory_->create_module("order_kernel_module", "test_module_graph", params_any);

    ASSERT_NE(module, nullptr);
    EXPECT_EQ(module->get_type_id(), "order_kernel_module");
    EXPECT_EQ(module->get_instance_id(), "test_module_graph");
}

/** Test module creation with invalid type */
TEST_F(OrderKernelPipelineTest, ModuleCreation_InvalidType) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;

    // Should throw on unsupported type
    EXPECT_THROW(
            std::ignore = module_factory_->create_module("invalid_type", "test_module", params_any),
            std::invalid_argument);
}

/** Test module creation with invalid params type */
TEST_F(OrderKernelPipelineTest, ModuleCreation_InvalidParams) {

    // Wrong parameter type
    const std::any wrong_params = std::string("not_a_struct");

    EXPECT_THROW(
            std::ignore = module_factory_->create_module(
                    "order_kernel_module", "test_module", wrong_params),
            std::bad_any_cast);
}

/** Test module creation with null GDR handle */
TEST_F(OrderKernelPipelineTest, ModuleCreation_NullGdrHandle) {

    // Null GDR handle should throw
    OrderKernelModule::StaticParams params;
    params.gdr_handle = nullptr;
    const std::any params_any = params;

    EXPECT_THROW(
            std::ignore = module_factory_->create_module(
                    "order_kernel_module", "test_module", params_any),
            gsl_lite::fail_fast);
}

// ============================================================================
// Module API Tests
// ============================================================================

/** Test module port names */
TEST_F(OrderKernelPipelineTest, ModuleAPI_PortNames) {

    // Create module
    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;
    auto module = module_factory_->create_module("order_kernel_module", "test_module", params_any);

    ASSERT_NE(module, nullptr);

    // Check input ports
    const auto input_ports = module->get_input_port_names();
    ASSERT_EQ(input_ports.size(), 1);
    EXPECT_EQ(input_ports[0], "doca_objects");

    // Check output ports
    const auto output_ports = module->get_output_port_names();
    ASSERT_EQ(output_ports.size(), 1);
    EXPECT_EQ(output_ports[0], "pusch");
}

/** Test input tensor information */
TEST_F(OrderKernelPipelineTest, ModuleAPI_InputTensorInfo) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;
    auto module = module_factory_->create_module("order_kernel_module", "test_module", params_any);

    // Get input tensor info for "doca_objects"
    const auto tensor_info = module->get_input_tensor_info("doca_objects");
    ASSERT_EQ(tensor_info.size(), 1);
    EXPECT_EQ(tensor_info[0].get_type(), tensor::TensorInfo::DataType::TensorR8U);
    EXPECT_EQ(tensor_info[0].get_dimensions().size(), 1);
    EXPECT_EQ(tensor_info[0].get_dimensions()[0], 8); // Pointer size

    // Invalid port should throw
    EXPECT_THROW(
            std::ignore = module->get_input_tensor_info("invalid_port"), std::invalid_argument);
}

/** Test output tensor information */
TEST_F(OrderKernelPipelineTest, ModuleAPI_OutputTensorInfo) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;
    auto module = module_factory_->create_module("order_kernel_module", "test_module", params_any);

    // Get output tensor info for "pusch"
    const auto tensor_info = module->get_output_tensor_info("pusch");
    ASSERT_EQ(tensor_info.size(), 1);
    EXPECT_EQ(tensor_info[0].get_type(), tensor::TensorInfo::DataType::TensorR16F);
    EXPECT_GT(tensor_info[0].get_dimensions()[0], 0); // PUSCH buffer size

    // Invalid port should throw
    EXPECT_THROW(
            std::ignore = module->get_output_tensor_info("invalid_port"), std::invalid_argument);
}

/** Test memory characteristics */
TEST_F(OrderKernelPipelineTest, ModuleAPI_MemoryCharacteristics) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;
    auto module = module_factory_->create_module("order_kernel_module", "test_module", params_any);

    // Input characteristics (zero-copy required for DOCA)
    const auto input_char = module->get_input_memory_characteristics("doca_objects");
    EXPECT_TRUE(input_char.requires_fixed_address_for_zero_copy);

    // Output characteristics (provides fixed address)
    const auto output_char = module->get_output_memory_characteristics("pusch");
    EXPECT_TRUE(output_char.provides_fixed_address_for_zero_copy);
}

/** Test interface casting */
TEST_F(OrderKernelPipelineTest, ModuleAPI_InterfaceCasting) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;
    auto module = module_factory_->create_module("order_kernel_module", "test_module", params_any);

    // Check interface access
    EXPECT_NE(module->as_stream_executor(), nullptr);
    EXPECT_NE(module->as_graph_node_provider(), nullptr);
}

// ============================================================================
// set_inputs() Tests
// ============================================================================

/** Test set_inputs no-op behavior */
TEST_F(OrderKernelPipelineTest, SetInputs_NoOp) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;
    auto module = module_factory_->create_module("order_kernel_module", "test_module", params_any);

    // set_inputs() is now a no-op (DOCA params come from constructor)
    // Should not throw regardless of input
    std::vector<pipeline::PortInfo> empty_inputs;
    EXPECT_NO_THROW(module->set_inputs(empty_inputs));
}

/** Test constructor with valid DOCA params */
TEST_F(OrderKernelPipelineTest, Constructor_ValidDocaParams) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    const std::any params_any = params;

    // Should succeed with valid parameters
    EXPECT_NO_THROW({
        auto module =
                module_factory_->create_module("order_kernel_module", "test_module", params_any);
    });
}

/** Test constructor rejects null DOCA params */
TEST_F(OrderKernelPipelineTest, Constructor_NullDocaParams) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;
    params.doca_rxq_params = nullptr; // Null DOCA params
    const std::any params_any = params;

    // Should throw due to null doca_rxq_params
    EXPECT_THROW(
            {
                auto module = module_factory_->create_module(
                        "order_kernel_module", "test_module", params_any);
            },
            gsl_lite::fail_fast);
}

/** Test constructor with custom timing parameters */
TEST_F(OrderKernelPipelineTest, Constructor_CustomTimingParams) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;

    // Set custom timing for 15kHz SCS (1ms slot duration)
    params.timing.slot_duration_ns = 1'000'000; // 1ms
    params.timing.ta4_min_ns = 100'000;         // 100us
    params.timing.ta4_max_ns = 900'000;         // 900us

    const std::any params_any = params;
    EXPECT_NO_THROW({
        auto module =
                module_factory_->create_module("order_kernel_module", "test_module", params_any);
    });
}

/** Test constructor rejects invalid timing parameters */
TEST_F(OrderKernelPipelineTest, Constructor_InvalidTimingParams) {

    OrderKernelModule::StaticParams params;
    params.gdr_handle = gdr_handle_.get();
    params.doca_rxq_params = doca_params_;

    // Invalid: ta4_min >= ta4_max
    params.timing.ta4_min_ns = 450'000;
    params.timing.ta4_max_ns = 50'000;

    const std::any params_any = params;
    EXPECT_THROW(
            {
                auto module = module_factory_->create_module(
                        "order_kernel_module", "test_module", params_any);
            },
            gsl_lite::fail_fast);
}

// ============================================================================
// Pipeline Creation Tests
// ============================================================================

/** Test successful pipeline creation */
TEST_F(OrderKernelPipelineTest, PipelineCreation_Success) {

    // Create pipeline spec with 1 module
    pipeline::PipelineSpec spec;
    spec.execution_mode = pipeline::ExecutionMode::Stream;

    // Module spec (pipeline will inject gdr_handle and doca_rxq_params)
    OrderKernelModule::StaticParams module_params;
    module_params.execution_mode = pipeline::ExecutionMode::Stream;
    // Note: gdr_handle and doca_rxq_params are injected by pipeline, not provided in spec

    const pipeline::ModuleCreationInfo module_info{
            .module_type = "order_kernel_module",
            .instance_id = "order_kernel_module_0",
            .init_params = module_params};
    spec.modules.emplace_back(module_info);

    // Create pipeline
    auto pipeline =
            pipeline_factory_->create_pipeline("order_kernel_pipeline", "test_pipeline", spec);

    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->get_pipeline_id(), "test_pipeline");
    EXPECT_EQ(pipeline->get_num_external_inputs(), 1);
    EXPECT_EQ(pipeline->get_num_external_outputs(), 1);
}

/** Test pipeline creation with invalid type */
TEST_F(OrderKernelPipelineTest, PipelineCreation_InvalidType) {

    pipeline::PipelineSpec spec;
    spec.execution_mode = pipeline::ExecutionMode::Stream;

    // Should throw on unsupported type
    EXPECT_THROW(
            std::ignore = pipeline_factory_->create_pipeline("invalid_type", "test_pipeline", spec),
            std::invalid_argument);
}

/** Test pipeline creation rejects wrong module count */
TEST_F(OrderKernelPipelineTest, PipelineCreation_WrongModuleCount) {

    // Spec with 0 modules
    pipeline::PipelineSpec spec_empty;
    spec_empty.execution_mode = pipeline::ExecutionMode::Stream;

    EXPECT_THROW(
            std::ignore = pipeline_factory_->create_pipeline(
                    "order_kernel_pipeline", "test_pipeline", spec_empty),
            std::invalid_argument);

    // Spec with 2 modules (should be exactly 1)
    pipeline::PipelineSpec spec_two;
    spec_two.execution_mode = pipeline::ExecutionMode::Stream;

    const OrderKernelModule::StaticParams params;
    // Note: gdr_handle and doca_rxq_params would be injected by pipeline (not needed in spec)

    const pipeline::ModuleCreationInfo mod1_info{
            .module_type = "order_kernel_module", .instance_id = "mod1", .init_params = params};
    spec_two.modules.emplace_back(mod1_info);

    const pipeline::ModuleCreationInfo mod2_info{
            .module_type = "order_kernel_module", .instance_id = "mod2", .init_params = params};
    spec_two.modules.emplace_back(mod2_info);

    EXPECT_THROW(
            std::ignore = pipeline_factory_->create_pipeline(
                    "order_kernel_pipeline", "test_pipeline", spec_two),
            std::invalid_argument);
}

} // namespace ran::fronthaul::tests

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
