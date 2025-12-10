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
 * @file pipeline_sample_tests.cpp
 * @brief Sample tests for pipeline library documentation
 */

#include <any>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <driver_types.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/types.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

namespace pipeline = framework::pipeline;
namespace tensor = framework::tensor;

// ============================================================================
// Simple Mock Module for Documentation Examples
// ============================================================================

/**
 * SimpleModule - Minimal module implementation for documentation
 *
 * Demonstrates the basic IModule interface without GPU complexity
 */
class SimpleModule final : public pipeline::IModule {
public:
    struct StaticParams {
        std::size_t tensor_size{1024};
    };

    SimpleModule(std::string instance_id, const StaticParams &params)
            : instance_id_(std::move(instance_id)), tensor_size_(params.tensor_size) {}

    [[nodiscard]] std::string_view get_type_id() const override { return "simple_module"; }

    [[nodiscard]] std::string_view get_instance_id() const override { return instance_id_; }

    void setup_memory(const pipeline::ModuleMemorySlice & /*memory_slice*/) override {}

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_input_tensor_info(std::string_view /*port_name*/) const override {
        return {tensor::TensorInfo{tensor::TensorInfo::DataType::TensorR32F, {tensor_size_}}};
    }

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_output_tensor_info(std::string_view /*port_name*/) const override {
        return {tensor::TensorInfo{tensor::TensorInfo::DataType::TensorR32F, {tensor_size_}}};
    }

    [[nodiscard]] std::vector<std::string> get_input_port_names() const override {
        return {"input"};
    }

    [[nodiscard]] std::vector<std::string> get_output_port_names() const override {
        return {"output"};
    }

    void set_inputs(std::span<const pipeline::PortInfo> /*inputs*/) override {}

    [[nodiscard]] std::vector<pipeline::PortInfo> get_outputs() const override { return {}; }

    void
    configure_io(const pipeline::DynamicParams & /*params*/, cudaStream_t /*stream*/) override {}

    [[nodiscard]] pipeline::IGraphNodeProvider *as_graph_node_provider() override {
        return nullptr;
    }

    [[nodiscard]] pipeline::IStreamExecutor *as_stream_executor() override { return nullptr; }

private:
    std::string instance_id_;
    std::size_t tensor_size_{};
};

/**
 * SimpleModuleFactory - Factory for creating SimpleModule instances
 */
class SimpleModuleFactory final : public pipeline::IModuleFactory {
public:
    [[nodiscard]] std::unique_ptr<pipeline::IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override {
        if (module_type != "simple_module") {
            return nullptr;
        }

        const auto &params = std::any_cast<const SimpleModule::StaticParams &>(static_params);
        return std::make_unique<SimpleModule>(instance_id, params);
    }

    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override {
        return module_type == "simple_module";
    }
};

// ============================================================================
// Documentation Sample Tests
// ============================================================================

TEST(PipelineSampleTests, ModuleCreation) {
    // example-begin module-creation-1
    // Create a module with configuration parameters
    const SimpleModule::StaticParams params{.tensor_size = 2048};
    auto module = std::make_unique<SimpleModule>("my_module", params);

    // Query module properties
    const auto type_id = module->get_type_id();
    const auto instance_id = module->get_instance_id();
    // example-end module-creation-1

    EXPECT_EQ(type_id, "simple_module");
    EXPECT_EQ(instance_id, "my_module");
}

TEST(PipelineSampleTests, ModulePorts) {
    const SimpleModule::StaticParams params{.tensor_size = 1024};
    auto module = std::make_unique<SimpleModule>("test_module", params);

    // example-begin module-ports-1
    // Query input and output ports
    const auto input_ports = module->get_input_port_names();
    const auto output_ports = module->get_output_port_names();

    // Get tensor information for a specific port
    const auto input_info = module->get_input_tensor_info("input");
    const auto output_info = module->get_output_tensor_info("output");
    // example-end module-ports-1

    EXPECT_EQ(input_ports.size(), 1);
    EXPECT_EQ(output_ports.size(), 1);
    EXPECT_FALSE(input_info.empty());
    EXPECT_FALSE(output_info.empty());
}

TEST(PipelineSampleTests, ModuleFactory) {
    // example-begin module-factory-1
    // Create a module factory
    auto factory = std::make_unique<SimpleModuleFactory>();

    // Check if a module type is supported
    const bool supported = factory->supports_module_type("simple_module");

    // Create a module using the factory
    const SimpleModule::StaticParams params{.tensor_size = 512};
    auto module = factory->create_module("simple_module", "factory_module", std::any(params));
    // example-end module-factory-1

    EXPECT_TRUE(supported);
    EXPECT_NE(module, nullptr);
    EXPECT_EQ(module->get_instance_id(), "factory_module");
}

TEST(PipelineSampleTests, PipelineSpecBasics) {
    // example-begin pipeline-spec-basic-1
    // Create a pipeline specification
    pipeline::PipelineSpec spec;
    spec.pipeline_name = "MyPipeline";
    spec.execution_mode = pipeline::ExecutionMode::Stream;

    // Define module configuration
    const SimpleModule::StaticParams module_params{.tensor_size = 1024};

    const pipeline::ModuleSpec module_spec(pipeline::ModuleCreationInfo{
            .module_type = "simple_module",
            .instance_id = "module_1",
            .init_params = std::any(module_params)});

    spec.modules.push_back(module_spec);

    // Define external I/O
    spec.external_inputs = {"input"};
    spec.external_outputs = {"output"};
    // example-end pipeline-spec-basic-1

    EXPECT_EQ(spec.pipeline_name, "MyPipeline");
    EXPECT_EQ(spec.modules.size(), 1);
    EXPECT_EQ(spec.external_inputs.size(), 1);
}

TEST(PipelineSampleTests, PipelineSpecWithConnections) {
    // example-begin pipeline-spec-connections-1
    pipeline::PipelineSpec spec;
    spec.pipeline_name = "TwoModulePipeline";

    // Add two modules
    const SimpleModule::StaticParams params{.tensor_size = 1024};

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "simple_module",
            .instance_id = "module_a",
            .init_params = std::any(params)});

    spec.modules.emplace_back(pipeline::ModuleCreationInfo{
            .module_type = "simple_module",
            .instance_id = "module_b",
            .init_params = std::any(params)});

    // Connect module_a output to module_b input
    const pipeline::PortConnection connection{
            .source_module = "module_a",
            .source_port = "output",
            .target_module = "module_b",
            .target_port = "input"};

    spec.connections.push_back(connection);
    // example-end pipeline-spec-connections-1

    EXPECT_EQ(spec.modules.size(), 2);
    EXPECT_EQ(spec.connections.size(), 1);
    EXPECT_EQ(spec.connections[0].source_module, "module_a");
    EXPECT_EQ(spec.connections[0].target_module, "module_b");
}

TEST(PipelineSampleTests, ExecutionModes) {
    // example-begin execution-modes-1
    // Stream mode - flexible addressing, suitable for development
    pipeline::PipelineSpec stream_spec;
    stream_spec.execution_mode = pipeline::ExecutionMode::Stream;

    // Graph mode - fixed addressing, optimal for production
    pipeline::PipelineSpec graph_spec;
    graph_spec.execution_mode = pipeline::ExecutionMode::Graph;
    // example-end execution-modes-1

    EXPECT_EQ(stream_spec.execution_mode, pipeline::ExecutionMode::Stream);
    EXPECT_EQ(graph_spec.execution_mode, pipeline::ExecutionMode::Graph);
}

TEST(PipelineSampleTests, PortInfo) {
    // example-begin port-info-1
    // Allocate device memory for a tensor
    const std::size_t tensor_size = 1024;
    void *device_ptr{};
    cudaMalloc(&device_ptr, tensor_size * sizeof(float));

    // Create tensor info describing the data
    const tensor::TensorInfo tensor_info{tensor::TensorInfo::DataType::TensorR32F, {tensor_size}};

    // Create device tensor wrapper
    const pipeline::DeviceTensor device_tensor{
            .device_ptr = device_ptr, .tensor_info = tensor_info};

    // Create port info for external input
    pipeline::PortInfo port_info{.name = "input0", .tensors = {device_tensor}};
    // example-end port-info-1

    EXPECT_EQ(port_info.name, "input0");
    EXPECT_EQ(port_info.tensors.size(), 1);
    EXPECT_EQ(port_info.tensors[0].device_ptr, device_ptr);

    cudaFree(device_ptr);
}

TEST(PipelineSampleTests, ConnectionCopyMode) {
    // example-begin connection-copy-mode-1
    // Configure zero-copy mode for a connection
    const auto module_params = SimpleModule::StaticParams{.tensor_size = 1024};
    auto module = std::make_unique<SimpleModule>("my_module", module_params);

    // Set connection to use zero-copy (if supported)
    module->set_connection_copy_mode("input", pipeline::ConnectionCopyMode::ZeroCopy);

    // Or configure to always copy data
    module->set_connection_copy_mode("input", pipeline::ConnectionCopyMode::Copy);
    // example-end connection-copy-mode-1

    // Test passes if no exceptions thrown
    EXPECT_NE(module, nullptr);
}

TEST(PipelineSampleTests, ModuleMemoryCharacteristics) {
    const auto params = SimpleModule::StaticParams{.tensor_size = 1024};
    auto module = std::make_unique<SimpleModule>("test_module", params);

    // example-begin memory-characteristics-1
    // Query input memory characteristics
    const auto input_chars = module->get_input_memory_characteristics("input");

    // Query output memory characteristics
    const auto output_chars = module->get_output_memory_characteristics("output");

    // Check if zero-copy is possible
    const bool can_use_zero_copy = pipeline::can_zero_copy(output_chars, input_chars);
    // example-end memory-characteristics-1

    // Module uses default characteristics
    EXPECT_FALSE(input_chars.requires_fixed_address_for_zero_copy);
    EXPECT_TRUE(output_chars.provides_fixed_address_for_zero_copy);
    EXPECT_TRUE(can_use_zero_copy);
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
