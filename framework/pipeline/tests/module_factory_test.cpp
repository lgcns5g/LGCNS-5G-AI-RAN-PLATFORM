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

#include <any>             // for any, any_cast, bad_any_cast
#include <cstddef>         // for size_t
#include <format>          // for format
#include <memory>          // for allocator, unique_ptr
#include <source_location> // for source_location
#include <span>
#include <stdexcept>   // for invalid_argument
#include <string>      // for operator==, string, oper...
#include <string_view> // for string_view
#include <tuple>       // for _Swallow_assign, ignore
#include <typeinfo>    // for type_info
#include <vector>      // for vector

#include <driver_types.h>

#include <gtest/gtest.h> // for AssertionResult, Message

#include "pipeline/imodule.hpp"                 // for IModule, IGraphNodeProvider
#include "pipeline/imodule_factory.hpp"         // for IModuleFactory
#include "pipeline/istream_executor.hpp"        // for IStreamExecutor
#include "pipeline/stream_executor_factory.hpp" // for StreamExecutorFactory
#include "pipeline/types.hpp"                   // for PortInfo, DynamicTickDat...
#include "tensor/data_types.hpp"                // for NvDataType
#include "tensor/tensor_info.hpp"               // for TensorInfo

namespace tensor = framework::tensor;

namespace {

// Configuration structures for modules
struct GemmConfig {
    struct Dimensions {
        std::size_t m{};
        std::size_t n{};
        std::size_t k{};
    } dims{};
    float alpha{};
    float beta{};
};

struct ReLUConfig final {
    std::size_t num_elements{};
    tensor::NvDataType data_type{};
};

// Base mock module class with common functionality
class MockModuleBase : public framework::pipeline::IModule {
public:
    MockModuleBase(const char *type_id, const char *instance_id /*,
                 StreamExecutorFactory *executor_factory*/)
      : type_id_{type_id}, instance_id_{instance_id} /*
         executor_factory_{executor_factory}*/
  {}

    [[nodiscard]] std::string_view get_type_id() const override { return type_id_; }
    [[nodiscard]] std::string_view get_instance_id() const override { return instance_id_; }

    void setup_memory(const framework::pipeline::ModuleMemorySlice & /*memory_slice*/) override {
        // Mock implementation
    }

    void set_inputs(std::span<const framework::pipeline::PortInfo> /*inputs*/) override {
        // Mock implementation
    }

    void configure_io(
            const framework::pipeline::DynamicParams & /*params*/,
            cudaStream_t /*stream*/) override {
        // Mock implementation
    }

    framework::pipeline::IGraphNodeProvider *as_graph_node_provider() override { return nullptr; }
    framework::pipeline::IStreamExecutor *as_stream_executor() override { return nullptr; }

protected:
    std::string type_id_;
    std::string instance_id_;
    // StreamExecutorFactory *executor_factory_{};
};

// Mock GEMM module
class GemmModule : public MockModuleBase {
public:
    GemmModule(const char *type_id, const char *instance_id,
             const std::any &static_params /*,
             StreamExecutorFactory *executorFactory*/)
      : MockModuleBase(type_id, instance_id /*, executorFactory*/),
        config_{std::any_cast<GemmConfig>(static_params)} {}

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override {
        if (port_name == "A") {
            return {tensor::TensorInfo{tensor::TensorR32F, {config_.dims.m, config_.dims.k}}};
        } else if (port_name == "B") {
            return {tensor::TensorInfo{tensor::TensorR32F, {config_.dims.k, config_.dims.n}}};
        } else if (port_name == "C") {
            return {tensor::TensorInfo{tensor::TensorR32F, {config_.dims.m, config_.dims.n}}};
        }
        throw std::invalid_argument(std::format("Unknown input port: {}", port_name));
    }

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override {
        if (port_name == "D") {
            return {tensor::TensorInfo{tensor::TensorR32F, {config_.dims.m, config_.dims.n}}};
        }
        throw std::invalid_argument(std::format("Unknown output port: {}", port_name));
    }

    [[nodiscard]] std::vector<std::string> get_input_port_names() const override {
        return {"A", "B", "C"};
    }

    [[nodiscard]] std::vector<std::string> get_output_port_names() const override { return {"D"}; }

    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override {
        framework::pipeline::PortInfo port;
        port.name = "D";
        port.tensors.push_back(framework::pipeline::DeviceTensor{
                .device_ptr = nullptr, .tensor_info = get_output_tensor_info("D")[0]});
        return {port};
    }

private:
    GemmConfig config_{};
};

// Mock ReLU module
class ReLUModule : public MockModuleBase {
public:
    ReLUModule(const char *type_id, const char *instance_id,
             const std::any &static_params /*,
             StreamExecutorFactory *executorFactory*/)
      : MockModuleBase(type_id, instance_id /*, executorFactory*/),
        config_{std::any_cast<ReLUConfig>(static_params)} {}

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override {
        if (port_name == "input") {
            return {tensor::TensorInfo{config_.data_type, {config_.num_elements}}};
        }
        throw std::invalid_argument(std::format("Unknown input port: {}", port_name));
    }

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override {
        if (port_name == "output") {
            return {tensor::TensorInfo{config_.data_type, {config_.num_elements}}};
        }
        throw std::invalid_argument(std::format("Unknown output port: {}", port_name));
    }

    [[nodiscard]] std::vector<std::string> get_input_port_names() const override {
        return {"input"};
    }

    [[nodiscard]] std::vector<std::string> get_output_port_names() const override {
        return {"output"};
    }

    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override {
        framework::pipeline::PortInfo port;
        port.name = "output";
        port.tensors.push_back(framework::pipeline::DeviceTensor{
                .device_ptr = nullptr, .tensor_info = get_output_tensor_info("output")[0]});
        return {port};
    }

private:
    ReLUConfig config_{};
};

// Helper function for creating modules
template <typename CONFIG, typename MODULE>
std::unique_ptr<framework::pipeline::IModule> create_module_impl(
        const std::string &instance_id,
        const std::any &static_params,
        const char *module_type,
        /*const framework::pipeline::StreamExecutorFactory *executor_factory,*/
        const std::source_location &location = std::source_location::current()) {
    // Validate that we have the correct config type
    try {
        std::any_cast<CONFIG>(static_params);
    } catch (const std::bad_any_cast &) {
        // Extract function name from source location
        const std::string function_name = location.function_name();

        // Format error message with the calling function name and expected config
        // type
        throw std::invalid_argument(
                std::format("{} expects {}", function_name, typeid(CONFIG).name()));
    }

    // Create module with type and instance IDs, static params and executor
    // factory
    auto module = std::make_unique<MODULE>(
            module_type, instance_id.c_str(), static_params /*, executor_factory*/);

    return module;
}

// ModuleFactory implementation
class ModuleFactory final : public framework::pipeline::IModuleFactory {
public:
    /**
     * Constructor
     *
     * @param[in] executorType The type of stream executors to create for modules
     */
    explicit ModuleFactory(
            framework::pipeline::StreamExecutorFactory::ExecutorType executor_type =
                    framework::pipeline::StreamExecutorFactory::ExecutorType::Null)
            : executor_factory_{std::make_unique<framework::pipeline::StreamExecutorFactory>(
                      executor_type)} {}

    /**
     * Create a module of the specified type.
     *
     * @param[in] module_type The type of module to create (currently supports
     * "gemm")
     * @param[in] instance_id The unique instance identifier for this module
     * @param[in] static_params Type-erased static parameters for module
     * initialization
     * @return Unique pointer to the created module
     * @throws std::invalid_argument if module_type is not supported
     * @throws std::bad_any_cast if static_params type doesn't match module
     * requirements
     */
    [[nodiscard]] std::unique_ptr<framework::pipeline::IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override {
        if (module_type == "gemm") {
            return create_gemm_module(instance_id, static_params);
        }

        if (module_type == "relu") {
            return create_relu_module(instance_id, static_params);
        }

        throw std::invalid_argument(std::format("Unsupported module type: {}", module_type));
    }

    /**
     * Check if a module type is supported by this factory.
     *
     * @param[in] module_type The type of module to check
     * @return true if the module type is supported, false otherwise
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override {
        return module_type == "gemm" || module_type == "relu";
    }

private:
    /**
     * Create a GEMM module with specific parameters.
     *
     * @param[in] instance_id The unique instance identifier
     * @param[in] static_params GemmConfig for the module
     * @return Unique pointer to the created GemmModule
     */
    [[nodiscard]] static std::unique_ptr<framework::pipeline::IModule>
    create_gemm_module(const std::string &instance_id, const std::any &static_params) {
        return create_module_impl<GemmConfig, GemmModule>(
                instance_id,
                static_params,
                "gemm", /*executor_factory_.get(),*/
                std::source_location::current());
    }

    /**
     * Create a ReLU module with specific parameters.
     *
     * @param[in] instance_id The unique instance identifier
     * @param[in] static_params ReLUConfig for the module
     * @return Unique pointer to the created ReLUModule
     */
    [[nodiscard]] static std::unique_ptr<framework::pipeline::IModule>
    create_relu_module(const std::string &instance_id, const std::any &static_params) {
        return create_module_impl<ReLUConfig, ReLUModule>(
                instance_id,
                static_params,
                "relu", /*executor_factory_.get(),*/
                std::source_location::current());
    }

    std::unique_ptr<framework::pipeline::StreamExecutorFactory>
            executor_factory_; //!< Factory for creating stream executors
};

class ModuleFactoryTest : public ::testing::Test {
protected:
    void SetUp() override { factory_ = std::make_unique<ModuleFactory>(); }

    void TearDown() override { factory_.reset(); }

    std::unique_ptr<ModuleFactory> factory_;
};

// Test: Verifies factory supports GEMM module type
TEST_F(ModuleFactoryTest, SupportsGemmModuleType) {
    EXPECT_TRUE(factory_->supports_module_type("gemm"));
}

// Test: Verifies factory supports ReLU module type
TEST_F(ModuleFactoryTest, SupportsReLUModuleType) {
    EXPECT_TRUE(factory_->supports_module_type("relu"));
}

// Test: Ensures factory correctly reports unsupported module types
TEST_F(ModuleFactoryTest, DoesNotSupportUnknownModuleType) {
    EXPECT_FALSE(factory_->supports_module_type("unknown"));
    EXPECT_FALSE(factory_->supports_module_type(""));
    EXPECT_FALSE(factory_->supports_module_type("fft"));
}

// Test: Validates successful GEMM module creation with valid parameters
TEST_F(ModuleFactoryTest, CreateGemmModuleWithValidParams) {
    GemmConfig config;
    static constexpr std::size_t M_SIZE = 64;
    static constexpr std::size_t N_SIZE = 32;
    static constexpr std::size_t K_SIZE = 16;
    static constexpr float ALPHA = 1.0F;
    static constexpr float BETA = 0.0F;

    config.dims.m = M_SIZE;
    config.dims.n = N_SIZE;
    config.dims.k = K_SIZE;
    config.alpha = ALPHA;
    config.beta = BETA;

    const auto module = factory_->create_module("gemm", "test_gemm", std::any(config));

    ASSERT_NE(module, nullptr);
    EXPECT_EQ(module->get_type_id(), "gemm");
    EXPECT_EQ(module->get_instance_id(), "test_gemm");
}

// Test: Validates successful ReLU module creation with valid parameters
TEST_F(ModuleFactoryTest, CreateReLUModuleWithValidParams) {
    ReLUConfig config;
    static constexpr std::size_t NUM_ELEMENTS = 1024;
    config.num_elements = NUM_ELEMENTS;
    config.data_type = tensor::TensorR32F;

    const auto module = factory_->create_module("relu", "test_relu", std::any(config));

    ASSERT_NE(module, nullptr);
    EXPECT_EQ(module->get_type_id(), "relu");
    EXPECT_EQ(module->get_instance_id(), "test_relu");
}

// Test: Ensures GEMM module creation fails with wrong parameter type
TEST_F(ModuleFactoryTest, CreateGemmModuleWithInvalidParams) {
    // Test with wrong parameter type
    EXPECT_THROW(
            std::ignore = factory_->create_module("gemm", "test_gemm", std::any(42)),
            std::invalid_argument);
}

// Test: Verifies creation fails for unsupported module types
TEST_F(ModuleFactoryTest, CreateUnsupportedModuleType) {
    GemmConfig config;
    static constexpr std::size_t M_SIZE = 64;
    static constexpr std::size_t N_SIZE = 32;
    static constexpr std::size_t K_SIZE = 16;

    config.dims.m = M_SIZE;
    config.dims.n = N_SIZE;
    config.dims.k = K_SIZE;

    EXPECT_THROW(
            std::ignore = factory_->create_module("unknown", "test_unknown", std::any(config)),
            std::invalid_argument);
}

// Test: Validates factory can create multiple modules with different instance
// IDs
TEST_F(ModuleFactoryTest, CreateMultipleModulesWithDifferentInstanceIds) {
    GemmConfig config1;
    static constexpr std::size_t M_SIZE_1 = 64;
    static constexpr std::size_t N_SIZE_1 = 32;
    static constexpr std::size_t K_SIZE_1 = 16;

    static constexpr std::size_t M_SIZE_2 = 128;
    static constexpr std::size_t N_SIZE_2 = 64;
    static constexpr std::size_t K_SIZE_2 = 32;

    config1.dims.m = M_SIZE_1;
    config1.dims.n = N_SIZE_1;
    config1.dims.k = K_SIZE_1;

    GemmConfig config2;
    config2.dims.m = M_SIZE_2;
    config2.dims.n = N_SIZE_2;
    config2.dims.k = K_SIZE_2;

    auto module1 = factory_->create_module("gemm", "gemm_1", std::any(config1));
    auto module2 = factory_->create_module("gemm", "gemm_2", std::any(config2));

    ASSERT_NE(module1, nullptr);
    ASSERT_NE(module2, nullptr);

    EXPECT_EQ(module1->get_instance_id(), "gemm_1");
    EXPECT_EQ(module2->get_instance_id(), "gemm_2");

    // Both should have the same type
    EXPECT_EQ(module1->get_type_id(), "gemm");
    EXPECT_EQ(module2->get_type_id(), "gemm");
}

// Test: Validates created modules have valid port information and tensor info
TEST_F(ModuleFactoryTest, CreatedModuleHasValidPortInformation) {
    GemmConfig config;
    static constexpr std::size_t M_SIZE = 64;
    static constexpr std::size_t N_SIZE = 32;
    static constexpr std::size_t K_SIZE = 16;

    config.dims.m = M_SIZE;
    config.dims.n = N_SIZE;
    config.dims.k = K_SIZE;

    const auto module = factory_->create_module("gemm", "test_gemm", std::any(config));

    // Check port information
    const auto input_ports = module->get_input_port_names();
    const auto output_ports = module->get_output_port_names();

    EXPECT_FALSE(input_ports.empty());
    EXPECT_FALSE(output_ports.empty());

    // For GEMM, we expect specific ports (this depends on GemmModule
    // implementation) We just check that we can get tensor info for the ports
    for (const auto &port_name : input_ports) {
        EXPECT_NO_THROW({ std::ignore = module->get_input_tensor_info(port_name); });
    }

    for (const auto &port_name : output_ports) {
        EXPECT_NO_THROW({ std::ignore = module->get_output_tensor_info(port_name); });
    }
}

} // namespace
