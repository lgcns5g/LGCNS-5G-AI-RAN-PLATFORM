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
#include <any>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include "sample_module_a.hpp"
#include "sample_module_b.hpp"
#include "sample_module_factories.hpp"
#include "sample_pipeline_factory.hpp"
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp"
#include "utils/error_macros.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace framework::pipelines::samples::tests {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;
namespace tensor = ::framework::tensor;

namespace {

/**
 * Get the path to the TensorRT engine file
 *
 * @return Absolute path to the TRT engine
 */
std::string get_trt_engine_path() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char *test_data_dir = std::getenv("TEST_DATA_DIR");
    if (test_data_dir == nullptr) {
        throw std::runtime_error(
                "TEST_DATA_DIR environment variable not set. This should be set by "
                "CMake/CTest.");
    }

    return std::string(test_data_dir) + "/tensorrt_cluster_engine_data.trtengine";
}

// TensorRT engine expects tensors of size 16384
constexpr std::size_t DEFAULT_TENSOR_SIZE = 16384;
constexpr float FLOAT_TOLERANCE = 1e-5F;

} // namespace

/**
 * Test fixture for SamplePipeline integration tests
 *
 * Provides helper methods for:
 * - CUDA stream management
 * - Device memory allocation
 * - Data copying (H2D, D2H)
 * - Output validation
 */
class SamplePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        RT_LOG_INFO("SamplePipelineTest: SetUp()");
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamCreate(&stream_));

        // Create factories for pipeline creation
        module_factory_ = std::make_unique<SampleModuleFactory>();
        pipeline_factory_ = std::make_unique<SamplePipelineFactory>(
                gsl_lite::not_null<pipeline::IModuleFactory *>(module_factory_.get()));

        RT_LOG_DEBUG("SamplePipelineTest: Factories initialized");
    }

    void TearDown() override {
        RT_LOG_INFO("SamplePipelineTest: TearDown()");

        // Destroy factories
        pipeline_factory_.reset();
        module_factory_.reset();

        // Free all allocated device memory
        for (void *ptr : device_allocations_) {
            if (ptr != nullptr) {
                FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaFree(ptr));
            }
        }
        device_allocations_.clear();

        if (stream_ != nullptr) {
            FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    /**
     * Allocate device memory and track for cleanup
     *
     * @param[in] size_bytes Size in bytes to allocate
     * @return Device pointer
     */
    void *allocate_device_memory(std::size_t size_bytes) {
        void *d_ptr = nullptr;
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMalloc(&d_ptr, size_bytes));
        device_allocations_.push_back(d_ptr);
        RT_LOG_DEBUG("Allocated {} bytes at device ptr {}", size_bytes, d_ptr);
        return d_ptr;
    }

    /**
     * Copy host data to device using stream-aware async copy
     *
     * @param[in] d_ptr Device pointer
     * @param[in] h_data Host data vector
     * @param[in] stream CUDA stream for async memory copy
     */
    static void copy_to_device(void *d_ptr, const std::vector<float> &h_data, cudaStream_t stream) {
        RT_LOG_DEBUG(
                "Copying {} floats ({} bytes) to device ptr {}",
                h_data.size(),
                h_data.size() * sizeof(float),
                d_ptr);
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                d_ptr,
                h_data.data(),
                h_data.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream));
        // Synchronize to ensure copy completes
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));
    }

    /**
     * Copy device data to host using stream-aware async copy
     *
     * @param[out] h_data Host data vector (will be resized)
     * @param[in] d_ptr Device pointer
     * @param[in] num_elements Number of float elements to copy
     * @param[in] stream CUDA stream for async memory copy
     */
    static void copy_from_device(
            std::vector<float> &h_data,
            void *d_ptr,
            std::size_t num_elements,
            cudaStream_t stream) {
        h_data.resize(num_elements);
        RT_LOG_DEBUG(
                "Copying {} floats ({} bytes) from device ptr {}",
                num_elements,
                num_elements * sizeof(float),
                d_ptr);
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                h_data.data(),
                d_ptr,
                num_elements * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream));
        // Synchronize to ensure copy completes before returning
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));
    }

    /**
     * Create PortInfo for external input
     *
     * @param[in] port_name Name of the port
     * @param[in] d_ptr Device pointer to tensor data
     * @param[in] tensor_size Number of elements
     * @return PortInfo structure
     */
    static pipeline::PortInfo
    create_port_info(const std::string &port_name, void *d_ptr, std::size_t tensor_size) {
        const tensor::TensorInfo tensor_info{
                tensor::TensorInfo::DataType::TensorR32F, {tensor_size}};
        const pipeline::DeviceTensor device_tensor{.device_ptr = d_ptr, .tensor_info = tensor_info};

        return pipeline::PortInfo{.name = port_name, .tensors = {device_tensor}};
    }

    /**
     * Validate output against expected values
     *
     * @param[in] actual Actual output from pipeline
     * @param[in] expected Expected output values
     * @param[in] tolerance Floating-point comparison tolerance
     * @return true if all values match within tolerance
     */
    static bool validate_output(
            const std::vector<float> &actual,
            const std::vector<float> &expected,
            float tolerance = FLOAT_TOLERANCE) {
        if (actual.size() != expected.size()) {
            RT_LOG_ERROR("Size mismatch: actual={}, expected={}", actual.size(), expected.size());
            return false;
        }

        bool all_match = true;
        std::size_t num_mismatches = 0;

        for (std::size_t i = 0; i < actual.size(); ++i) {
            const float diff = std::abs(actual[i] - expected[i]);
            if (diff > tolerance) {
                if (num_mismatches < 10) { // Log first 10 mismatches
                    RT_LOG_ERROR(
                            "Mismatch at index {}: actual={}, expected={}, diff={}",
                            i,
                            actual[i],
                            expected[i],
                            diff);
                }
                all_match = false;
                ++num_mismatches;
            }
        }

        if (num_mismatches > 0) {
            RT_LOG_ERROR("Total mismatches: {} / {}", num_mismatches, actual.size());
        } else {
            RT_LOG_INFO("All {} values match within tolerance {}", actual.size(), tolerance);
        }

        return all_match;
    }

    /**
     * Compute expected output for add + relu pipeline
     *
     * @param[in] input0 First input vector
     * @param[in] input1 Second input vector
     * @return Expected output after add and relu
     */
    static std::vector<float>
    compute_expected_output(const std::vector<float> &input0, const std::vector<float> &input1) {
        std::vector<float> expected(input0.size());
        for (std::size_t i = 0; i < input0.size(); ++i) {
            const float add_result = input0[i] + input1[i];
            expected[i] = std::max(0.0F, add_result); // ReLU
        }
        return expected;
    }

    /**
     * Create a PipelineSpec for SamplePipeline
     *
     * @param[in] tensor_size Size of tensors for both modules
     * @param[in] trt_engine_path Path to TensorRT engine file
     * @param[in] execution_mode Pipeline execution mode (Stream or Graph)
     * @return Configured PipelineSpec
     */
    // example-begin pipeline-spec-creation-1
    [[nodiscard]] static pipeline::PipelineSpec create_pipeline_spec(
            std::size_t tensor_size,
            const std::string &trt_engine_path,
            pipeline::ExecutionMode execution_mode) {

        pipeline::PipelineSpec spec;
        spec.pipeline_name = "SamplePipeline";
        spec.execution_mode = execution_mode;

        // Module A configuration
        const SampleModuleA::StaticParams module_a_params{
                .tensor_size = tensor_size,
                .trt_engine_path = trt_engine_path,
                .execution_mode = spec.execution_mode};

        const pipeline::ModuleCreationInfo module_a_info{
                .module_type = "sample_module_a",
                .instance_id = "module_a",
                .init_params = std::any(module_a_params)};

        spec.modules.emplace_back(module_a_info);

        // Module B configuration
        const SampleModuleB::StaticParams module_b_params{
                .tensor_size = tensor_size, .execution_mode = spec.execution_mode};

        const pipeline::ModuleCreationInfo module_b_info{
                .module_type = "sample_module_b",
                .instance_id = "module_b",
                .init_params = std::any(module_b_params)};

        spec.modules.emplace_back(module_b_info);

        // Connections
        const pipeline::PortConnection connection{
                .source_module = "module_a",
                .source_port = "output",
                .target_module = "module_b",
                .target_port = "input"};

        spec.connections.push_back(connection);

        // External I/O
        spec.external_inputs = {"input0", "input1"};
        spec.external_outputs = {"output"};

        return spec;
    }
    // example-end pipeline-spec-creation-1

    cudaStream_t stream_{nullptr};
    std::vector<void *> device_allocations_;
    std::unique_ptr<SampleModuleFactory> module_factory_;
    std::unique_ptr<SamplePipelineFactory> pipeline_factory_;
};

/**
 * Test stream execution mode with positive values
 *
 * Verifies:
 * - Pipeline construction and setup
 * - External input/output handling
 * - TensorRT module execution (add)
 * - CUDA kernel module execution (ReLU)
 * - Output correctness
 */
TEST_F(SamplePipelineTest, StreamExecution_ValidatesCorrectOutput) {
    RT_LOG_INFO("=== StreamExecution_ValidatesCorrectOutput ===");

    // Configuration
    const std::size_t tensor_size = DEFAULT_TENSOR_SIZE;
    const std::string engine_path = get_trt_engine_path();

    RT_LOG_INFO("Using TensorRT engine: {}", engine_path);
    ASSERT_TRUE(std::filesystem::exists(engine_path))
            << "TensorRT engine not found at: " << engine_path;

    // Create pipeline via factory
    RT_LOG_INFO("Creating pipeline via factory");
    // example-begin pipeline-creation-1
    const auto spec =
            create_pipeline_spec(tensor_size, engine_path, pipeline::ExecutionMode::Stream);
    auto pipeline = pipeline_factory_->create_pipeline("sample", "test_pipeline", spec);

    // Setup pipeline (allocates memory, initializes modules)
    pipeline->setup();
    // example-end pipeline-creation-1
    RT_LOG_INFO("Calling pipeline setup()");

    // Prepare host input data
    std::vector<float> h_input0(tensor_size);
    std::vector<float> h_input1(tensor_size);

    for (std::size_t i = 0; i < tensor_size; ++i) {
        h_input0[i] = static_cast<float>(i + 1); // [1, 2, 3, ...]
        h_input1[i] = static_cast<float>(i + 2); // [2, 3, 4, ...]
    }

    // Compute expected output: (input0 + input1) -> relu
    const std::vector<float> expected_output = compute_expected_output(h_input0, h_input1);

    RT_LOG_DEBUG(
            "First 5 expected values: [{}, {}, {}, {}, {}]",
            expected_output[0],
            expected_output[1],
            expected_output[2],
            expected_output[3],
            expected_output[4]);

    // Allocate device memory for external inputs
    void *d_input0 = allocate_device_memory(tensor_size * sizeof(float));
    void *d_input1 = allocate_device_memory(tensor_size * sizeof(float));

    // Copy input data to device using stream-aware operations
    copy_to_device(d_input0, h_input0, stream_);
    copy_to_device(d_input1, h_input1, stream_);

    // Create external input port infos
    std::vector<pipeline::PortInfo> external_inputs;
    external_inputs.push_back(create_port_info("input0", d_input0, tensor_size));
    external_inputs.push_back(create_port_info("input1", d_input1, tensor_size));

    // Create placeholder for external output (pipeline will fill this in)
    std::vector<pipeline::PortInfo> external_outputs(1);

    // example-begin pipeline-configure-execute-1
    // Configure I/O with external inputs/outputs
    RT_LOG_INFO("Calling configure_io()");
    const pipeline::DynamicParams params{.module_specific_params = {}};
    pipeline->configure_io(params, external_inputs, external_outputs, stream_);

    // Perform one-time warmup (loads TRT engine, captures CUDA graph)
    RT_LOG_INFO("Calling warmup()");
    pipeline->warmup(stream_);

    // Execute pipeline in stream mode
    RT_LOG_INFO("Calling execute_stream()");
    pipeline->execute_stream(stream_);

    // Synchronize to ensure completion
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream_));
    RT_LOG_INFO("Stream execution completed");
    // example-end pipeline-configure-execute-1

    // Get actual output pointer from pipeline (filled in by configure_io)
    ASSERT_FALSE(external_outputs.empty());
    ASSERT_FALSE(external_outputs[0].tensors.empty());
    void *d_output = external_outputs[0].tensors[0].device_ptr;
    RT_LOG_DEBUG("Pipeline output at device ptr: {}", d_output);

    // Copy output back to host using stream-aware operation
    std::vector<float> h_output;
    copy_from_device(h_output, d_output, tensor_size, stream_);

    RT_LOG_DEBUG(
            "First 5 actual values: [{}, {}, {}, {}, {}]",
            h_output[0],
            h_output[1],
            h_output[2],
            h_output[3],
            h_output[4]);

    // Validate output
    EXPECT_TRUE(validate_output(h_output, expected_output));

    RT_LOG_INFO("=== StreamExecution_ValidatesCorrectOutput PASSED ===");
}

/**
 * Test graph execution mode with positive values
 *
 * Verifies:
 * - Graph construction
 * - Graph execution
 * - Output matches stream execution
 */
TEST_F(SamplePipelineTest, GraphExecution_ValidatesCorrectOutput) {
    RT_LOG_INFO("=== GraphExecution_ValidatesCorrectOutput ===");

    // Configuration
    const std::size_t tensor_size = DEFAULT_TENSOR_SIZE;
    const std::string engine_path = get_trt_engine_path();

    RT_LOG_INFO("Using TensorRT engine: {}", engine_path);
    ASSERT_TRUE(std::filesystem::exists(engine_path))
            << "TensorRT engine not found at: " << engine_path;

    // Create pipeline via factory
    RT_LOG_INFO("Creating pipeline via factory");
    // example-begin pipeline-graph-mode-1
    const auto spec =
            create_pipeline_spec(tensor_size, engine_path, pipeline::ExecutionMode::Graph);
    auto pipeline = pipeline_factory_->create_pipeline("sample", "test_pipeline_graph", spec);

    // Setup pipeline
    pipeline->setup();
    // example-end pipeline-graph-mode-1
    RT_LOG_INFO("Calling pipeline setup()");

    // Prepare host input data (same as stream test)
    std::vector<float> h_input0(tensor_size);
    std::vector<float> h_input1(tensor_size);

    for (std::size_t i = 0; i < tensor_size; ++i) {
        h_input0[i] = static_cast<float>(i + 1); // [1, 2, 3, ...]
        h_input1[i] = static_cast<float>(i + 2); // [2, 3, 4, ...]
    }

    // Compute expected output
    const std::vector<float> expected_output = compute_expected_output(h_input0, h_input1);

    // Allocate device memory for external inputs
    void *d_input0 = allocate_device_memory(tensor_size * sizeof(float));
    void *d_input1 = allocate_device_memory(tensor_size * sizeof(float));

    // Copy input data to device using stream-aware operations
    copy_to_device(d_input0, h_input0, stream_);
    copy_to_device(d_input1, h_input1, stream_);

    // Create external inputs/outputs
    std::vector<pipeline::PortInfo> external_inputs;
    external_inputs.push_back(create_port_info("input0", d_input0, tensor_size));
    external_inputs.push_back(create_port_info("input1", d_input1, tensor_size));

    // Create placeholder for external output (pipeline will fill this in)
    std::vector<pipeline::PortInfo> external_outputs(1);

    // CRITICAL ORDER:
    // example-begin pipeline-graph-execute-1
    // Step 1: configure_io FIRST (provides tensor addresses, establishes
    // connections)
    RT_LOG_INFO("Calling configure_io() - establishes connections");
    const pipeline::DynamicParams params{.module_specific_params = {}};
    pipeline->configure_io(params, external_inputs, external_outputs, stream_);

    // Step 2: warmup() (loads TRT engine, captures CUDA graph)
    RT_LOG_INFO("Calling warmup() - loads engine, captures graph");
    pipeline->warmup(stream_);

    // Step 3: execute graph (build_graph() called automatically on first
    // execution)
    RT_LOG_INFO("Calling execute_graph()");
    pipeline->execute_graph(stream_);

    // Synchronize
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream_));
    RT_LOG_INFO("Graph execution completed");
    // example-end pipeline-graph-execute-1

    // Get actual output pointer from pipeline
    ASSERT_FALSE(external_outputs.empty());
    ASSERT_FALSE(external_outputs[0].tensors.empty());
    void *d_output = external_outputs[0].tensors[0].device_ptr;

    // Copy output back using stream-aware operation
    std::vector<float> h_output;
    copy_from_device(h_output, d_output, tensor_size, stream_);

    RT_LOG_DEBUG(
            "First 5 actual values: [{}, {}, {}, {}, {}]",
            h_output[0],
            h_output[1],
            h_output[2],
            h_output[3],
            h_output[4]);

    // Validate output
    EXPECT_TRUE(validate_output(h_output, expected_output));

    RT_LOG_INFO("=== GraphExecution_ValidatesCorrectOutput PASSED ===");
}

/**
 * Test graph execution with multiple iterations and different data
 *
 * Verifies:
 * - Graph reusability across iterations
 * - Correct handling of negative values (ReLU clipping)
 * - Dynamic parameter updates
 */
TEST_F(SamplePipelineTest, GraphExecution_MultipleIterations) {
    RT_LOG_INFO("=== GraphExecution_MultipleIterations ===");

    // Configuration
    const std::size_t tensor_size = DEFAULT_TENSOR_SIZE;
    const std::string engine_path = get_trt_engine_path();

    RT_LOG_INFO("Using TensorRT engine: {}", engine_path);
    ASSERT_TRUE(std::filesystem::exists(engine_path))
            << "TensorRT engine not found at: " << engine_path;

    // Create pipeline using factory
    const auto spec =
            create_pipeline_spec(tensor_size, engine_path, pipeline::ExecutionMode::Graph);
    auto pipeline = pipeline_factory_->create_pipeline("sample", "test_pipeline_multi_iter", spec);

    // Setup pipeline
    RT_LOG_INFO("Calling pipeline setup()");
    pipeline->setup();

    // Allocate device memory for external inputs (reused across iterations)
    void *d_input0 = allocate_device_memory(tensor_size * sizeof(float));
    void *d_input1 = allocate_device_memory(tensor_size * sizeof(float));

    // === Iteration 1: Build everything (warmup + graph) ===
    RT_LOG_INFO("--- Iteration 1: Initial graph build ---");

    std::vector<float> h_input0_iter1(tensor_size);
    std::vector<float> h_input1_iter1(tensor_size);

    for (std::size_t i = 0; i < tensor_size; ++i) {
        h_input0_iter1[i] = static_cast<float>(i + 1); // [1, 2, 3, ...]
        h_input1_iter1[i] = static_cast<float>(i + 2); // [2, 3, 4, ...]
    }

    const std::vector<float> expected_output_iter1 =
            compute_expected_output(h_input0_iter1, h_input1_iter1);

    copy_to_device(d_input0, h_input0_iter1, stream_);
    copy_to_device(d_input1, h_input1_iter1, stream_);

    std::vector<pipeline::PortInfo> external_inputs;
    external_inputs.push_back(create_port_info("input0", d_input0, tensor_size));
    external_inputs.push_back(create_port_info("input1", d_input1, tensor_size));

    // Create placeholder for external output (pipeline will fill this in)
    std::vector<pipeline::PortInfo> external_outputs(1);

    // Step 1: configure_io (establishes connections)
    const pipeline::DynamicParams params_iter1{.module_specific_params = {}};
    pipeline->configure_io(params_iter1, external_inputs, external_outputs, stream_);

    // Step 2: warmup (loads TRT engine, captures graph)
    RT_LOG_INFO("Calling warmup() - loads engine, captures graph");
    pipeline->warmup(stream_);

    // Get actual output pointer from pipeline
    ASSERT_FALSE(external_outputs.empty());
    ASSERT_FALSE(external_outputs[0].tensors.empty());
    void *d_output = external_outputs[0].tensors[0].device_ptr;

    // Step 3: execute (build_graph() called automatically on first execution)
    RT_LOG_INFO("Calling execute_graph() - first time");
    pipeline->execute_graph(stream_);
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream_));

    std::vector<float> h_output_iter1;
    copy_from_device(h_output_iter1, d_output, tensor_size, stream_);

    RT_LOG_DEBUG(
            "Iter1 - First 5 actual: [{}, {}, {}, {}, {}]",
            h_output_iter1[0],
            h_output_iter1[1],
            h_output_iter1[2],
            h_output_iter1[3],
            h_output_iter1[4]);
    RT_LOG_DEBUG(
            "Iter1 - First 5 expected: [{}, {}, {}, {}, {}]",
            expected_output_iter1[0],
            expected_output_iter1[1],
            expected_output_iter1[2],
            expected_output_iter1[3],
            expected_output_iter1[4]);

    EXPECT_TRUE(validate_output(h_output_iter1, expected_output_iter1))
            << "Iteration 1 output mismatch";

    // === Iteration 2: Reuse graph (no warmup, no build_graph) ===
    RT_LOG_INFO("--- Iteration 2: Reuse existing graph ---");

    std::vector<float> h_input0_iter2(tensor_size);
    std::vector<float> h_input1_iter2(tensor_size);

    for (std::size_t i = 0; i < tensor_size; ++i) {
        // Create pattern with some negative results after addition
        const auto val = static_cast<float>(i);
        h_input0_iter2[i] = -val;       // [0, -1, -2, -3, ...]
        h_input1_iter2[i] = val * 2.0F; // [0, 2, 4, 6, ...]
                                        // Result after add: [0, 1, 2, 3, ...]
        // Result after ReLU: [0, 1, 2, 3, ...] (no clipping in this pattern)
    }

    // Modify some values to create negative results
    for (std::size_t i = 0; i < 256; ++i) {
        h_input0_iter2[i] = -10.0F; // Large negative
        h_input1_iter2[i] = 5.0F;   // Smaller positive
                                    // Result after add: -5.0
                                    // Result after ReLU: 0.0 (clipped)
    }

    const std::vector<float> expected_output_iter2 =
            compute_expected_output(h_input0_iter2, h_input1_iter2);

    copy_to_device(d_input0, h_input0_iter2, stream_);
    copy_to_device(d_input1, h_input1_iter2, stream_);

    // configure_io only (no warmup - already done, no build_graph - already
    // built)
    const pipeline::DynamicParams params_iter2{.module_specific_params = {}};
    pipeline->configure_io(params_iter2, external_inputs, external_outputs, stream_);

    // execute with same graph
    pipeline->execute_graph(stream_);
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream_));

    std::vector<float> h_output_iter2;
    copy_from_device(h_output_iter2, d_output, tensor_size, stream_);

    RT_LOG_DEBUG(
            "Iter2 - First 5 actual: [{}, {}, {}, {}, {}]",
            h_output_iter2[0],
            h_output_iter2[1],
            h_output_iter2[2],
            h_output_iter2[3],
            h_output_iter2[4]);
    RT_LOG_DEBUG(
            "Iter2 - First 5 expected: [{}, {}, {}, {}, {}]",
            expected_output_iter2[0],
            expected_output_iter2[1],
            expected_output_iter2[2],
            expected_output_iter2[3],
            expected_output_iter2[4]);

    EXPECT_TRUE(validate_output(h_output_iter2, expected_output_iter2))
            << "Iteration 2 output mismatch";

    RT_LOG_INFO("=== GraphExecution_MultipleIterations PASSED ===");
}

} // namespace framework::pipelines::samples::tests

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
