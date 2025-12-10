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

#ifndef FRAMEWORK_CORE_IPIPELINE_HPP
#define FRAMEWORK_CORE_IPIPELINE_HPP

#include <any>
#include <span>
#include <string_view>
#include <vector>

#include <cuda_runtime.h> // for cudaStream_t

#include "pipeline/types.hpp"

namespace framework::pipeline {

/**
 * @class IPipeline
 * @brief Base interface for all processing pipelines.
 *
 * This interface defines the contract that all pipelines must adhere to,
 * including initialization, setup, and execution phases. Pipelines coordinate
 * the execution of multiple modules and manage data flow between them.
 */
class IPipeline {
public:
    /**
     * Virtual destructor.
     */
    virtual ~IPipeline() = default;

    /**
     * Get the pipeline identifier.
     *
     * @return The pipeline ID as a string view
     */
    [[nodiscard]] virtual std::string_view get_pipeline_id() const = 0;

    /**
     * Default constructor.
     */
    IPipeline() = default;

    /**
     * Copy constructor.
     */
    IPipeline(const IPipeline &) = default;

    /**
     * Move constructor.
     */
    IPipeline(IPipeline &&) = default;

    /**
     * Copy assignment operator.
     *
     * @return Reference to this IPipeline
     */
    IPipeline &operator=(const IPipeline &) = default;

    /**
     * Move assignment operator.
     *
     * @return Reference to this IPipeline
     */
    IPipeline &operator=(IPipeline &&) = default;

    /**
     * Perform one-time setup after initialization.
     *
     * This method should create modules, allocate memory, and establish
     * data flow connections between modules.
     */
    virtual void setup() = 0;

    /**
     * Perform one-time warmup and initialization of all modules.
     *
     * This method calls warmup() on all modules after connections are
     * established. It should be called once after the first set_inputs() call
     * and before any execution. This is where expensive one-time operations
     * occur, such as:
     * - Loading models to device memory (TensorRT engines)
     * - Capturing CUDA graphs for graph-mode execution
     *
     * Typical pipeline lifecycle:
     * 1. setup() - allocate memory, create modules
     * 2. configure_io() - establish connections, set external inputs (first call)
     * 3. warmup(..., stream) - one-time initialization (expensive, called once)
     * 4. loop: configure_io() + execute() - process data
     *
     * @param[in] stream CUDA stream to use for warmup operations (passed to
     * modules)
     * @throws std::runtime_error if any module warmup fails
     * @note Default implementation is no-op (pipelines can override if needed)
     * @note Must be called before build_graph() for graph-mode execution
     * @note Should only be called once in the pipeline lifecycle
     * @note TensorRT graph capture requires a non-default stream (cannot use
     * cudaStreamDefault)
     * @note Stream parameter is last to follow standard C++ convention
     */
    virtual void warmup(cudaStream_t stream) { (void)stream; }

    /**
     * Configure the pipeline I/O for the next iteration.
     *
     * This method configures external inputs/outputs and updates dynamic
     * parameters for all modules. It must be called before execute().
     *
     * @param[in] params Dynamic parameters for the current iteration
     * @param[in] external_inputs Span of external input port information
     * @param[out] external_outputs Span of external output port information.
     *                              Caller pre-allocates span storage; pipeline
     *                              writes PortInfo metadata (including
     * device_ptr). Pipeline retains ownership of device memory; caller must not
     * free device pointers. Device pointers remain valid until pipeline
     *                              destruction or next setup() call.
     * @param[in] stream CUDA stream to use for I/O configuration operations
     * (e.g., descriptor copies)
     * @note For the first call, this establishes connections between modules
     * @note After the first call, warmup() should be called before execution
     * @note Stream parameter is last to follow standard C++ convention
     */
    virtual void configure_io(
            const DynamicParams &params,
            std::span<const PortInfo> external_inputs,
            std::span<PortInfo> external_outputs,
            cudaStream_t stream) = 0;

    /**
     * Execute the pipeline using CUDA streams.
     *
     * This method launches all module kernels sequentially using the parameters
     * configured in configure_io(). The pipeline must have been initialized,
     * setup, and configure_io must have been called before this method.
     *
     * @param[in] stream The CUDA stream to execute on
     * @throws std::runtime_error if execution fails
     */
    virtual void execute_stream(cudaStream_t stream) = 0;

    /**
     * Execute the pipeline using CUDA graphs.
     *
     * This method launches the pre-built CUDA graph. The graph must have been
     * created, instantiated, and uploaded during the setup() phase. The pipeline
     * must have been initialized, setup, and configure_io must have been called
     * before this method.
     *
     * @param[in] stream The CUDA stream to launch the graph on
     * @throws std::runtime_error if graph execution is not supported or fails
     */
    virtual void execute_graph(cudaStream_t stream) = 0;

    /**
     * Get the number of external inputs required by this pipeline.
     *
     * @return Number of external input tensors needed
     */
    [[nodiscard]] virtual std::size_t get_num_external_inputs() const = 0;

    /**
     * Get the number of external outputs produced by this pipeline.
     *
     * @return Number of external output tensors produced
     */
    [[nodiscard]] virtual std::size_t get_num_external_outputs() const = 0;

    /**
     * Get pipeline output port information.
     *
     * Provides access to the pipeline's output buffers. This allows external
     * components to access output buffer addresses without executing the pipeline.
     * Typical use case: getting fixed buffer addresses after warmup for zero-copy
     * data passing between pipelines.
     *
     * @return Vector of output PortInfo describing each output port
     * @throws std::logic_error Default implementation throws - must be overridden
     *                          by pipelines that need to expose outputs
     * @note Buffer addresses are typically stable after warmup()
     * @note Not all pipelines need to implement this - only those that expose
     *       outputs for external consumption
     */
    [[nodiscard]] virtual std::vector<PortInfo> get_outputs() const {
        throw std::logic_error("get_outputs() not implemented for this pipeline type");
    }
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IPIPELINE_HPP
