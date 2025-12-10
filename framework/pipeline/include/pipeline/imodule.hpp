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

#ifndef FRAMEWORK_CORE_IMODULE_HPP
#define FRAMEWORK_CORE_IMODULE_HPP

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"

namespace framework::pipeline {

// Forward declaration
struct ModuleMemorySlice;
class IGraphNodeProvider;
class IStreamExecutor;

/**
 * @class IModule
 * @brief Base interface for all algorithm modules.
 *
 * This interface defines the contract that all modules in the processing
 * pipeline must adhere to, including initialization, tensor information, and
 * capability interfaces.
 */
class IModule {
public:
    /**
     * Default constructor.
     */
    IModule() = default;

    /**
     * Copy constructor.
     */
    IModule(const IModule &) = default;

    /**
     * Move constructor.
     */
    IModule(IModule &&) = default;

    /**
     * Copy assignment operator.
     * @return Reference to this object
     */
    IModule &operator=(const IModule &) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    IModule &operator=(IModule &&) = default;

    /**
     * Virtual destructor.
     */
    virtual ~IModule() = default;

    /**
     * Get the type identifier of the module.
     *
     * @return The type ID as a string_view
     */
    [[nodiscard]] virtual std::string_view get_type_id() const = 0;

    /**
     * Get the instance identifier of the module.
     *
     * @return The instance ID as a string_view
     */
    [[nodiscard]] virtual std::string_view get_instance_id() const = 0;

    /**
     * Perform one-time setup after memory allocation.
     *
     * @param[in] memory_slice Memory slice allocated by PipelineMemoryManager
     */
    virtual void setup_memory(const ModuleMemorySlice &memory_slice) = 0;

    /**
     * Get the input tensor information for a specified port.
     *
     * @param[in] port_name The name of the input port
     * @return Vector of tensor information for all tensors on this port
     */
    [[nodiscard]] virtual std::vector<tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const = 0;

    /**
     * Get the output tensor information for a specified port.
     *
     * @param[in] port_name The name of the output port
     * @return Vector of tensor information for all tensors on this port
     */
    [[nodiscard]] virtual std::vector<tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const = 0;

    /**
     * Get the names of all input ports.
     *
     * @return A vector of port names
     */
    [[nodiscard]] virtual std::vector<std::string> get_input_port_names() const = 0;

    /**
     * Get the names of all output ports.
     *
     * @return A vector of port names
     */
    [[nodiscard]] virtual std::vector<std::string> get_output_port_names() const = 0;

    /**
     * Set input connections for the module.
     *
     * This method is called by the pipeline to connect input ports to their data
     * sources. The module should validate that all required inputs are provided
     * and that the port names match expected inputs.
     *
     * @param[in] inputs Span of port information with device pointers to input
     * data
     * @throws std::invalid_argument if required inputs are missing or port names
     * don't match
     * @note This is typically called during pipeline configuration after all
     * modules are created
     */
    virtual void set_inputs(std::span<const PortInfo> inputs) = 0;

    /**
     * Get output port information.
     *
     * Returns information about all output ports including their device pointers
     * and tensor metadata. This is used by the pipeline to route data between
     * modules.
     *
     * @return Vector of port information for all outputs
     * @note Device pointers are only valid after setup_memory() has been called
     */
    [[nodiscard]] virtual std::vector<PortInfo> get_outputs() const = 0;

    /**
     * Perform one-time warmup and initialization after connections are
     * established.
     *
     * This method is called once after set_inputs() to perform any expensive
     * one-time initialization that requires knowledge of input/output
     * connections. Examples include:
     * - Loading machine learning models to device memory (TensorRT, PyTorch)
     * - Capturing CUDA graphs for graph-mode execution
     * - Allocating and initializing lookup tables
     *
     * Typical pipeline lifecycle:
     * 1. setup() - allocate memory, initialize data structures
     * 2. set_inputs() - establish data flow connections (lightweight)
     * 3. warmup(stream) - one-time initialization (expensive, called once)
     * 4. loop: configure_io() + execute() - process data
     *
     * @param[in] stream CUDA stream to use for warmup operations (e.g., graph
     * capture)
     * @throws std::runtime_error if warmup fails
     * @note Default implementation is no-op (most modules don't need warmup)
     * @note This should only be called once after the first set_inputs() call
     * @note Warmup must complete before build_graph() for graph-mode execution
     * @note TensorRT graph capture requires a non-default stream (cannot use
     * cudaStreamDefault)
     */
    virtual void warmup([[maybe_unused]] cudaStream_t stream) {}

    /**
     * Configure I/O for the current iteration.
     *
     * This method is called before execute() to update any parameters that
     * change per execution. The module should use this opportunity to update
     * internal state, kernel parameters, or any iteration-specific configuration.
     * The subsequent execute() call will use this prepared state.
     *
     * Execution flow for both stream and graph modes:
     * 1. configure_io(params, stream) - prepare internal state
     * 2. execute(stream) - launch work using prepared state
     *
     * @param[in] params Dynamic parameters for the current iteration
     * @param[in] stream CUDA stream for async operations during configuration
     * @note This is called after warmup() has completed one-time initialization
     * @note For graph mode, this is called before each graph launch to update
     * captured parameters
     */
    virtual void configure_io(const DynamicParams &params, cudaStream_t stream) = 0;

    /**
     * Get memory characteristics for input ports (for zero-copy optimization).
     *
     * Allows modules to declare whether they require fixed input addresses
     * for zero-copy. Called during pipeline setup to optimize memory allocation
     * strategy.
     *
     * IMPORTANT: Only the requires_fixed_address_for_zero_copy field is used.
     *
     * @param[in] port_name Input port name
     * @return Input port memory characteristics
     * @note Default implementation: flexible (doesn't require fixed addresses)
     */
    [[nodiscard]] virtual InputPortMemoryCharacteristics
    get_input_memory_characteristics([[maybe_unused]] std::string_view port_name) const {
        return InputPortMemoryCharacteristics{};
    }

    /**
     * Get memory characteristics for output ports (for zero-copy optimization).
     *
     * Allows modules to declare whether they provide fixed addresses for
     * outputs. Called during pipeline setup to optimize memory allocation
     * strategy.
     *
     * IMPORTANT: Only the provides_fixed_address_for_zero_copy field is used.
     *
     * @param[in] port_name Output port name
     * @return Output port memory characteristics
     * @note Default implementation: provides fixed addresses (typical case)
     */
    [[nodiscard]] virtual OutputPortMemoryCharacteristics
    get_output_memory_characteristics([[maybe_unused]] std::string_view port_name) const {
        return OutputPortMemoryCharacteristics{};
    }

    /**
     * Configure connection copy mode for an input port (for zero-copy
     * optimization).
     *
     * Called by the pipeline during setup() to inform the module about input
     * characteristics. This allows modules to optimize memory
     * allocation in get_requirements():
     * - ConnectionCopyMode::Copy: Module must allocate input buffer and copy data
     * - ConnectionCopyMode::ZeroCopy: Module can use input address directly
     * (skip allocation)
     *
     * Typical flow:
     * 1. Pipeline analyzes input/output memory characteristics
     * 2. Pipeline calls set_connection_copy_mode() to configure each input port
     * 3. Module's get_requirements() uses this info to calculate memory needs
     * 4. Module allocates only what it needs (skips buffers for zero-copy inputs)
     *
     * @param[in] port_name Input port name
     * @param[in] mode Connection copy mode (Copy or ZeroCopy)
     * @note Default implementation is no-op (modules that don't support zero-copy
     * inputs ignore)
     * @note Must be called before get_requirements() for memory optimization to
     * work
     */
    virtual void set_connection_copy_mode(
            [[maybe_unused]] std::string_view port_name, [[maybe_unused]] ConnectionCopyMode mode) {
    }

    /**
     * @brief Cast the module to a specific type, using dynamic_cast
     * @tparam T The type to cast to
     * @return The casted module
     */
    template <typename T>
    [[nodiscard]] std::add_pointer_t<const std::remove_reference_t<T>> as_type() const {
        return dynamic_cast<std::add_pointer_t<const std::remove_reference_t<T>>>(this);
    }

    /**
     * @brief Cast the module to a IGraphNodeProvider
     * @return The casted module
     */
    virtual IGraphNodeProvider *as_graph_node_provider() = 0;

    /**
     * @brief Cast the module to a IStreamExecutor
     * @return The casted module
     */
    virtual IStreamExecutor *as_stream_executor() = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IMODULE_HPP
