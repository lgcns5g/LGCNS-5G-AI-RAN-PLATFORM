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

#ifndef FRAMEWORK_CORE_TYPES_HPP
#define FRAMEWORK_CORE_TYPES_HPP

#include <any>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include <NamedType/named_type.hpp>

#include <wise_enum.h>

#include "tensor/tensor_info.hpp"

namespace framework::pipeline {

/**
 * @struct DynamicParams
 * @brief Container for dynamic parameters that can change per iteration.
 *
 * This structure holds parameters that may vary for each iteration or unit of
 * work being processed through the pipeline. The module_specific_params field
 * allows modules to receive custom parameters for dynamic updates like matrix
 * dimensions, kernel launch configurations, or other per-iteration variations.
 */
struct DynamicParams final {
    std::any module_specific_params; //!< Module-specific parameters for dynamic
                                     //!< updates (e.g., matrix dimensions, kernel
                                     //!< configs)
};

/**
 * Memory allocation requirements for a module
 *
 * Follows cuBB's pattern with static/dynamic kernel descriptors and device
 * tensor allocation.
 */
struct ModuleMemoryRequirements final {
    static constexpr std::size_t DEFAULT_ALIGNMENT = 128; //!< Memory alignment requirement in bytes
    std::size_t static_kernel_descriptor_bytes{0}; //!< Size of static kernel parameters (set once)
    std::size_t dynamic_kernel_descriptor_bytes{
            0};                         //!< Size of dynamic kernel parameters (updated per frame)
    std::size_t device_tensor_bytes{0}; //!< Size of module's device tensor allocation (for any use)
    std::size_t alignment{DEFAULT_ALIGNMENT}; //!< Memory alignment requirement
};

/**
 * Stream output operator for ModuleMemoryRequirements
 *
 * @param[in,out] oss Output stream to write to
 * @param[in] req ModuleMemoryRequirements to output
 * @return Reference to the output stream
 */
inline std::ostream &operator<<(std::ostream &oss, const ModuleMemoryRequirements &req) {
    oss << "ModuleMemoryRequirements: " << '\n';
    oss << "  static_kernel_descriptor_bytes: " << req.static_kernel_descriptor_bytes << '\n';
    oss << "  dynamic_kernel_descriptor_bytes: " << req.dynamic_kernel_descriptor_bytes << '\n';
    oss << "  device_tensor_bytes: " << req.device_tensor_bytes << '\n';
    return oss;
}

/**
 * Memory slice assigned to a module
 *
 * Contains pointers to memory regions allocated by the pipeline.
 * Follows cuBB pattern: CPU/GPU descriptor pairs + device tensor slice.
 */
struct ModuleMemorySlice final {
    // Static kernel descriptor slices (both CPU and GPU)
    std::byte *static_kernel_descriptor_cpu_ptr{
            nullptr}; //!< Pinned memory for static kernel parameters (CPU)
    std::byte *static_kernel_descriptor_gpu_ptr{
            nullptr}; //!< Device memory for static kernel parameters (GPU)

    // Dynamic kernel descriptor slices (both CPU and GPU)
    std::byte *dynamic_kernel_descriptor_cpu_ptr{
            nullptr}; //!< Pinned memory for dynamic kernel parameters (CPU)
    std::byte *dynamic_kernel_descriptor_gpu_ptr{
            nullptr}; //!< Device memory for dynamic kernel parameters (GPU)

    // Device tensor slice (device memory) - module uses this for any purpose
    std::byte *device_tensor_ptr{nullptr}; //!< Device memory for module's tensor
                                           //!< data (intermediate/output/scratch)

    // Sizes of allocated slices
    std::size_t static_kernel_descriptor_bytes{
            0}; //!< Size of static descriptor slices (same for CPU and GPU)
    std::size_t dynamic_kernel_descriptor_bytes{
            0}; //!< Size of dynamic descriptor slices (same for CPU and GPU)
    std::size_t device_tensor_bytes{0}; //!< Size of device tensor slice
};

/**
 * Stream output operator for ModuleMemorySlice
 *
 * @param[in,out] oss Output stream to write to
 * @param[in] req ModuleMemorySlice to output
 * @return Reference to the output stream
 */
inline std::ostream &operator<<(std::ostream &oss, const ModuleMemorySlice &req) {
    oss << "ModuleMemorySlice: " << '\n';
    oss << "  static_kernel_descriptor_bytes: " << req.static_kernel_descriptor_bytes << '\n';
    oss << "  dynamic_kernel_descriptor_bytes: " << req.dynamic_kernel_descriptor_bytes << '\n';
    oss << "  device_tensor_bytes: " << req.device_tensor_bytes << '\n';
    return oss;
}

/**
 * @struct DeviceTensor
 * @brief Represents a tensor with its device memory location and metadata
 *
 * Each device tensor contains the device pointer to tensor data along with
 * its associated metadata (dimensions, type, etc.). This allows ports to
 * contain multiple tensors, each with its own device memory address.
 */
struct DeviceTensor final {
    void *device_ptr{nullptr};      //!< Device pointer to tensor data
    tensor::TensorInfo tensor_info; //!< Tensor metadata (dimensions, type, size)
};

/**
 * Mode for data transfer between modules
 *
 * Specifies whether data copying is required between connected modules.
 */
enum class ConnectionCopyMode {
    Copy,    //!< Allocate buffer and copy data (cudaMemcpy)
    ZeroCopy //!< Use upstream address directly (no copy)
};

/**
 * Pipeline execution mode determining addressing and memory allocation strategy
 *
 * The execution mode is a static configuration set at pipeline construction
 * time and cannot change during the pipeline's lifetime. It determines:
 * - Memory allocation strategy (fixed vs. dynamic addressing)
 * - Zero-copy optimization possibilities
 * - TRT engine configuration (graph capture vs. stream mode)
 */
enum class ExecutionMode {
    Stream, //!< Stream mode: flexible addressing, supports dynamic
            //!< set_tensor_address() per iteration, enables zero-copy with
            //!< dynamic upstream addresses
    Graph   //!< Graph mode: fixed addressing required for CUDA graph
            //!< capture/replay, zero-copy only possible with fixed upstream
            //!< addresses
};

} // namespace framework::pipeline

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::pipeline::ConnectionCopyMode, Copy, ZeroCopy)
WISE_ENUM_ADAPT(framework::pipeline::ExecutionMode, Stream, Graph)

namespace framework::pipeline {

/**
 * @struct InputPortMemoryCharacteristics
 * @brief Memory characteristics for a module INPUT port (for zero-copy
 * optimization)
 *
 * Describes what an input port requires from its upstream connection to enable
 * zero-copy. Used by get_input_memory_characteristics() to declare input
 * requirements.
 */
struct InputPortMemoryCharacteristics final {
    // clang-format off
  /**
   * Whether this input port requires fixed upstream addresses for zero-copy optimization.
   *
   * This field determines if the module can zero-copy with any upstream or only with fixed ones:
   *
   * false = Can zero-copy with ANY upstream (fixed OR dynamic addresses)
   *         - Module is flexible: accepts any address, even if it changes per iteration
   *         - Example: TRT engine in STREAM MODE (uses set_tensor_address() per iteration)
   *         - Example: CUDA kernel with dynamic descriptors (pointer updated per iteration)
   *         - Zero-copy: ALWAYS possible! Just use whatever upstream provides
   *                      - If upstream fixed: same address every iteration (no copy)
   *                      - If upstream dynamic: different address each iteration (still no copy!)
   *
   * true = Can ONLY zero-copy if upstream provides fixed addresses
   *        - Module MUST have fixed address BEFORE warmup()
   *        - Reason: Address needed for CUDA graph capture or other pre-warmup operations
   *        - Example: TRT engine in GRAPH MODE (address captured during graph warmup)
   *        - Zero-copy: Only possible if upstream provides_fixed_address_for_zero_copy=true
   *                     (downstream uses upstream's fixed address, no allocation needed)
   *        - If upstream provides_fixed_address_for_zero_copy=false: MUST allocate + cudaMemcpy each iteration
   */
  bool requires_fixed_address_for_zero_copy{false};
    // clang-format on
};

/**
 * @struct OutputPortMemoryCharacteristics
 * @brief Memory characteristics for a module OUTPUT port (for zero-copy
 * optimization)
 *
 * Describes what an output port provides to its downstream connections to
 * enable zero-copy. Used by get_output_memory_characteristics() to declare
 * output capabilities.
 */
struct OutputPortMemoryCharacteristics final {
    /**
     * Whether this output port provides fixed device addresses (for zero-copy
     * optimization).
     *
     * true = Address allocated once in setup_memory(), never changes
     *        - Enables downstream zero-copy (if downstream can accept it)
     *        - This is the typical case for most modules
     * false = Address may change per iteration
     *         - Examples: external inputs, ping-pong buffers
     *         - Limits zero-copy to flexible consumers only
     */
    bool provides_fixed_address_for_zero_copy{true};
};

/**
 * Helper function to determine if zero-copy is possible for a connection.
 *
 * Zero-copy decision matrix:
 *
 * | Upstream Provides Fixed | Downstream Requires Fixed | Zero-Copy? |
 * Explanation |
 * |-------------------------|---------------------------|------------|--------------------------------------------------------------------------------|
 * | true                    | true                      | YES        | Graph
 * mode: downstream uses upstream's fixed address (no allocation)          | |
 * true                    | false                     | YES        | Stream
 * mode: downstream uses upstream's fixed address each tick                | |
 * false                   | false                     | YES        | Stream
 * mode: downstream uses upstream's changing address each tick             | |
 * false                   | true                      | NO         |
 * Incompatible: downstream needs fixed address but upstream changes → must copy
 * |
 *
 * The ONLY case requiring allocation + copy: upstream dynamic AND downstream
 * requires fixed (e.g., graph mode with changing external inputs)
 *
 * @param[in] upstream Output characteristics from the producing module
 * @param[in] downstream Input characteristics from the consuming module
 * @return true if zero-copy is possible, false if copy is required
 */
[[nodiscard]] inline bool can_zero_copy(
        const OutputPortMemoryCharacteristics &upstream,
        const InputPortMemoryCharacteristics &downstream) {
    return !downstream.requires_fixed_address_for_zero_copy ||
           upstream.provides_fixed_address_for_zero_copy;
}

/**
 * Information about a module's input or output port
 *
 * This structure represents a named port containing one or more tensors,
 * each with its own device memory pointer and tensor metadata. Used for
 * module interconnection in pipelines.
 *
 * Usage contexts:
 * - ModuleRouter: Topology definition and connection routing
 * - IModule::set_inputs(): Receives port info and extracts device pointers
 * - IModule::get_outputs(): Returns port info for routing to other modules
 *
 * Note: Modules typically extract and cache just the device pointers (void*)
 * from PortInfo for execution, not the entire structure.
 */
struct PortInfo final {
    std::string name;                  //!< Port name (e.g., "input0", "matrixA", "output0")
    std::vector<DeviceTensor> tensors; //!< Vector of device tensors for this port
    // Note: Memory characteristics (InputPortMemoryCharacteristics,
    // OutputPortMemoryCharacteristics) are queried during pipeline setup via
    // IModule::get_input_memory_characteristics() and
    // IModule::get_output_memory_characteristics(), not carried in runtime
    // PortInfo
};

/**
 * @struct ModuleCreationInfo
 * @brief Information needed to create a module instance
 *
 * Contains all information needed to create and initialize a module
 * through the factory pattern. This is the underlying data for ModuleSpec.
 */
struct ModuleCreationInfo final {
    std::string module_type; //!< Module type identifier (e.g., "gemm", "relu")
    std::string instance_id; //!< Unique instance identifier for this module
    std::any init_params;    //!< Type-erased initialization parameters
};

/**
 * Strong type for module specifications using NamedType
 *
 * Provides type safety and clearer intent when working with module
 * specifications in pipeline configurations.
 */
using ModuleSpec = fluent::NamedType<ModuleCreationInfo, struct ModuleSpecTag>;

/**
 * @struct PipelineModuleConfig
 * @brief Configuration for creating all modules in a pipeline
 *
 * Contains a list of module creation specifications that define what modules
 * a pipeline should create and in what order. The order in the vector
 * determines the execution order of the modules.
 *
 */
struct PipelineModuleConfig final {
    std::vector<ModuleSpec> modules; //!< Ordered list of modules to create
};

/**
 * @struct PortConnection
 * @brief Represents a connection between module ports
 */
struct PortConnection final {
    std::string source_module; //!< Source module ID
    std::string source_port;   //!< Source port name (e.g., "output0")
    std::string target_module; //!< Target module ID
    std::string target_port;   //!< Target port name (e.g., "input0")
};

/**
 * @struct PipelineSpec
 * @brief Complete specification for constructing a pipeline
 *
 * Contains all modules, connections, and external I/O specifications
 * needed to construct a complete pipeline through the factory.
 */
struct PipelineSpec final {
    std::string pipeline_name;                 //!< Pipeline name for identification
    std::vector<ModuleSpec> modules;           //!< Ordered list of modules to create
    std::vector<PortConnection> connections;   //!< Module interconnections
    std::vector<std::string> external_inputs;  //!< External input identifiers
    std::vector<std::string> external_outputs; //!< External output identifiers
    ExecutionMode execution_mode{
            ExecutionMode::Graph}; //!< Execution mode (Graph or Stream), default
                                   //!< Graph for backward compatibility
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_TYPES_HPP
