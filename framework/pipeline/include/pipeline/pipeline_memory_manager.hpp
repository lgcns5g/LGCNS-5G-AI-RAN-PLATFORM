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

#ifndef FRAMEWORK_CORE_PIPELINE_MEMORY_MANAGER_HPP
#define FRAMEWORK_CORE_PIPELINE_MEMORY_MANAGER_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pipeline/types.hpp"
#include "tensor/tensor_arena.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

// Forward declarations
class IModule;

/**
 * Align offset to specified boundary
 *
 * @param[in] offset Current offset
 * @param[in] alignment Alignment requirement
 * @return Aligned offset
 */
[[nodiscard]] std::size_t align_memory_offset(std::size_t offset, std::size_t alignment);

/**
 * Calculate total memory requirements for a collection of modules
 *
 * This function simulates the allocation process to account for alignment
 * padding between modules. It ensures the calculated total matches what will
 * actually be needed during allocation.
 *
 * @param[in] module_requirements Vector of memory requirements from each module
 * @return Total memory requirements including alignment padding
 */
[[nodiscard]] ModuleMemoryRequirements calculate_pipeline_memory_requirements(
        const std::vector<ModuleMemoryRequirements> &module_requirements);

/**
 * Calculate total memory requirements for a collection of modules
 *
 * This overload extracts requirements from the modules directly.
 *
 * @param[in] modules Vector of modules to calculate requirements for
 * @return Total memory requirements including alignment padding
 * @throws std::runtime_error if any module doesn't implement
 * IAllocationInfoProvider
 */
[[nodiscard]] ModuleMemoryRequirements
calculate_pipeline_memory_requirements(const std::vector<IModule *> &modules);

/**
 * Pipeline-level memory manager
 *
 * Follows cuBB pattern where pipeline owns large contiguous memory arenas and
 * assigns memory slices to modules. Five-tier allocation:
 * 1. Static kernel descriptors CPU: Small pinned memory (like cuBB's
 * m_kernelStatDescr CPU)
 * 2. Static kernel descriptors GPU: Small device memory (like cuBB's
 * m_kernelStatDescr GPU)
 * 3. Dynamic kernel descriptors CPU: Small pinned memory (like cuBB's
 * m_kernelDynDescr CPU)
 * 4. Dynamic kernel descriptors GPU: Small device memory (like cuBB's
 * m_kernelDynDescr GPU)
 * 5. Device tensors: Large device memory (like cuBB's m_LinearAlloc)
 *
 * Note: Pipeline only allocates memory slices. Each module is responsible for
 * copying its own descriptors from CPU to GPU when ready.
 */
class PipelineMemoryManager final {
public:
    /**
     * Constructor
     *
     * @param[in] total_static_kernel_descriptor_bytes Total memory for all static
     * kernel descriptors (both CPU and GPU)
     * @param[in] total_dynamic_kernel_descriptor_bytes Total memory for all
     * dynamic kernel descriptors (both CPU and GPU)
     * @param[in] total_device_tensor_bytes Total device memory for all module
     * tensor allocations
     */
    PipelineMemoryManager(
            std::size_t total_static_kernel_descriptor_bytes,
            std::size_t total_dynamic_kernel_descriptor_bytes,
            std::size_t total_device_tensor_bytes);

    /**
     * Factory method to create a PipelineMemoryManager sized for the given
     * modules
     *
     * This method calculates the total memory requirements for all modules
     * (including alignment) and creates a PipelineMemoryManager with the
     * appropriate arena sizes.
     *
     * @param[in] modules Vector of modules that will use this memory manager
     * @return Unique pointer to a properly sized PipelineMemoryManager
     * @throws std::runtime_error if any module doesn't implement
     * IAllocationInfoProvider
     */
    [[nodiscard]] static std::unique_ptr<PipelineMemoryManager>
    create_for_modules(const std::vector<IModule *> &modules);

    // Non-copyable, movable
    PipelineMemoryManager(const PipelineMemoryManager &) = delete;
    PipelineMemoryManager &operator=(const PipelineMemoryManager &) = delete;

    /**
     * Move constructor.
     */
    PipelineMemoryManager(PipelineMemoryManager &&) = default;

    /**
     * Move assignment operator.
     *
     * @return Reference to this object
     */
    PipelineMemoryManager &operator=(PipelineMemoryManager &&) = default;

    /**
     * Destructor.
     */
    ~PipelineMemoryManager() = default;

    /**
     * Allocate memory slice for a module
     *
     * @param[in] module_id Unique identifier for the module
     * @param[in] requirements Memory requirements for the module
     * @return Memory slice assigned to the module
     * @throws std::runtime_error if allocation fails or exceeds arena capacity
     */
    [[nodiscard]] ModuleMemorySlice
    allocate_module_slice(std::string_view module_id, const ModuleMemoryRequirements &requirements);

    /**
     * Get memory slice for a previously allocated module
     *
     * @param[in] module_id Module identifier
     * @return Memory slice for the module
     * @throws std::runtime_error if module not found
     */
    [[nodiscard]] const ModuleMemorySlice &get_module_slice(std::string_view module_id) const;

    /**
     * Get total memory usage statistics
     *
     * @return Current memory usage across all arenas
     */
    struct MemoryUsage final {
        std::size_t static_kernel_descriptor_used{
                0}; //!< Used static kernel descriptor memory (CPU + GPU)
        std::size_t dynamic_kernel_descriptor_used{
                0};                        //!< Used dynamic kernel descriptor memory (CPU + GPU)
        std::size_t device_tensor_used{0}; //!< Used device tensor memory
        std::size_t static_kernel_descriptor_total{
                0}; //!< Total static kernel descriptor memory (CPU + GPU)
        std::size_t dynamic_kernel_descriptor_total{
                0};                         //!< Total dynamic kernel descriptor memory (CPU + GPU)
        std::size_t device_tensor_total{0}; //!< Total device tensor memory
    };

    /**
     * Get current memory usage statistics.
     *
     * @return Memory usage information across all arenas
     */
    [[nodiscard]] MemoryUsage get_memory_usage() const;

    /**
     * Pre-allocate memory slices for all modules
     *
     * This method calculates requirements and allocates memory slices for all
     * modules in one atomic operation, ensuring consistency between calculation
     * and allocation order. After calling this method, get_module_slice() can be
     * used to retrieve pre-computed slices.
     *
     * @param[in] modules Vector of modules that need memory allocation
     * @throws std::runtime_error if any module doesn't implement
     * IAllocationInfoProvider
     * @throws std::runtime_error if allocation fails
     */
    void allocate_all_module_slices(const std::vector<IModule *> &modules);

    /**
     * Copy all static kernel descriptors to device in one bulk operation
     *
     * Copies the entire contiguous static descriptor region (all modules) from
     * CPU pinned memory to GPU device memory. This should be called once at the
     * end of pipeline initialization after all modules have initialized their
     * static parameters.
     *
     * @param[in] stream CUDA stream for async copy operation
     * @throws std::runtime_error if copy fails or pointers are null
     */
    void copy_all_static_descriptors_to_device(cudaStream_t stream) const;

    /**
     * Copy all dynamic kernel descriptors to device in one bulk operation
     *
     * Copies the entire contiguous dynamic descriptor region (all modules) from
     * CPU pinned memory to GPU device memory. This should be called every
     * iteration in configure_io() after all modules have updated their dynamic
     * parameters.
     *
     * @param[in] stream CUDA stream for async copy operation
     * @throws std::runtime_error if copy fails or pointers are null
     */
    void copy_all_dynamic_descriptors_to_device(cudaStream_t stream) const;

private:
    // Memory arenas following cuBB pattern (CPU + GPU for descriptors)
    tensor::TensorArena static_kernel_descriptor_arena_cpu_;  //!< Pinned memory for
                                                              //!< static kernel params
                                                              //!< (CPU)
    tensor::TensorArena static_kernel_descriptor_arena_gpu_;  //!< Device memory for
                                                              //!< static kernel params
                                                              //!< (GPU)
    tensor::TensorArena dynamic_kernel_descriptor_arena_cpu_; //!< Pinned memory for dynamic
                                                              //!< kernel params (CPU)
    tensor::TensorArena dynamic_kernel_descriptor_arena_gpu_; //!< Device memory for dynamic
                                                              //!< kernel params (GPU)
    tensor::TensorArena device_tensor_arena_;                 //!< Device memory for module tensors

    // Current allocation offsets (like cuBB's arena allocation tracking)
    std::size_t static_kernel_descriptor_offset_{0}; //!< Current offset in static descriptor arenas
    std::size_t dynamic_kernel_descriptor_offset_{
            0};                           //!< Current offset in dynamic descriptor arenas
    std::size_t device_tensor_offset_{0}; //!< Current offset in device tensor
                                          //!< arena

    // Module memory slice registry with heterogeneous lookup support
    std::unordered_map<
            std::string,
            ModuleMemorySlice,
            utils::TransparentStringHash,
            std::equal_to<>>
            module_slices_;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_PIPELINE_MEMORY_MANAGER_HPP
