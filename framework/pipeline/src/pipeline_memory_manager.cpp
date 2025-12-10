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

#include <bit>
#include <cstddef>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda_runtime_api.h>

#include "log/rt_log_macros.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/pipeline_memory_manager.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_arena.hpp"
#include "utils/core_log.hpp"
#include "utils/error_macros.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

std::size_t align_memory_offset(const std::size_t offset, const std::size_t alignment) {
    // Validate that alignment is a power of 2
    FRAMEWORK_NV_THROW_IF(
            alignment == 0 || !std::has_single_bit(alignment),
            std::runtime_error,
            std::format(
                    "Invalid alignment value: {}. Alignment must be a "
                    "non-zero power-of-two.",
                    alignment));
    return (offset + alignment - 1) & ~(alignment - 1);
}

PipelineMemoryManager::PipelineMemoryManager(
        const std::size_t total_static_kernel_descriptor_bytes,
        const std::size_t total_dynamic_kernel_descriptor_bytes,
        const std::size_t total_device_tensor_bytes)
        : static_kernel_descriptor_arena_cpu_(
                  total_static_kernel_descriptor_bytes, tensor::MemoryType::HostPinned),
          static_kernel_descriptor_arena_gpu_(
                  total_static_kernel_descriptor_bytes, tensor::MemoryType::Device),
          dynamic_kernel_descriptor_arena_cpu_(
                  total_dynamic_kernel_descriptor_bytes, tensor::MemoryType::HostPinned),
          dynamic_kernel_descriptor_arena_gpu_(
                  total_dynamic_kernel_descriptor_bytes, tensor::MemoryType::Device),
          device_tensor_arena_(total_device_tensor_bytes, tensor::MemoryType::Device) {

    RT_LOGC_INFO(
            utils::Core::CorePipeline,
            "Created PipelineMemoryManager: static={} dynamic={} tensor={} "
            "bytes",
            total_static_kernel_descriptor_bytes,
            total_dynamic_kernel_descriptor_bytes,
            total_device_tensor_bytes);
}

ModuleMemorySlice PipelineMemoryManager::allocate_module_slice(
        std::string_view module_id, const ModuleMemoryRequirements &requirements) {

    RT_LOGC_DEBUG(utils::Core::CorePipeline, "Allocating memory slice for module: {}", module_id);
    RT_LOGC_DEBUG(
            utils::Core::CoreCudaRuntime,
            "Requirements: static={} dynamic={} device={} alignment={}",
            requirements.static_kernel_descriptor_bytes,
            requirements.dynamic_kernel_descriptor_bytes,
            requirements.device_tensor_bytes,
            requirements.alignment);

    // Check if module already exists (C++20 heterogeneous lookup)
    if (module_slices_.contains(module_id)) {
        const std::string error_msg =
                std::format("Module '{}' already has allocated memory slice", module_id);
        RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Validate alignment is a non-zero power-of-two
    if (requirements.alignment == 0 || !std::has_single_bit(requirements.alignment)) {
        const std::string error_msg = std::format(
                "Module '{}' has invalid alignment value: {}. Alignment "
                "must be a non-zero power-of-two.",
                module_id,
                requirements.alignment);
        RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Align offsets
    const std::size_t aligned_static_offset =
            align_memory_offset(static_kernel_descriptor_offset_, requirements.alignment);
    const std::size_t aligned_dynamic_offset =
            align_memory_offset(dynamic_kernel_descriptor_offset_, requirements.alignment);
    const std::size_t aligned_tensor_offset =
            align_memory_offset(device_tensor_offset_, requirements.alignment);

    RT_LOGC_DEBUG(
            utils::Core::CoreCudaRuntime,
            "Aligned offsets: static={} dynamic={} tensor={}",
            aligned_static_offset,
            aligned_dynamic_offset,
            aligned_tensor_offset);

    // Check bounds
    if (aligned_static_offset + requirements.static_kernel_descriptor_bytes >
        static_kernel_descriptor_arena_cpu_.total_bytes()) {
        const std::string error_msg = std::format(
                "Static kernel descriptor allocation for module '{}' exceeds arena "
                "capacity: {} + {} > {}",
                module_id,
                aligned_static_offset,
                requirements.static_kernel_descriptor_bytes,
                static_kernel_descriptor_arena_cpu_.total_bytes());
        RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    if (aligned_dynamic_offset + requirements.dynamic_kernel_descriptor_bytes >
        dynamic_kernel_descriptor_arena_cpu_.total_bytes()) {
        const std::string error_msg = std::format(
                "Dynamic kernel descriptor allocation for module '{}' exceeds arena "
                "capacity: {} + {} > {}",
                module_id,
                aligned_dynamic_offset,
                requirements.dynamic_kernel_descriptor_bytes,
                dynamic_kernel_descriptor_arena_cpu_.total_bytes());
        RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    if (aligned_tensor_offset + requirements.device_tensor_bytes >
        device_tensor_arena_.total_bytes()) {
        const std::string error_msg = std::format(
                "Device tensor allocation for module '{}' exceeds arena capacity: {} + "
                "{} > {}",
                module_id,
                aligned_tensor_offset,
                requirements.device_tensor_bytes,
                device_tensor_arena_.total_bytes());
        RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Allocate memory slices
    ModuleMemorySlice slice{};

    if (requirements.static_kernel_descriptor_bytes > 0) {
        slice.static_kernel_descriptor_cpu_ptr =
                static_kernel_descriptor_arena_cpu_.allocate_at<std::byte>(aligned_static_offset);
        slice.static_kernel_descriptor_gpu_ptr =
                static_kernel_descriptor_arena_gpu_.allocate_at<std::byte>(aligned_static_offset);
        slice.static_kernel_descriptor_bytes = requirements.static_kernel_descriptor_bytes;
        RT_LOGC_DEBUG(
                utils::Core::CoreCudaRuntime,
                "Static descriptors: CPU={}, GPU={}",
                static_cast<void *>(slice.static_kernel_descriptor_cpu_ptr),
                static_cast<void *>(slice.static_kernel_descriptor_gpu_ptr));
    }

    if (requirements.dynamic_kernel_descriptor_bytes > 0) {
        slice.dynamic_kernel_descriptor_cpu_ptr =
                dynamic_kernel_descriptor_arena_cpu_.allocate_at<std::byte>(aligned_dynamic_offset);
        slice.dynamic_kernel_descriptor_gpu_ptr =
                dynamic_kernel_descriptor_arena_gpu_.allocate_at<std::byte>(aligned_dynamic_offset);
        slice.dynamic_kernel_descriptor_bytes = requirements.dynamic_kernel_descriptor_bytes;
        RT_LOGC_DEBUG(
                utils::Core::CoreCudaRuntime,
                "Dynamic descriptors: CPU={}, GPU={}",
                static_cast<void *>(slice.dynamic_kernel_descriptor_cpu_ptr),
                static_cast<void *>(slice.dynamic_kernel_descriptor_gpu_ptr));
    }

    if (requirements.device_tensor_bytes > 0) {
        slice.device_tensor_ptr =
                device_tensor_arena_.allocate_at<std::byte>(aligned_tensor_offset);
        slice.device_tensor_bytes = requirements.device_tensor_bytes;
        RT_LOGC_DEBUG(
                utils::Core::CoreCudaRuntime,
                "Device tensor: {}",
                static_cast<void *>(slice.device_tensor_ptr));
    }

    // Update offsets
    static_kernel_descriptor_offset_ =
            aligned_static_offset + requirements.static_kernel_descriptor_bytes;
    dynamic_kernel_descriptor_offset_ =
            aligned_dynamic_offset + requirements.dynamic_kernel_descriptor_bytes;
    device_tensor_offset_ = aligned_tensor_offset + requirements.device_tensor_bytes;

    // Store slice (requires std::string key for insertion)
    module_slices_.emplace(module_id, slice);

    RT_LOGC_DEBUG(utils::Core::CorePipeline, "Memory slice allocated for module {}", module_id);

    return slice;
}

const ModuleMemorySlice &PipelineMemoryManager::get_module_slice(std::string_view module_id) const {
    // Use C++20 heterogeneous lookup with string_view
    const auto it = module_slices_.find(module_id);
    if (it == module_slices_.end()) {
        const std::string error_msg =
                std::format("Module '{}' not found in memory slice registry", module_id);
        RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }
    return it->second;
}

PipelineMemoryManager::MemoryUsage PipelineMemoryManager::get_memory_usage() const {
    MemoryUsage usage{};

    usage.static_kernel_descriptor_used = static_kernel_descriptor_offset_;
    usage.dynamic_kernel_descriptor_used = dynamic_kernel_descriptor_offset_;
    usage.device_tensor_used = device_tensor_offset_;

    usage.static_kernel_descriptor_total = static_kernel_descriptor_arena_cpu_.total_bytes();
    usage.dynamic_kernel_descriptor_total = dynamic_kernel_descriptor_arena_cpu_.total_bytes();
    usage.device_tensor_total = device_tensor_arena_.total_bytes();

    return usage;
}

ModuleMemoryRequirements calculate_pipeline_memory_requirements(
        const std::vector<ModuleMemoryRequirements> &module_requirements) {

    RT_LOGC_DEBUG(
            utils::Core::CorePipeline,
            "Calculating total requirements for {} modules",
            module_requirements.size());

    ModuleMemoryRequirements total_reqs{};

    // Simulate the allocation process to account for alignment
    std::size_t static_offset = 0;
    std::size_t dynamic_offset = 0;
    std::size_t tensor_offset = 0;

    for (const auto
                 &[static_kernel_descriptor_bytes,
                   dynamic_kernel_descriptor_bytes,
                   device_tensor_bytes,
                   alignment] : module_requirements) {
        // Validate alignment is a non-zero power-of-two
        if (alignment == 0 || !std::has_single_bit(alignment)) {
            const std::string error_msg = std::format(
                    "Invalid alignment value: {}. Alignment must be a "
                    "non-zero power-of-two.",
                    alignment);
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        // Apply alignment (same logic as allocate_module_slice)
        static_offset = align_memory_offset(static_offset, alignment);
        dynamic_offset = align_memory_offset(dynamic_offset, alignment);
        tensor_offset = align_memory_offset(tensor_offset, alignment);

        // Add the module's requirements
        static_offset += static_kernel_descriptor_bytes;
        dynamic_offset += dynamic_kernel_descriptor_bytes;
        tensor_offset += device_tensor_bytes;
    }

    // Return the total with alignment accounted for
    total_reqs.static_kernel_descriptor_bytes = static_offset;
    total_reqs.dynamic_kernel_descriptor_bytes = dynamic_offset;
    total_reqs.device_tensor_bytes = tensor_offset;

    RT_LOGC_DEBUG(
            utils::Core::CorePipeline,
            "Total requirements: static={} dynamic={} device={}",
            total_reqs.static_kernel_descriptor_bytes,
            total_reqs.dynamic_kernel_descriptor_bytes,
            total_reqs.device_tensor_bytes);

    return total_reqs;
}

ModuleMemoryRequirements
calculate_pipeline_memory_requirements(const std::vector<IModule *> &modules) {

    RT_LOGC_DEBUG(
            utils::Core::CorePipeline,
            "Calculating total requirements for {} module pointers",
            modules.size());

    // Collect requirements from all modules
    std::vector<ModuleMemoryRequirements> module_requirements;
    module_requirements.reserve(modules.size());

    for (const auto *module : modules) {
        if (module == nullptr) {
            const std::string error_msg = "Null module pointer in calculate_total_requirements";
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        const auto *alloc_provider = dynamic_cast<const IAllocationInfoProvider *>(module);
        if (alloc_provider == nullptr) {
            const std::string error_msg = std::format(
                    "Module '{}' with type '{}' does not implement "
                    "IAllocationInfoProvider interface",
                    module->get_instance_id(),
                    module->get_type_id());
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        module_requirements.push_back(alloc_provider->get_requirements());
    }

    // Use the existing overload to calculate with alignment
    return calculate_pipeline_memory_requirements(module_requirements);
}

void PipelineMemoryManager::allocate_all_module_slices(const std::vector<IModule *> &modules) {

    RT_LOGC_INFO(
            utils::Core::CorePipeline, "Allocating memory slices for {} modules", modules.size());

    // Collect all requirements and module IDs in order
    std::vector<ModuleMemoryRequirements> module_requirements;
    std::vector<std::string> module_ids;

    module_requirements.reserve(modules.size());
    module_ids.reserve(modules.size());

    for (const auto *module : modules) {
        if (module == nullptr) {
            const std::string error_msg = "Null module pointer in allocate_all_module_slices";
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        const auto *alloc_provider = dynamic_cast<const IAllocationInfoProvider *>(module);
        if (alloc_provider == nullptr) {
            const std::string error_msg = std::format(
                    "Module '{}' with type '{}' does not implement "
                    "IAllocationInfoProvider interface",
                    module->get_instance_id(),
                    module->get_type_id());
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        module_requirements.push_back(alloc_provider->get_requirements());
        module_ids.emplace_back(module->get_instance_id());
    }

    // Calculate total requirements to verify we have enough space
    const ModuleMemoryRequirements total_reqs =
            calculate_pipeline_memory_requirements(module_requirements);

    if (total_reqs.static_kernel_descriptor_bytes >
                static_kernel_descriptor_arena_cpu_.total_bytes() ||
        total_reqs.dynamic_kernel_descriptor_bytes >
                dynamic_kernel_descriptor_arena_cpu_.total_bytes() ||
        total_reqs.device_tensor_bytes > device_tensor_arena_.total_bytes()) {
        const std::string error_msg = "Total memory requirements exceed arena capacity";
        RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Allocate memory slice for each module in the same order
    for (std::size_t i = 0; i < modules.size(); ++i) {
        std::ignore = allocate_module_slice(module_ids[i], module_requirements[i]);
    }

    RT_LOGC_INFO(utils::Core::CorePipeline, "All module slices allocated successfully");
}

std::unique_ptr<PipelineMemoryManager>
PipelineMemoryManager::create_for_modules(const std::vector<IModule *> &modules) {

    RT_LOGC_INFO(
            utils::Core::CorePipeline,
            "Creating PipelineMemoryManager for {} modules",
            modules.size());

    // Collect requirements from all modules
    std::vector<ModuleMemoryRequirements> module_requirements;
    module_requirements.reserve(modules.size());

    for (const auto *module : modules) {
        if (module == nullptr) {
            const std::string error_msg = "Null module pointer in create_for_modules";
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        const auto *alloc_provider = dynamic_cast<const IAllocationInfoProvider *>(module);
        if (alloc_provider == nullptr) {
            const std::string error_msg = std::format(
                    "Module '{}' with type '{}' does not implement "
                    "IAllocationInfoProvider interface",
                    module->get_instance_id(),
                    module->get_type_id());
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        module_requirements.push_back(alloc_provider->get_requirements());
    }

    // Calculate total requirements with alignment
    const ModuleMemoryRequirements total_reqs =
            calculate_pipeline_memory_requirements(module_requirements);

    // Create the memory manager with the calculated sizes
    auto manager = std::make_unique<PipelineMemoryManager>(
            total_reqs.static_kernel_descriptor_bytes,
            total_reqs.dynamic_kernel_descriptor_bytes,
            total_reqs.device_tensor_bytes);

    RT_LOGC_INFO(utils::Core::CorePipeline, "PipelineMemoryManager created successfully");

    return manager;
}

void PipelineMemoryManager::copy_all_static_descriptors_to_device(cudaStream_t stream) const {

    // Only copy if we have allocated static descriptors
    if (static_kernel_descriptor_offset_ > 0) {
        // Get base pointers from arenas using raw_ptr_mutable()
        auto *cpu_ptr = static_kernel_descriptor_arena_cpu_.raw_ptr_mutable();
        auto *gpu_ptr = static_kernel_descriptor_arena_gpu_.raw_ptr_mutable();

        if (cpu_ptr == nullptr || gpu_ptr == nullptr) {
            const std::string error_msg =
                    "Null pointer in static descriptor arenas during bulk copy";
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        RT_LOGC_DEBUG(
                utils::Core::CorePipeline,
                "Copying all static descriptors to device: {} bytes from {} to {}",
                static_kernel_descriptor_offset_,
                cpu_ptr,
                gpu_ptr);

        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                gpu_ptr,
                cpu_ptr,
                static_kernel_descriptor_offset_,
                cudaMemcpyHostToDevice,
                stream));

        RT_LOGC_INFO(
                utils::Core::CorePipeline,
                "Bulk copy of static descriptors complete: {} bytes",
                static_kernel_descriptor_offset_);
    } else {
        RT_LOGC_DEBUG(utils::Core::CorePipeline, "No static descriptors to copy (offset=0)");
    }
}

void PipelineMemoryManager::copy_all_dynamic_descriptors_to_device(cudaStream_t stream) const {

    // Only copy if we have allocated dynamic descriptors
    if (dynamic_kernel_descriptor_offset_ > 0) {
        // Get base pointers from arenas using raw_ptr_mutable()
        auto *cpu_ptr = dynamic_kernel_descriptor_arena_cpu_.raw_ptr_mutable();
        auto *gpu_ptr = dynamic_kernel_descriptor_arena_gpu_.raw_ptr_mutable();

        if (cpu_ptr == nullptr || gpu_ptr == nullptr) {
            const std::string error_msg =
                    "Null pointer in dynamic descriptor arenas during bulk copy";
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        RT_LOGC_DEBUG(
                utils::Core::CorePipeline,
                "Copying all dynamic descriptors to device: {} bytes from {} to {}",
                dynamic_kernel_descriptor_offset_,
                cpu_ptr,
                gpu_ptr);

        const cudaError_t result = cudaMemcpyAsync(
                gpu_ptr,
                cpu_ptr,
                dynamic_kernel_descriptor_offset_,
                cudaMemcpyHostToDevice,
                stream);

        if (result != cudaSuccess) {
            const std::string error_msg = std::format(
                    "Failed to copy dynamic descriptors to device: {}", cudaGetErrorString(result));
            RT_LOGC_ERROR(utils::Core::CorePipeline, "{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        RT_LOGC_INFO(
                utils::Core::CorePipeline,
                "Bulk copy of dynamic descriptors complete: {} bytes",
                dynamic_kernel_descriptor_offset_);
    } else {
        RT_LOGC_DEBUG(utils::Core::CorePipeline, "No dynamic descriptors to copy (offset=0)");
    }
}

} // namespace framework::pipeline
