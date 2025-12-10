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

#ifndef FRAMEWORK_CORE_KERNEL_DESCRIPTOR_ACCESSOR_HPP
#define FRAMEWORK_CORE_KERNEL_DESCRIPTOR_ACCESSOR_HPP

#include <cstddef>
#include <format>
#include <iterator>
#include <new>
#include <stdexcept>

#include <driver_types.h>

#include "pipeline/types.hpp"

namespace framework::pipeline {

/**
 * Kernel descriptor accessor for type-safe parameter handling
 *
 * This is Per-Module class type instance.
 *
 * Provides access to both static and dynamic kernel parameter structures in
 * pinned memory. Each module gets its own KernelDescriptorAccessor and is
 * responsible for copying descriptors from CPU to GPU when ready.
 */
class KernelDescriptorAccessor final {
public:
    /**
     * Constructor
     *
     * @param[in] memory_slice Memory slice containing kernel descriptor regions
     */
    explicit KernelDescriptorAccessor(const ModuleMemorySlice &memory_slice);

    // Non-copyable and non-movable
    KernelDescriptorAccessor(const KernelDescriptorAccessor &) = delete;
    KernelDescriptorAccessor &operator=(const KernelDescriptorAccessor &) = delete;
    KernelDescriptorAccessor(KernelDescriptorAccessor &&) = delete;
    KernelDescriptorAccessor &operator=(KernelDescriptorAccessor &&) = delete;

    /**
     * Destructor
     */
    ~KernelDescriptorAccessor() = default;

    /**
     * Create type-safe static kernel parameter at specific offset
     *
     * In-place construction (placement new) so no UB when accessing the
     * descriptor.

     * @note The created object's destructor is not called automatically.
     *       This is safe for POD types typically used as kernel parameters.
     *       Users must ensure proper cleanup for non-POD types.
     *
     * @tparam T Kernel parameter structure type
     * @param[in] offset_bytes Byte offset within the module's static descriptor
     * region
     * @return Reference to constructed kernel parameter object (CPU memory)
     * @throws std::runtime_error if allocation exceeds slice bounds
     */
    template <typename T>
    [[nodiscard]]
    T &create_static_param(const std::size_t offset_bytes) {
        if (offset_bytes + sizeof(T) > memory_slice_.static_kernel_descriptor_bytes) {
            throw std::runtime_error(std::format(
                    "Static kernel parameter allocation exceeds slice "
                    "bounds: {} + {} > {}",
                    offset_bytes,
                    sizeof(T),
                    memory_slice_.static_kernel_descriptor_bytes));
        }

        std::byte *location = std::next(
                memory_slice_.static_kernel_descriptor_cpu_ptr,
                static_cast<std::ptrdiff_t>(offset_bytes));
        return *new (location) T{}; // Placement new with default initialization
    }

    /**
     * Create type-safe dynamic kernel parameter at specific offset
     *
     * @tparam T Kernel parameter structure type
     * @param[in] offset_bytes Byte offset within the module's dynamic descriptor
     * region
     * @return Reference to constructed kernel parameter object (CPU memory)
     * @throws std::runtime_error if allocation exceeds slice bounds
     */
    template <typename T>
    [[nodiscard]]
    T &create_dynamic_param(const std::size_t offset_bytes) {
        if (offset_bytes + sizeof(T) > memory_slice_.dynamic_kernel_descriptor_bytes) {
            throw std::runtime_error(std::format(
                    "Dynamic kernel parameter allocation exceeds slice "
                    "bounds: {} + {} > {}",
                    offset_bytes,
                    sizeof(T),
                    memory_slice_.dynamic_kernel_descriptor_bytes));
        }

        std::byte *location = std::next(
                memory_slice_.dynamic_kernel_descriptor_cpu_ptr,
                static_cast<std::ptrdiff_t>(offset_bytes));
        return *new (location) T{}; // Placement new with default initialization
    }

    /**
     * Copy static descriptors from CPU to GPU memory (async)
     *
     * Module calls this when static descriptors are ready.
     * Typically called once during module initialization.
     *
     * @param[in] stream CUDA stream for async operation
     */
    void copy_static_descriptors_to_device(cudaStream_t stream) const;

    /**
     * Copy dynamic descriptors from CPU to GPU memory (async)
     *
     * Module calls this when dynamic descriptors are ready.
     * Typically called every frame/slot when parameters change.
     *
     * @param[in] stream CUDA stream for async operation
     */
    void copy_dynamic_descriptors_to_device(cudaStream_t stream) const;

    /**
     * Get GPU device pointer for static kernel parameters
     *
     * @tparam T Kernel parameter structure type
     * @param[in] offset_bytes Byte offset within the static descriptor region
     * @return Device pointer for kernel launch
     */
    template <typename T>
    [[nodiscard]]
    T *get_static_device_ptr(const std::size_t offset_bytes) const {
        if (offset_bytes + sizeof(T) > memory_slice_.static_kernel_descriptor_bytes) {
            throw std::runtime_error(std::format(
                    "Static kernel parameter access exceeds slice bounds: "
                    "{} + {} > {}",
                    offset_bytes,
                    sizeof(T),
                    memory_slice_.static_kernel_descriptor_bytes));
        }

        std::byte *location = std::next(
                memory_slice_.static_kernel_descriptor_gpu_ptr,
                static_cast<std::ptrdiff_t>(offset_bytes));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T *>(location);
    }

    /**
     * Get GPU device pointer for dynamic kernel parameters
     *
     * @tparam T Kernel parameter structure type
     * @param[in] offset_bytes Byte offset within the dynamic descriptor region
     * @return Device pointer for kernel launch
     */
    template <typename T>
    [[nodiscard]]
    T *get_dynamic_device_ptr(const std::size_t offset_bytes) const {
        if (offset_bytes + sizeof(T) > memory_slice_.dynamic_kernel_descriptor_bytes) {
            throw std::runtime_error(std::format(
                    "Dynamic kernel parameter access exceeds slice bounds: "
                    "{} + {} > {}",
                    offset_bytes,
                    sizeof(T),
                    memory_slice_.dynamic_kernel_descriptor_bytes));
        }

        std::byte *location = std::next(
                memory_slice_.dynamic_kernel_descriptor_gpu_ptr,
                static_cast<std::ptrdiff_t>(offset_bytes));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T *>(location);
    }

private:
    const ModuleMemorySlice memory_slice_; //!< Module's memory slice
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_KERNEL_DESCRIPTOR_ACCESSOR_HPP
