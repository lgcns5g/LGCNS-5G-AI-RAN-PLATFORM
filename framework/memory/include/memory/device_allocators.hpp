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

#ifndef FRAMEWORK_DEVICE_ALLOCATORS_HPP
#define FRAMEWORK_DEVICE_ALLOCATORS_HPP

#include <source_location>

#include <cuda_runtime.h>

#include "utils/error_macros.hpp"
#include "utils/exceptions.hpp"

namespace framework::memory {
/**
 * Device memory allocator for CUDA GPU memory
 *
 * Provides static methods for allocating and deallocating memory on the GPU
 * device. Uses cudaMalloc/cudaFree for memory management.
 */
struct DeviceAlloc final {
    /**
     * Allocate memory on the GPU device
     *
     * @param[in] nbytes Number of bytes to allocate
     * @return Pointer to allocated device memory
     * @throws CudaRuntimeException If allocation fails
     */
    [[nodiscard]]
    static void *allocate(const std::size_t nbytes) {
        void *addr{};
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMalloc(&addr, nbytes));
        return addr;
    }

    /**
     * Deallocate memory on the GPU device
     *
     * @param[in] addr Pointer to device memory to deallocate
     * @throws CudaRuntimeException If deallocation fails
     */
    static void deallocate(void *addr) { FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaFree(addr)); }
};

/**
 * Pinned host memory allocator for CUDA
 *
 * Provides static methods for allocating and deallocating pinned (page-locked)
 * host memory. Pinned memory can be transferred to/from the GPU more
 * efficiently than pageable memory. Uses cudaHostAlloc/cudaFreeHost for memory
 * management.
 */
struct PinnedAlloc final {
    /**
     * Allocate pinned host memory
     *
     * @param[in] nbytes Number of bytes to allocate
     * @return Pointer to allocated pinned host memory
     * @throws CudaRuntimeException If allocation fails
     * @note No tracking of host pinned memory currently
     */
    [[nodiscard]]
    static void *allocate(const std::size_t nbytes) {
        void *addr{};
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaHostAlloc(&addr, nbytes, 0));
        return addr;
    }

    /**
     * Deallocate pinned host memory
     *
     * @param[in] addr Pointer to pinned host memory to deallocate
     * @throws CudaRuntimeException If deallocation fails
     */
    static void deallocate(void *addr) { FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaFreeHost(addr)); }
};

} // namespace framework::memory

#endif // FRAMEWORK_DEVICE_ALLOCATORS_HPP
