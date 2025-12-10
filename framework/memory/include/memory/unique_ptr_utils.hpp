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

#ifndef FRAMEWORK_UNIQUE_PTR_UTILS_HPP
#define FRAMEWORK_UNIQUE_PTR_UTILS_HPP

#include <limits>
#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

#include "memory/device_allocators.hpp"
#include "utils/error_macros.hpp"
#include "utils/exceptions.hpp"

namespace framework::memory {

/**
 * Custom deleter for device memory allocated with CUDA
 *
 * This deleter is designed to work with std::unique_ptr to provide RAII
 * management of CUDA device memory. It automatically calls cudaFree() when
 * the unique_ptr is destroyed.
 *
 * @tparam T The type of the pointer being managed (can be array types)
 */
template <typename T> struct DeviceDeleter final {
    using PtrT = std::remove_all_extents_t<T>; //!< Pointer type after removing
                                               //!< array extents

    /**
     * Deletes device memory using cudaFree
     *
     * @param[in] ptr Pointer to device memory to be freed
     * @note Best-effort cleanup; never throws exceptions from deleter
     */
    void operator()(PtrT *ptr) const noexcept {
        FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_NO_THROW(cudaFree(ptr));
    }
};

/**
 * Custom deleter for pinned host memory allocated with CUDA
 *
 * This deleter is designed to work with std::unique_ptr to provide RAII
 * management of CUDA pinned host memory. It automatically calls cudaFreeHost()
 * when the unique_ptr is destroyed.
 *
 * @tparam T The type of the pointer being managed (can be array types)
 */
template <typename T> struct PinnedDeleter final {
    using PtrT = std::remove_all_extents_t<T>; //!< Pointer type after removing
                                               //!< array extents

    /**
     * Deletes pinned host memory using cudaFreeHost
     *
     * @param[in] ptr Pointer to pinned host memory to be freed
     * @note Best-effort cleanup; never throws exceptions from deleter
     */
    void operator()(PtrT *ptr) const noexcept {
        FRAMEWORK_CUDA_RUNTIME_EXPR_CHECK_NO_THROW(cudaFreeHost(ptr));
    }
};

/**
 * Type alias for std::unique_ptr with device memory deleter
 *
 * This provides automatic RAII management of CUDA device memory.
 * The memory is automatically freed when the unique_ptr goes out of scope.
 *
 * @tparam T The type of the pointer being managed
 */
template <typename T> using UniqueDevicePtr = std::unique_ptr<T, DeviceDeleter<T>>;

/**
 * Type alias for std::unique_ptr with pinned host memory deleter
 *
 * This provides automatic RAII management of CUDA pinned host memory.
 * The memory is automatically freed when the unique_ptr goes out of scope.
 *
 * @tparam T The type of the pointer being managed
 */
template <typename T> using UniquePinnedPtr = std::unique_ptr<T, PinnedDeleter<T>>;

/**
 * Creates a unique_ptr managing device memory
 *
 * Allocates device memory using DeviceAlloc and wraps it in a unique_ptr
 * with automatic cleanup. (Note: cudaMalloc leaves device memory
 * uninitialised.)
 *
 * @tparam T The type of elements to allocate
 * @param[in] count Number of elements to allocate (default: 1)
 * @return UniqueDevicePtr managing the allocated memory
 * @throws CudaRuntimeException if device allocation fails
 * @throws std::overflow_error if size calculation overflows
 */
template <typename T> UniqueDevicePtr<T> make_unique_device(const std::size_t count = 1) {
    FRAMEWORK_NV_THROW_IF(
            count > std::numeric_limits<std::size_t>::max() / sizeof(T),
            std::overflow_error,
            "make_unique_device size calculation overflow");

    using PointerT = typename UniqueDevicePtr<T>::pointer;
    PointerT ptr = nullptr;
    if (count != 0) {
        ptr = static_cast<PointerT>(DeviceAlloc::allocate(count * sizeof(T)));
    }
    return UniqueDevicePtr<T>(ptr);
}

/**
 * Creates a unique_ptr managing pinned host memory
 *
 * Allocates pinned host memory using PinnedAlloc and wraps it in a unique_ptr
 * with automatic cleanup.
 *
 * @tparam T The type of elements to allocate
 * @param[in] count Number of elements to allocate (default: 1)
 * @return UniquePinnedPtr managing the allocated memory
 * @throws CudaRuntimeException if pinned allocation fails
 * @throws std::overflow_error if size calculation overflows
 */
template <typename T> UniquePinnedPtr<T> make_unique_pinned(const std::size_t count = 1) {
    FRAMEWORK_NV_THROW_IF(
            count > std::numeric_limits<std::size_t>::max() / sizeof(T),
            std::overflow_error,
            "make_unique_pinned size calculation overflow");

    using PointerT = typename UniquePinnedPtr<T>::pointer;
    PointerT ptr = nullptr;
    if (count != 0) {
        ptr = static_cast<PointerT>(PinnedAlloc::allocate(count * sizeof(T)));
    }
    return UniquePinnedPtr<T>(ptr);
}

} // namespace framework::memory

#endif // FRAMEWORK_UNIQUE_PTR_UTILS_HPP
