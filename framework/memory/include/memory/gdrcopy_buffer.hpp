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

#ifndef FRAMEWORK_CORE_GDRCOPY_BUFFER_HPP
#define FRAMEWORK_CORE_GDRCOPY_BUFFER_HPP

/**
 * @file gdrcopy_buffer.hpp
 * @brief GDRCopy utility for pinned GPU memory with CPU access
 *
 * Provides RAII wrapper for GDRCopy pinned GPU memory (CPU-visible GPU memory).
 * Enables zero-copy access patterns where NIC/CPU can access GPU memory directly.
 *
 * @note TEGRA hardware is not supported in this implementation (x86_64 only).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>

#include <gdrapi.h> // for gdr_t, gdr_pin_buffer, gdr_map, gdr_unmap, gdr_unpin_buffer

#include <cuda.h> // for CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, cuMemAlloc, cuPointerSetAttribute

#include "utils/error_macros.hpp" // for FRAMEWORK_CUDA_DRIVER_CHECK_THROW

namespace framework::memory {

/// Minimum GDRCopy pin size in bytes (64KB page alignment requirement)
/// Note: GPU_PAGE_SIZE and GPU_PAGE_MASK are defined in gdrapi.h
inline constexpr std::size_t GPU_MIN_PIN_SIZE = GPU_PAGE_SIZE;

/**
 * Custom deleter for GDRCopy handle
 *
 * This deleter is designed to work with std::unique_ptr to provide RAII
 * management of GDRCopy handles. It automatically calls gdr_close() when
 * the unique_ptr is destroyed.
 *
 * The deleter takes gdr_t by value (not pointer) which allows unique_ptr
 * to directly store the gdr_t handle without heap allocation.
 */
struct GdrHandleDeleter final {
    /**
     * Closes GDRCopy handle using gdr_close
     *
     * @param[in] handle GDRCopy handle to be closed (gdr_t is struct gdr*)
     * @note Best-effort cleanup; never throws exceptions from deleter
     * @note gdr_close returns 0 on success, we ignore errors in destructor
     */
    void operator()(gdr_t handle) const noexcept {
        if (handle != nullptr) {
            // gdr_close returns 0 on success, ignore errors in deleter
            std::ignore = gdr_close(handle);
        }
    }
};

/**
 * Type alias for std::unique_ptr with GDRCopy handle deleter
 *
 * This provides automatic RAII management of GDRCopy handles.
 * The handle is automatically closed when the unique_ptr goes out of scope.
 *
 * Note: We use a custom deleter that takes gdr_t by value, allowing
 * unique_ptr<struct gdr, Deleter> to directly store the gdr_t handle
 * without heap allocation. The struct gdr is opaque but that's fine
 * since we never dereference it - we only pass the pointer to gdr_close().
 *
 * Usage:
 * ```cpp
 * auto gdr_handle = make_unique_gdr_handle();
 * // Use gdr_handle.get() to get the raw gdr_t (struct gdr*)
 * // Automatically closed when gdr_handle goes out of scope
 * ```
 */
using UniqueGdrHandle = std::unique_ptr<struct gdr, GdrHandleDeleter>;

/**
 * Creates a unique_ptr managing a GDRCopy handle
 *
 * Opens a GDRCopy handle using gdr_open() and wraps it in a unique_ptr
 * with automatic cleanup.
 *
 * @return UniqueGdrHandle managing the opened handle
 * @throws std::runtime_error if gdr_open() fails
 */
inline UniqueGdrHandle make_unique_gdr_handle() {
    gdr_t handle = gdr_open();
    if (handle == nullptr) {
        throw std::runtime_error("make_unique_gdr_handle: gdr_open() failed");
    }
    // No heap allocation needed - unique_ptr stores the gdr_t directly
    return UniqueGdrHandle(handle);
}

/**
 * @brief RAII wrapper for GDRCopy pinned GPU memory
 *
 * Provides CPU-visible access to GPU memory via GDRCopy. The NIC can write
 * directly to this memory, and the CPU can read it without GPU→CPU copies.
 *
 * This is useful for:
 * - Direct NIC-to-GPU memory access (DOCA GPUNetIO)
 * - CPU polling of GPU-written status flags
 * - Zero-copy data sharing between CPU and GPU
 *
 * Usage with UniqueGdrHandle (recommended):
 * ```cpp
 * auto gdr_handle = make_unique_gdr_handle();  // RAII-managed handle
 * GpinnedBuffer buf(gdr_handle.get(), sizeof(uint32_t));
 *
 * // CPU writes to GPU memory
 * *static_cast<uint32_t*>(buf.get_host_addr()) = 42;
 *
 * // Kernel reads from GPU memory
 * uint32_t* device_ptr = static_cast<uint32_t*>(buf.get_device_addr());
 * kernel<<<>>>(..., device_ptr, ...);
 *
 * // Cleanup automatic - handle closed when gdr_handle goes out of scope
 * ```
 *
 * Legacy usage with raw gdr_t:
 * ```cpp
 * gdr_t handle = gdr_open();
 * GpinnedBuffer buf(handle, sizeof(uint32_t));
 * // ... use buffer ...
 * gdr_close(handle);  // Manual cleanup required
 * ```
 */
class GpinnedBuffer final {
public:
    /**
     * @brief Construct a GDRCopy pinned buffer
     *
     * @param[in] gdr_handle Non-owning GDRCopy handle (gdr_t is already a pointer)
     * @param[in] size_bytes Requested size in bytes (minimum GPU_MIN_PIN_SIZE)
     *
     * @throws std::invalid_argument if gdr_handle is null or size_bytes is 0
     * @throws std::runtime_error if GDRCopy operations fail
     */
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    GpinnedBuffer(gdr_t gdr_handle, const std::size_t size_bytes)
            : gdr_handle_(gdr_handle), size_input_(size_bytes) {
        CUdeviceptr dev_addr{};
        void *host_ptr{};
        constexpr unsigned int K_SYNC_MEMOPS_FLAG = 1;
        std::size_t pin_size{};
        std::size_t alloc_size{};
        std::size_t rounded_size{};

        if (gdr_handle_ == nullptr || size_input_ == 0) {
            throw std::invalid_argument("GpinnedBuffer: invalid input arguments");
        }

        const std::size_t adjusted_size = std::max(size_input_, GPU_MIN_PIN_SIZE);

        // GDRDRV and the GPU driver require GPU page size-aligned address and size
        // arguments to gdr_pin_buffer, so we need to be paranoid here and allocate
        // an extra page so we can safely pass the rounded size
        rounded_size = (adjusted_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        pin_size = rounded_size;
        alloc_size = rounded_size + GPU_PAGE_SIZE;

        // Allocate device memory (TEGRA not supported)
        FRAMEWORK_CUDA_DRIVER_CHECK_THROW(cuMemAlloc(&dev_addr, alloc_size));

        addr_allocated_ = static_cast<std::uintptr_t>(dev_addr);

        // Offset into a page-aligned address if necessary
        if (dev_addr % GPU_PAGE_SIZE != 0) {
            dev_addr += (GPU_PAGE_SIZE - (dev_addr % GPU_PAGE_SIZE));
        }

        // Set attributes for the allocated device memory
        FRAMEWORK_CUDA_DRIVER_CHECK_THROW(cuPointerSetAttribute(
                &K_SYNC_MEMOPS_FLAG, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_addr));

        // Pin the device buffer
        if (gdr_pin_buffer(gdr_handle_, dev_addr, pin_size, 0, 0, &mh_) != 0) {
            cuMemFree(static_cast<CUdeviceptr>(addr_allocated_)); // Ignore error in cleanup
            throw std::runtime_error("GpinnedBuffer: gdr_pin_buffer failed");
        }

        // Map the buffer to user space
        if (gdr_map(gdr_handle_, mh_, &host_ptr, pin_size) != 0) {
            gdr_unpin_buffer(gdr_handle_, mh_);
            cuMemFree(static_cast<CUdeviceptr>(addr_allocated_)); // Ignore error in cleanup
            throw std::runtime_error("GpinnedBuffer: gdr_map failed");
        }

        // Retrieve info about the mapping
        if (gdr_get_info(gdr_handle_, mh_, &info_) != 0) {
            gdr_unmap(gdr_handle_, mh_, host_ptr, pin_size);
            gdr_unpin_buffer(gdr_handle_, mh_);
            cuMemFree(static_cast<CUdeviceptr>(addr_allocated_)); // Ignore error in cleanup
            throw std::runtime_error("GpinnedBuffer: gdr_get_info failed");
        }

        addr_device_ = static_cast<std::uintptr_t>(dev_addr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        addr_host_ = reinterpret_cast<std::uintptr_t>(host_ptr);
        size_free_ = pin_size;
    }

    /**
     * @brief Destructor - unmaps and unpins GDRCopy buffer
     */
    ~GpinnedBuffer() {
        if (gdr_handle_ != nullptr) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
            gdr_unmap(gdr_handle_, mh_, reinterpret_cast<void *>(addr_host_), size_free_);
            gdr_unpin_buffer(gdr_handle_, mh_);
            // Ignore errors in destructor
            cuMemFree(static_cast<CUdeviceptr>(addr_allocated_));
        }
    }

    // Delete copy operations
    GpinnedBuffer(const GpinnedBuffer &) = delete;
    GpinnedBuffer &operator=(const GpinnedBuffer &) = delete;

    // Delete move operations (could be implemented if needed)
    GpinnedBuffer(GpinnedBuffer &&) = delete;
    GpinnedBuffer &operator=(GpinnedBuffer &&) = delete;

    /**
     * @brief Get host-side pointer for CPU access
     * @return Host pointer (CPU-visible)
     */
    [[nodiscard]] void *get_host_addr() const {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
        return reinterpret_cast<void *>(addr_host_);
    }

    /**
     * @brief Get device-side pointer for GPU kernel access
     * @return Device pointer (GPU-visible)
     */
    [[nodiscard]] void *get_device_addr() const {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
        return reinterpret_cast<void *>(addr_device_);
    }

    /**
     * @brief Get requested buffer size
     * @return Original requested size in bytes
     */
    [[nodiscard]] std::size_t get_size() const { return size_input_; }

    /**
     * @brief Get actual allocated size (page-aligned)
     * @return Actual pinned size in bytes
     */
    [[nodiscard]] std::size_t get_size_free() const { return size_free_; }

private:
    gdr_t gdr_handle_{};           //!< Non-owning GDRCopy handle (gdr_t is already a pointer)
    std::size_t size_input_{};     //!< Requested buffer size
    gdr_mh_t mh_{};                //!< GDRCopy memory handle
    gdr_info_t info_{};            //!< GDRCopy buffer info
    std::uintptr_t addr_device_{}; //!< Device-side address (GPU-visible)
    std::uintptr_t addr_host_{};   //!< Host-side address (CPU-visible)
    std::uintptr_t
            addr_allocated_{}; //!< Original allocated address from cuMemAlloc (for cuMemFree)
    std::size_t size_free_{};  //!< Actual pinned size (page-aligned)
};

} // namespace framework::memory

#endif // FRAMEWORK_CORE_GDRCOPY_BUFFER_HPP
