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

#ifndef FRAMEWORK_MONOTONIC_ALLOC_HPP
#define FRAMEWORK_MONOTONIC_ALLOC_HPP

#include <bit>
#include <format>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "utils/error_macros.hpp"

namespace framework::memory {

/**
 * A monotonic memory allocator that provides fast, aligned memory allocation
 * from a pre-allocated buffer.
 *
 * This allocator manages a contiguous block of memory and provides
 * sub-allocations from it in a linear fashion. Once memory is allocated, it
 * cannot be individually freed - only the entire allocator can be reset. This
 * design makes it very fast for temporary allocations with known lifetimes.
 *
 * Note: CUDA memory allocation alignment guarantees:
 * • cudaMalloc: From the CUDA Programming Guide (§3.2 of 12.x): the returned
 *   pointer "is always at least 256-byte aligned."
 * • cudaHostAlloc / cudaMallocHost (pinned host memory): Guarantees alignment
 *   to at least the host page size (commonly 4 KB) and never less than 64
 * bytes.
 *
 * @tparam ALLOC_ALIGN_BYTES Alignment requirement for all allocations (must be
 * power of 2)
 * @tparam TAlloc Allocator type that provides static allocate() and
 * deallocate() methods
 */
template <std::uint32_t ALLOC_ALIGN_BYTES, typename TAlloc> class MonotonicAlloc final {
public:
    /**
     * Constructs a monotonic allocator with the specified buffer size.
     *
     * @param[in] bufsize Size of the buffer to allocate in bytes
     * @throws std::bad_alloc if the underlying allocation fails
     */
    explicit MonotonicAlloc(const std::size_t bufsize)
            : buffer_(TAlloc::allocate(bufsize)), size_(bufsize) {
        static_assert(
                std::has_single_bit(ALLOC_ALIGN_BYTES), "ALLOC_ALIGN_BYTES must be a power of 2");
    }

    /**
     * Move constructor - transfers ownership of the buffer from another
     * allocator.
     *
     * @param[in] allocator The allocator to move from (will be left in a valid
     * but empty state)
     */
    MonotonicAlloc(MonotonicAlloc &&allocator) noexcept
            : buffer_(std::exchange(allocator.buffer_, nullptr)),
              size_(std::exchange(allocator.size_, 0)),
              offset_(std::exchange(allocator.offset_, 0)) {}

    /**
     * Move assignment operator - transfers ownership of the buffer from another
     * allocator.
     *
     * @param[in] allocator The allocator to move from (will be left in a valid
     * but empty state)
     * @return Reference to this allocator
     */
    MonotonicAlloc &operator=(MonotonicAlloc &&allocator) noexcept {
        if (this != &allocator) {
            if (buffer_ != nullptr) {
                TAlloc::deallocate(buffer_);
            }
            buffer_ = std::exchange(allocator.buffer_, nullptr);
            size_ = std::exchange(allocator.size_, 0);
            offset_ = std::exchange(allocator.offset_, 0);
        }
        return *this;
    }

    MonotonicAlloc &operator=(const MonotonicAlloc &) = delete;
    MonotonicAlloc(const MonotonicAlloc &) = delete;

    /**
     * Destructor - deallocates the managed buffer if it exists.
     */
    ~MonotonicAlloc() {
        if (buffer_ != nullptr) {
            try {
                TAlloc::deallocate(buffer_);
            } catch (const std::exception &e) {
                RT_LOGC_ERROR(
                        utils::Core::CoreCudaRuntime, "Error deallocating buffer: {}", e.what());
            } catch (...) {
                RT_LOGC_ERROR(utils::Core::CoreCudaRuntime, "Unknown error deallocating buffer");
            }
        }
    }

    /**
     * Resets the allocator to its initial state, making all allocated memory
     * available for reuse.
     *
     * This does not deallocate the underlying buffer, just resets the allocation
     * offset to zero. All previously returned pointers become invalid after this
     * call.
     */
    void reset() noexcept { offset_ = 0; }

    /**
     * Allocates a block of memory from the linear buffer.
     *
     * The returned memory is aligned according to ALLOC_ALIGN_BYTES. The
     * allocation is performed by advancing the internal offset pointer, so this
     * operation is very fast.
     *
     * @param[in] nbytes Number of bytes to allocate
     * @return Pointer to the allocated memory block
     * @throws std::runtime_error if the requested size would exceed the buffer
     * capacity
     */
    void *allocate(const std::size_t nbytes) {
        // Calculate the aligned size first to ensure safe allocation
        const std::size_t aligned_size =
                ((nbytes + (ALLOC_ALIGN_BYTES - 1)) & ~(ALLOC_ALIGN_BYTES - 1));

        if ((offset_ + aligned_size) > size_) {
            throw std::runtime_error(std::format(
                    "monotonic_alloc::allocate(): Buffer size exceeded. "
                    "offset = {}, num_bytes = {}, aligned_bytes = {}, block_size = {}",
                    offset_,
                    nbytes,
                    aligned_size,
                    size_));
        }
        // Store the current offset for returning...
        void *ptr = std::next(static_cast<char *>(buffer_), static_cast<std::ptrdiff_t>(offset_));
        // Increment the offset for the next allocation using the aligned size
        offset_ += aligned_size;
        return ptr;
    }

    /**
     * Sets the entire buffer to a specific value using CUDA memory operations.
     *
     * @note This method only works with device memory (cudaMalloc) or managed
     * memory (cudaMallocManaged). It will not work with pinned host memory
     * (cudaHostAlloc/ cudaMallocHost) as cudaMemsetAsync cannot operate on host
     * memory.
     *
     * @param[in] val The value to set each byte to
     * @param[in] strm CUDA stream to use for the asynchronous operation (default:
     * 0)
     */
    // NOLINTNEXTLINE(hicpp-use-nullptr,modernize-use-nullptr)
    void memset(const int val, cudaStream_t strm = 0) const {
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemsetAsync(buffer_, val, size_, strm));
    }

    /**
     * Gets the total size of the managed buffer.
     *
     * @return Total buffer size in bytes
     */
    [[nodiscard]]
    std::size_t size() const noexcept {
        return size_;
    }

    /**
     * Gets the current allocation offset within the buffer.
     *
     * @return Current offset in bytes from the start of the buffer
     */
    [[nodiscard]]
    std::size_t offset() const noexcept {
        return offset_;
    }

    /**
     * Gets the base address of the managed buffer.
     *
     * @return Pointer to the start of the buffer, or nullptr if no buffer is
     * allocated
     */
    [[nodiscard]]
    void *address() const noexcept {
        return buffer_;
    }

private:
    void *buffer_{};       //!< Parent allocation
    std::size_t size_{};   //!< Total allocation size
    std::size_t offset_{}; //!< Offset of next allocation
};

} // namespace framework::memory

#endif // FRAMEWORK_MONOTONIC_ALLOC_HPP
