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

#ifndef FRAMEWORK_BUFFER_HPP
#define FRAMEWORK_BUFFER_HPP

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "memory/memcpy_helper.hpp"
#include "utils/error_macros.hpp"
#include "utils/exceptions.hpp"

namespace framework::memory {

/**
 * Concept to identify host-accessible allocators
 *
 * An allocator is considered host-accessible if it's not a DeviceAlloc.
 * This allows for direct CPU access to the allocated memory.
 *
 * @tparam TAlloc The allocator type to check
 */
template <typename TAlloc>
concept HostAccessible = !std::same_as<TAlloc, DeviceAlloc>;

/**
 * Generic buffer class for managing memory allocations with different allocator
 * types
 *
 * This class provides a RAII wrapper for memory buffers that can be allocated
 * using different allocators (host, device, pinned, etc.). It supports copy and
 * move semantics with automatic memory management and CUDA-aware memory
 * transfers.
 *
 * @tparam T The element type stored in the buffer
 * @tparam TAlloc The allocator type used for memory allocation/deallocation
 */
template <typename T, class TAlloc> class Buffer final {
    // Allow all buffer template instantiations to access each other's private
    // members
    template <typename U, class UAlloc> friend class Buffer;

public:
    using ElementType = T;        //!< Element type stored in the buffer
    using AllocatorType = TAlloc; //!< Allocator type used for memory management

    /**
     * Default constructor creates an empty buffer
     */
    Buffer() = default;

    /**
     * Constructor that allocates memory for the specified number of elements
     *
     * @param[in] num_elements Number of elements to allocate space for
     * @throw std::bad_alloc if allocation fails
     */
    explicit Buffer(const std::size_t num_elements)
            : addr_(static_cast<ElementType *>(AllocatorType::allocate(num_elements * sizeof(T)))),
              size_(num_elements) {}

    /**
     * Destructor deallocates the buffer memory
     */
    ~Buffer() {
        try {
            deallocate_buffer();
        } catch (const std::exception &e) {
            RT_LOGC_ERROR(utils::Core::CoreCudaRuntime, "Error deallocating buffer: {}", e.what());
        } catch (...) {
            RT_LOGC_ERROR(utils::Core::CoreCudaRuntime, "Unknown error deallocating buffer");
        }
    }

    /**
     * Manually deallocate the buffer memory and reset internal state
     *
     * This method can be called multiple times safely. After calling this method,
     * the buffer will be in an empty state.
     */
    void deallocate_buffer() {
        if (addr_) {
            AllocatorType::deallocate(addr_);
            addr_ = nullptr;
            size_ = 0;
        }
    }

    /**
     * Cross-allocator copy constructor
     *
     * Creates a new buffer by copying data from another buffer with a different
     * allocator type. Uses CUDA memory copy operations to transfer data between
     * different memory spaces.
     *
     * @tparam TAlloc2 The allocator type of the source buffer
     * @param[in] other The source buffer to copy from
     * @throw utils::CudaRuntimeException if CUDA memory copy fails
     */
    template <class TAlloc2>
    explicit Buffer(const Buffer<T, TAlloc2> &other)
            : addr_(static_cast<ElementType *>(AllocatorType::allocate(other.size_ * sizeof(T)))),
              size_(other.size_) {
        if (const cudaError_t error = cudaMemcpy(
                    addr_, other.addr_, sizeof(T) * size_, MemcpyHelper<TAlloc, TAlloc2>::KIND);
            error != cudaSuccess) {
            AllocatorType::deallocate(addr_);
            addr_ = nullptr;
            size_ = 0;
            throw utils::CudaRuntimeException(error);
        }
    }

    /**
     * Copy constructor
     *
     * Creates a new buffer by copying data from another buffer with the same
     * allocator type.
     *
     * @param[in] other The source buffer to copy from
     * @throw utils::CudaRuntimeException if CUDA memory copy fails
     */
    Buffer(const Buffer &other)
            : addr_(static_cast<ElementType *>(AllocatorType::allocate(other.size_ * sizeof(T)))),
              size_(other.size_) {
        if (const cudaError_t error = cudaMemcpy(
                    addr_, other.addr_, sizeof(T) * size_, MemcpyHelper<TAlloc, TAlloc>::KIND);
            error != cudaSuccess) {
            AllocatorType::deallocate(addr_);
            addr_ = nullptr;
            size_ = 0;
            throw utils::CudaRuntimeException(error);
        }
    }

    /**
     * Copy assignment operator
     *
     * Copies data from another buffer with the same allocator type.
     * Uses copy-and-swap idiom for strong exception safety.
     *
     * @param[in] other The source buffer to copy from
     * @return Reference to this buffer
     * @throw utils::CudaRuntimeException if CUDA memory copy fails
     */
    Buffer &operator=(const Buffer &other) {
        if (this == &other) {
            return *this;
        }

        Buffer tmp(other); // may throw, but leaves *this unchanged
        std::swap(addr_, tmp.addr_);
        std::swap(size_, tmp.size_);
        return *this;
    }

    /**
     * Move constructor
     *
     * Transfers ownership of the buffer from another buffer, leaving the source
     * buffer empty.
     *
     * @param[in,out] other The source buffer to move from (will be left empty)
     */
    Buffer(Buffer &&other) noexcept
            : addr_(std::exchange(other.addr_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    /**
     * Cross-allocator move constructor (deleted)
     *
     * Cross-allocator moves are prohibited because memory allocated with one
     * allocator must be deallocated with the same allocator. Moving memory
     * between allocators would violate this contract and lead to undefined
     * behavior.
     *
     * For cross-allocator operations, use copy semantics instead:
     * - Buffer<T, DestAlloc> dest(source);  // Copy constructor
     *
     * @tparam TAlloc2 The allocator type of the source buffer
     * @param[in,out] other The source buffer (this operation is not allowed)
     */
    template <class TAlloc2> explicit Buffer(Buffer<T, TAlloc2> &&other) = delete;

    /**
     * Constructor from std::vector
     *
     * Creates a buffer by copying data from a std::vector using CUDA memory
     * operations.
     *
     * @param[in] src_vec The source vector to copy data from
     * @throw utils::CudaRuntimeException if CUDA memory copy fails
     */
    explicit Buffer(const std::vector<T> &src_vec)
            : addr_(static_cast<ElementType *>(
                      AllocatorType::allocate(src_vec.size() * sizeof(T)))),
              size_(src_vec.size()) {
        if (const cudaError_t error =
                    cudaMemcpy(addr_, src_vec.data(), sizeof(T) * size_, cudaMemcpyDefault);
            error != cudaSuccess) {
            AllocatorType::deallocate(addr_);
            addr_ = nullptr;
            size_ = 0;
            throw utils::CudaRuntimeException(error);
        }
    }

    /**
     * Move assignment operator
     *
     * Transfers ownership of the buffer from another buffer, properly
     * deallocating any existing memory in this buffer first.
     *
     * @param[in,out] other The source buffer to move from (will be left empty)
     * @return Reference to this buffer
     */
    Buffer &operator=(Buffer &&other) noexcept {
        // Self-assignment protection
        if (this == &other) {
            return *this;
        }
        try {
            if (addr_) {
                AllocatorType::deallocate(addr_);
            }
        } catch (const std::exception &e) {
            RT_LOGC_ERROR(utils::Core::CoreCudaRuntime, "Error deallocating buffer: {}", e.what());
        } catch (...) {
            RT_LOGC_ERROR(utils::Core::CoreCudaRuntime, "Unknown error deallocating buffer");
        }
        addr_ = std::exchange(other.addr_, nullptr);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    /**
     * Get mutable pointer to the buffer memory
     *
     * @return Pointer to the first element of the buffer, or nullptr if empty
     */
    [[nodiscard]]
    ElementType *addr() noexcept {
        return addr_;
    }

    /**
     * Get const pointer to the buffer memory
     *
     * @return Const pointer to the first element of the buffer, or nullptr if
     * empty
     */
    [[nodiscard]]
    const ElementType *addr() const noexcept {
        return addr_;
    }

    /**
     * Get the number of elements in the buffer
     *
     * @return Number of elements the buffer can hold
     */
    [[nodiscard]]
    std::size_t size() const noexcept {
        return size_;
    }

    /**
     * Indexed access operator for host-accessible buffers
     *
     * Provides unchecked array-like access to buffer elements. Only enabled for
     * allocators that allow host access (i.e., not device allocators).
     *
     * @param[in] idx Index of the element to access
     * @return Reference to the element at the specified index
     * @pre idx < size()
     * @note This operator is only available for host-accessible allocators
     * @note No bounds checking is performed. Use at() for bounds-checked access
     */
    [[nodiscard]]
    ElementType &operator[](const std::size_t idx)
        requires HostAccessible<TAlloc>
    {
        return *std::next(addr_, static_cast<std::ptrdiff_t>(idx));
    }

    /**
     * Read-only element access operator
     *
     * Provides unchecked const access to elements in the buffer. Only available
     * for allocators that allow host access (i.e., not device allocators).
     *
     * @param[in] idx Index of the element to access
     * @return Const reference to the element at the specified index
     * @pre idx < size()
     * @note This operator is only available for host-accessible allocators
     * @note No bounds checking is performed. Use at() for bounds-checked access
     */
    [[nodiscard]]
    const ElementType &operator[](const std::size_t idx) const
        requires HostAccessible<TAlloc>
    {
        return *std::next(addr_, static_cast<std::ptrdiff_t>(idx));
    }

    /**
     * Bounds-checked element access
     *
     * Provides mutable access to elements in the buffer with bounds checking.
     * Only available for allocators that allow host access.
     *
     * @param[in] idx Index of the element to access
     * @return Reference to the element at the specified index
     * @throw std::out_of_range if idx >= size()
     * @note This function is only available for host-accessible allocators
     */
    [[nodiscard]]
    ElementType &at(const std::size_t idx)
        requires HostAccessible<TAlloc>
    {
        FRAMEWORK_NV_THROW_IF(idx >= size_, std::out_of_range, "buffer index out of range");
        return *std::next(addr_, static_cast<std::ptrdiff_t>(idx));
    }

    /**
     * Bounds-checked read-only element access
     *
     * Provides const access to elements in the buffer with bounds checking.
     * Only available for allocators that allow host access.
     *
     * @param[in] idx Index of the element to access
     * @return Const reference to the element at the specified index
     * @throw std::out_of_range if idx >= size()
     * @note This function is only available for host-accessible allocators
     */
    [[nodiscard]]
    const ElementType &at(const std::size_t idx) const
        requires HostAccessible<TAlloc>
    {
        FRAMEWORK_NV_THROW_IF(idx >= size_, std::out_of_range, "buffer index out of range");
        return *std::next(addr_, static_cast<std::ptrdiff_t>(idx));
    }

private:
    ElementType *addr_{}; //!< Pointer to the allocated memory
    std::size_t size_{};  //!< Number of elements in the buffer
};

/**
 * Abstract base class for buffer wrappers
 *
 * This polymorphic design allows silent mixing of different allocator types,
 * which can lead to memory corruption and undefined behavior. The type erasure
 * hides critical allocator information needed for safe memory management.
 *
 * **Dangerous Example:**
 * ```cpp
 * std::unique_ptr<BufferWrapper> device_wrapper =
 *     std::make_unique<BufferImpl<DeviceAlloc>>(1024);
 * std::unique_ptr<BufferWrapper> pinned_wrapper =
 *     std::make_unique<BufferImpl<PinnedAlloc>>(1024);
 *
 * // Problem: Silent mixing of allocator types!
 * device_wrapper = std::move(pinned_wrapper);
 * // Now device_wrapper contains pinned memory but user expects device memory
 * // Destructor will call DeviceAlloc::deallocate() on pinned memory -> UB
 * ```
 *
 * **Recommended Alternative - Variant-Based Approach:**
 * ```cpp
 * using buffer_variant = std::variant<
 *     Buffer<uint8_t, DeviceAlloc>,
 *     Buffer<uint8_t, PinnedAlloc>
 * >;
 *
 * // Type-safe usage:
 * buffer_variant device_buffer = Buffer<uint8_t, DeviceAlloc>(1024);
 * buffer_variant pinned_buffer = Buffer<uint8_t, PinnedAlloc>(1024);
 *
 * // Compile-time error prevents dangerous mixing:
 * // device_buffer = pinned_buffer;  // ERROR: types don't match
 *
 * // Safe access with std::visit:
 * std::visit([](auto& buf) {
 *     void* addr = buf.addr();
 * }, device_buffer);
 * ```
 *
 * **Why Variant is Better:**
 * - Compile-time type safety prevents allocator mixing
 * - No virtual function overhead
 * - Clear type information preserved
 * - std::visit provides type-safe polymorphic operations
 * - No risk of memory corruption from allocator mismatches
 *
 * **If You Must Use This Class:**
 * - Never use `auto` with BufferWrapper pointers
 * - Always use explicit types: `std::unique_ptr<BufferWrapper>`
 * - Document allocator types clearly in variable names
 * - Consider runtime type checking for additional safety
 *
 * Consider using std::variant<Buffer<T, Alloc>...> for type safety
 */
class BufferWrapper {
public:
    virtual ~BufferWrapper() = default;

    /**
     * Default constructor
     */
    BufferWrapper() = default;

    /**
     * Copy constructor
     */
    BufferWrapper(const BufferWrapper &) = default;

    /**
     * Move constructor
     */
    BufferWrapper(BufferWrapper &&) = default;

    /**
     * Copy assignment operator
     *
     * @return Reference to this buffer wrapper
     */
    BufferWrapper &operator=(const BufferWrapper &) = default;

    /**
     * Move assignment operator
     *
     * @return Reference to this buffer wrapper
     */
    BufferWrapper &operator=(BufferWrapper &&) = default;
    /**
     * Get the address of the buffer
     *
     * @return Pointer to the first element of the buffer
     */
    [[nodiscard]] virtual void *addr() = 0;
};

/**
 * Concrete implementation of BufferWrapper for a specific allocator type
 *
 * This class is used to store a buffer of type Buffer<uint8_t, AllocType> in a
 * single container. It is used to store different types of buffers in a single
 * container.
 */
template <typename AllocType> class BufferImpl final : public BufferWrapper {
public:
    /**
     * Constructor
     *
     * @param[in] size The size of the buffer
     */
    explicit BufferImpl(const std::size_t size) : buffer_(size) {}

    /**
     * Get the address of the buffer
     *
     * @return Pointer to the first element of the buffer
     */
    [[nodiscard]] void *addr() override { return buffer_.addr(); }

private:
    Buffer<std::uint8_t, AllocType> buffer_;
};

} // namespace framework::memory

#endif // FRAMEWORK_BUFFER_HPP
