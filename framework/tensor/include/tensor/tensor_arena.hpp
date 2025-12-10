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

#ifndef FRAMEWORK_CORE_TENSOR_ARENA_HPP
#define FRAMEWORK_CORE_TENSOR_ARENA_HPP

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <wise_enum.h>

namespace framework::tensor {

/**
 * @brief Memory allocation type for tensor arenas
 */
enum class MemoryType {
    Device,    //!< GPU device memory
    HostPinned //!< CPU pinned (page-locked) memory
};

} // namespace framework::tensor

// WISE_ENUM_ADAPT must be called at global namespace scope for ADL to work
WISE_ENUM_ADAPT(framework::tensor::MemoryType, Device, HostPinned)

namespace framework::tensor {

/**
 * @brief Type-safe memory arena for tensor data
 *
 * Provides type-safe allocation and access to tensor memory regions.
 * Supports both device and pinned host memory allocation.
 */
class TensorArena final {
public:
    /**
     * @brief Constructor - allocates memory arena
     *
     * @param[in] total_bytes Total size of arena in bytes
     * @param[in] memory_type Type of memory to allocate (Device or HostPinned)
     */
    explicit TensorArena(std::size_t total_bytes, MemoryType memory_type = MemoryType::Device);

    /**
     * @brief Destructor - cleanup memory
     */
    ~TensorArena();

    // Non-copyable, movable
    TensorArena(const TensorArena &) = delete;
    TensorArena &operator=(const TensorArena &) = delete;
    /**
     * Move constructor.
     * @param other Arena to move from
     */
    TensorArena(TensorArena &&other) noexcept;
    /**
     * Move assignment operator.
     * @param other Arena to move from
     * @return Reference to this arena
     */
    TensorArena &operator=(TensorArena &&other) noexcept;

    /**
     * @brief Return an already allocated memory region at specific offset
     *
     * @tparam T Type to allocate
     * @param[in] offset_bytes Byte offset from arena start
     * @return Type-safe pointer to allocated region
     * @throws std::runtime_error if allocation would exceed arena bounds
     */
    template <typename T> T *allocate_at(const std::size_t offset_bytes) {
        if (offset_bytes > total_bytes_ || sizeof(T) > total_bytes_ - offset_bytes) {
            throw std::runtime_error("Tensor allocation exceeds arena bounds");
        }

        // Type-safe: std::byte* arithmetic is well-defined
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        std::byte *location = memory_bytes_ + offset_bytes;

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T *>(location);
    }

    /**
     * @brief Get raw memory pointer for external APIs
     * @return Pointer to raw memory
     */
    [[nodiscard]] void *raw_ptr() { return raw_memory_; }

    /**
     * @brief Get const raw memory pointer
     * @return Const pointer to raw memory
     */
    [[nodiscard]] const void *raw_ptr() const { return raw_memory_; }

    /**
     * @brief Get mutable raw memory pointer from const context
     *
     * Provides mutable access to arena memory from const member functions when
     * the operation is logically const but requires mutable pointer for external
     * APIs (e.g., CUDA memory transfer operations where dst must be non-const).
     *
     * @return Mutable pointer to raw memory
     */
    [[nodiscard]] void *raw_ptr_mutable() const { return raw_memory_; }

    /**
     * @brief Get total bytes allocated for this arena
     * @return Total arena size in bytes
     */
    [[nodiscard]] std::size_t total_bytes() const { return total_bytes_; }

    /**
     * @brief Get memory type of this arena
     * @return Memory type (Device or HostPinned)
     */
    [[nodiscard]] MemoryType memory_type() const { return memory_type_; }

private:
    void *raw_memory_{nullptr};                  //!< Raw memory pointer (void* for CUDA)
    std::byte *memory_bytes_{nullptr};           //!< Typed memory bytes for arithmetic
    std::size_t total_bytes_{0};                 //!< Total arena size
    MemoryType memory_type_{MemoryType::Device}; //!< Type of memory
};

} // namespace framework::tensor

#endif // FRAMEWORK_CORE_TENSOR_ARENA_HPP
