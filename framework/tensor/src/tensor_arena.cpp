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

#include <cstddef>

#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "memory/device_allocators.hpp"
#include "tensor/tensor_arena.hpp"
#include "utils/core_log.hpp"

namespace framework::tensor {

TensorArena::TensorArena(const std::size_t total_bytes, const MemoryType memory_type)
        : total_bytes_(total_bytes), memory_type_(memory_type) {

    if (total_bytes == 0) {
        RT_LOGC_DEBUG(utils::Core::CoreCudaRuntime, "Creating empty TensorArena");
        return;
    }

    // Allocate memory based on type
    if (memory_type == MemoryType::Device) {
        raw_memory_ = memory::DeviceAlloc::allocate(total_bytes);
        RT_LOGC_DEBUG(
                utils::Core::CoreCudaRuntime,
                "Allocated device memory arena: {} bytes at {}",
                total_bytes,
                raw_memory_);
    } else {
        raw_memory_ = memory::PinnedAlloc::allocate(total_bytes);
        RT_LOGC_DEBUG(
                utils::Core::CoreCudaRuntime,
                "Allocated pinned host memory arena: {} bytes at {}",
                total_bytes,
                raw_memory_);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    memory_bytes_ = reinterpret_cast<std::byte *>(raw_memory_);
}

TensorArena::~TensorArena() {
    if (raw_memory_ == nullptr) {
        return;
    }

    RT_LOGC_DEBUG(
            utils::Core::CoreCudaRuntime,
            "Destroying TensorArena: {} bytes at {}",
            total_bytes_,
            raw_memory_);

    // Deallocate based on type
    // Note: memory::DeviceAlloc/memory::PinnedAlloc may throw, but destructor must not propagate
    try {
        if (memory_type_ == MemoryType::Device) {
            memory::DeviceAlloc::deallocate(raw_memory_);
        } else {
            memory::PinnedAlloc::deallocate(raw_memory_);
        }
        // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
        // Suppress exceptions in destructor - cannot safely propagate
    }

    raw_memory_ = nullptr;
    memory_bytes_ = nullptr;
}

TensorArena::TensorArena(TensorArena &&other) noexcept
        : raw_memory_(other.raw_memory_), memory_bytes_(other.memory_bytes_),
          total_bytes_(other.total_bytes_), memory_type_(other.memory_type_) {
    other.raw_memory_ = nullptr;
    other.memory_bytes_ = nullptr;
    other.total_bytes_ = 0;
}

TensorArena &TensorArena::operator=(TensorArena &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    // Clean up existing memory
    if (raw_memory_ != nullptr) {
        // Note: memory::DeviceAlloc/memory::PinnedAlloc may throw, but noexcept must not propagate
        try {
            if (memory_type_ == MemoryType::Device) {
                memory::DeviceAlloc::deallocate(raw_memory_);
            } else {
                memory::PinnedAlloc::deallocate(raw_memory_);
            }
            // NOLINTNEXTLINE(bugprone-empty-catch)
        } catch (...) {
            // Suppress exceptions in noexcept move operator - cannot safely propagate
        }
    }

    // Move from other
    raw_memory_ = other.raw_memory_;
    memory_bytes_ = other.memory_bytes_;
    total_bytes_ = other.total_bytes_;
    memory_type_ = other.memory_type_;

    // Reset other
    other.raw_memory_ = nullptr;
    other.memory_bytes_ = nullptr;
    other.total_bytes_ = 0;

    return *this;
}

} // namespace framework::tensor
