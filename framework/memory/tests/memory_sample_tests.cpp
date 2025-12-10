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

/**
 * @file memory_sample_tests.cpp
 * @brief Sample tests for memory library documentation
 */

#include <cstdint>
#include <memory>
#include <utility>

#include <driver_types.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "log/components.hpp"
#include "memory/buffer.hpp"
#include "memory/device_allocators.hpp"
#include "memory/monotonic_alloc.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "utils/core_log.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

template <typename T, typename Alloc> using Buffer = framework::memory::Buffer<T, Alloc>;
using DeviceAlloc = framework::memory::DeviceAlloc;
using PinnedAlloc = framework::memory::PinnedAlloc;
template <std::uint32_t ALIGNMENT, typename Alloc>
using MonotonicAlloc = framework::memory::MonotonicAlloc<ALIGNMENT, Alloc>;
using framework::memory::make_unique_device;
using framework::memory::make_unique_pinned;

TEST(MemorySampleTests, DeviceBuffer) {

    // example-begin device-buffer-1
    // Allocate buffer on GPU device
    Buffer<float, DeviceAlloc> device_buffer(1024);

    // Query buffer properties
    const auto size = device_buffer.size();
    auto *const addr = device_buffer.addr();
    // example-end device-buffer-1

    EXPECT_EQ(size, 1024);
    EXPECT_NE(addr, nullptr);
}

TEST(MemorySampleTests, PinnedBuffer) {

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    // example-begin pinned-buffer-1
    // Allocate pinned host buffer for efficient CPU-GPU transfers
    Buffer<int32_t, PinnedAlloc> pinned_buffer(512);

    // Access data on host
    pinned_buffer.addr()[0] = 42;
    pinned_buffer.addr()[1] = 100;

    const auto first_value = pinned_buffer.addr()[0];
    const auto second_value = pinned_buffer.addr()[1];
    // example-end pinned-buffer-1
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    EXPECT_EQ(first_value, 42);
    EXPECT_EQ(second_value, 100);
}

TEST(MemorySampleTests, BufferCopy) {

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    // example-begin buffer-copy-1
    // Create pinned buffer and initialize data
    Buffer<float, PinnedAlloc> host_buffer(256);
    host_buffer.addr()[0] = 3.14F;

    // Copy data to device buffer
    const Buffer<float, DeviceAlloc> device_buffer(host_buffer);
    // example-end buffer-copy-1
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    EXPECT_EQ(device_buffer.size(), 256);
}

TEST(MemorySampleTests, BufferMove) {

    // example-begin buffer-move-1
    // Create buffer
    Buffer<double, DeviceAlloc> buffer1(128);
    auto *const original_addr = buffer1.addr();

    // Transfer ownership via move
    Buffer<double, DeviceAlloc> buffer2(std::move(buffer1));

    // buffer1 is now empty, buffer2 owns the memory
    auto *const moved_addr = buffer2.addr();
    // example-end buffer-move-1

    EXPECT_EQ(moved_addr, original_addr);
}

TEST(MemorySampleTests, UniqueDevicePtr) {

    // example-begin unique-device-ptr-1
    // Allocate device memory with automatic cleanup
    auto device_ptr = make_unique_device<float>(1024);

    // Memory is automatically freed when device_ptr goes out of scope
    auto *const ptr_value = device_ptr.get();
    // example-end unique-device-ptr-1

    EXPECT_NE(ptr_value, nullptr);
}

TEST(MemorySampleTests, UniquePinnedPtr) {

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    // example-begin unique-pinned-ptr-1
    // Allocate pinned host memory with automatic cleanup
    auto pinned_ptr = make_unique_pinned<int32_t>(2048);

    // Access memory on host
    pinned_ptr.get()[0] = 123;
    const auto value = pinned_ptr.get()[0];
    // example-end unique-pinned-ptr-1
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    EXPECT_EQ(value, 123);
}

TEST(MemorySampleTests, MonotonicAllocator) {

    // example-begin monotonic-alloc-1
    // Create monotonic allocator with 4KB buffer on device
    constexpr std::uint32_t ALIGNMENT = 256;
    MonotonicAlloc<ALIGNMENT, DeviceAlloc> allocator(4096);

    // Perform fast sequential allocations
    void *block1 = allocator.allocate(512);
    void *block2 = allocator.allocate(256);

    // Check current offset (bytes used)
    const auto offset = allocator.offset();
    const auto total_size = allocator.size();
    // example-end monotonic-alloc-1

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);
    EXPECT_GE(offset, 512 + 256);
    EXPECT_EQ(total_size, 4096);
}

TEST(MemorySampleTests, MonotonicAllocReset) {

    // example-begin monotonic-alloc-reset-1
    constexpr std::uint32_t ALIGNMENT = 64;
    MonotonicAlloc<ALIGNMENT, PinnedAlloc> allocator(2048);

    // Allocate some memory
    void *block1 = allocator.allocate(1000);
    const auto offset_before = allocator.offset();

    // Reset allocator to reuse memory
    allocator.reset();
    const auto offset_after = allocator.offset();

    // Memory is available for reuse
    void *block2 = allocator.allocate(1000);
    // example-end monotonic-alloc-reset-1

    EXPECT_GT(offset_before, 0);
    EXPECT_EQ(offset_after, 0);
    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);
}

TEST(MemorySampleTests, AllocatorBasics) {

    // example-begin allocator-basics-1
    // Direct allocator usage
    void *device_mem = DeviceAlloc::allocate(1024);
    void *pinned_mem = PinnedAlloc::allocate(2048);

    // Manual deallocation
    DeviceAlloc::deallocate(device_mem);
    PinnedAlloc::deallocate(pinned_mem);
    // example-end allocator-basics-1

    SUCCEED();
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
