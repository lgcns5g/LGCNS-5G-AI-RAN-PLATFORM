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

#include <algorithm> // for max
#include <cstddef>   // for size_t, byte
#include <cstring>   // for memset
#include <limits>    // for numeric_limits
#include <span>      // for span
#include <string>    // for allocator, basic_string, char_...
#include <tuple>     // for _Swallow_assign, ignore
#include <vector>    // for vector

#include <driver_types.h> // for cudaPointerAttributes, cudaError

#include <gtest/gtest.h> // for Message, TestPartResult, CmpHe...

#include <cuda_runtime.h> // for cudaPointerGetAttributes, cuda...

#include "memory/device_allocators.hpp" // for DeviceAlloc, PinnedAlloc
#include "utils/exceptions.hpp"         // for CudaRuntimeException

namespace {

// Test fixture for device allocator tests
class DeviceAllocatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context
        const cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            FAIL() << "CUDA device not available: " << cudaGetErrorString(error);
        }
    }

    void TearDown() override {
        // Note: We don't call cudaDeviceReset() here as it destroys the CUDA
        // context and interferes with subsequent tests in the same executable. CUDA
        // will clean up automatically when the process exits.
    }
};

// Test: Validates DeviceAlloc can allocate and deallocate device memory
TEST_F(DeviceAllocatorsTest, DeviceAllocBasicAllocation) {
    static constexpr std::size_t ALLOC_SIZE = 1024; // 1KB

    void *ptr = framework::memory::DeviceAlloc::allocate(ALLOC_SIZE);

    EXPECT_NE(ptr, nullptr);

    // Verify the pointer is in device memory space
    cudaPointerAttributes attributes{};
    const cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    EXPECT_EQ(error, cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    // Clean up
    EXPECT_NO_THROW(framework::memory::DeviceAlloc::deallocate(ptr));
}

// Test: Validates DeviceAlloc handles zero-byte allocations correctly
TEST_F(DeviceAllocatorsTest, DeviceAllocZeroBytes) {
    void *ptr = framework::memory::DeviceAlloc::allocate(0);

    // CUDA allows zero-byte allocations and may return nullptr or a valid pointer
    // Both behaviors are acceptable according to CUDA documentation

    // If a pointer was returned, it should be deallocatable
    if (ptr != nullptr) {
        EXPECT_NO_THROW(framework::memory::DeviceAlloc::deallocate(ptr));
    }
}

// Test: Validates DeviceAlloc handles large allocations appropriately
TEST_F(DeviceAllocatorsTest, DeviceAllocLargeAllocation) {
    static constexpr std::size_t LARGE_SIZE = 100ULL * 1024 * 1024; // 100MB

    void *ptr = nullptr;
    EXPECT_NO_THROW(ptr = framework::memory::DeviceAlloc::allocate(LARGE_SIZE));

    if (ptr != nullptr) {
        // Verify the allocation is in device memory
        cudaPointerAttributes attributes{};
        const cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
        EXPECT_EQ(error, cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

        EXPECT_NO_THROW(framework::memory::DeviceAlloc::deallocate(ptr));
    }
}

// Test: Validates DeviceAlloc throws exception on allocation failure
TEST_F(DeviceAllocatorsTest, DeviceAllocFailureHandling) {
    // Attempt to allocate an impossibly large amount of memory
    static constexpr std::size_t IMPOSSIBLE_SIZE = std::numeric_limits<std::size_t>::max();

    EXPECT_THROW(
            { std::ignore = framework::memory::DeviceAlloc::allocate(IMPOSSIBLE_SIZE); },
            framework::utils::CudaRuntimeException);

    // Clear any CUDA error state left by the intentional allocation failure
    cudaGetLastError();
}

// Test: Validates DeviceAlloc handles multiple allocations correctly
TEST_F(DeviceAllocatorsTest, DeviceAllocMultipleAllocations) {
    static constexpr std::size_t NUM_ALLOCATIONS = 10;
    static constexpr std::size_t ALLOCATION_SIZE = 1024; // 1KB each

    std::vector<void *> pointers{};
    pointers.reserve(NUM_ALLOCATIONS);

    // Allocate multiple blocks
    for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
        void *ptr = framework::memory::DeviceAlloc::allocate(ALLOCATION_SIZE);
        EXPECT_NE(ptr, nullptr);
        pointers.push_back(ptr);
    }

    // Verify all pointers are unique
    for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
        for (std::size_t j = i + 1; j < NUM_ALLOCATIONS; ++j) {
            EXPECT_NE(pointers[i], pointers[j]);
        }
    }

    // Deallocate all blocks
    for (void *ptr : pointers) {
        EXPECT_NO_THROW(framework::memory::DeviceAlloc::deallocate(ptr));
    }
}

// Test: Validates PinnedAlloc can allocate and deallocate host pinned memory
TEST_F(DeviceAllocatorsTest, PinnedAllocBasicAllocation) {
    static constexpr std::size_t ALLOC_SIZE = 1024; // 1KB

    void *ptr = framework::memory::PinnedAlloc::allocate(ALLOC_SIZE);

    EXPECT_NE(ptr, nullptr);

    // Verify the pointer is in host pinned memory space
    cudaPointerAttributes attributes{};
    const cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    EXPECT_EQ(error, cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

    // Test that we can write to and read from the memory
    auto *byte_ptr = static_cast<std::byte *>(ptr);
    static constexpr std::byte TEST_VALUE{0xAB};

    // Use std::span for safe memory access without pointer arithmetic
    const std::span<std::byte> memory_span{byte_ptr, ALLOC_SIZE};

    memory_span[0] = TEST_VALUE;
    memory_span[ALLOC_SIZE - 1] = TEST_VALUE;

    EXPECT_EQ(memory_span[0], TEST_VALUE);
    EXPECT_EQ(memory_span[ALLOC_SIZE - 1], TEST_VALUE);

    // Clean up
    EXPECT_NO_THROW(framework::memory::PinnedAlloc::deallocate(ptr));
}

// Test: Validates PinnedAlloc handles zero-byte allocations correctly
TEST_F(DeviceAllocatorsTest, PinnedAllocZeroBytes) {
    void *ptr = framework::memory::PinnedAlloc::allocate(0);

    // CUDA allows zero-byte allocations and may return nullptr or a valid pointer

    // If a pointer was returned, it should be deallocatable
    if (ptr != nullptr) {
        EXPECT_NO_THROW(framework::memory::PinnedAlloc::deallocate(ptr));
    }
}

// Test: Validates PinnedAlloc handles large allocations appropriately
TEST_F(DeviceAllocatorsTest, PinnedAllocLargeAllocation) {
    static constexpr std::size_t LARGE_SIZE =
            50ULL * 1024 * 1024; // 50MB (smaller than device test due to host memory limits)

    void *ptr = nullptr;
    EXPECT_NO_THROW(ptr = framework::memory::PinnedAlloc::allocate(LARGE_SIZE));

    if (ptr != nullptr) {
        // Verify the allocation is in host pinned memory
        cudaPointerAttributes attributes{};
        const cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
        EXPECT_EQ(error, cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

        EXPECT_NO_THROW(framework::memory::PinnedAlloc::deallocate(ptr));
    }
}

// Test: Validates PinnedAlloc throws exception on allocation failure
TEST_F(DeviceAllocatorsTest, PinnedAllocFailureHandling) {
    // Attempt to allocate an impossibly large amount of memory
    static constexpr std::size_t IMPOSSIBLE_SIZE = std::numeric_limits<std::size_t>::max();

    EXPECT_THROW(
            { std::ignore = framework::memory::PinnedAlloc::allocate(IMPOSSIBLE_SIZE); },
            framework::utils::CudaRuntimeException);

    // Clear any CUDA error state left by the intentional allocation failure
    cudaGetLastError();
}

// Test: Validates PinnedAlloc handles multiple allocations correctly
TEST_F(DeviceAllocatorsTest, PinnedAllocMultipleAllocations) {
    static constexpr std::size_t NUM_ALLOCATIONS = 10;
    static constexpr std::size_t ALLOCATION_SIZE = 1024; // 1KB each

    std::vector<void *> pointers{};
    pointers.reserve(NUM_ALLOCATIONS);

    // Allocate multiple blocks
    for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
        void *ptr = framework::memory::PinnedAlloc::allocate(ALLOCATION_SIZE);
        EXPECT_NE(ptr, nullptr);
        pointers.push_back(ptr);
    }

    // Verify all pointers are unique
    for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
        for (std::size_t j = i + 1; j < NUM_ALLOCATIONS; ++j) {
            EXPECT_NE(pointers[i], pointers[j]);
        }
    }

    // Deallocate all blocks
    for (void *ptr : pointers) {
        EXPECT_NO_THROW(framework::memory::PinnedAlloc::deallocate(ptr));
    }
}

// Test: Validates memory transfer between device and pinned allocations
TEST_F(DeviceAllocatorsTest, DeviceAndPinnedMemoryTransfer) {
    static constexpr std::size_t DATA_SIZE = 1024 * sizeof(float);
    static constexpr std::size_t NUM_ELEMENTS = DATA_SIZE / sizeof(float);

    // Allocate host pinned memory
    void *host_ptr = framework::memory::PinnedAlloc::allocate(DATA_SIZE);
    const std::span<std::byte> host_span{static_cast<std::byte *>(host_ptr), DATA_SIZE};
    EXPECT_NE(host_ptr, nullptr);

    // Allocate device memory
    void *device_ptr = framework::memory::DeviceAlloc::allocate(DATA_SIZE);
    EXPECT_NE(device_ptr, nullptr);

    // Initialize host data
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        host_span[i] = static_cast<std::byte>(i);
    }

    // Copy host to device
    cudaError_t error = cudaMemcpy(device_ptr, host_ptr, DATA_SIZE, cudaMemcpyHostToDevice);
    EXPECT_EQ(error, cudaSuccess);

    // Clear host data to verify copy
    std::memset(host_ptr, 0, DATA_SIZE);

    // Copy device back to host
    error = cudaMemcpy(host_ptr, device_ptr, DATA_SIZE, cudaMemcpyDeviceToHost);
    EXPECT_EQ(error, cudaSuccess);

    // Verify data integrity
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        EXPECT_EQ(host_span[i], static_cast<std::byte>(i));
    }

    // Clean up
    EXPECT_NO_THROW(framework::memory::DeviceAlloc::deallocate(device_ptr));
    EXPECT_NO_THROW(framework::memory::PinnedAlloc::deallocate(host_ptr));
}

// Test: Validates allocator interface consistency
TEST_F(DeviceAllocatorsTest, AllocatorInterfaceConsistency) {
    static constexpr std::size_t TEST_SIZE = 512;

    // Test that both allocators follow the same interface pattern
    void *device_ptr = nullptr;
    void *pinned_ptr = nullptr;

    // Both should have static allocate methods that return void*
    EXPECT_NO_THROW(device_ptr = framework::memory::DeviceAlloc::allocate(TEST_SIZE));
    EXPECT_NO_THROW(pinned_ptr = framework::memory::PinnedAlloc::allocate(TEST_SIZE));

    EXPECT_NE(device_ptr, nullptr);
    EXPECT_NE(pinned_ptr, nullptr);

    // Both should have static deallocate methods that accept void*
    EXPECT_NO_THROW(framework::memory::DeviceAlloc::deallocate(device_ptr));
    EXPECT_NO_THROW(framework::memory::PinnedAlloc::deallocate(pinned_ptr));
}

// Test: Validates proper handling of nullptr deallocation
TEST_F(DeviceAllocatorsTest, NullptrDeallocation) {
    // Both allocators should handle nullptr deallocation gracefully
    // CUDA functions cudaFree and cudaFreeHost should handle nullptr
    EXPECT_NO_THROW(framework::memory::DeviceAlloc::deallocate(nullptr));
    EXPECT_NO_THROW(framework::memory::PinnedAlloc::deallocate(nullptr));
}

} // namespace
