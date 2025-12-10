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

#include <cstring>     // for size_t, memset
#include <limits>      // for numeric_limits
#include <memory>      // for allocator, unique_ptr
#include <numbers>     // for e
#include <string>      // for basic_string, char_traits
#include <tuple>       // for _Swallow_assign, ignore
#include <type_traits> // for is_same_v
#include <utility>     // for move

#include <driver_types.h> // for cudaError, cudaPointerAttributes

#include <gtest/gtest.h> // for CmpHelperNE, Message, TestPartR...

#include <cuda_runtime.h> // for cudaGetErrorString, cudaGetLast...

#include "log/components.hpp"          // for LogLevel, format_component_name
#include "memory/unique_ptr_utils.hpp" // for make_unique_device, make_unique...
#include "utils/core_log.hpp"          // for Core
#include "utils/exceptions.hpp"        // for CudaRuntimeException

namespace {

// Test fixture for unique_ptr_utils tests
class UniquePtrUtilsTest : public ::testing::Test {
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

// Test: Validates DeviceDeleter properly frees device memory
TEST_F(UniquePtrUtilsTest, DeviceDeleterBasicFunctionality) {
    // Allocate device memory directly
    float *device_ptr{};
    ASSERT_EQ(cudaMalloc(&device_ptr, sizeof(float)), cudaSuccess);
    ASSERT_NE(device_ptr, nullptr);

    // Verify the pointer is in device memory
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, device_ptr), cudaSuccess);
    ASSERT_EQ(attributes.type, cudaMemoryTypeDevice);

    // Test deleter functionality
    const framework::memory::DeviceDeleter<float> deleter{};
    EXPECT_NO_THROW(deleter(device_ptr));
}

// Test: Validates PinnedDeleter properly frees pinned host memory
TEST_F(UniquePtrUtilsTest, PinnedDeleterBasicFunctionality) {
    // Allocate pinned host memory directly
    float *pinned_ptr{};
    ASSERT_EQ(cudaMallocHost(&pinned_ptr, sizeof(float)), cudaSuccess);
    ASSERT_NE(pinned_ptr, nullptr);

    // Verify the pointer is in host memory
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, pinned_ptr), cudaSuccess);
    ASSERT_EQ(attributes.type, cudaMemoryTypeHost);

    // Test deleter functionality
    const framework::memory::PinnedDeleter<float> deleter{};
    EXPECT_NO_THROW(deleter(pinned_ptr));
}

// Test: Validates make_unique_device creates proper unique_ptr with device
// memory
TEST_F(UniquePtrUtilsTest, MakeUniqueDeviceBasicAllocation) {
    static constexpr std::size_t NUM_ELEMENTS = 10;

    auto device_ptr = framework::memory::make_unique_device<float>(NUM_ELEMENTS);

    EXPECT_NE(device_ptr.get(), nullptr);

    // Verify the pointer is in device memory
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, device_ptr.get()), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    // Test that it's a proper unique_ptr
    static_assert(std::is_same_v<decltype(device_ptr), framework::memory::UniqueDevicePtr<float>>);
}

// Test: Validates make_unique_pinned creates proper unique_ptr with pinned
// memory
TEST_F(UniquePtrUtilsTest, MakeUniquePinnedBasicAllocation) {
    static constexpr std::size_t NUM_ELEMENTS = 10;

    auto pinned_ptr = framework::memory::make_unique_pinned<float>(NUM_ELEMENTS);

    EXPECT_NE(pinned_ptr.get(), nullptr);

    // Verify the pointer is in host memory
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, pinned_ptr.get()), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

    // Test that it's a proper unique_ptr
    static_assert(std::is_same_v<decltype(pinned_ptr), framework::memory::UniquePinnedPtr<float>>);

    // Test that we can write to and read from the memory
    static constexpr float FIRST_VALUE = 42.0F;
    static constexpr float LAST_VALUE = 99.0F;
    pinned_ptr.get()[0] = FIRST_VALUE;
    pinned_ptr.get()[NUM_ELEMENTS - 1] = LAST_VALUE;

    EXPECT_EQ(pinned_ptr.get()[0], FIRST_VALUE);
    EXPECT_EQ(pinned_ptr.get()[NUM_ELEMENTS - 1], LAST_VALUE);
}

// Test: Validates default parameter behavior (count = 1)
TEST_F(UniquePtrUtilsTest, DefaultParameterBehavior) {
    // Test device allocation with default count
    auto device_ptr = framework::memory::make_unique_device<int>();
    EXPECT_NE(device_ptr.get(), nullptr);

    // Test pinned allocation with default count
    auto pinned_ptr = framework::memory::make_unique_pinned<int>();
    EXPECT_NE(pinned_ptr.get(), nullptr);

    // Verify we can use the single element
    static constexpr int TEST_VALUE = 123;
    pinned_ptr.get()[0] = TEST_VALUE;
    EXPECT_EQ(pinned_ptr.get()[0], TEST_VALUE);
}

// Test: Validates zero-element allocation behavior
TEST_F(UniquePtrUtilsTest, ZeroElementAllocation) {
    // Test device allocation with zero elements
    auto device_ptr = framework::memory::make_unique_device<float>(0);
    // Zero-size allocation may return nullptr or a valid pointer - both are valid
    // No assertion needed as both behaviors are acceptable per CUDA documentation

    // Test pinned allocation with zero elements
    auto pinned_ptr = framework::memory::make_unique_pinned<float>(0);
    // Zero-size allocation may return nullptr or a valid pointer - both are valid
    // No assertion needed as both behaviors are acceptable per CUDA documentation

    // If pointers are not null, they should be safely managed by the unique_ptr
    // (cleanup will be tested by the RAII behavior when unique_ptrs go out of
    // scope)
}

// Test: Validates RAII behavior - automatic cleanup when unique_ptr goes out of
// scope
TEST_F(UniquePtrUtilsTest, RAIIBehavior) {
    void *captured_device_addr = nullptr;
    void *captured_pinned_addr = nullptr;

    static constexpr std::size_t NUM_ELEMENTS = 100;

    {
        // Create unique_ptrs in limited scope
        auto device_ptr = framework::memory::make_unique_device<float>(NUM_ELEMENTS);
        auto pinned_ptr = framework::memory::make_unique_pinned<float>(NUM_ELEMENTS);

        captured_device_addr = device_ptr.get();
        captured_pinned_addr = pinned_ptr.get();

        EXPECT_NE(captured_device_addr, nullptr);
        EXPECT_NE(captured_pinned_addr, nullptr);

        // unique_ptrs should be automatically destroyed here
    }

    // Memory should have been automatically deallocated
    // We can't directly verify this, but CUDA will detect leaks if RAII isn't
    // working
}

// Test: Validates move semantics work correctly
TEST_F(UniquePtrUtilsTest, MoveSemanticsSupport) {
    static constexpr std::size_t NUM_ELEMENTS = 50;

    // Create initial unique_ptrs
    auto device_ptr1 = framework::memory::make_unique_device<int>(NUM_ELEMENTS);
    auto pinned_ptr1 = framework::memory::make_unique_pinned<int>(NUM_ELEMENTS);

    const void *original_device_addr = device_ptr1.get();
    const void *original_pinned_addr = pinned_ptr1.get();

    EXPECT_NE(original_device_addr, nullptr);
    EXPECT_NE(original_pinned_addr, nullptr);

    // Move to new unique_ptrs
    auto device_ptr2 = std::move(device_ptr1);
    auto pinned_ptr2 = std::move(pinned_ptr1);

    // New pointers should have the original addresses (verifies successful move)
    EXPECT_EQ(device_ptr2.get(), original_device_addr);
    EXPECT_EQ(pinned_ptr2.get(), original_pinned_addr);

    // Verify move semantics by checking moved-from objects are in valid state
    // Note: Using reset() instead of get() to avoid static analyzer warnings
    // about moved-from objects
    EXPECT_NO_THROW(device_ptr1.reset());
    EXPECT_NO_THROW(pinned_ptr1.reset());
}

// Test: Validates unique_ptr reset and release functionality
TEST_F(UniquePtrUtilsTest, UniquePtrResetAndRelease) {
    static constexpr std::size_t NUM_ELEMENTS = 25;

    // Test reset functionality
    auto device_ptr = framework::memory::make_unique_device<float>(NUM_ELEMENTS);
    EXPECT_NE(device_ptr.get(), nullptr);

    device_ptr.reset();
    EXPECT_EQ(device_ptr.get(), nullptr);

    // Test release functionality
    auto pinned_ptr = framework::memory::make_unique_pinned<float>(NUM_ELEMENTS);
    float *raw_ptr = pinned_ptr.release();
    EXPECT_NE(raw_ptr, nullptr);
    EXPECT_EQ(pinned_ptr.get(), nullptr);

    // Manually clean up released pointer
    const framework::memory::PinnedDeleter<float> deleter{};
    EXPECT_NO_THROW(deleter(raw_ptr));
}

// Test: Validates different data types work correctly
TEST_F(UniquePtrUtilsTest, MultipleDataTypesSupport) {
    static constexpr std::size_t NUM_ELEMENTS = 10;

    // Test various data types
    auto int_device_ptr = framework::memory::make_unique_device<int>(NUM_ELEMENTS);
    auto float_device_ptr = framework::memory::make_unique_device<float>(NUM_ELEMENTS);
    auto double_device_ptr = framework::memory::make_unique_device<double>(NUM_ELEMENTS);

    auto int_pinned_ptr = framework::memory::make_unique_pinned<int>(NUM_ELEMENTS);
    auto float_pinned_ptr = framework::memory::make_unique_pinned<float>(NUM_ELEMENTS);
    auto double_pinned_ptr = framework::memory::make_unique_pinned<double>(NUM_ELEMENTS);

    // Verify all allocations succeeded
    EXPECT_NE(int_device_ptr.get(), nullptr);
    EXPECT_NE(float_device_ptr.get(), nullptr);
    EXPECT_NE(double_device_ptr.get(), nullptr);
    EXPECT_NE(int_pinned_ptr.get(), nullptr);
    EXPECT_NE(float_pinned_ptr.get(), nullptr);
    EXPECT_NE(double_pinned_ptr.get(), nullptr);

    // Test that we can use the pinned memory
    static constexpr int INT_VALUE = 42;
    static constexpr float FLOAT_VALUE = 3.14F;
    static constexpr double DOUBLE_VALUE = std::numbers::e;

    int_pinned_ptr.get()[0] = INT_VALUE;
    float_pinned_ptr.get()[0] = FLOAT_VALUE;
    double_pinned_ptr.get()[0] = DOUBLE_VALUE;

    EXPECT_EQ(int_pinned_ptr.get()[0], INT_VALUE);
    EXPECT_FLOAT_EQ(float_pinned_ptr.get()[0], FLOAT_VALUE);
    EXPECT_DOUBLE_EQ(double_pinned_ptr.get()[0], DOUBLE_VALUE);
}

// Test: Validates large allocation handling
TEST_F(UniquePtrUtilsTest, LargeAllocationHandling) {
    static constexpr std::size_t LARGE_DEVICE_SIZE = 10UL * 1024 * 1024; // 10M elements
    static constexpr std::size_t LARGE_PINNED_SIZE =
            5UL * 1024 * 1024; // 5M elements (smaller due to host memory limits)

    // Test large device allocation
    auto large_device_ptr = framework::memory::make_unique_device<float>(LARGE_DEVICE_SIZE);
    if (large_device_ptr.get() != nullptr) {
        // Verify it's device memory
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, large_device_ptr.get()), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
    }

    // Test large pinned allocation
    auto large_pinned_ptr = framework::memory::make_unique_pinned<float>(LARGE_PINNED_SIZE);
    if (large_pinned_ptr.get() != nullptr) {
        // Verify it's host memory
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, large_pinned_ptr.get()), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    }
}

// Test: Validates exception handling on allocation failure
TEST_F(UniquePtrUtilsTest, AllocationFailureHandling) {
    // Attempt to allocate impossibly large amounts of memory
    static constexpr std::size_t IMPOSSIBLE_SIZE =
            std::numeric_limits<std::size_t>::max() / sizeof(float);

    EXPECT_THROW(
            { std::ignore = framework::memory::make_unique_device<float>(IMPOSSIBLE_SIZE); },
            framework::utils::CudaRuntimeException);

    EXPECT_THROW(
            { std::ignore = framework::memory::make_unique_pinned<float>(IMPOSSIBLE_SIZE); },
            framework::utils::CudaRuntimeException);

    // Clear any CUDA error state left by the intentional allocation failures
    cudaGetLastError();
}

// Test: Validates memory transfer between device and pinned allocations
TEST_F(UniquePtrUtilsTest, MemoryTransferBetweenAllocations) {
    static constexpr std::size_t NUM_ELEMENTS = 1024;

    // Allocate both device and pinned memory
    auto device_ptr = framework::memory::make_unique_device<float>(NUM_ELEMENTS);
    auto pinned_ptr = framework::memory::make_unique_pinned<float>(NUM_ELEMENTS);

    ASSERT_NE(device_ptr.get(), nullptr);
    ASSERT_NE(pinned_ptr.get(), nullptr);

    // Initialize pinned memory with test data
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        pinned_ptr.get()[i] = static_cast<float>(i);
    }

    // Copy from pinned to device
    ASSERT_EQ(
            cudaMemcpy(
                    device_ptr.get(),
                    pinned_ptr.get(),
                    NUM_ELEMENTS * sizeof(float),
                    cudaMemcpyHostToDevice),
            cudaSuccess);

    // Clear pinned memory to verify copy
    std::memset(pinned_ptr.get(), 0, NUM_ELEMENTS * sizeof(float));

    // Copy from device back to pinned
    ASSERT_EQ(
            cudaMemcpy(
                    pinned_ptr.get(),
                    device_ptr.get(),
                    NUM_ELEMENTS * sizeof(float),
                    cudaMemcpyDeviceToHost),
            cudaSuccess);

    // Verify data integrity
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        EXPECT_EQ(pinned_ptr.get()[i], static_cast<float>(i));
    }
}

// Test: Validates type traits and template instantiation
TEST_F(UniquePtrUtilsTest, TypeTraitsAndTemplateInstantiation) {
    // Test that the type aliases work correctly
    static_assert(std::is_same_v<
                  framework::memory::UniqueDevicePtr<int>,
                  std::unique_ptr<int, framework::memory::DeviceDeleter<int>>>);
    static_assert(std::is_same_v<
                  framework::memory::UniquePinnedPtr<float>,
                  std::unique_ptr<float, framework::memory::PinnedDeleter<float>>>);

    // Test that deleters handle array types correctly
    static constexpr std::size_t ARRAY_SIZE = 10;
    static_assert(std::is_same_v<
                  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
                  framework::memory::DeviceDeleter<int[]>::PtrT,
                  int>);
    static_assert(std::is_same_v<
                  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
                  framework::memory::PinnedDeleter<float[ARRAY_SIZE]>::PtrT,
                  float>);

    // Test template instantiation with different types
    static constexpr std::size_t CHAR_ELEMENTS = 100;
    static constexpr std::size_t DOUBLE_ELEMENTS = 50;

    auto char_device_ptr = framework::memory::make_unique_device<char>(CHAR_ELEMENTS);
    auto double_device_ptr = framework::memory::make_unique_device<double>(DOUBLE_ELEMENTS);

    EXPECT_NE(char_device_ptr.get(), nullptr);
    EXPECT_NE(double_device_ptr.get(), nullptr);
}

// Test: Validates nullptr handling in deleters
TEST_F(UniquePtrUtilsTest, NullptrHandlingInDeleters) {
    // Both deleters should handle nullptr gracefully
    const framework::memory::DeviceDeleter<float> device_deleter{};
    const framework::memory::PinnedDeleter<float> pinned_deleter{};

    EXPECT_NO_THROW(device_deleter(nullptr));
    EXPECT_NO_THROW(pinned_deleter(nullptr));

    // Test with unique_ptr reset to nullptr
    static constexpr std::size_t NUM_ELEMENTS = 10;
    auto device_ptr = framework::memory::make_unique_device<float>(NUM_ELEMENTS);
    auto pinned_ptr = framework::memory::make_unique_pinned<float>(NUM_ELEMENTS);

    EXPECT_NO_THROW(device_ptr.reset(nullptr));
    EXPECT_NO_THROW(pinned_ptr.reset(nullptr));
}

// Test: Validates custom deleter behavior with unique_ptr
TEST_F(UniquePtrUtilsTest, CustomDeleterBehavior) {
    // Test that custom deleters are properly called
    {
        static constexpr std::size_t NUM_ELEMENTS = 5;
        auto device_ptr = framework::memory::make_unique_device<int>(NUM_ELEMENTS);
        auto pinned_ptr = framework::memory::make_unique_pinned<int>(NUM_ELEMENTS);

        const void *device_addr = device_ptr.get();
        const void *pinned_addr = pinned_ptr.get();

        EXPECT_NE(device_addr, nullptr);
        EXPECT_NE(pinned_addr, nullptr);

        // Deleters will be called automatically when unique_ptrs go out of scope
    }

    // Test manual deleter usage
    const framework::memory::DeviceDeleter<float> device_deleter{};
    const framework::memory::PinnedDeleter<float> pinned_deleter{};

    // Allocate memory manually
    float *device_ptr{};
    float *pinned_ptr{};

    ASSERT_EQ(cudaMalloc(&device_ptr, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMallocHost(&pinned_ptr, sizeof(float)), cudaSuccess);

    // Use deleters manually
    EXPECT_NO_THROW(device_deleter(device_ptr));
    EXPECT_NO_THROW(pinned_deleter(pinned_ptr));
}

} // namespace
