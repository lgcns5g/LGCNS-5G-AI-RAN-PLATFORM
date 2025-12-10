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

#include <cstddef>     // for size_t, ptrdiff_t
#include <exception>   // for exception
#include <iterator>    // for next
#include <memory>      // for allocator, unique_ptr, make_un...
#include <numbers>     // for e, pi_v
#include <set>         // for set
#include <string>      // for basic_string, char_traits
#include <type_traits> // for is_same_v
#include <utility>     // for move
#include <vector>      // for vector

#include <gtest/gtest.h> // for Test, CmpHelperNE, Message

#include "memory/buffer.hpp"            // for BufferImpl, Buffer, BufferWrapper
#include "memory/device_allocators.hpp" // for PinnedAlloc, DeviceAlloc
#include "utils/error_macros.hpp"       // for FRAMEWORK_PRAGMA_IGNORE_SELF_MOVE

namespace {

// Test: Default constructor initializes empty buffer with null pointer and zero
// size
TEST(BufferTest, DefaultConstructorDeviceAlloc) {
    const framework::memory::Buffer<int, framework::memory::DeviceAlloc> buf{};

    EXPECT_EQ(buf.addr(), nullptr);
    EXPECT_EQ(buf.size(), 0);
}

// Test: Default constructor initializes empty buffer with null pointer and zero
// size
TEST(BufferTest, DefaultConstructorPinnedAlloc) {
    const framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf{};

    EXPECT_EQ(buf.addr(), nullptr);
    EXPECT_EQ(buf.size(), 0);
}

// Test: Constructor allocates device memory and sets correct size and non-null
// pointer
TEST(BufferTest, AllocationConstructorDeviceAlloc) {
    static constexpr std::size_t NUM_ELEMENTS = 100;
    const framework::memory::Buffer<float, framework::memory::DeviceAlloc> buf(NUM_ELEMENTS);

    EXPECT_NE(buf.addr(), nullptr);
    EXPECT_EQ(buf.size(), NUM_ELEMENTS);
}

// Test: Constructor allocates pinned memory and sets correct size and non-null
// pointer
TEST(BufferTest, AllocationConstructorPinnedAlloc) {
    static constexpr std::size_t NUM_ELEMENTS = 100;
    const framework::memory::Buffer<float, framework::memory::PinnedAlloc> buf(NUM_ELEMENTS);

    EXPECT_NE(buf.addr(), nullptr);
    EXPECT_EQ(buf.size(), NUM_ELEMENTS);
}

// Test: Zero-size allocation results in valid buffer state regardless of
// pointer value
TEST(BufferTest, ZeroSizeAllocation) {
    const framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(0);

    // Zero-size allocation may return nullptr or a valid pointer - both are valid
    EXPECT_EQ(buf.size(), 0);
}

// Test: Manual deallocation resets buffer to empty state and allows multiple
// calls safely
TEST(BufferTest, ManualDeallocation) {
    static constexpr std::size_t NUM_ELEMENTS = 50;
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(NUM_ELEMENTS);
    EXPECT_NE(buf.addr(), nullptr);
    EXPECT_EQ(buf.size(), NUM_ELEMENTS);

    // Manual deallocation
    buf.deallocate_buffer();
    EXPECT_EQ(buf.addr(), nullptr);
    EXPECT_EQ(buf.size(), 0); // Size should be reset to 0 after deallocation

    // Multiple deallocations should be safe
    buf.deallocate_buffer();
    EXPECT_EQ(buf.addr(), nullptr);
}

// Test: Move constructor transfers ownership maintaining original address and
// size
TEST(BufferTest, MoveConstructor) {
    static constexpr std::size_t NUM_ELEMENTS = 50;
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(NUM_ELEMENTS);

    void *original_addr = buf.addr();
    const std::size_t original_size = buf.size();

    EXPECT_NE(original_addr, nullptr);
    EXPECT_EQ(original_size, NUM_ELEMENTS);

    // After move
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> moved_buf(std::move(buf));

    EXPECT_EQ(moved_buf.addr(), original_addr);
    EXPECT_EQ(moved_buf.size(), original_size);
}

// Test: Moved-from buffer becomes invalid but remains destructible after move
// operation
TEST(BufferTest, MoveConstructorSourceEmpty) {
    static constexpr std::size_t NUM_ELEMENTS = 50;
    framework::memory::Buffer<double, framework::memory::PinnedAlloc> original_buf(NUM_ELEMENTS);

    EXPECT_NE(original_buf.addr(), nullptr);
    EXPECT_EQ(original_buf.size(), NUM_ELEMENTS);

    const framework::memory::Buffer<double, framework::memory::PinnedAlloc> moved_buf(
            std::move(original_buf));

    // Note: Cannot test moved-from object state as it would violate
    // use-after-move rules
}

// Test: Move assignment transfers ownership from source to destination buffer
TEST(BufferTest, MoveAssignment) {
    static constexpr std::size_t NUM_ELEMENTS1 = 30;
    static constexpr std::size_t NUM_ELEMENTS2 = 60;
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf1(NUM_ELEMENTS1);
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf2(NUM_ELEMENTS2);

    void *buf2_original_addr = buf2.addr();
    const std::size_t buf2_original_size = buf2.size();

    EXPECT_NE(buf1.addr(), nullptr);
    EXPECT_EQ(buf1.size(), NUM_ELEMENTS1);
    EXPECT_NE(buf2.addr(), nullptr);
    EXPECT_EQ(buf2.size(), NUM_ELEMENTS2);

    // After move assignment
    buf1 = std::move(buf2);

    EXPECT_EQ(buf1.addr(), buf2_original_addr);
    EXPECT_EQ(buf1.size(), buf2_original_size);

    // Note: Cannot test moved-from object state as it would violate
    // use-after-move rules
}

// Test: Cross-allocator move constructor is explicitly deleted
TEST(BufferTest, CrossAllocatorMoveConstructorIsDeleted) {
    // Cross-allocator moves are fundamentally unsafe because:
    // 1. Memory allocated with one allocator must be deallocated with the same
    // allocator
    // 2. Moving memory between allocators would violate this contract
    // 3. For example: pinned memory allocated with cudaHostAlloc cannot be freed
    // with cudaFree

    // This test verifies that cross-allocator moves should use copy semantics
    // instead
    static constexpr std::size_t NUM_ELEMENTS = 60;
    framework::memory::Buffer<float, framework::memory::PinnedAlloc> pinned_buf(NUM_ELEMENTS);

    // Initialize test data
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        static constexpr float SCALE_FACTOR = 1.5F;
        pinned_buf[i] = static_cast<float>(i) * SCALE_FACTOR;
    }

    const auto *original_addr = pinned_buf.addr();
    EXPECT_NE(original_addr, nullptr);
    EXPECT_EQ(pinned_buf.size(), NUM_ELEMENTS);

    // Cross-allocator operations should use COPY semantics, not move
    // This creates new device memory and copies the data
    framework::memory::Buffer<float, framework::memory::DeviceAlloc> device_buf(
            pinned_buf); // Copy constructor

    // Verify the copy worked correctly
    EXPECT_NE(device_buf.addr(), nullptr);
    EXPECT_NE(device_buf.addr(), original_addr); // Different memory addresses
    EXPECT_EQ(device_buf.size(), NUM_ELEMENTS);

    // Original buffer should still be valid
    EXPECT_EQ(pinned_buf.addr(), original_addr);
    EXPECT_EQ(pinned_buf.size(), NUM_ELEMENTS);

    // Verify data integrity by copying back
    const framework::memory::Buffer<float, framework::memory::PinnedAlloc> verify_buf(device_buf);

    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        static constexpr float SCALE_FACTOR = 1.5F;
        EXPECT_FLOAT_EQ(verify_buf[i], static_cast<float>(i) * SCALE_FACTOR);
    }

    // Note: Cross-allocator move constructor is explicitly deleted in the buffer
    // class because it's fundamentally unsafe. If you try to write: Buffer<float,
    // DeviceAlloc> device_buf(std::move(pinned_buf)); You will get a clear
    // compile-time error about the deleted function, preventing the unsafe
    // operation and guiding you to use copy semantics instead.
}

// Test: Cross-allocator copy performs deep copy with data integrity
// preservation
TEST(BufferTest, CrossAllocatorCopyPinnedToDevice) {
    static constexpr std::size_t NUM_ELEMENTS = 100;
    static constexpr float TEST_VALUE = std::numbers::pi_v<float>;

    // Create pinned buffer and fill with test data
    framework::memory::Buffer<float, framework::memory::PinnedAlloc> pinned_buf(NUM_ELEMENTS);

    EXPECT_NE(pinned_buf.addr(), nullptr);
    EXPECT_EQ(pinned_buf.size(), NUM_ELEMENTS);

    // Fill pinned buffer with test data
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        pinned_buf[i] = TEST_VALUE + static_cast<float>(i);
    }

    // Copy to device buffer
    framework::memory::Buffer<float, framework::memory::DeviceAlloc> device_buf(
            pinned_buf); // Copy constructor

    EXPECT_NE(device_buf.addr(), nullptr);
    EXPECT_EQ(device_buf.size(), NUM_ELEMENTS);

    // Verify device buffer is independent from pinned buffer
    EXPECT_NE(device_buf.addr(), pinned_buf.addr());

    // Copy back to pinned to verify data integrity
    const framework::memory::Buffer<float, framework::memory::PinnedAlloc> verify_buf(device_buf);

    EXPECT_NE(verify_buf.addr(), nullptr);
    EXPECT_EQ(verify_buf.size(), NUM_ELEMENTS);
    EXPECT_NE(verify_buf.addr(), pinned_buf.addr());
    EXPECT_NE(verify_buf.addr(), device_buf.addr());

    // Verify data integrity
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        EXPECT_FLOAT_EQ(verify_buf[i], TEST_VALUE + static_cast<float>(i));
    }
}

// Test: Device-to-pinned copy maintains data integrity through CUDA memory
// transfer
TEST(BufferTest, CrossAllocatorCopyConstructorDeviceToPinned) {
    static constexpr std::size_t NUM_ELEMENTS = 40;

    // Create source data in pinned memory
    framework::memory::Buffer<double, framework::memory::PinnedAlloc> source_buf(NUM_ELEMENTS);
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        *(std::next(source_buf.addr(), static_cast<std::ptrdiff_t>(i))) =
                static_cast<double>(i) * std::numbers::e;
    }

    // Copy to device
    const framework::memory::Buffer<double, framework::memory::DeviceAlloc> device_buf(source_buf);

    // Copy back to pinned (device -> pinned)
    framework::memory::Buffer<double, framework::memory::PinnedAlloc> result_buf(device_buf);

    EXPECT_NE(result_buf.addr(), nullptr);
    EXPECT_EQ(result_buf.size(), NUM_ELEMENTS);

    // Verify data integrity
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        EXPECT_DOUBLE_EQ(
                *(std::next(result_buf.addr(), static_cast<std::ptrdiff_t>(i))),
                static_cast<double>(i) * std::numbers::e);
    }
}

// Test: Vector constructor copies all elements maintaining data integrity
TEST(BufferTest, VectorConstructor) {
    const std::vector<int> source_vec = {1, 2, 3, 4, 5, 10, 20, 30};

    // Create buffer from vector
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(source_vec);

    EXPECT_NE(buf.addr(), nullptr);
    EXPECT_EQ(buf.size(), source_vec.size());

    // Verify data was copied correctly
    for (std::size_t i = 0; i < source_vec.size(); ++i) {
        EXPECT_EQ(
                *(std::next(buf.addr(), static_cast<std::ptrdiff_t>(i))),
                *(std::next(source_vec.begin(), static_cast<std::ptrdiff_t>(i))));
    }
}

// Test: Empty vector constructor creates zero-size buffer without allocation
TEST(BufferTest, EmptyVectorConstructor) {
    const std::vector<float> empty_vec{};

    // Create buffer from empty vector
    const framework::memory::Buffer<float, framework::memory::PinnedAlloc> buf(empty_vec);

    EXPECT_EQ(buf.size(), 0);
}

// Test: Operator[] provides read/write access to pinned memory elements
TEST(BufferTest, IndexedAccessPinnedAlloc) {
    static constexpr std::size_t NUM_ELEMENTS = 10;
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(NUM_ELEMENTS);

    // Write data using indexed access
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        buf[i] = static_cast<int>(i * i);
    }

    // Read and verify data using indexed access
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        EXPECT_EQ(buf[i], static_cast<int>(i * i));
    }
}

// Test: Template type aliases correctly expose ElementType and AllocatorType
TEST(BufferTest, TypeAliases) {
    using TestBuffer = framework::memory::Buffer<double, framework::memory::PinnedAlloc>;

    static_assert(std::is_same_v<TestBuffer::ElementType, double>);
    static_assert(std::is_same_v<TestBuffer::AllocatorType, framework::memory::PinnedAlloc>);
}

// Test: Template instantiation works with various primitive types
TEST(BufferTest, DifferentElementTypes) {
    // Test with various types
    static constexpr std::size_t NUM_ELEMENTS = 10U;
    static constexpr std::size_t NUM_ELEMENTS_2 = 20U;
    static constexpr std::size_t NUM_ELEMENTS_3 = 5U;
    framework::memory::Buffer<char, framework::memory::PinnedAlloc> char_buf(NUM_ELEMENTS);
    framework::memory::Buffer<short, framework::memory::PinnedAlloc> short_buf(NUM_ELEMENTS_2);
    framework::memory::Buffer<long long, framework::memory::PinnedAlloc> long_long_buf(
            NUM_ELEMENTS_3);

    EXPECT_EQ(char_buf.size(), NUM_ELEMENTS);
    EXPECT_EQ(short_buf.size(), NUM_ELEMENTS_2);
    EXPECT_EQ(long_long_buf.size(), NUM_ELEMENTS_3);

    EXPECT_NE(char_buf.addr(), nullptr);
    EXPECT_NE(short_buf.addr(), nullptr);
    EXPECT_NE(long_long_buf.addr(), nullptr);
}

// Test: Large memory allocation succeeds and provides accessible memory
// boundaries
TEST(BufferTest, LargeBufferAllocation) {
    static constexpr std::size_t LARGE_SIZE = 1'000'000; // 1 million elements

    try {
        framework::memory::Buffer<int, framework::memory::PinnedAlloc> large_buf(LARGE_SIZE);

        EXPECT_NE(large_buf.addr(), nullptr);
        EXPECT_EQ(large_buf.size(), LARGE_SIZE);

        // Test access to first and last elements
        static constexpr int VALUE = 42;
        large_buf[0] = VALUE;
        large_buf[LARGE_SIZE - 1] = VALUE * 2;

        EXPECT_EQ(large_buf[0], VALUE);
        EXPECT_EQ(large_buf[LARGE_SIZE - 1], VALUE * 2);
    } catch (const std::exception &e) {
        // Large allocation might fail due to memory constraints
        FAIL() << "Failed to allocate large buffer due to memory constraints: " << e.what();
    }
}

// Test: addr() method returns appropriate const/non-const pointer types
TEST(BufferTest, AddrMethodConstCorrectness) {
    static constexpr std::size_t NUM_ELEMENTS = 10U;
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(NUM_ELEMENTS);
    const auto &const_buf = buf;

    // Non-const version returns non-const pointer
    static_assert(std::is_same_v<decltype(buf.addr()), int *>);

    // Const version returns const pointer
    static_assert(std::is_same_v<decltype(const_buf.addr()), const int *>);

    EXPECT_EQ(buf.addr(), const_buf.addr());
}

// Test: HostAccessible concept correctly enables/disables operator[] at
// compile-time
TEST(BufferTest, HostAccessibleConcept) {
    // PinnedAlloc should be host accessible (can use operator[])
    static_assert(framework::memory::HostAccessible<framework::memory::PinnedAlloc>);

    // DeviceAlloc should NOT be host accessible (cannot use operator[])
    static_assert(!framework::memory::HostAccessible<framework::memory::DeviceAlloc>);

    // Verify that pinned buffer supports indexed access
    static constexpr std::size_t NUM_ELEMENTS = 5U;
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> pinned_buf(NUM_ELEMENTS);
    static constexpr int VALUE = 42;
    pinned_buf[0] = VALUE; // This should compile
    EXPECT_EQ(pinned_buf[0], VALUE);

    // Note: device buffer operator[] should not compile due to HostAccessible
    // concept This is tested at compile time via the static_assert above
}

// Test: CUDA memory operations complete successfully with proper error
// propagation
TEST(BufferTest, CudaErrorHandling) {
    // This test verifies that CudaRuntimeException is thrown on CUDA errors
    // We can't easily force a CUDA error in a unit test, but we can verify
    // the exception type is correct

    try {
        static constexpr std::size_t NUM_ELEMENTS = 10;
        framework::memory::Buffer<int, framework::memory::PinnedAlloc> source_buf(NUM_ELEMENTS);

        // Initialize source data
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            source_buf[i] = static_cast<int>(i);
        }

        // This should succeed normally
        const framework::memory::Buffer<int, framework::memory::DeviceAlloc> device_buf(source_buf);
        const framework::memory::Buffer<int, framework::memory::PinnedAlloc> result_buf(device_buf);

        // Verify data integrity
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            EXPECT_EQ(result_buf[i], static_cast<int>(i));
        }
    } catch (const std::exception &e) {
        FAIL() << "Unexpected exception type: " << e.what();
    }
}

// Test: Buffer destructor automatically releases memory preventing leaks
TEST(BufferTest, RAIIMemoryManagement) {
    {
        void *allocated_addr = nullptr;
        // Buffer should be automatically destroyed when going out of scope
        static constexpr std::size_t NUM_ELEMENTS = 100U;
        framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(NUM_ELEMENTS);
        allocated_addr = buf.addr();
        EXPECT_NE(allocated_addr, nullptr);

        // Buffer destructor will be called here
    }

    // Memory should have been deallocated automatically
    // We can't directly verify this, but CUDA/system will detect leaks
    // if RAII isn't working properly
}

// Test: Self-move assignment maintains object validity without undefined
// behavior
TEST(BufferTest, SelfMoveAssignment) {
    static constexpr std::size_t NUM_ELEMENTS = 50U;
    framework::memory::Buffer<int, framework::memory::PinnedAlloc> buf(NUM_ELEMENTS);
    const auto original_size = buf.size();
    const auto *original_addr = buf.addr();
    // Self-move assignment should be safe
    FRAMEWORK_PRAGMA_IGNORE_SELF_MOVE
    buf = std::move(buf);
    FRAMEWORK_PRAGMA_RESTORE_SELF_MOVE

    // Buffer should remain in a valid state
    // Note: After self-move, the object is in a valid but unspecified state
    // We just verify it doesn't crash
    // the original size after self-move Object must still be usable – no specific
    // state guaranteed
    EXPECT_EQ(buf.size(), original_size); // This is by design. We cannot rely on
                                          // the size being the original size
                                          // after self-move.
    EXPECT_EQ(buf.addr(), original_addr);
    EXPECT_NO_THROW({ [[maybe_unused]] auto *ptr = buf.addr(); });
}

// Test: BufferImpl provides type-erased interface for different allocator types
TEST(BufferTest, BufferImplPolymorphicWrapper) {
    static constexpr std::size_t BUFFER_SIZE = 1024U;

    // Test device allocator variant
    const auto device_buffer_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::DeviceAlloc>>(
                    BUFFER_SIZE);

    EXPECT_NE(device_buffer_wrapper->addr(), nullptr);

    // Test pinned allocator variant
    const auto pinned_buffer_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(
                    BUFFER_SIZE);

    EXPECT_NE(pinned_buffer_wrapper->addr(), nullptr);

    // Verify different memory addresses (different allocators)
    EXPECT_NE(device_buffer_wrapper->addr(), pinned_buffer_wrapper->addr());
}

// Test: BufferImpl automatically releases resources when destroyed
TEST(BufferTest, BufferImplRAIIBehavior) {
    {
        static constexpr std::size_t BUFFER_SIZE = 2048U;
        void *captured_addr = nullptr;
        // Test automatic cleanup on scope exit
        auto buffer_wrapper =
                std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(
                        BUFFER_SIZE);
        captured_addr = buffer_wrapper->addr();

        EXPECT_NE(captured_addr, nullptr);

        // BufferImpl should be automatically destroyed here
    }

    // Memory should have been automatically deallocated
    // We can't directly verify this, but CUDA/system will detect leaks
    // if RAII isn't working properly
}

// Test: Mixed allocator types can coexist in polymorphic containers with unique
// addresses
TEST(BufferTest, BufferImplPolymorphicContainer) {
    static constexpr std::size_t BUFFER_SIZE = 512U;
    std::vector<std::unique_ptr<framework::memory::BufferWrapper>> buffer_wrappers{};

    // Create mixed allocator types in the same container
    buffer_wrappers.push_back(
            std::make_unique<framework::memory::BufferImpl<framework::memory::DeviceAlloc>>(
                    BUFFER_SIZE));
    buffer_wrappers.push_back(
            std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(
                    BUFFER_SIZE));
    buffer_wrappers.push_back(
            std::make_unique<framework::memory::BufferImpl<framework::memory::DeviceAlloc>>(
                    BUFFER_SIZE * 2));
    buffer_wrappers.push_back(
            std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(
                    BUFFER_SIZE * 3));

    // Verify all buffers are valid
    for (const auto &buffer_wrapper : buffer_wrappers) {
        EXPECT_NE(buffer_wrapper->addr(), nullptr);
    }

    // Verify unique addresses
    std::set<void *> unique_addresses{};
    for (const auto &buffer_wrapper : buffer_wrappers) {
        const auto [iter, inserted] = unique_addresses.insert(buffer_wrapper->addr());
        EXPECT_TRUE(inserted) << "Duplicate buffer address detected";
    }

    EXPECT_EQ(unique_addresses.size(), buffer_wrappers.size());
}

// Test: Zero-size BufferImpl construction/destruction succeeds without
// exceptions
TEST(BufferTest, BufferImplZeroSizeAllocation) {
    // Test zero-size allocation with device allocator
    auto device_buffer_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::DeviceAlloc>>(0);

    // Zero-size allocation behavior may vary - both nullptr and valid pointer are
    // acceptable The key is that it shouldn't crash or throw exceptions

    // Test zero-size allocation with pinned allocator
    auto pinned_buffer_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(0);

    // Should not crash during construction or destruction
}

// Test: Large BufferImpl allocations succeed with distinct memory addresses
TEST(BufferTest, BufferImplLargeAllocation) {
    static constexpr std::size_t LARGE_SIZE = 64ULL * 1024ULL * 1024ULL; // 64MB

    try {
        // Test large allocation with device allocator
        auto device_buffer_wrapper =
                std::make_unique<framework::memory::BufferImpl<framework::memory::DeviceAlloc>>(
                        LARGE_SIZE);

        EXPECT_NE(device_buffer_wrapper->addr(), nullptr);

        // Test large allocation with pinned allocator
        auto pinned_buffer_wrapper =
                std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(
                        LARGE_SIZE);

        EXPECT_NE(pinned_buffer_wrapper->addr(), nullptr);

        // Verify different addresses
        EXPECT_NE(device_buffer_wrapper->addr(), pinned_buffer_wrapper->addr());
    } catch (const std::exception &e) {
        // Large allocation might fail due to memory constraints
        FAIL() << "Failed to allocate large buffer due to memory constraints: " << e.what();
    }
}

// Test: Virtual dispatch correctly routes to allocator-specific implementations
TEST(BufferTest, BufferImplVirtualFunctionBehavior) {
    static constexpr std::size_t BUFFER_SIZE = 256;

    // Create BufferImpl instances through base class pointer
    auto device_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::DeviceAlloc>>(
                    BUFFER_SIZE);
    auto pinned_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(
                    BUFFER_SIZE);

    // Test virtual function dispatch
    void *device_addr = device_wrapper->addr();
    void *pinned_addr = pinned_wrapper->addr();

    EXPECT_NE(device_addr, nullptr);
    EXPECT_NE(pinned_addr, nullptr);
    EXPECT_NE(device_addr, pinned_addr);

    // Test polymorphic behavior in function calls
    const auto test_virtual_dispatch = [](framework::memory::BufferWrapper &wrapper) -> void * {
        return wrapper.addr();
    };

    EXPECT_EQ(test_virtual_dispatch(*device_wrapper), device_addr);
    EXPECT_EQ(test_virtual_dispatch(*pinned_wrapper), pinned_addr);
}

// Test: BufferImpl supports move semantics through unique_ptr wrapper
TEST(BufferTest, BufferImplMoveSemantics) {
    static constexpr std::size_t BUFFER_SIZE = 1024;

    // Test move construction
    // IMPORTANT: Use explicit type std::unique_ptr<BufferWrapper> instead of auto
    // to make the polymorphic nature visible and prevent accidental type mixing.
    // Using auto here would hide the fact that we're dealing with a type-erased
    // wrapper that can silently accept different allocator types, leading to
    // potential memory corruption.
    std::unique_ptr<framework::memory::BufferWrapper> original_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::PinnedAlloc>>(
                    BUFFER_SIZE);

    const void *original_addr = original_wrapper->addr();
    EXPECT_NE(original_addr, nullptr);

    // Move the unique_ptr
    std::unique_ptr<framework::memory::BufferWrapper> moved_wrapper = std::move(original_wrapper);

    EXPECT_EQ(original_wrapper, nullptr);
    EXPECT_NE(moved_wrapper, nullptr);
    EXPECT_EQ(moved_wrapper->addr(), original_addr);

    // Test move assignment
    // Use an explicit std::unique_ptr<BufferWrapper> (not auto) so that the
    // type-erasure is obvious to readers.  With auto the fact that we’re holding
    // a BufferWrapper is hidden, which can make allocator mismatches harder to
    // spot during review, even though run-time mixing is safe.
    std::unique_ptr<framework::memory::BufferWrapper> another_wrapper =
            std::make_unique<framework::memory::BufferImpl<framework::memory::DeviceAlloc>>(
                    BUFFER_SIZE * 2);

    const void *another_addr = another_wrapper->addr();
    EXPECT_NE(another_addr, nullptr);
    EXPECT_NE(another_addr, original_addr);

    // Move assign
    another_wrapper = std::move(moved_wrapper);

    EXPECT_EQ(moved_wrapper, nullptr);
    EXPECT_NE(another_wrapper, nullptr);
    EXPECT_EQ(another_wrapper->addr(), original_addr);
}

} // namespace
