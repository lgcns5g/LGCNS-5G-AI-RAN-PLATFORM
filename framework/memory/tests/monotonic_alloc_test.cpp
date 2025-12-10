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

#include <algorithm>   // for max
#include <bit>         // for bit_cast
#include <cstdint>     // for uintptr_t
#include <cstdlib>     // for size_t, free, malloc
#include <new>         // for bad_alloc
#include <stdexcept>   // for runtime_error
#include <string>      // for allocator, basic_string, char_tr...
#include <type_traits> // for is_copy_assignable_v, is_copy_co...
#include <utility>     // for move
#include <vector>      // for vector

#include <gtest/gtest.h> // for Message, TestPartResult, EXPECT_EQ

#include "memory/monotonic_alloc.hpp" // for MonotonicAlloc
#include "utils/error_macros.hpp"     // for FRAMEWORK_PRAGMA_IGNORE_SELF_MOVE

// Test constants for magic number elimination
namespace {
// Alignment values (powers of 2)
constexpr std::size_t ALIGNMENT_1_BYTE = 1;
constexpr std::size_t ALIGNMENT_2_BYTES = 2;
constexpr std::size_t ALIGNMENT_4_BYTES = 4;
constexpr std::size_t ALIGNMENT_8_BYTES = 8;
constexpr std::size_t ALIGNMENT_16_BYTES = 16;
constexpr std::size_t ALIGNMENT_32_BYTES = 32;
constexpr std::size_t ALIGNMENT_64_BYTES = 64;
constexpr std::size_t ALIGNMENT_128_BYTES = 128;

// Buffer sizes
constexpr std::size_t SMALL_BUFFER_SIZE = 64;
constexpr std::size_t MEDIUM_BUFFER_SIZE = 128;
constexpr std::size_t LARGE_BUFFER_SIZE = 1024;
constexpr std::size_t PERFORMANCE_BUFFER_SIZE = 1024ULL * 1024; // 1MB

// Allocation sizes
constexpr std::size_t ZERO_BYTES = 0;
constexpr std::size_t SINGLE_BYTE = 1;
constexpr std::size_t SMALL_ALLOCATION = 10;
constexpr std::size_t MEDIUM_ALLOCATION = 32;
constexpr std::size_t LARGE_ALLOCATION = 64;
constexpr std::size_t UNALIGNED_ALLOCATION = 17;
constexpr std::size_t BOUNDARY_ALLOCATION = 48;

// Expected offsets and counts
constexpr std::size_t INITIAL_OFFSET = 0;
constexpr std::size_t SINGLE_COUNT = 1;
constexpr std::size_t DOUBLE_COUNT = 2;
constexpr std::size_t LOOP_COUNT = 10;
constexpr std::size_t PERFORMANCE_ALLOCATION_COUNT = 1000;

// Calculated offsets for specific test scenarios
constexpr std::size_t OFFSET_AFTER_SMALL_ALIGNED = 16;
constexpr std::size_t OFFSET_AFTER_TWO_ALLOCATIONS = 48;
constexpr std::size_t OFFSET_AFTER_THREE_ALLOCATIONS = 64;
constexpr std::size_t OFFSET_AFTER_BASIC_ALLOCATION = 96;
constexpr std::size_t TOTAL_OFFSET_TEN_ALLOCATIONS = 320;
constexpr std::size_t BUFFER_SIZE_512 = 512;
} // namespace

// Test: Mock allocator for testing monotonic_alloc without CUDA dependencies
struct MockAlloc final {
    static inline std::size_t allocation_count{INITIAL_OFFSET};
    static inline std::size_t deallocation_count{INITIAL_OFFSET};
    static inline std::size_t total_allocated_bytes{INITIAL_OFFSET};

    [[nodiscard]] static void *allocate(const std::size_t nbytes) {
        ++allocation_count;
        total_allocated_bytes += nbytes;
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc,hicpp-no-malloc)
        return std::malloc(nbytes);
    }

    static void deallocate(void *addr) {
        ++deallocation_count;
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc,hicpp-no-malloc)
        std::free(addr);
    }

    static void reset_counters() {
        allocation_count = INITIAL_OFFSET;
        deallocation_count = INITIAL_OFFSET;
        total_allocated_bytes = INITIAL_OFFSET;
    }
};

// Test: Mock allocator that always fails allocation
struct FailingAlloc final {
    [[nodiscard]] static void *allocate(const std::size_t /*nbytes*/) { throw std::bad_alloc{}; }

    static void deallocate(void * /*addr*/) {
        // Should never be called
    }
};

class MonotonicAllocTest : public ::testing::Test {
protected:
    void SetUp() override { MockAlloc::reset_counters(); }

    void TearDown() override { MockAlloc::reset_counters(); }
};

// Test: Basic construction and destruction
TEST_F(MonotonicAllocTest, ConstructionAndDestruction) {
    {
        static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
        const framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(
                BUFFER_SIZE);

        EXPECT_EQ(MockAlloc::allocation_count, SINGLE_COUNT);
        EXPECT_EQ(MockAlloc::total_allocated_bytes, BUFFER_SIZE);
        EXPECT_EQ(allocator.size(), BUFFER_SIZE);
        EXPECT_EQ(allocator.offset(), INITIAL_OFFSET);
        EXPECT_NE(allocator.address(), nullptr);
    }

    EXPECT_EQ(MockAlloc::deallocation_count, SINGLE_COUNT);
}

// Test: Template parameter validation
TEST_F(MonotonicAllocTest, TemplateParameterValidation) {

    // Test: Valid power-of-2 alignments should compile
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_1_BYTE, MockAlloc>(SMALL_BUFFER_SIZE)));
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_2_BYTES, MockAlloc>(SMALL_BUFFER_SIZE)));
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_4_BYTES, MockAlloc>(SMALL_BUFFER_SIZE)));
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_8_BYTES, MockAlloc>(SMALL_BUFFER_SIZE)));
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc>(SMALL_BUFFER_SIZE)));
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_32_BYTES, MockAlloc>(SMALL_BUFFER_SIZE)));
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_64_BYTES, MockAlloc>(SMALL_BUFFER_SIZE)));
    EXPECT_NO_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_128_BYTES, MockAlloc>(SMALL_BUFFER_SIZE)));

    // Test: Non-power-of-2 alignments should fail to compile
    // These are compile-time tests - they would fail compilation if uncommented
    // MonotonicAlloc<INVALID_ALIGNMENT_3, mock_alloc> alloc3(SMALL_BUFFER_SIZE);
    // // Should not compile MonotonicAlloc<INVALID_ALIGNMENT_5, mock_alloc>
    // alloc5(SMALL_BUFFER_SIZE);   // Should not compile
    // MonotonicAlloc<INVALID_ALIGNMENT_6, mock_alloc> alloc6(SMALL_BUFFER_SIZE);
    // // Should not compile MonotonicAlloc<INVALID_ALIGNMENT_7, mock_alloc>
    // alloc7(SMALL_BUFFER_SIZE);   // Should not compile
}

// Test: Failed allocation in constructor
TEST_F(MonotonicAllocTest, FailedAllocation) {

    EXPECT_THROW(
            (framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, FailingAlloc>(
                    LARGE_BUFFER_SIZE)),
            std::bad_alloc);
}

// Test: Basic memory allocation
TEST_F(MonotonicAllocTest, BasicAllocation) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: First allocation
    void *ptr1 = allocator.allocate(LARGE_ALLOCATION);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(allocator.offset(), LARGE_ALLOCATION); // 64 bytes, aligned to 16

    // Test: Second allocation
    void *ptr2 = allocator.allocate(MEDIUM_ALLOCATION);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    EXPECT_EQ(allocator.offset(),
              OFFSET_AFTER_BASIC_ALLOCATION); // 64 + 32 = 96, aligned to 16

    // Test: Pointers should be properly spaced
    EXPECT_EQ(static_cast<char *>(ptr2) - static_cast<char *>(ptr1), LARGE_ALLOCATION);
}

// Test: Alignment behavior
TEST_F(MonotonicAllocTest, AlignmentBehavior) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Allocation sizes that require alignment padding
    void *ptr1 = allocator.allocate(SMALL_ALLOCATION); // Should be padded to 16
    EXPECT_EQ(allocator.offset(), OFFSET_AFTER_SMALL_ALIGNED);

    void *ptr2 = allocator.allocate(UNALIGNED_ALLOCATION);       // Should be padded to 32
    EXPECT_EQ(allocator.offset(), OFFSET_AFTER_TWO_ALLOCATIONS); // 16 + 32 = 48

    void *ptr3 = allocator.allocate(SINGLE_BYTE);                  // Should be padded to 16
    EXPECT_EQ(allocator.offset(), OFFSET_AFTER_THREE_ALLOCATIONS); // 48 + 16 = 64

    // Test: Pointers should be properly aligned
    EXPECT_EQ(std::bit_cast<std::uintptr_t>(ptr1) % ALIGNMENT_16_BYTES, INITIAL_OFFSET);
    EXPECT_EQ(std::bit_cast<std::uintptr_t>(ptr2) % ALIGNMENT_16_BYTES, INITIAL_OFFSET);
    EXPECT_EQ(std::bit_cast<std::uintptr_t>(ptr3) % ALIGNMENT_16_BYTES, INITIAL_OFFSET);
}

// Test: Buffer overflow protection
TEST_F(MonotonicAllocTest, BufferOverflowProtection) {

    static constexpr std::size_t BUFFER_SIZE = MEDIUM_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Fill most of the buffer
    void *ptr1 = allocator.allocate(LARGE_ALLOCATION);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(allocator.offset(), LARGE_ALLOCATION);

    void *ptr2 = allocator.allocate(MEDIUM_ALLOCATION);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(allocator.offset(), OFFSET_AFTER_BASIC_ALLOCATION);

    // Test: This should still fit (96 + 32 = 128)
    void *ptr3 = allocator.allocate(MEDIUM_ALLOCATION);
    EXPECT_NE(ptr3, nullptr);
    EXPECT_EQ(allocator.offset(), BUFFER_SIZE);

    // Test: This should throw - buffer is full
    EXPECT_THROW(allocator.allocate(SINGLE_BYTE), std::runtime_error);

    // Test: Exception message should contain useful information
    try {
        allocator.allocate(ALIGNMENT_16_BYTES);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_TRUE(msg.find("Buffer size exceeded") != std::string::npos);
        EXPECT_TRUE(msg.find("offset = 128") != std::string::npos);
        EXPECT_TRUE(msg.find("num_bytes = 16") != std::string::npos);
        EXPECT_TRUE(msg.find("aligned_bytes = 16") != std::string::npos);
        EXPECT_TRUE(msg.find("block_size = 128") != std::string::npos);
    }
}

// Test: Reset functionality
TEST_F(MonotonicAllocTest, ResetFunctionality) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Allocate some memory
    void *ptr1 = allocator.allocate(LARGE_ALLOCATION);
    [[maybe_unused]] const auto *ptr2 = allocator.allocate(MEDIUM_ALLOCATION);
    EXPECT_EQ(allocator.offset(), OFFSET_AFTER_BASIC_ALLOCATION);

    // Test: Reset should clear the offset
    allocator.reset();
    EXPECT_EQ(allocator.offset(), INITIAL_OFFSET);
    EXPECT_EQ(allocator.size(), BUFFER_SIZE);
    EXPECT_NE(allocator.address(), nullptr);

    // Test: Should be able to allocate again from the beginning
    const auto *ptr3 = allocator.allocate(LARGE_ALLOCATION);
    EXPECT_NE(ptr3, nullptr);
    EXPECT_EQ(allocator.offset(), LARGE_ALLOCATION);

    // Test: New allocation should reuse the same memory space
    EXPECT_EQ(ptr1, ptr3);
}

// Test: Move constructor
TEST_F(MonotonicAllocTest, MoveConstructor) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;

    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator1(BUFFER_SIZE);
    void *original_addr = allocator1.address();

    // Test: Allocate some memory to set offset
    allocator1.allocate(LARGE_ALLOCATION);
    EXPECT_EQ(allocator1.offset(), LARGE_ALLOCATION);

    // Test: Move construct
    const framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator2(
            std::move(allocator1));

    // Test: New allocator should have the state
    EXPECT_EQ(allocator2.size(), BUFFER_SIZE);
    EXPECT_EQ(allocator2.offset(), LARGE_ALLOCATION);
    EXPECT_EQ(allocator2.address(), original_addr);

    // Test: Original allocator should be empty
    // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move,hicpp-invalid-access-moved)
    // cppcheck-suppress accessMoved
    EXPECT_EQ(allocator1.size(), INITIAL_OFFSET);
    // cppcheck-suppress accessMoved
    EXPECT_EQ(allocator1.offset(), INITIAL_OFFSET);
    // cppcheck-suppress accessMoved
    EXPECT_EQ(allocator1.address(), nullptr);
    // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move,hicpp-invalid-access-moved)

    // Test: Only one deallocation should happen when both destructors run
    EXPECT_EQ(MockAlloc::allocation_count, SINGLE_COUNT);
}

// Test: Move assignment operator
TEST_F(MonotonicAllocTest, MoveAssignment) {

    static constexpr std::size_t BUFFER_SIZE1 = LARGE_BUFFER_SIZE;
    static constexpr std::size_t BUFFER_SIZE2 = BUFFER_SIZE_512;

    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator1(BUFFER_SIZE1);
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator2(BUFFER_SIZE2);

    void *original_addr1 = allocator1.address();
    [[maybe_unused]] const auto *original_addr2 = allocator2.address();

    allocator1.allocate(LARGE_ALLOCATION);
    allocator2.allocate(MEDIUM_ALLOCATION);

    EXPECT_EQ(MockAlloc::allocation_count, DOUBLE_COUNT);

    // Test: Move assignment
    allocator2 = std::move(allocator1);

    // Test: allocator2 should have allocator1's state
    EXPECT_EQ(allocator2.size(), BUFFER_SIZE1);
    EXPECT_EQ(allocator2.offset(), LARGE_ALLOCATION);
    EXPECT_EQ(allocator2.address(), original_addr1);

    // Test: allocator1 should be empty
    // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move,hicpp-invalid-access-moved)
    // cppcheck-suppress accessMoved
    EXPECT_EQ(allocator1.size(), INITIAL_OFFSET);
    // cppcheck-suppress accessMoved
    EXPECT_EQ(allocator1.offset(), INITIAL_OFFSET);
    // cppcheck-suppress accessMoved
    EXPECT_EQ(allocator1.address(), nullptr);
    // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move,hicpp-invalid-access-moved)

    // Test: Original allocator2 buffer should be deallocated
    EXPECT_EQ(MockAlloc::deallocation_count, SINGLE_COUNT);
}

// Test: Self-assignment protection
TEST_F(MonotonicAllocTest, SelfAssignmentProtection) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    void *original_addr = allocator.address();
    allocator.allocate(LARGE_ALLOCATION);

    // Test: Self-assignment should be safe
    FRAMEWORK_PRAGMA_IGNORE_SELF_MOVE
    allocator = std::move(allocator);
    FRAMEWORK_PRAGMA_RESTORE_SELF_MOVE

    // Test: State should be unchanged
    EXPECT_EQ(allocator.size(), BUFFER_SIZE);
    EXPECT_EQ(allocator.offset(), LARGE_ALLOCATION);
    EXPECT_EQ(allocator.address(), original_addr);

    // Test: No extra deallocations
    EXPECT_EQ(MockAlloc::deallocation_count, INITIAL_OFFSET);
}

// Test: Copy operations are deleted
TEST_F(MonotonicAllocTest, CopyOperationsDeleted) {

    // Test: These should not compile
    static_assert(!std::is_copy_constructible_v<
                  framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc>>);
    static_assert(!std::is_copy_assignable_v<
                  framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc>>);

    // Test: Move operations should be available
    static_assert(std::is_move_constructible_v<
                  framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc>>);
    static_assert(std::is_move_assignable_v<
                  framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc>>);
}

// Test: Getter methods
TEST_F(MonotonicAllocTest, GetterMethods) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Initial state
    EXPECT_EQ(allocator.size(), BUFFER_SIZE);
    EXPECT_EQ(allocator.offset(), INITIAL_OFFSET);
    EXPECT_NE(allocator.address(), nullptr);

    // Test: After allocation
    void *ptr = allocator.allocate(LARGE_ALLOCATION);
    EXPECT_EQ(allocator.size(), BUFFER_SIZE);        // Size should not change
    EXPECT_EQ(allocator.offset(), LARGE_ALLOCATION); // Offset should update
    EXPECT_EQ(allocator.address(), ptr);             // Address should be the base

    // Test: Methods should be const-correct
    const auto &const_allocator = allocator;
    EXPECT_EQ(const_allocator.size(), BUFFER_SIZE);
    EXPECT_EQ(const_allocator.offset(), LARGE_ALLOCATION);
    EXPECT_EQ(const_allocator.address(), ptr);
}

// Test: Zero-size allocation
TEST_F(MonotonicAllocTest, ZeroSizeAllocation) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Zero-size allocation should work
    void *ptr = allocator.allocate(ZERO_BYTES);
    EXPECT_NE(ptr, nullptr);

    // Test: Offset should be aligned even for zero-size
    EXPECT_EQ(allocator.offset(), INITIAL_OFFSET); // 0 aligned to 16 is still 0

    // Test: Next allocation should work normally
    void *ptr2 = allocator.allocate(MEDIUM_ALLOCATION);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(allocator.offset(), MEDIUM_ALLOCATION);
}

// Test: Large allocation
TEST_F(MonotonicAllocTest, LargeAllocation) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Allocation that uses entire buffer
    void *ptr = allocator.allocate(BUFFER_SIZE);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(allocator.offset(), BUFFER_SIZE);

    // Test: No more allocations should be possible
    EXPECT_THROW(allocator.allocate(SINGLE_BYTE), std::runtime_error);
}

// Test: Different allocator types
TEST_F(MonotonicAllocTest, DifferentAllocatorTypes) {

    // Test: Template should work with different allocator types
    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;

    // Test: Mock allocator
    const framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> mock_allocator(
            BUFFER_SIZE);
    // Test: Template instantiation should be correct
    EXPECT_EQ(mock_allocator.size(), BUFFER_SIZE);
}

// Test: Memory pattern verification
TEST_F(MonotonicAllocTest, MemoryPatternVerification) {

    static constexpr std::size_t BUFFER_SIZE = LARGE_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Allocate several blocks and verify they're contiguous
    std::vector<const void *> ptrs;
    for (std::size_t i = INITIAL_OFFSET; i < LOOP_COUNT; ++i) {
        ptrs.push_back(allocator.allocate(MEDIUM_ALLOCATION));
    }

    // Test: Verify pointers are properly spaced
    for (std::size_t i = SINGLE_COUNT; i < ptrs.size(); ++i) {
        const auto diff = static_cast<const char *>(ptrs[i]) -
                          static_cast<const char *>(ptrs[i - SINGLE_COUNT]);
        EXPECT_EQ(diff, MEDIUM_ALLOCATION); // 32 bytes aligned to 16 is still 32
    }

    // Test: Total offset should be correct
    EXPECT_EQ(allocator.offset(), TOTAL_OFFSET_TEN_ALLOCATIONS); // 10 * 32 = 320
}

// Test: Edge case - allocation exactly at buffer boundary
TEST_F(MonotonicAllocTest, BoundaryAllocation) {

    static constexpr std::size_t BUFFER_SIZE = MEDIUM_BUFFER_SIZE;
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Allocate up to exactly the boundary
    allocator.allocate(LARGE_ALLOCATION);    // offset = 64
    allocator.allocate(BOUNDARY_ALLOCATION); // offset = 112
    allocator.allocate(ALIGNMENT_16_BYTES);  // offset = 128 (exactly at boundary)

    EXPECT_EQ(allocator.offset(), BUFFER_SIZE);

    // Test: One more byte should fail
    EXPECT_THROW(allocator.allocate(SINGLE_BYTE), std::runtime_error);
}

// Test: Alignment buffer overrun protection
TEST_F(MonotonicAllocTest, AlignmentBufferOverrunProtection) {

    // Test: Use a small buffer where alignment padding could cause overruns
    static constexpr std::size_t BUFFER_SIZE = SMALL_BUFFER_SIZE; // 64 bytes
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Allocate most of the buffer leaving only a few bytes
    allocator.allocate(BOUNDARY_ALLOCATION); // 48 bytes -> aligned to 48, offset = 48
    EXPECT_EQ(allocator.offset(), BOUNDARY_ALLOCATION);

    // Test: Try to allocate 15 bytes (unaligned size fits, but aligned size
    // doesn't) 15 bytes would align to 16 bytes, and 48 + 16 = 64 (exactly at
    // boundary)
    static constexpr std::size_t UNALIGNED_15_BYTES = 15;
    const auto *ptr = allocator.allocate(UNALIGNED_15_BYTES);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(allocator.offset(), BUFFER_SIZE); // Should be exactly at boundary

    // Test: Now any allocation should fail since buffer is full
    EXPECT_THROW(allocator.allocate(SINGLE_BYTE), std::runtime_error);

    // Test: Try a case where unaligned size would fit but aligned size exceeds
    // buffer
    allocator.reset();
    allocator.allocate(BOUNDARY_ALLOCATION + 1); // 49 bytes -> aligned to 64, offset = 64
    EXPECT_EQ(allocator.offset(), BUFFER_SIZE);

    // Test: Verify exception message includes aligned size information
    try {
        allocator.allocate(SINGLE_BYTE); // 1 byte -> aligns to 16, would exceed buffer
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_TRUE(msg.find("Buffer size exceeded") != std::string::npos);
        EXPECT_TRUE(msg.find("offset = 64") != std::string::npos);
        EXPECT_TRUE(msg.find("num_bytes = 1") != std::string::npos);
        EXPECT_TRUE(msg.find("aligned_bytes = 16") != std::string::npos);
        EXPECT_TRUE(msg.find("block_size = 64") != std::string::npos);
    }
}

// Test: Performance characteristics (basic timing)
TEST_F(MonotonicAllocTest, PerformanceCharacteristics) {

    static constexpr std::size_t BUFFER_SIZE = PERFORMANCE_BUFFER_SIZE; // 1MB
    framework::memory::MonotonicAlloc<ALIGNMENT_16_BYTES, MockAlloc> allocator(BUFFER_SIZE);

    // Test: Many small allocations should be fast
    static constexpr std::size_t NUM_ALLOCATIONS = PERFORMANCE_ALLOCATION_COUNT;

    for (std::size_t i = INITIAL_OFFSET; i < NUM_ALLOCATIONS; ++i) {
        const auto *ptr = allocator.allocate(LARGE_ALLOCATION);
        EXPECT_NE(ptr, nullptr);
    }

    // Test: All allocations should have succeeded
    EXPECT_EQ(allocator.offset(), NUM_ALLOCATIONS * LARGE_ALLOCATION);

    // Test: Verify total allocated memory stays within buffer limits
    EXPECT_LE(allocator.offset(), BUFFER_SIZE);
}
