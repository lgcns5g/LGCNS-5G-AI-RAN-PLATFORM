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

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <rte_mbuf.h>
#include <rte_mbuf_core.h>
#include <rte_memory.h>
#include <rte_mempool.h>

#include <gtest/gtest.h>

#include "oran/dpdk_buf.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// Test fixture for MBuf tests
class MBufTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a mempool for testing
        mempool_ = rte_pktmbuf_pool_create(
                "test_pool",  // name
                256,          // n (number of elements)
                0,            // cache_size
                0,            // priv_size
                2048,         // data_room_size
                SOCKET_ID_ANY // socket_id
        );
        ASSERT_NE(mempool_, nullptr) << "Failed to create mempool";

        // Allocate a test mbuf
        test_mbuf_ = rte_pktmbuf_alloc(mempool_);
        ASSERT_NE(test_mbuf_, nullptr) << "Failed to allocate mbuf";
    }

    void TearDown() override {
        if (test_mbuf_ != nullptr) {
            rte_pktmbuf_free(test_mbuf_);
            test_mbuf_ = nullptr;
        }
        if (mempool_ != nullptr) {
            rte_mempool_free(mempool_);
            mempool_ = nullptr;
        }
    }

    rte_mempool *mempool_{nullptr};
    rte_mbuf *test_mbuf_{nullptr};
};

// Test construction and basic accessors
TEST_F(MBufTest, ConstructionAndBasicAccessors) {
    ran::oran::MBuf buffer(test_mbuf_);

    // Check that data pointer is valid
    EXPECT_NE(buffer.data(), nullptr);

    // Check const data accessor
    const auto &const_buffer = buffer;
    EXPECT_NE(const_buffer.data(), nullptr);
    EXPECT_EQ(const_buffer.data(), buffer.data());

    // Check initial size and capacity
    EXPECT_EQ(buffer.size(), 0U);
    EXPECT_GT(buffer.capacity(), 0U);

    // Verify get_mbuf returns the same mbuf
    EXPECT_EQ(buffer.get_mbuf(), test_mbuf_);
}

// Test set_size functionality
TEST_F(MBufTest, SetSize) {
    ran::oran::MBuf buffer(test_mbuf_);

    // Set a valid size
    const std::size_t new_size = 100;
    buffer.set_size(new_size);
    EXPECT_EQ(buffer.size(), new_size);

    // Set size to zero
    buffer.set_size(0);
    EXPECT_EQ(buffer.size(), 0U);

    // Set size to capacity
    const std::size_t capacity = buffer.capacity();
    buffer.set_size(capacity);
    EXPECT_EQ(buffer.size(), capacity);

    // Try to set size beyond capacity (should throw)
    const std::size_t oversized = capacity + 100;
    EXPECT_THROW(buffer.set_size(oversized), std::length_error);
}

// Test data writing and reading
TEST_F(MBufTest, DataWriteAndRead) {
    ran::oran::MBuf buffer(test_mbuf_);

    // Write some data
    const std::array<std::uint8_t, 5> test_data = {0x01, 0x02, 0x03, 0x04, 0x05};
    const std::size_t test_size = test_data.size();

    buffer.set_size(test_size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::memcpy(buffer.data(), test_data.data(), test_size);

    // Verify data can be read back
    for (std::size_t i = 0; i < test_size; ++i) {
        EXPECT_EQ(buffer.at(i), test_data.at(i));
    }

    // Verify const accessor works
    const auto &const_buffer = buffer;
    for (std::size_t i = 0; i < test_size; ++i) {
        EXPECT_EQ(const_buffer.at(i), test_data.at(i));
    }
}

// Test clear_flags functionality
TEST_F(MBufTest, ClearFlags) {
    ran::oran::MBuf buffer(test_mbuf_);

    // Set some flags
    test_mbuf_->ol_flags = 0xFFFFFFFFFFFFFFFFULL;

    // Clear all flags
    buffer.clear_flags();
    EXPECT_EQ(test_mbuf_->ol_flags, 0U);
}

// Test set_timestamp functionality (automatically sets flag)
TEST_F(MBufTest, SetTimestamp) {
    ran::oran::MBuf buffer(test_mbuf_);

    // Set timestamp (this automatically sets the timestamp flag)
    constexpr std::uint64_t TEST_TIMESTAMP = 123456789ULL;
    buffer.set_timestamp(TEST_TIMESTAMP);

    // Set another timestamp
    buffer.set_timestamp(0xDEADBEEFULL);

    // The timestamp flag is automatically set by set_timestamp()
}

// Test MBuf timestamp for accurate send scheduling
TEST_F(MBufTest, AccurateSendScheduling) {
    using namespace std::chrono;

    // example-begin mbuf-timestamp-1
    // Wrap DPDK mbuf for ORAN packet transmission
    ran::oran::MBuf buffer(test_mbuf_);

    // Get current time in nanoseconds since epoch
    const auto now = system_clock::now().time_since_epoch();
    const auto ns = duration_cast<nanoseconds>(now).count();
    const auto current_time_ns = static_cast<std::uint64_t>(ns);

    // Set timestamp for hardware-assisted send scheduling
    // Timestamp specifies when NIC should transmit the packet
    buffer.set_timestamp(current_time_ns);
    // example-end mbuf-timestamp-1

    // Verify the timestamp flag was set in the mbuf
    EXPECT_NE(test_mbuf_->ol_flags, 0U);
}

// Test multiple operations in sequence
TEST_F(MBufTest, SequentialOperations) {
    ran::oran::MBuf buffer(test_mbuf_);

    // Set size
    buffer.set_size(64);
    EXPECT_EQ(buffer.size(), 64U);

    // Write data
    std::memset(buffer.data(), 0xAB, 64);

    // Set timestamp (automatically sets flag)
    buffer.set_timestamp(987654321ULL);

    // Verify data still intact
    for (std::size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(buffer.at(i), 0xAB);
    }

    // Clear flags
    buffer.clear_flags();
    EXPECT_EQ(test_mbuf_->ol_flags, 0U);

    // Verify data still intact after clearing flags
    for (std::size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(buffer.at(i), 0xAB);
    }

    // Change size
    buffer.set_size(32);
    EXPECT_EQ(buffer.size(), 32U);
}

// Test zero-size buffer
TEST_F(MBufTest, ZeroSizeBuffer) {
    ran::oran::MBuf buffer(test_mbuf_);

    // Initially zero size
    EXPECT_EQ(buffer.size(), 0U);

    // Can still access data pointer
    EXPECT_NE(buffer.data(), nullptr);

    // Setting zero size should work
    buffer.set_size(0);
    EXPECT_EQ(buffer.size(), 0U);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
