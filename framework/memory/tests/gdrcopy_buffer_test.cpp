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
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <driver_types.h>
#include <gdrapi.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "memory/gdrcopy_buffer.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace framework::memory::tests {

// Test fixture for GDRCopy buffer tests
class GdrCopyBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context for framework::memory::GpinnedBuffer tests
        // framework::memory::GpinnedBuffer requires CUDA to allocate device memory
        const cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            FAIL() << "CUDA device not available: " << cudaGetErrorString(error);
        }
    }

    void TearDown() override {
        // Note: We don't call cudaDeviceReset() here as it destroys the CUDA
        // context and interferes with subsequent tests in the same executable.
        // CUDA will clean up automatically when the process exits.
    }
};

// Test UniqueGdrHandle RAII wrapper
TEST_F(GdrCopyBufferTest, UniqueGdrHandleCreation) {
    // Test that we can create a UniqueGdrHandle
    // Note: This will fail if GDRCopy driver is not available, which is expected
    try {
        auto handle = framework::memory::make_unique_gdr_handle();
        ASSERT_NE(handle.get(), nullptr) << "GDRCopy handle should not be null";
        // Handle will be automatically closed when going out of scope
    } catch (const std::runtime_error &e) {
        // GDRCopy driver is required for these tests
        FAIL() << "GDRCopy driver not available: " << e.what();
    }
}

// Test UniqueGdrHandle RAII cleanup
TEST_F(GdrCopyBufferTest, UniqueGdrHandleRAII) {
    try {
        {
            auto handle = framework::memory::make_unique_gdr_handle();
            ASSERT_NE(handle.get(), nullptr);
            // Handle should be closed automatically when exiting scope
        }
        // If we get here without crash, RAII cleanup worked
        SUCCEED();
    } catch (const std::runtime_error &e) {
        FAIL() << "GDRCopy driver not available: " << e.what();
    }
}

// Test framework::memory::GpinnedBuffer with UniqueGdrHandle
TEST_F(GdrCopyBufferTest, GpinnedBufferWithUniqueHandle) {
    try {
        auto gdr_handle = framework::memory::make_unique_gdr_handle();
        ASSERT_NE(gdr_handle.get(), nullptr);

        // Create a small pinned buffer
        constexpr std::size_t BUFFER_SIZE = sizeof(std::uint32_t);
        auto buffer =
                std::make_unique<framework::memory::GpinnedBuffer>(gdr_handle.get(), BUFFER_SIZE);

        // Verify buffer was created
        ASSERT_NE(buffer, nullptr);
        EXPECT_NE(buffer->get_device_addr(), nullptr);
        EXPECT_NE(buffer->get_host_addr(), nullptr);
        EXPECT_GE(buffer->get_size(), BUFFER_SIZE);

    } catch (const std::runtime_error &e) {
        FAIL() << "GDRCopy or CUDA not available: " << e.what();
    } catch (const std::invalid_argument &e) {
        FAIL() << "Invalid argument (should not happen): " << e.what();
    }
}

// Test framework::memory::GpinnedBuffer with raw gdr_t (legacy pattern)
TEST_F(GdrCopyBufferTest, GpinnedBufferWithRawHandle) {
    gdr_t handle = gdr_open();
    if (handle == nullptr) {
        FAIL() << "GDRCopy driver not available";
    }

    try {
        // Create buffer with raw handle
        constexpr std::size_t BUFFER_SIZE = sizeof(std::uint64_t);
        auto buffer = std::make_unique<framework::memory::GpinnedBuffer>(handle, BUFFER_SIZE);

        ASSERT_NE(buffer, nullptr);
        EXPECT_NE(buffer->get_device_addr(), nullptr);
        EXPECT_NE(buffer->get_host_addr(), nullptr);

        // Cleanup
        buffer.reset();
        gdr_close(handle);

    } catch (const std::runtime_error &e) {
        gdr_close(handle);
        FAIL() << "CUDA not available: " << e.what();
    }
}

// Test framework::memory::GpinnedBuffer memory access (CPU write, verify addresses)
TEST_F(GdrCopyBufferTest, GpinnedBufferMemoryAccess) {
    try {
        auto gdr_handle = framework::memory::make_unique_gdr_handle();
        ASSERT_NE(gdr_handle.get(), nullptr);

        constexpr std::size_t BUFFER_SIZE = sizeof(std::uint32_t);
        auto buffer =
                std::make_unique<framework::memory::GpinnedBuffer>(gdr_handle.get(), BUFFER_SIZE);

        // Write to host memory
        auto *host_ptr = static_cast<std::uint32_t *>(buffer->get_host_addr());
        ASSERT_NE(host_ptr, nullptr);
        *host_ptr = 0x12345678U;

        // Verify write (basic sanity check)
        EXPECT_EQ(*host_ptr, 0x12345678U);

        // Verify device address is different from host address
        auto *device_ptr = static_cast<std::uint32_t *>(buffer->get_device_addr());
        EXPECT_NE(device_ptr, host_ptr);

    } catch (const std::runtime_error &e) {
        FAIL() << "GDRCopy or CUDA not available: " << e.what();
    }
}

// Test that null handle is rejected
TEST_F(GdrCopyBufferTest, NullHandleRejected) {
    gdr_t null_handle = nullptr;
    EXPECT_THROW(
            {
                auto buffer = std::make_unique<framework::memory::GpinnedBuffer>(
                        null_handle, sizeof(std::uint32_t));
            },
            std::invalid_argument);
}

// Test that zero size is rejected
TEST_F(GdrCopyBufferTest, ZeroSizeRejected) {
    try {
        auto gdr_handle = framework::memory::make_unique_gdr_handle();
        EXPECT_THROW(
                {
                    auto buffer =
                            std::make_unique<framework::memory::GpinnedBuffer>(gdr_handle.get(), 0);
                },
                std::invalid_argument);
    } catch (const std::runtime_error &e) {
        FAIL() << "GDRCopy driver not available: " << e.what();
    }
}

// Test multiple buffers with same handle
TEST_F(GdrCopyBufferTest, MultipleBuffersWithSameHandle) {
    try {
        auto gdr_handle = framework::memory::make_unique_gdr_handle();
        ASSERT_NE(gdr_handle.get(), nullptr);

        // Create multiple buffers using the same handle
        auto buffer1 = std::make_unique<framework::memory::GpinnedBuffer>(
                gdr_handle.get(), sizeof(std::uint32_t));
        auto buffer2 = std::make_unique<framework::memory::GpinnedBuffer>(
                gdr_handle.get(), sizeof(std::uint64_t));
        auto buffer3 = std::make_unique<framework::memory::GpinnedBuffer>(
                gdr_handle.get(), sizeof(std::uint32_t) * 4);

        EXPECT_NE(buffer1->get_device_addr(), nullptr);
        EXPECT_NE(buffer2->get_device_addr(), nullptr);
        EXPECT_NE(buffer3->get_device_addr(), nullptr);

        // All buffers should have different addresses
        EXPECT_NE(buffer1->get_device_addr(), buffer2->get_device_addr());
        EXPECT_NE(buffer1->get_device_addr(), buffer3->get_device_addr());
        EXPECT_NE(buffer2->get_device_addr(), buffer3->get_device_addr());

    } catch (const std::runtime_error &e) {
        FAIL() << "GDRCopy or CUDA not available: " << e.what();
    }
}

} // namespace framework::memory::tests
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
