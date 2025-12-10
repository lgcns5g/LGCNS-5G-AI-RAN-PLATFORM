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
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/types.hpp"

namespace {

// Mock kernel parameter structure for testing
struct KernelParams {
    int value{};
    float factor{};
    char flag{};
};

/**
 * Test fixture for KernelDescriptorAccessor tests
 */
class KernelDescriptorAccessorTest : public ::testing::Test {
protected:
    // Test constants
    static constexpr std::size_t MEMORY_SIZE = 2048;

    void SetUp() override { setup_memory_slices(); }

    void TearDown() override { cleanup_memory(); }

    void setup_memory_slices() {
        // Allocate memory for test scenarios
        // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
        static_cpu_memory_ = std::make_unique<std::byte[]>(MEMORY_SIZE);
        dynamic_cpu_memory_ = std::make_unique<std::byte[]>(MEMORY_SIZE);
        static_gpu_memory_ = std::make_unique<std::byte[]>(MEMORY_SIZE);
        dynamic_gpu_memory_ = std::make_unique<std::byte[]>(MEMORY_SIZE);
        // NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

        // Setup valid memory slice
        valid_memory_slice_.static_kernel_descriptor_cpu_ptr = static_cpu_memory_.get();
        valid_memory_slice_.static_kernel_descriptor_gpu_ptr = static_gpu_memory_.get();
        valid_memory_slice_.dynamic_kernel_descriptor_cpu_ptr = dynamic_cpu_memory_.get();
        valid_memory_slice_.dynamic_kernel_descriptor_gpu_ptr = dynamic_gpu_memory_.get();
        valid_memory_slice_.static_kernel_descriptor_bytes = MEMORY_SIZE;
        valid_memory_slice_.dynamic_kernel_descriptor_bytes = MEMORY_SIZE;
        valid_memory_slice_.device_tensor_bytes = 0;

        // Setup empty memory slice
        empty_memory_slice_ = framework::pipeline::ModuleMemorySlice{};
    }

    void cleanup_memory() {
        static_cpu_memory_.reset();
        dynamic_cpu_memory_.reset();
        static_gpu_memory_.reset();
        dynamic_gpu_memory_.reset();
    }

    // Memory slices for different test scenarios
    framework::pipeline::ModuleMemorySlice valid_memory_slice_{};
    framework::pipeline::ModuleMemorySlice empty_memory_slice_{};

    // Memory storage - using C-style arrays with unique_ptr for byte allocation
    // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
    std::unique_ptr<std::byte[]> static_cpu_memory_;
    std::unique_ptr<std::byte[]> dynamic_cpu_memory_;
    std::unique_ptr<std::byte[]> static_gpu_memory_;
    std::unique_ptr<std::byte[]> dynamic_gpu_memory_;
    // NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
};

TEST_F(KernelDescriptorAccessorTest, ConstructorBasic) {
    EXPECT_NO_THROW(
            { const framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_); });
}

TEST_F(KernelDescriptorAccessorTest, CreateStaticParamBasic) {
    framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);

    auto &params = accessor.create_static_param<KernelParams>(0);
    EXPECT_EQ(params.value, 0);
    EXPECT_EQ(params.factor, 0.0F);
    EXPECT_EQ(params.flag, 0);
}

TEST_F(KernelDescriptorAccessorTest, CreateDynamicParamBasic) {
    framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);

    auto &params = accessor.create_dynamic_param<KernelParams>(0);
    EXPECT_EQ(params.value, 0);
    EXPECT_EQ(params.factor, 0.0F);
    EXPECT_EQ(params.flag, 0);
}

TEST_F(KernelDescriptorAccessorTest, CreateParamWithOffset) {
    framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);
    const std::size_t offset = sizeof(KernelParams);

    auto &static_params = accessor.create_static_param<KernelParams>(offset);
    auto &dynamic_params = accessor.create_dynamic_param<KernelParams>(offset);

    EXPECT_EQ(static_params.value, 0);
    EXPECT_EQ(dynamic_params.value, 0);
}

TEST_F(KernelDescriptorAccessorTest, CreateStaticParamBoundsCheckSuccess) {
    framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);
    const std::size_t max_valid_offset = MEMORY_SIZE - sizeof(KernelParams);

    EXPECT_NO_THROW(
            { std::ignore = accessor.create_static_param<KernelParams>(max_valid_offset); });
}

TEST_F(KernelDescriptorAccessorTest, CreateStaticParamBoundsCheckFailure) {
    framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);
    const std::size_t invalid_offset = MEMORY_SIZE - sizeof(KernelParams) + 1;

    EXPECT_THROW(
            { std::ignore = accessor.create_static_param<KernelParams>(invalid_offset); },
            std::runtime_error);
}

TEST_F(KernelDescriptorAccessorTest, CreateDynamicParamBoundsCheckFailure) {
    framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);
    const std::size_t invalid_offset = MEMORY_SIZE - sizeof(KernelParams) + 1;

    EXPECT_THROW(
            { std::ignore = accessor.create_dynamic_param<KernelParams>(invalid_offset); },
            std::runtime_error);
}

TEST_F(KernelDescriptorAccessorTest, CreateParamEmptySlice) {
    framework::pipeline::KernelDescriptorAccessor accessor(empty_memory_slice_);

    EXPECT_THROW(
            { std::ignore = accessor.create_static_param<KernelParams>(0); }, std::runtime_error);

    EXPECT_THROW(
            { std::ignore = accessor.create_dynamic_param<KernelParams>(0); }, std::runtime_error);
}

TEST_F(KernelDescriptorAccessorTest, GetDevicePtrBasic) {
    const framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);

    auto *static_ptr = accessor.get_static_device_ptr<KernelParams>(0);
    auto *dynamic_ptr = accessor.get_dynamic_device_ptr<KernelParams>(0);

    EXPECT_NE(static_ptr, nullptr);
    EXPECT_NE(dynamic_ptr, nullptr);

    // Verify device pointers point to expected locations
    EXPECT_EQ(
            static_cast<void *>(static_ptr),
            static_cast<void *>(valid_memory_slice_.static_kernel_descriptor_gpu_ptr));
    EXPECT_EQ(
            static_cast<void *>(dynamic_ptr),
            static_cast<void *>(valid_memory_slice_.dynamic_kernel_descriptor_gpu_ptr));
}

TEST_F(KernelDescriptorAccessorTest, GetDevicePtrBoundsCheckFailure) {
    const framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);
    const std::size_t invalid_offset = MEMORY_SIZE - sizeof(KernelParams) + 1;

    EXPECT_THROW(
            { std::ignore = accessor.get_static_device_ptr<KernelParams>(invalid_offset); },
            std::runtime_error);

    EXPECT_THROW(
            { std::ignore = accessor.get_dynamic_device_ptr<KernelParams>(invalid_offset); },
            std::runtime_error);
}

TEST_F(KernelDescriptorAccessorTest, MultipleParameterCreation) {
    framework::pipeline::KernelDescriptorAccessor accessor(valid_memory_slice_);

    // Create multiple parameters at different offsets
    const std::size_t offset1 = 0;
    const std::size_t offset2 = sizeof(KernelParams);
    const std::size_t offset3 = sizeof(KernelParams) * 2;

    auto &params1 = accessor.create_static_param<KernelParams>(offset1);
    auto &params2 = accessor.create_static_param<KernelParams>(offset2);
    auto &params3 = accessor.create_static_param<KernelParams>(offset3);

    // Verify they're different objects
    EXPECT_NE(&params1, &params2);
    EXPECT_NE(&params2, &params3);
    EXPECT_NE(&params1, &params3);
}

} // namespace
