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
 * @file tensor_sample_tests.cpp
 * @brief Sample tests for tensor library documentation
 */

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/data_types.hpp"
#include "tensor/tensor_arena.hpp"
#include "tensor/tensor_info.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

TEST(TensorSampleTests, BasicTensorInfo) {
    // example-begin basic-tensor-info-1
    // Create tensor descriptor for a 3D array of float32 values
    const framework::tensor::TensorInfo tensor(framework::tensor::TensorR32F, {10, 20, 30});

    // Get tensor properties
    const auto type = tensor.get_type();
    const auto &dimensions = tensor.get_dimensions();
    const auto total_elements = tensor.get_total_elements();
    // example-end basic-tensor-info-1

    EXPECT_EQ(type, framework::tensor::TensorR32F);
    EXPECT_EQ(dimensions.size(), 3);
    EXPECT_EQ(dimensions[0], 10);
    EXPECT_EQ(dimensions[1], 20);
    EXPECT_EQ(dimensions[2], 30);
    EXPECT_EQ(total_elements, 6000);
}

TEST(TensorSampleTests, TensorCompatibility) {
    // example-begin tensor-compatibility-1
    // Create two tensor descriptors
    const framework::tensor::TensorInfo tensor1(framework::tensor::TensorR32F, {100, 200});
    const framework::tensor::TensorInfo tensor2(framework::tensor::TensorR32F, {100, 200});
    const framework::tensor::TensorInfo tensor3(framework::tensor::TensorC32F, {100, 200});

    // Check compatibility
    const auto compatible_same = tensor1.is_compatible_with(tensor2);
    const auto compatible_different = tensor1.is_compatible_with(tensor3);
    // example-end tensor-compatibility-1

    EXPECT_TRUE(compatible_same);
    EXPECT_FALSE(compatible_different);
}

TEST(TensorSampleTests, DataTypeStrings) {
    // example-begin data-type-strings-1
    // Get string representation of data types
    const auto *const float_name =
            framework::tensor::nv_get_data_type_string(framework::tensor::TensorR32F);
    const auto *const complex_name =
            framework::tensor::nv_get_data_type_string(framework::tensor::TensorC32F);
    const auto *const int_name =
            framework::tensor::nv_get_data_type_string(framework::tensor::TensorR32I);
    // example-end data-type-strings-1

    EXPECT_STREQ(float_name, "TensorR32F");
    EXPECT_STREQ(complex_name, "TensorC32F");
    EXPECT_STREQ(int_name, "TensorR32I");
}

TEST(TensorSampleTests, DataTypeTraits) {
    // example-begin data-type-traits-1
    // Use type traits to get C++ types from tensor types
    using FloatType = framework::tensor::data_type_traits<framework::tensor::TensorR32F>::Type;
    using IntType = framework::tensor::data_type_traits<framework::tensor::TensorR32I>::Type;

    // Reverse mapping from C++ types to tensor types
    constexpr auto FLOAT_TYPE = framework::tensor::type_to_tensor_type<float>::VALUE;
    constexpr auto INT_TYPE = framework::tensor::type_to_tensor_type<int>::VALUE;

    // Check type sizes
    const auto float_size = sizeof(FloatType);
    const auto int_size = sizeof(IntType);
    // example-end data-type-traits-1

    EXPECT_EQ(FLOAT_TYPE, framework::tensor::TensorR32F);
    EXPECT_EQ(INT_TYPE, framework::tensor::TensorR32I);
    EXPECT_EQ(float_size, sizeof(float));
    EXPECT_EQ(int_size, sizeof(int));
}

TEST(TensorSampleTests, StorageElementSize) {
    // example-begin storage-element-size-1
    // Get storage size for different data types
    const auto float_size =
            framework::tensor::get_nv_type_storage_element_size(framework::tensor::TensorR32F);
    const auto complex_size =
            framework::tensor::get_nv_type_storage_element_size(framework::tensor::TensorC32F);
    const auto half_size =
            framework::tensor::get_nv_type_storage_element_size(framework::tensor::TensorR16F);
    // example-end storage-element-size-1

    EXPECT_EQ(float_size, 4);
    EXPECT_EQ(complex_size, 8);
    EXPECT_EQ(half_size, 2);
}

TEST(TensorSampleTests, DeviceArena) {
    // example-begin device-arena-1
    // Allocate device memory arena
    const std::size_t arena_size = static_cast<std::size_t>(1024) * 1024; // 1 MB
    framework::tensor::TensorArena arena(arena_size, framework::tensor::MemoryType::Device);

    // Allocate typed regions within arena
    auto *float_ptr = arena.allocate_at<float>(0);
    auto *int_ptr = arena.allocate_at<int>(1024);

    // Get arena properties
    const auto total_bytes = arena.total_bytes();
    const auto mem_type = arena.memory_type();
    // example-end device-arena-1

    EXPECT_NE(float_ptr, nullptr);
    EXPECT_NE(int_ptr, nullptr);
    EXPECT_EQ(total_bytes, arena_size);
    EXPECT_EQ(mem_type, framework::tensor::MemoryType::Device);
}

TEST(TensorSampleTests, HostPinnedArena) {
    // example-begin host-pinned-arena-1
    // Allocate pinned host memory arena for efficient CPU-GPU transfers
    const std::size_t buffer_size = 4096;
    framework::tensor::TensorArena host_arena(
            buffer_size, framework::tensor::MemoryType::HostPinned);

    // Get raw memory pointer for CUDA operations
    void *raw_memory = host_arena.raw_ptr();

    const auto mem_type = host_arena.memory_type();
    // example-end host-pinned-arena-1

    EXPECT_NE(raw_memory, nullptr);
    EXPECT_EQ(mem_type, framework::tensor::MemoryType::HostPinned);
}

TEST(TensorSampleTests, CompleteExample) {
    // example-begin complete-example-1
    // Create tensor descriptor
    const framework::tensor::TensorInfo tensor_desc(framework::tensor::TensorR32F, {128, 256});
    const auto element_count = tensor_desc.get_total_elements();
    const auto element_size =
            framework::tensor::get_nv_type_storage_element_size(framework::tensor::TensorR32F);
    const auto total_size = element_count * element_size;

    // Allocate device memory for tensor
    framework::tensor::TensorArena device_memory(total_size, framework::tensor::MemoryType::Device);
    auto *tensor_data = device_memory.allocate_at<float>(0);

    // Allocate host pinned memory for transfers
    framework::tensor::TensorArena host_memory(
            total_size, framework::tensor::MemoryType::HostPinned);
    auto *host_data = host_memory.allocate_at<float>(0);
    // example-end complete-example-1

    EXPECT_NE(tensor_data, nullptr);
    EXPECT_NE(host_data, nullptr);
    EXPECT_EQ(element_count, 128 * 256);
    EXPECT_EQ(total_size, element_count * sizeof(float));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
