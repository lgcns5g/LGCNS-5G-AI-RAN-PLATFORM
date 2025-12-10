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
 * @file utils_sample_tests.cpp
 * @brief Sample tests for utils library documentation
 */

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/arr.hpp"
#include "utils/cuda_stream.hpp"
#include "utils/error_macros.hpp"
#include "utils/errors.hpp"
#include "utils/exceptions.hpp"
#include "utils/string_hash.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {

using CudaStream = framework::utils::CudaStream;
using NvErrc = framework::utils::NvErrc;
using CudaRuntimeException = framework::utils::CudaRuntimeException;
using CudaDriverException = framework::utils::CudaDriverException;
template <typename T, std::size_t N> using Arr = framework::utils::Arr<T, N>;
using TransparentStringHash = framework::utils::TransparentStringHash;

TEST(UtilsSampleTests, BasicCudaStream) {
    // example-begin basic-stream-1
    // Create a CUDA stream with automatic lifetime management
    const CudaStream stream;

    // Use the stream with CUDA operations
    cudaStream_t handle = stream.get();

    // Synchronize the stream
    const bool success = stream.synchronize();
    // example-end basic-stream-1

    EXPECT_TRUE(success);
    EXPECT_NE(handle, nullptr);
}

TEST(UtilsSampleTests, MoveCudaStream) {
    // example-begin move-stream-1
    // Create initial stream
    CudaStream stream1;
    cudaStream_t handle1 = stream1.get();

    // Move stream ownership
    const CudaStream stream2 = std::move(stream1);
    cudaStream_t handle2 = stream2.get();
    // example-end move-stream-1

    EXPECT_EQ(handle1, handle2);
    EXPECT_NE(handle2, nullptr);
}

TEST(UtilsSampleTests, ErrorCodeBasic) {
    // example-begin error-code-1
    // Create error code from NvErrc enum
    const NvErrc error_code = NvErrc::Success;

    // Check if error represents success
    const bool is_success = (error_code == NvErrc::Success);

    // Convert to std::error_code for standard error handling
    const std::error_code std_error = make_error_code(error_code);
    // example-end error-code-1

    EXPECT_TRUE(is_success);
    EXPECT_FALSE(std_error);
}

TEST(UtilsSampleTests, CudaRuntimeException) {
    // Check CUDA device availability
    int device_count{};
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        FAIL() << "No CUDA devices available. cudaGetDeviceCount returned: "
               << cudaGetErrorString(error) << " (device count: " << device_count << ")";
    }

    // example-begin cuda-runtime-exception-1
    try {
        // Simulate CUDA error by calling invalid API
        FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaSetDevice(9999));
    } catch (const CudaRuntimeException &ex) {
        // Exception caught and error message available
        const char *error_msg = ex.what();
        const bool caught = true;
        // example-end cuda-runtime-exception-1

        EXPECT_TRUE(caught);
        EXPECT_NE(error_msg, nullptr);
    }
}

TEST(UtilsSampleTests, CudaDriverException) {
    // example-begin cuda-driver-exception-1
    try {
        // Initialize CUDA driver
        FRAMEWORK_CUDA_DRIVER_CHECK_THROW(cuInit(0));

        CUdevice device{};
        // Simulate driver API error
        FRAMEWORK_CUDA_DRIVER_CHECK_THROW(cuDeviceGet(&device, 9999));
    } catch (const CudaDriverException &ex) {
        // Exception caught with driver error details
        const std::string error_msg = ex.what();
        const bool caught = true;
        // example-end cuda-driver-exception-1

        EXPECT_TRUE(caught);
        EXPECT_FALSE(error_msg.empty());
    }
}

TEST(UtilsSampleTests, ThrowIfCondition) {
    // example-begin throw-if-1
    const int value = 42;

    try {
        // Throw exception if condition is met
        FRAMEWORK_NV_THROW_IF(value > 100, std::runtime_error, "Value exceeds maximum");

        // This code executes because condition is false
        const bool condition_passed = true;
        // example-end throw-if-1

        EXPECT_TRUE(condition_passed);
    } catch (const std::runtime_error &) {
        FAIL() << "Should not throw";
    }
}

TEST(UtilsSampleTests, ArrBasic) {
    // example-begin arr-basic-1
    // Create fixed-size array with 3 elements
    Arr<float, 3> vec;

    // Access elements
    vec[0] = 1.0F;
    vec[1] = 2.0F;
    vec[2] = 3.0F;

    // Get size
    const std::size_t size = vec.size();
    // example-end arr-basic-1

    EXPECT_EQ(size, 3);
    EXPECT_FLOAT_EQ(vec[0], 1.0F);
}

TEST(UtilsSampleTests, ArrIterators) {
    // example-begin arr-iterators-1
    Arr<int, 4> arr;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;

    // Iterate using range-based for loop
    int sum = 0;
    for (const int value : arr) {
        sum += value;
    }
    // example-end arr-iterators-1

    EXPECT_EQ(sum, 100);
}

TEST(UtilsSampleTests, ArrAccessData) {
    // example-begin arr-access-1
    Arr<double, 5> arr;
    arr[0] = 1.5;
    arr[1] = 2.5;
    arr[2] = 3.5;

    // Access underlying data pointer
    const double *data_ptr = arr.data();

    // Get array size
    const std::size_t array_size = arr.size();
    // example-end arr-access-1

    EXPECT_EQ(array_size, 5);
    EXPECT_NE(data_ptr, nullptr);
    EXPECT_DOUBLE_EQ(*data_ptr, 1.5);
}

TEST(UtilsSampleTests, TransparentStringHashBasic) {
    // example-begin transparent-hash-1
    // Create unordered_map with transparent string hash
    std::unordered_map<std::string, int, TransparentStringHash, std::equal_to<>> map;

    // Insert entries
    map["first"] = 1;
    map["second"] = 2;

    // Lookup using string_view without allocating temporary string
    const std::string_view key = "first";
    const auto it = map.find(key);
    const bool found = (it != map.end());
    // example-end transparent-hash-1

    EXPECT_TRUE(found);
    EXPECT_EQ(it->second, 1);
}

TEST(UtilsSampleTests, TransparentStringHashLookup) {
    // example-begin transparent-hash-lookup-1
    std::unordered_map<std::string, std::string, TransparentStringHash, std::equal_to<>> modules;
    modules["cuda"] = "CUDA Runtime";
    modules["driver"] = "CUDA Driver";

    // Efficient lookup with string literal (no allocation)
    const bool has_cuda = modules.contains("cuda");

    // Lookup with string_view (no allocation)
    const std::string_view driver_key = "driver";
    const auto driver_it = modules.find(driver_key);
    // example-end transparent-hash-lookup-1

    EXPECT_TRUE(has_cuda);
    EXPECT_NE(driver_it, modules.end());
}

TEST(UtilsSampleTests, ErrorCodeConversion) {
    // example-begin error-conversion-1
    // Work with NvErrc error codes
    const NvErrc nv_error = NvErrc::InvalidArgument;

    // Convert to standard error_code
    const std::error_code ec = make_error_code(nv_error);

    // Check error category
    const std::string category_name = ec.category().name();

    // Get error message
    const std::string message = ec.message();
    // example-end error-conversion-1

    EXPECT_TRUE(ec);
    EXPECT_FALSE(category_name.empty());
    EXPECT_FALSE(message.empty());
}

} // namespace

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
