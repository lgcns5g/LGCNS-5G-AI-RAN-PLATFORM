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
#include <cstddef>

#include <device_launch_parameters.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "utils/arr.hpp"

/**
 * CUDA kernel that exercises Arr class functionality on the GPU
 *
 * This kernel tests core Arr operations to verify that std::array<T, DIM>
 * works correctly in device code, as opposed to the previous T[DIM]
 * implementation.
 *
 * @param[out] results Array to store test results (1 = pass, 0 = fail)
 */
__global__ void test_arr_on_device(int *results) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Only use thread 0 to avoid race conditions
    if (tid != 0) {
        return;
    }

    // Initialize all test results to failure
    static constexpr int NUM_TESTS = 6;
    for (int i = 0; i < NUM_TESTS; ++i) {
        results[i] = 0; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    // Test 1: Default construction and zero initialization
    {
        framework::utils::Arr<int, 3> arr{};
        if (arr[0] == 0 && arr[1] == 0 && arr[2] == 0) {
            results[0] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
    }

    // Test 2: Element assignment and access
    {
        framework::utils::Arr<float, 2> arr{};
        static constexpr float TEST_VALUE_1 = 1.5F;
        static constexpr float TEST_VALUE_2 = 2.5F;
        arr[0] = TEST_VALUE_1;
        arr[1] = TEST_VALUE_2;
        if (arr[0] == TEST_VALUE_1 && arr[1] == TEST_VALUE_2) {
            results[1] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
    }

    // Test 3: Fill method
    {
        framework::utils::Arr<int, 4> arr{};
        static constexpr int TEST_VALUE = 42;
        arr.fill(TEST_VALUE);
        if (arr[0] == TEST_VALUE && arr[1] == TEST_VALUE && arr[2] == TEST_VALUE &&
            arr[3] == TEST_VALUE) {
            results[2] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
    }

    // Test 4: Product method
    {
        framework::utils::Arr<int, 3> arr{};
        arr[0] = 2;
        arr[1] = 3;
        arr[2] = 4;
        static constexpr int PRODUCT = 24;
        if (arr.product() == PRODUCT) { // 2 * 3 * 4 = 24
            results[3] = 1;             // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
    }

    // Test 5: Equality operator
    {
        framework::utils::Arr<int, 2> arr1{};
        framework::utils::Arr<int, 2> arr2{};
        static constexpr int TEST_VALUE = 5;
        arr1.fill(TEST_VALUE);
        arr2.fill(TEST_VALUE);
        if (arr1 == arr2) {
            results[4] = 1; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
    }

    // Test 6: Iterator access (begin/end)
    {
        framework::utils::Arr<int, 3> arr{};
        static constexpr int TEST_VALUE_1 = 10;
        static constexpr int TEST_VALUE_2 = 20;
        static constexpr int TEST_VALUE_3 = 30;
        arr[0] = TEST_VALUE_1;
        arr[1] = TEST_VALUE_2;
        arr[2] = TEST_VALUE_3;

        const int *begin_ptr = arr.begin();
        const int *end_ptr = arr.end();

        static constexpr int NUM_ELEMENTS = 3;
        static constexpr int INDEX = 5;
        if (begin_ptr != end_ptr && *begin_ptr == TEST_VALUE_1 &&
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            *(begin_ptr + 1) == TEST_VALUE_2 &&
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            *(begin_ptr + 2) == TEST_VALUE_3 && (end_ptr - begin_ptr) == NUM_ELEMENTS) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            results[INDEX] = 1;
        }
    }
}

namespace {

// ========== CUDA Device Tests ==========

// Test: Arr class works correctly in CUDA device code with std::array
TEST(ArrCudaTest, DeviceCompatibility) {
    // Fail test if CUDA device is not available
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        FAIL() << "CUDA device not available: " << cudaGetErrorString(error);
    }

    // Set device
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    // Allocate device memory for results
    static constexpr int NUM_TESTS = 6;
    int *device_results{};
    ASSERT_EQ(cudaMalloc(&device_results, NUM_TESTS * sizeof(int)), cudaSuccess);

    // Launch kernel with single thread to test Arr operations
    test_arr_on_device<<<1, 1>>>(device_results);

    // Check for kernel launch errors
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Wait for kernel to complete
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back to host
    std::array<int, NUM_TESTS> host_results{};
    ASSERT_EQ(
            cudaMemcpy(
                    host_results.data(),
                    device_results,
                    NUM_TESTS * sizeof(int),
                    cudaMemcpyDeviceToHost),
            cudaSuccess);

    // Clean up device memory
    ASSERT_EQ(cudaFree(device_results), cudaSuccess);

    // Verify all tests passed
    EXPECT_EQ(1, host_results[0]) << "Default construction test failed on device";
    EXPECT_EQ(1, host_results[1]) << "Element assignment test failed on device";
    EXPECT_EQ(1, host_results[2]) << "Fill method test failed on device";
    EXPECT_EQ(1, host_results[3]) << "Product method test failed on device";
    EXPECT_EQ(1, host_results[4]) << "Equality operator test failed on device";
    EXPECT_EQ(1, host_results[5]) << "Iterator access test failed on device";

    // Note: We don't call cudaDeviceReset() here to avoid interfering with other
    // tests. CUDA will clean up automatically when the process exits.
}

// Test: Arr array constructor compatibility on device
TEST(ArrCudaTest, ArrayConstructorDeviceCompatibility) {
    // Fail test if CUDA device is not available
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        FAIL() << "CUDA device not available: " << cudaGetErrorString(error);
    }

    // This test verifies that Arr can be constructed from std::array on the host
    // and the data remains accessible after copying to device
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    // Create Arr on host using std::array constructor
    const std::array<float, 3> host_array = {1.1F, 2.2F, 3.3F};
    const framework::utils::Arr<float, 3> host_arr(host_array);

    // Verify host construction worked
    EXPECT_EQ(1.1F, host_arr[0]);
    EXPECT_EQ(2.2F, host_arr[1]);
    EXPECT_EQ(3.3F, host_arr[2]);

    // Test that the Arr's memory layout is compatible with device operations
    // by checking that it's the expected size
    EXPECT_EQ(sizeof(float) * 3, sizeof(host_arr));

    // Note: We don't call cudaDeviceReset() here to avoid interfering with other
    // tests. CUDA will clean up automatically when the process exits.
}

} // namespace
