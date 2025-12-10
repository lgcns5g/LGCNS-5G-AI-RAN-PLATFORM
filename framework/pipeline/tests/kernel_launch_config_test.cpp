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

#include <array>     // for array
#include <cstddef>   // for size_t
#include <limits>    // for numeric_limits
#include <stdexcept> // for runtime_error, inva...
#include <string>    // for allocator, basic_st...

#include <vector_types.h> // for dim3, __global__

#include <gtest/gtest.h> // for AssertionResult

#include "kernel_test_dummy.cuh"                     // for dummy_test_kernel
#include "pipeline/dynamic_kernel_launch_config.hpp" // for DynamicKernelLaunch...
#include "pipeline/ikernel_launch_config.hpp"        // for IKernelLaunchConfig
#include "pipeline/kernel_launch_config.hpp"         // for KernelLaunchConfig

// Safe way to get dummy kernel pointer
// Note: Linter may show false positive about undefined internal linkage
// The kernel function is defined in kernel_test_dummy.cu and linked correctly
const void *get_dummy_kernel_ptr() {
    // Required for CUDA kernel pointer
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const void *>(dummy_test_kernel);
}

namespace {

constexpr int TEST_VALUE_1 = 42;
constexpr int TEST_VALUE_2 = 100;
constexpr float TEST_FLOAT_VALUE = 3.14F;
constexpr int TEST_BLOCK_SIZE = 256;
constexpr std::size_t TEST_SHARED_MEM_SIZE = 1024;

class KernelLaunchConfigTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(KernelLaunchConfigTest, DualKernelLaunchConfigBasicSetup) {
    framework::pipeline::DualKernelLaunchConfig config;

    config.setup_kernel_function(get_dummy_kernel_ptr());
    config.setup_kernel_dimensions(dim3(1), dim3(TEST_BLOCK_SIZE));

    // Setup arguments
    int value1 = TEST_VALUE_1;
    int value2 = TEST_VALUE_2;

    config.setup_kernel_arguments(&value1, &value2);

    SUCCEED();
}

TEST_F(KernelLaunchConfigTest, DynamicKernelLaunchConfigBasicSetup) {
    framework::pipeline::DynamicKernelLaunchConfig config;

    // Setup kernel function (using a dummy function pointer)
    const void *dummy_kernel = get_dummy_kernel_ptr();
    config.setup_kernel_function(dummy_kernel);

    // Setup dimensions
    const dim3 grid_dim(2, 1, 1);
    const dim3 block_dim(512, 1, 1);
    config.setup_kernel_dimensions(grid_dim, block_dim);

    // Setup arguments
    int value1 = TEST_VALUE_1;
    int value2 = TEST_VALUE_2;
    float value3 = TEST_FLOAT_VALUE;

    config.setup_kernel_arguments(&value1, &value2, &value3);

    // Config was set up successfully if we reach this point
    EXPECT_TRUE(true);
}

TEST_F(KernelLaunchConfigTest, DualKernelLaunchConfigArgumentLimit) {
    framework::pipeline::DualKernelLaunchConfig config;

    config.setup_kernel_function(get_dummy_kernel_ptr());
    config.setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1));

    // Should work with 2 arguments
    int arg1 = 1;
    int arg2 = 2;
    EXPECT_NO_THROW(config.setup_kernel_arguments(&arg1, &arg2));

    // Should throw with more than 2 arguments
    int arg3 = 3;
    EXPECT_THROW(config.setup_kernel_arguments(&arg1, &arg2, &arg3), std::runtime_error);
}

TEST_F(KernelLaunchConfigTest, DynamicKernelLaunchConfigManyArguments) {
    framework::pipeline::DynamicKernelLaunchConfig config;

    config.setup_kernel_function(get_dummy_kernel_ptr());
    config.setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1));

    // Should work with many arguments
    static constexpr int ARG1 = 1;
    static constexpr int ARG2 = 2;
    static constexpr int ARG3 = 3;
    static constexpr int ARG4 = 4;
    static constexpr int ARG5 = 5;
    static constexpr float ARG6 = 6.0;
    static constexpr double ARG7 = 7.0;

    int arg1 = ARG1;
    int arg2 = ARG2;
    int arg3 = ARG3;
    int arg4 = ARG4;
    int arg5 = ARG5;
    float arg6 = ARG6;
    double arg7 = ARG7;

    EXPECT_NO_THROW(config.setup_kernel_arguments(&arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7));
}

TEST_F(KernelLaunchConfigTest, DualKernelLaunchConfigCompatibility) {
    // Test DualKernelLaunchConfig with the 2-argument pattern
    framework::pipeline::DualKernelLaunchConfig config;

    config.setup_kernel_function(get_dummy_kernel_ptr());
    config.setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1));

    int arg1 = 1;
    int arg2 = 2;

    // Test the standard setup method
    config.setup_kernel_arguments(&arg1, &arg2);

    // Config was set up successfully if we reach this point
    EXPECT_TRUE(true);
}

TEST_F(KernelLaunchConfigTest, PolymorphicInterface) {
    // Test that both config types work through the interface
    framework::pipeline::DualKernelLaunchConfig dual_config;
    framework::pipeline::DynamicKernelLaunchConfig dyn_config;

    const std::array<framework::pipeline::IKernelLaunchConfig *, 2> configs = {
            &dual_config, &dyn_config};

    for (auto *config : configs) {
        config->setup_kernel_function(get_dummy_kernel_ptr());
        config->setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1));

        int arg1 = 1;
        int arg2 = 2;

        config->setup_kernel_arguments(&arg1, &arg2);

        // Config was set up successfully if we reach this point without exception
        EXPECT_TRUE(true);
    }
}

TEST_F(KernelLaunchConfigTest, SharedMemorySetup) {
    framework::pipeline::DualKernelLaunchConfig config;

    config.setup_kernel_function(get_dummy_kernel_ptr());

    // Test shared memory setup
    const std::size_t shared_mem_bytes = TEST_SHARED_MEM_SIZE;
    config.setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1), shared_mem_bytes);

    // Verify setup was successful - if we reach this point without exception,
    // the shared memory configuration was set up correctly
    EXPECT_TRUE(true);
}

TEST_F(KernelLaunchConfigTest, InvalidSharedMemorySize) {
    framework::pipeline::DualKernelLaunchConfig config;

    config.setup_kernel_function(get_dummy_kernel_ptr());

    // Test that oversized shared memory throws
    const std::size_t oversized_shared_mem = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(
            config.setup_kernel_dimensions(
                    dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1), oversized_shared_mem),
            std::invalid_argument);
}

TEST_F(KernelLaunchConfigTest, TemplateKernelLaunchConfig3Args) {
    // Test the template with 3 arguments
    framework::pipeline::KernelLaunchConfig<3> config;

    config.setup_kernel_function(get_dummy_kernel_ptr());
    config.setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1));

    int arg1 = 1;
    int arg2 = 2;
    static constexpr float ARG3 = 3.0F;
    float arg3 = ARG3;

    // Should work with exactly 3 arguments
    EXPECT_NO_THROW(config.setup_kernel_arguments(&arg1, &arg2, &arg3));
}

TEST_F(KernelLaunchConfigTest, TemplateKernelLaunchConfig4Args) {
    // Test the template with 4 arguments
    framework::pipeline::KernelLaunchConfig<4> config;

    config.setup_kernel_function(get_dummy_kernel_ptr());
    config.setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1));

    int arg1 = 1;
    int arg2 = 2;
    static constexpr float ARG3 = 3.0F;
    float arg3 = ARG3;
    static constexpr double ARG4 = 4.0;
    double arg4 = ARG4;

    // Should work with exactly 4 arguments
    EXPECT_NO_THROW(config.setup_kernel_arguments(&arg1, &arg2, &arg3, &arg4));

    // Should throw with 5 arguments
    static constexpr int ARG5 = 5;
    int arg5 = ARG5;
    EXPECT_THROW(
            config.setup_kernel_arguments(&arg1, &arg2, &arg3, &arg4, &arg5), std::runtime_error);
}

TEST_F(KernelLaunchConfigTest, PolymorphicTemplateUsage) {
    // Test that template configs work through the interface
    framework::pipeline::KernelLaunchConfig<2> config2;
    framework::pipeline::KernelLaunchConfig<3> config3;
    framework::pipeline::DynamicKernelLaunchConfig config_dyn;

    const std::array<framework::pipeline::IKernelLaunchConfig *, 3> configs = {
            &config2, &config3, &config_dyn};

    for (auto *config : configs) {
        config->setup_kernel_function(get_dummy_kernel_ptr());
        config->setup_kernel_dimensions(dim3(1, 1, 1), dim3(TEST_BLOCK_SIZE, 1, 1));

        int arg1 = 1;
        int arg2 = 2;

        // All configs should at least support 2 arguments
        config->setup_kernel_arguments(&arg1, &arg2);

        // Config was set up successfully if we reach this point without exception
        EXPECT_TRUE(true);
    }
}

} // namespace
