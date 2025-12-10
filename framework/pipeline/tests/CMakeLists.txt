# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pipeline Domain Tests

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

add_executable(
    framework_pipeline_tests
    kernel_descriptor_accessor_test.cpp
    kernel_launch_config_test.cpp
    module_factory_test.cpp
    pipeline_test.cpp
    pipeline_sample_tests.cpp
    # CUDA test helpers from utils
    ${CMAKE_SOURCE_DIR}/framework/utils/tests/kernel_test_dummy.cu
    ${CMAKE_SOURCE_DIR}/framework/utils/tests/kernel_test_dummy.cuh)

target_link_libraries(
    framework_pipeline_tests
    PRIVATE framework::utils
            framework::memory
            framework::tensor
            framework::tensorrt
            framework::pipeline
            framework-test-utils
            GTest::gtest_main
            framework::options
            framework::warnings)

# Need access to kernel_test_dummy.cuh from utils tests
target_include_directories(framework_pipeline_tests
                           PRIVATE ${CMAKE_SOURCE_DIR}/framework/utils/tests)

target_link_cuda(framework_pipeline_tests)

add_test(
    NAME pipeline.all_tests
    COMMAND framework_pipeline_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
label_parallel(pipeline.all_tests)
