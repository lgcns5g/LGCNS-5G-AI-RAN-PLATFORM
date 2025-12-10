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

# Sample Pipeline Integration Tests
#
# End-to-end tests for SamplePipeline with TensorRT and CUDA kernel modules

include("${CMAKE_SOURCE_DIR}/cmake/CpuArchitecture.cmake")
include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

set(TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/framework/pipeline/samples/engines")

add_executable(sample_pipeline_tests sample_pipeline_test.cpp)

# Link dependencies
target_link_libraries(sample_pipeline_tests PRIVATE sample-pipeline framework::pipeline rt_log
                                                    framework::options framework::warnings)

target_link_system_libraries(sample_pipeline_tests PRIVATE GTest::gtest_main CUDA::cudart)

# Enable CUDA support
target_link_cuda(sample_pipeline_tests)

# Detect target architecture
detect_cpu_architecture(TARGET_ARCH)

# Register with CTest
add_test(
    NAME sample_pipeline.integration_tests
    COMMAND sample_pipeline_tests --gtest_output=xml:sample_pipeline_tests.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
label_parallel(sample_pipeline.integration_tests)

# Set test properties - ensure TensorRT engine path is accessible
set_tests_properties(sample_pipeline.integration_tests PROPERTIES ENVIRONMENT
                                                                  "TEST_DATA_DIR=${TEST_DATA_DIR}")

# Disable test as it may not work on all architectures.  The pre-compiled TRT engine is for x86_64
# and sm90 only.
set_tests_properties(sample_pipeline.integration_tests PROPERTIES DISABLED TRUE)
message(STATUS "Sample pipeline tests disabled")
