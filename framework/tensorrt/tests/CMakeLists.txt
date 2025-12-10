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

enable_testing()

include(GoogleTest)
include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

add_executable(tensorrt_tests_sample tensorrt_sample_tests.cpp mlir_trt_engine_test.cpp
                                     mlir_add_kernel_engine_test.cpp)

target_include_directories(tensorrt_tests_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensorrt_tests_sample PRIVATE framework::warnings framework::options
                                                    framework::tensorrt)
target_link_system_libraries(tensorrt_tests_sample PRIVATE GTest::gtest_main)

add_test(
    NAME tensorrt_tests.sample
    COMMAND tensorrt_tests_sample --gtest_output=xml:tensorrt_tests_sample.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
label_parallel(tensorrt_tests.sample)

add_custom_target(tensorrt_tests DEPENDS tensorrt_tests_sample)
