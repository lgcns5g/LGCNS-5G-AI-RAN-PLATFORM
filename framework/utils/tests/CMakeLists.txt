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

# Framework Test Utilities Library Shared test utilities for framework component tests This library
# is defined in utils/tests/ but used across all framework domain tests

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

add_library(framework-test-utils STATIC test_logging_setup.cpp)
add_library(framework::test_utils ALIAS framework-test-utils)

target_link_libraries(framework-test-utils PUBLIC framework::utils framework::rt_log GTest::gtest
                                                  framework::options framework::warnings)

# No public headers for this internal test utility library
target_include_directories(framework-test-utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Utils Domain Tests
add_executable(
    framework_utils_tests arr_test.cpp arr_cuda_test.cu errors_test.cpp exceptions_test.cpp
                          kernel_test_dummy.cu utils_sample_tests.cpp)

target_link_libraries(
    framework_utils_tests PRIVATE framework::utils framework-test-utils GTest::gtest_main
                                  framework::options framework::warnings)

target_link_cuda(framework_utils_tests)

add_test(NAME utils.all_tests COMMAND framework_utils_tests)
label_parallel(utils.all_tests)
