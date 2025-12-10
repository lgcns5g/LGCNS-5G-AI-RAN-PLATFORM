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

add_executable(rt_log_tests rt_log_tests.cpp rt_log_sample_tests.cpp temp_file.cpp)
target_include_directories(rt_log_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_system_libraries(rt_log_tests PRIVATE GTest::gtest_main)
target_link_libraries(rt_log_tests PRIVATE framework::warnings framework::options framework::rt_log)

add_test(
    NAME rt_log_tests.all_tests
    COMMAND rt_log_tests --gtest_output=xml:rt_log_tests.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
label_parallel(rt_log_tests.all_tests)

add_executable(rt_log_custom_tests rt_log_custom_tests.cpp temp_file.cpp)
target_include_directories(rt_log_custom_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_system_libraries(rt_log_custom_tests PRIVATE GTest::gtest_main)
target_link_libraries(rt_log_custom_tests PRIVATE framework::warnings framework::options
                                                  framework::rt_log)

add_test(
    NAME rt_log_custom_tests.all_tests
    COMMAND rt_log_custom_tests --gtest_output=xml:rt_log_custom_tests.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
label_parallel(rt_log_custom_tests.all_tests)
