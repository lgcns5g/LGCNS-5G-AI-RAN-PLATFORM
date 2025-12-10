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
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Dpdk.cmake)

# Create net_test library with helper functions for test initialization
add_library(net_test STATIC net_test_helpers.cpp)
add_library(framework::net_test ALIAS net_test)

target_link_cuda(net_test)
target_include_directories(net_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_system_libraries(net_test PUBLIC GTest::gtest)
target_link_libraries(net_test PUBLIC framework::warnings framework::options framework::net)

function(add_net_test TEST_NAME)
    set(TARGET_NAME net_${TEST_NAME}_tests)
    add_executable(${TARGET_NAME} ${ARGN})

    target_link_cuda(${TARGET_NAME})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_system_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main)
    target_link_libraries(${TARGET_NAME} PRIVATE framework::warnings framework::options
                                                 framework::net framework::net_test)

    set_dpdk_capabilities(${TARGET_NAME})

    add_test(
        NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME} --gtest_output=xml:${TARGET_NAME}.xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    label_test_requires_nic(${TARGET_NAME})
    label_parallel(${TARGET_NAME})

endfunction()

add_net_test(utils doca_utils_tests.cpp dpdk_utils_tests.cpp utils_test_env.cpp)
add_net_test(env net_env_tests.cpp log_test_env.cpp)
add_net_test(cpu_env net_cpu_only_tests.cpp log_test_env.cpp)
add_net_test(sample net_sample_tests.cpp log_test_env.cpp)

add_custom_target(net_tests DEPENDS net_utils_tests net_env_tests net_cpu_env_tests
                                    net_sample_tests)
