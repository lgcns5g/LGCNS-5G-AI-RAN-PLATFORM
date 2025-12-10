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

# Order Kernel Pipeline Tests
#
# Factory instantiation and API validation tests These tests exercise the factory pattern,
# module/pipeline creation, and API contracts WITHOUT requiring DOCA hardware or actual kernel
# execution

add_executable(order_kernel_pipeline_tests order_kernel_pipeline_tests.cpp)

# Link dependencies
target_link_libraries(
    order_kernel_pipeline_tests PRIVATE fronthaul-pipelines framework::pipeline rt_log
                                        framework::options framework::warnings)

target_link_system_libraries(order_kernel_pipeline_tests PRIVATE GTest::gtest_main CUDA::cudart)

# Note: We do NOT call target_link_cuda() here because: 1. This test is pure C++ (not .cu), so no
# device code to resolve 2. order-kernel-pipeline already has device symbols resolved 3. Calling
# target_link_cuda() would cause double device linking and duplicate symbols

# Link GDRCopy for order kernel tests
target_link_gdrcopy(order_kernel_pipeline_tests)

# Register with CTest
add_test(
    NAME order_kernel_pipeline.basic_tests
    COMMAND order_kernel_pipeline_tests --gtest_output=xml:order_kernel_pipeline_tests.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Dpdk.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Doca.cmake)

label_test_requires_nic(order_kernel_pipeline.basic_tests)
label_parallel(order_kernel_pipeline.basic_tests)

# Function to create fronthaul test executables with common configuration
function(add_fronthaul_test TARGET_NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

    # Create executable
    add_executable(${TARGET_NAME} ${ARG_SOURCES})
    add_executable(ran::${TARGET_NAME} ALIAS ${TARGET_NAME})

    # Link libraries
    target_link_system_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main)
    target_link_libraries(
        ${TARGET_NAME} PRIVATE framework::warnings framework::options ran::fronthaul
                               framework::rt_log framework::net_test fapi_test_utils)

    # Set DPDK capabilities (required for all fronthaul tests)
    set_dpdk_capabilities(${TARGET_NAME})

    # Add dependency on fapi_sample
    add_dependencies(${TARGET_NAME} fapi_sample)

    # Register test with CTest
    add_test(
        NAME ${TARGET_NAME}
        COMMAND $<TARGET_FILE:${TARGET_NAME}> --gtest_output=xml:${TARGET_NAME}.xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # cmake-format: off
    # Runtime configuration:
    # - TEST_CELLS (default: 1), TEST_VECTOR (default: TVnr_7201_gNB_FAPI_s0.h5)
    #   Override: TEST_CELLS=2 TEST_VECTOR=other.h5 ctest -R ${TARGET_NAME}
    # - C++ tests construct capture filename: fapi_capture_fapi_sample_${TEST_CELLS}C.fapi
    # - FAPI_CAPTURE_DIR is test-specific (not user-configurable) so set it here
    # cmake-format: on
    set_tests_properties(
        ${TARGET_NAME} PROPERTIES ENVIRONMENT "FAPI_CAPTURE_DIR=${FAPI_CAPTURE_FILE_PATH}"
                                  FIXTURES_REQUIRED fapi_capture_file)

    label_test_requires_nic(${TARGET_NAME})
    label_integration_test(${TARGET_NAME})
    label_real_time(${TARGET_NAME})
endfunction()

# Define base directory for FAPI capture files (tests construct full filename at runtime)
set(FAPI_INTEGRATION_TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/testMAC/testMAC")
set(FAPI_CAPTURE_FILE_PATH "${FAPI_INTEGRATION_TEST_OUTPUT_DIR}")

# Fronthaul tests - Fronthaul constructor initializes DPDK (uses log_test_env.cpp)
add_fronthaul_test(
    fronthaul_tests
    SOURCES
    log_test_env.cpp
    fronthaul_cplane_tests.cpp
    packet_send_time_tests.cpp
    packet_header_tests.cpp
    fronthaul_parser_tests.cpp
    fronthaul_sample_tests.cpp)

target_link_gdrcopy(fronthaul_tests)

# Add dependency on fapi_sample and testmac and ru_emulator to ensure they're built and fapi file
# exists
add_dependencies(fronthaul_tests fapi_sample test_mac test_mac_set_cap ru_emulator
                 ru_emulator_set_cap)
