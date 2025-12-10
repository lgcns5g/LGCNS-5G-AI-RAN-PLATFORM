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

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Dpdk.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Doca.cmake)

add_executable(
    oran_fapi_tests
    oran_tests.cpp
    oran_utils_tests.cpp
    oran_sample_tests.cpp
    cplane_tests.cpp
    fapi_to_cplane_tests.cpp
    fapi_to_oran_tests.cpp
    numerology_tests.cpp
    dpdk_buf_tests.cpp
    log_test_env.cpp)
add_executable(ran::oran_fapi_tests ALIAS oran_fapi_tests)
target_link_cuda(oran_fapi_tests)
target_link_system_libraries(oran_fapi_tests PRIVATE GTest::gtest_main)
target_link_libraries(oran_fapi_tests PRIVATE framework::warnings framework::options ran::oran
                                              framework::rt_log framework::net_test fapi_test_utils)

# Set DPDK capabilities (required for dpdk_buf_tests)
set_dpdk_capabilities(oran_fapi_tests)

# Add dependency on fapi_sample and test_mac to ensure they're built and FAPI file exists
add_dependencies(oran_fapi_tests fapi_sample test_mac test_mac_set_cap)

# Define base directory for FAPI capture files (tests construct full filename at runtime)
set(FAPI_INTEGRATION_TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/testMAC/testMAC")

# Run all oran unit tests as single test
add_test(
    NAME ran.oran_fapi.all_tests
    COMMAND $<TARGET_FILE:oran_fapi_tests> --gtest_output=xml:oran_fapi_tests.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# cmake-format: off
# Runtime configuration:
# - TEST_CELLS (default: 1), TEST_VECTOR (default: TVnr_7201_gNB_FAPI_s0.h5)
#   Override: TEST_CELLS=2 TEST_VECTOR=other.h5 ctest -R ran.oran_fapi
# - C++ tests construct capture filename: fapi_capture_fapi_sample_${TEST_CELLS}C.fapi
# - FAPI_CAPTURE_DIR is test-specific (not user-configurable) so set it here
# cmake-format: on
set_tests_properties(
    ran.oran_fapi.all_tests
    PROPERTIES ENVIRONMENT "FAPI_CAPTURE_DIR=${FAPI_INTEGRATION_TEST_OUTPUT_DIR}" FIXTURES_REQUIRED
               fapi_capture_file)

label_test_requires_nic(ran.oran_fapi.all_tests)
label_integration_test(ran.oran_fapi.all_tests)
label_parallel(ran.oran_fapi.all_tests)
