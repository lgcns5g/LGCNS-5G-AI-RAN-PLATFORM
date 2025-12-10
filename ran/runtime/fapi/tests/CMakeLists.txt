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

# Test utilities library
add_library(fapi_test_utils STATIC fapi_test_utils.cpp)
target_include_directories(fapi_test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_system_libraries(fapi_test_utils PUBLIC tl::expected)
target_link_libraries(fapi_test_utils PUBLIC framework::warnings framework::options)

# Define base directory for FAPI capture files (tests construct full filename at runtime)
set(FAPI_INTEGRATION_TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/testMAC/testMAC")

# Function to add FAPI tests with common configuration
function(add_fapi_test SUFFIX SOURCES)
    set(TEST_NAME fapi_tests_${SUFFIX})

    add_executable(${TEST_NAME} ${SOURCES})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_system_libraries(${TEST_NAME} PRIVATE GTest::gtest_main gsl-lite::gsl-lite)
    target_link_libraries(${TEST_NAME} PRIVATE framework::warnings framework::options
                                               framework::rt_log ran::fapi fapi_test_utils)

    add_test(
        NAME fapi_tests.${SUFFIX}
        COMMAND ${TEST_NAME} --gtest_output=xml:${TEST_NAME}.xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

add_fapi_test(state "fapi_state_tests.cpp;fapi_file_io_tests.cpp;fapi_sample_tests.cpp")
label_parallel(fapi_tests.state)
add_fapi_test(file_replay "fapi_file_replay_tests.cpp;log_env.cpp")

# Add dependency on fapi_sample and test_mac to ensure they're built and FAPI file exists
add_dependencies(fapi_tests_file_replay fapi_sample test_mac test_mac_set_cap)

# cmake-format: off
# Runtime configuration:
# - TEST_CELLS (default: 1), TEST_VECTOR (default: TVnr_7201_gNB_FAPI_s0.h5)
#   Override: TEST_CELLS=2 TEST_VECTOR=other.h5 ctest -R fapi_tests
# - C++ tests construct capture filename: fapi_capture_fapi_sample_${TEST_CELLS}C.fapi
# - FAPI_CAPTURE_DIR is test-specific (not user-configurable) so set it here
# cmake-format: on
set_tests_properties(
    fapi_tests.file_replay
    PROPERTIES ENVIRONMENT "FAPI_CAPTURE_DIR=${FAPI_INTEGRATION_TEST_OUTPUT_DIR}" FIXTURES_REQUIRED
               fapi_capture_file)

label_integration_test(fapi_tests.file_replay)
label_real_time(fapi_tests.file_replay)
