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

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)

# Python tooling targets for integration test script (just linting/formatting)
add_python_ruff_targets("fapi_samples" ${CMAKE_CURRENT_SOURCE_DIR})

# Function to create a FAPI sample application
function(add_fapi_sample_app APP_NAME)
    add_executable(${APP_NAME} src/${APP_NAME}.cpp src/fapi_sample_utils.cpp)

    target_link_libraries(${APP_NAME} PRIVATE framework::options framework::warnings
                                              framework::rt_log framework::task ran::fapi)
    target_link_system_libraries(${APP_NAME} PRIVATE CLI11::CLI11 gsl-lite::gsl-lite)
    target_include_directories(${APP_NAME} PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

endfunction()

# Function to add tests for a FAPI sample application
function(add_fapi_sample_tests APP_NAME)
    add_version_and_help_tests(${APP_NAME})
endfunction()

# Function to add FAPI integration test with test_mac
function(add_fapi_integration_test APP_NAME)
    # Leverage CUBB_HOME environment variable for test_mac file resolution test_mac's
    # get_root_path() checks CUBB_HOME first, then falls back to relative paths

    # Set up directories under CUBB_HOME (build directory)
    set(TEST_VECTORS_DIR "${CMAKE_BINARY_DIR}/testVectors")
    set(MULTI_CELL_DIR "${TEST_VECTORS_DIR}/multi-cell")
    set(TEST_MAC_CONFIG_DIR "${CMAKE_BINARY_DIR}/cuPHY-CP/testMAC/testMAC")
    set(NVLOG_CONFIG_DIR "${CMAKE_BINARY_DIR}/cuPHY/nvlog/config")
    file(MAKE_DIRECTORY "${MULTI_CELL_DIR}")
    file(MAKE_DIRECTORY "${TEST_MAC_CONFIG_DIR}")
    file(MAKE_DIRECTORY "${NVLOG_CONFIG_DIR}")

    # Copy nvlog config that test_mac needs (via CUBB_HOME or relative path navigation)
    configure_file("${aerial_sdk_SOURCE_DIR}/cuPHY/nvlog/config/nvlog_config.yaml"
                   "${NVLOG_CONFIG_DIR}/nvlog_config.yaml" COPYONLY)

    # Symlink all test vectors from test_data directory (zero disk space overhead)
    # Note: Cannot symlink entire directory - need writable multi-cell/ subdirectory for generated YAML files
    file(GLOB TEST_VECTOR_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/ran/test_data/*.h5")
    foreach(TV_FILE ${TEST_VECTOR_FILES})
        get_filename_component(TV_FILENAME ${TV_FILE} NAME)
        file(CREATE_LINK "${TV_FILE}" "${TEST_VECTORS_DIR}/${TV_FILENAME}" SYMBOLIC)
    endforeach()

    # Build test command
    set(TEST_MAC_OUTPUT_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/testMAC/testMAC")
    set(TEST_MAC_CONFIG_FILENAME "test_mac_fapi_sample.yaml")
    set(LAUNCH_PATTERN_TEMPLATE
        "${CMAKE_CURRENT_SOURCE_DIR}/config/launch_pattern_fapi_sample.yaml.in")
    set(TEST_MAC_CONFIG_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/config/test_mac_fapi_sample.yaml.in")

    set(BASE_TEST_COMMAND
        ${CMAKE_COMMAND} -E env CUBB_HOME=${CMAKE_BINARY_DIR} uv run
        "${CMAKE_CURRENT_SOURCE_DIR}/src/run_fapi_integration_test.py" "$<TARGET_FILE:test_mac>"
        "${TEST_MAC_OUTPUT_DIR}" "$<TARGET_FILE:${APP_NAME}>" "${TEST_MAC_CONFIG_FILENAME}"
        "${LAUNCH_PATTERN_TEMPLATE}" "${TEST_MAC_CONFIG_TEMPLATE}")

    # Main integration test - always regenerates capture file for validation
    add_test(NAME ${APP_NAME}.integration_test COMMAND ${BASE_TEST_COMMAND} --overwrite-capture)

    # Fixture setup test - only generates if file doesn't exist (for downstream test dependencies)
    add_test(NAME ${APP_NAME}.fixture_setup COMMAND ${BASE_TEST_COMMAND})

    # cmake-format: off
    # Runtime configuration:
    # - TEST_CELLS (default: 1), TEST_VECTOR (default: TVnr_7201_gNB_FAPI_s0.h5)
    # - Automatically detects launch pattern cycle length and runs one cycle
    # - Generates test_mac config and launch pattern YAML at runtime
    # cmake-format: on
    set_tests_properties(${APP_NAME}.integration_test PROPERTIES DEPENDS py_fapi_samples_setup)

    # TSAN reports known false positives from nvIPC library (uses C99 atomics that TSAN cannot see)
    # Allow the test to pass despite TSAN warnings
    if(ENABLE_SANITIZER_THREAD)
        set_tests_properties(${APP_NAME}.integration_test PROPERTIES PASS_REGULAR_EXPRESSION
                                                                     "ALL CELLS VALIDATION PASSED")
        # fixture_setup may skip if file exists, so accept either validation passed or skip message
        set_tests_properties(
            ${APP_NAME}.fixture_setup
            PROPERTIES PASS_REGULAR_EXPRESSION
                       "ALL CELLS VALIDATION PASSED|Skipping test execution")
    endif()

    set_tests_properties(${APP_NAME}.fixture_setup PROPERTIES FIXTURES_SETUP fapi_capture_file
                                                              DEPENDS py_fapi_samples_setup)

    label_integration_test(${APP_NAME}.integration_test)
    label_integration_test(${APP_NAME}.fixture_setup)
    label_real_time(${APP_NAME}.integration_test)
    label_parallel(${APP_NAME}.fixture_setup)
endfunction()

# Create the FAPI sample application
add_fapi_sample_app(fapi_sample)
add_fapi_sample_tests(fapi_sample)
add_fapi_integration_test(fapi_sample)

# Set capabilities to allow FIFO scheduling without root
set_cap_sys_nice(fapi_sample)
create_cap_sys_nice_target(test_mac)

# Add dependency on testmac to ensure they're built since integration test depends on it
add_dependencies(fapi_sample test_mac test_mac_set_cap)
