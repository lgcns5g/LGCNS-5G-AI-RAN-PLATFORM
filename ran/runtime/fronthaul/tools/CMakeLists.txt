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

include(${CMAKE_SOURCE_DIR}/cmake/Python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)

# Just linting/formatting targets for tool scripts
add_python_ruff_targets("fronthaul_tools" ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_TESTING)
    enable_testing()

    # Test 1: Verify script runs with --help
    add_test(
        NAME fronthaul_tools.loopback_help
        COMMAND uv --directory ${CMAKE_CURRENT_SOURCE_DIR} run src/detect_loopback.py --help
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(fronthaul_tools.loopback_help PROPERTIES DEPENDS py_fronthaul_tools_setup)
    label_parallel(fronthaul_tools.loopback_help)

    # Test 2: Dry-run mode (no actual packet sending, just interface discovery)
    add_test(
        NAME fronthaul_tools.loopback_dry_run
        COMMAND uv --directory ${CMAKE_CURRENT_SOURCE_DIR} run src/detect_loopback.py --dry-run
                --output ${CMAKE_CURRENT_BINARY_DIR}/loopback_dry_run.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(fronthaul_tools.loopback_dry_run PROPERTIES DEPENDS
                                                                     py_fronthaul_tools_setup)
    label_parallel(fronthaul_tools.loopback_dry_run)

    # Test 3: Full detection (requires CAP_NET_RAW capability) NOTE: Script will attempt to set
    # CAP_NET_RAW on first run if missing Uses default behavior (stop after first detection) for
    # faster testing
    add_test(
        NAME fronthaul_tools.detect_loopback
        COMMAND uv --directory ${CMAKE_CURRENT_SOURCE_DIR} run src/detect_loopback.py --output
                ${CMAKE_CURRENT_BINARY_DIR}/loopback_results.json --verbose
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(fronthaul_tools.detect_loopback PROPERTIES DEPENDS
                                                                    py_fronthaul_tools_setup)
    label_test_requires_nic(fronthaul_tools.detect_loopback)
    label_integration_test(fronthaul_tools.detect_loopback)
endif()

# Function to add fronthaul + RU emulator integration test
function(add_fronthaul_integration_test APP_NAME)
    # CUBB_HOME directory structure (only needed by ru_emulator, not fronthaul_app)
    set(RU_EMULATOR_CONFIG_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/ru-emulator/ru_emulator")

    file(MAKE_DIRECTORY "${RU_EMULATOR_CONFIG_DIR}")

    # FAPI capture file base directory (Python script constructs full filename at runtime)
    set(FAPI_INTEGRATION_TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/testMAC/testMAC")

    # ru_emulator working directory
    set(RU_EMULATOR_WORKING_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/ru-emulator/ru_emulator")

    # RU emulator config template
    set(RU_EMULATOR_CONFIG_TEMPLATE
        "${CMAKE_SOURCE_DIR}/ran/runtime/fronthaul/tools/config/ru_emulator_config.yaml.in")
    if(NOT EXISTS "${RU_EMULATOR_CONFIG_TEMPLATE}")
        message(FATAL_ERROR "RU emulator config template not found: ${RU_EMULATOR_CONFIG_TEMPLATE}")
    endif()

    # Build test command - Python script constructs capture filename dynamically. Pass FAPI capture
    # directory, not full filename (filename depends on runtime TEST_CELLS)
    set(TEST_COMMAND
        ${CMAKE_COMMAND} -E env CUBB_HOME=${CMAKE_BINARY_DIR} uv --directory
        "${CMAKE_SOURCE_DIR}/ran/runtime/fronthaul/tools" run src/run_fronthaul_integration_test.py
        "$<TARGET_FILE:ru_emulator>" "${RU_EMULATOR_WORKING_DIR}" "${RU_EMULATOR_CONFIG_TEMPLATE}"
        "$<TARGET_FILE:${APP_NAME}>" "${FAPI_INTEGRATION_TEST_OUTPUT_DIR}")

    # Add CTest
    add_test(NAME ${APP_NAME}.integration_test COMMAND ${TEST_COMMAND})

    # cmake-format: off
    # Runtime configuration:
    # - TEST_CELLS (default: 1) controls number of cells to test
    # - TEST_SLOTS (default: 100) controls test duration; timeout auto-calculated
    # - FAPI_CAPTURE_DIR is test-specific (not user-configurable)
    # cmake-format: on
    set_tests_properties(
        ${APP_NAME}.integration_test
        PROPERTIES FIXTURES_REQUIRED fapi_capture_file ENVIRONMENT
                   "FAPI_CAPTURE_DIR=${FAPI_INTEGRATION_TEST_OUTPUT_DIR}")

    label_test_requires_nic(${APP_NAME}.integration_test)
    label_integration_test(${APP_NAME}.integration_test)
    label_real_time(${APP_NAME}.integration_test)
endfunction()

# Set DPDK capabilities on ru_emulator (imported target from aerial_sdk)
create_dpdk_capabilities_target(ru_emulator)
