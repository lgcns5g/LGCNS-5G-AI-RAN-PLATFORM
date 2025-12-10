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

include("${CMAKE_SOURCE_DIR}/cmake/RanTrtPlugin.cmake")

# Include necessary CMake modules
include(${CMAKE_SOURCE_DIR}/cmake/Python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GDRCopy.cmake)

# Python tooling targets for integration test script (linting/formatting)
add_python_ruff_targets("phy_ran_app_samples" ${CMAKE_CURRENT_SOURCE_DIR})

# PHY RAN App Executable
add_executable(phy_ran_app src/phy_ran_app.cpp src/phy_ran_app_utils.cpp)

# Add configured files include directory for cmake version info
target_include_directories(phy_ran_app PRIVATE ${CMAKE_BINARY_DIR}/configured_files/include)

target_link_libraries(
    phy_ran_app
    PRIVATE ran::phy_ran_utils # Brings: fapi, options, warnings (PUBLIC dependencies)
            ran::fronthaul # Fronthaul library (DOCA/DPDK networking)
            framework::task # TaskScheduler and TaskGraph
            framework::net # Networking configuration utilities
            framework::rt_log) # RT-safe logging

# System libraries
target_link_system_libraries(phy_ran_app PRIVATE CLI11::CLI11 tl::expected)

# GDRCopy for GPU Direct RDMA
target_link_gdrcopy(phy_ran_app)

# Installation
install(TARGETS phy_ran_app RUNTIME DESTINATION bin)

# Version and help tests (on the binary name)
add_version_and_help_tests(phy_ran_app)

# Function to add PHY RAN app integration test (modernized with runtime config)
function(add_phy_ran_app_integration_test APP_NAME)
    # Set up directories under CUBB_HOME (build directory)
    set(TEST_VECTORS_DIR "${CMAKE_BINARY_DIR}/testVectors")
    set(MULTI_CELL_DIR "${TEST_VECTORS_DIR}/multi-cell")
    set(TEST_MAC_CONFIG_DIR "${CMAKE_BINARY_DIR}/cuPHY-CP/testMAC/testMAC")
    set(NVLOG_CONFIG_DIR "${CMAKE_BINARY_DIR}/cuPHY/nvlog/config")
    set(RU_EMULATOR_CONFIG_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/ru-emulator/ru_emulator")
    set(RU_EMULATOR_WORKING_DIR "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/ru-emulator/ru_emulator")

    file(MAKE_DIRECTORY "${RU_EMULATOR_CONFIG_DIR}")

    # Template paths (passed to Python script for runtime generation)
    set(LAUNCH_PATTERN_TEMPLATE
        "${CMAKE_CURRENT_SOURCE_DIR}/config/launch_pattern_fapi_sample.yaml.in")
    set(TEST_MAC_CONFIG_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/config/test_mac_fapi_sample.yaml.in")
    set(RU_EMULATOR_CONFIG_TEMPLATE
        "${CMAKE_SOURCE_DIR}/ran/runtime/fronthaul/tools/config/ru_emulator_config.yaml.in")

    # Verify templates exist
    if(NOT EXISTS "${LAUNCH_PATTERN_TEMPLATE}")
        message(FATAL_ERROR "Launch pattern template not found: ${LAUNCH_PATTERN_TEMPLATE}")
    endif()
    if(NOT EXISTS "${TEST_MAC_CONFIG_TEMPLATE}")
        message(FATAL_ERROR "Test MAC config template not found: ${TEST_MAC_CONFIG_TEMPLATE}")
    endif()
    if(NOT EXISTS "${RU_EMULATOR_CONFIG_TEMPLATE}")
        message(FATAL_ERROR "RU emulator config template not found: ${RU_EMULATOR_CONFIG_TEMPLATE}")
    endif()

    # Test command - passes template paths for runtime generation
    set(TEST_COMMAND
        ${CMAKE_COMMAND} -E env CUBB_HOME=${CMAKE_BINARY_DIR} uv run
        "${CMAKE_CURRENT_SOURCE_DIR}/src/run_phy_ran_app_integration_test.py"
        "$<TARGET_FILE:test_mac>" "${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/testMAC/testMAC"
        "$<TARGET_FILE:ru_emulator>" "${RU_EMULATOR_WORKING_DIR}" "$<TARGET_FILE:${APP_NAME}>"
        "${LAUNCH_PATTERN_TEMPLATE}" "${TEST_MAC_CONFIG_TEMPLATE}" "${RU_EMULATOR_CONFIG_TEMPLATE}")

    # Add CTest
    add_test(NAME ${APP_NAME}.integration_test COMMAND ${TEST_COMMAND})

    # Set TRT engine path for tests (handles both ENABLE_MLIR_TRT=ON and OFF)
    test_set_trt_engine_path(
        ${APP_NAME}.integration_test "pusch_inner_receiver_free_energy_filter" FIXTURE_NAME
        "pusch_inner_receiver_trt_engines" ENGINE_DIR "ran/runtime/pusch/engine")

    # cmake-format: off
    # Runtime configuration:
    # - TEST_CELLS (default: 1) controls number of cells to test
    # - TEST_SLOTS (default: 100) controls test duration
    # - TEST_VECTOR (default: TVnr_7201_gNB_FAPI_s0.h5) controls test vector
    # - Python script generates YAML configs at runtime from templates
    # cmake-format: on
    label_test_requires_nic(${APP_NAME}.integration_test)
    label_integration_test(${APP_NAME}.integration_test)
    label_real_time(${APP_NAME}.integration_test)

    message(STATUS "PHY RAN App integration test setup complete")
    message(STATUS "  Launch pattern template: ${LAUNCH_PATTERN_TEMPLATE}")
    message(STATUS "  Test MAC config template: ${TEST_MAC_CONFIG_TEMPLATE}")
    message(STATUS "  RU config template: ${RU_EMULATOR_CONFIG_TEMPLATE}")
endfunction()

# Call the integration test setup
add_phy_ran_app_integration_test(phy_ran_app)

# Disable test when sanitizers are enabled (known data race issue)
if(ENABLE_SANITIZER_ADDRESS OR ENABLE_SANITIZER_THREAD)
    set_tests_properties(phy_ran_app.integration_test PROPERTIES DISABLED TRUE)
    message(STATUS "Disabled phy_ran_app.integration_test (sanitizers enabled)")
endif()

# Set capabilities (from both patterns)
set_cap_sys_nice(phy_ran_app)
set_dpdk_capabilities(phy_ran_app)
create_dpdk_capabilities_target(ru_emulator)
create_cap_sys_nice_target(test_mac)

# Add dependencies (from fapi_sample pattern)
add_dependencies(phy_ran_app test_mac test_mac_set_cap ru_emulator ru_emulator_set_cap)
