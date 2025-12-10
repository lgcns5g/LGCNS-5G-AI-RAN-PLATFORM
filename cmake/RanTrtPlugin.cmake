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

include_guard(GLOBAL)

include("${CMAKE_SOURCE_DIR}/cmake/CpuArchitecture.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Nvinfer.cmake")

# Internal helper to build list of available files for error messages
function(_build_available_files_list OUT_LIST DIRECTORY PATTERN)
    file(GLOB available_files "${DIRECTORY}/${PATTERN}")
    set(file_list "")
    foreach(file IN LISTS available_files)
        get_filename_component(file_name "${file}" NAME)
        set(file_list "${file_list}\n  - ${file_name}")
    endforeach()
    set(${OUT_LIST}
        "${file_list}"
        PARENT_SCOPE)
endfunction()

# Internal helper to build and report architecture mismatch error for engines
function(_report_arch_error CPU_ARCH GPU_ARCH LOOKING_FOR DIRECTORY)
    _build_available_files_list(available_list "${DIRECTORY}" "*.trtengine")

    set(error_msg "TensorRT files not found for detected architecture.\n")
    string(APPEND error_msg "Detected: CPU=${CPU_ARCH}, GPU=${GPU_ARCH}\n")
    string(APPEND error_msg "Looking for: ${LOOKING_FOR}\n")
    string(APPEND error_msg "In directory: ${DIRECTORY}\n")
    string(APPEND error_msg "Available files:${available_list}")
    message(FATAL_ERROR "${error_msg}")
endfunction()

# Internal helper to report plugin and engine missing error
function(_report_plugin_engine_error CPU_ARCH GPU_ARCH ENGINE_FILE PLUGIN_FILE DIRECTORY)
    _build_available_files_list(engine_list "${DIRECTORY}" "*.trtengine")
    _build_available_files_list(plugin_list "${DIRECTORY}" "*.so")

    set(error_msg "TensorRT files not found for detected architecture.\n")
    string(APPEND error_msg "Detected: CPU=${CPU_ARCH}, GPU=${GPU_ARCH}\n")
    string(APPEND error_msg "Looking for: ${ENGINE_FILE} and ${PLUGIN_FILE}\n")
    string(APPEND error_msg "In directory: ${DIRECTORY}\n")
    string(APPEND error_msg "Available TRT engine files:${engine_list}\n")
    string(APPEND error_msg "Available TRT plugin files:${plugin_list}")
    message(FATAL_ERROR "${error_msg}")
endfunction()

# Internal function to setup pre-checked-in plugin (called lazily when needed)
function(_setup_prechecked_trt_plugin)
    # Guard to ensure setup only runs once
    if(TARGET trt_plugin_lib)
        return()
    endif()

    detect_cpu_architecture(CPU_ARCH)
    resolve_gpu_architecture(GPU_ARCH)

    # Select architecture-specific TRT files
    set(TRT_ENGINE_FILE "pusch_inner_receiver_free_energy_filter.${CPU_ARCH}.${GPU_ARCH}.trtengine")
    set(TRT_PLUGIN_FILE "libran_trt_plugin.${CPU_ARCH}.${GPU_ARCH}.so")
    set(TRT_ENGINE_DIR "${CMAKE_SOURCE_DIR}/ran/runtime/pusch/engine")

    # Set TRT engine and plugin paths
    set(TRT_ENGINE_PATH "${TRT_ENGINE_DIR}/${TRT_ENGINE_FILE}")
    set(TRT_PLUGIN_PATH "${TRT_ENGINE_DIR}/${TRT_PLUGIN_FILE}")

    if(NOT EXISTS "${TRT_ENGINE_PATH}" OR NOT EXISTS "${TRT_PLUGIN_PATH}")
        _report_plugin_engine_error("${CPU_ARCH}" "${GPU_ARCH}" "${TRT_ENGINE_FILE}"
                                    "${TRT_PLUGIN_FILE}" "${TRT_ENGINE_DIR}")
    endif()

    # Copy arch-specific plugin to build directory with generic name
    set(TRT_PLUGIN_BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}/libran_trt_plugin.so")
    file(COPY "${TRT_PLUGIN_PATH}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/${TRT_PLUGIN_FILE}" "${TRT_PLUGIN_BUILD_PATH}")

    # Create imported library target pointing to copied plugin (GLOBAL scope)
    add_library(trt_plugin_lib SHARED IMPORTED GLOBAL)
    set_target_properties(trt_plugin_lib PROPERTIES IMPORTED_LOCATION "${TRT_PLUGIN_BUILD_PATH}")

    message(STATUS "Using TRT engine: ${TRT_ENGINE_FILE}")
    message(STATUS "Using TRT plugin: ${TRT_PLUGIN_FILE}")
    message(STATUS "TRT plugin copied to: ${TRT_PLUGIN_BUILD_PATH}")
endfunction()

# Function to link RAN TRT plugin to a target (handles both MLIR_TRT modes)
function(target_link_ran_trt_plugin TARGET_NAME VISIBILITY)
    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR "target_link_ran_trt_plugin: Target '${TARGET_NAME}' does not exist")
    endif()

    if(NOT VISIBILITY MATCHES "^(PRIVATE|PUBLIC|INTERFACE)$")
        message(
            FATAL_ERROR
                "target_link_ran_trt_plugin: VISIBILITY must be PRIVATE, PUBLIC, or INTERFACE")
    endif()

    if(ENABLE_MLIR_TRT)
        # Link against build-time plugin
        target_link_libraries(${TARGET_NAME} ${VISIBILITY} ran_trt_plugin)
        message(STATUS "Linked ${TARGET_NAME} to build-time ran_trt_plugin (${VISIBILITY})")
    else()
        # Setup pre-checked-in plugin (only runs once)
        _setup_prechecked_trt_plugin()

        # Link against pre-checked-in plugin
        target_link_libraries(${TARGET_NAME} ${VISIBILITY} trt_plugin_lib)

        # Set RPATH so the executable can find the plugin at runtime
        set_target_properties(${TARGET_NAME} PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
                                                        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib")

        message(STATUS "Linked ${TARGET_NAME} to pre-checked-in trt_plugin_lib (${VISIBILITY})")
    endif()
endfunction()

# Function to set TRT engine path for tests (handles both MLIR_TRT modes)
function(test_set_trt_engine_path TEST_NAME ENGINE_NAME)
    cmake_parse_arguments(ARG "" "FIXTURE_NAME;ENGINE_DIR" "" ${ARGN})

    # Unified engine location for both modes
    set(TRT_ENGINE_BUILD_DIR "${CMAKE_BINARY_DIR}/ran/py/trt_engines")
    set(ENGINE_BUILD_PATH "${TRT_ENGINE_BUILD_DIR}/${ENGINE_NAME}.trtengine")

    if(ENABLE_MLIR_TRT)
        # Python tests generate engine at unified location
        if(ARG_FIXTURE_NAME)
            set(FIXTURE_REQ "${ARG_FIXTURE_NAME}")
        else()
            message(WARNING "${TEST_NAME}: MLIR_TRT=ON but no FIXTURE_NAME provided")
            set(FIXTURE_REQ "")
        endif()

        message(STATUS "${TEST_NAME}: Using generated engine at ${ENGINE_BUILD_PATH}")
    else()
        # Copy pre-checked-in engine to unified location (only if not already there)
        if(NOT EXISTS "${ENGINE_BUILD_PATH}")
            if(NOT ARG_ENGINE_DIR)
                message(
                    FATAL_ERROR
                        "test_set_trt_engine_path: ENGINE_DIR is required when ENABLE_MLIR_TRT=OFF")
            endif()

            detect_cpu_architecture(CPU_ARCH)
            resolve_gpu_architecture(GPU_ARCH)

            # Build arch-specific engine filename
            set(ENGINE_FILE "${ENGINE_NAME}.${CPU_ARCH}.${GPU_ARCH}.trtengine")

            # Determine source directory
            set(ENGINE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/${ARG_ENGINE_DIR}")
            set(ENGINE_SOURCE_PATH "${ENGINE_SOURCE_DIR}/${ENGINE_FILE}")

            # Verify source engine exists
            if(NOT EXISTS "${ENGINE_SOURCE_PATH}")
                _report_arch_error("${CPU_ARCH}" "${GPU_ARCH}" "${ENGINE_FILE}"
                                   "${ENGINE_SOURCE_DIR}")
            endif()

            # Copy engine to unified location
            file(MAKE_DIRECTORY "${TRT_ENGINE_BUILD_DIR}")
            file(COPY "${ENGINE_SOURCE_PATH}" DESTINATION "${TRT_ENGINE_BUILD_DIR}")
            file(RENAME "${TRT_ENGINE_BUILD_DIR}/${ENGINE_FILE}" "${ENGINE_BUILD_PATH}")

            message(STATUS "${TEST_NAME}: Copied ${ENGINE_FILE} -> ${ENGINE_BUILD_PATH}")
        else()
            message(STATUS "${TEST_NAME}: Using cached engine at ${ENGINE_BUILD_PATH}")
        endif()

        set(FIXTURE_REQ "")
    endif()

    # Set test properties (identical for both modes)
    set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT
                                                 "RAN_TRT_ENGINE_PATH=${ENGINE_BUILD_PATH}")

    # Append to existing fixtures if provided
    if(FIXTURE_REQ)
        get_test_property(${TEST_NAME} FIXTURES_REQUIRED _existing_fixtures)
        if(_existing_fixtures)
            set_tests_properties(${TEST_NAME} PROPERTIES FIXTURES_REQUIRED
                                                         "${_existing_fixtures};${FIXTURE_REQ}")
        else()
            set_tests_properties(${TEST_NAME} PROPERTIES FIXTURES_REQUIRED "${FIXTURE_REQ}")
        endif()
    endif()
endfunction()
