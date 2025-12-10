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

include(cmake/CPM.cmake)
include(cmake/CpuArchitecture.cmake)
include(cmake/Patches.cmake)

# ACAR configuration options
set(ACAR_REPO
    "https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git"
    CACHE STRING "Git repository URL for aerial_sdk")
set(ACAR_SOURCE_DIR
    ""
    CACHE PATH "Directory to download/find aerial_sdk source code (if empty, CPM will manage it)")

# Configure aerial_sdk before adding as subdirectory
function(configure_acar_subdirectory source_dir)
    # Detect target architecture to select appropriate toolchain
    detect_cpu_architecture(TARGET_ARCH)

    # Map architecture to toolchain file
    if(TARGET_ARCH STREQUAL "x86_64")
        set(ACAR_TOOLCHAIN_FILE "${source_dir}/cuPHY/cmake/toolchains/x86-64")
    elseif(TARGET_ARCH STREQUAL "aarch64")
        set(ACAR_TOOLCHAIN_FILE "${source_dir}/cuPHY/cmake/toolchains/grace-cross")
    else()
        message(FATAL_ERROR "Unsupported architecture for aerial_sdk: ${TARGET_ARCH}. "
                            "Supported architectures: x86_64, aarch64")
    endif()

    # Set toolchain file if not already configured
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "")
        if(EXISTS "${ACAR_TOOLCHAIN_FILE}")
            message(STATUS "Setting CMAKE_TOOLCHAIN_FILE: ${ACAR_TOOLCHAIN_FILE}")
            set(CMAKE_TOOLCHAIN_FILE
                "${ACAR_TOOLCHAIN_FILE}"
                CACHE FILEPATH "Toolchain file" FORCE)
        else()
            message(WARNING "Toolchain file not found at: ${ACAR_TOOLCHAIN_FILE}")
        endif()
    else()
        message(STATUS "CMAKE_TOOLCHAIN_FILE already set: ${CMAKE_TOOLCHAIN_FILE}")
    endif()

    # Configure aerial_sdk to build minimal components
    set(ENABLE_TESTS
        OFF
        PARENT_SCOPE)
    set(ENABLE_YANG_PARSER
        OFF
        PARENT_SCOPE)
    set(AERIAL_METRICS
        OFF
        PARENT_SCOPE)
    set(ENABLE_SCTP_CHECK
        OFF
        PARENT_SCOPE)
    set(ENABLE_DOCA_GPU_COMM
        OFF
        PARENT_SCOPE)
    set(ENABLE_CUMAC
        OFF
        PARENT_SCOPE)
    set(ENABLE_PYAERIAL
        OFF
        PARENT_SCOPE)
endfunction()

# cmake-format: off
# Setup aerial_sdk in two steps to handle dependencies correctly:
# 1. download_acar_source() - Downloads aerial_sdk source, making aerial_sdk_SOURCE_DIR available
#    for patch paths and allowing other dependencies (fmtlog) to be configured first
# 2. setup_acar_targets() - Configures and adds aerial_sdk as subdirectory after its dependencies
#    are available (e.g., fmtlog is needed by nvlog)
# cmake-format: on

# Download ACAR source (Step 1: make aerial_sdk_SOURCE_DIR available)
function(download_acar_source)
    # Check if we need to reconfigure due to ACAR_SOURCE_DIR change
    if(DEFINED aerial_sdk_SOURCE_DIR AND DEFINED _ACAR_SOURCE_DIR_LAST)
        if(_ACAR_SOURCE_DIR_LAST STREQUAL ACAR_SOURCE_DIR)
            return() # Already configured with same ACAR_SOURCE_DIR
        else()
            message(
                STATUS
                    "ACAR_SOURCE_DIR changed from '${_ACAR_SOURCE_DIR_LAST}' to '${ACAR_SOURCE_DIR}', reconfiguring..."
            )
            unset(aerial_sdk_SOURCE_DIR CACHE)
        endif()
    endif()

    # Track ACAR_SOURCE_DIR for change detection
    set(_ACAR_SOURCE_DIR_LAST
        "${ACAR_SOURCE_DIR}"
        CACHE INTERNAL "Last value of ACAR_SOURCE_DIR used")

    if(SKIP_ACAR_DOWNLOAD)
        # Use existing source
        if(ACAR_SOURCE_DIR STREQUAL "")
            message(FATAL_ERROR "SKIP_ACAR_DOWNLOAD=ON requires ACAR_SOURCE_DIR to be specified")
        endif()
        if(NOT EXISTS "${ACAR_SOURCE_DIR}/CMakeLists.txt")
            message(
                FATAL_ERROR
                    "ACAR_SOURCE_DIR '${ACAR_SOURCE_DIR}' does not contain aerial_sdk source")
        endif()
        message(STATUS "Using existing aerial_sdk at: ${ACAR_SOURCE_DIR}")
        # Propagate aerial_sdk_SOURCE_DIR to cache for global access
        set(aerial_sdk_SOURCE_DIR
            ${ACAR_SOURCE_DIR}
            CACHE INTERNAL "Path to aerial_sdk source directory")
    else()
        # Download aerial_sdk
        if(NOT ACAR_SOURCE_DIR STREQUAL "")
            message(STATUS "Downloading aerial_sdk to: ${ACAR_SOURCE_DIR}")
            set(SOURCE_DIR_ARG SOURCE_DIR ${ACAR_SOURCE_DIR})
        else()
            message(
                STATUS
                    "Downloading aerial_sdk to CPM location: ${CMAKE_BINARY_DIR}/_deps/aerial_sdk-src"
            )
            set(SOURCE_DIR_ARG "")
        endif()

        cpmaddpackage(
            NAME
            aerial_sdk
            GIT_REPOSITORY
            ${ACAR_REPO}
            GIT_TAG
            25.3.0
            GIT_SHALLOW
            YES
            GIT_SUBMODULES_RECURSE
            YES
            DOWNLOAD_ONLY
            YES
            SYSTEM
            YES
            ${SOURCE_DIR_ARG})

        # cmake-format: off
        # Apply patch, skipping if already applied
        # if(aerial_sdk_ADDED)
        #    apply_patch_once(${CMAKE_SOURCE_DIR}/cmake/patches/aerial_sdk.patch
        #                     ${aerial_sdk_SOURCE_DIR})
        # endif()
        # cmake-format: on

        # Propagate aerial_sdk_SOURCE_DIR to cache for global access
        set(aerial_sdk_SOURCE_DIR
            ${aerial_sdk_SOURCE_DIR}
            CACHE INTERNAL "Path to aerial_sdk source directory")
    endif()
endfunction()

# Setup ACAR targets (Step 2: configure and add subdirectory)
function(setup_acar_targets)
    if(TARGET nvlog)
        return() # Already set up
    endif()

    if(NOT DEFINED aerial_sdk_SOURCE_DIR)
        message(FATAL_ERROR "download_acar_source() must be called before setup_acar_targets()")
    endif()

    configure_acar_subdirectory(${aerial_sdk_SOURCE_DIR})

    # cmake-format: off
    # CMake 3.27+: Suppress CMP0144 warnings in aerial_sdk subdirectory
    # aerial_sdk uses uppercase MATHDX_ROOT env var, which triggers CMP0144 warnings.
    # aerial_sdk's CMakeLists.txt calls project(), which creates a new policy scope.
    # cmake_policy(SET ...) doesn't propagate through project() boundaries, but
    # CMAKE_POLICY_DEFAULT_CMP0144 does - it tells the new scope what default to use.
    # cmake-format: on
    if(POLICY CMP0144)
        set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
    endif()

    add_subdirectory(${aerial_sdk_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/aerial_sdk
                     EXCLUDE_FROM_ALL)

    # Propagate aerial_sdk_BINARY_DIR to cache for global access
    set(aerial_sdk_BINARY_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/aerial_sdk
        CACHE INTERNAL "Path to aerial_sdk binary directory")
endfunction()

# Download and install LDPC cubin files.
function(download_ldpc_decoder_cubin)
    # Skip download if using user-provided ACAR_SOURCE_DIR
    if(NOT ACAR_SOURCE_DIR STREQUAL "")
        message(
            STATUS
                "LDPC decoder cubin: Skipping download, using user-provided ACAR_SOURCE_DIR: ${ACAR_SOURCE_DIR}"
        )
        return()
    endif()

    if(NOT DEFINED aerial_sdk_SOURCE_DIR)
        message(
            FATAL_ERROR "download_acar_source() must be called before download_ldpc_decoder_cubin()"
        )
    endif()

    set(LDPC_DECODER_CUBIN_VERSION
        "2f9dcb"
        CACHE STRING "LDPC decoder cubin version")
    # Install to aerial_sdk source tree
    set(CUBIN_INSTALL_DIR "${aerial_sdk_SOURCE_DIR}/cuPHY/src/cuphy/error_correction")

    set(NGC_API_KEY "$ENV{NGC_API_KEY}")
    # Assume that if NGC_API_KEY is set, we are downloading from nvstaging, otherwise from public
    # NGC
    if(NGC_API_KEY STREQUAL "")
        set(NGC_ORG_TEAM "nvidia/team/aerial")
    else()
        set(NGC_ORG_TEAM "nvstaging/team/aerial")
    endif()

    # Construct NGC download URL
    set(CUBIN_DOWNLOAD_URL
        "https://api.ngc.nvidia.com/v2/resources/org/${NGC_ORG_TEAM}/ldpc-decoder-cubin/${LDPC_DECODER_CUBIN_VERSION}/files?redirect=true&path=ldpc_decoder_cubin.zip"
    )

    message(STATUS "LDPC decoder cubin: Downloading from: ${CUBIN_DOWNLOAD_URL}")
    message(STATUS "LDPC decoder cubin: Installing to: ${CUBIN_INSTALL_DIR}")

    # Build curl command
    set(CUBIN_ZIP_FILE "${CMAKE_BINARY_DIR}/ldpc_decoder_cubin.zip")
    set(CURL_CMD curl --fail -H "Content-Type: application/json")
    if(NOT NGC_API_KEY STREQUAL "")
        list(APPEND CURL_CMD -H "Authorization: Bearer ${NGC_API_KEY}")
    endif()
    list(APPEND CURL_CMD -L -o ${CUBIN_ZIP_FILE} ${CUBIN_DOWNLOAD_URL})

    # Download zip file using curl
    execute_process(
        COMMAND ${CURL_CMD}
        RESULT_VARIABLE DOWNLOAD_RESULT
        ERROR_VARIABLE DOWNLOAD_ERROR
        OUTPUT_QUIET)

    if(NOT DOWNLOAD_RESULT EQUAL 0)
        message(
            WARNING
                "Failed to download LDPC decoder cubins. Using fallback LDPC decoder implementation."
        )
        return()
    endif()

    # Remove existing install dir (symlink or directory)
    file(REMOVE_RECURSE "${CUBIN_INSTALL_DIR}/ldpc_decoder_cubin")

    # Extract archive to installation directory
    file(ARCHIVE_EXTRACT INPUT ${CUBIN_ZIP_FILE} DESTINATION ${CUBIN_INSTALL_DIR})
    if(NOT EXISTS "${CUBIN_INSTALL_DIR}/ldpc_decoder_cubin")
        message(WARNING "Failed to extract LDPC decoder cubins")
        return()
    endif()

    # Cleanup downloaded zip file
    file(REMOVE "${CUBIN_ZIP_FILE}")
endfunction()
