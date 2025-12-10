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

# Prevent multiple expensive DOCA detection and configuration operations
include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/CpuArchitecture.cmake")
find_package(PkgConfig REQUIRED)

# Set up DOCA packages
set(DOCA_ROOT
    "/opt/mellanox/doca"
    CACHE PATH "Path to DOCA installation directory")
set(DOCA_PATH ${DOCA_ROOT})
detect_cpu_architecture(DOCA_ARCH)
get_arch_lib_suffix(${DOCA_ARCH} DOCA_LIB_SUFFIX)
# Update PKG_CONFIG_PATH safely to avoid leading colon when undefined
if(DEFINED ENV{PKG_CONFIG_PATH})
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${DOCA_ROOT}/lib/${DOCA_LIB_SUFFIX}/pkgconfig")
else()
    set(ENV{PKG_CONFIG_PATH} "${DOCA_ROOT}/lib/${DOCA_LIB_SUFFIX}/pkgconfig")
endif()
pkg_search_module(DOCA REQUIRED doca-gpunetio)

if(EXISTS "${DOCA_PATH}")
    message(
        STATUS
            "DOCA found:\n  - Includes: ${DOCA_INCLUDE_DIRS}\n  - Lib Dirs: ${DOCA_LIBRARY_DIRS}\n  - Libs: ${DOCA_LIBRARIES}"
    )
else()
    message(
        FATAL_ERROR
            "DOCA_PATH does not exist: ${DOCA_PATH} - DOCA GPU functionality requires DOCA installation"
    )
endif()

# Links DOCA libraries to the specified target
#
# This function configures a target to use DOCA GPU libraries including:
#   - doca_gpunetio: DOCA GPU network I/O library
#   - doca_common: Common DOCA utilities
#   - doca_argp: DOCA argument parsing
#
# Arguments:
#   target     - Target to link DOCA libraries to
#   visibility - Required: PUBLIC, PRIVATE, or INTERFACE
function(target_link_doca target visibility)
    # Validate visibility argument
    if(NOT visibility MATCHES "^(PUBLIC|PRIVATE|INTERFACE)$")
        message(
            FATAL_ERROR
                "target_link_doca requires VISIBILITY argument (PUBLIC, PRIVATE, or INTERFACE). Usage: target_link_doca(target PUBLIC|PRIVATE|INTERFACE)"
        )
    endif()

    set(DOCA_VISIBILITY ${visibility})

    # Add DOCA compile definitions
    target_compile_definitions(${target} ${DOCA_VISIBILITY} ALLOW_EXPERIMENTAL_API)
    target_compile_definitions(${target} ${DOCA_VISIBILITY} DOCA_ALLOW_EXPERIMENTAL_API)

    # Add DOCA include directories
    target_include_directories(${target} SYSTEM ${DOCA_VISIBILITY} ${DOCA_INCLUDE_DIRS})

    # Add DOCA library directories
    target_link_directories(${target} ${DOCA_VISIBILITY} ${DOCA_LIBRARY_DIRS})

    # Link DOCA libraries
    target_link_system_libraries(
        ${target}
        ${DOCA_VISIBILITY}
        doca_gpunetio
        doca_common
        doca_argp
        doca_eth
        doca_dpdk_bridge
        doca_flow
        doca_rdma)

    # Add special DOCA GPU device library with whole-archive linking
    pkg_get_variable(DOCA_GPUNETIO_LIBDIR doca-gpunetio libdir)
    target_link_directories(${target} ${DOCA_VISIBILITY} ${DOCA_GPUNETIO_LIBDIR})
    target_link_system_libraries(${target} ${DOCA_VISIBILITY}
                                 "$<LINK_LIBRARY:WHOLE_ARCHIVE,doca_gpunetio_device>")

    # Set CUDA properties for DOCA GPU targets
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()
