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

# Prevent multiple expensive DPDK detection and configuration operations
include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/CpuArchitecture.cmake")
find_package(PkgConfig REQUIRED)

# Set up DPDK
set(DPDK_ROOT
    "/opt/mellanox/dpdk"
    CACHE PATH "Path to DPDK installation directory")

set(DPDK_PATH ${DPDK_ROOT})
detect_cpu_architecture(DPDK_ARCH)
get_arch_lib_suffix(${DPDK_ARCH} DPDK_LIB_SUFFIX)
# Update PKG_CONFIG_PATH safely to avoid leading colon when undefined
if(DEFINED ENV{PKG_CONFIG_PATH})
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${DPDK_ROOT}/lib/${DPDK_LIB_SUFFIX}/pkgconfig")
else()
    set(ENV{PKG_CONFIG_PATH} "${DPDK_ROOT}/lib/${DPDK_LIB_SUFFIX}/pkgconfig")
endif()
pkg_search_module(DPDK REQUIRED libdpdk)

if(EXISTS "${DPDK_PATH}")
    message(
        STATUS
            "DPDK found:\n  - Includes: ${DPDK_INCLUDE_DIRS}\n  - Lib Dirs: ${DPDK_LIBRARY_DIRS}\n  - Libs: ${DPDK_LIBRARIES}"
    )
else()
    message(
        FATAL_ERROR
            "DPDK_PATH does not exist: ${DPDK_PATH} - DPDK functionality requires DPDK installation"
    )
endif()

# Links DPDK libraries to the specified target
#
# Arguments:
#   target     - Target to link DPDK libraries to
#   visibility - Required: PUBLIC, PRIVATE, or INTERFACE
function(target_link_dpdk target visibility)
    # Validate visibility argument
    if(NOT visibility MATCHES "^(PUBLIC|PRIVATE|INTERFACE)$")
        message(
            FATAL_ERROR
                "target_link_dpdk requires VISIBILITY argument (PUBLIC, PRIVATE, or INTERFACE). Usage: target_link_dpdk(target PUBLIC|PRIVATE|INTERFACE)"
        )
    endif()

    set(DPDK_VISIBILITY ${visibility})

    target_compile_definitions(${target} ${DPDK_VISIBILITY} ALLOW_EXPERIMENTAL_API)
    target_compile_options(${target} ${DPDK_VISIBILITY} ${DPDK_CFLAGS})
    target_include_directories(${target} SYSTEM ${DPDK_VISIBILITY} ${DPDK_INCLUDE_DIRS})
    target_link_directories(${target} ${DPDK_VISIBILITY} ${DPDK_LIBRARY_DIRS})
    target_link_directories(${target} ${DPDK_VISIBILITY} ${DPDK_LIBRARY_DIRS}/dpdk/pmds-23.0)
    # Pass linker flags via target_link_options, not as pseudo-library Disable --as-needed for DPDK
    # libraries to ensure they're linked even if not directly used
    target_link_options(${target} ${DPDK_VISIBILITY} -Wl,--no-as-needed)
    target_link_system_libraries(
        ${target}
        ${DPDK_VISIBILITY}
        ${DPDK_LIBRARIES}
        rte_net_mlx5
        rte_bus_pci
        rte_bus_vdev
        rte_common_mlx5
        rte_bus_auxiliary
        rte_gpu_cuda)
    # Re-enable --as-needed for subsequent libraries
    target_link_options(${target} ${DPDK_VISIBILITY} -Wl,--as-needed)
endfunction()
