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

# Prevent multiple expensive MathDx detection and configuration operations
include_guard(GLOBAL)

include(${CMAKE_SOURCE_DIR}/cmake/CpuArchitecture.cmake)

# CMake 3.27+: Suppress CMP0144 warning about uppercase MATHDX_ROOT The warning is harmless
# (NO_DEFAULT_PATH disables automatic ROOT search), but we set policy NEW to acknowledge intentional
# use of uppercase env var.  Note the policy scope is limited to this file only.
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

# Set up MathDx package root User can override via -Dmathdx_ROOT=<path> or environment variable
if(NOT DEFINED MATHDX_ROOT AND DEFINED ENV{MATHDX_ROOT})
    set(MATHDX_ROOT
        "$ENV{MATHDX_ROOT}"
        CACHE PATH "Path to MathDx installation directory")
elseif(NOT DEFINED MATHDX_ROOT)
    set(MATHDX_ROOT
        "/opt/nvidia/mathdx"
        CACHE PATH "Path to MathDx installation directory")
endif()

# Find MathDx via CMake CONFIG mode (REQUIRED) MathDx provides mathdx-config.cmake with multiple
# components
find_package(
    mathdx
    COMPONENTS cufftdx cublasdx cusolverdx curanddx CONFIG
    REQUIRED PATHS ${MATHDX_ROOT} NO_DEFAULT_PATH)

message(STATUS "MathDx found: ${MATHDX_ROOT}")
message(STATUS "  - cuFFTDx available via mathdx::cufftdx target (use target_link_mathdx())")
message(STATUS "  - cuBLASDx available via mathdx::cublasdx target (use target_link_mathdx())")
message(STATUS "  - cuRANDDx available via mathdx::curanddx target (use target_link_mathdx())")
message(
    STATUS "  - cuSOLVERDx available via mathdx::cusolverdx target (use target_link_cusolverdx())")

# Links MathDx libraries to the specified target
#
# This function configures a target to use MathDx libraries for various operations on CUDA GPUs.
# You must explicitly specify which components to link.
#
# Available components:
#   - cufftdx: FFT operations (header-only)
#   - cublasdx: BLAS operations (header-only)
#   - curanddx: Random number generation (header-only)
#
# Note: cuSOLVERDx has a static library requiring LTO flags and must be linked via
# target_link_cusolverdx() instead.
#
# Arguments:
#   target     - Target to link MathDx libraries to
#   visibility - Required: PUBLIC, PRIVATE, or INTERFACE
#   components - List of components to link (e.g., cufftdx cublasdx)
#
# Example usage:
#   add_executable(my_app main.cu)
#   target_link_mathdx(my_app PRIVATE cufftdx cublasdx)
#
function(target_link_mathdx target visibility)
    # Validate visibility argument
    if(NOT visibility MATCHES "^(PUBLIC|PRIVATE|INTERFACE)$")
        message(
            FATAL_ERROR
                "target_link_mathdx requires VISIBILITY argument (PUBLIC, PRIVATE, or INTERFACE). Usage: target_link_mathdx(target PUBLIC|PRIVATE|INTERFACE component1 component2 ...)"
        )
    endif()

    # Get components from remaining arguments
    set(components ${ARGN})

    # Validate that components were provided
    if(NOT components)
        message(
            FATAL_ERROR
                "target_link_mathdx requires at least one component. Usage: target_link_mathdx(target ${visibility} cufftdx cublasdx ...)"
        )
    endif()

    # Build list of component targets to link
    set(component_targets)
    foreach(component IN LISTS components)
        # Validate component name
        if(NOT component MATCHES "^(cufftdx|cublasdx|curanddx)$")
            message(
                FATAL_ERROR
                    "Invalid MathDx component '${component}'. Valid components: cufftdx, cublasdx, curanddx. Note: Use target_link_cusolverdx() for cuSOLVERDx."
            )
        endif()
        list(APPEND component_targets mathdx::${component})
    endforeach()

    # Link MathDx component targets
    target_link_system_libraries(${target} ${visibility} ${component_targets})

    # MathDx components require CUDA separable compilation for device code
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

# Links cuSOLVERDx library to the specified target
#
# This function configures a target to use cuSOLVERDx for solver operations on CUDA GPUs. cuSOLVERDx
# provides QR decomposition and other linear algebra solver operations.
#
# Note: cuSOLVERDx has a static library containing LTO objects. This function automatically adds the
# required -dlto flag for CUDA device linking.
#
# Arguments: target     - Target to link cuSOLVERDx library to visibility - Required: PUBLIC,
# PRIVATE, or INTERFACE
#
# Example usage: add_executable(my_solver_app main.cu) target_link_cusolverdx(my_solver_app PRIVATE)
#
function(target_link_cusolverdx target visibility)
    # Validate visibility argument
    if(NOT visibility MATCHES "^(PUBLIC|PRIVATE|INTERFACE)$")
        message(
            FATAL_ERROR
                "target_link_cusolverdx requires VISIBILITY argument (PUBLIC, PRIVATE, or INTERFACE). Usage: target_link_cusolverdx(target PUBLIC|PRIVATE|INTERFACE)"
        )
    endif()

    detect_cpu_architecture(TARGET_ARCH)

    # Static library (.a) is only available for x86_64; link it for host + device code
    if(TARGET_ARCH STREQUAL "x86_64")
        target_link_system_libraries(${target} ${visibility} mathdx::cusolverdx)
    else()
        # On other architectures: use fatbin (static library not available)
        if(NOT TARGET mathdx::cusolverdx_fatbin)
            message(
                FATAL_ERROR
                    "${TARGET_ARCH} requires cuSOLVERDx fatbin library (CUDA >= 12.8), but mathdx::cusolverdx_fatbin target not found"
            )
        endif()
        target_link_system_libraries(${target} ${visibility} mathdx::cusolverdx_fatbin)
    endif()

    # cmake-format: off
    # Handle LTO flags based on CMAKE_CUDA_ARCHITECTURES
    # When explicit architectures are set (e.g., 90-real), we need to add lto_<arch> codes
    # in addition to sm_<arch> codes to avoid -dlto/-gencode conflicts
    set(use_explicit_lto_codes FALSE)
    if(CMAKE_CUDA_ARCHITECTURES)
        string(FIND "${CMAKE_CUDA_ARCHITECTURES}" "native" native_pos)
        if(native_pos EQUAL -1)
            set(use_explicit_lto_codes TRUE)
        endif()
    endif()

    if(use_explicit_lto_codes)
        # Explicit architectures - add supplementary gencode flags with LTO codes
        # CMake will generate sm_<arch> from CUDA_ARCHITECTURES, we add lto_<arch>
        foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            # Strip -real/-virtual suffix to get base architecture number
            string(REGEX REPLACE "[-].*$" "" base_arch "${arch}")
            # Add gencode flag with LTO code (CMake already adds SM code from CUDA_ARCHITECTURES)
            target_compile_options(${target} PRIVATE 
                $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${base_arch},code=lto_${base_arch}>)
        endforeach()
    else()
        # Using "native" or no explicit architectures - standalone -dlto flag works without conflicts
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dlto>)
    endif()
    # cmake-format: on

    # Add -dlto flag to device link stage (required by nvlink for LTO objects)
    target_link_options(${target} PRIVATE $<DEVICE_LINK:-dlto>)

    # cuSOLVERDx requires CUDA separable compilation for device code
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()
