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

# Prevent multiple expensive TensorRT detection and configuration operations
include_guard(GLOBAL)

# Detect target architecture and find TensorRT libraries in architecture-specific paths This runs
# once when the file is included due to include_guard(GLOBAL)
#
# Creates the following imported target: - nvinfer::nvinfer: TensorRT inference library target
#
# Architecture-specific library paths: - x86_64: /usr/lib/x86_64-linux-gnu - aarch64:
# /usr/lib/aarch64-linux-gnu

detect_cpu_architecture(TARGET_ARCH)
get_arch_lib_suffix(${TARGET_ARCH} ARCH_LIB_SUFFIX)

# Find TensorRT library in architecture-specific path
find_library(
    NVINFER_LIBRARY
    NAMES nvinfer
    PATHS /usr/lib/${ARCH_LIB_SUFFIX} REQUIRED)

# Find TensorRT headers
find_path(
    NVINFER_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS /usr/local/cuda/include /usr/include REQUIRED)

# Create imported target for nvinfer
if(NOT TARGET nvinfer::nvinfer)
    add_library(nvinfer::nvinfer SHARED IMPORTED)
    set_target_properties(
        nvinfer::nvinfer PROPERTIES IMPORTED_LOCATION "${NVINFER_LIBRARY}"
                                    INTERFACE_INCLUDE_DIRECTORIES "${NVINFER_INCLUDE_DIR}")
endif()

message(STATUS "TensorRT (nvinfer) found: ${NVINFER_LIBRARY}")
message(STATUS "  - nvinfer available via nvinfer::nvinfer target (use target_link_nvinfer())")

# Links TensorRT (nvinfer) library to the specified target
#
# This function configures a target to use TensorRT for inference operations. It automatically:
#   - Links the nvinfer::nvinfer imported target
#   - Includes TensorRT headers (from the imported target's INTERFACE_INCLUDE_DIRECTORIES)
#   - Configures CUDA compilation settings via target_link_cuda
#
# Arguments:
#   target     - Target to link TensorRT library to
#   visibility - Required: PUBLIC, PRIVATE, or INTERFACE
#
# Example usage:
#   add_library(my_trt_plugin SHARED plugin.cpp plugin.cu)
#   target_link_nvinfer(my_trt_plugin PUBLIC)
#
function(target_link_nvinfer target visibility)
    # Validate visibility argument
    if(NOT visibility MATCHES "^(PUBLIC|PRIVATE|INTERFACE)$")
        message(
            FATAL_ERROR
                "target_link_nvinfer requires VISIBILITY argument (PUBLIC, PRIVATE, or INTERFACE). Usage: target_link_nvinfer(target PUBLIC|PRIVATE|INTERFACE)"
        )
    endif()

    # Configure CUDA compilation settings
    target_link_cuda(${target})

    # Link nvinfer target (includes headers via INTERFACE_INCLUDE_DIRECTORIES)
    target_link_system_libraries(${target} ${visibility} nvinfer::nvinfer)
endfunction()
