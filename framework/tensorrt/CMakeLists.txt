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

# Framework TensorRT Library
#
# TensorRT integration providing engine interfaces and MLIR support

# Find TensorRT libraries
find_package(TensorRT QUIET)
if(NOT TensorRT_FOUND)
    find_library(
        NVINFER
        NAMES nvinfer libnvinfer
        PATHS /usr/local/tensorrt/lib /opt/tensorrt/lib
        PATH_SUFFIXES lib lib64)
    if(NOT NVINFER)
        message(FATAL_ERROR "TensorRT library not found")
    endif()
else()
    set(NVINFER ${TensorRT_LIBRARIES})
endif()

add_library(framework-tensorrt STATIC)

target_sources(
    framework-tensorrt
    PRIVATE src/trt_engine_logger.cpp src/trt_engine.cpp src/trt_pre_post_enqueue_stream_cap.cpp
            src/mlir_trt_engine.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/tensorrt/trt_engine_logger.hpp
           include/tensorrt/trt_engine_params.hpp
           include/tensorrt/trt_engine_interface.hpp
           include/tensorrt/trt_engine_interfaces.hpp
           include/tensorrt/trt_engine.hpp
           include/tensorrt/trt_null_pre_post_enqueue.hpp
           include/tensorrt/trt_pre_post_enqueue_stream_cap.hpp
           include/tensorrt/mlir_trt_engine.hpp)

# Create the namespaced alias
add_library(framework::tensorrt ALIAS framework-tensorrt)

target_include_directories(
    framework-tensorrt PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:include>)

target_link_cuda(framework-tensorrt)
target_link_libraries(framework-tensorrt PUBLIC framework::utils framework::tensor
                                                framework::options framework::warnings)
target_link_system_libraries(framework-tensorrt PUBLIC ${NVINFER} gsl-lite)

set_target_properties(framework-tensorrt PROPERTIES LINKER_LANGUAGE CXX)

# Install targets
install(
    TARGETS framework-tensorrt
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# Install headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt
    DESTINATION include
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.cuh")

# Install sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/tensorrt
    COMPONENT src
    PATTERN "CMakeLists.txt" EXCLUDE)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
