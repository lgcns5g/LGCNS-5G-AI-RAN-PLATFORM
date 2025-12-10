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

include(GenerateExportHeader)
include(${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GDRCopy.cmake)

# Create oran_order_kernels library as OBJECT library OBJECT libraries work better with CUDA device
# code than STATIC libraries because device linking happens at the final executable level
add_library(oran_order_kernels OBJECT src/oran_order_kernels.cu)

# Configure CUDA support using target_link_cuda macro This handles: enable_language(CUDA), CUDA
# standard, PIC, separable compilation, and relaxed constexpr
target_link_cuda(oran_order_kernels)

# Link required libraries
target_link_libraries(oran_order_kernels PUBLIC framework::options framework::warnings
                                                framework::net framework::rt_log)

# OBJECT libraries should not resolve device symbols - that happens in the final executable
set_target_properties(oran_order_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Set include directories
target_include_directories(oran_order_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_system_libraries(oran_order_kernels PUBLIC comp_decomp_lib oran_h)

# Order Kernel Pipeline - ORAN UL Receiver
#
# Components: - OrderKernelModule: Wraps order_kernel_doca_single_subSlot_pingpong CUDA kernel -
# OrderKernelPipeline: Pipeline for ORAN packet reception and ordering
add_library(fronthaul-pipelines STATIC src/order_kernel_module.cpp src/order_kernel_pipeline.cpp
                                       src/order_kernel_factories.cpp)

target_include_directories(fronthaul-pipelines PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(
    fronthaul-pipelines PUBLIC framework::pipeline rt_log framework::net ran::oran
                               oran_order_kernels # Link CUDA kernels from oran_order_kernels.cu
)

target_link_cuda(fronthaul-pipelines)
target_link_gdrcopy(fronthaul-pipelines)

# Add alias for consistent naming
add_library(fronthaul::fronthaul_pipelines ALIAS fronthaul-pipelines)

add_library(fronthaul STATIC)
add_library(ran::fronthaul ALIAS fronthaul)

target_sources(
    fronthaul
    PRIVATE src/fronthaul.cpp src/fronthaul_parser.cpp src/uplane_network_config.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/fronthaul/fronthaul.hpp
           include/fronthaul/fronthaul_log.hpp
           include/fronthaul/fronthaul_parser.hpp
           include/fronthaul/uplane_config.hpp
           include/fronthaul/uplane_network_config.hpp)

target_link_libraries(
    fronthaul
    PUBLIC ran::oran ran::fapi framework::net framework::task fronthaul-pipelines
           framework::pipeline
    PRIVATE framework::options framework::warnings framework::rt_log)

target_link_system_libraries(fronthaul PUBLIC yaml gsl-lite::gsl-lite tl::expected CUDA::cudart)
target_link_gdrcopy(fronthaul)

target_include_directories(fronthaul PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

set_target_properties(
    fronthaul
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

install(
    TARGETS fronthaul
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/fronthaul
    DESTINATION include
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h")

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/fronthaul
    COMPONENT src)

generate_export_header(fronthaul EXPORT_FILE_NAME
                       ${CMAKE_CURRENT_BINARY_DIR}/include/fronthaul/fronthaul_export.hpp)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(fronthaul PUBLIC FRONTHAUL_STATIC_DEFINE)
endif()
