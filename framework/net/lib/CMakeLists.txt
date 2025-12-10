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
include(${CMAKE_SOURCE_DIR}/cmake/Doca.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Dpdk.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GDRCopy.cmake)

add_library(net STATIC)
add_library(framework::net ALIAS net)

target_sources(
    net
    PRIVATE src/doca_utils.cpp
            src/dpdk_utils.cpp
            src/dpdk_txq.cpp
            src/env.cpp
            src/gpu.cpp
            src/nic.cpp
            src/doca_rxq.cpp
            src/doca_txq.cpp
            src/mempool.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/net/details/doca_utils.hpp
           include/net/details/dpdk_utils.hpp
           include/net/doca_types.hpp
           include/net/dpdk_types.hpp
           include/net/dpdk_txq.hpp
           include/net/env.hpp
           include/net/gpu.hpp
           include/net/mempool.hpp
           include/net/nic.hpp
           include/net/doca_rxq.hpp
           include/net/doca_txq.hpp)

install(
    TARGETS net
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/net
    DESTINATION include
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "details" EXCLUDE)

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/net
    COMPONENT src)

target_include_directories(
    net
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include/net>)

set_target_properties(
    net
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(net EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/net/net_export.hpp)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(net PUBLIC NET_STATIC_DEFINE)
endif()

target_link_libraries(net PUBLIC framework::options framework::warnings framework::rt_log)
set_target_properties(net PROPERTIES LINKER_LANGUAGE CXX)
target_link_cuda(net)
target_link_system_libraries(net PUBLIC gsl-lite tl::expected CUDA::cudart CUDA::cuda_driver)
target_link_doca(net PUBLIC)
target_link_dpdk(net PUBLIC)
target_link_gdrcopy(net)
