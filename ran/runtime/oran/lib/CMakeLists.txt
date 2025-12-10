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

add_library(oran STATIC)
add_library(ran::oran ALIAS oran)

target_sources(
    oran
    PRIVATE src/cplane_utils.cpp src/cplane_message.cpp src/fapi_to_cplane.cpp src/numerology.cpp
            src/dpdk_buf.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/oran/oran_log.hpp
           include/oran/oran_errors.hpp
           include/oran/oran_buf.hpp
           include/oran/vec_buf.hpp
           include/oran/cplane_types.hpp
           include/oran/cplane_utils.hpp
           include/oran/cplane_message.hpp
           include/oran/fapi_to_cplane.hpp
           include/oran/dpdk_buf.hpp
           include/oran/numerology.hpp)
target_link_cuda(oran)
target_link_libraries(
    oran
    PUBLIC ran::fapi framework::net
    PRIVATE framework::options framework::warnings framework::rt_log)
target_link_system_libraries(oran PUBLIC CUDA::cudart CUDA::cuda_driver scf_5g_fapi_h oran_h)

target_include_directories(
    oran
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/oran/details
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

set_target_properties(
    oran
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

install(
    TARGETS oran
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/oran
    DESTINATION include/
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h")

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/oran
    COMPONENT src)

generate_export_header(oran EXPORT_FILE_NAME
                       ${CMAKE_CURRENT_BINARY_DIR}/include/oran/oran_export.hpp)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(oran PUBLIC ORAN_STATIC_DEFINE)
endif()
