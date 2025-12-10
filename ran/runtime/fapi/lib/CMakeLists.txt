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

add_library(fapi STATIC)
add_library(ran::fapi ALIAS fapi)

target_sources(
    fapi
    PRIVATE src/fapi_state.cpp src/fapi_file_writer.cpp src/fapi_file_reader.cpp
            src/fapi_file_replay.cpp src/fapi_utils.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/fapi/fapi_buffer.hpp
           include/fapi/fapi_state.hpp
           include/fapi/fapi_log.hpp
           include/fapi/fapi_file_writer.hpp
           include/fapi/fapi_file_reader.hpp
           include/fapi/fapi_file_replay.hpp
           include/fapi/fapi_utils.hpp)

target_link_libraries(fapi PUBLIC framework::options framework::warnings framework::rt_log)
target_link_system_libraries(
    fapi
    PUBLIC
    nvipc
    scf_5g_fapi_h
    yaml
    nvlog
    gsl-lite::gsl-lite
    wise_enum::wise_enum)
set_target_properties(fapi PROPERTIES LINKER_LANGUAGE CXX)

install(
    TARGETS fapi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/fapi
    DESTINATION include/
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h")

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/fapi
    COMPONENT src)

target_include_directories(
    fapi
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include/>)

set_target_properties(
    fapi
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(fapi EXPORT_FILE_NAME
                       ${CMAKE_CURRENT_BINARY_DIR}/include/fapi/fapi_export.hpp)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(fapi PUBLIC FAPI_STATIC_DEFINE)
endif()
