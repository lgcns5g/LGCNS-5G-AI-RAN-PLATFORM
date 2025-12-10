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

add_library(rt_log STATIC)
add_library(framework::rt_log ALIAS rt_log)

target_sources(
    rt_log
    PRIVATE src/rt_log.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/log/rt_log.hpp
           include/log/rt_log_macros.hpp
           include/log/components.hpp)

target_link_libraries(rt_log PUBLIC framework::options framework::warnings)
target_link_system_libraries(rt_log PUBLIC quill::quill wise_enum::wise_enum)
set_target_properties(rt_log PROPERTIES LINKER_LANGUAGE CXX)

install(
    TARGETS rt_log
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/log
    DESTINATION include/log
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h")

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/log
    COMPONENT src)

target_include_directories(
    rt_log
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include/log>)

set_target_properties(
    rt_log
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(rt_log EXPORT_FILE_NAME
                       ${CMAKE_CURRENT_BINARY_DIR}/include/log/rt_log_export.hpp)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(rt_log PUBLIC RT_LOG_STATIC_DEFINE)
endif()
