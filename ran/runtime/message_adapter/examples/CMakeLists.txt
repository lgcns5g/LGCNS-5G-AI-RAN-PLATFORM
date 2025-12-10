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

add_library(sample_5g_pipeline STATIC)
add_library(ran::sample_5g_pipeline ALIAS sample_5g_pipeline)

target_sources(
    sample_5g_pipeline
    PRIVATE sample_5g_pipelines.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${CMAKE_CURRENT_SOURCE_DIR}
           FILES
           cell.hpp
           sample_5g_pipelines.hpp)

target_include_directories(
    sample_5g_pipeline
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
           $<BUILD_INTERFACE:${MESSAGE_ADAPTER_INCLUDE}> $<INSTALL_INTERFACE:include/>)

target_link_libraries(
    sample_5g_pipeline
    PUBLIC ran::common
           ran::driver
           ran::fapi
           ran::message_adapter
           framework::pipeline
           framework::rt_log
           framework::warnings
           framework::options
    PRIVATE ran::fronthaul ${CMAKE_DL_LIBS})

target_link_system_libraries(
    sample_5g_pipeline
    PUBLIC
    nvipc
    scf_5g_fapi_h
    PRIVATE
    nvlog
    yaml)

set_target_properties(sample_5g_pipeline PROPERTIES LINKER_LANGUAGE CXX)

install(
    TARGETS sample_5g_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    FILES cell.hpp sample_5g_pipelines.hpp
    DESTINATION include/sample_5g_pipeline
    COMPONENT dev)

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/sample_5g_pipeline
    COMPONENT src
    FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.hpp")

set_target_properties(
    sample_5g_pipeline
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(sample_5g_pipeline PUBLIC SAMPLE_5G_PIPELINE_STATIC_DEFINE)
endif()

message(STATUS "Building sample_5g_pipeline library")
