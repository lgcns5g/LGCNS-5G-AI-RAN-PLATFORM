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

add_library(driver STATIC)
add_library(ran::driver ALIAS driver)

target_sources(
    driver
    PRIVATE lib/src/driver.cpp lib/src/pusch_pipeline_context.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
           FILES
           lib/include/driver/driver.hpp
           lib/include/driver/driver_log.hpp
           lib/include/driver/pusch_pipeline_context.hpp)

target_include_directories(
    driver PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include>
                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<INSTALL_INTERFACE:include/>)

target_link_libraries(
    driver
    PUBLIC ran::common
           ran::pusch
           framework::pipeline
           framework::rt_log
           framework::warnings
           framework::options
           CUDA::cudart)

target_link_system_libraries(driver PUBLIC scf_5g_fapi_h)

set_target_properties(driver PROPERTIES LINKER_LANGUAGE CXX)

install(
    TARGETS driver
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    FILES lib/include/driver/driver.hpp lib/include/driver/driver_log.hpp
          lib/include/driver/pusch_pipeline_context.hpp
    DESTINATION include/driver
    COMPONENT dev)

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    DESTINATION src/aerial-framework/driver
    COMPONENT src
    FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.hpp")

set_target_properties(
    driver
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(driver PUBLIC DRIVER_STATIC_DEFINE)
endif()
