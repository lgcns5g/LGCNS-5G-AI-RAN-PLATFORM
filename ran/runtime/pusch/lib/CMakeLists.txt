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

include("${CMAKE_SOURCE_DIR}/cmake/RanTrtPlugin.cmake")

add_library(pusch STATIC)
add_library(ran::pusch ALIAS pusch)

target_sources(
    pusch
    PRIVATE src/inner_rx_module.cpp src/pusch_module_factories.cpp src/pusch_pipeline.cpp
            src/pusch_trt_utils.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${CMAKE_CURRENT_SOURCE_DIR}/include
           FILES
           include/pusch/inner_rx_module.hpp
           include/pusch/pusch_defines.hpp
           include/pusch/pusch_log.hpp
           include/pusch/pusch_module_factories.hpp
           include/pusch/pusch_pipeline.hpp
           include/pusch/pusch_trt_utils.hpp)

target_include_directories(
    pusch PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<INSTALL_INTERFACE:include/>)
target_link_libraries(
    pusch
    PUBLIC ran::common
           framework::utils
           framework::memory
           framework::tensor
           framework::tensorrt
           framework::pipeline
           ran::ldpc
           framework::rt_log
           framework::warnings
           framework::options
           CUDA::cudart)
target_link_system_libraries(pusch PUBLIC wise_enum::wise_enum gsl-lite::gsl-lite)

# Link RAN TRT plugin (handles both ENABLE_MLIR_TRT=ON and OFF)
target_link_ran_trt_plugin(pusch PUBLIC)

set_target_properties(pusch PROPERTIES LINKER_LANGUAGE CXX)

install(
    TARGETS pusch
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    FILES include/pusch/inner_rx_module.hpp include/pusch/pusch_defines.hpp
          include/pusch/pusch_log.hpp include/pusch/pusch_module_factories.hpp
          include/pusch/pusch_pipeline.hpp include/pusch/pusch_trt_utils.hpp
    DESTINATION include/pusch
    COMPONENT dev)

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION src/aerial-framework/pusch
    COMPONENT src
    FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.hpp")

set_target_properties(
    pusch
    PROPERTIES VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(pusch PUBLIC PUSCH_STATIC_DEFINE)
endif()
