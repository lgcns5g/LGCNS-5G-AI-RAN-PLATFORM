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

# Framework Pipeline Library
#
# Pipeline framework providing modules, pipelines, and graph/stream/kernel execution infrastructure

add_library(framework-pipeline STATIC)

target_sources(
    framework-pipeline
    PRIVATE src/module_factory.cpp
            src/module_router.cpp
            src/pipeline_factory.cpp
            src/pipeline_memory_manager.cpp
            src/graph.cpp
            src/graph_manager.cpp
            src/kernel_node_helper.cpp
            src/dynamic_kernel_launch_config.cpp
            src/kernel_descriptor_accessor.cpp
            src/stream_executor.cpp
            src/stream_executor_factory.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           # Core types
           include/pipeline/types.hpp
           # Module system
           include/pipeline/imodule.hpp
           include/pipeline/imodule_factory.hpp
           include/pipeline/module_factory.hpp
           include/pipeline/module_router.hpp
           # Pipeline system
           include/pipeline/ipipeline.hpp
           include/pipeline/ipipeline_factory.hpp
           include/pipeline/pipeline_factory.hpp
           include/pipeline/pipeline_memory_manager.hpp
           include/pipeline/iallocation_info_provider.hpp
           # CUDA graph execution
           include/pipeline/igraph.hpp
           include/pipeline/graph.hpp
           include/pipeline/igraph_manager.hpp
           include/pipeline/igraph_node_provider.hpp
           include/pipeline/graph_manager.hpp
           # Stream execution
           include/pipeline/istream_executor.hpp
           include/pipeline/stream_executor.hpp
           include/pipeline/null_stream_executor.hpp
           include/pipeline/stream_executor_factory.hpp
           # Kernel execution config
           include/pipeline/ikernel_launch_config.hpp
           include/pipeline/kernel_launch_config.hpp
           include/pipeline/dynamic_kernel_launch_config.hpp
           include/pipeline/kernel_node_helper.hpp
           include/pipeline/kernel_descriptor_accessor.hpp)

# Create the namespaced alias
add_library(framework::pipeline ALIAS framework-pipeline)

target_include_directories(
    framework-pipeline PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:include>)

target_link_cuda(framework-pipeline)
target_link_libraries(
    framework-pipeline PUBLIC framework::utils framework::memory framework::tensor
                              framework::tensorrt framework::options framework::warnings)
target_link_system_libraries(
    framework-pipeline
    PUBLIC
    wise_enum::wise_enum
    CUDA::cudart
    CUDA::cuda_driver
    gsl-lite
    NamedType)

set_target_properties(framework-pipeline PROPERTIES LINKER_LANGUAGE CXX)

# Install targets
install(
    TARGETS framework-pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# Install headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline
    DESTINATION include
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.cuh")

# Install sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/pipeline
    COMPONENT src
    PATTERN "CMakeLists.txt" EXCLUDE)

# Add subdirectories
add_subdirectory(tests)
add_subdirectory(samples)
