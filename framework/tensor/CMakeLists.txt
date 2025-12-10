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

# Framework Tensor Library
#
# Tensor infrastructure providing types, layouts and memory arenas

add_library(framework-tensor STATIC)

target_sources(
    framework-tensor
    PRIVATE src/data_types.cpp src/tensor_info.cpp src/tensor_arena.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/tensor/data_types.hpp
           include/tensor/tensor_numerology.hpp
           include/tensor/tensor_info.hpp
           include/tensor/tensor_arena.hpp)

# Create the namespaced alias
add_library(framework::tensor ALIAS framework-tensor)

target_include_directories(
    framework-tensor PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:include>)

target_link_cuda(framework-tensor)
target_link_libraries(framework-tensor PUBLIC framework::utils framework::memory framework::options
                                              framework::warnings)
target_link_system_libraries(framework-tensor PUBLIC CUDA::cudart wise_enum::wise_enum)

set_target_properties(framework-tensor PROPERTIES LINKER_LANGUAGE CXX)

# Install targets
install(
    TARGETS framework-tensor
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# Install headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tensor
    DESTINATION include
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.cuh")

# Install sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/tensor
    COMPONENT src
    PATTERN "CMakeLists.txt" EXCLUDE)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
