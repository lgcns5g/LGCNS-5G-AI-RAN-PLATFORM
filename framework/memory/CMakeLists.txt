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

# Framework Memory Library
#
# Memory management layer providing CUDA allocators, buffers, and memory wrappers

add_library(framework-memory INTERFACE)

target_sources(
    framework-memory
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/memory/device_allocators.hpp
           include/memory/memcpy_helper.hpp
           include/memory/monotonic_alloc.hpp
           include/memory/gdrcopy_buffer.hpp
           include/memory/buffer.hpp
           include/memory/unique_ptr_utils.hpp)

# Create the namespaced alias
add_library(framework::memory ALIAS framework-memory)

target_include_directories(
    framework-memory INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                               $<INSTALL_INTERFACE:include>)

target_link_libraries(framework-memory INTERFACE framework::utils framework::options
                                                 framework::warnings)
target_link_system_libraries(framework-memory INTERFACE CUDA::cudart)

# Install targets
install(
    TARGETS framework-memory
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# Install headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/memory
    DESTINATION include
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.cuh")

# Install sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/memory
    COMPONENT src
    PATTERN "CMakeLists.txt" EXCLUDE)

# Add tests subdirectory
add_subdirectory(tests)
