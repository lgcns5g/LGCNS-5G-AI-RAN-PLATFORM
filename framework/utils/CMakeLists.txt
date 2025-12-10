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

# Framework Utilities Library
#
# Foundation layer providing CUDA utilities, error handling, logging, and string utilities

add_library(framework-utils STATIC)

target_sources(
    framework-utils
    PRIVATE src/cuda_stream.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           include
           FILES
           include/utils/arr.hpp
           include/utils/string_hash.hpp
           include/utils/cuda_defines.hpp
           include/utils/cuda_stream.hpp
           include/utils/variant_types.hpp
           include/utils/errors.hpp
           include/utils/exceptions.hpp
           include/utils/error_macros.hpp
           include/utils/core_log.hpp
           include/utils/details/underlying.hpp)

# Create the namespaced alias
add_library(framework::utils ALIAS framework-utils)

target_include_directories(
    framework-utils PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_link_cuda(framework-utils)
target_link_libraries(framework-utils PUBLIC framework::options framework::warnings
                                             framework::rt_log)
target_link_system_libraries(framework-utils PUBLIC CUDA::cudart CUDA::cuda_driver gsl-lite)

set_target_properties(framework-utils PROPERTIES LINKER_LANGUAGE CXX)

# Install targets
install(
    TARGETS framework-utils
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# Install headers – component dev
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    DESTINATION include
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.cuh")

# Install sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/utils
    COMPONENT src
    PATTERN "CMakeLists.txt" EXCLUDE)

# Add tests subdirectory
add_subdirectory(tests)
