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

cmake_minimum_required(VERSION 3.23) # required for FILE_SET HEADERS

# Only set the cxx_standard if it is not set by someone else
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

# strongly encouraged to enable this globally to avoid conflicts between -Wpedantic being enabled
# and -std=c++20 and -std=gnu++20 for example when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position-independent code globally for all static libraries Required for CUDA device
# linking and potential shared library usage
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Read full version from VERSION file
file(STRINGS "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION_FULL)
string(STRIP "${PROJECT_VERSION_FULL}" PROJECT_VERSION_FULL)

# Extract base version for CMake's VERSION (0.1.0 from 0.1.0-rc.3)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJECT_VERSION_BASE "${PROJECT_VERSION_FULL}")

# Set the project name and language
project(
    framework
    VERSION ${PROJECT_VERSION_BASE}
    DESCRIPTION ""
    HOMEPAGE_URL "%%myurl%%"
    LANGUAGES CXX C ASM CUDA)

# Validate CUDA architectures are set
include(cmake/Cuda.cmake)
check_cuda_architectures()

include(cmake/PreventInSourceBuilds.cmake)
include(cmake/ProjectOptions.cmake)

setup_options()

local_options()

global_options()

include(cmake/Dependencies.cmake)
setup_dependencies()

include(cmake/Python.cmake)
enable_static_analysis()

if(ENABLE_IWYU)
    add_fix_includes_target(DIRECTORIES framework ran)
endif()

# Add copyright and include guard checking targets
include(cmake/CopyrightTargets.cmake)
include(cmake/IncludeGuardTargets.cmake)

# Add figure management targets
include(cmake/ImageMetadata.cmake)
include(cmake/Mermaid.cmake)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(GIT_SHA
    "Unknown"
    CACHE STRING "SHA this build was generated from")
string(SUBSTRING "${GIT_SHA}" 0 8 GIT_SHORT_SHA)

target_compile_features(framework_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

add_library(framework::options ALIAS framework_options)
add_library(framework::warnings ALIAS framework_warnings)

# Adding the tests:
include(CTest)

# configure files based on CMake configuration options
add_subdirectory(cmake/configured_files)

add_subdirectory(framework)
add_subdirectory(ran)

# cmake-format: off
# Add docs after the main project targets so doxygen and sphinx targets can reference them
# When BUILD_DOCS or ENFORCE_DOCSTRINGS is OFF, docs become warning-only targets
# cmake-format: on
add_subdirectory(docs)

# Don't even look at tests if we're not top level
if(NOT PROJECT_IS_TOP_LEVEL)
    return()
endif()

if(CMAKE_SKIP_INSTALL_RULES)
    return()
endif()
