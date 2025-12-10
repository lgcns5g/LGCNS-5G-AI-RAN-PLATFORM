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

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui, ccmake
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                                 "RelWithDebInfo")
endif()

# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enhance error reporting and compiler messages
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    if(WIN32)
        # On Windows cuda nvcc uses cl and not clang
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-fcolor-diagnostics>
                            $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
    else()
        # Force colour even when output is not a TTY (e.g. CI logs)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-fdiagnostics-color=always>
                            $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(WIN32)
        # On Windows cuda nvcc uses cl and not gcc
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-fdiagnostics-color=always>
                            $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-fdiagnostics-color=always>
                            $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND MSVC_VERSION GREATER 1900)
    add_compile_options(/diagnostics:column)
else()
    message(STATUS "No colored compiler diagnostic set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
endif()
