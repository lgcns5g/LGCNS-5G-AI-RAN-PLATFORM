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

# CopyrightTargets.cmake
#
# Creates targets for checking and fixing copyright headers to ensure SPDX compliance. These targets
# scan C/C++/CUDA/Python/Shell/CMake files for proper copyright headers.

include_guard(GLOBAL)

# Find Python3 for running the copyright checking scripts
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Path to consolidated CLI script
set(COPYRIGHT_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/helpers/copyright.py")

# Verify script exists
if(NOT EXISTS "${COPYRIGHT_SCRIPT}")
    message(FATAL_ERROR "CopyrightTargets.cmake: copyright.py not found at ${COPYRIGHT_SCRIPT}")
endif()

# Find git
find_package(Git REQUIRED)

# Create check-copyright target (validation only, exit 1 if violations found)
add_custom_target(
    check-copyright
    COMMAND ${Python3_EXECUTABLE} ${COPYRIGHT_SCRIPT} check
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking copyright headers for SPDX compliance"
    VERBATIM)

# Create fix-copyright target (modifies files to fix copyright headers)
add_custom_target(
    fix-copyright
    COMMAND ${Python3_EXECUTABLE} ${COPYRIGHT_SCRIPT} fix
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Fixing copyright headers to ensure SPDX compliance"
    VERBATIM)

message(
    STATUS "CopyrightTargets: Created copyright checking targets (check-copyright, fix-copyright)")
