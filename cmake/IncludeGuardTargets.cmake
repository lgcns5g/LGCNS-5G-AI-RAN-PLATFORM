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

# IncludeGuardTargets.cmake
#
# Creates targets for checking include guard compliance in C/C++/CUDA header files. Ensures that all
# headers use proper include guards (not #pragma once) with correct prefixes.

include_guard(GLOBAL)

# Find Python3 for running the include guard checking scripts
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Path to consolidated CLI script
set(INCLUDE_GUARDS_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/helpers/include_guards.py")

# Verify script exists
if(NOT EXISTS "${INCLUDE_GUARDS_SCRIPT}")
    message(
        FATAL_ERROR
            "IncludeGuardTargets.cmake: include_guards.py not found at ${INCLUDE_GUARDS_SCRIPT}")
endif()

# Find git
find_package(Git REQUIRED)

# Create check-include-guards target (validation only, exit 1 if violations found)
add_custom_target(
    check-include-guards
    COMMAND ${Python3_EXECUTABLE} ${INCLUDE_GUARDS_SCRIPT} check
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking include guard compliance for header files"
    VERBATIM)

# Create fix-include-guards target (modifies files to fix include guards)
add_custom_target(
    fix-include-guards
    COMMAND ${Python3_EXECUTABLE} ${INCLUDE_GUARDS_SCRIPT} fix
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Fixing include guards in header files"
    VERBATIM)

message(
    STATUS
        "IncludeGuardTargets: Created include guard checking targets (check-include-guards, fix-include-guards)"
)
