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

# FormatTargets.cmake
#
# Creates user-friendly clang-format checking targets that provide clear error messages instead of
# Python stacktraces. These targets complement (but don't override) the targets created by
# Format.cmake.
#
# This file should be included after Format.cmake has been loaded via CPMAddPackage.

# Only proceed if Format.cmake was successfully loaded (check for targets it creates)
if(NOT TARGET format OR NOT TARGET check-format)
    message(
        WARNING "FormatTargets.cmake: Format.cmake targets not found, skipping friendly targets")
    return()
endif()

# Find Python3 for running the wrapper script
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Path to our wrapper script
set(CLANG_FORMAT_WRAPPER_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/helpers/check_clang_format_wrapper.py")

# Path to the original git-clang-format.py from Format.cmake (from CPM source directory)
set(GIT_CLANG_FORMAT_SCRIPT "${Format.cmake_SOURCE_DIR}/git-clang-format.py")

# Verify our wrapper script exists
if(NOT EXISTS "${CLANG_FORMAT_WRAPPER_SCRIPT}")
    message(
        WARNING "FormatTargets.cmake: Wrapper script not found at ${CLANG_FORMAT_WRAPPER_SCRIPT}")
    return()
endif()

# Find clang-format binary
find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format)
if(NOT CLANG_FORMAT_EXECUTABLE)
    message(WARNING "FormatTargets.cmake: clang-format not found, skipping friendly targets")
    return()
endif()

# Find git
find_package(Git REQUIRED)

# Get the commit hash to compare against (Git's empty tree hash for checking all files)
set(CLANG_FORMAT_BASE_COMMIT "4b825dc642cb6eb9a060e54bf8d69288fbee4904")

# Create our user-friendly check-clang-format target
add_custom_target(
    check-clang-format-friendly
    COMMAND ${Python3_EXECUTABLE} ${CLANG_FORMAT_WRAPPER_SCRIPT} ${GIT_CLANG_FORMAT_SCRIPT}
            --binary=${CLANG_FORMAT_EXECUTABLE} --ci ${CLANG_FORMAT_BASE_COMMIT}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking code formatting with clang-format (user-friendly errors)"
    VERBATIM)

# Create user-friendly check-format target that depends on both clang and cmake format checks
if(TARGET check-cmake-format)
    add_custom_target(
        check-format-friendly
        DEPENDS check-clang-format-friendly check-cmake-format
        COMMENT "Checking all code formatting (C++ and CMake)")
else()
    add_custom_target(
        check-format-friendly
        DEPENDS check-clang-format-friendly
        COMMENT "Checking all code formatting (C++ only)")
endif()

message(
    STATUS
        "FormatTargets: Created user-friendly formatting targets (check-clang-format-friendly, check-format-friendly)"
)
