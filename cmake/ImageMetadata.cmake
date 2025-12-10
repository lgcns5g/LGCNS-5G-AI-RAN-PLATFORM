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

# ImageMetadata.cmake
#
# Creates targets for clearing metadata from image files (XML and SVG) to reduce noise in diffs.

include_guard(GLOBAL)

# Find Python3 for running the scripts
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Path to script
set(CLEAR_METADATA_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/helpers/clear_metadata.py")

# Verify script exists
if(NOT EXISTS "${CLEAR_METADATA_SCRIPT}")
    message(
        FATAL_ERROR "ImageMetadata.cmake: clear_metadata.py not found at ${CLEAR_METADATA_SCRIPT}")
endif()

# Create clear-figure-metadata target
add_custom_target(
    clear-figure-metadata
    COMMAND ${Python3_EXECUTABLE} ${CLEAR_METADATA_SCRIPT}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Clearing metadata from figure files (XML/SVG)"
    VERBATIM)

message(STATUS "ImageMetadata: Created target clear-figure-metadata")
