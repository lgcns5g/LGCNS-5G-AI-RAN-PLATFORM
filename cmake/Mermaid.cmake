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

# Mermaid.cmake
#
# Creates targets for generating SVG diagrams from Mermaid source files.

include_guard(GLOBAL)

# Find Python3 for running the scripts
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Path to script
set(GENERATE_MERMAID_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/generate_mermaid_svgs.py")

# Verify script exists
if(NOT EXISTS "${GENERATE_MERMAID_SCRIPT}")
    message(
        FATAL_ERROR
            "Mermaid.cmake: generate_mermaid_svgs.py not found at ${GENERATE_MERMAID_SCRIPT}")
endif()

# Create generate-mermaid-svgs target
add_custom_target(
    generate-mermaid-svgs
    COMMAND ${Python3_EXECUTABLE} ${GENERATE_MERMAID_SCRIPT}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating SVG diagrams from Mermaid source files"
    VERBATIM)

message(STATUS "Mermaid: Created target generate-mermaid-svgs")
