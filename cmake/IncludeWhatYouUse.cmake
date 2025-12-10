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

# cmake-format: off
# Helper macro to get common IWYU configuration Sets command-line options, mapping file location,
# and checks file existence Used consistently across compile-time checks and fix-includes target
# OUT_OPTIONS: List of IWYU command-line options 
# OUT_MAPPING_FILE: Path to mapping file (only set if file exists)
# OUT_FILE_EXISTS: TRUE if mapping file exists, FALSE otherwise
# cmake-format: on
macro(_get_common_iwyu_config OUT_OPTIONS OUT_MAPPING_FILE OUT_FILE_EXISTS)
    set(${OUT_OPTIONS} "--cxx17ns" "--comment_style=short" "--no_fwd_decls")
    set(_IWYU_MAPPING_PATH "${CMAKE_SOURCE_DIR}/tools/iwyu_mappings.imp")

    if(EXISTS ${_IWYU_MAPPING_PATH})
        set(${OUT_MAPPING_FILE} ${_IWYU_MAPPING_PATH})
        set(${OUT_FILE_EXISTS} TRUE)
    else()
        set(${OUT_MAPPING_FILE} "")
        set(${OUT_FILE_EXISTS} FALSE)
        message(WARNING "IWYU mapping file not found: ${_IWYU_MAPPING_PATH}")
    endif()
endmacro()

macro(enable_include_what_you_use)
    find_program(INCLUDE_WHAT_YOU_USE include-what-you-use)
    if(INCLUDE_WHAT_YOU_USE)
        # Get common IWYU configuration
        _get_common_iwyu_config(COMMON_IWYU_OPTS IWYU_MAPPING_FILE IWYU_MAPPING_EXISTS)

        # Configure IWYU with appropriate driver mode based on compiler
        set(IWYU_OPTIONS ${INCLUDE_WHAT_YOU_USE} -Xiwyu)

        # Add mapping file if it exists
        if(IWYU_MAPPING_EXISTS)
            list(APPEND IWYU_OPTIONS --mapping_file=${IWYU_MAPPING_FILE})
        endif()

        # Add common IWYU options
        foreach(OPT ${COMMON_IWYU_OPTS})
            list(APPEND IWYU_OPTIONS -Xiwyu ${OPT})
        endforeach()

        set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_OPTIONS})

    else()
        message(WARNING "include-what-you-use requested but executable not found")
    endif()
endmacro()

# Creates IWYU auto-fix target Usage: add_fix_includes_target(DIRECTORIES dir1 dir2 ...)
function(add_fix_includes_target)
    # Parse arguments
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs DIRECTORIES)
    cmake_parse_arguments(IWYU_FIX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Require directories argument
    if(NOT IWYU_FIX_DIRECTORIES)
        message(FATAL_ERROR "add_fix_includes_target requires DIRECTORIES argument")
    endif()

    find_program(IWYU_TOOL iwyu_tool)
    find_package(Python3 COMPONENTS Interpreter)

    if(NOT IWYU_TOOL)
        message(WARNING "iwyu_tool not found - IWYU auto-fix targets not created")
        return()
    endif()

    if(NOT Python3_FOUND)
        message(WARNING "Python3 interpreter not found - IWYU auto-fix targets not created")
        return()
    endif()

    # cmake-format: off
    # Use custom fix_include wrapper that prevents group consolidation 
    # The wrapper patches fix_include to treat ALL #includes as barriers, 
    # preventing consolidation of include groups separated by blank lines. 
    # This preserves manual grouping structure while still allowing add/remove operations.
    # cmake-format: on
    set(FIX_INCLUDE_WRAPPER "${CMAKE_SOURCE_DIR}/cmake/helpers/fix_include_no_regroup.py")

    if(NOT EXISTS ${FIX_INCLUDE_WRAPPER})
        message(FATAL_ERROR "fix_include wrapper not found: ${FIX_INCLUDE_WRAPPER}")
    endif()

    include(ProcessorCount)
    ProcessorCount(NUM_CORES)
    if(NUM_CORES GREATER 0)
        math(EXPR IWYU_JOBS "${NUM_CORES} / 2")
        if(IWYU_JOBS LESS 1)
            set(IWYU_JOBS 1)
        endif()
    else()
        set(IWYU_JOBS 4)
    endif()

    # Get common IWYU configuration
    _get_common_iwyu_config(COMMON_IWYU_OPTS IWYU_MAPPING_FILE IWYU_MAPPING_EXISTS)

    # Build IWYU args string with common options
    set(IWYU_COMMON_ARGS "")
    foreach(OPT ${COMMON_IWYU_OPTS})
        string(APPEND IWYU_COMMON_ARGS " -Xiwyu ${OPT}")
    endforeach()

    if(IWYU_MAPPING_EXISTS)
        set(IWYU_ARGS "-- -Xiwyu --mapping_file=${IWYU_MAPPING_FILE}${IWYU_COMMON_ARGS}")
    else()
        set(IWYU_ARGS "-- ${IWYU_COMMON_ARGS}")
    endif()

    # Convert CMake list to space-separated string for shell command
    string(REPLACE ";" " " IWYU_DIRS_STR "${IWYU_FIX_DIRECTORIES}")

    # cmake-format: off
    # Common fix_include flags:
    # --noreorder:      Prevents alphabetical sorting of includes. Preserves manual ordering
    #                   and blank line groupings. Final ordering handled by clang-format.
    # --nosafe_headers: Allows removing unused includes from headers (default only adds).
    #                   Strictly enforces "include what you use" in both .cpp and .hpp files.
    # cmake-format: on
    set(FIX_INCLUDE_FLAGS "--noreorder --nosafe_headers")

    # Common IWYU command base
    set(IWYU_CMD_BASE
        "${IWYU_TOOL} -p ${CMAKE_BINARY_DIR} -j ${IWYU_JOBS} ${IWYU_DIRS_STR} ${IWYU_ARGS}")

    # Create the fix-includes target
    add_custom_target(
        fix-includes
        COMMAND
            bash -c
            "${IWYU_CMD_BASE} | ${Python3_EXECUTABLE} ${FIX_INCLUDE_WRAPPER} ${FIX_INCLUDE_FLAGS}"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running IWYU auto-fix with ${IWYU_JOBS} parallel jobs on: ${IWYU_DIRS_STR}"
        VERBATIM)

    # Create the check-includes target
    add_custom_target(
        check-includes
        COMMAND
            bash -c
            "${IWYU_CMD_BASE} | ${Python3_EXECUTABLE} ${FIX_INCLUDE_WRAPPER} ${FIX_INCLUDE_FLAGS} --dry_run"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT
            "Checking includes with IWYU (dry-run, ${IWYU_JOBS} parallel jobs) on: ${IWYU_DIRS_STR}"
        VERBATIM)

    message(
        STATUS
            "IWYU targets created (${IWYU_JOBS} parallel jobs) for directories: ${IWYU_FIX_DIRECTORIES}"
    )
    message(STATUS "  - fix-includes: Apply IWYU suggestions")
    message(STATUS "  - check-includes: Check without modifying files (dry-run)")
endfunction()
