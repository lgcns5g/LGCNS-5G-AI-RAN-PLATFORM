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

include("${CMAKE_CURRENT_LIST_DIR}/Utilities.cmake")

# Enable doxygen doc builds of source
function(enable_doxygen DOXYGEN_THEME)
    # Don't use README files as mainpage since they're excluded from Doxygen processing README files
    # are user documentation, not API documentation

    # set better defaults for doxygen
    is_verbose(_is_verbose)
    if(NOT ${_is_verbose})
        set(DOXYGEN_QUIET YES)
    endif()
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)

    # Configure graph generation only if dot is available DOXYGEN_HAVE_DOT is automatically set by
    # find_package(Doxygen)
    if(DOXYGEN_HAVE_DOT)
        set(DOXYGEN_CALLER_GRAPH YES)
        set(DOXYGEN_CALL_GRAPH YES)
        set(DOXYGEN_DOT_IMAGE_FORMAT svg)
        set(DOXYGEN_DOT_TRANSPARENT YES)
        set(DOXYGEN_DOT_GRAPH_MAX_NODES 200)
        message(STATUS "Graphviz dot found - enabling call graphs")
    else()
        set(DOXYGEN_HAVE_DOT NO)
        set(DOXYGEN_CALLER_GRAPH NO)
        set(DOXYGEN_CALL_GRAPH NO)
        set(DOXYGEN_GRAPHICAL_HIERARCHY NO)
        set(DOXYGEN_DIRECTORY_GRAPH NO)
        set(DOXYGEN_COLLABORATION_GRAPH NO)
        set(DOXYGEN_GROUP_GRAPHS NO)
        set(DOXYGEN_INCLUDE_GRAPH NO)
        set(DOXYGEN_INCLUDED_BY_GRAPH NO)
        message(STATUS "Graphviz dot not found - all graph generation disabled")
    endif()

    # Configure XML output for Breathe/Sphinx integration
    set(DOXYGEN_GENERATE_XML YES)
    set(DOXYGEN_XML_OUTPUT xml)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_HTML_OUTPUT html)

    # Restrict to C/C++/CUDA files only
    set(DOXYGEN_FILE_PATTERNS
        "*.h"
        "*.hpp"
        "*.hxx"
        "*.c"
        "*.cpp"
        "*.cxx"
        "*.cc"
        "*.cu"
        "*.cuh")

    # Enable source browsing for source links in documentation
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_REFERENCED_BY_RELATION YES)
    set(DOXYGEN_REFERENCES_RELATION YES)
    set(DOXYGEN_REFERENCES_LINK_SOURCE YES)
    set(DOXYGEN_SOURCE_TOOLTIPS YES)
    set(DOXYGEN_VERBATIM_HEADERS YES)

    # Set consistent output directory
    if(NOT DOXYGEN_OUTPUT_DIRECTORY)
        set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs/doxygen)
    endif()

    # Configure warnings as errors for documentation build
    set(DOXYGEN_WARN_AS_ERROR YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_WARN_NO_PARAMDOC YES)
    set(DOXYGEN_WARN_IF_DOC_ERROR YES)
    set(DOXYGEN_WARNINGS YES)

    # Configure preprocessing to handle export macros
    set(DOXYGEN_ENABLE_PREPROCESSING YES)
    set(DOXYGEN_MACRO_EXPANSION YES)
    set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
    set(DOXYGEN_SEARCH_INCLUDES YES)

    # Get common predefined macros for preprocessing
    get_doxygen_predefined_macros(DOXYGEN_PREDEFINED)

    # If not specified, exclude the vcpkg files and the files CMake downloads under _deps (like
    # project_options)
    if(NOT DOXYGEN_EXCLUDE_PATTERNS)
        set(DOXYGEN_EXCLUDE_PATTERNS
            "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/*" "${CMAKE_BINARY_DIR}/vcpkg_installed/*"
            "${PROJECT_SOURCE_DIR}/out/*" "${PROJECT_SOURCE_DIR}/external/*" "*/_deps/*" "*.md")
    endif()

    if("${DOXYGEN_THEME}" STREQUAL "")
        set(DOXYGEN_THEME "awesome-sidebar")
    endif()

    if("${DOXYGEN_THEME}" STREQUAL "awesome" OR "${DOXYGEN_THEME}" STREQUAL "awesome-sidebar")
        # use a modern doxygen theme https://github.com/jothepro/doxygen-awesome-css v1.6.1
        FetchContent_Declare(
            _doxygen_theme
            URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v1.6.1.zip)
        FetchContent_MakeAvailable(_doxygen_theme)
        if("${DOXYGEN_THEME}" STREQUAL "awesome" OR "${DOXYGEN_THEME}" STREQUAL "awesome-sidebar")
            set(DOXYGEN_HTML_EXTRA_STYLESHEET "${_doxygen_theme_SOURCE_DIR}/doxygen-awesome.css")
        endif()
        if("${DOXYGEN_THEME}" STREQUAL "awesome-sidebar")
            set(DOXYGEN_HTML_EXTRA_STYLESHEET
                ${DOXYGEN_HTML_EXTRA_STYLESHEET}
                "${_doxygen_theme_SOURCE_DIR}/doxygen-awesome-sidebar-only.css")
        endif()
    else()
        # use the original doxygen theme
    endif()

    # find doxygen and dot if available
    find_package(Doxygen REQUIRED OPTIONAL_COMPONENTS dot)

    # add doxygen-docs target
    message(STATUS "Adding `doxygen-docs` target that builds the documentation.")
    message(STATUS "Doxygen XML output: ${DOXYGEN_OUTPUT_DIRECTORY}/xml")
    doxygen_add_docs(
        doxygen-docs ALL ${PROJECT_SOURCE_DIR}
        COMMENT "Generating documentation - entry file: ${CMAKE_CURRENT_BINARY_DIR}/html/index.html"
    )
endfunction()

#[=======================================================================[
Doxygen Documentation Enforcement Functions

target_enforce_docstrings - Enforce documentation for a specific CMake target
enforce_docstrings - Enforce documentation for an entire directory tree
generate_doxygen_config - Helper to generate Doxygen configuration
generate_warnings_check_script - Helper to generate warnings check script
#]=======================================================================]

# Helper function to generate Doxygen configuration content
function(generate_doxygen_config output_var project_name output_dir input_dirs)
    # Get common predefined macros for preprocessing
    get_doxygen_predefined_macros(PREDEFINED_MACROS)

    # Convert list to space-separated string for Doxygen config file
    string(REPLACE ";" " " PREDEFINED_MACROS_STR "${PREDEFINED_MACROS}")

    set(${output_var}
        "
PROJECT_NAME           = \"${project_name}\"
OUTPUT_DIRECTORY       = ${output_dir}
INPUT                  = ${input_dirs}

# File processing
RECURSIVE              = YES
FILE_PATTERNS          = *.h *.hpp *.hxx *.c *.cpp *.cxx *.cc *.cu *.cuh

# Preprocessing settings to handle attributes and export macros
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = YES
EXPAND_ONLY_PREDEF     = YES
SEARCH_INCLUDES        = YES
PREDEFINED             = ${PREDEFINED_MACROS_STR}

# Documentation warnings
WARN_IF_UNDOCUMENTED   = YES
WARN_NO_PARAMDOC       = YES
WARN_IF_DOC_ERROR      = YES
WARN_AS_ERROR          = NO
WARNINGS               = YES
WARN_FORMAT            = \"$file:$line: warning: $text\"
WARN_LOGFILE           = ${output_dir}/warnings.log

# Extraction settings
EXTRACT_ALL            = NO
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = NO
EXTRACT_LOCAL_CLASSES  = NO
EXTRACT_LOCAL_METHODS  = NO
EXTRACT_ANON_NSPACES   = NO

# Enable minimal HTML generation (required for parameter validation)
GENERATE_HTML          = YES
HTML_OUTPUT            = html
HTML_TIMESTAMP         = NO
HTML_DYNAMIC_SECTIONS  = NO

# Disable graph generation (graphviz/dot not required)
HAVE_DOT               = NO
CLASS_DIAGRAMS         = NO
CALL_GRAPH             = NO
CALLER_GRAPH           = NO
GRAPHICAL_HIERARCHY    = NO
DIRECTORY_GRAPH        = NO
COLLABORATION_GRAPH    = NO
GROUP_GRAPHS           = NO
INCLUDE_GRAPH          = NO
INCLUDED_BY_GRAPH      = NO

# Disable everything else
GENERATE_LATEX         = NO
GENERATE_XML           = NO
GENERATE_RTF           = NO
GENERATE_MAN           = NO
GENERATE_DOCBOOK       = NO
QUIET                  = YES
"
        PARENT_SCOPE)
endfunction()

# Helper function to generate the warnings check script
function(generate_warnings_check_script output_file target_name output_dir fail_on_warnings)
    # Convert boolean to string for injection into the script
    if(fail_on_warnings)
        set(FAIL_ON_WARNINGS_VALUE "TRUE")
    else()
        set(FAIL_ON_WARNINGS_VALUE "FALSE")
    endif()

    file(
        WRITE ${output_file}
        "
# Run Doxygen for docstring checking
execute_process(
    COMMAND ${DOXYGEN_EXECUTABLE} ${output_dir}/Doxyfile
    RESULT_VARIABLE RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR \"Doxygen failed to run for docstring check\")
endif()

# Clean up HTML output (we only needed it for validation)
file(REMOVE_RECURSE \"${output_dir}/html\")

# Check warnings
set(WARNING_LOG \"${output_dir}/warnings.log\")
if(EXISTS \"\${WARNING_LOG}\")
    file(READ \"\${WARNING_LOG}\" WARNINGS)
    if(WARNINGS)
        # Split warnings into individual lines for more accurate counting
        string(REGEX REPLACE \"\\r?\\n\" \";\" WARNING_LINES \"\${WARNINGS}\")

        # Count lines that contain warning indicators (case-insensitive)
        set(COUNT 0)
        foreach(line IN LISTS WARNING_LINES)
            if(line MATCHES \".*[Ww]arning:.*\")
                math(EXPR COUNT \"\${COUNT} + 1\")
            endif()
        endforeach()

        message(STATUS \"=== Docstring Check: ${target_name} ===\")
        message(STATUS \"Found \${COUNT} undocumented function(s):\")
        message(STATUS \"\${WARNINGS}\")
        message(STATUS \"====================================\")

        # Define the fail_on_warnings variable based on the parameter
        set(fail_on_warnings ${FAIL_ON_WARNINGS_VALUE})

        if(fail_on_warnings)
            message(FATAL_ERROR \"Build failed: undocumented functions found in ${target_name}\")
        else()
            message(WARNING \"Found undocumented functions in ${target_name}\")
        endif()
    else()
        message(STATUS \"✓ All functions documented in ${target_name}\")
    endif()
else()
    message(STATUS \"✓ No documentation warnings found for ${target_name}\")
endif()
")
endfunction()

#[=======================================================================[
target_enforce_docstrings - Enforce documentation for a specific CMake target

Usage:
  target_enforce_docstrings(
    TARGET <target_name>
    DIRECTORIES <dir1> [<dir2>...]
    [FAIL_ON_WARNINGS]
  )

Example:
  target_enforce_docstrings(
    TARGET mylib
    DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
    FAIL_ON_WARNINGS
  )

Creates target: <target_name>_docstring_check
#]=======================================================================]
function(target_enforce_docstrings)
    # Parse arguments
    set(options FAIL_ON_WARNINGS)
    set(oneValueArgs TARGET)
    set(multiValueArgs DIRECTORIES)

    cmake_parse_arguments(DOC_ENFORCE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT DOC_ENFORCE_TARGET)
        message(FATAL_ERROR "TARGET is required for target_enforce_docstrings")
    endif()

    if(NOT DOC_ENFORCE_DIRECTORIES)
        message(FATAL_ERROR "DIRECTORIES is required for target_enforce_docstrings")
    endif()

    # Ensure Doxygen is available
    find_package(Doxygen REQUIRED)

    # Set output directory for this target's checks
    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docstring_check_${DOC_ENFORCE_TARGET}")
    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    # Convert directory list to space-separated string
    string(REPLACE ";" " " DOXYGEN_INPUT_DIRS "${DOC_ENFORCE_DIRECTORIES}")

    # Generate Doxygen configuration
    generate_doxygen_config(DOXYGEN_CONFIG_CONTENT "${DOC_ENFORCE_TARGET} Docstring Check"
                            "${OUTPUT_DIR}" "${DOXYGEN_INPUT_DIRS}")

    # Write the enforcement Doxyfile
    set(DOXYGEN_CONFIG_FILE "${OUTPUT_DIR}/Doxyfile")
    file(WRITE ${DOXYGEN_CONFIG_FILE} "${DOXYGEN_CONFIG_CONTENT}")

    # Create check script
    set(CHECK_SCRIPT "${OUTPUT_DIR}/check.cmake")
    generate_warnings_check_script(${CHECK_SCRIPT} "${DOC_ENFORCE_TARGET}" "${OUTPUT_DIR}"
                                   ${DOC_ENFORCE_FAIL_ON_WARNINGS})

    # Create the docstring check target
    add_custom_target(
        ${DOC_ENFORCE_TARGET}_docstring_check
        COMMAND ${CMAKE_COMMAND} -P ${CHECK_SCRIPT}
        COMMENT "Checking docstrings for ${DOC_ENFORCE_TARGET}")

    # Create a check_all_docstrings target if it doesn't exist
    if(NOT TARGET check_all_docstrings)
        add_custom_target(check_all_docstrings COMMENT "Checking docstrings for all targets")
    endif()

    # Add this target's check to the global check
    add_dependencies(check_all_docstrings ${DOC_ENFORCE_TARGET}_docstring_check)

    # Optionally make the docstring check run before the main target
    if(TARGET ${DOC_ENFORCE_TARGET})
        add_dependencies(${DOC_ENFORCE_TARGET} ${DOC_ENFORCE_TARGET}_docstring_check)
    endif()

endfunction()

#[=======================================================================[
enforce_docstrings - Enforce documentation for an entire directory tree

Usage:
  enforce_docstrings(
    NAME <name>
    ROOT_DIRECTORY <directory>
    [EXCLUDE_DIRECTORIES <dir1> [<dir2>...]]
    [EXCLUDE_PATTERNS <pattern1> [<pattern2>...]]
    [FAIL_ON_WARNINGS]
  )

Examples:
  # Check entire framework directory
  enforce_docstrings(
    NAME framework
    ROOT_DIRECTORY ${CMAKE_SOURCE_DIR}/framework
    EXCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/framework/build
    EXCLUDE_PATTERNS "*/test/*" "*/tests/*" "*_test.cpp"
    FAIL_ON_WARNINGS
  )

  # Check ran with pattern exclusions
  enforce_docstrings(
    NAME ran
    ROOT_DIRECTORY ${CMAKE_SOURCE_DIR}/ran
    EXCLUDE_PATTERNS "*/test/*" "*_test.cpp"
  )

Creates target: <name>_docstring_check
#]=======================================================================]
function(enforce_docstrings)
    # Parse arguments
    set(options FAIL_ON_WARNINGS)
    set(oneValueArgs NAME ROOT_DIRECTORY)
    set(multiValueArgs EXCLUDE_DIRECTORIES EXCLUDE_PATTERNS)

    cmake_parse_arguments(DOC_ENFORCE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Set defaults
    if(NOT DOC_ENFORCE_NAME)
        get_filename_component(DOC_ENFORCE_NAME ${DOC_ENFORCE_ROOT_DIRECTORY} NAME)
    endif()

    if(NOT DOC_ENFORCE_ROOT_DIRECTORY)
        message(FATAL_ERROR "ROOT_DIRECTORY is required for enforce_docstrings")
    endif()

    # Ensure Doxygen is available
    find_package(Doxygen REQUIRED)

    # Set output directory for this check
    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docstring_check_${DOC_ENFORCE_NAME}")
    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    # Generate Doxygen configuration
    generate_doxygen_config(DOXYGEN_CONFIG_CONTENT "${DOC_ENFORCE_NAME} Docstring Check"
                            "${OUTPUT_DIR}" "${DOC_ENFORCE_ROOT_DIRECTORY}")

    # Add exclude patterns if specified
    if(DOC_ENFORCE_EXCLUDE_DIRECTORIES OR DOC_ENFORCE_EXCLUDE_PATTERNS)
        set(EXCLUDE_PATTERNS_STR "")
        foreach(exclude_dir ${DOC_ENFORCE_EXCLUDE_DIRECTORIES})
            set(EXCLUDE_PATTERNS_STR "${EXCLUDE_PATTERNS_STR} \"${exclude_dir}/*\"")
        endforeach()
        foreach(exclude_pattern ${DOC_ENFORCE_EXCLUDE_PATTERNS})
            set(EXCLUDE_PATTERNS_STR "${EXCLUDE_PATTERNS_STR} \"${exclude_pattern}\"")
        endforeach()

        set(DOXYGEN_CONFIG_CONTENT "${DOXYGEN_CONFIG_CONTENT}
EXCLUDE_PATTERNS       = ${EXCLUDE_PATTERNS_STR}")
    endif()

    # Write the enforcement Doxyfile
    set(DOXYGEN_CONFIG_FILE "${OUTPUT_DIR}/Doxyfile")
    file(WRITE ${DOXYGEN_CONFIG_FILE} "${DOXYGEN_CONFIG_CONTENT}")

    # Create check script
    set(CHECK_SCRIPT "${OUTPUT_DIR}/check.cmake")
    generate_warnings_check_script(${CHECK_SCRIPT} "${DOC_ENFORCE_NAME}" "${OUTPUT_DIR}"
                                   ${DOC_ENFORCE_FAIL_ON_WARNINGS})

    # Create the docstring check target
    add_custom_target(
        ${DOC_ENFORCE_NAME}_docstring_check
        COMMAND ${CMAKE_COMMAND} -P ${CHECK_SCRIPT}
        COMMENT "Checking docstrings for ${DOC_ENFORCE_NAME}")

    # Create a check target if it doesn't exist
    if(NOT TARGET check_all_docstrings)
        add_custom_target(check_all_docstrings COMMENT "Checking docstrings for all targets")
    endif()

    # Add this target's check to the global check
    add_dependencies(check_all_docstrings ${DOC_ENFORCE_NAME}_docstring_check)

endfunction()

#[=======================================================================[
Build Targets Created:

Individual checks:
  cmake --build <builddir> --target <target_name>_docstring_check
  cmake --build <builddir> --target <name>_docstring_check

All checks:
  cmake --build <builddir> --target check_all_docstrings
#]=======================================================================]

# Function to discover export macros automatically
function(discover_export_macros output_var)
    # Find all *_export.hpp files
    file(GLOB_RECURSE EXPORT_HEADERS "${CMAKE_BINARY_DIR}/*_export.hpp"
         "${CMAKE_SOURCE_DIR}/*_export.hpp")

    set(EXPORT_MACROS "")

    foreach(HEADER ${EXPORT_HEADERS})
        # Extract the base name (e.g., sample_export.hpp -> SAMPLE)
        get_filename_component(BASENAME ${HEADER} NAME_WE)
        string(REPLACE "_export" "" MODULE_NAME ${BASENAME})
        string(TOUPPER ${MODULE_NAME} MODULE_NAME_UPPER)

        # Add the export macro
        list(APPEND EXPORT_MACROS "${MODULE_NAME_UPPER}_EXPORT=")
    endforeach()

    set(${output_var}
        ${EXPORT_MACROS}
        PARENT_SCOPE)
endfunction()

# Helper function to get common Doxygen PREDEFINED macros Returns a list of macro definitions for
# preprocessing
function(get_doxygen_predefined_macros output_var)
    # Auto-discover export macros
    discover_export_macros(DISCOVERED_EXPORT_MACROS)

    # Define common macros that need to be handled during preprocessing - Export macros:
    # Automatically discovered *_EXPORT macros - Compiler attributes: __attribute__, __declspec__ -
    # CUDA macros: CUDA_BOTH_INLINE, CUDA_HOST, CUDA_DEVICE - WISE_ENUM_ADAPT: Hide from Doxygen
    # (used after standard enum declarations)
    set(${output_var}
        ${DISCOVERED_EXPORT_MACROS}
        "__cplusplus"
        "__attribute__(x)="
        "__declspec(x)="
        "WISE_ENUM_ADAPT(...)="
        "CUDA_BOTH_INLINE="
        "CUDA_HOST="
        "CUDA_DEVICE="
        "__global__="
        PARENT_SCOPE)
endfunction()
