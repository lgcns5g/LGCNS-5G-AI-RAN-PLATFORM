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

# Sphinx documentation functions

# Find required tools
find_package(Doxygen REQUIRED)

# Function to set up Sphinx documentation
function(setup_sphinx_docs)
    # Set up directories
    set(SPHINX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/docs)
    set(SPHINX_BUILD_DIR ${CMAKE_BINARY_DIR}/docs/sphinx)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs/doxygen)

    # Include existing Doxygen configuration and enable it This will now properly configure XML
    # output for Breathe
    include(${CMAKE_SOURCE_DIR}/cmake/Doxygen.cmake)
    enable_doxygen("awesome-sidebar")

    # Add target to verify documentation samples
    add_custom_target(
        py_docs_samples
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/verify_docs_samples.py docs/api
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Verifying API documentation code samples"
        VERBATIM)

    # Sphinx HTML build command
    set(SPHINX_DOCS_COMMAND
        ${CMAKE_COMMAND} -E env uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py sphinx_docs
        docs --build-dir "${CMAKE_BINARY_DIR}")

    # Sphinx linkcheck command
    set(SPHINX_LINKCHECK_COMMAND
        ${CMAKE_COMMAND} -E env uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py
        sphinx_linkcheck docs --build-dir "${CMAKE_BINARY_DIR}")

    # Sphinx HTML documentation target
    add_custom_target(
        sphinx-docs
        COMMAND ${SPHINX_DOCS_COMMAND}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS doxygen-docs py_docs_all
        COMMENT "Building Sphinx HTML documentation"
        VERBATIM)

    # Sphinx linkcheck target
    add_custom_target(
        sphinx-linkcheck
        COMMAND ${SPHINX_LINKCHECK_COMMAND}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS doxygen-docs py_docs_all
        COMMENT "Checking external links in documentation"
        VERBATIM)

    # Full docs target depends on both sphinx-docs and sphinx-linkcheck (can run in parallel)
    add_custom_target(
        docs
        DEPENDS sphinx-docs sphinx-linkcheck
        COMMENT "Building all documentation (HTML + linkcheck)"
        VERBATIM)

    # Print helpful information
    # cmake-format: off
    message(STATUS "Documentation targets:")
    message(STATUS "  py_docs_setup    - Install Python dependencies (including Sphinx)")
    message(STATUS "  py_docs_all      - Run all docs package checks (test, lint, format, doc8)")
    message(STATUS "  py_docs_doc8     - Run doc8 documentation linter (Sphinx docs)")
    message(STATUS "  py_docs_samples  - Verify API documentation code samples")
    message(STATUS "  doxygen-docs     - Build C++ API documentation")
    message(STATUS "  sphinx-docs      - Build Sphinx HTML documentation")
    message(STATUS "  sphinx-linkcheck - Check external links in documentation")
    message(STATUS "  docs             - Build all documentation (sphinx-docs + sphinx-linkcheck)")
    message(STATUS "")
    message(STATUS "Sphinx output will be in: ${SPHINX_BUILD_DIR}/index.html")
    message(STATUS "Doxygen output will be in: ${DOXYGEN_OUTPUT_DIR}/html/index.html")
    message(STATUS "Doxygen XML for Breathe: ${DOXYGEN_OUTPUT_DIR}/xml/index.xml")
    # cmake-format: on
endfunction()
