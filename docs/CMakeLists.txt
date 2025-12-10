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

include(${CMAKE_SOURCE_DIR}/cmake/Python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Doxygen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Sphinx.cmake)

# If documentation building is enabled, set up the full documentation pipeline
if(BUILD_DOCS)

    # Setup docs python package for building documentation
    if(ENABLE_MLIR_TRT)
        set(DOCS_PYTHON_EXTRAS "dev" "ran_mlir_trt")
    else()
        set(DOCS_PYTHON_EXTRAS "dev" "ran_base")
    endif()

    add_python_ruff_targets("docs" ${CMAKE_CURRENT_SOURCE_DIR} EXTRAS ${DOCS_PYTHON_EXTRAS})

    # Add doc8 target specifically for docs package (contains Sphinx documentation)
    add_python_doc8_target(py_docs_doc8 ${CMAKE_CURRENT_SOURCE_DIR} py_docs_setup)

    # Ensure MLIR-TRT wheels are downloaded before docs setup (when enabled)
    if(ENABLE_MLIR_TRT)
        add_dependencies(py_docs_setup py_ran_mlir_trt_setup)
    endif()

    # Must be after python package targets are setup
    add_subdirectory(tutorials)

    # py_docs_all = all checks/builds including notebook conversion
    add_dependencies(py_docs_all py_docs_doc8 py_notebook_convert py_docs_samples)
endif()

# Docstring checking - create targets whether enabled or disabled
if(ENFORCE_DOCSTRINGS)
    enforce_docstrings(
        NAME
        framework
        ROOT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/framework/
        EXCLUDE_DIRECTORIES
        ${CMAKE_BUILD_DIR}/framework/
        EXCLUDE_PATTERNS
        "*/test/*"
        "*/tests/*"
        "*_test.cpp"
        FAIL_ON_WARNINGS)
    enforce_docstrings(
        NAME
        ran_runtime
        ROOT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/ran/runtime/
        EXCLUDE_DIRECTORIES
        ${CMAKE_BUILD_DIR}/ran/runtime/
        FAIL_ON_WARNINGS)
    enforce_docstrings(
        NAME
        ran_py
        ROOT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/ran/py/
        EXCLUDE_DIRECTORIES
        ${CMAKE_BUILD_DIR}/ran/py/
        FAIL_ON_WARNINGS)
else()
    # Create stub targets that warn when ENFORCE_DOCSTRINGS is disabled
    add_custom_target(
        framework_docstring_check
        COMMAND ${CMAKE_COMMAND} -E echo "WARNING: ENFORCE_DOCSTRINGS is OFF. "
                "Enable with -DENFORCE_DOCSTRINGS=ON to check docstrings."
        COMMENT "Docstring checking is disabled")

    add_custom_target(
        ran_runtime_docstring_check
        COMMAND ${CMAKE_COMMAND} -E echo "WARNING: ENFORCE_DOCSTRINGS is OFF. "
                "Enable with -DENFORCE_DOCSTRINGS=ON to check docstrings."
        COMMENT "Docstring checking is disabled")

    add_custom_target(
        ran_py_docstring_check
        COMMAND ${CMAKE_COMMAND} -E echo "WARNING: ENFORCE_DOCSTRINGS is OFF. "
                "Enable with -DENFORCE_DOCSTRINGS=ON to check docstrings."
        COMMENT "Docstring checking is disabled")

    # Create the global check_all_docstrings target
    add_custom_target(
        check_all_docstrings
        COMMAND ${CMAKE_COMMAND} -E echo "WARNING: ENFORCE_DOCSTRINGS is OFF. "
                "Enable with -DENFORCE_DOCSTRINGS=ON to check docstrings."
        COMMENT "Docstring checking is disabled"
        DEPENDS framework_docstring_check ran_runtime_docstring_check ran_py_docstring_check)
endif()

# Setup Sphinx docs - create targets whether enabled or disabled
if(BUILD_DOCS)
    setup_sphinx_docs()
else()
    # Create stub documentation targets that warn when BUILD_DOCS is disabled
    add_custom_target(
        doxygen-docs
        COMMAND ${CMAKE_COMMAND} -E echo
                "WARNING: BUILD_DOCS is OFF. Enable with -DBUILD_DOCS=ON to build documentation."
        COMMENT "Documentation building is disabled")

    add_custom_target(
        docs
        COMMAND ${CMAKE_COMMAND} -E echo
                "WARNING: BUILD_DOCS is OFF. Enable with -DBUILD_DOCS=ON to build documentation."
        COMMENT "Documentation building is disabled")
endif()
