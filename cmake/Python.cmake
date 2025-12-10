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

# cmake/Python.cmake - Generic Python development functions

# Function to add a Python setup target
function(add_python_setup_target prefix package_dir)
    # Parse optional EXTRAS argument
    cmake_parse_arguments(ARG "" "" "EXTRAS" ${ARGN})

    # Create the aggregate target if it doesn't exist yet
    if(NOT TARGET py_all_setup)
        add_custom_target(py_all_setup COMMENT "Setting up Python environments for all packages")
    endif()

    # Build command with optional extras
    set(setup_cmd uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py setup ${package_dir})
    if(ARG_EXTRAS)
        list(APPEND setup_cmd --extras ${ARG_EXTRAS})
    endif()

    add_custom_target(
        ${prefix}_setup
        COMMAND ${setup_cmd}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT
            "Setting up Python environment for ${package_dir} (first time may take a few minutes...)"
    )

    # Add to aggregate target
    add_dependencies(py_all_setup ${prefix}_setup)

    # Create corresponding clean venv target
    add_python_venv_clean_target(${prefix}_clean_venv ${package_dir})
endfunction()

# Function to add a Python test target
function(add_python_test_target target_name package_dir setup_target)
    # Parse optional EXTRAS argument
    cmake_parse_arguments(ARG "" "" "EXTRAS" ${ARGN})

    # Build test command with optional extras
    set(test_cmd
        uv
        run
        ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py
        test
        ${package_dir}
        --build-dir
        ${CMAKE_CURRENT_BINARY_DIR})
    if(ARG_EXTRAS)
        list(APPEND test_cmd --extras ${ARG_EXTRAS})
    endif()

    # Create custom target using the same command
    add_custom_target(
        ${target_name}
        COMMAND ${test_cmd}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running Python tests for ${package_dir}"
        DEPENDS ${setup_target})

    # Register with CTest for unified test reporting
    if(BUILD_TESTING)
        add_test(
            NAME ${target_name}
            COMMAND ${test_cmd}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endfunction()

# Function to add a Python code quality lint target
function(add_python_lint_target target_name package_dir setup_target)
    add_custom_target(
        ${target_name}
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py lint ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running Python code quality checks for ${package_dir}"
        DEPENDS ${setup_target})
endfunction()

# Function to add a Python code formatting target
function(add_python_fix_format_target target_name package_dir setup_target)
    # Create the aggregate target if it doesn't exist yet
    if(NOT TARGET py_all_fix_format)
        add_custom_target(py_all_fix_format COMMENT "Formatting Python code for all packages")
    endif()

    # Create the individual fix format target
    add_custom_target(
        ${target_name}
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py fix_format ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatting Python code for ${package_dir}"
        DEPENDS ${setup_target})

    # Add this target to the aggregate target
    add_dependencies(py_all_fix_format ${target_name})
endfunction()

# Function to add a Python wheel build target
function(add_python_wheel_build_target target_name package_dir setup_target)
    add_custom_target(
        ${target_name}
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py wheel_build ${package_dir}
                --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Building Python wheel for ${package_dir}"
        DEPENDS ${setup_target})
endfunction()

# Function to add a Python wheel install target
function(add_python_wheel_install_target target_name package_dir wheel_build_target)
    add_custom_target(
        ${target_name}
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py wheel_install ${package_dir}
                --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Installing Python wheel for ${package_dir}"
        DEPENDS ${wheel_build_target})
endfunction()

# Function to add a Python wheel test target
function(add_python_wheel_test_target target_name package_dir wheel_install_target)
    add_custom_target(
        ${target_name}
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py wheel_test ${package_dir}
                --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Testing Python wheel for ${package_dir}"
        DEPENDS ${wheel_install_target})
endfunction()

# Function to add a Python venv clean target
function(add_python_venv_clean_target target_name package_dir)
    # Create the aggregate target if it doesn't exist yet
    if(NOT TARGET py_all_clean_venv)
        add_custom_target(py_all_clean_venv
                          COMMENT "Cleaning Python virtual environments for all packages")
    endif()

    # Check if we've already created a clean target for this directory This may be the case for
    # multiple cmake targets using the same package directory
    get_filename_component(package_dir_abs "${package_dir}" ABSOLUTE)
    get_property(
        cleaned_dirs
        TARGET py_all_clean_venv
        PROPERTY CLEANED_DIRS)

    if(NOT package_dir_abs IN_LIST cleaned_dirs)
        # First time seeing this directory - create the actual clean command
        add_custom_target(
            ${target_name}
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${package_dir}/.venv"
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Removing Python virtual environment from ${package_dir}")
        add_dependencies(py_all_clean_venv ${target_name})

        # Track this directory on the aggregate target
        list(APPEND cleaned_dirs ${package_dir_abs})
        set_property(TARGET py_all_clean_venv PROPERTY CLEANED_DIRS "${cleaned_dirs}")
    endif()
endfunction()

# Function to add a Python format check target
function(add_python_check_format_target target_name package_dir setup_target)
    # Create the aggregate target if it doesn't exist yet
    if(NOT TARGET py_all_check_format)
        add_custom_target(py_all_check_format
                          COMMENT "Checking Python code formatting for all packages")
    endif()

    # Create the individual check format target
    add_custom_target(
        ${target_name}
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py check_format ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${setup_target}
        COMMENT "Checking Python code formatting for ${package_dir}")

    # Add this target to the aggregate target
    add_dependencies(py_all_check_format ${target_name})
endfunction()

# Function to add a Python doc8 target (for docs package only)
function(add_python_doc8_target target_name package_dir setup_target)
    add_custom_target(
        ${target_name}
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py doc8 ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running doc8 documentation linter for ${package_dir}"
        DEPENDS ${setup_target})
endfunction()

# Function to add individual Python linting targets
function(add_python_individual_lint_targets package_name package_dir setup_target)
    set(prefix "py_${package_name}")

    # Create aggregate targets if they don't exist yet
    if(NOT TARGET py_all_mypy)
        add_custom_target(py_all_mypy COMMENT "Running mypy for all packages")
    endif()
    if(NOT TARGET py_all_ruff_check)
        add_custom_target(py_all_ruff_check COMMENT "Running ruff check for all packages")
    endif()
    if(NOT TARGET py_all_ruff_fix)
        add_custom_target(py_all_ruff_fix COMMENT "Running ruff fix for all packages")
    endif()
    if(NOT TARGET py_all_lint)
        add_custom_target(py_all_lint COMMENT "Running lint checks for all packages")
    endif()
    if(NOT TARGET py_all_fix_lint)
        add_custom_target(py_all_fix_lint COMMENT "Running lint fixes for all packages")
    endif()

    # Individual linting targets
    add_custom_target(
        ${prefix}_black
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py black ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running black on ${package_dir}"
        DEPENDS ${setup_target})

    add_custom_target(
        ${prefix}_isort
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py isort ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running isort on ${package_dir}"
        DEPENDS ${setup_target})

    add_custom_target(
        ${prefix}_flake8
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py flake8 ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running flake8 on ${package_dir}"
        DEPENDS ${setup_target})

    add_custom_target(
        ${prefix}_pylint
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py pylint ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running pylint on ${package_dir}"
        DEPENDS ${setup_target})

    # mypy target
    add_custom_target(
        ${prefix}_mypy
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py mypy ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running mypy on ${package_dir}"
        DEPENDS ${setup_target})
    add_dependencies(py_all_mypy ${prefix}_mypy)

    # Ruff targets
    add_custom_target(
        ${prefix}_ruff_check
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py ruff_check ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running ruff check on ${package_dir}"
        DEPENDS ${setup_target})
    add_dependencies(py_all_ruff_check ${prefix}_ruff_check)

    add_custom_target(
        ${prefix}_ruff_fix
        COMMAND uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py ruff_fix ${package_dir}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running ruff fix on ${package_dir}"
        DEPENDS ${setup_target})
    add_dependencies(py_all_ruff_fix ${prefix}_ruff_fix)

    # Aggregate lint target (ruff_check + mypy)
    add_custom_target(
        ${prefix}_lint
        DEPENDS ${prefix}_ruff_check ${prefix}_mypy
        COMMENT "Running lint checks for ${package_dir}")
    add_dependencies(py_all_lint ${prefix}_lint)

    # Aggregate fix lint target (ruff_fix + mypy)
    add_custom_target(
        ${prefix}_fix_lint
        DEPENDS ${prefix}_ruff_fix ${prefix}_mypy
        COMMENT "Running lint fixes for ${package_dir}")
    add_dependencies(py_all_fix_lint ${prefix}_fix_lint)
endfunction()

# Option to enable MLIR-TensorRT
option(ENABLE_MLIR_TRT "Enable MLIR-TensorRT with MLIR-TensorRT" ON)

# cmake-format: off
# Function to setup MLIR-TensorRT: downloads wheels AND tarball, creates compiler target
# Downloads occur at BUILD TIME (via add_custom_target), not during CMake configuration.
# The target_name parameter creates a buildable CMake target that performs the setup.
# Additionally creates the imported executable target 'mlir-tensorrt::compiler'.
# Note: Wheels are downloaded but NOT installed. Use py_ran_setup to install them.
# All version configuration is managed in scripts/setup_python_env.py.
# cmake-format: on
function(add_mlir_trt_setup_target target_name)
    # Check if already configured
    if(TARGET mlir-tensorrt::compiler)
        message(STATUS "MLIR-TensorRT compiler target already exists")
        return()
    endif()

    # Extract to build dir (per-build isolation, fast to re-extract if needed)
    set(MLIR_TARBALL_EXTRACT_DIR "${CMAKE_BINARY_DIR}/_deps")

    # Construct expected compiler path (Python script normalizes to mlir-tensorrt/)
    set(MLIR_TENSORRT_COMPILER_EXECUTABLE
        "${MLIR_TARBALL_EXTRACT_DIR}/mlir-tensorrt/bin/mlir-tensorrt-compiler")

    # Build minimal command - all versions come from Python defaults
    set(SETUP_CMD uv run ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py setup_mlir_trt
                  --mlir-tarball-extract-dir ${MLIR_TARBALL_EXTRACT_DIR})

    # Create custom target to run setup at build time
    add_custom_target(
        ${target_name}
        COMMAND ${SETUP_CMD}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Setting up MLIR-TensorRT (versions from Python defaults)"
        BYPRODUCTS ${MLIR_TENSORRT_COMPILER_EXECUTABLE})

    message(STATUS "MLIR-TensorRT setup target created: ${target_name}")
    message(STATUS "  Expected compiler: ${MLIR_TENSORRT_COMPILER_EXECUTABLE}")
    message(STATUS "  Note: Wheels will be downloaded. Run py_ran_setup to install them.")

    # Create imported executable target (globally visible)
    add_executable(mlir-tensorrt::compiler IMPORTED GLOBAL)
    set_target_properties(mlir-tensorrt::compiler PROPERTIES IMPORTED_LOCATION
                                                             ${MLIR_TENSORRT_COMPILER_EXECUTABLE})

    # Derive root directory from compiler path
    get_filename_component(MLIR_TENSORRT_BIN_DIR "${MLIR_TENSORRT_COMPILER_EXECUTABLE}" DIRECTORY)

    message(STATUS "MLIR-TensorRT configuration complete")
    message(STATUS "  Compiler directory: ${MLIR_TENSORRT_BIN_DIR}")
endfunction()

# Function to add lightweight Python ruff targets (for tools/samples/scripts)
function(add_python_ruff_targets package_name package_dir)
    # Parse optional EXTRAS argument
    cmake_parse_arguments(ARG "" "" "EXTRAS" ${ARGN})

    set(prefix "py_${package_name}")

    # Setup target - foundation for all other targets
    if(ARG_EXTRAS)
        add_python_setup_target(${prefix} ${package_dir} EXTRAS ${ARG_EXTRAS})
    else()
        add_python_setup_target(${prefix} ${package_dir})
    endif()

    # Ruff linting and formatting targets
    add_python_individual_lint_targets(${package_name} ${package_dir} ${prefix}_setup)
    add_python_fix_format_target(${prefix}_fix_format ${package_dir} ${prefix}_setup)
    add_python_check_format_target(${prefix}_check_format ${package_dir} ${prefix}_setup)

    # All-in-one target
    add_custom_target(
        ${prefix}_all
        DEPENDS ${prefix}_setup ${prefix}_ruff_check ${prefix}_ruff_fix ${prefix}_fix_format
                ${prefix}_check_format
        COMMENT "Running all ruff checks for ${package_dir}")
endfunction()

# cmake-format: off
# Function to add a notebook test as a CTest
# Usage: add_notebook_ctest("getting_started" ${CMAKE_SOURCE_DIR}/docs ${setup_target} 
#                          FIXTURE py_docs_env ENVIRONMENT "VAR1=value1" "VAR2=value2")
function(add_notebook_ctest notebook_name package_dir setup_target)
    # Parse optional TIMEOUT, FIXTURE, and ENVIRONMENT arguments
    cmake_parse_arguments(ARG "" "TIMEOUT;FIXTURE" "ENVIRONMENT" ${ARGN})
    
    # Set default timeout if not provided
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 300)
    endif()
    
    # Create the aggregate target if it doesn't exist yet
    if(NOT TARGET py_all_notebook_test)
        add_custom_target(py_all_notebook_test COMMENT "Running all notebook tests")
    endif()
    
    # Build the test command using setup_python_env.py notebook_test
    set(test_cmd
        ${CMAKE_COMMAND}
        -E
        env
        RAN_ENV_PYTHON_FILE=${CMAKE_BINARY_DIR}/ran/py/.env.python)
    
    # Add optional environment variables
    if(ARG_ENVIRONMENT)
        list(APPEND test_cmd ${ARG_ENVIRONMENT})
    endif()
    
    list(APPEND test_cmd
        uv
        run
        ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py
        notebook_test
        ${package_dir}
        --notebook-name
        ${notebook_name}
        --notebook-timeout
        ${ARG_TIMEOUT})

    # Create custom target using the same command
    add_custom_target(
        py_notebook_${notebook_name}
        COMMAND ${test_cmd}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Testing ${notebook_name} notebook"
        DEPENDS ${setup_target})
    
    # Add to aggregate target
    add_dependencies(py_all_notebook_test py_notebook_${notebook_name})

    # Register with CTest for unified test reporting
    if(BUILD_TESTING)
        add_test(
            NAME py_notebook_${notebook_name}
            COMMAND ${test_cmd}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        
        # Set test properties (base properties)
        if(ARG_FIXTURE)
            set_tests_properties(
                py_notebook_${notebook_name}
                PROPERTIES TIMEOUT ${ARG_TIMEOUT} LABELS "notebook"
                FIXTURES_REQUIRED ${ARG_FIXTURE})
        else()
            set_tests_properties(
                py_notebook_${notebook_name}
                PROPERTIES TIMEOUT ${ARG_TIMEOUT} LABELS "notebook")
        endif()
    endif()
endfunction()

# Function to add a comprehensive Python package target suite
#
# Usage: add_python_package_targets("mypackage" ${CMAKE_CURRENT_SOURCE_DIR} [EXTRAS "dev" "mlir_trt_wheels"])
# cmake-format: on
function(add_python_package_targets package_name package_dir)
    # Parse optional EXTRAS argument
    cmake_parse_arguments(ARG "" "" "EXTRAS" ${ARGN})

    set(prefix "py_${package_name}")

    # Setup target - foundation for all other targets
    if(ARG_EXTRAS)
        add_python_setup_target(${prefix} ${package_dir} EXTRAS ${ARG_EXTRAS})
    else()
        add_python_setup_target(${prefix} ${package_dir})
    endif()

    # Development targets
    if(ARG_EXTRAS)
        add_python_test_target(${prefix}_test ${package_dir} ${prefix}_setup EXTRAS ${ARG_EXTRAS})
    else()
        add_python_test_target(${prefix}_test ${package_dir} ${prefix}_setup)
    endif()
    add_python_individual_lint_targets(${package_name} ${package_dir} ${prefix}_setup)
    add_python_fix_format_target(${prefix}_fix_format ${package_dir} ${prefix}_setup)
    add_python_check_format_target(${prefix}_check_format ${package_dir} ${prefix}_setup)

    # Wheel targets
    add_python_wheel_build_target(${prefix}_wheel_build ${package_dir} ${prefix}_setup)
    add_python_wheel_install_target(${prefix}_wheel_install ${package_dir} ${prefix}_wheel_build)
    add_python_wheel_test_target(${prefix}_wheel_test ${package_dir} ${prefix}_wheel_install)

    # All-in-one target - depends on individual CMake targets for parallelism
    add_custom_target(
        ${prefix}_all
        DEPENDS ${prefix}_setup ${prefix}_test ${prefix}_lint ${prefix}_fix_format
                ${prefix}_check_format ${prefix}_wheel_test
        COMMENT "Running all Python operations for ${package_dir}")
endfunction()
