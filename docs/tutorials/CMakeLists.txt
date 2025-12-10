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

# Tutorials CMake Configuration Tutorials are part of the docs package and use docs/.venv

include(${CMAKE_SOURCE_DIR}/cmake/Python.cmake)

# Optional: Configure notebook test timeout
set(NOTEBOOK_TEST_TIMEOUT
    "600"
    CACHE STRING "Timeout for notebook execution in seconds")

# Setup docs python package for building documentation Use same EXTRAS as docs since tutorials use
# docs/.venv DOCS_PYTHON_EXTRAS is set in parent docs/CMakeLists.txt
set(pyproject_dir ${CMAKE_SOURCE_DIR}/docs)
add_python_ruff_targets("tutorials" "${pyproject_dir}" EXTRAS ${DOCS_PYTHON_EXTRAS})

# Ensure MLIR-TRT wheels are downloaded before tutorials setup (when enabled)
if(ENABLE_MLIR_TRT)
    add_dependencies(py_tutorials_setup py_ran_mlir_trt_setup)
endif()

# Define notebook conversion command (reused by custom target and CTest)
set(NOTEBOOK_CONVERT_CMD
    uv
    run
    ${CMAKE_SOURCE_DIR}/scripts/setup_python_env.py
    jupytext_convert
    ${CMAKE_SOURCE_DIR}/docs
    --extras
    ${DOCS_PYTHON_EXTRAS})

# Tutorial convert target (for manual use) Passes docs as package_dir (contains pyproject.toml and
# tutorials/ subdirectory)
add_custom_target(
    py_notebook_convert
    COMMAND ${NOTEBOOK_CONVERT_CMD}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Converting tutorials to notebooks"
    DEPENDS py_docs_setup)

# Create docs environment fixture for notebook tests
if(BUILD_TESTING)
    # Step 1: Setup Python environment
    add_test(
        NAME py_docs_env_setup
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target py_docs_setup
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    set_tests_properties(py_docs_env_setup PROPERTIES FIXTURES_SETUP py_docs_env LABELS
                                                      "fixture;docs;tutorial")

    # Step 2: Convert notebooks (depends on environment, sets up notebooks for tests)
    add_test(
        NAME py_notebook_convert_test
        COMMAND ${NOTEBOOK_CONVERT_CMD}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    set_tests_properties(
        py_notebook_convert_test PROPERTIES FIXTURES_REQUIRED py_docs_env FIXTURES_SETUP
                                            py_notebooks_ready LABELS "fixture;docs;tutorial")
endif()

# Helper function to add a tutorial notebook test
function(add_tutorial_notebook_test notebook_name)
    # Parse optional DEPENDS and ENVIRONMENT arguments
    cmake_parse_arguments(ARG "" "" "DEPENDS;ENVIRONMENT" ${ARGN})

    # Build arguments for add_notebook_ctest
    set(notebook_args
        ${notebook_name}
        ${CMAKE_SOURCE_DIR}/docs
        py_docs_setup
        TIMEOUT
        ${NOTEBOOK_TEST_TIMEOUT}
        FIXTURE
        py_notebooks_ready)

    # Add optional environment variables
    if(ARG_ENVIRONMENT)
        list(APPEND notebook_args ENVIRONMENT ${ARG_ENVIRONMENT})
    endif()

    # Add the notebook test
    add_notebook_ctest(${notebook_args})

    # Make notebook convert a dependency of the test target
    add_dependencies(py_notebook_${notebook_name} py_notebook_convert sync_env_python)

    # Setup fixture for this notebook so other notebooks can depend on it
    set_tests_properties(py_notebook_${notebook_name}
                         PROPERTIES FIXTURES_SETUP "notebook_${notebook_name}_complete")

    # Add optional dependencies on other notebook tests using fixtures
    if(ARG_DEPENDS)
        set(required_fixtures "py_notebooks_ready")
        foreach(dep ${ARG_DEPENDS})
            list(APPEND required_fixtures "notebook_${dep}_complete")
        endforeach()
        set_tests_properties(py_notebook_${notebook_name} PROPERTIES FIXTURES_REQUIRED
                                                                     "${required_fixtures}")
    endif()
endfunction()

# cmake-format: off
# ==============================================================================
# Notebook Test Dependency Graph (TRT Engine Race Condition Prevention)
# ==============================================================================
#
#                          getting_started (Wave 1)
#                                   |
#          +------------------------+------------------------+
#          |                        |                        |
#    fronthaul_       ai_tukey_filter_training_      mlir_trt_tutorial
#    tutorial              tutorial                     (Wave 2)
#    (Wave 2)              (Wave 2)
#          |                        |
#          |          +-------------+
#          |          |             |
#          |   pusch_receiver_  pusch_channel_estimation_
#          |   lowering_tut.    lowering_tutorial
#          |     (Wave 3)          (Wave 3)
#          |          |             |
#          +----------+-------------+
#                     |
#                     v
#            pipeline_tutorial ← Uses TRT engines from lowering tutorials
#                (Wave 4)
#                     |
#                     v
#            phy_ran_app_tutorial
#                (Wave 5)
#
# KEY INSIGHT: Lowering tutorials generate TRT engines that pipeline_tutorial
#              consumes. Running lowering tutorials first ensures all engines
#              are ready before C++ tests execute.
#
# Wave 1: Foundation (1 test)
# Wave 2: Basic tutorials + MLIR intro (3 tests in parallel)
# Wave 3: TRT engine generation (2 tests in parallel)
# Wave 4: Pipeline tutorial with C++ tests (uses engines from Wave 3)
# Wave 5: PHY RAN App integration test
# ==============================================================================
# cmake-format: on

# Individual notebook tests as CTests Getting started is the foundation notebook
add_tutorial_notebook_test("getting_started")

# Other notebooks depend on getting_started
add_tutorial_notebook_test("fronthaul_tutorial" DEPENDS getting_started)
add_tutorial_notebook_test("pusch_receiver_tutorial" DEPENDS getting_started)

# MLIR-TRT lowering tutorials
if(ENABLE_MLIR_TRT)
    add_tutorial_notebook_test("ai_tukey_filter_training_tutorial" DEPENDS getting_started)
    add_tutorial_notebook_test("mlir_trt_tutorial" DEPENDS getting_started)
    # pusch_receiver_lowering_tutorial generates TRT engines needed by pipeline_tutorial
    add_tutorial_notebook_test("pusch_receiver_lowering_tutorial" DEPENDS getting_started)
    # Channel estimation lowering depends on AI Tukey filter model being trained first
    add_tutorial_notebook_test("pusch_channel_estimation_lowering_tutorial" DEPENDS
                               "ai_tukey_filter_training_tutorial")
    # pipeline_tutorial depends on TRT engines from lowering tutorials
    set(PIPELINE_DEPS "fronthaul_tutorial;pusch_receiver_lowering_tutorial")
    set(PIPELINE_DEPS "${PIPELINE_DEPS};pusch_channel_estimation_lowering_tutorial")
    add_tutorial_notebook_test("pipeline_tutorial" DEPENDS "${PIPELINE_DEPS}")
    # phy_ran_app_tutorial runs after pipeline_tutorial
    add_tutorial_notebook_test("phy_ran_app_tutorial" DEPENDS pipeline_tutorial)
else()
    # Without MLIR-TRT, pipeline_tutorial runs directly after fronthaul
    add_tutorial_notebook_test("pipeline_tutorial" DEPENDS fronthaul_tutorial)
    add_tutorial_notebook_test("phy_ran_app_tutorial" DEPENDS pipeline_tutorial)
endif()
