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

# Ensure mlir-trt-downloads directory exists in source tree Required for: uv find-links validation
# during wheel builds
set(MLIR_TRT_DOWNLOADS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mlir-trt-downloads)
if(NOT EXISTS ${MLIR_TRT_DOWNLOADS_DIR})
    file(MAKE_DIRECTORY ${MLIR_TRT_DOWNLOADS_DIR})
endif()

# ------------------------------------------------------------------------------
# PHY - Setup and Configuration
# ------------------------------------------------------------------------------

if(ENABLE_MLIR_TRT)
    # Setup MLIR-TensorRT compiler (downloads wheels + tarball, doesn't install)
    add_mlir_trt_setup_target("py_ran_mlir_trt_setup" MLIR_DOWNLOAD_DIR ${MLIR_TRT_DOWNLOADS_DIR})
    message(STATUS "MLIR TRT enabled - building TensorRT plugins")

    find_package(CUDAToolkit REQUIRED)
    set(PLUGIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ran/trt_plugins)
    set(PLUGIN_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/trt_plugins)
    set(MLIR_TRT_WRAPPER_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir_trt_wrapper)
    set(PHY_JAX_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/phy/jax)

    # TensorRT engine directory for tests
    set(RAN_TRT_ENGINE_PATH ${CMAKE_CURRENT_BINARY_DIR}/trt_engines)
    # Build directory for Python tests
    set(RAN_PYTEST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

    # Custom target to clean all TRT engine files
    add_custom_target(
        py_ran_clean_trt_engines
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.trtengine" -type f -delete
        COMMENT "Removing all *.trtengine files from ${CMAKE_BINARY_DIR}")
endif()

# ------------------------------------------------------------------------------
# PHY - Plugin Base Classes
# ------------------------------------------------------------------------------

if(ENABLE_MLIR_TRT)
    # TensorRT Plugin Base Classes - lightweight static library
    add_library(trt_plugin_base STATIC)

    set(TRT_PLUGIN_BASE_DIR ${PLUGIN_SRC_DIR}/base)

    target_sources(
        trt_plugin_base
        PRIVATE ${TRT_PLUGIN_BASE_DIR}/trt_plugin_base.cpp
        PUBLIC FILE_SET
               HEADERS
               BASE_DIRS
               ${TRT_PLUGIN_BASE_DIR}
               FILES
               ${TRT_PLUGIN_BASE_DIR}/trt_plugin_base.hpp
               ${TRT_PLUGIN_BASE_DIR}/trt_plugin_creator_base.hpp)

    target_include_directories(trt_plugin_base PUBLIC $<BUILD_INTERFACE:${TRT_PLUGIN_BASE_DIR}>)

    target_link_system_libraries(trt_plugin_base PUBLIC CUDA::cudart)
    target_link_nvinfer(trt_plugin_base PUBLIC)
endif()

# ------------------------------------------------------------------------------
# PHY - Test Infrastructure
# ------------------------------------------------------------------------------

if(ENABLE_MLIR_TRT)
    # Test utilities library
    add_library(trt_test_utils STATIC)

    set(TRT_UTILS_DIR ${PLUGIN_TEST_DIR}/trt_utils)

    target_sources(
        trt_test_utils
        PRIVATE ${TRT_UTILS_DIR}/trt_test_utils.cpp
        PUBLIC FILE_SET
               HEADERS
               BASE_DIRS
               ${TRT_UTILS_DIR}
               FILES
               ${TRT_UTILS_DIR}/trt_test_utils.hpp
               ${TRT_UTILS_DIR}/trt_test_utils_impl.hpp)

    target_include_directories(trt_test_utils PUBLIC $<BUILD_INTERFACE:${TRT_UTILS_DIR}>)

    target_link_nvinfer(trt_test_utils PUBLIC)
    target_link_system_libraries(trt_test_utils PUBLIC CUDA::cudart)

    # Function to add PHY Python test targets with environment fixture
    function(add_phy_python_test target_name test_dir)
        # Use same extras as main ran package
        add_python_test_target(${target_name} ${test_dir} py_ran_setup EXTRAS ${RAN_PYTHON_EXTRAS})
        # Priority 1: Pass build-specific .env.python path via RAN_ENV_PYTHON_FILE Priority 2:
        # Fixture syncs to source dir as fallback
        set_tests_properties(
            ${target_name}
            PROPERTIES ENVIRONMENT "RAN_ENV_PYTHON_FILE=${CMAKE_CURRENT_BINARY_DIR}/.env.python"
                       FIXTURES_REQUIRED "ran_env")
    endfunction()

    # Function to add PHY tests with common configuration Usage: add_phy_test(TEST_NAME <name>
    # SOURCE_FILES <file1> [<file2> ...] FIXTURE_NAME <fixture> [GTEST_FILTERS <filter1> [<filter2>
    # ...]])
    function(add_phy_test)
        cmake_parse_arguments(ARG "" "TEST_NAME;FIXTURE_NAME" "SOURCE_FILES;GTEST_FILTERS" ${ARGN})

        if(BUILD_TESTING)
            # Create aggregated target if it doesn't exist
            if(NOT TARGET phy_tests_all)
                add_custom_target(phy_tests_all COMMENT "Building all PHY tests")
            endif()

            add_executable(${ARG_TEST_NAME} ${ARG_SOURCE_FILES})
            target_link_libraries(${ARG_TEST_NAME} PRIVATE ran_trt_plugin trt_test_utils)
            target_link_system_libraries(${ARG_TEST_NAME} PRIVATE GTest::gtest_main)

            # Add this test to the aggregated target
            add_dependencies(phy_tests_all ${ARG_TEST_NAME})

            # If no filters specified, use placeholder to ensure foreach iterates once (empty string
            # doesn't work - foreach skips empty lists)
            if(NOT ARG_GTEST_FILTERS)
                set(ARG_GTEST_FILTERS "__NO_FILTER__")
            endif()

            foreach(filter IN LISTS ARG_GTEST_FILTERS)
                # Generate test name with optional suffix
                if(filter AND NOT filter STREQUAL "__NO_FILTER__")
                    set(test_suffix ".${filter}")
                    set(filter_arg "--gtest_filter=*${filter}*")
                    set(xml_suffix "_${filter}")
                else()
                    set(test_suffix "")
                    set(filter_arg "")
                    set(xml_suffix "")
                endif()

                # Register test
                add_test(
                    NAME ran.phy_test.${ARG_TEST_NAME}${test_suffix}
                    COMMAND $<TARGET_FILE:${ARG_TEST_NAME}> ${filter_arg}
                            --gtest_output=xml:${ARG_TEST_NAME}${xml_suffix}.xml
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
                set_tests_properties(
                    ran.phy_test.${ARG_TEST_NAME}${test_suffix}
                    PROPERTIES LABELS "phy;parallel" ENVIRONMENT
                               "RAN_TRT_ENGINE_PATH=${RAN_TRT_ENGINE_PATH}")
                if(ARG_FIXTURE_NAME)
                    set_tests_properties(ran.phy_test.${ARG_TEST_NAME}${test_suffix}
                                         PROPERTIES FIXTURES_REQUIRED ${ARG_FIXTURE_NAME})
                endif()
            endforeach()
        endif()
    endfunction()

    # Function to add PHY benchmarks with common configuration Usage: add_phy_bench(BENCH_NAME
    # <name> SOURCE_FILES <file1> [<file2> ...] FIXTURE_NAME <fixture> [BENCHMARK_FILTERS <filter1>
    # [<filter2> ...]])
    function(add_phy_bench)
        cmake_parse_arguments(ARG "" "BENCH_NAME;FIXTURE_NAME" "SOURCE_FILES;BENCHMARK_FILTERS"
                              ${ARGN})

        if(BUILD_TESTING)
            # Create aggregated build target if it doesn't exist
            if(NOT TARGET phy_bench_all)
                add_custom_target(phy_bench_all COMMENT "Building all PHY benchmarks")
            endif()

            # Create output directories for benchmark and profiling results
            file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark_results)
            file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/nsys_results)

            # Create benchmark executable
            add_executable(${ARG_BENCH_NAME} ${ARG_SOURCE_FILES})
            target_link_libraries(${ARG_BENCH_NAME} PRIVATE ran_trt_plugin trt_test_utils)
            target_link_system_libraries(${ARG_BENCH_NAME} PRIVATE benchmark::benchmark)

            # Add to aggregated build target
            add_dependencies(phy_bench_all ${ARG_BENCH_NAME})

            # If no filters specified, use placeholder to ensure foreach iterates once (empty string
            # doesn't work - foreach skips empty lists)
            if(NOT ARG_BENCHMARK_FILTERS)
                set(ARG_BENCHMARK_FILTERS "__NO_FILTER__")
            endif()

            foreach(filter IN LISTS ARG_BENCHMARK_FILTERS)
                # Generate test name with optional suffix
                if(filter AND NOT filter STREQUAL "__NO_FILTER__")
                    set(test_suffix ".${filter}")
                    set(filter_arg "--benchmark_filter=.*${filter}.*")
                    set(output_suffix "_${filter}")
                else()
                    set(test_suffix "")
                    set(filter_arg "")
                    set(output_suffix "")
                endif()

                # Register benchmark run
                add_test(
                    NAME ran.phy_bench.${ARG_BENCH_NAME}${test_suffix}
                    COMMAND
                        ${CMAKE_COMMAND} -E env RAN_TRT_ENGINE_PATH=${RAN_TRT_ENGINE_PATH}
                        $<TARGET_FILE:${ARG_BENCH_NAME}> ${filter_arg} --benchmark_format=console
                        --benchmark_out_format=json
                        --benchmark_out=${CMAKE_BINARY_DIR}/benchmark_results/${ARG_BENCH_NAME}${output_suffix}_results.json
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
                set_tests_properties(ran.phy_bench.${ARG_BENCH_NAME}${test_suffix}
                                     PROPERTIES LABELS "phy;benchmark;parallel")
                if(ARG_FIXTURE_NAME)
                    set_tests_properties(ran.phy_bench.${ARG_BENCH_NAME}${test_suffix}
                                         PROPERTIES FIXTURES_REQUIRED ${ARG_FIXTURE_NAME})
                endif()

                # Register nsys profiling as CTest test
                add_test(
                    NAME ran.phy_nsys.${ARG_BENCH_NAME}${test_suffix}
                    COMMAND
                        ${CMAKE_COMMAND} -E env RAN_TRT_ENGINE_PATH=${RAN_TRT_ENGINE_PATH} nsys
                        profile --trace=cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage=true
                        --stats=true
                        --output=${CMAKE_BINARY_DIR}/nsys_results/${ARG_BENCH_NAME}${output_suffix}.nsys-rep
                        --force-overwrite=true $<TARGET_FILE:${ARG_BENCH_NAME}> ${filter_arg}
                        --benchmark_min_warmup_time=0.1 --benchmark_min_time=1000x
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
                if(ARG_FIXTURE_NAME)
                    set_tests_properties(
                        ran.phy_nsys.${ARG_BENCH_NAME}${test_suffix}
                        PROPERTIES LABELS "phy;nsys;parallel" FIXTURES_REQUIRED
                                   "${ARG_FIXTURE_NAME}")
                else()
                    set_tests_properties(ran.phy_nsys.${ARG_BENCH_NAME}${test_suffix}
                                         PROPERTIES LABELS "phy;nsys;parallel")
                endif()

                # Disable profiling tests when sanitizers are enabled due to ASan/TSan runtime
                # library loading order conflicts with nsys
                if(ENABLE_SANITIZER_ADDRESS OR ENABLE_SANITIZER_THREAD)
                    set_tests_properties(ran.phy_nsys.${ARG_BENCH_NAME}${test_suffix}
                                         PROPERTIES DISABLED TRUE)
                endif()
            endforeach()
        endif()
    endfunction()
endif()

# ------------------------------------------------------------------------------
# MLIR_TRT - Main Plugin Library
# ------------------------------------------------------------------------------

if(ENABLE_MLIR_TRT)
    add_library(ran_trt_plugin SHARED)

    target_sources(
        ran_trt_plugin
        PRIVATE ${PLUGIN_SRC_DIR}/manager/trt_plugin_registry.cpp
                ${PLUGIN_SRC_DIR}/sample/sequential_sum_plugin.cpp
                ${PLUGIN_SRC_DIR}/sample/sequential_sum_kernel.cu
                ${PLUGIN_SRC_DIR}/dmrs/dmrs_trt_plugin.cpp
                ${PLUGIN_SRC_DIR}/dmrs/dmrs_trt_plugin.cu
                ${PLUGIN_SRC_DIR}/fft/fft_trt_plugin.cpp
                ${PLUGIN_SRC_DIR}/fft/fft_trt_plugin.cu
                ${PLUGIN_SRC_DIR}/cholesky_factor_inv/cholesky_factor_inv_trt_plugin.cpp
                ${PLUGIN_SRC_DIR}/cholesky_factor_inv/cholesky_factor_inv_trt_plugin.cu
        PUBLIC FILE_SET
               HEADERS
               BASE_DIRS
               include
               src
               FILES
               ${PLUGIN_SRC_DIR}/sample/sequential_sum_plugin.hpp
               ${PLUGIN_SRC_DIR}/dmrs/dmrs_trt_plugin.hpp
               ${PLUGIN_SRC_DIR}/fft/fft_trt_plugin.hpp
               ${PLUGIN_SRC_DIR}/cholesky_factor_inv/cholesky_factor_inv_trt_plugin.hpp)

    # Create namespaced alias
    add_library(ran::trt_plugin ALIAS ran_trt_plugin)

    target_include_directories(
        ran_trt_plugin
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include>)

    # Add dependency on aerial_sdk LUT generation and include the LUTS directory
    if(NOT TARGET CRCLUTGEN)
        message(FATAL_ERROR "CRCLUTGEN target not found - required for LUT generation")
    endif()

    if(NOT DEFINED aerial_sdk_BINARY_DIR)
        message(
            FATAL_ERROR
                "aerial_sdk_BINARY_DIR not defined - aerial_sdk must be added as subdirectory")
    endif()

    add_dependencies(ran_trt_plugin CRCLUTGEN)
    set(AERIAL_LUTS_DIR "${aerial_sdk_BINARY_DIR}/cuPHY/src/cuphy/LUTS")
    target_include_directories(ran_trt_plugin PRIVATE ${AERIAL_LUTS_DIR})
    message(STATUS "Added aerial_sdk LUTS directory: ${AERIAL_LUTS_DIR}")

    target_link_libraries(ran_trt_plugin PUBLIC trt_plugin_base)
    target_link_mathdx(ran_trt_plugin PRIVATE cufftdx)
    target_link_cusolverdx(ran_trt_plugin PRIVATE)
    target_link_nvinfer(ran_trt_plugin PUBLIC)
    target_link_system_libraries(ran_trt_plugin PUBLIC CUDA::cudart CUDA::cuda_driver)
    set_target_properties(ran_trt_plugin PROPERTIES SKIP_BUILD_RPATH ON)

    message(STATUS "TensorRT plugin library configured: ran_trt_plugin")
endif()

# ------------------------------------------------------------------------------
# Python Package Configuration
# ------------------------------------------------------------------------------

# Get MLIR_TENSORRT_COMPILER_EXECUTABLE for .env.python generation
if(ENABLE_MLIR_TRT)
    if(NOT TARGET mlir-tensorrt::compiler)
        message(FATAL_ERROR "ENABLE_MLIR_TRT is ON but mlir-tensorrt::compiler target not found!")
    endif()
    get_target_property(MLIR_TENSORRT_COMPILER_EXECUTABLE mlir-tensorrt::compiler IMPORTED_LOCATION)
    if(NOT MLIR_TENSORRT_COMPILER_EXECUTABLE)
        message(
            FATAL_ERROR
                "ENABLE_MLIR_TRT is ON but mlir-tensorrt::compiler executable path not found!")
    endif()
    message(STATUS "MLIR-TensorRT compiler path: ${MLIR_TENSORRT_COMPILER_EXECUTABLE}")
endif()

# cmake-format: off
# Generate .env.python file for direct pytest usage
# Generate in BUILD directory (preset-specific)
# CTest passes path via RAN_ENV_PYTHON_FILE (priority 1) with fixture sync as fallback (priority 2)
if(ENABLE_MLIR_TRT)
    # Configure: Generate .env.python in BUILD directory (preset-specific)
    file(GENERATE 
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.env.python
        CONTENT "# Auto-generated by CMake - DO NOT EDIT
# Re-run CMake configure to update these paths
# Generated from: ran/py/CMakeLists.txt

RAN_TRT_PLUGIN_DSO_PATH=$<TARGET_FILE:ran_trt_plugin>
RAN_TRT_ENGINE_PATH=${RAN_TRT_ENGINE_PATH}
RAN_PYTEST_BUILD_DIR=${RAN_PYTEST_BUILD_DIR}
MLIR_TRT_COMPILER_PATH=${MLIR_TENSORRT_COMPILER_EXECUTABLE}
ENABLE_MLIR_TRT=ON
"
    )
    message(STATUS "Will generate .env.python in build dir: ${CMAKE_CURRENT_BINARY_DIR}/.env.python")
else()
    # Configure: Generate minimal .env.python when MLIR_TRT is disabled
    file(GENERATE 
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.env.python
        CONTENT "# Auto-generated by CMake - DO NOT EDIT
# Generated from: ran/py/CMakeLists.txt

RAN_PYTEST_BUILD_DIR=${RAN_PYTEST_BUILD_DIR}
ENABLE_MLIR_TRT=OFF
"
    )
    message(STATUS "Will generate .env.python in build dir: ${CMAKE_CURRENT_BINARY_DIR}/.env.python")
endif()
# cmake-format: on

# Build: Custom target to manually sync from build dir to source dir
add_custom_target(
    sync_env_python
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/.env.python
            ${CMAKE_CURRENT_SOURCE_DIR}/.env.python
    COMMENT
        "Syncing .env.python: ${CMAKE_CURRENT_BINARY_DIR}/.env.python -> ${CMAKE_CURRENT_SOURCE_DIR}/.env.python"
)

# CTest: Fixture to sync .env.python to source dir as fallback (Priority 2) Tests get
# RAN_ENV_PYTHON_PATH (Priority 1) but fixture ensures source dir has current version
if(BUILD_TESTING)
    # Build py_ran_setup target to ensure MLIR-TRT compiler and Python env are ready.This ensures
    # the .env.python file references valid paths.
    add_test(NAME ran_build_setup COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
                                          py_ran_setup)
    set_tests_properties(ran_build_setup PROPERTIES FIXTURES_SETUP ran_env LABELS "parallel")

    # Sync .env.python after setup is complete
    add_test(NAME ran_sync_env_python
             COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/.env.python
                     ${CMAKE_CURRENT_SOURCE_DIR}/.env.python)
    set_tests_properties(ran_sync_env_python PROPERTIES FIXTURES_SETUP ran_env LABELS "parallel")
endif()

# Define Python extras based on MLIR_TRT configuration
if(ENABLE_MLIR_TRT)
    set(RAN_PYTHON_EXTRAS "dev" "mlir_trt_wheels" "phy_jax" "datasets")
else()
    set(RAN_PYTHON_EXTRAS "dev" "datasets")
endif()

add_python_package_targets("ran" ${CMAKE_CURRENT_SOURCE_DIR} EXTRAS ${RAN_PYTHON_EXTRAS})

# Pass the build-specific .env.python path to the main test target via environment variable
if(BUILD_TESTING)
    # Disabled because we already have individual pytest targets for each test directory
    set_tests_properties(
        py_ran_test
        PROPERTIES ENVIRONMENT "RAN_ENV_PYTHON_FILE=${CMAKE_CURRENT_BINARY_DIR}/.env.python"
                   FIXTURES_REQUIRED "ran_env" DISABLED TRUE)
endif()

# Add target dependency chain for MLIR TRT
if(ENABLE_MLIR_TRT)
    # MLIR-TRT setup must run BEFORE py_ran_setup so it can download the wheels
    add_dependencies(py_ran_setup py_ran_mlir_trt_setup)
    message(STATUS "Python setup (venv creation) will run after MLIR TRT downloads")

    # Python setup must depend on the TensorRT plugin build (ran_trt_plugin) so it can use the
    # plugin in the tests
    add_dependencies(py_ran_setup ran_trt_plugin)
    message(STATUS "Python setup will depend on the TensorRT plugin build (ran_trt_plugin)")
endif()

# cmake-format: off
# Helper macro to create fixture test targets for TRT engine generation
# Creates _fixture variant that uses cached engines when SKIP_TRT_ENGINE_GENERATION=1
# Sets up named fixture (e.g., dmrs_trt_engines) for C++ tests to depend on
# Test depends on py_ran_setup (inherited from add_phy_python_test)
# cmake-format: on
macro(add_trt_engine_python_test TEST_NAME TEST_DIR FIXTURE_NAME)
    add_phy_python_test(${TEST_NAME}_fixture ${TEST_DIR})
    # Add SKIP_TRT_ENGINE_GENERATION to existing ENVIRONMENT property set by add_phy_python_test
    set_tests_properties(
        ${TEST_NAME}_fixture
        PROPERTIES
            LABELS
            "parallel"
            ENVIRONMENT
            "RAN_ENV_PYTHON_FILE=${CMAKE_CURRENT_BINARY_DIR}/.env.python;SKIP_TRT_ENGINE_GENERATION=1"
            FIXTURES_REQUIRED
            "ran_env"
            FIXTURES_SETUP
            ${FIXTURE_NAME})
endmacro()

# Granular test targets for specific test directories (generates .trtengine files)
if(ENABLE_MLIR_TRT)
    # cmake-format: off
    # Fixture test targets for TRT engine generation
    # Uses cached engines when SKIP_TRT_ENGINE_GENERATION=1 (set by fixture property)
    # Generates engines when they don't exist or when explicitly requested
    # Example: py_ran_test_dmrs_plugin_fixture sets up fixture dmrs_trt_engines
    # cmake-format: on

    add_trt_engine_python_test(py_ran_test_sample_plugin ${PLUGIN_TEST_DIR}/sample
                               sample_trt_engines)
    add_trt_engine_python_test(py_ran_test_dmrs_plugin ${PLUGIN_TEST_DIR}/dmrs dmrs_trt_engines)
    add_trt_engine_python_test(py_ran_test_fft_plugin ${PLUGIN_TEST_DIR}/fft fft_trt_engines)
    add_trt_engine_python_test(py_ran_test_cholesky_factor_inv_plugin
                               ${PLUGIN_TEST_DIR}/cholesky_factor_inv cholesky_trt_engines)
    add_trt_engine_python_test(py_ran_test_mlir_trt_wrapper ${MLIR_TRT_WRAPPER_TEST_DIR}
                               mlir_trt_engines)
    add_trt_engine_python_test(
        py_ran_test_pusch_inner_receiver ${PHY_JAX_TEST_DIR}/pusch/test_pusch_inner_receiver.py
        pusch_inner_receiver_trt_engines)
    add_trt_engine_python_test(
        py_ran_test_channel_estimation ${PHY_JAX_TEST_DIR}/pusch/test_channel_estimation.py
        channel_estimation_trt_engines)

    message(STATUS "Added granular Python test targets for TensorRT engine generation")

    # Add individual Python test for numpy PHY tests (doesn't require TRT engines)
    set(PHY_NUMPY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/phy/numpy)
    add_phy_python_test(py_ran_test_phy_numpy ${PHY_NUMPY_TEST_DIR})
    set_tests_properties(py_ran_test_phy_numpy PROPERTIES LABELS "parallel")
endif()

# ------------------------------------------------------------------------------
# PHY - Plugin Tests and Benchmarks
# ------------------------------------------------------------------------------

if(ENABLE_MLIR_TRT)
    # C++ tests and benchmarks depend on TRT engine fixtures from Python tests
    add_phy_test(TEST_NAME mlir_trt_add_test SOURCE_FILES ${MLIR_TRT_WRAPPER_TEST_DIR}/add_test.cpp
                 FIXTURE_NAME mlir_trt_engines)

    add_phy_bench(BENCH_NAME mlir_trt_add_bench SOURCE_FILES
                  ${MLIR_TRT_WRAPPER_TEST_DIR}/add_bench.cpp FIXTURE_NAME mlir_trt_engines)

    add_phy_test(TEST_NAME sample_plugin_tests SOURCE_FILES
                 ${PLUGIN_TEST_DIR}/sample/sample_plugin_tests.cpp FIXTURE_NAME sample_trt_engines)

    add_phy_bench(BENCH_NAME sample_plugin_bench SOURCE_FILES
                  ${PLUGIN_TEST_DIR}/sample/sample_plugin_bench.cpp FIXTURE_NAME sample_trt_engines)

    add_phy_test(TEST_NAME dmrs_plugin_tests SOURCE_FILES
                 ${PLUGIN_TEST_DIR}/dmrs/dmrs_plugin_tests.cpp FIXTURE_NAME dmrs_trt_engines)

    add_phy_bench(BENCH_NAME dmrs_plugin_bench SOURCE_FILES
                  ${PLUGIN_TEST_DIR}/dmrs/dmrs_plugin_bench.cpp FIXTURE_NAME dmrs_trt_engines)

    add_phy_test(TEST_NAME fft_plugin_tests SOURCE_FILES
                 ${PLUGIN_TEST_DIR}/fft/fft_plugin_tests.cpp FIXTURE_NAME fft_trt_engines)

    add_phy_bench(BENCH_NAME fft_plugin_bench SOURCE_FILES
                  ${PLUGIN_TEST_DIR}/fft/fft_plugin_bench.cpp FIXTURE_NAME fft_trt_engines)

    add_phy_test(
        TEST_NAME cholesky_factor_inv_plugin_tests SOURCE_FILES
        ${PLUGIN_TEST_DIR}/cholesky_factor_inv/cholesky_factor_inv_plugin_tests.cpp FIXTURE_NAME
        cholesky_trt_engines)

    add_phy_bench(
        BENCH_NAME cholesky_factor_inv_plugin_bench SOURCE_FILES
        ${PLUGIN_TEST_DIR}/cholesky_factor_inv/cholesky_factor_inv_plugin_bench.cpp FIXTURE_NAME
        cholesky_trt_engines)

    add_phy_test(
        TEST_NAME
        pusch_inner_receiver_tests
        SOURCE_FILES
        ${PHY_JAX_TEST_DIR}/pusch/pusch_inner_receiver_tests.cpp
        ${PHY_JAX_TEST_DIR}/pusch/pusch_inner_receiver_test_utils.cpp
        FIXTURE_NAME
        pusch_inner_receiver_trt_engines
        GTEST_FILTERS
        "free_energy"
        "weighted_threshold"
        "ai_tukey")

    add_phy_bench(
        BENCH_NAME
        pusch_inner_receiver_bench
        SOURCE_FILES
        ${PHY_JAX_TEST_DIR}/pusch/pusch_inner_receiver_bench.cpp
        ${PHY_JAX_TEST_DIR}/pusch/pusch_inner_receiver_test_utils.cpp
        FIXTURE_NAME
        pusch_inner_receiver_trt_engines
        BENCHMARK_FILTERS
        "free_energy"
        "weighted_threshold"
        "ai_tukey")

    add_phy_bench(
        BENCH_NAME
        channel_estimation_bench
        SOURCE_FILES
        ${PHY_JAX_TEST_DIR}/pusch/channel_estimation_bench.cpp
        FIXTURE_NAME
        channel_estimation_trt_engines
        BENCHMARK_FILTERS
        "free_energy"
        "weighted_threshold"
        "identity")

    # Aggregate C++ build target must also depend on py_ran_setup to ensure the Python environment
    # is set up
    add_custom_target(phy_all DEPENDS phy_tests_all phy_bench_all py_ran_setup
                                      py_ran_mlir_trt_setup)
endif()
