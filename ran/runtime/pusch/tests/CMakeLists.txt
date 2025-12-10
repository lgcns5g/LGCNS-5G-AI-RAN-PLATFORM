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

include("${CMAKE_SOURCE_DIR}/cmake/RanTrtPlugin.cmake")

# Create a shared test utilities library
add_library(pusch_test_utils STATIC pusch_test_utils.cpp pusch_pipeline_runner.cpp
                                    inner_rx_module_runner.cpp)

target_link_libraries(
    pusch_test_utils
    PUBLIC framework::pipeline
           framework::memory
           framework::utils
           ran::pusch
           ran::aerial_tv
           ran::ldpc
           ran::common
           CUDA::cudart)

target_include_directories(pusch_test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(pusch_test_utils
                           PUBLIC TEST_VECTOR_PATH="${CMAKE_SOURCE_DIR}/ran/test_data")

# GTest executable
add_executable(pusch_tests inner_rx_module_test.cpp pusch_pipeline_test.cpp pusch_test_env.cpp
                           pusch_sample_tests.cpp)

target_link_system_libraries(pusch_tests PRIVATE GTest::gtest_main)
target_link_libraries(pusch_tests PRIVATE framework::warnings framework::options framework::rt_log
                                          pusch_test_utils)

add_test(
    NAME pusch_tests
    COMMAND pusch_tests --gtest_output=xml:pusch_tests.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Create output directories for benchmark and profiling results
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark_results)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/nsys_results)

# Common TRT engine configuration
set(PUSCH_TRT_ENGINE_NAME "pusch_inner_receiver_free_energy_filter")
set(PUSCH_TRT_FIXTURE_NAME "pusch_inner_receiver_trt_engines")
set(PUSCH_TRT_ENGINE_DIR "ran/runtime/pusch/engine")

# Helper function to set TRT engine path with common parameters
function(pusch_set_trt_engine_path TEST_NAME)
    test_set_trt_engine_path(${TEST_NAME} ${PUSCH_TRT_ENGINE_NAME} FIXTURE_NAME
                             ${PUSCH_TRT_FIXTURE_NAME} ENGINE_DIR ${PUSCH_TRT_ENGINE_DIR})
endfunction()

# Helper function to add a benchmark executable with corresponding benchmark and nsys test targets
# Parameters: PREFIX - Prefix for the benchmark (e.g., "pusch_pipeline" creates
# "pusch_pipeline_bench") SOURCE_FILE - Source file for the benchmark
function(add_pusch_bench PREFIX SOURCE_FILE)
    set(BENCH_NAME ${PREFIX}_bench)
    set(NSYS_NAME ${PREFIX}_nsys)

    # Create benchmark executable
    add_executable(${BENCH_NAME} ${SOURCE_FILE})

    target_link_system_libraries(${BENCH_NAME} PRIVATE benchmark::benchmark
                                 benchmark::benchmark_main)
    target_link_libraries(
        ${BENCH_NAME} PRIVATE framework::warnings framework::options framework::rt_log
                              framework::task_instrument pusch_test_utils)

    # Register benchmark run with JSON output
    add_test(
        NAME ${BENCH_NAME}
        COMMAND $<TARGET_FILE:${BENCH_NAME}> --benchmark_format=console --benchmark_out_format=json
                --benchmark_out=${CMAKE_BINARY_DIR}/benchmark_results/${BENCH_NAME}_results.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(${BENCH_NAME} PROPERTIES LABELS "benchmark")

    # Register nsys profiling as CTest test
    add_test(
        NAME ${NSYS_NAME}
        COMMAND
            nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage=true --stats=true
            --output=${CMAKE_BINARY_DIR}/nsys_results/${PREFIX}.nsys-rep --force-overwrite=true
            $<TARGET_FILE:${BENCH_NAME}> --benchmark_min_warmup_time=0.1 --benchmark_min_time=1000x
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(${NSYS_NAME} PROPERTIES LABELS "nsys")

    # Disable profiling tests when sanitizers are enabled
    if(ENABLE_SANITIZER_ADDRESS OR ENABLE_SANITIZER_THREAD)
        set_tests_properties(${NSYS_NAME} PROPERTIES DISABLED TRUE)
    endif()

    # Set TRT engine path for both benchmark and nsys tests
    pusch_set_trt_engine_path(${BENCH_NAME})
    pusch_set_trt_engine_path(${NSYS_NAME})
endfunction()

# Benchmarks PUSCH Pipeline benchmark (creates pusch_pipeline_bench and pusch_pipeline_nsys)
add_pusch_bench(pusch_pipeline pusch_pipeline_bench.cpp)

# InnerRx Module benchmark (creates inner_rx_module_bench and inner_rx_module_nsys)
add_pusch_bench(inner_rx_module inner_rx_module_bench.cpp)

# Set TRT engine path for test executable
pusch_set_trt_engine_path(pusch_tests)
