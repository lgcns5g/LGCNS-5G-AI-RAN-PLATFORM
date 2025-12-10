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

# Just linting/formatting targets (real tests are integration tests below)
add_python_ruff_targets("task_tools" ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_TESTING)
    enable_testing()

    # Add stress test that generates JSON stats and analyzes them
    add_test(
        NAME task_tools.stress_run
        COMMAND
            ${CMAKE_COMMAND} -E env
            GTEST_FILTER=TaskScheduler.StressTestMultipleWorkersWithDependencies
            $<TARGET_FILE:task_tests_library> --gtest_output=xml:task_tests_stress_run.xml
        WORKING_DIRECTORY $<TARGET_FILE_DIR:task_tests_library>)
    set_tests_properties(task_tools.stress_run PROPERTIES FIXTURES_SETUP stress_test_data DEPENDS
                                                          task_tests_library)

    # Add test to analyze the generated stress test stats
    add_test(
        NAME task_tools.stress_analysis
        COMMAND uv --directory ${CMAKE_CURRENT_SOURCE_DIR} run src/task_stats.py
                $<TARGET_FILE_DIR:task_tests_library>/stress_test_stats.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(task_tools.stress_analysis PROPERTIES FIXTURES_REQUIRED stress_test_data
                                                               DEPENDS py_task_tools_setup)

    # Add NVTX test that profiles the test execution with nsys
    add_test(
        NAME task_tools.nvtx_profile
        COMMAND ${CMAKE_COMMAND} -E env GTEST_FILTER=Nvtx.* nsys profile --trace=nvtx,cuda
                --output=nvtx_test_trace --force-overwrite=true $<TARGET_FILE:task_tests_instrument>
        WORKING_DIRECTORY $<TARGET_FILE_DIR:task_tests_instrument>)
    set_tests_properties(
        task_tools.nvtx_profile PROPERTIES FIXTURES_SETUP nvtx_profile_data DEPENDS
                                           "task_tests_instrument;py_task_tools_setup")

    # Add test to analyze the generated NVTX profile and check for instrumented functions
    add_test(
        NAME task_tools.nvtx_analysis
        COMMAND
            uv --directory ${CMAKE_CURRENT_SOURCE_DIR} run src/nsys_stats.py
            $<TARGET_FILE_DIR:task_tests_instrument>/nvtx_test_trace.nsys-rep test_nvtx_function
            compute_sum compute_product perform_iterations fibonacci orchestrate_computations
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(task_tools.nvtx_analysis PROPERTIES FIXTURES_REQUIRED nvtx_profile_data
                                                             DEPENDS py_task_tools_setup)

    # Disable profiling tests when sanitizers are enabled due to ASan/TSan runtime library loading
    # order conflicts with nsys
    if(ENABLE_SANITIZER_ADDRESS OR ENABLE_SANITIZER_THREAD)
        set_tests_properties(task_tools.nvtx_profile PROPERTIES DISABLED TRUE)
        set_tests_properties(task_tools.nvtx_analysis PROPERTIES DISABLED TRUE)
    endif()
endif()
