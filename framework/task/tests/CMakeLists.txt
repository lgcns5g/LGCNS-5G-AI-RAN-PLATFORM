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

enable_testing()

include(GoogleTest)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

function(add_task_test TEST_NAME)
    set(TARGET_NAME task_tests_${TEST_NAME})
    add_executable(${TARGET_NAME} ${ARGN})

    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE framework::warnings framework::options
                                                 framework::task framework::task_instrument)
    target_link_system_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main gsl-lite::gsl-lite)

    set_cap_sys_nice(${TARGET_NAME})

    add_test(
        NAME task_tests.${TEST_NAME}
        COMMAND ${TARGET_NAME} --gtest_output=xml:${TARGET_NAME}.xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

add_task_test(
    library
    spinlock_tests.cpp
    flat_map_tests.cpp
    task_scheduler_tests.cpp
    task_monitor_tests.cpp
    bounded_queue_tests.cpp
    memory_trigger_tests.cpp
    timed_trigger_tests.cpp
    task_worker_tests.cpp
    task_tests.cpp
    task_pool_tests.cpp
    task_graph_tests.cpp
    task_category_tests.cpp
    task_utils_tests.cpp
    task_visualizer_tests.cpp
    time_tests.cpp
    task_sample_tests.cpp
    log_env.cpp)

label_real_time(task_tests.library)

add_task_test(instrument nvtx_tests.cpp instrumented_functions.cpp function_cache_tests.cpp
              log_env.cpp)

# Enable function instrumentation for the instrumented functions file
set_source_files_properties(instrumented_functions.cpp PROPERTIES COMPILE_FLAGS
                                                                  "-finstrument-functions")

# Ensure debug symbols are available for function name resolution
target_compile_options(task_tests_instrument PRIVATE -g)
target_link_options(task_tests_instrument PRIVATE -rdynamic)

add_custom_target(task_tests DEPENDS task_tests_library task_tests_instrument)
