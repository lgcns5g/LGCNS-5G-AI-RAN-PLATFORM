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

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)

# Function to create a task sample application
function(add_task_sample_app APP_NAME)
    add_executable(${APP_NAME} ${APP_NAME}.cpp)

    target_link_libraries(${APP_NAME} PRIVATE framework::options framework::warnings
                                              framework::rt_log framework::task)
    target_link_system_libraries(${APP_NAME} PRIVATE CLI11::CLI11 tl::expected)
    target_include_directories(${APP_NAME} PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

endfunction()

# Function to add tests for a task sample application
function(add_task_sample_tests APP_NAME)
    add_version_and_help_tests(${APP_NAME})

    add_test(NAME ${APP_NAME}.console COMMAND $<TARGET_FILE:${APP_NAME}> --interval_ms 1 --count
                                              2000)
    label_parallel(${APP_NAME}.console)
endfunction()

# Create the taskwork sample applications
add_task_sample_app(task_sample)
add_task_sample_tests(task_sample)
