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

add_executable(rt_log_sample rt_log_sample.cpp)

target_link_libraries(rt_log_sample PRIVATE framework::options framework::warnings
                                            framework::rt_log)

target_link_system_libraries(rt_log_sample PRIVATE CLI11::CLI11)

target_include_directories(rt_log_sample PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

add_version_and_help_tests(rt_log_sample)

add_test(NAME rt_log_sample.console COMMAND $<TARGET_FILE:rt_log_sample> --level INFO)
label_parallel(rt_log_sample.console)
add_test(NAME rt_log_sample.file
         COMMAND $<TARGET_FILE:rt_log_sample> --file
                 ${CMAKE_CURRENT_BINARY_DIR}/rt_log_sample_test.log --level DEBUG)
label_parallel(rt_log_sample.file)
