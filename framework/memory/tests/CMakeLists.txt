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

# Memory Domain Tests

include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

add_executable(
    framework_memory_tests
    buffer_test.cpp
    device_allocators_test.cpp
    gdrcopy_buffer_test.cpp
    memcpy_helper_test.cpp
    monotonic_alloc_test.cpp
    unique_ptr_utils_test.cpp
    memory_sample_tests.cpp)

target_link_libraries(
    framework_memory_tests PRIVATE framework::utils framework::memory framework-test-utils
                                   GTest::gtest_main framework::options framework::warnings)

target_link_cuda(framework_memory_tests)

# Link GDRCopy for gdrcopy_buffer_test.cpp
target_link_gdrcopy(framework_memory_tests)

add_test(NAME memory.all_tests COMMAND framework_memory_tests)
label_parallel(memory.all_tests)
