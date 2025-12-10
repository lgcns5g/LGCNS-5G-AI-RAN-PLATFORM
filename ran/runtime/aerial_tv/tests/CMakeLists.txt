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

# Create the cuphy_pusch_tv_test executable
add_executable(cuphy_pusch_tv_test cuphy_pusch_tv_test.cpp cuphy_pusch_tv_sample_test.cpp)

target_link_libraries(cuphy_pusch_tv_test PRIVATE ran::aerial_tv framework::warnings
                                                  framework::options)
target_link_system_libraries(cuphy_pusch_tv_test PRIVATE GTest::gtest_main)
target_compile_definitions(cuphy_pusch_tv_test
                           PRIVATE TEST_VECTOR_PATH="${CMAKE_SOURCE_DIR}/ran/test_data")

add_test(NAME cuphy_pusch_tv_test COMMAND cuphy_pusch_tv_test)
label_parallel(cuphy_pusch_tv_test)
