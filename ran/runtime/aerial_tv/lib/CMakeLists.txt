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

# Create the aerial_tv library
add_library(aerial_tv src/cuphy_pusch_tv.cpp src/aerial_tv_utils.cpp)
add_library(ran::aerial_tv ALIAS aerial_tv)

target_include_directories(aerial_tv PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(aerial_tv PRIVATE framework::options framework::warnings)
target_link_libraries(aerial_tv PUBLIC ran::common ran::ldpc)
target_link_system_libraries(aerial_tv PUBLIC hdf5hpp::hdf5hpp range-v3::range-v3
                             gsl-lite::gsl-lite)
target_link_system_libraries(aerial_tv PRIVATE GTest::gtest GTest::gmock)
