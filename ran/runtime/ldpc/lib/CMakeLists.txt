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

add_library(
    ldpc src/ldpc_params.cpp src/outer_rx_params.cpp src/crc_decoder_module.cpp
         src/ldpc_derate_match_module.cpp src/ldpc_decoder_module.cpp src/ldpc_module_factories.cpp)

add_library(ran::ldpc ALIAS ldpc)

target_include_directories(ldpc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(ldpc PUBLIC framework::pipeline framework::rt_log framework::warnings
                                  framework::options ran::common)
target_link_system_libraries(ldpc PUBLIC cuphy::cuphy_ldpc wise_enum::wise_enum range-v3::range-v3)
