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

set(LIB_SOURCES message_adapter.cpp phy_stats.cpp)

set(APP_SOURCES main.cpp)

message(STATUS "Building Message Adapter")

# Create message_adapter library
set(LIB_TARGET message_adapter)
add_library(${LIB_TARGET} STATIC ${LIB_SOURCES})
add_library(ran::message_adapter ALIAS message_adapter)

target_include_directories(${LIB_TARGET} PUBLIC ${MESSAGE_ADAPTER_INCLUDE})
target_include_directories(${LIB_TARGET} PRIVATE ${EXAMPLE_PIPELINE_INCLUDE})

target_link_libraries(${LIB_TARGET} PUBLIC framework::rt_log framework::warnings framework::options)
target_link_system_libraries(${LIB_TARGET} PUBLIC nvlog nvipc yaml)

# Create message_adapter_app executable
set(APP_TARGET message_adapter_app)
add_executable(${APP_TARGET} ${APP_SOURCES})

target_include_directories(${APP_TARGET} PRIVATE ${MESSAGE_ADAPTER_INCLUDE}
                                                 ${EXAMPLE_PIPELINE_INCLUDE})
target_include_directories(${APP_TARGET} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(${APP_TARGET} PRIVATE ${LIB_TARGET} sample_5g_pipeline)
target_link_system_libraries(${APP_TARGET} PRIVATE nvlog nvipc yaml scf::scf_5g_fapi_h)

# Enable CUDA for device linking (required by DOCA GPU libraries in dependency chain)
target_link_cuda(${APP_TARGET})
