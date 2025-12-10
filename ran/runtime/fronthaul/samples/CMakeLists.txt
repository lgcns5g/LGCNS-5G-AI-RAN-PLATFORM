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

include(${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Doca.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GDRCopy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CapPermissions.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Dpdk.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)

add_executable(fronthaul_app fronthaul_app.cpp fronthaul_app_utils.cpp)
target_link_libraries(fronthaul_app PRIVATE ran::fronthaul framework::options framework::warnings
                                            framework::rt_log framework::net framework::task)
target_link_system_libraries(fronthaul_app PRIVATE CLI11::CLI11 CUDA::cudart CUDA::cuda_driver
                             tl::expected)
target_include_directories(fronthaul_app PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")
target_link_gdrcopy(fronthaul_app)
set_dpdk_capabilities(fronthaul_app)

install(TARGETS fronthaul_app RUNTIME DESTINATION bin)
add_version_and_help_tests(fronthaul_app)

# Fronthaul + RU emulator integration test
if(BUILD_TESTING)
    add_fronthaul_integration_test(fronthaul_app)
endif()
