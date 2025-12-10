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
include(${CMAKE_SOURCE_DIR}/cmake/Dpdk.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Doca.cmake)

# Function to create a network sample application
function(add_net_sample_app APP_NAME)
    add_executable(${APP_NAME} net_samples.cpp ${APP_NAME}.cpp gpunetio_kernels.cu)

    target_link_cuda(${APP_NAME})

    target_link_libraries(
        ${APP_NAME} PRIVATE framework::options framework::warnings framework::rt_log framework::net
                            framework::memory framework::utils)
    target_link_system_libraries(${APP_NAME} PRIVATE CLI11::CLI11 CUDA::cudart CUDA::cuda_driver)
    target_include_directories(${APP_NAME} PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")
    set_dpdk_capabilities(${APP_NAME})

endfunction()

# Function to add tests for a network sample application
function(add_net_sample_tests APP_NAME)
    add_version_and_help_tests(${APP_NAME})

    if(APP_NAME STREQUAL "net_receiver")
        add_test(NAME ${APP_NAME}.console COMMAND $<TARGET_FILE:${APP_NAME}> --timeout 1)
        label_test_requires_nic(${APP_NAME}.console)
        label_parallel(${APP_NAME}.console)
    else()
        add_test(NAME ${APP_NAME}.console COMMAND $<TARGET_FILE:${APP_NAME}>)
        add_test(NAME ${APP_NAME}.cpu_only COMMAND $<TARGET_FILE:${APP_NAME}> --cpu-only)
        label_test_requires_nic(${APP_NAME}.console)
        label_test_requires_nic(${APP_NAME}.cpu_only)
        label_parallel(${APP_NAME}.console)
        label_parallel(${APP_NAME}.cpu_only)
    endif()
endfunction()

# Create the network sample applications
add_net_sample_app(net_sender)
add_net_sample_app(net_receiver)

# Create the network sample tests
add_net_sample_tests(net_sender)
add_net_sample_tests(net_receiver)

add_custom_target(net_samples DEPENDS net_sender net_receiver)
