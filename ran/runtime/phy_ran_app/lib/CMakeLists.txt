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

# PHY RAN App Utilities Library

# Include GDRCopy CMake module (needed for fronthaul dependency)
include(${CMAKE_SOURCE_DIR}/cmake/GDRCopy.cmake)

add_library(phy_ran_utils src/fapi_rx_handler.cpp src/task_factories.cpp)

target_include_directories(
    phy_ran_utils PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_link_libraries(
    phy_ran_utils
    PUBLIC ran::fapi # FAPI state machine and types
           ran::sample_5g_pipeline # Sample5GPipeline (Message Adapter) - PUBLIC because interfaces
                                   # are exposed in fapi_rx_handler.hpp
           framework::options # Project-wide compiler options
           framework::warnings # Project-wide warnings
    PRIVATE ran::fronthaul # Fronthaul interface (only used in task_factories.cpp implementation)
)

# GDRCopy for GPU Direct RDMA (needed by fronthaul)
target_link_gdrcopy(phy_ran_utils)

# Create alias for consistent naming
add_library(ran::phy_ran_utils ALIAS phy_ran_utils)

# Installation
install(
    TARGETS phy_ran_utils
    EXPORT phy_ran_utils_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES
    DESTINATION include)

install(DIRECTORY include/ DESTINATION include)
