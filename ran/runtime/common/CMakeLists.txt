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

add_library(ran_common INTERFACE)
add_library(ran::common ALIAS ran_common)

target_sources(ran_common INTERFACE FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                                    ran_common.hpp)

target_include_directories(ran_common INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                $<INSTALL_INTERFACE:include/>)

install(
    TARGETS ran_common
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# headers – component dev
install(
    FILES ran_common.hpp
    DESTINATION include/ran
    COMPONENT dev)

# sources – component src
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION src/aerial-framework/ran/common
    COMPONENT src
    FILES_MATCHING
    PATTERN "*.hpp")

set_target_properties(ran_common PROPERTIES VERSION ${PROJECT_VERSION})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(ran_common INTERFACE RAN_COMMON_STATIC_DEFINE)
endif()
