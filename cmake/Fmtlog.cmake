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

include_guard(GLOBAL)

include(cmake/CPM.cmake)
include(cmake/Patches.cmake)

# Setup fmtlog library with aerial_sdk patches Requires: fmt::fmt to be available Arguments:
# aerial_sdk_source_dir - Path to aerial_sdk source directory
function(setup_fmtlog aerial_sdk_source_dir)
    if(TARGET fmtlog::fmtlog)
        return() # Already set up
    endif()

    if(NOT aerial_sdk_source_dir)
        message(FATAL_ERROR "setup_fmtlog() requires aerial_sdk_source_dir argument")
    endif()

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    cpmaddpackage(
        NAME
        fmtlog
        GITHUB_REPOSITORY
        MengRao/fmtlog
        GIT_TAG
        acd521b1a64480354136a745c511358da1ec7dc5
        GIT_SUBMODULES
        ""
        DOWNLOAD_ONLY
        YES
        SYSTEM
        YES)

    if(fmtlog_ADDED)
        # Apply patch from aerial_sdk (ignore exit code since fmt submodule removal will fail
        # harmlessly)
        apply_patch_once(${aerial_sdk_source_dir}/cuPHY-CP/container/patches/fmtlog.patch
                         ${fmtlog_SOURCE_DIR})

        # Add fmtlog as subdirectory
        add_subdirectory(${fmtlog_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/fmtlog EXCLUDE_FROM_ALL)

        # Apply custom compile definition for queue size and ensure headers are accessible Configure
        # both shared and static targets
        foreach(fmtlog_target fmtlog-shared fmtlog-static)
            if(TARGET ${fmtlog_target})
                target_compile_definitions(${fmtlog_target} PUBLIC FMTLOG_QUEUE_SIZE=0x1000000)
                target_include_directories(
                    ${fmtlog_target} PUBLIC $<BUILD_INTERFACE:${fmtlog_SOURCE_DIR}>
                                            $<INSTALL_INTERFACE:include>)
            endif()
        endforeach()

        # Create aliases (prefer shared library)
        if(TARGET fmtlog-shared)
            if(NOT TARGET fmtlog::fmtlog)
                add_library(fmtlog::fmtlog ALIAS fmtlog-shared)
            endif()
            if(NOT TARGET fmtlog)
                add_library(fmtlog ALIAS fmtlog-shared)
            endif()
        elseif(TARGET fmtlog-static)
            if(NOT TARGET fmtlog::fmtlog)
                add_library(fmtlog::fmtlog ALIAS fmtlog-static)
            endif()
            if(NOT TARGET fmtlog)
                add_library(fmtlog ALIAS fmtlog-static)
            endif()
        endif()
    endif()
endfunction()
