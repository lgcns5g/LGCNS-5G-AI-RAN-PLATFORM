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

# Setup accessor library (C++ member accessor with C++20 compatibility fixes)
function(setup_accessor)
    if(TARGET accessor::accessor)
        return() # Already set up
    endif()

    cmake_policy(PUSH)
    # Enable portable IPO/LTO support detection
    cmake_policy(SET CMP0069 NEW)

    cpmaddpackage(
        NAME
        accessor_lib
        GITHUB_REPOSITORY
        hliberacki/cpp-member-accessor
        GIT_TAG
        5e4d52f82006d14c63488eadca757f893b85824b
        SYSTEM
        YES)

    if(NOT TARGET accessor::accessor AND TARGET accessor)
        add_library(accessor::accessor ALIAS accessor)
    endif()

    cmake_policy(POP)
endfunction()
