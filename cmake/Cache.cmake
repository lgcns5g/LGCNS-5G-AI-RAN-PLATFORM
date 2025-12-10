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

# Enable cache if available
function(enable_cache)
    set(CACHE_OPTION
        "ccache"
        CACHE STRING "Compiler cache to be used")
    set(CACHE_OPTION_VALUES "ccache" "sccache")
    set_property(CACHE CACHE_OPTION PROPERTY STRINGS ${CACHE_OPTION_VALUES})
    list(FIND CACHE_OPTION_VALUES ${CACHE_OPTION} CACHE_OPTION_INDEX)

    if(${CACHE_OPTION_INDEX} EQUAL -1)
        message(
            STATUS
                "Using custom compiler cache system: '${CACHE_OPTION}', explicitly supported entries are ${CACHE_OPTION_VALUES}"
        )
    endif()

    find_program(CACHE_BINARY NAMES ${CACHE_OPTION_VALUES})
    if(CACHE_BINARY)
        message(STATUS "${CACHE_BINARY} found and enabled")
        set(CMAKE_CXX_COMPILER_LAUNCHER
            ${CACHE_BINARY}
            CACHE FILEPATH "CXX compiler cache used")
        set(CMAKE_C_COMPILER_LAUNCHER
            ${CACHE_BINARY}
            CACHE FILEPATH "C compiler cache used")
    else()
        message(WARNING "${CACHE_OPTION} is enabled but was not found. Not using it")
    endif()
endfunction()
