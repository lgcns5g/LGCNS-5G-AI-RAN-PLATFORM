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

macro(configure_linker project_name)
    include(CheckCXXCompilerFlag)

    set(USER_LINKER_OPTION
        "lld"
        CACHE STRING "Linker to be used")
    set(USER_LINKER_OPTION_VALUES "lld" "gold" "bfd" "mold")
    set_property(CACHE USER_LINKER_OPTION PROPERTY STRINGS ${USER_LINKER_OPTION_VALUES})
    list(FIND USER_LINKER_OPTION_VALUES ${USER_LINKER_OPTION} USER_LINKER_OPTION_INDEX)

    if(${USER_LINKER_OPTION_INDEX} EQUAL -1)
        message(
            STATUS
                "Using custom linker: '${USER_LINKER_OPTION}', explicitly supported entries are ${USER_LINKER_OPTION_VALUES}"
        )
    endif()

    if(NOT DEFINED ENABLE_USER_LINKER OR NOT ENABLE_USER_LINKER)
        return()
    endif()

    set(LINKER_FLAG "-fuse-ld=${USER_LINKER_OPTION}")

    check_cxx_compiler_flag(${LINKER_FLAG} CXX_SUPPORTS_USER_LINKER)
    if(CXX_SUPPORTS_USER_LINKER)
        target_compile_options(${project_name} INTERFACE ${LINKER_FLAG})
    endif()
endmacro()
