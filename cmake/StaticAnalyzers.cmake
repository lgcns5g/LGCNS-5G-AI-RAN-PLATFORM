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

macro(enable_cppcheck WARNINGS_AS_ERRORS CPPCHECK_OPTIONS)
    find_program(CPPCHECK cppcheck)
    if(CPPCHECK)

        if(CMAKE_GENERATOR MATCHES ".*Visual Studio.*")
            set(CPPCHECK_TEMPLATE "vs")
        else()
            set(CPPCHECK_TEMPLATE "gcc")
        endif()

        if("${CPPCHECK_OPTIONS}" STREQUAL "")
            # Enable all warnings that are actionable by the user of this toolset style should
            # enable the other 3, but we'll be explicit just in case
            set(SUPPRESS_DIR "*:${CMAKE_CURRENT_BINARY_DIR}/_deps/*.h")
            set(SUPPRESS_DIR_HPP "*:${CMAKE_CURRENT_BINARY_DIR}/_deps/*.hpp")
            set(SUPPRESS_DPDK "*:/opt/mellanox/dpdk/include/dpdk/*.h")
            set(SUPPRESS_DPDK_ARCH "*:/opt/mellanox/dpdk/include/*/dpdk/*.h")
            set(SUPPRESS_DOCA "*:/opt/mellanox/doca/include/*.h")
            set(SUPPRESS_TENSORRT "*:/usr/include/x86_64-linux-gnu/NvInfer*.h")
            set(CMAKE_CXX_CPPCHECK
                ${CPPCHECK}
                --template=${CPPCHECK_TEMPLATE}
                --enable=style,performance,warning,portability
                --inline-suppr
                # We cannot act on a bug/missing feature of cppcheck
                --suppress=cppcheckError
                --suppress=internalAstError
                # if a file does not have an internalAstError, we get an unmatchedSuppression error
                --suppress=unmatchedSuppression
                # noisy and incorrect sometimes
                --suppress=passedByValue
                # ignores code that cppcheck thinks is invalid C++
                --suppress=syntaxError
                --suppress=preprocessorErrorDirective
                # ignores static_assert type failures
                --suppress=knownConditionTrueFalse
                # ignores unknown macros from third-party libraries like wise_enum
                --suppress=unknownMacro
                --inconclusive
                --suppress=${SUPPRESS_DIR}
                --suppress=${SUPPRESS_DIR_HPP}
                --suppress=${SUPPRESS_DPDK}
                --suppress=${SUPPRESS_DPDK_ARCH}
                --suppress=${SUPPRESS_DOCA}
                --suppress=${SUPPRESS_TENSORRT})
        else()
            # if the user provides a CPPCHECK_OPTIONS with a template specified, it will override
            # this template
            set(CMAKE_CXX_CPPCHECK ${CPPCHECK} --template=${CPPCHECK_TEMPLATE} ${CPPCHECK_OPTIONS})
        endif()

        if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
            set(CMAKE_CXX_CPPCHECK ${CMAKE_CXX_CPPCHECK} --std=c++${CMAKE_CXX_STANDARD})
        endif()
        if(${WARNINGS_AS_ERRORS})
            list(APPEND CMAKE_CXX_CPPCHECK --error-exitcode=2)
        endif()
    else()
        message(WARNING "cppcheck requested but executable not found")
    endif()
endmacro()

macro(enable_clang_tidy target WARNINGS_AS_ERRORS)

    find_program(CLANGTIDY clang-tidy)
    if(CLANGTIDY)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")

            get_target_property(TARGET_PCH ${target} INTERFACE_PRECOMPILE_HEADERS)

            if("${TARGET_PCH}" STREQUAL "TARGET_PCH-NOTFOUND")
                get_target_property(TARGET_PCH ${target} PRECOMPILE_HEADERS)
            endif()

            if(NOT ("${TARGET_PCH}" STREQUAL "TARGET_PCH-NOTFOUND"))
                message(
                    SEND_ERROR
                        "clang-tidy cannot be enabled with non-clang compiler and PCH, clang-tidy fails to handle gcc's PCH file"
                )
            endif()
        endif()

        # construct the clang-tidy command line
        set(CLANG_TIDY_OPTIONS
            ${CLANGTIDY} -extra-arg=-Wno-unknown-warning-option
            -extra-arg=-Wno-ignored-optimization-argument
            -extra-arg=-Wno-unused-command-line-argument -p)
        # set standard
        if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
            if("${CLANG_TIDY_OPTIONS_DRIVER_MODE}" STREQUAL "cl")
                set(CLANG_TIDY_OPTIONS ${CLANG_TIDY_OPTIONS}
                                       -extra-arg=/std:c++${CMAKE_CXX_STANDARD})
            else()
                set(CLANG_TIDY_OPTIONS ${CLANG_TIDY_OPTIONS}
                                       -extra-arg=-std=c++${CMAKE_CXX_STANDARD})
            endif()
        endif()

        # set warnings as errors
        if(${WARNINGS_AS_ERRORS})
            list(APPEND CLANG_TIDY_OPTIONS -warnings-as-errors=*)
        endif()

        message(STATUS "Also setting clang-tidy globally")
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_OPTIONS})
    else()
        message(WARNING "clang-tidy requested but executable not found")
    endif()
endmacro()
