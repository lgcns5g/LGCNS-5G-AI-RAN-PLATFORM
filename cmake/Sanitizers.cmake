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

function(
    enable_sanitizers
    project_name
    ENABLE_SANITIZER_ADDRESS
    ENABLE_SANITIZER_LEAK
    ENABLE_SANITIZER_UNDEFINED_BEHAVIOR
    ENABLE_SANITIZER_THREAD
    ENABLE_SANITIZER_MEMORY)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(SANITIZERS "")

        if(${ENABLE_SANITIZER_ADDRESS})
            list(APPEND SANITIZERS "address")
        endif()

        if(${ENABLE_SANITIZER_LEAK})
            list(APPEND SANITIZERS "leak")
        endif()

        if(${ENABLE_SANITIZER_UNDEFINED_BEHAVIOR})
            list(APPEND SANITIZERS "undefined")
        endif()

        if(${ENABLE_SANITIZER_THREAD})
            if("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS)
                message(
                    WARNING "Thread sanitizer does not work with Address and Leak sanitizer enabled"
                )
            else()
                list(APPEND SANITIZERS "thread")
            endif()
        endif()

        if(${ENABLE_SANITIZER_MEMORY} AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
            message(
                WARNING
                    "Memory sanitizer requires all the code (including libc++) to be MSan-instrumented otherwise it reports false positives"
            )
            if("address" IN_LIST SANITIZERS
               OR "thread" IN_LIST SANITIZERS
               OR "leak" IN_LIST SANITIZERS)
                message(
                    WARNING
                        "Memory sanitizer does not work with Address, Thread or Leak sanitizer enabled"
                )
            else()
                list(APPEND SANITIZERS "memory")
            endif()
        endif()
    elseif(MSVC)
        if(${ENABLE_SANITIZER_ADDRESS})
            list(APPEND SANITIZERS "address")
        endif()
        if(${ENABLE_SANITIZER_LEAK}
           OR ${ENABLE_SANITIZER_UNDEFINED_BEHAVIOR}
           OR ${ENABLE_SANITIZER_THREAD}
           OR ${ENABLE_SANITIZER_MEMORY})
            message(WARNING "MSVC only supports address sanitizer")
        endif()
    endif()

    list(JOIN SANITIZERS "," LIST_OF_SANITIZERS)

    if(LIST_OF_SANITIZERS)
        if(NOT "${LIST_OF_SANITIZERS}" STREQUAL "")
            if(NOT MSVC)
                # Apply sanitizer flags only to C and C++ languages, not CUDA. CUDA/nvcc has issues
                # with sanitizer flags during device linking
                target_compile_options(
                    ${project_name}
                    INTERFACE $<$<COMPILE_LANGUAGE:C>:-fsanitize=${LIST_OF_SANITIZERS}>
                              $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${LIST_OF_SANITIZERS}>)
                target_link_options(
                    ${project_name} INTERFACE
                    $<$<COMPILE_LANGUAGE:C>:-fsanitize=${LIST_OF_SANITIZERS}>
                    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${LIST_OF_SANITIZERS}>)
            else()
                string(FIND "$ENV{PATH}" "$ENV{VSINSTALLDIR}" index_of_vs_install_dir)
                if("${index_of_vs_install_dir}" STREQUAL "-1")
                    message(
                        SEND_ERROR
                            "Using MSVC sanitizers requires setting the MSVC environment before building the project. Please manually open the MSVC command prompt and rebuild the project."
                    )
                endif()
                target_compile_options(${project_name} INTERFACE /fsanitize=${LIST_OF_SANITIZERS}
                                                                 /Zi /INCREMENTAL:NO)
                target_compile_definitions(${project_name} INTERFACE _DISABLE_VECTOR_ANNOTATION
                                                                     _DISABLE_STRING_ANNOTATION)
                target_link_options(${project_name} INTERFACE /INCREMENTAL:NO)
            endif()
        endif()
    endif()

endfunction()
