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

macro(enable_ipo)
    include(CheckIPOSupported)

    check_ipo_supported(
        RESULT c_ipo_supported
        LANGUAGES C
        OUTPUT c_output)
    if(c_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_C ON)
        message(STATUS "IPO enabled for C language")
    else()
        message(WARNING "IPO not supported for C: ${c_output}")
    endif()

    check_ipo_supported(
        RESULT cxx_ipo_supported
        LANGUAGES CXX
        OUTPUT cxx_output)
    if(cxx_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_CXX ON)
        message(STATUS "IPO enabled for C++ language")
    else()
        message(WARNING "IPO not supported for C++: ${cxx_output}")
    endif()

    # Only check CUDA IPO when CUDA is enabled
    if(CMAKE_CUDA_COMPILER)
        check_ipo_supported(
            RESULT cuda_ipo_supported
            LANGUAGES CUDA
            OUTPUT cuda_output)
        if(cuda_ipo_supported)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_CUDA ON)
            message(STATUS "IPO enabled for CUDA language")
        else()
            message(WARNING "IPO not supported for CUDA: ${cuda_output}")
        endif()
    endif()
endmacro()
