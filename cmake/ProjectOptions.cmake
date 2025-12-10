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

include(cmake/SystemLink.cmake)
include(cmake/LibFuzzer.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

include(CheckCXXSourceCompiles)

macro(supports_sanitizers)
    if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*")
       AND NOT WIN32)

        message(
            STATUS
                "Sanity checking UndefinedBehaviorSanitizer, it should be supported on this platform"
        )
        set(TEST_PROGRAM "int main() { return 0; }")

        # Check if UndefinedBehaviorSanitizer works at link time
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined")
        set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
        check_cxx_source_compiles("${TEST_PROGRAM}" HAS_UBSAN_LINK_SUPPORT)

        if(HAS_UBSAN_LINK_SUPPORT)
            message(STATUS "UndefinedBehaviorSanitizer is supported at both compile and link time.")
            set(SUPPORTS_UBSAN ON)
        else()
            message(WARNING "UndefinedBehaviorSanitizer is NOT supported at link time.")
            set(SUPPORTS_UBSAN OFF)
        endif()
    else()
        set(SUPPORTS_UBSAN OFF)
    endif()

    if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*")
       AND WIN32)
        set(SUPPORTS_ASAN OFF)
    else()
        if(NOT WIN32)
            message(
                STATUS "Sanity checking AddressSanitizer, it should be supported on this platform")
            set(TEST_PROGRAM "int main() { return 0; }")

            # Check if AddressSanitizer works at link time
            set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
            set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address")
            check_cxx_source_compiles("${TEST_PROGRAM}" HAS_ASAN_LINK_SUPPORT)

            if(HAS_ASAN_LINK_SUPPORT)
                message(STATUS "AddressSanitizer is supported at both compile and link time.")
                set(SUPPORTS_ASAN ON)
            else()
                message(WARNING "AddressSanitizer is NOT supported at link time.")
                set(SUPPORTS_ASAN OFF)
            endif()
        else()
            set(SUPPORTS_ASAN ON)
        endif()
    endif()
endmacro()

macro(setup_options)
    option(MAINTAINER_MODE "Enable maintainer mode with strict checks" OFF)
    option(ENABLE_HARDENING "Enable hardening" ON)
    option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
    option(AERIAL_TENSOR_BOUNDS_CHECK "Enable tensor bounds checking for safety" ON)
    option(GSL_CONTRACT_VIOLATION_THROWS
           "Configure GSL contract violations to throw exceptions instead of asserting" ON)
    option(SKIP_ACAR_DOWNLOAD "Skip downloading aerial_sdk if it already exists" OFF)
    cmake_dependent_option(
        ENABLE_GLOBAL_HARDENING "Attempt to push hardening options to built dependencies" ON
        ENABLE_HARDENING OFF)

    supports_sanitizers()

    if(NOT PROJECT_IS_TOP_LEVEL OR NOT MAINTAINER_MODE)
        option(ENABLE_IPO "Enable IPO/LTO" ON)
        option(WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
        option(ENABLE_USER_LINKER "Enable user-selected linker" OFF)
        option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
        option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
        option(ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
        option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
        option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
        option(ENABLE_UNITY_BUILD "Enable unity builds" OFF)
        option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
        option(ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
        option(ENABLE_IWYU "Enable include what you use" OFF)
        option(ENABLE_PCH "Enable precompiled headers" OFF)
        option(ENABLE_CACHE "Enable ccache" ON)
        option(BUILD_TESTING "Enable testing" ON)
        option(ENFORCE_DOCSTRINGS "Enforce documentation strings" OFF)
        option(BUILD_DOCS "Build documentation" ON)
        option(NVTX_ENABLE "Enable NVTX" ON)
        message(STATUS "Not building in maintainer mode")
    else()
        option(ENABLE_IPO "Enable IPO/LTO" ON)
        option(WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
        option(ENABLE_USER_LINKER "Enable user-selected linker" OFF)
        option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
        option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
        option(ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
        option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
        option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
        option(ENABLE_UNITY_BUILD "Enable unity builds" OFF)
        option(ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
        option(ENABLE_CPPCHECK "Enable cpp-check analysis" ON)
        option(ENABLE_IWYU "Enable include what you use" ON)
        option(ENABLE_PCH "Enable precompiled headers" OFF)
        option(ENABLE_CACHE "Enable ccache" ON)
        option(BUILD_TESTING "Enable testing" ON)
        option(ENFORCE_DOCSTRINGS "Enforce documentation strings" ON)
        option(BUILD_DOCS "Build documentation" ON)
        option(NVTX_ENABLE "Enable NVTX" ON)
        message(STATUS "Building in maintainer mode")
    endif()

    if(NOT PROJECT_IS_TOP_LEVEL)
        mark_as_advanced(
            ENABLE_IPO
            WARNINGS_AS_ERRORS
            ENABLE_USER_LINKER
            ENABLE_SANITIZER_ADDRESS
            ENABLE_SANITIZER_LEAK
            ENABLE_SANITIZER_UNDEFINED
            ENABLE_SANITIZER_THREAD
            ENABLE_SANITIZER_MEMORY
            ENABLE_UNITY_BUILD
            ENABLE_CLANG_TIDY
            ENABLE_CPPCHECK
            ENABLE_COVERAGE
            ENABLE_PCH
            ENABLE_CACHE
            ENFORCE_DOCSTRINGS
            BUILD_DOCS
            AERIAL_TENSOR_BOUNDS_CHECK
            SKIP_ACAR_DOWNLOAD)
    endif()

    check_libfuzzer_support(LIBFUZZER_SUPPORTED)
    if(LIBFUZZER_SUPPORTED
       AND (ENABLE_SANITIZER_ADDRESS
            OR ENABLE_SANITIZER_THREAD
            OR ENABLE_SANITIZER_UNDEFINED))
        set(DEFAULT_FUZZER ON)
    else()
        set(DEFAULT_FUZZER OFF)
    endif()

    option(BUILD_FUZZ_TESTS "Enable fuzz testing executable" ${DEFAULT_FUZZER})

endmacro()

macro(global_options)
    if(ENABLE_IPO)
        include(cmake/InterproceduralOptimization.cmake)
        enable_ipo()
    endif()

    supports_sanitizers()

    if(ENABLE_HARDENING AND ENABLE_GLOBAL_HARDENING)
        include(cmake/Hardening.cmake)
        if(NOT SUPPORTS_UBSAN
           OR ENABLE_SANITIZER_UNDEFINED
           OR ENABLE_SANITIZER_ADDRESS
           OR ENABLE_SANITIZER_THREAD
           OR ENABLE_SANITIZER_LEAK)
            set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
        else()
            set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
        endif()
        message("${ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${ENABLE_SANITIZER_UNDEFINED}")
        enable_hardening(framework_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
    endif()
endmacro()

macro(local_options)
    if(PROJECT_IS_TOP_LEVEL)
        include(cmake/StandardProjectSettings.cmake)
    endif()

    add_library(framework_warnings INTERFACE)
    add_library(framework_options INTERFACE)

    include(cmake/CompilerWarnings.cmake)
    set_project_warnings(framework_warnings ${WARNINGS_AS_ERRORS} "" "" "" "")

    if(ENABLE_USER_LINKER)
        include(cmake/Linker.cmake)
        configure_linker(framework_options)
    endif()

    include(cmake/Sanitizers.cmake)
    enable_sanitizers(
        framework_options ${ENABLE_SANITIZER_ADDRESS} ${ENABLE_SANITIZER_LEAK}
        ${ENABLE_SANITIZER_UNDEFINED} ${ENABLE_SANITIZER_THREAD} ${ENABLE_SANITIZER_MEMORY})

    set_target_properties(framework_options PROPERTIES UNITY_BUILD ${ENABLE_UNITY_BUILD})

    if(ENABLE_PCH)
        target_precompile_headers(framework_options INTERFACE <vector> <string> <utility>)
    endif()

    if(ENABLE_CACHE)
        include(cmake/Cache.cmake)
        enable_cache()
    endif()

    if(ENABLE_COVERAGE)
        include(cmake/Tests.cmake)
        enable_coverage(framework_options)
        setup_coverage_targets()
    endif()

    if(WARNINGS_AS_ERRORS)
        check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
        if(LINKER_FATAL_WARNINGS)
            # This is not working consistently, so disabling for now
            # target_link_options(framework_options INTERFACE -Wl,--fatal-warnings)
        endif()
    endif()

    if(ENABLE_HARDENING AND NOT ENABLE_GLOBAL_HARDENING)
        include(cmake/Hardening.cmake)
        if(NOT SUPPORTS_UBSAN
           OR ENABLE_SANITIZER_UNDEFINED
           OR ENABLE_SANITIZER_ADDRESS
           OR ENABLE_SANITIZER_THREAD
           OR ENABLE_SANITIZER_LEAK)
            set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
        else()
            set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
        endif()
        enable_hardening(framework_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
    endif()

    # Configure tensor bounds checking
    if(AERIAL_TENSOR_BOUNDS_CHECK)
        target_compile_definitions(framework_options INTERFACE AERIAL_TENSOR_BOUNDS_CHECK=1)
        message(STATUS "Tensor bounds checking is enabled")
    else()
        target_compile_definitions(framework_options INTERFACE AERIAL_TENSOR_BOUNDS_CHECK=0)
        message(STATUS "Tensor bounds checking is disabled")
    endif()

    # Configure GSL contract violation behavior. When enabled, gsl_Expects/gsl_Ensures throw
    # exceptions; when disabled, they assert
    if(GSL_CONTRACT_VIOLATION_THROWS)
        target_compile_definitions(framework_options INTERFACE gsl_CONFIG_CONTRACT_VIOLATION_THROWS)
        message(STATUS "GSL contract violations will throw exceptions")
    else()
        target_compile_definitions(framework_options
                                   INTERFACE gsl_CONFIG_CONTRACT_VIOLATION_ASSERTS)
        message(STATUS "GSL contract violations will assert")
    endif()
endmacro()

macro(enable_static_analysis)
    include(cmake/StaticAnalyzers.cmake)
    include(cmake/IncludeWhatYouUse.cmake)
    if(ENABLE_CLANG_TIDY)
        enable_clang_tidy(framework_options ${WARNINGS_AS_ERRORS})
    endif()

    if(ENABLE_CPPCHECK)
        enable_cppcheck(${WARNINGS_AS_ERRORS} "" # override cppcheck options
        )
    endif()

    if(ENABLE_IWYU)
        enable_include_what_you_use()
    endif()
endmacro()
