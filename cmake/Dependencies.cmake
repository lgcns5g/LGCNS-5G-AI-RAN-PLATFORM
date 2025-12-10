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

# Set default CPM source cache if not already configured Users can override via: cmake
# -DCPM_SOURCE_CACHE=/custom/path or export CPM_SOURCE_CACHE=/custom/path
if(NOT DEFINED CPM_SOURCE_CACHE AND NOT DEFINED ENV{CPM_SOURCE_CACHE})
    set(ENV{CPM_SOURCE_CACHE} "$ENV{HOME}/.cache/CPM")
    message(STATUS "CPM source cache: $ENV{HOME}/.cache/CPM (default)")
else()
    if(DEFINED CPM_SOURCE_CACHE)
        message(STATUS "CPM source cache: ${CPM_SOURCE_CACHE} (CMake variable)")
    else()
        message(STATUS "CPM source cache: $ENV{CPM_SOURCE_CACHE} (environment variable)")
    endif()
endif()

include(cmake/CPM.cmake)
include(cmake/CpuArchitecture.cmake)
include(cmake/Acar.cmake)

# Done as a function so that updates to variables like CMAKE_CXX_FLAGS don't propagate out to other
# targets
function(setup_dependencies)

    # For each dependency, see if it's already been provided to us by a parent project

    # Use `@` for tag with v prefix, `#` for non v prefix in tag.

    if(BUILD_TESTING)
        if(NOT TARGET GTest::gtest_main)
            cpmaddpackage(
                NAME
                googletest
                GITHUB_REPOSITORY
                google/googletest
                GIT_TAG
                v1.15.2
                SYSTEM
                YES)
        endif()

        if(NOT TARGET benchmark::benchmark)
            cpmaddpackage(
                NAME
                benchmark
                GITHUB_REPOSITORY
                google/benchmark
                GIT_TAG
                v1.9.4
                OPTIONS
                "BENCHMARK_ENABLE_TESTING OFF"
                "BENCHMARK_ENABLE_GTEST_TESTS OFF"
                SYSTEM
                YES)
        endif()

        if(NOT TARGET accessor::accessor)
            include(${CMAKE_SOURCE_DIR}/cmake/CppMemberAccessor.cmake)
            setup_accessor()
        endif()
    endif() # BUILD_TESTING

    if(NOT TARGET CLI11::CLI11)
        cpmaddpackage(
            NAME
            CLI11
            GITHUB_REPOSITORY
            CLIUtils/CLI11
            GIT_TAG
            v2.5.0
            SYSTEM
            YES)
    endif()

    if(NOT TARGET quill::quill)
        cpmaddpackage(
            NAME
            quill
            GITHUB_REPOSITORY
            odygrd/quill
            GIT_TAG
            v10.0.1
            OPTIONS
            "QUILL_BUILD_TESTS OFF"
            "QUILL_BUILD_BENCHMARKS OFF"
            "QUILL_BUILD_EXAMPLES OFF"
            SYSTEM
            YES)
    endif()

    if(NOT TARGET wise_enum)
        cpmaddpackage(
            NAME
            WiseEnum
            GITHUB_REPOSITORY
            quicknir/wise_enum
            GIT_TAG
            3.1.0
            OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_EXAMPLES OFF"
            SYSTEM
            YES)

        # WiseEnum only sets INSTALL_INTERFACE includes, add BUILD_INTERFACE for CPM usage
        if(TARGET wise_enum)
            target_include_directories(wise_enum SYSTEM
                                       INTERFACE $<BUILD_INTERFACE:${WiseEnum_SOURCE_DIR}>)

            # Create lowercase alias for compatibility
            if(NOT TARGET wise_enum::wise_enum)
                add_library(wise_enum::wise_enum ALIAS wise_enum)
            endif()
        endif()
    endif()

    # backward-cpp - stack trace library
    if(NOT TARGET Backward::Backward)
        cpmaddpackage(
            NAME
            Backward
            GITHUB_REPOSITORY
            bombela/backward-cpp
            GIT_TAG
            v1.6
            SYSTEM
            YES)

        # Backward::Backward should be created by BackwardConfig.cmake If not, create it manually
        # from the backward library target
        if(NOT TARGET Backward::Backward AND TARGET backward)
            add_library(Backward::Backward ALIAS backward)
        endif()
    endif()

    if(NOT TARGET gsl::gsl-lite)
        cpmaddpackage(
            NAME
            gsl-lite
            GITHUB_REPOSITORY
            gsl-lite/gsl-lite
            GIT_TAG
            v1.0.1
            SYSTEM
            YES)
    endif()

    # Setup GDRCopy for GPU Direct RDMA memory copy
    if(NOT TARGET gdrcopy::gdrcopy)
        include(${CMAKE_SOURCE_DIR}/cmake/GDRCopy.cmake)
        setup_gdrcopy()
    endif()

    # Setup MathDx for GPU-accelerated operations (REQUIRED) Provides: cuFFTDx (FFT), cuBLASDx
    # (BLAS), cuRANDDx (random), cuSOLVERDx (solver) Note: Use target_link_mathdx() with explicit
    # component arguments to link header-only components (e.g., cufftdx, cublasdx, curanddx)
    if(NOT TARGET mathdx::cufftdx)
        include(${CMAKE_SOURCE_DIR}/cmake/MathDx.cmake)
    endif()

    # Setup TensorRT nvinfer library for GPU inference operations Provides nvinfer::nvinfer target
    # (use target_link_nvinfer())
    if(NOT TARGET nvinfer::nvinfer)
        include(${CMAKE_SOURCE_DIR}/cmake/Nvinfer.cmake)
    endif()

    if(NOT TARGET NamedType)
        cpmaddpackage(
            NAME
            NamedType
            GITHUB_REPOSITORY
            joboccara/NamedType
            GIT_TAG
            v1.1.0
            OPTIONS
            "ENABLE_TEST OFF"
            SYSTEM
            YES)
    endif()

    # Header-only parallel hashmap library
    if(NOT TARGET phmap::phmap)
        cpmaddpackage(
            NAME
            parallel_hashmap
            GITHUB_REPOSITORY
            greg7mdp/parallel-hashmap
            GIT_TAG
            v2.0.0
            OPTIONS
            "PHMAP_BUILD_TESTS OFF"
            "PHMAP_BUILD_EXAMPLES OFF"
            DOWNLOAD_ONLY
            YES
            SYSTEM
            YES)

        # Manually create an interface target and propagate the include directory
        add_library(phmap INTERFACE)
        target_include_directories(phmap SYSTEM INTERFACE "${parallel_hashmap_SOURCE_DIR}/")

        # Optional: provide an alias so downstreams can use phmap::phmap
        add_library(phmap::phmap ALIAS phmap)
    endif()

    # libyaml C library (needed for yaml.hpp wrapper)
    if(NOT TARGET libyaml::libyaml)
        cpmaddpackage(
            NAME
            libyaml
            GITHUB_REPOSITORY
            yaml/libyaml
            GIT_TAG
            0.2.5
            OPTIONS
            "BUILD_TESTING OFF"
            SYSTEM
            YES)

        # Create a stable alias whatever target name upstream uses
        if(TARGET yaml)
            add_library(libyaml::libyaml ALIAS yaml)
        elseif(TARGET yaml_static)
            add_library(libyaml::libyaml ALIAS yaml_static)
        elseif(TARGET libyaml)
            add_library(libyaml::libyaml ALIAS libyaml)
        endif()
    endif()

    # YAML-CPP for YAML parsing (needed by nvlog shim headers)
    if(NOT TARGET yaml-cpp)
        cpmaddpackage(
            NAME
            yaml-cpp
            GITHUB_REPOSITORY
            jbeder/yaml-cpp
            GIT_TAG
            0.8.0
            OPTIONS
            "YAML_CPP_BUILD_TESTS OFF"
            "YAML_CPP_BUILD_TOOLS OFF"
            "YAML_BUILD_SHARED_LIBS OFF"
            "YAML_CPP_FORMAT_SOURCE OFF"
            "YAML_CPP_INSTALL OFF"
            SYSTEM
            YES)

        # Provide common alias if upstream doesn't
        if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
            add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
        endif()
    endif()

    # fmt library for formatting (needed by aerial_sdk's nvlog)
    if(NOT TARGET fmt::fmt)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        cpmaddpackage(
            NAME
            fmt
            GITHUB_REPOSITORY
            fmtlib/fmt
            GIT_TAG
            10.2.1
            OPTIONS
            "BUILD_SHARED_LIBS ON"
            "FMT_INSTALL OFF"
            SYSTEM
            YES)
    endif()

    # Download aerial_sdk source to access fmtlog patch file
    download_acar_source()
    download_ldpc_decoder_cubin()

    # Setup fmtlog with patches from aerial_sdk
    if(NOT TARGET fmtlog::fmtlog)
        include(${CMAKE_SOURCE_DIR}/cmake/Fmtlog.cmake)
        setup_fmtlog(${aerial_sdk_SOURCE_DIR})
    endif()

    # Configure and add aerial_sdk as subdirectory (now that fmt and fmtlog are available)
    setup_acar_targets()

    # clang-format and cmake-format tools
    cpmaddpackage(
        NAME
        Format.cmake
        GITHUB_REPOSITORY
        TheLartians/Format.cmake
        GIT_TAG
        v1.8.2
        SYSTEM
        YES)

    # Configure the formatting tool to exclude json files, which do not seem to be supported by
    # clang format. Enable other extensions.
    find_package(Git REQUIRED)

    # Check if we're in a git repo first
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --is-inside-work-tree
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE IN_REPO
        OUTPUT_QUIET ERROR_QUIET)

    if(IN_REPO EQUAL 0)
        set(CLANGFORMAT_EXTENSIONS
            "c,h,m,mm,cc,cp,cpp,c++,cxx,hh,hpp,hxx,inc,inl,ccm,cppm,cxxm,c++m,cu,cuh,proto,protodevel,java,js,ts,cs,sv,svh,v,vh"
        )

        # Check if clangformat.extensions is already set correctly
        execute_process(
            COMMAND ${GIT_EXECUTABLE} config --get clangformat.extensions
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE GET_RESULT
            OUTPUT_VARIABLE CURRENT_EXTENSIONS
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

        if(GET_RESULT EQUAL 0 AND CURRENT_EXTENSIONS STREQUAL CLANGFORMAT_EXTENSIONS)
            message(STATUS "clangformat.extensions already configured correctly")
        endif()

        # Try to set if not configured (ignore lock failures - another parallel build will succeed)
        if(NOT GET_RESULT EQUAL 0 OR NOT CURRENT_EXTENSIONS STREQUAL CLANGFORMAT_EXTENSIONS)
            message(STATUS "Setting clangformat.extensions in git config...")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} config clangformat.extensions "${CLANGFORMAT_EXTENSIONS}"
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                RESULT_VARIABLE SET_RESULT
                ERROR_QUIET)

            # Verify it got set (either by us or another parallel build)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} config --get clangformat.extensions
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                RESULT_VARIABLE VERIFY_RESULT
                OUTPUT_VARIABLE FINAL_EXTENSIONS
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

            # Fatal only if nobody managed to set it correctly
            if(NOT VERIFY_RESULT EQUAL 0 OR NOT FINAL_EXTENSIONS STREQUAL CLANGFORMAT_EXTENSIONS)
                message(
                    FATAL_ERROR
                        "clangformat.extensions not set correctly. "
                        "This is required to prevent formatting JSON files which causes failures. "
                        "Expected: ${CLANGFORMAT_EXTENSIONS}, Got: ${FINAL_EXTENSIONS}")
            endif()
        endif()
    endif()

    # Range library for C++14/17/20 Link using: target_link_system_libraries(your_target
    # range-v3::range-v3) Include using: #include <range/v3/all.hpp>
    if(NOT TARGET range-v3::range-v3)
        cpmaddpackage(
            NAME
            range-v3
            GITHUB_REPOSITORY
            ericniebler/range-v3
            GIT_TAG
            ca1388fb9da8e69314dda222dc7b139ca84e092f
            OPTIONS
            "RANGE_V3_TESTS OFF"
            "RANGE_V3_EXAMPLES OFF"
            "RANGE_V3_PERF OFF"
            "RANGE_V3_HEADER_CHECKS OFF"
            "RANGE_V3_DOCS OFF"
            "RANGE_V3_INSTALL ON"
            SYSTEM
            YES)
    endif()

    # C++11/14/17 std::expected implementation with functional-style extensions Link using:
    # target_link_system_libraries(your_target tl::expected) Include using: #include
    # <tl/expected.hpp>
    if(NOT TARGET tl::expected)
        cpmaddpackage(
            NAME
            expected
            GITHUB_REPOSITORY
            TartanLlama/expected
            GIT_TAG
            v1.3.1
            OPTIONS
            "EXPECTED_BUILD_TESTS OFF"
            SYSTEM
            YES)

        # Create modern alias if upstream doesn't provide it
        if(TARGET expected AND NOT TARGET tl::expected)
            add_library(tl::expected ALIAS expected)
        endif()
    endif()

    # Create user-friendly format checking targets
    include(${CMAKE_SOURCE_DIR}/cmake/FormatTargets.cmake)
endfunction()
