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

# Simplified GDRCopy setup using Makefile build
function(setup_gdrcopy)
    if(NOT TARGET gdrcopy::gdrcopy)
        find_package(CUDAToolkit REQUIRED)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_BIN_DIR} DIRECTORY)

        if(NOT COMMAND cpmaddpackage)
            message(
                FATAL_ERROR
                    "CPM not available. Call setup_gdrcopy() from within setup_dependencies().")
        endif()

        cpmaddpackage(
            NAME
            gdrcopy
            GITHUB_REPOSITORY
            NVIDIA/gdrcopy
            GIT_TAG
            v2.5
            DOWNLOAD_ONLY
            YES
            SYSTEM
            YES)

        if(gdrcopy_ADDED)
            # Build and install to build tree (no sudo needed)
            set(GDRCOPY_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/gdrcopy-install)
            set(GDRCOPY_LIB_FILE ${GDRCOPY_INSTALL_DIR}/lib/libgdrapi.so)

            add_custom_command(
                OUTPUT ${GDRCOPY_LIB_FILE}
                COMMAND make CUDA=${CUDA_TOOLKIT_ROOT_DIR} lib
                COMMAND make prefix=${GDRCOPY_INSTALL_DIR} CUDA=${CUDA_TOOLKIT_ROOT_DIR} lib_install
                WORKING_DIRECTORY ${gdrcopy_SOURCE_DIR}
                COMMENT "Building and installing GDRCopy library"
                VERBATIM)

            # Create IMPORTED SHARED library target for libgdrapi
            add_library(gdrcopy_libgdrapi SHARED IMPORTED)
            set_target_properties(gdrcopy_libgdrapi PROPERTIES IMPORTED_LOCATION
                                                               ${GDRCOPY_LIB_FILE})

            # Create interface library that links to the imported target
            add_library(gdrcopy INTERFACE)
            target_include_directories(gdrcopy SYSTEM INTERFACE ${GDRCOPY_INSTALL_DIR}/include)
            target_link_system_libraries(gdrcopy INTERFACE gdrcopy_libgdrapi)

            # Add dependency on the build command
            add_custom_target(gdrcopy_build DEPENDS ${GDRCOPY_LIB_FILE})
            add_dependencies(gdrcopy_libgdrapi gdrcopy_build)
            add_dependencies(gdrcopy gdrcopy_build)

            # Create alias
            add_library(gdrcopy::gdrcopy ALIAS gdrcopy)

            message(STATUS "GDRCopy will be built and installed to: ${GDRCOPY_INSTALL_DIR}")
        endif()
    endif()
endfunction()

# Simplified linking function
function(target_link_gdrcopy target)
    if(NOT TARGET gdrcopy::gdrcopy)
        message(FATAL_ERROR "GDRCopy not available for ${target}. Call setup_gdrcopy() first.")
    endif()

    target_link_system_libraries(${target} PRIVATE gdrcopy::gdrcopy)
    target_compile_definitions(${target} PRIVATE HAVE_GDRCOPY=1)
endfunction()
