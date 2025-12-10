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

# Create a common library setup function to reduce duplication
function(add_task_library)
    set(options "")
    set(oneValueArgs TARGET_NAME NAMESPACE EXPORT_HEADER_NAME STATIC_DEFINE_NAME)
    set(multiValueArgs SOURCES PUBLIC_HEADERS SYSTEM_LIBRARIES LIBRARIES COMPILE_DEFINITIONS)

    cmake_parse_arguments(LIB "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT LIB_TARGET_NAME)
        message(FATAL_ERROR "TARGET_NAME is required")
    endif()

    if(NOT LIB_NAMESPACE)
        message(FATAL_ERROR "NAMESPACE is required")
    endif()

    # Set default values
    if(NOT LIB_EXPORT_HEADER_NAME)
        set(LIB_EXPORT_HEADER_NAME "${LIB_TARGET_NAME}_export.hpp")
    endif()

    if(NOT LIB_STATIC_DEFINE_NAME)
        string(TOUPPER "${LIB_TARGET_NAME}" UPPER_TARGET_NAME)
        set(LIB_STATIC_DEFINE_NAME "${UPPER_TARGET_NAME}_STATIC_DEFINE")
    endif()

    include(GenerateExportHeader)

    # Create the library
    add_library(${LIB_TARGET_NAME} STATIC)
    add_library(${LIB_NAMESPACE} ALIAS ${LIB_TARGET_NAME})

    # Add sources
    if(LIB_SOURCES OR LIB_PUBLIC_HEADERS)
        target_sources(
            ${LIB_TARGET_NAME}
            PRIVATE ${LIB_SOURCES}
            PUBLIC FILE_SET HEADERS BASE_DIRS include FILES ${LIB_PUBLIC_HEADERS})
    endif()

    # Link system libraries
    if(LIB_SYSTEM_LIBRARIES)
        target_link_system_libraries(${LIB_TARGET_NAME} PUBLIC ${LIB_SYSTEM_LIBRARIES})
    endif()

    # Link regular libraries
    if(LIB_LIBRARIES)
        target_link_libraries(${LIB_TARGET_NAME} PUBLIC ${LIB_LIBRARIES})
    endif()

    # Set linker language
    set_target_properties(${LIB_TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

    # Add compile definitions
    if(LIB_COMPILE_DEFINITIONS)
        target_compile_definitions(${LIB_TARGET_NAME} PUBLIC ${LIB_COMPILE_DEFINITIONS})
    endif()

    # Install targets
    install(
        TARGETS ${LIB_TARGET_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

    # Install headers
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/task
        DESTINATION include
        COMPONENT dev
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "*.cuh")

    # Install sources
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION src/aerial-framework/${LIB_TARGET_NAME}
        COMPONENT src)

    # Set include directories
    target_include_directories(
        ${LIB_TARGET_NAME}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
               $<INSTALL_INTERFACE:include/task>)

    # Set target properties
    set_target_properties(
        ${LIB_TARGET_NAME}
        PROPERTIES VERSION ${PROJECT_VERSION}
                   CXX_VISIBILITY_PRESET hidden
                   VISIBILITY_INLINES_HIDDEN YES)

    # Generate export header
    generate_export_header(${LIB_TARGET_NAME} EXPORT_FILE_NAME
                           ${CMAKE_CURRENT_BINARY_DIR}/include/task/${LIB_EXPORT_HEADER_NAME})

    # Handle static library definitions
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(${LIB_TARGET_NAME} PUBLIC ${LIB_STATIC_DEFINE_NAME})
    endif()

endfunction()

# Use the common library setup function
add_task_library(
    TARGET_NAME
    task
    NAMESPACE
    framework::task
    SOURCES
    src/task.cpp
    src/task_pool.cpp
    src/task_graph.cpp
    src/task_scheduler.cpp
    src/task_monitor.cpp
    src/time.cpp
    src/task_worker.cpp
    src/task_utils.cpp
    src/task_visualizer.cpp
    src/timed_trigger.cpp
    src/memory_trigger.cpp
    PUBLIC_HEADERS
    include/task/time.hpp
    include/task/task_log.hpp
    include/task/flat_map.hpp
    include/task/task.hpp
    include/task/task_pool.hpp
    include/task/task_graph.hpp
    include/task/bounded_queue.hpp
    include/task/spinlock.hpp
    include/task/task_category.hpp
    include/task/task_utils.hpp
    include/task/task_visualizer.hpp
    include/task/timed_trigger.hpp
    include/task/memory_trigger.hpp
    include/task/task_scheduler.hpp
    include/task/task_monitor.hpp
    SYSTEM_LIBRARIES
    wise_enum::wise_enum
    phmap::phmap
    LIBRARIES
    framework::rt_log
    framework::options
    framework::warnings)

# Create separate task_instrument library so we can explicitly disable instrumentation for the NVTX
# and function cache files
add_task_library(
    TARGET_NAME
    task_instrument
    NAMESPACE
    framework::task_instrument
    SOURCES
    src/nvtx.cpp
    src/function_cache.cpp
    PUBLIC_HEADERS
    include/task/nvtx.hpp
    include/task/function_cache.hpp
    include/task/flat_map.hpp
    SYSTEM_LIBRARIES
    CUDA::nvtx3
    phmap::phmap
    gsl-lite::gsl-lite
    LIBRARIES
    framework::rt_log
    framework::options
    framework::warnings)

# Handle NVTX configuration for task_instrument library
if(NVTX_ENABLE)
    target_compile_definitions(task_instrument PUBLIC NVTX_ENABLE)
endif()
