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

# DPDK capabilities required for DevX/TIS operations
set(DPDK_CAPABILITIES
    "cap_net_raw,cap_net_admin,cap_sys_rawio,cap_ipc_lock,cap_dac_override,cap_sys_admin,cap_sys_nice=eip"
)

# Internal function to set capabilities on a target (regular targets in same directory only) Uses
# POST_BUILD which is simpler but has CMake limitation: only works for targets created in the same
# CMakeLists.txt where this function is called
function(_set_capabilities_post_build TARGET_NAME CAPABILITIES DESCRIPTION)
    find_program(SETCAP_EXECUTABLE setcap)

    if(SETCAP_EXECUTABLE)
        add_custom_command(
            TARGET ${TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo
                    "Setting ${DESCRIPTION} capabilities for ${TARGET_NAME}"
            COMMAND ${SETCAP_EXECUTABLE} ${CAPABILITIES} $<TARGET_FILE:${TARGET_NAME}>
            COMMAND ${CMAKE_COMMAND} -E echo "Capabilities set successfully for ${TARGET_NAME}"
            COMMENT "Setting ${DESCRIPTION} capabilities for non-privileged execution"
            VERBATIM)
    else()
        message(WARNING "setcap not found - applications may require sudo or privileged containers")
    endif()
endfunction()

# Generic function to set capabilities on any target (regular or imported, any directory) Uses stamp
# file approach to work around CMake limitation: POST_BUILD can only attach to targets in the same
# directory. This approach works for: - Imported targets (e.g., from find_package) - Targets created
# in different CMakeLists.txt directories - Regular targets in the same directory The stamp file
# tracks whether setcap needs to run: the DEPENDS clause ensures setcap only executes when the
# target binary actually changes, not on every build.
function(set_target_capabilities TARGET_NAME CAPABILITIES DESCRIPTION)
    # Check if target exists and get its type
    if(NOT TARGET ${TARGET_NAME})
        message(WARNING "Target ${TARGET_NAME} does not exist, skipping capability setup")
        return()
    endif()

    get_target_property(target_type ${TARGET_NAME} TYPE)

    # Only set capabilities on executable targets
    if(NOT target_type STREQUAL "EXECUTABLE")
        message(
            WARNING
                "Target ${TARGET_NAME} is not an executable (type: ${target_type}), skipping setcap"
        )
        return()
    endif()

    find_program(SETCAP_EXECUTABLE setcap)

    if(SETCAP_EXECUTABLE)
        set(stamp_file "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_cap.stamp")
        set(cap_target "${TARGET_NAME}_set_cap")

        if(NOT TARGET ${cap_target})
            # Create custom command that depends on the target binary itself This ensures setcap
            # runs only when the binary is rebuilt/changed
            add_custom_command(
                OUTPUT ${stamp_file}
                COMMAND ${CMAKE_COMMAND} -E echo
                        "Setting ${DESCRIPTION} capabilities for ${TARGET_NAME}"
                COMMAND ${SETCAP_EXECUTABLE} ${CAPABILITIES} $<TARGET_FILE:${TARGET_NAME}>
                COMMAND ${CMAKE_COMMAND} -E touch ${stamp_file}
                DEPENDS $<TARGET_FILE:${TARGET_NAME}>
                COMMENT "Setting ${DESCRIPTION} capabilities for non-privileged execution"
                VERBATIM)

            # Custom target runs as part of ALL (default build) and depends on stamp file
            add_custom_target(${cap_target} ALL DEPENDS ${stamp_file})
        endif()
    else()
        message(
            WARNING "setcap not found - ${TARGET_NAME} may require sudo or privileged containers")
    endif()
endfunction()

function(set_dpdk_capabilities TARGET_NAME)
    # Set required capabilities for DPDK DevX/TIS operations
    _set_capabilities_post_build(${TARGET_NAME} "${DPDK_CAPABILITIES}" "DPDK")
endfunction()

function(set_cap_sys_nice TARGET_NAME)
    _set_capabilities_post_build(${TARGET_NAME} "cap_sys_nice=eip" "CAP_SYS_NICE")
endfunction()

function(create_cap_sys_nice_target TARGET_NAME)
    set_target_capabilities(${TARGET_NAME} "cap_sys_nice=eip" "CAP_SYS_NICE")
endfunction()

function(create_dpdk_capabilities_target TARGET_NAME)
    # Set required capabilities for DPDK DevX/TIS operations on imported/external targets
    set_target_capabilities(${TARGET_NAME} "${DPDK_CAPABILITIES}" "DPDK")
endfunction()
