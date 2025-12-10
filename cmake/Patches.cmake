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

# Apply a patch file to a directory, skipping already-applied patches without prompting
#
# cmake-format: off
# Args:
#   patch_file - Path to the patch file to apply
#   working_dir - Directory where the patch should be applied
#   patch_level - Strip N leading path components (default: 1)
# cmake-format: on
function(apply_patch_once patch_file working_dir)
    set(options "")
    set(oneValueArgs PATCH_LEVEL)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default patch level
    if(NOT DEFINED ARG_PATCH_LEVEL)
        set(ARG_PATCH_LEVEL 1)
    endif()

    message(STATUS "Applying patch: ${patch_file}")

    # cmake-format: off
    # Apply patch with -N flag to skip already-applied patches without prompting
    # -p${ARG_PATCH_LEVEL}: Strip N leading path components
    # -N/--forward: Skip already-applied patches without prompting
    # --reject-file=-: Send reject output to stdout instead of creating .rej files
    # cmake-format: on
    execute_process(
        COMMAND patch -p${ARG_PATCH_LEVEL} -N --forward --reject-file=-
        INPUT_FILE ${patch_file}
        WORKING_DIRECTORY ${working_dir}
        RESULT_VARIABLE PATCH_RESULT
        OUTPUT_VARIABLE PATCH_OUTPUT
        ERROR_VARIABLE PATCH_ERROR)

    message(STATUS "Patch result: ${PATCH_RESULT}")
    if(PATCH_OUTPUT)
        message(STATUS "Patch output:\n${PATCH_OUTPUT}")
    endif()
    if(PATCH_ERROR)
        message(STATUS "Patch errors:\n${PATCH_ERROR}")
    endif()
endfunction()
