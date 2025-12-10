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

# Create global benchmark target
if(NOT TARGET all_benchmarks)
    add_custom_target(all_benchmarks COMMENT "Running all benchmarks")
endif()

# Helper function to add benchmark executables to the global target Note: Assumes executables are
# built with Google Benchmark and support --benchmark_format, --benchmark_out_format, and
# --benchmark_out flags
function(add_benchmark target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "add_benchmark: Target '${target_name}' does not exist")
    endif()
    # Create combined console + JSON output target
    add_custom_target(
        run_${target_name}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/benchmark_results
        COMMAND ${CMAKE_COMMAND} -E echo "=== Running ${target_name} ==="
        COMMAND ${target_name} --benchmark_format=console --benchmark_out_format=json
                --benchmark_out=${CMAKE_BINARY_DIR}/benchmark_results/${target_name}_results.json
        COMMAND ${CMAKE_COMMAND} -E echo
                "Results saved to benchmark_results/${target_name}_results.json"
        DEPENDS ${target_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running ${target_name} with console and JSON output"
        USES_TERMINAL)

    # Add to the global target
    add_dependencies(all_benchmarks run_${target_name})

    # Store benchmark names for potential future use
    set_property(GLOBAL APPEND PROPERTY BENCHMARK_TARGETS ${target_name})
endfunction()
