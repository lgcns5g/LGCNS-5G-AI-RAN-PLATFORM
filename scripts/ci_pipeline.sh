#!/usr/bin/env bash
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

set -euo pipefail -o errtrace

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Runs the CI (Continuous Integration) pipeline for Aerial Framework.

OPTIONS:
    -h, --help          Show this help message

ENVIRONMENT VARIABLES:
    PRESETS             Comma-separated preset names (default: gcc-release)
                       Examples: clang-debug,gcc-release or single: clang-release
                       Up to 3 presets supported
    CHECK_FORMAT_ONLY   Exit after formatting check: 1 or empty (default: empty)
    BUILD_DOCS_ONLY     Exit after building docs: 1 or empty (default: empty)
    SKIP_CLEAN          Skip cleaning build/install directories: 1 or empty (default: empty)
    CMAKE_OPTIONS       Space-separated CMake options forwarded to cmake --preset commands
                       Examples: -DACAR_REPO=<url>, -DENABLE_CLANG_TIDY=OFF

Examples:
    PRESETS=clang-release ./scripts/ci_pipeline.sh
    CHECK_FORMAT_ONLY=1 ./scripts/ci_pipeline.sh
    CMAKE_OPTIONS="-DACAR_REPO=https://example.com/aerial_sdk.git -DENABLE_CLANG_TIDY=OFF" ./scripts/ci_pipeline.sh
    PRESETS=gcc-debug,clang-release CMAKE_OPTIONS="-DACAR_REPO=https://example.com/aerial_sdk.git" ./scripts/ci_pipeline.sh

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Configuration variables
PRESETS="${PRESETS:-gcc-release}"
CHECK_FORMAT_ONLY="${CHECK_FORMAT_ONLY:-}"
BUILD_DOCS_ONLY="${BUILD_DOCS_ONLY:-}"
SKIP_CLEAN="${SKIP_CLEAN:-}"
CMAKE_OPTIONS="${CMAKE_OPTIONS:-}"

# Parse CMAKE_OPTIONS into array
CMAKE_OPTIONS_ARRAY=()
if [[ -n "$CMAKE_OPTIONS" ]]; then
    read -ra CMAKE_OPTIONS_ARRAY <<< "$CMAKE_OPTIONS"
fi

# Function to reorder presets with gcc-release first
reorder_presets() {
    local input_array=("$@")
    local has_gcc_release=false
    local temp_array=()
    
    for preset in "${input_array[@]}"; do
        if [[ "$preset" == "gcc-release" ]]; then
            has_gcc_release=true
        else
            temp_array+=("$preset")
        fi
    done
    
    if [[ "$has_gcc_release" == "true" ]]; then
        PRESET_ARRAY=("gcc-release" "${temp_array[@]}")
        HAS_GCC_RELEASE=true
    else
        PRESET_ARRAY=("${input_array[@]}")
        HAS_GCC_RELEASE=false
    fi
}

# Parse presets into array (limit to 3 presets)
IFS=',' read -ra PRESET_ARRAY <<< "$PRESETS"
(( ${#PRESET_ARRAY[@]} <= 3 )) || { echo "✗ Maximum of 3 presets supported" >&2; exit 1; }

# Move gcc-release to front if present to speed up notebook tests that use this preset
reorder_presets "${PRESET_ARRAY[@]}"

# Colors and output functions
RED='\033[0;31m' GREEN='\033[0;32m' BLUE='\033[0;34m' NC='\033[0m'
print_step() { echo -e "$(date '+%H:%M:%S') ${BLUE}==> ${1}${NC}"; }
print_success() { echo -e "$(date '+%H:%M:%S') ${GREEN}✓ ${1}${NC}"; }
print_error() { echo -e "$(date '+%H:%M:%S') ${RED}✗ ${1}${NC}" >&2; }

# Timer functions with integrated step/success printing
declare -A TIMER_STARTS
step_start() {
    TIMER_STARTS["$1"]=$SECONDS
    printf "[TIMER] %s %s start\n" "$(date '+%H:%M:%S')" "$1"
    print_step "$2"
}
step_end() {
    local name="$1"
    local start=${TIMER_STARTS["$name"]}
    local elapsed=$((SECONDS - start))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    print_success "$2"
    printf "[TIMER] %s %s: %dm %ds\n" "$(date '+%H:%M:%S')" "$name" "$mins" "$secs"
}

# Configuration variables
FIRST_PRESET="${PRESET_ARRAY[0]}"
# Use gcc-release for docs if it exists (has MLIR_TRT=ON for notebooks), otherwise use first preset
if [[ "$HAS_GCC_RELEASE" == "true" ]]; then
    DOCS_PRESET="gcc-release"
else
    DOCS_PRESET="${FIRST_PRESET}"
fi
BUILD_DIR="out/build/${DOCS_PRESET}"
PROJECT_ROOT="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Function to configure a single preset
configure_preset() {
    local preset="$1"
    local is_first="$2"
    local args=()
    
    # Configure gcc-release to match notebook tests
    if [[ "$preset" == "gcc-release" ]]; then
        args+=("-DENABLE_MLIR_TRT=ON" "-DENABLE_IWYU=OFF" "-DENABLE_CLANG_TIDY=OFF")
    # If gcc-release exists, all other presets disable MLIR_TRT
    elif [[ "$HAS_GCC_RELEASE" == "true" ]]; then
        args+=("-DENABLE_MLIR_TRT=OFF")
        [[ "$is_first" != "true" ]] && args+=("-DENABLE_IWYU=OFF")
    # If no gcc-release, only first preset has MLIR_TRT enabled (default)
    elif [[ "$is_first" != "true" ]]; then
        args+=("-DENABLE_CLANG_TIDY=OFF" "-DENABLE_MLIR_TRT=OFF" "-DENABLE_IWYU=OFF")
    fi
    
    [[ "$preset" == *"debug"* ]] && args+=("-DENABLE_COVERAGE=ON")
    [[ ${#CMAKE_OPTIONS_ARRAY[@]} -gt 0 ]] && args+=("${CMAKE_OPTIONS_ARRAY[@]}")
    cmake --preset "$preset" "${args[@]}"
}

# Function to run commands in parallel or sequential
run_parallel_or_sequential() {
    local action="$1"
    shift
    local commands=("$@")

    if [[ ${#PRESET_ARRAY[@]} -eq 1 ]]; then
        print_step "$action (${FIRST_PRESET})..."
        eval "${commands[0]}"
    else
        print_step "$action multiple presets in parallel..."
        local pids=()
        for cmd in "${commands[@]}"; do
            eval "$cmd" &
            pids+=($!)
        done

        # Wait for all jobs and check exit codes
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done

        if [[ $failed -eq 1 ]]; then
            print_error "$action failed for one or more presets"
            return 1
        fi
    fi
}

# Main execution
step_start "Full pipeline" "Starting CI Tests for Aerial Framework"
print_step "Environment: PRESETS=${PRESETS}, CHECK_FORMAT_ONLY=${CHECK_FORMAT_ONLY:-<none>}, CMAKE_OPTIONS=${CMAKE_OPTIONS:-<none>}"
print_step "Docs/format checks preset: ${DOCS_PRESET}, Build directory: ${BUILD_DIR}"

cd "$PROJECT_ROOT"

# Step 0: Clean build and install directories
if [[ "${SKIP_CLEAN,,}" =~ ^(1|yes|true)$ ]]; then
    print_step "Skipping clean (SKIP_CLEAN=${SKIP_CLEAN})"
else
    print_step "Cleaning build and install directories..."
    for preset in "${PRESET_ARRAY[@]}"; do
        rm -rf "out/build/${preset}" && mkdir -p "out/build/${preset}"
        rm -rf "out/install/${preset}" && mkdir -p "out/install/${preset}"
    done
fi

# Step 1: Configure all presets
step_start "Configure" "Configuring presets..."
configure_commands=()
is_first_preset=true
for preset in "${PRESET_ARRAY[@]}"; do
    configure_commands+=("configure_preset '$preset' '$is_first_preset'")
    is_first_preset=false
done
run_parallel_or_sequential "Configuring" "${configure_commands[@]}"
step_end "Configure" "Configuration completed"

# Step 2: Check code formatting
step_start "Format check" "Checking code formatting..."
cmake --build "$BUILD_DIR" --target check-format-friendly || {
    echo "cmake check-format-friendly failed - run cmake format to see changes needed or fix-format to automatically correct formatting"
    exit 1
}
cmake --build "$BUILD_DIR" --target py_all_check_format
print_success "Code formatting check passed"

# Run ruff static analysis on all python packages (fast)
cmake --build "$BUILD_DIR" --target py_all_ruff_check
print_success "Python static analysis check passed"

# Run mypy on ran python packages (fast)
cmake --build "$BUILD_DIR" --target py_ran_mypy
step_end "Format check" "Python RAN mypy check passed"

# Exit early if only checking formatting
[[ "${CHECK_FORMAT_ONLY,,}" =~ ^(1|yes|true)$ ]] && {
    print_success "Formatting check completed successfully!"
    exit 0
}

# Step 2.1: Check copyright headers
step_start "Copyright check" "Checking copyright headers for SPDX compliance..."
cmake --build "$BUILD_DIR" --target check-copyright || {
    print_error "cmake check-copyright failed - run cmake fix-copyright to automatically fix copyright headers"
    exit 1
}
step_end "Copyright check" "Copyright header check passed"

# Step 2.2: Check include guards
step_start "Include guards check" "Checking include guard compliance..."
cmake --build "$BUILD_DIR" --target check-include-guards || {
    print_error "cmake check-include-guards failed - run cmake fix-include-guards to automatically fix include guards"
    exit 1
}
step_end "Include guards check" "Include guard check passed"

# Step 3: Check documentation standards and build docs
step_start "Docs build" "Checking documentation standards..."
cmake --build "$BUILD_DIR" --target check_all_docstrings
print_success "Documentation standards check passed"
print_step "Building documentation..."
cmake --build "$BUILD_DIR" --target docs
step_end "Docs build" "Documentation built successfully"

# Exit early if only building docs
[[ "${BUILD_DOCS_ONLY,,}" =~ ^(1|yes|true)$ ]] && {
    print_success "Documentation available at: ${BUILD_DIR}/docs/sphinx/index.html"
    exit 0
}

# Step 4: Test ran python wheel (remaining ran tests are run with ctest)
step_start "Python wheel test" "Building Python packages..."
for preset in "${PRESET_ARRAY[@]}"; do
    print_step "Building Python packages for $preset..."
    cmake --build "out/build/$preset" --target py_ran_wheel_test
done
step_end "Python wheel test" "Python packages built and tested"

# Step 5: Build all presets
step_start "C++ build" "Building C++ project..."
build_commands=()
for preset in "${PRESET_ARRAY[@]}"; do
    build_commands+=("cmake --build 'out/build/$preset'")
done
run_parallel_or_sequential "Building C++ project" "${build_commands[@]}"
step_end "C++ build" "C++ build completed"

# Step 6: Test CMake install
step_start "CMake install" "Testing CMake install..."
install_commands=()
for preset in "${PRESET_ARRAY[@]}"; do
    install_commands+=("cmake --install 'out/build/$preset' --prefix 'out/install/$preset'")
done
run_parallel_or_sequential "Testing CMake install" "${install_commands[@]}"
step_end "CMake install" "CMake install completed"

# Step 7: Run tests sequentially
step_start "C++ tests" "Running C++ tests..."
TEST_TIMEOUT=3600  # 60 minutes
TEST_PARALLEL_JOBS=16

# Save time by running notebook tests once for docs preset only.
# Most of the ctest calls in the notebooks will be exercised in the loop below anyway.
print_step "Running notebook tests for ${DOCS_PRESET}..."
ctest --preset "${DOCS_PRESET}" -L notebook --parallel "$TEST_PARALLEL_JOBS" --timeout "$TEST_TIMEOUT"
print_success "Notebook tests passed for ${DOCS_PRESET}"

# Rebuild docs so notebook outputs from the tests are included in the sphinx output
print_step "Rebuilding docs with notebook outputs..."
cmake --build "$BUILD_DIR" --target docs
print_success "Docs rebuilt with notebook outputs"

for preset in "${PRESET_ARRAY[@]}"; do
    print_step "Testing preset: $preset"
    
    # Run parallel-only tests in parallel
    print_step "Running parallel tests for $preset..."
    ctest --preset "$preset" -L parallel --parallel "$TEST_PARALLEL_JOBS" --timeout "$TEST_TIMEOUT"
    print_success "Parallel tests passed for $preset"
    
    # Run all other tests serially
    print_step "Running remaining tests for $preset..."
    ctest --preset "$preset" -LE "notebook|parallel" --timeout "$TEST_TIMEOUT"
    print_success "Remaining tests passed for $preset"
    
    print_success "All tests passed for $preset"
done
step_end "C++ tests" "All C++ tests passed"

# Step 8: Generate coverage reports
coverage_generated=false
for preset in "${PRESET_ARRAY[@]}"; do
    if [[ "$preset" == *"debug"* ]]; then
        step_start "Coverage generation" "Generating C++ coverage report for $preset..."
        cmake --build "out/build/$preset" --target coverage
        step_end "Coverage generation" "Coverage report generated: out/build/$preset/coverage/index.html"
        coverage_generated=true
    fi
done

# Summary
print_step "Test Summary"
for item in "Configuration: ${FIRST_PRESET}" "C++ Build: Completed successfully" \
           "CMake Install: Installation completed successfully" "Python Build: Packages built and tested" \
           "Formatting: Code formatting checks passed" "Documentation: Documentation standards check passed" \
           "Tests: All tests passed" "Documentation: Sphinx documentation built successfully"; do
    print_success "✓ $item"
done

if [[ "$coverage_generated" == "true" ]]; then
    print_success "✓ Coverage: Coverage report(s) generated"
else
    print_success "✓ Coverage: Skipped (no debug builds)"
fi

echo ""
step_end "Full pipeline" "All CI tests completed successfully!"
