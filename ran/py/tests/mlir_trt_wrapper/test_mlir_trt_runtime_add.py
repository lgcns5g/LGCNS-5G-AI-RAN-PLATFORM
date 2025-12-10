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

"""Test MLIR-TensorRT runtime compilation and execution."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Only import JAX and internal dependencies when MLIR_TRT is enabled
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    import jax  # noqa: E402
    import jax.export  # noqa: E402
    from jax import numpy as jnp  # noqa: E402

    from ran import mlir_trt_wrapper as mtw  # noqa: E402
    from ran.trt_plugins.manager.trt_plugin_manager import (  # noqa: E402
        copy_trt_engine_for_cpp_tests,
        should_skip_engine_generation,
    )

    os.environ["JAX_PLATFORMS"] = "cuda"

try:
    from ml_dtypes import bfloat16

    HAS_ML_DTYPES = True
except ImportError:
    # Define these so pytest can resolve the types during
    # test collection, even if they are skipped by pytestmark
    HAS_ML_DTYPES = False
    bfloat16 = None

# Compiler flag test configurations: (flags, test_id)
TENSORRT_FLAG_TEST_CASES = [
    (
        ["--tensorrt-builder-opt-level=0"],
        "opt_level_0",
    ),
    (
        ["--tensorrt-builder-opt-level=3"],
        "opt_level_3",
    ),
    (
        ["--tensorrt-builder-opt-level=5"],
        "opt_level_5",
    ),
]

# Check if engines already exist (evaluated at module import time)
# Only check for engines that are copied to persistent location
# (test_compile_with_different_flags tests don't save engines)
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    _skip_engine_gen = should_skip_engine_generation(
        [
            "add_func_float32.trtengine",
            "add_func_float16.trtengine",
            "add_func_bfloat16.trtengine",
            "add_func_int32.trtengine",
        ]
    )
else:
    _skip_engine_gen = False

# All tests in this module require MLIR_TRT to be enabled
pytestmark = [
    pytest.mark.skipif(
        os.getenv("ENABLE_MLIR_TRT", "OFF") != "ON",
        reason="Requires MLIR-TensorRT compiler (ENABLE_MLIR_TRT=OFF)",
    ),
    pytest.mark.skipif(
        _skip_engine_gen,
        reason="TRT engines already exist and SKIP_TRT_ENGINE_GENERATION=1",
    ),
]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for compiled artifacts.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns
    -------
        Path to temporary output directory
    """
    output_dir = tmp_path / "mlir_trt_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_compile_and_execute_add_function(temp_output_dir: Path) -> None:
    """Test compilation and execution of simple addition function with various dtypes.

    This test verifies the complete MLIR-TensorRT workflow:
    1. Export JAX function to StableHLO MLIR
    2. Compile MLIR to executable (parallelized across dtypes)
    3. Execute on GPU with runtime
    4. Verify correctness of output

    Tested data types:
    - float32: Standard 32-bit floating point
    - float16: 16-bit floating point (enables FP16 TensorRT kernels)
    - bfloat16: Brain float 16-bit (Google's ML-optimized format)
    - int32: 32-bit integer

    Args:
        temp_output_dir: Temporary directory for compilation artifacts
    """
    logger.info("Starting dtype compilation test")

    # Define simple addition function
    def add_func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x + y

    # Test configurations: (dtype, x_data, y_data, expected_data, test_id)
    dtype_configs = [
        (np.float32, [1.0, 2.0], [3.0, 4.0], [4.0, 6.0], "float32"),
        (np.float16, [1.0, 2.0], [3.0, 4.0], [4.0, 6.0], "float16"),
        (np.int32, [1, 2], [3, 4], [4, 6], "int32"),
    ]

    # Add bfloat16 if available
    if HAS_ML_DTYPES:
        dtype_configs.append((bfloat16, [1.0, 2.0], [3.0, 4.0], [4.0, 6.0], "bfloat16"))

    # Store MLIR modules for parallel compilation
    mlir_modules = {}
    test_data = {}

    # Sequential: Prepare MLIR modules for all dtypes
    for dtype, x_data, y_data, expected_data, test_id in dtype_configs:
        # Prepare test inputs
        x_input = np.array(x_data, dtype=dtype)
        y_input = np.array(y_data, dtype=dtype)
        expected_output = np.array(expected_data, dtype=dtype)

        # Validate arrays
        mtw.validate_array(x_input, "x_input")
        mtw.validate_array(y_input, "y_input")

        # Export JAX function to StableHLO MLIR
        jit_func = jax.jit(add_func)
        exported = jax.export.export(jit_func)(x_input, y_input)
        stablehlo_mlir = exported.mlir_module()

        mlir_modules[test_id] = stablehlo_mlir
        test_data[test_id] = (dtype, x_input, y_input, expected_output)

    # Parallel: Compile all dtypes
    logger.info(f"Compiling {len(dtype_configs)} dtypes in parallel...")

    def compile_dtype(test_id: str) -> tuple[str, object, float]:
        """Compile MLIR for a single dtype."""
        start_time = time.time()
        func_name = f"add_func_{test_id}"
        output_dir = temp_output_dir / test_id
        output_dir.mkdir(parents=True, exist_ok=True)

        exe = mtw.compile(
            stablehlo_mlir=mlir_modules[test_id],
            name=func_name,
            export_path=output_dir,
        )

        compilation_time = time.time() - start_time
        logger.info(f"  Compiled {test_id} in {compilation_time:.2f}s")
        return test_id, exe, compilation_time

    parallel_start = time.time()
    executables = {}

    with ThreadPoolExecutor(max_workers=len(dtype_configs)) as executor:
        future_to_id = {
            executor.submit(compile_dtype, test_id): test_id
            for _, _, _, _, test_id in dtype_configs
        }

        for future in as_completed(future_to_id):
            test_id = future_to_id[future]
            try:
                result_id, exe, comp_time = future.result()
                executables[result_id] = exe
            except Exception as exc:
                logger.error(f"Compilation failed for {test_id}: {exc}")
                raise

    parallel_total = time.time() - parallel_start
    logger.info(f"Parallel compilation completed in {parallel_total:.2f}s")

    # Sequential: Execute and verify all dtypes
    for dtype, x_data, y_data, expected_data, test_id in dtype_configs:
        logger.info(f"Verifying {test_id}...")
        test_dtype, x_input, y_input, expected_output = test_data[test_id]
        exe = executables[test_id]
        func_name = f"add_func_{test_id}"
        output_dir = temp_output_dir / test_id

        # Prepare output buffer
        output = np.zeros_like(expected_output)
        mtw.validate_array(output, "output")

        # Execute compiled function
        inputs = (x_input, y_input)
        outputs = (output,)
        mtw.execute(exe=exe, inputs=inputs, outputs=outputs, sync_stream=True, validate=False)

        # Verify output correctness with dtype-appropriate tolerances
        if np.issubdtype(test_dtype, np.integer):
            np.testing.assert_array_equal(outputs[0], expected_output)
        elif HAS_ML_DTYPES and test_dtype == bfloat16:
            np.testing.assert_allclose(
                np.asarray(outputs[0]).astype(np.float32),
                np.asarray(expected_output).astype(np.float32),
                rtol=1e-2,
                atol=1e-2,
            )
        elif test_dtype == np.float16:
            np.testing.assert_allclose(outputs[0], expected_output, rtol=1e-2, atol=1e-2)
        else:
            np.testing.assert_allclose(outputs[0], expected_output, rtol=1e-6, atol=1e-6)

        # Verify artifacts were created
        assert (output_dir / f"{func_name}.bin").exists()
        assert (output_dir / f"{func_name}.original.stablehlo.mlir").exists()

        # Copy engines to configured directory for C++ tests
        if dtype in (np.float32, np.float16, np.int32) or (HAS_ML_DTYPES and dtype == bfloat16):
            copy_trt_engine_for_cpp_tests(output_dir, f"add_func_{test_id}.trtengine")

    logger.info("All dtype compilations verified successfully!")


def test_compile_with_different_flags(temp_output_dir: Path) -> None:
    """Test compilation with various MLIR-TensorRT compiler flags.

    This test verifies that different compiler configurations work correctly:
    - Different optimization levels (0, 3, 5)

    Compilation is parallelized across all flag configurations using ThreadPoolExecutor.

    Args:
        temp_output_dir: Temporary directory for compilation artifacts
    """
    logger.info("Starting compiler flags test")

    # Define simple addition function
    def add_func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x + y

    # Use float32 for all compiler flag tests
    dtype = np.float32
    x_data = [1.0, 2.0, 3.0]
    y_data = [4.0, 5.0, 6.0]
    expected_data = [5.0, 7.0, 9.0]

    # Prepare test inputs
    x_input = np.ascontiguousarray(np.array(x_data, dtype=dtype))
    y_input = np.ascontiguousarray(np.array(y_data, dtype=dtype))
    expected_output = np.ascontiguousarray(np.array(expected_data, dtype=dtype))

    # Export JAX function to StableHLO MLIR (only once, reused for all compilations)
    jit_func = jax.jit(add_func)
    exported = jax.export.export(jit_func)(x_input, y_input)
    stablehlo_mlir = exported.mlir_module()

    # Parallel: Compile with all flag configurations
    logger.info(f"Compiling {len(TENSORRT_FLAG_TEST_CASES)} configurations in parallel...")

    def compile_with_flags(compilation_flags: list[str], test_id: str) -> tuple[str, object, float]:
        """Compile MLIR with specified flags."""
        start_time = time.time()
        func_name = f"add_func_{test_id}"
        output_dir = temp_output_dir / test_id
        output_dir.mkdir(parents=True, exist_ok=True)

        exe = mtw.compile(
            stablehlo_mlir=stablehlo_mlir,
            name=func_name,
            export_path=output_dir,
            mlir_tensorrt_compilation_flags=compilation_flags,
        )

        compilation_time = time.time() - start_time
        logger.info(f"  Compiled {test_id} in {compilation_time:.2f}s")
        return test_id, exe, compilation_time

    parallel_start = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=len(TENSORRT_FLAG_TEST_CASES)) as executor:
        future_to_config = {
            executor.submit(compile_with_flags, flags, test_id): (flags, test_id)
            for flags, test_id in TENSORRT_FLAG_TEST_CASES
        }

        for future in as_completed(future_to_config):
            flags, test_id = future_to_config[future]
            try:
                result_id, exe, comp_time = future.result()
                results[result_id] = (exe, comp_time, flags)
            except Exception as exc:
                logger.error(f"Compilation failed for {test_id}: {exc}")
                raise

    parallel_total = time.time() - parallel_start
    logger.info(f"Parallel compilation completed in {parallel_total:.2f}s")

    # Sequential: Verify all compilations
    for test_id, (exe, comp_time, flags) in results.items():
        logger.info(f"Verifying {test_id}...")
        func_name = f"add_func_{test_id}"
        output_dir = temp_output_dir / test_id

        # Prepare output buffer
        output = np.zeros_like(expected_output)

        # Execute compiled function
        inputs = (x_input, y_input)
        outputs = (output,)
        mtw.execute(exe=exe, inputs=inputs, outputs=outputs, sync_stream=True, validate=False)

        # Verify output correctness
        np.testing.assert_allclose(
            outputs[0],
            expected_output,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Execution with flags {flags} failed",
        )

        # Verify compilation artifacts were created
        assert (output_dir / f"{func_name}.bin").exists(), (
            f"Binary artifact not created for {test_id}"
        )
        assert (output_dir / f"{func_name}.original.stablehlo.mlir").exists(), (
            f"MLIR artifact not created for {test_id}"
        )

    logger.info("All flag configurations verified successfully!")
