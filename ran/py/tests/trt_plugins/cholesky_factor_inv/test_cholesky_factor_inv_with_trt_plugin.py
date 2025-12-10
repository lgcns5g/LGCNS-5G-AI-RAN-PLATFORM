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

"""Test Cholesky factorization and inversion JAX implementation with custom TensorRT plugin."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Limit JAX GPU memory pre-allocation to prevent OOM issues
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

# Only import JAX and internal dependencies when MLIR_TRT is enabled
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax import export  # noqa: E402

    from ran import mlir_trt_wrapper as mtw  # noqa: E402
    from ran.trt_plugins.cholesky_factor_inv.cholesky_factor_inv_trt_plugin import (
        CholeskyFactorInvTrtPlugin,
    )  # noqa: E402
    from ran.trt_plugins.manager.trt_plugin_manager import (  # noqa: E402
        copy_trt_engine_for_cpp_tests,
        should_skip_engine_generation,
    )

# Check if engines already exist (evaluated at module import time)
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    _skip_engine_gen = should_skip_engine_generation(["cholesky_test.trtengine"])
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

logger = logging.getLogger(__name__)


def test_cusolver_cholesky_inv_with_trt_plugin() -> None:
    """Test Cholesky factorization and inversion JAX implementation with custom TensorRT plugin.

    The test compiles the JAX Cholesky inversion function to a TensorRT engine
    executable using the MLIR-TensorRT compiler. The JAX function includes a
    custom TensorRT plugin that is used to perform device-side Cholesky factor inversion
    using cuSOLVERDx. A custom C++ plugin is required because TensorRT does not have
    native support for this operation. The test checks the functional correctness of the
    JAX function with respect to JAX's reference implementation using three
    different runtimes: The standard cpython interpreter, the MLIR-TensorRT runtime, and
    a standalone C++ app.

    The test uses separate real and imaginary parts to avoid complex types in TensorRT.

    Returns
    -------
        None
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # --------------------------------
        # Test parameters
        # --------------------------------
        # Cholesky inversion inputs:
        matrix_size = 2  # Matrix size (2x2, 4x4, or 8x8 supported)
        batch_size = 1  # Number of matrices to process

        # Expected outputs:
        # - l_inv: Inverse of Cholesky factor L^{-1}, shape (batch_size, matrix_size, matrix_size)

        # --------------------------------
        # Build directory
        # --------------------------------

        test_build_dir = Path(temp_dir) / "cholesky_factor_inv_plugin_test"
        test_build_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Test build directory: {test_build_dir}")

        # --------------------------------
        # Generate test input (positive semi-definite matrix)
        # --------------------------------

        # Create a positive semi-definite matrix for testing
        # For simplicity, use a known PSD matrix
        if matrix_size == 2:
            # 2x2 PSD matrix
            cov_matrix = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        elif matrix_size == 4:
            # 4x4 PSD matrix (identity + small perturbation)
            cov_matrix = np.eye(4, dtype=np.float32) * 2.0
            cov_matrix[0, 1] = cov_matrix[1, 0] = 0.5
        elif matrix_size == 8:
            # 8x8 PSD matrix (identity + small perturbation)
            cov_matrix = np.eye(8, dtype=np.float32) * 2.0
            cov_matrix[0, 1] = cov_matrix[1, 0] = 0.5
        else:
            raise ValueError(f"Unsupported matrix size: {matrix_size}")

        # Add batch dimension
        cov_batch = np.expand_dims(cov_matrix, axis=0)  # Shape: (1, matrix_size, matrix_size)

        # --------------------------------
        # JAX reference Cholesky inversion
        # --------------------------------

        # Compute reference result using JAX/NumPy
        l_chol_ref = jax.numpy.linalg.cholesky(cov_batch[0])
        l_inv_ref = np.linalg.inv(l_chol_ref)
        l_inv_ref_batch = np.expand_dims(l_inv_ref, axis=0)

        # --------------------------------
        # JAX Cholesky factorization and inversion function
        # --------------------------------

        # Create the cuSOLVER plugin instance
        cusolver_plugin = CholeskyFactorInvTrtPlugin(
            matrix_size=matrix_size, is_complex=False, name="cusolver_cholesky_inv_plugin"
        )
        trt_plugin_config = cusolver_plugin.trt_plugin_config

        # Test the plugin directly
        l_inv_result = cusolver_plugin(jnp.asarray(cov_batch))

        # --------------------------------
        # Check functional correctness using cpython runtime
        # --------------------------------

        np.testing.assert_allclose(l_inv_result, l_inv_ref_batch, rtol=1e-3, atol=1e-3)
        logger.info("JAX Cholesky inversion function matches JAX reference")

        # --------------------------------
        # Compile JAX Cholesky function to TensorRT engine executable using MLIR-TensorRT
        # executor backend
        # --------------------------------

        # Inputs to the JAX Cholesky function
        inputs = (cov_batch,)

        # Prepare the JAX Cholesky function for compilation
        jit_cusolver_plugin = jax.jit(cusolver_plugin)

        mlir_tensorrt_compilation_flags = [
            "tensorrt-builder-opt-level=0",
            "tensorrt-workspace-memory-pool-limit=50MiB",
        ]

        # Export JAX function to StableHLO MLIR
        exported = jax.export.export(
            jit_cusolver_plugin,
            disabled_checks=[
                export.DisabledSafetyCheck.custom_call("tensorrt_cholesky_inv_plugin")
            ],
        )(*inputs)
        stablehlo_mlir = exported.mlir_module()

        # Wrap the single plugin config in a dictionary mapping target name to config
        trt_plugin_configs = {
            "tensorrt_cholesky_inv_plugin": trt_plugin_config,
        }

        mlir_trt_exe = mtw.compile(
            stablehlo_mlir=stablehlo_mlir,
            name="cusolver_cholesky_inv_jax",
            export_path=test_build_dir,
            mlir_entry_point="main",
            mlir_tensorrt_compilation_flags=mlir_tensorrt_compilation_flags,
            trt_plugin_configs=trt_plugin_configs,  # type: ignore[arg-type]
        )

        # --------------------------------
        # Check functional correctness using MLIR-TensorRT runtime
        # --------------------------------

        l_inv_mlir_trt: np.ndarray = np.zeros(
            (batch_size, matrix_size, matrix_size), dtype=np.float32
        )

        mtw.execute(
            exe=mlir_trt_exe,
            inputs=(cov_batch,),
            outputs=(l_inv_mlir_trt,),
            sync_stream=True,
        )

        # Verify results are approximately equal
        np.testing.assert_allclose(l_inv_mlir_trt, l_inv_result, rtol=1e-3, atol=1e-3)

        # --------------------------------
        # Copy TensorRT engine for C++ test
        # --------------------------------

        engine_dest = copy_trt_engine_for_cpp_tests(
            test_build_dir, "cholesky_test.trtengine", required=False
        )
        logger.debug(f"Copied TensorRT engine to {engine_dest}")

        # --------------------------------
        # Python test completed - C++ application testing moved to separate CMake target
        # --------------------------------

        logger.info(
            "Python Cholesky factorization and inversion TensorRT plugin test completed successfully!"
        )
        logger.info(
            "Note: Testing of generated C++ artifacts is handled by "
            "the 'ran_py_cpp_integration_tests' CMake target"
        )
