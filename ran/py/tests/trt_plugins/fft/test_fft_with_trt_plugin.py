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

"""Test FFT JAX implementation with custom TensorRT plugin."""

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
    from ran.trt_plugins import fft  # noqa: E402
    from ran.trt_plugins.manager.trt_plugin_manager import (  # noqa: E402
        copy_trt_engine_for_cpp_tests,
        should_skip_engine_generation,
    )

# Check if engines already exist (evaluated at module import time)
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    _skip_engine_gen = should_skip_engine_generation(["fft_test.trtengine"])
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


def test_fft_with_trt_plugin() -> None:
    """Test FFT JAX implementation with custom TensorRT plugin.

    The test compiles the JAX FFT function to a TensorRT engine executable
    using the MLIR-TensorRT compiler. The JAX FFT function includes a custom
    TensorRT engine plugin that is used to perform device-side FFT computation using
    cuFFTDx. A custom C++ plugin is required for FFT computation because TensorRT does not
    have native FFT support. The test checks the functional correctness of the JAX FFT
    function with respect to JAX's reference FFT implementation using three different
    runtimes: The standard cpython interpreter, the MLIR-TensorRT runtime, and a
    standalone C++ app.

    The test now uses separate real and imaginary parts to avoid complex types in TensorRT.

    Returns
    -------
        None
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # --------------------------------
        # Test parameters
        # --------------------------------
        # FFT inputs:
        fft_size = 2048

        # Expected FFT outputs:
        # - fft_result: tuple of (real, imag) arrays of shape (fft_size,)

        # --------------------------------
        # Build directory
        # --------------------------------

        test_build_dir = Path(temp_dir) / "fft_plugin_test"
        test_build_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Test build directory: {test_build_dir}")

        # --------------------------------
        # JAX reference FFT
        # --------------------------------

        # Create test input signal with random data (separate real and imaginary parts)
        # Use small scaling factor to keep values reasonable
        # Test with batched input to match channel estimation use case
        np.random.seed(42)  # For reproducibility
        alpha = 0.001
        batch_size = 4  # Test with multiple batches like in channel estimation
        x_real = alpha * np.random.randn(batch_size, fft_size).astype(np.float32)
        x_imag = alpha * np.random.randn(batch_size, fft_size).astype(np.float32)

        # JAX reference FFT (convert to complex for reference computation)
        # For batched input, apply FFT along last axis
        x_complex = x_real + 1j * x_imag
        fft_ref_complex = jax.numpy.fft.fft(x_complex, axis=-1)
        fft_ref_real = fft_ref_complex.real
        fft_ref_imag = fft_ref_complex.imag

        # --------------------------------
        # JAX FFT function
        # --------------------------------

        # The JAX FFT function uses a custom TensorRT plugin. The plugin needs to be loaded
        # and created with the creator parameter fft_size.
        fft_func, trt_plugin_config = fft.get_fft_jax(fft_size=fft_size)

        # Call the FFT function with separate real and imaginary parts
        fft_result_real, fft_result_imag = fft_func(jnp.asarray(x_real), jnp.asarray(x_imag))

        # --------------------------------
        # Check functional correctness using cpython runtime
        # --------------------------------

        np.testing.assert_allclose(fft_result_real, fft_ref_real, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fft_result_imag, fft_ref_imag, rtol=1e-5, atol=1e-5)
        logger.info("JAX FFT function matches JAX reference FFT")

        # --------------------------------
        # Test IFFT → FFT round trip
        # --------------------------------

        # Get IFFT function
        ifft_func, _ = fft.get_fft_jax(fft_size=fft_size, direction=1)

        # IFFT of the original signal
        y_real, y_imag = ifft_func(jnp.asarray(x_real), jnp.asarray(x_imag))

        # FFT of the IFFT result (should recover original signal)
        z_real, z_imag = fft_func(y_real, y_imag)

        # Check round trip recovers original signal
        logger.info(
            f"Round trip error (real): max={np.max(np.abs(z_real - x_real)):.6e}, mean={np.mean(np.abs(z_real - x_real)):.6e}"
        )
        logger.info(
            f"Round trip error (imag): max={np.max(np.abs(z_imag - x_imag)):.6e}, mean={np.mean(np.abs(z_imag - x_imag)):.6e}"
        )

        np.testing.assert_allclose(
            z_real, x_real, rtol=1e-3, atol=1e-6, err_msg="IFFT→FFT round trip failed for real part"
        )
        np.testing.assert_allclose(
            z_imag,
            x_imag,
            rtol=1e-3,
            atol=1e-6,
            err_msg="IFFT→FFT round trip failed for imaginary part",
        )
        logger.info("IFFT → FFT round trip test passed")

        # --------------------------------
        # Compile JAX FFT function to TensorRT engine executable using MLIR-TensorRT
        # executor backend
        # --------------------------------

        # Inputs to the JAX FFT function
        inputs = (x_real, x_imag)

        # Prepare the JAX FFT function for compilation (nothing is traced at this stage)
        # Static args are required for fft_size (this is fixed during tracing)
        jit_hybrid_trt_plugin = jax.jit(fft_func, static_argnums=())

        mlir_tensorrt_compilation_flags = [
            "tensorrt-builder-opt-level=0",
            "tensorrt-workspace-memory-pool-limit=50MiB",
        ]

        # Export JAX function to StableHLO MLIR
        # Safety check disabled for custom TensorRT plugin call: JAX export validates
        # standard ops but cannot verify custom plugin semantics. The plugin is validated
        # through functional correctness tests (lines 109-110) before and after export.
        exported = jax.export.export(
            jit_hybrid_trt_plugin,
            disabled_checks=[export.DisabledSafetyCheck.custom_call("tensorrt_fft_plugin")],
        )(*inputs)
        stablehlo_mlir = exported.mlir_module()

        # Wrap the single plugin config in a dictionary mapping target name to config
        trt_plugin_configs = {
            "tensorrt_fft_plugin": trt_plugin_config,
        }

        mlir_trt_exe = mtw.compile(
            stablehlo_mlir=stablehlo_mlir,
            name="fft_jax",
            export_path=test_build_dir,
            mlir_entry_point="main",
            mlir_tensorrt_compilation_flags=mlir_tensorrt_compilation_flags,
            trt_plugin_configs=trt_plugin_configs,  # type: ignore[arg-type]
        )

        # --------------------------------
        # Check functional correctness using MLIR-TensorRT runtime
        # --------------------------------

        fft_real_mlir_trt: np.ndarray = np.zeros((batch_size, fft_size), dtype=np.float32)
        fft_imag_mlir_trt: np.ndarray = np.zeros((batch_size, fft_size), dtype=np.float32)

        mtw.execute(
            exe=mlir_trt_exe,
            inputs=(x_real, x_imag),
            outputs=(fft_real_mlir_trt, fft_imag_mlir_trt),
            sync_stream=True,
        )

        # Verify results are approximately equal
        np.testing.assert_allclose(fft_real_mlir_trt, fft_result_real, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fft_imag_mlir_trt, fft_result_imag, rtol=1e-5, atol=1e-5)

        # --------------------------------
        # Copy TensorRT engine for C++ test
        # --------------------------------

        engine_dest = copy_trt_engine_for_cpp_tests(
            test_build_dir, "fft_test.trtengine", required=False
        )
        logger.debug(f"Copied TensorRT engine to {engine_dest}")

        # --------------------------------
        # Python test completed - C++ application testing moved to separate CMake target
        # --------------------------------

        logger.info("Python FFT TensorRT plugin test completed successfully!")
        logger.info(
            "Note: Testing of generated C++ artifacts is handled by "
            "the 'ran_py_cpp_integration_tests' CMake target"
        )


def test_fft_singleton_instances() -> None:
    """Test FFT singleton instances with forward and inverse transforms.

    Verifies FFT/IFFT round-trip accuracy for singleton instances
    against JAX reference implementation.
    """
    import jax.numpy as jnp
    from ran.trt_plugins.fft.fft_trt_plugin import fft_128, ifft_128, fft_2048, ifft_2048

    logger.info("Testing FFT singleton instances")

    fft_sizes = [128, 2048]
    atol = 1e-5
    alpha = 1e-3  # scaling factor

    for fft_size in fft_sizes:
        logger.info("=" * 48)
        logger.info("Testing FFT size: %d", fft_size)
        logger.info("=" * 48)

        # Select appropriate singleton instances
        if fft_size == 128:
            fft = fft_128
            ifft = ifft_128
        elif fft_size == 2048:
            fft = fft_2048
            ifft = ifft_2048
        else:
            raise ValueError(f"Unsupported FFT size: {fft_size}")

        # Generate random test data
        key = jax.random.PRNGKey(42)
        x_real = alpha * jax.random.normal(key, (fft_size,), dtype=jnp.float32)
        x_imag = alpha * jax.random.normal(key, (fft_size,), dtype=jnp.float32)
        x = x_real + 1j * x_imag

        # Test inverse FFT
        logger.info("Testing inverse FFT: fft_size=%d", fft_size)
        y_jax = jax.numpy.fft.ifft(x)
        y_real, y_imag = ifft(x.real, x.imag)
        y = y_real + 1j * y_imag

        # Assert inverse FFT results match JAX reference
        np.testing.assert_allclose(
            y, y_jax, atol=atol, err_msg=f"Inverse FFT failed for size {fft_size}"
        )
        logger.info("Inverse FFT results match JAX reference")

        # Test forward FFT
        logger.info("Testing forward FFT: fft_size=%d", fft_size)
        z_jax = jax.numpy.fft.fft(y_jax)
        z_real, z_imag = fft(y.real, y.imag)
        z = z_real + 1j * z_imag

        # Assert forward FFT results match JAX reference
        np.testing.assert_allclose(
            z, z_jax, atol=atol, err_msg=f"Forward FFT failed for size {fft_size}"
        )
        logger.info("Forward FFT results match JAX reference")

        # Assert round-trip (IFFT → FFT) recovers original signal
        np.testing.assert_allclose(
            z, x, atol=atol, err_msg=f"FFT/IFFT round-trip failed for size {fft_size}"
        )
        logger.info("FFT/IFFT round-trip successfully recovered original signal")

    logger.info("All FFT singleton instance tests passed")
