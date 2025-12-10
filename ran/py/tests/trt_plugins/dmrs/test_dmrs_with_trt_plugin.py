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

"""Test DMRS JAX implementation with custom TensorRT plugin."""

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
    from jax import export  # noqa: E402

    from ran import mlir_trt_wrapper as mtw  # noqa: E402
    from ran.phy import numpy as phy  # noqa: E402
    from ran.trt_plugins.dmrs import dmrs_3276  # noqa: E402
    from ran.trt_plugins.manager.trt_plugin_manager import (  # noqa: E402
        copy_trt_engine_for_cpp_tests,
        should_skip_engine_generation,
    )

# Check if engines already exist (evaluated at module import time)
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    _skip_engine_gen = should_skip_engine_generation(["dmrs_test.trtengine"])
else:
    _skip_engine_gen = False

# All tests in this module require MLIR_TRT to be enabled
pytestmark = [
    pytest.mark.skipif(
        os.getenv("ENABLE_MLIR_TRT", "OFF") != "ON",
        reason="MLIR TRT not enabled (ENABLE_MLIR_TRT=OFF)",
    ),
    pytest.mark.skipif(
        _skip_engine_gen,
        reason="TRT engines already exist and SKIP_TRT_ENGINE_GENERATION=1",
    ),
]

logger = logging.getLogger(__name__)


def test_gen_dmrs_sym_jax_with_trt_plugin() -> None:
    """Test DMRS JAX implementation with custom TensorRT plugin.

    The test compiles the JAX DMRS gen_dmrs_seq function to a TensorRT engine executable
    using the MLIR-TensorRT compiler. The JAX DMRS function includes a custom
    TensorRT engine plugin that is used to generate the underlying DMRS sequence.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # --------------------------------
        # Test parameters
        # --------------------------------
        # DMRS inputs:
        slot_number = 0  # slot_number: integer slot number
        n_f = 3276  # length of DMRS sequence per port (must be even)
        n_t = 14  # Number of OFDM symbols per slot
        n_dmrs_id = 0  # DMRS identity (integer)

        # Expected DMRS outputs (row-major/C-style layout):
        # - r_dmrs: stacked real/imag array of shape (2, n_t, 2, n_f//2)
        # - scr_seq: integer array of shape (n_t, 2, n_f)

        # --------------------------------
        # Test output directory
        # --------------------------------

        test_build_dir = Path(temp_dir) / "dmrs_plugin_test"
        test_build_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Test build directory: {test_build_dir}")

        # --------------------------------
        # 5G reference model
        # --------------------------------

        r_dmrs_ref, scr_seq_ref = phy.pusch.gen_dmrs_sym(
            slot_number=slot_number, n_f=n_f, n_t=n_t, n_dmrs_id=n_dmrs_id
        )

        # --------------------------------
        # JAX DMRS function using singleton instance
        # --------------------------------

        # Use the singleton directly - it's callable via __call__
        # Assign to 'gen_dmrs_sym_jax' to maintain stable C++ function names in MLIR export
        # Plugin is pre-configured with sequence_length=3276, n_t=14
        gen_dmrs_sym_jax = dmrs_3276
        r_dmrs__ri_sym_cdm_sc, scr_seq__sym_cdm_sc = gen_dmrs_sym_jax(slot_number, n_dmrs_id)

        # Get config for MLIR compilation - wrap in dictionary mapping target name to config
        trt_plugin_configs = {
            "tensorrt_dmrs_plugin": gen_dmrs_sym_jax.get_config(),
        }

        # Convert stacked real/imag to complex for comparison
        r_dmrs = np.array(r_dmrs__ri_sym_cdm_sc[0, :, :, :]) + 1j * np.array(
            r_dmrs__ri_sym_cdm_sc[1, :, :, :]
        )
        scr_seq = scr_seq__sym_cdm_sc

        # --------------------------------
        # Check functional correctness using cpython runtime
        # --------------------------------

        scr_seq_np = np.array(scr_seq)
        # Reference model uses column-major (n_f, n_t, 2), transpose to row-major (n_t, 2, n_f)
        r_dmrs_ref_row_major = np.einsum("abc->bca", r_dmrs_ref)
        scr_seq_ref_row_major = np.einsum("abc->bca", scr_seq_ref)
        np.testing.assert_allclose(r_dmrs, r_dmrs_ref_row_major, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(scr_seq_np, scr_seq_ref_row_major)
        logger.debug("JAX DMRS function matches 5G reference model")

        # --------------------------------
        # Compile JAX DMRS function to TensorRT engine executable using MLIR-TensorRT
        # executor backend
        # --------------------------------

        # Inputs to the JAX DMRS function (sequence_length and n_t are compile-time constants)
        inputs = (slot_number, n_dmrs_id)

        # Prepare the JAX DMRS function for compilation (nothing is traced at this stage)
        # No static args needed - both inputs are runtime parameters
        jit_hybrid_trt_plugin = jax.jit(gen_dmrs_sym_jax)

        mlir_tensorrt_compilation_flags = [
            "tensorrt-builder-opt-level=0",
            "tensorrt-workspace-memory-pool-limit=50MiB",
        ]

        # Export JAX function to StableHLO MLIR
        exported = jax.export.export(
            jit_hybrid_trt_plugin,
            disabled_checks=[export.DisabledSafetyCheck.custom_call("tensorrt_dmrs_plugin")],
        )(*inputs)
        stablehlo_mlir = exported.mlir_module()

        mlir_trt_exe = mtw.compile(
            stablehlo_mlir=stablehlo_mlir,
            name="gen_dmrs_sym_jax",
            export_path=test_build_dir,
            mlir_entry_point="main",
            mlir_tensorrt_compilation_flags=mlir_tensorrt_compilation_flags,
            trt_plugin_configs=trt_plugin_configs,  # type: ignore[arg-type]
        )

        # --------------------------------
        # Check functional correctness using MLIR-TensorRT runtime
        # --------------------------------

        # Allocate output buffers with correct shapes
        r_dmrs_mlir_trt__ri_sym_cdm_sc = np.zeros(r_dmrs__ri_sym_cdm_sc.shape, dtype=np.float32)
        scr_seq_out_mlir_trt = np.zeros_like(scr_seq)

        # Execute the TensorRT engine with stream synchronization to fix race condition
        # where downstream operations may read results before TensorRT plugin completes
        mtw.execute(
            exe=mlir_trt_exe,
            inputs=(
                np.array(slot_number, dtype=np.int32),
                np.array(n_dmrs_id, dtype=np.int32),
            ),
            outputs=(r_dmrs_mlir_trt__ri_sym_cdm_sc, scr_seq_out_mlir_trt),
            sync_stream=True,  # Synchronize CUDA stream after execution
        )

        # Convert stacked real/imag to complex for comparison
        r_dmrs_mlir_trt = (
            r_dmrs_mlir_trt__ri_sym_cdm_sc[0, :, :, :]
            + 1j * r_dmrs_mlir_trt__ri_sym_cdm_sc[1, :, :, :]
        )

        # Verify results are approximately equal
        np.testing.assert_allclose(r_dmrs_mlir_trt, r_dmrs, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(scr_seq_out_mlir_trt, scr_seq_ref_row_major)
        logger.debug("MLIR-TensorRT runtime output matches reference")

        # --------------------------------
        # Copy TensorRT engine to location expected by C++ test
        # --------------------------------

        engine_dest = copy_trt_engine_for_cpp_tests(test_build_dir, "dmrs_test.trtengine")
        logger.debug(f"Copied TensorRT engine to {engine_dest}")

        # --------------------------------
        # Python test completed - C++ application testing moved to separate CMake target
        # --------------------------------

        logger.debug("Python DMRS TensorRT plugin test completed successfully!")
        logger.debug(
            "Note: Testing of generated C++ artifacts is handled by "
            "the 'ran_py_cpp_integration_tests' CMake target"
        )
