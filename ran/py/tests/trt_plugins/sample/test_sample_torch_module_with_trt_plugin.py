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

"""Tests for PyTorch-TensorRT hybrid model functionality."""

from __future__ import annotations

import logging
import os

import pytest

logger = logging.getLogger(__name__)

# Only import TensorRT, PyTorch and internal dependencies when MLIR_TRT is enabled
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    import tensorrt as trt  # noqa: E402
    import torch  # noqa: E402

    from ran.trt_plugins.manager.trt_plugin_manager import (  # noqa: E402
        TensorRTPluginManager,
        get_ran_trt_engine_path,
        should_skip_engine_generation,
    )
    from ran.trt_plugins.sample import create_sequential_sum_module_with_trt_plugin  # noqa: E402

# Check if engines already exist (evaluated at module import time)
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    _skip_engine_gen = should_skip_engine_generation(
        ["sequential_sum_test.trtengine", "torch_model_with_trt_plugin.trtengine"]
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


def test_sequential_sum_plugin_forward_pass() -> None:
    """Test complete TensorRT engine creation and forward pass with sequential sum plugin."""

    try:
        # Load plugins first
        plugin_manager = TensorRTPluginManager()
        plugin_manager.load_plugin_library()

        # Test data using PyTorch tensors
        input_tensor_data = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda"
        )
        expected_output = torch.tensor(
            [1.0, 3.0, 6.0, 10.0, 15.0], dtype=torch.float32, device="cuda"
        )

        # Reshape input for TensorRT (batch dimension)
        input_tensor_data = input_tensor_data.unsqueeze(0)  # Shape: (1, 5)
        expected_output = expected_output.unsqueeze(0)  # Shape: (1, 5)

        # Create TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)

        # Create builder and network
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        # Add input tensor
        input_tensor = network.add_input("input", trt.float32, (1, 5))

        # Create and add sequential sum plugin
        plugin = plugin_manager.create_plugin("SequentialSum", fields=None)
        assert plugin is not None, "Failed to create sequential sum plugin"

        plugin_layer = network.add_plugin_v3([input_tensor], [], plugin)
        assert plugin_layer is not None, "Failed to add plugin layer to network"

        # Mark output
        output_tensor = plugin_layer.get_output(0)
        network.mark_output(output_tensor)
        output_tensor.name = "output"

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        assert serialized_engine is not None, "Failed to build TensorRT engine"

        # Save engine for C++ test consumption
        engine_dir = get_ran_trt_engine_path()
        os.makedirs(engine_dir, exist_ok=True)
        engine_path = os.path.join(engine_dir, "sequential_sum_test.trtengine")
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        logger.info(f"Saved TensorRT engine to: {engine_path}")

        # Create runtime and deserialize engine
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        assert engine is not None, "Failed to deserialize engine"

        # Create execution context
        context = engine.create_execution_context()
        assert context is not None, "Failed to create execution context"

        # Allocate PyTorch tensors on GPU
        output_tensor_data = torch.zeros_like(expected_output, device="cuda")

        # Set up bindings using PyTorch tensor data pointers
        bindings = [input_tensor_data.data_ptr(), output_tensor_data.data_ptr()]

        # Execute
        success = context.execute_v2(bindings)
        assert success, "Failed to execute TensorRT"

        # Verify results using PyTorch tensor comparison
        torch.testing.assert_close(output_tensor_data, expected_output, rtol=1e-5, atol=1e-6)

    except Exception as e:
        # Let the actual exception propagate for better debugging
        import traceback

        traceback.print_exc()
        assert False, f"TensorRT engine test failed: {e}"


def test_pytorch_trt_model_with_trt_plugin() -> None:
    """Test PyTorch-TRT hybrid model using modular design with torch tensors."""

    try:
        # Test data - batch of 2 samples
        input_data = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32
        )

        # Create hybrid model using our modular design
        model = create_sequential_sum_module_with_trt_plugin(device="cuda")
        model.eval()  # Set to evaluation mode

        # Run the hybrid model
        with torch.no_grad():
            output = model(input_data)

        # Expected computation (TensorRT plugin processes flattened tensor):
        # Input: [[1,2,3,4,5], [2,3,4,5,6]]
        # After scale by 2: [[2,4,6,8,10], [4,6,8,10,12]]
        # Flattened: [2,4,6,8,10,4,6,8,10,12]
        # Sequential sum: [2,6,12,20,30,34,40,48,58,70]
        # Reshaped: [[2,6,12,20,30], [34,40,48,58,70]]
        # After add bias 1: [[3,7,13,21,31], [35,41,49,59,71]]
        expected_output = torch.tensor(
            [[3.0, 7.0, 13.0, 21.0, 31.0], [35.0, 41.0, 49.0, 59.0, 71.0]],
            dtype=torch.float32,
            device="cuda",
        )

        # Verify results
        torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-6)

    except Exception as e:
        import traceback

        traceback.print_exc()
        assert False, f"PyTorch-TRT hybrid model test failed: {e}"
