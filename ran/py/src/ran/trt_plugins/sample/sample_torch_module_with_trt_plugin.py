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

"""
PyTorch-TensorRT hybrid model integration.

This module provides a PyTorch nn.Module that incorporates TensorRT engines
as intermediate layers, enabling seamless integration of custom TensorRT plugins
within PyTorch models.
"""

import logging
import os
from typing import Optional

import tensorrt as trt
import torch
import torch.nn as nn

from ran.trt_plugins.manager.trt_plugin_manager import (
    TensorRTPluginManager,
    get_ran_trt_engine_path,
)

logger = logging.getLogger(__name__)


class TensorRTLayer(nn.Module):
    """PyTorch layer that wraps a TensorRT engine for execution."""

    def __init__(
        self, trt_engine: object, input_name: str = "input", output_name: str = "output"
    ) -> None:
        """
        Initialize TensorRT layer.

        Args:
            trt_engine: Compiled TensorRT engine
            input_name: Name of the input tensor in the TensorRT network
            output_name: Name of the output tensor in the TensorRT network
        """
        super().__init__()

        self.trt_engine = trt_engine
        self.context = trt_engine.create_execution_context()  # type: ignore[attr-defined]
        self.input_name = input_name
        self.output_name = output_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TensorRT engine using PyTorch CUDA tensors.

        Args:
            x: Input tensor (must be on CUDA device)

        Returns:
            Output tensor from TensorRT engine
        """
        if not x.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")

        # Ensure contiguous memory layout
        x = x.contiguous()

        # Set input shape for dynamic batch size
        self.context.set_input_shape(self.input_name, x.shape)

        # Allocate output tensor with same shape and device as input
        output = torch.empty_like(x, device=x.device, dtype=x.dtype)

        # Set up bindings using tensor data pointers
        bindings = [
            x.data_ptr(),  # input binding
            output.data_ptr(),  # output binding
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")

        return output


class SequentialSumModuleWithTrtPlugin(nn.Module):
    """
    PyTorch model that uses TensorRT sequential sum plugin as an intermediate layer.

    This demonstrates how to combine PyTorch operations with custom TensorRT plugins
    in a seamless, end-to-end differentiable model.
    """

    def __init__(self, trt_engine: Optional[object] = None):
        """
        Initialize hybrid model.

        Args:
            trt_engine: Pre-built TensorRT engine with sequential sum plugin.
                       If None, will be built automatically.
        """
        super().__init__()

        if trt_engine is None:
            trt_engine = self._build_sequential_sum_engine()

        # PyTorch layers
        self.pre_scale = nn.Parameter(torch.tensor(2.0))
        self.trt_layer = TensorRTLayer(trt_engine)
        self.post_bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: PyTorch preprocessing -> TensorRT plugin -> PyTorch postprocessing.

        Args:
            x: Input tensor of shape (batch_size, 5)

        Returns:
            Output tensor after hybrid processing
        """
        # Ensure input is on CUDA
        if not x.is_cuda:
            x = x.cuda()

        # PyTorch preprocessing: scale input
        x = x * self.pre_scale

        # TensorRT sequential sum plugin processing
        x = self.trt_layer(x)

        # PyTorch postprocessing: add bias
        x = x + self.post_bias

        return x

    def _build_sequential_sum_engine(self) -> object:
        """Build TensorRT engine with sequential sum plugin."""
        # Load plugins first
        plugin_manager = TensorRTPluginManager()
        plugin_manager.load_plugin_library()

        # Create TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)

        # Create builder and network
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        # Add input tensor with dynamic batch size
        input_tensor = network.add_input("input", trt.float32, (-1, 5))

        # Create and add sequential sum plugin
        plugin = plugin_manager.create_plugin("SequentialSum", fields=None)
        plugin_layer = network.add_plugin_v3([input_tensor], [], plugin)

        # Mark output
        output_tensor = plugin_layer.get_output(0)
        network.mark_output(output_tensor)
        output_tensor.name = "output"

        # Set optimization profile for dynamic batch size
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 5), (4, 5), (16, 5))  # min, opt, max batch sizes
        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine for C++ test consumption
        engine_dir = get_ran_trt_engine_path()
        os.makedirs(engine_dir, exist_ok=True)
        engine_path = os.path.join(engine_dir, "torch_model_with_trt_plugin.trtengine")
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        logger.info(f"Saved hybrid TensorRT engine to: {engine_path}")

        # Create runtime and deserialize engine
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        return engine


def create_sequential_sum_module_with_trt_plugin(
    device: str = "cuda",
) -> SequentialSumModuleWithTrtPlugin:
    """
    Factory function to create a hybrid PyTorch-TRT model.

    Args:
        device: Target device for the model

    Returns:
        Initialized hybrid model
    """
    model = SequentialSumModuleWithTrtPlugin()
    return model.to(device)
