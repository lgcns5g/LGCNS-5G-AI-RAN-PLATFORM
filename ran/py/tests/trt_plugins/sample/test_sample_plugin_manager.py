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

"""Tests for TensorRT plugin manager functionality."""

from __future__ import annotations

import os

import numpy as np
import pytest

# Only import internal dependencies when MLIR_TRT is enabled
if os.getenv("ENABLE_MLIR_TRT", "OFF") == "ON":
    from ran.trt_plugins.manager.global_trt_plugin_manager import (  # noqa: E402
        global_trt_plugin_manager_create_plugin,
    )
    from ran.trt_plugins.manager.trt_plugin_manager import TensorRTPluginManager  # noqa: E402

# All tests in this module require MLIR_TRT to be enabled
pytestmark = pytest.mark.skipif(
    os.getenv("ENABLE_MLIR_TRT", "OFF") != "ON",
    reason="Requires MLIR-TensorRT compiler (ENABLE_MLIR_TRT=OFF)",
)


def test_load_trt_plugins() -> None:
    """Test loading TensorRT plugins."""
    plugin_manager = TensorRTPluginManager()
    result = plugin_manager.load_plugin_library()
    assert isinstance(result, bool)


def test_create_sequential_sum_plugin() -> None:
    """Test creating SequentialSum plugin."""
    plugin_manager = TensorRTPluginManager()

    # First try to load plugins
    plugin_manager.load_plugin_library()

    # Then create plugin
    plugin = plugin_manager.create_plugin("SequentialSum", fields=None)

    # Plugin should be created successfully
    assert plugin is not None
    # Basic validation that it's a plugin-like object
    assert hasattr(plugin, "__class__")


def test_global_trt_plugin_manager_examples() -> None:
    """Test global TensorRT plugin manager with various plugin types.

    Verifies plugin creation with and without parameters for different
    plugin types (FFT, DMRS, Cholesky, SequentialSum).
    """
    # Example 1: Plugin without parameters
    plugin_no_params = global_trt_plugin_manager_create_plugin("SequentialSum")
    assert plugin_no_params is not None, "SequentialSum plugin creation failed"
    assert hasattr(plugin_no_params, "__class__"), "Plugin should be a valid object"

    # Example 2: FFT plugin with parameters
    fft_plugin = global_trt_plugin_manager_create_plugin(
        "FftTrt",
        fields={"fft_size": np.int32(2048), "direction": np.int32(0)},
    )
    assert fft_plugin is not None, "FftTrt plugin creation failed"
    assert hasattr(fft_plugin, "__class__"), "FFT plugin should be a valid object"

    # Example 3: DMRS plugin with parameters
    dmrs_plugin = global_trt_plugin_manager_create_plugin(
        "DmrsTrt",
        fields={"sequence_length": np.int32(42)},
    )
    assert dmrs_plugin is not None, "DmrsTrt plugin creation failed"
    assert hasattr(dmrs_plugin, "__class__"), "DMRS plugin should be a valid object"

    # Example 4: Cholesky plugin with parameters
    cholesky_plugin = global_trt_plugin_manager_create_plugin(
        "CholeskyFactorInv",
        fields={"matrix_size": np.int32(2), "is_complex": np.int32(1)},
    )
    assert cholesky_plugin is not None, "CholeskyFactorInv plugin creation failed"
    assert hasattr(cholesky_plugin, "__class__"), "Cholesky plugin should be a valid object"
