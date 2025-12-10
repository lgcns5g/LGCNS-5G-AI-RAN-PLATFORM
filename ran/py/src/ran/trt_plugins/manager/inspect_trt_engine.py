#!/usr/bin/env python3
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

"""TensorRT Engine Inspector

Utility to inspect TensorRT engine files and display detailed information
about inputs, outputs, shapes, and data types.

Usage:
    python inspect_trt_engine.py <engine_file>
    python inspect_trt_engine.py ai_tukey_filter/tensorrt_cluster_engine_data.trtengine
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path
from typing import Any

try:
    import tensorrt as trt
except ImportError:
    print("Error: tensorrt module not found. Please install TensorRT Python bindings.")
    sys.exit(1)

# TensorRT DataType enum mapping to human-readable names
DTYPE_MAP = {
    trt.DataType.FLOAT: "float32 (fp32)",
    trt.DataType.HALF: "float16 (fp16)",
    trt.DataType.INT8: "int8",
    trt.DataType.INT32: "int32",
    trt.DataType.BOOL: "bool",
    trt.DataType.UINT8: "uint8",
    trt.DataType.FP8: "float8 (fp8)",
    trt.DataType.BF16: "bfloat16 (bf16)",
    trt.DataType.INT64: "int64",
    trt.DataType.INT4: "int4",
}

# TensorRT DataType enum to integer mapping (for C++ code generation)
DTYPE_TO_INT = {
    trt.DataType.FLOAT: 0,
    trt.DataType.HALF: 1,
    trt.DataType.INT8: 2,
    trt.DataType.INT32: 3,
    trt.DataType.BOOL: 4,
    trt.DataType.UINT8: 5,
    trt.DataType.FP8: 6,
    trt.DataType.BF16: 7,
    trt.DataType.INT64: 8,
    trt.DataType.INT4: 9,
}


def get_dtype_name(dtype: trt.DataType) -> str:
    """Get human-readable name for TensorRT data type."""
    return DTYPE_MAP.get(dtype, f"Unknown({int(dtype)})")


def get_dtype_cpp_type(dtype: trt.DataType) -> str:
    """Get C++ type string for TensorRT data type."""
    cpp_type_map = {
        trt.DataType.FLOAT: "float",
        trt.DataType.HALF: "__half",
        trt.DataType.INT8: "int8_t",
        trt.DataType.INT32: "int32_t",
        trt.DataType.BOOL: "bool",
        trt.DataType.UINT8: "uint8_t",
        trt.DataType.FP8: "__nv_fp8_e4m3",
        trt.DataType.BF16: "__nv_bfloat16",
        trt.DataType.INT64: "int64_t",
        trt.DataType.INT4: "/* int4 (packed) */",
    }
    return cpp_type_map.get(dtype, f"/* unknown type {int(dtype)} */")


def format_shape(shape: tuple) -> str:
    """Format tensor shape as string."""
    return f"({', '.join(map(str, shape))})"


def calculate_elements(shape: tuple) -> int:
    """Calculate total number of elements in tensor."""
    result = 1
    for dim in shape:
        result *= dim
    return result


def calculate_strides(shape: tuple) -> tuple:
    """Calculate row-major (C-style) strides for a given shape.

    For contiguous tensors, strides are calculated as:
    stride[i] = product of all dimensions after i

    Example: shape (4, 14, 3276, 2)
    - stride[0] = 14 * 3276 * 2 = 91,728
    - stride[1] = 3276 * 2 = 6,552
    - stride[2] = 2
    - stride[3] = 1
    """
    if not shape:
        return tuple()

    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def format_strides(strides: tuple) -> str:
    """Format tensor strides as string."""
    return f"({', '.join(map(str, strides))})"


def load_ran_trt_plugins() -> bool:
    """Load custom RAN TensorRT plugins.

    Returns:
        True if plugins loaded successfully, False otherwise.
    """
    # Get plugin DSO path from environment
    plugin_dso_path = os.environ.get("RAN_TRT_PLUGIN_DSO_PATH")
    if not plugin_dso_path:
        print("Warning: RAN_TRT_PLUGIN_DSO_PATH not set. Custom plugins may not load.")
        print("Attempting to use default path...")
        # Try to find it in the build directory
        build_dir = os.environ.get("RAN_BUILD_DIR", "out/build/clang-debug")
        plugin_dso_path = f"{build_dir}/ran/py/libran_trt_plugin.so"

    plugin_path = Path(plugin_dso_path)
    if not plugin_path.exists():
        print(f"Warning: Plugin library not found at: {plugin_path}")
        return False

    try:
        print(f"Loading plugin library: {plugin_path}")
        plugin_lib = ctypes.CDLL(str(plugin_path))

        # Initialize standard TensorRT plugins
        trt.init_libnvinfer_plugins(None, "")

        # Initialize custom RAN plugins
        if not hasattr(plugin_lib, "init_ran_plugins"):
            print(f"Error: init_ran_plugins function not found in {plugin_path}")
            return False

        init_func = plugin_lib.init_ran_plugins
        init_func.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        init_func.restype = ctypes.c_bool

        result = init_func(None, b"")
        if not result:
            print("Error: Failed to initialize custom TensorRT plugins")
            return False

        print("Custom TensorRT plugins loaded successfully")
        return True

    except Exception as e:
        print(f"Error loading plugins: {e}")
        return False


def inspect_engine(engine_path: Path, verbose: bool = False) -> None:
    """Inspect a TensorRT engine file and display information."""
    if not engine_path.exists():
        print(f"Error: Engine file not found: {engine_path}")
        sys.exit(1)

    print(f"Inspecting TensorRT Engine: {engine_path}")
    print("=" * 80)

    # Load custom RAN TensorRT plugins
    print("\nLoading TensorRT plugins...")
    if not load_ran_trt_plugins():
        print("Warning: Failed to load custom plugins. Engine may not deserialize correctly.")
    print()

    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)

    # Load engine
    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        print("Error: Failed to deserialize engine")
        sys.exit(1)

    # Engine metadata
    print("\nEngine Metadata:")
    print(f"  TensorRT Version: {trt.__version__}")

    # Use v2 API for TensorRT 10.x compatibility
    try:
        mem_size = engine.get_device_memory_size_v2()
    except AttributeError:
        mem_size = engine.device_memory_size

    print(f"  Device Memory Size: {mem_size:,} bytes")
    print(f"  Number of I/O Tensors: {engine.num_io_tensors}")

    # Tensor information
    print("\nTensor Information:")
    print("-" * 80)

    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)

        is_input = mode == trt.TensorIOMode.INPUT
        tensor_list = inputs if is_input else outputs

        tensor_list.append(
            {
                "index": i,
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "mode": "INPUT" if is_input else "OUTPUT",
            }
        )

    # Display inputs
    print(f"\nInputs ({len(inputs)}):")
    for tensor in inputs:
        num_elements = calculate_elements(tensor["shape"])
        strides = calculate_strides(tensor["shape"])
        print(f"  [{tensor['index']}] {tensor['name']}")
        print(f"      Shape: {format_shape(tensor['shape'])}")
        print(f"      Strides (row-major): {format_strides(strides)}")
        print(
            f"      Type:  {get_dtype_name(tensor['dtype'])} (TRT type code: {DTYPE_TO_INT[tensor['dtype']]})"
        )
        print(f"      C++ Type: {get_dtype_cpp_type(tensor['dtype'])}")
        print(f"      Elements: {num_elements:,}")
        print()

    # Display outputs
    print(f"Outputs ({len(outputs)}):")
    for tensor in outputs:
        num_elements = calculate_elements(tensor["shape"])
        strides = calculate_strides(tensor["shape"])
        print(f"  [{tensor['index']}] {tensor['name']}")
        print(f"      Shape: {format_shape(tensor['shape'])}")
        print(f"      Strides (row-major): {format_strides(strides)}")
        print(
            f"      Type:  {get_dtype_name(tensor['dtype'])} (TRT type code: {DTYPE_TO_INT[tensor['dtype']]})"
        )
        print(f"      C++ Type: {get_dtype_cpp_type(tensor['dtype'])}")
        print(f"      Elements: {num_elements:,}")
        print()

    # Generate C++ code snippet
    print("=" * 80)
    print("C++ Code Snippet (for reference):")
    print("-" * 80)
    print()

    for tensor in inputs:
        cpp_type = get_dtype_cpp_type(tensor["dtype"])
        shape_str = format_shape(tensor["shape"])
        print(f"// Input: {tensor['name']} {shape_str}")
        print(f"CudaTensor<{cpp_type}> {tensor['name'].replace('arg', 'input')}{shape_str};")
        print()

    for tensor in outputs:
        cpp_type = get_dtype_cpp_type(tensor["dtype"])
        shape_str = format_shape(tensor["shape"])
        print(f"// Output: {tensor['name']} {shape_str}")
        print(f"CudaTensor<{cpp_type}> {tensor['name'].replace('result', 'output')}{shape_str};")
        print()

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect TensorRT engine files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "engine_file",
        type=Path,
        help="Path to TensorRT engine file (.trtengine)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    inspect_engine(args.engine_file, args.verbose)


if __name__ == "__main__":
    main()
