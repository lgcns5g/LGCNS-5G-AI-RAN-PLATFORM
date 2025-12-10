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

"""Executor for compiled StableHLO MLIR modules using MLIR-TensorRT runtime."""

import ctypes
import logging

import mlir_tensorrt.compiler.api as compiler  # type: ignore
import mlir_tensorrt.runtime.api as runtime  # type: ignore
import numpy as np
from jax.typing import ArrayLike

from ran.mlir_trt_wrapper.mlir_trt_types import (
    HAS_ML_DTYPES,
    validate_array,
)

if HAS_ML_DTYPES:
    from ml_dtypes import bfloat16
else:
    bfloat16 = None

logger = logging.getLogger(__name__)


def execute(
    exe: compiler.Executable,
    inputs: tuple[ArrayLike, ...],
    outputs: tuple[ArrayLike, ...],
    *,
    sync_stream: bool,
    mlir_entry_point: str = "main",
    validate: bool = True,
) -> tuple[ArrayLike, ...]:
    """Execute MLIR-TensorRT executable using the runtime.

    Args:
        exe: MLIR-TensorRT executable
        inputs: Input arrays (must be C-contiguous, float32/int32/complex64)
        outputs: Output arrays (pre-allocated, must be C-contiguous)
        mlir_entry_point: Entry point function name
        sync_stream: Whether to synchronize CUDA stream before/after copying results.
                     Set to True when using TensorRT plugins that execute asynchronously.
                     Set to False for standard operations without custom plugins.
        validate: Whether to validate array requirements (recommended)

    Returns
    -------
        Output arrays with results copied from device to host

    Raises
    ------
        ValueError: If arrays don't meet runtime requirements (when validate=True)
        RuntimeError: If runtime initialization, execution, or result copying fails

    Important:
        Arrays must meet these requirements:
        - **C-contiguous memory layout** (use `np.ascontiguousarray(arr)`)
        - **Supported dtypes**: float32, float16, bfloat16 (if ml_dtypes available), int32, bool
        - **Not supported**: float64, int64, complex64, complex128 (convert to supported types)

    Examples
    --------
        >>> import numpy as np
        >>> import mlir_tensorrt.compiler.api as compiler
        >>> from pathlib import Path
        >>>
        >>> # Load compiled executable
        >>> with open(Path("output/add_func.bin"), "rb") as f:
        ...     exe = compiler.Executable.from_bytes(f.read())
        >>>
        >>> # Prepare C-contiguous float32 arrays
        >>> x = np.array([1.0, 2.0], dtype=np.float32)
        >>> y = np.array([3.0, 4.0], dtype=np.float32)
        >>> output = np.zeros(2, dtype=np.float32)
        >>>
        >>> # Execute
        >>> result = execute(exe, (x, y), (output,))
        >>> print(result[0])  # [4.0, 6.0]

    Notes
    -----
        For JAX arrays, convert to numpy first:
        `jax_array = jnp.array([1.0, 2.0])`
        `np_array = np.asarray(jax_array, dtype=np.float32, order='C')`
    """
    if validate:
        for i, inp in enumerate(inputs):
            validate_array(inp, f"input[{i}]")
        for i, out in enumerate(outputs):
            validate_array(out, f"output[{i}]")

    try:
        client = runtime.RuntimeClient()
        devices = client.get_devices()
        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        session = runtime.RuntimeSession(client, session_options, exe)
    except Exception as e:
        error_msg = "Failed to initialize MLIR-TensorRT runtime session"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    # Create device memrefs for inputs and outputs
    # Special handling for bfloat16: reinterpret as uint16 for memref creation
    try:
        inputs_device = []
        for x in inputs:
            arr = np.asarray(x)
            if HAS_ML_DTYPES and arr.dtype == bfloat16:
                # Reinterpret bfloat16 as uint16 for memref creation
                arr_reinterpreted = arr.view(np.uint16)
                memref = client.create_memref(
                    arr_reinterpreted, dtype=runtime.ScalarTypeCode.bf16, device=devices[0]
                )
            else:
                memref = client.create_memref(arr, device=devices[0])
            inputs_device.append(memref)

        outputs_device = []
        for y in outputs:
            arr = np.asarray(y)
            if HAS_ML_DTYPES and arr.dtype == bfloat16:
                # Reinterpret bfloat16 as uint16 for memref creation
                arr_reinterpreted = arr.view(np.uint16)
                memref = client.create_memref(
                    arr_reinterpreted, dtype=runtime.ScalarTypeCode.bf16, device=devices[0]
                )
            else:
                memref = client.create_memref(arr, device=devices[0])
            outputs_device.append(memref)
    except Exception as e:
        error_msg = "Failed to create device memrefs for inputs/outputs"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    # Execute on device
    try:
        session.execute_function(
            name=mlir_entry_point,
            in_args=inputs_device,
            out_args=outputs_device,
        )

        # Optionally synchronize stream to ensure all GPU work completes before copying results
        # This is necessary when TensorRT plugins execute asynchronously on their own streams
        if sync_stream:
            devices[0].stream.sync()
            logger.debug("CUDA stream synchronized after execute")
    except Exception as e:
        error_msg = f"TensorRT execution failed for function '{mlir_entry_point}'"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    # Copy results from device to host output buffers
    try:
        for i, output_device in enumerate(outputs_device):
            result = client.copy_to_host(output_device)
            output_arr = np.asarray(outputs[i])
            ctypes.memmove(output_arr.ctypes.data, result.ptr, output_arr.nbytes)

        # Synchronize again after copy to ensure host buffers are populated before returning
        # This ensures copy_to_host (which may be asynchronous) completes before we use the data
        if sync_stream:
            devices[0].stream.sync()
            logger.debug("CUDA stream synchronized after copy to host")
    except Exception as e:
        error_msg = "Failed to copy results from device to host"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    return outputs
