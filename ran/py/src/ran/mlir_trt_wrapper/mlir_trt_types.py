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

"""Type definitions and validation for MLIR-TensorRT runtime."""

import numpy as np
from jax.typing import ArrayLike

try:
    from ml_dtypes import bfloat16, float8_e4m3fn

    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False
    bfloat16 = None
    float8_e4m3fn = None

__all__ = [
    "HAS_ML_DTYPES",
    "SUPPORTED_DTYPES",
    "UNSUPPORTED_DTYPES",
    "bfloat16",
    "float8_e4m3fn",
    "validate_array",
]

# Supported datatypes for MLIR-TensorRT runtime
# Maps numpy dtypes to human-readable names and MLIR types
SUPPORTED_DTYPES = {
    np.float32: {"name": "float32", "mlir_type": "f32"},
    np.float16: {"name": "float16", "mlir_type": "f16"},
    np.int32: {"name": "int32", "mlir_type": "i32"},
    np.bool_: {"name": "bool", "mlir_type": "i1"},
}

# Add bfloat16 if ml_dtypes is available
if HAS_ML_DTYPES:
    SUPPORTED_DTYPES[bfloat16] = {"name": "bfloat16", "mlir_type": "bf16"}

# Unsupported datatypes with suggested alternatives
UNSUPPORTED_DTYPES = {
    np.float64: "float32",
    np.int64: "int32",
    np.complex64: "float32",  # Complex types not yet supported - see MLIR-TensorRT issue
    np.complex128: "float32",
    np.int8: "int32",
    np.int16: "int32",
    np.uint8: "int32",
    np.uint16: "int32",
    np.uint32: "int32",
    np.uint64: "int32",
}


def validate_array(arr: ArrayLike, arg_name: str = "array") -> None:
    """Validate array meets MLIR-TensorRT runtime requirements.

    Args:
        arr: Array to validate
        arg_name: Name of the argument for error messages

    Raises
    ------
        TypeError: If array is not numpy-compatible
        ValueError: If array format is unsupported

    Examples
    --------
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0], dtype=np.float32)
        >>> validate_array(x, "input_0")  # OK
        >>>
        >>> bad = np.array([1.0, 2.0], dtype=np.float64)
        >>> validate_array(bad, "input_1")  # Raises ValueError

    Notes
    -----
        MLIR-TensorRT runtime requires:
        - C-contiguous memory layout
        - Supported dtypes: float32, float16, bfloat16 (if ml_dtypes available), int32, bool
        - Not supported: float64, int64, complex64, complex128 (convert to supported types)
    """
    arr_np = np.asarray(arr)

    # Check data type support
    if arr_np.dtype.type in UNSUPPORTED_DTYPES:
        suggested = UNSUPPORTED_DTYPES[arr_np.dtype.type]
        supported_list = ", ".join(info["name"] for info in SUPPORTED_DTYPES.values())
        raise ValueError(
            f"{arg_name} has unsupported dtype '{arr_np.dtype}'. "
            f"MLIR-TensorRT runtime supports: {supported_list}. "
            f"Suggested alternative: np.{suggested}. "
            f"Convert with: array.astype(np.{suggested})"
        )

    # Special handling for ml_dtypes types
    dtype_to_check = arr_np.dtype.type
    if HAS_ML_DTYPES and arr_np.dtype == bfloat16:
        dtype_to_check = bfloat16

    if dtype_to_check not in SUPPORTED_DTYPES:
        supported_list = ", ".join(info["name"] for info in SUPPORTED_DTYPES.values())
        raise ValueError(
            f"{arg_name} has unknown dtype '{arr_np.dtype}'. "
            f"MLIR-TensorRT runtime supports: {supported_list}."
        )

    # Check memory layout
    if not arr_np.flags.c_contiguous:
        raise ValueError(
            f"{arg_name} is not C-contiguous. "
            f"MLIR-TensorRT requires C-contiguous arrays. "
            f"Convert with: np.ascontiguousarray(array)"
        )
