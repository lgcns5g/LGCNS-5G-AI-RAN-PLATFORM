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

"""TensorRT plugin field type map."""

import numpy as np
import tensorrt as trt
from jax import numpy as jnp

TRT_PLUGIN_FIELD_TYPE_MAP = {
    "INT4": {
        "trt": trt.PluginFieldType.INT4,
        "types": [jnp.int4],
    },
    "INT8": {
        "trt": trt.PluginFieldType.INT8,
        "types": [np.int8, jnp.int8],
    },
    "INT16": {
        "trt": trt.PluginFieldType.INT16,
        "types": [np.int16, jnp.int16],
    },
    "INT32": {
        "trt": trt.PluginFieldType.INT32,
        "types": [np.int32, jnp.int32],
    },
    "INT64": {
        "trt": trt.PluginFieldType.INT64,
        "types": [np.int64, jnp.int64],
    },
    "FLOAT16": {
        "trt": trt.PluginFieldType.FLOAT16,
        "types": [np.float16, jnp.float16],
    },
    "FLOAT32": {
        "trt": trt.PluginFieldType.FLOAT32,
        "types": [np.float32, jnp.float32],
    },
    "FLOAT64": {
        "trt": trt.PluginFieldType.FLOAT64,
        "types": [np.float64, jnp.float64],
    },
    "CHAR": {
        "trt": trt.PluginFieldType.CHAR,
        "types": [str],
    },
}


def _convert_type_to_trt(field_type: object) -> trt.DataType:
    """Convert Python/NumPy/JAX type to TensorRT DataType.

    Args:
        field_type: Type instance to convert

    Returns
    -------
        Corresponding TensorRT DataType

    Raises
    ------
        ValueError: If field type is not supported
    """
    for type_info in TRT_PLUGIN_FIELD_TYPE_MAP.values():
        # Check if field_type matches any of the supported types
        if any(isinstance(field_type, t) for t in type_info["types"]):
            return type_info["trt"]

    raise ValueError(f"Unsupported field type: {field_type}")
