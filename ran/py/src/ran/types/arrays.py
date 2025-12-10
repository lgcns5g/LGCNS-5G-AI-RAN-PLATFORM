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
Centralized array type aliases for PHY (NumPy) and PHY (JAX).

Public API exports only the canonical choices per backend to enforce a
single dtype across the codebase and avoid accidental casting:

- NumPy: IntNP, FloatNP, ComplexNP (always available)
- JAX: IntJAX, FloatJAX, ComplexJAX (only when ENABLE_MLIR_TRT=ON)
"""

import numpy as np
import numpy.typing as npt

# NumPy scalar types (always available)
IntNP = np.int64
"""NumPy integer scalar type (int64)."""

FloatNP = np.float64
"""NumPy floating-point scalar type (float64)."""

ComplexNP = np.complex128
"""NumPy complex scalar type (complex128)."""

# NumPy array types for type hinting
type IntArrayNP = npt.NDArray[IntNP]
"""NumPy integer array type (NDArray[int64])."""

type FloatArrayNP = npt.NDArray[FloatNP]
"""NumPy floating-point array type (NDArray[float64])."""

type ComplexArrayNP = npt.NDArray[ComplexNP]
"""NumPy complex array type (NDArray[complex128])."""

# JAX types (only available when ENABLE_MLIR_TRT=ON)
try:
    import jax.numpy as jnp

    IntJAX = jnp.int32
    """JAX integer scalar type (int32). Only available when ENABLE_MLIR_TRT=ON."""

    FloatJAX = jnp.float64
    """JAX floating-point scalar type (float64). Only available when ENABLE_MLIR_TRT=ON."""

    ComplexJAX = jnp.complex128
    """JAX complex scalar type (complex128). Only available when ENABLE_MLIR_TRT=ON."""
except ImportError:
    # JAX not available - MLIR_TRT features disabled
    pass
