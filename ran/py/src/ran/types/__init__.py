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
Public entry point for canonical array type aliases.

Exports only the selected dtypes to keep the codebase consistent:
  - NumPy: IntArrayNP (int32), FloatArrayNP (float64), ComplexArrayNP (complex128)
  - JAX:   IntJAX, FloatJAX, ComplexJAX (only when ENABLE_MLIR_TRT=ON)
"""

# NumPy types (always available)
from .arrays import (
    ComplexArrayNP,
    ComplexNP,
    FloatArrayNP,
    FloatNP,
    IntArrayNP,
    IntNP,
)

__all__ = [
    "ComplexArrayNP",
    "ComplexNP",
    "FloatArrayNP",
    "FloatNP",
    "IntArrayNP",
    "IntNP",
]

# JAX types (only when ENABLE_MLIR_TRT=ON)
try:
    from .arrays import ComplexJAX, FloatJAX, IntJAX  # noqa: F401

    __all__.extend(["ComplexJAX", "FloatJAX", "IntJAX"])
except ImportError:
    # JAX not available - MLIR_TRT features disabled
    pass
