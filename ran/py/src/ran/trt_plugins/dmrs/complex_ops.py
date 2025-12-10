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

"""Complex arithmetic operations on stacked real/imag tensors.

Operations on tensors where dimension 0 contains [real, imag] components.
"""

import jax.numpy as jnp
from jax import Array


def complex_mul_conj(
    a__ri: Array,
    b__ri: Array,
) -> Array:
    """Complex multiplication with conjugate: a * conj(b).

    Computes (a_real + i*a_imag) * (b_real - i*b_imag) for tensors where
    the first dimension contains [real, imag] components.

    Args:
        a__ri: Complex tensor with shape (2, ...)
        b__ri: Complex tensor with shape (2, ...)

    Returns
    -------
        result__ri: Complex product with shape (2, ...)

    Notes
    -----
        Formula: (a+bi) * (c-di) = (ac+bd) + (bc-ad)i
        - result[0] = real part = a[0]*b[0] + a[1]*b[1]
        - result[1] = imag part = a[1]*b[0] - a[0]*b[1]
    """
    real_part = a__ri[0] * b__ri[0] + a__ri[1] * b__ri[1]
    imag_part = a__ri[1] * b__ri[0] - a__ri[0] * b__ri[1]

    return jnp.stack([real_part, imag_part], axis=0)


def complex_mul(
    a__ri: Array,
    b__ri: Array,
) -> Array:
    """Complex multiplication: a * b.

    Computes (a_real + i*a_imag) * (b_real + i*b_imag) for tensors where
    the first dimension contains [real, imag] components.

    Args:
        a__ri: Complex tensor with shape (2, ...)
        b__ri: Complex tensor with shape (2, ...)

    Returns
    -------
        result__ri: Complex product with shape (2, ...)

    Notes
    -----
        Formula: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        - result[0] = real part = a[0]*b[0] - a[1]*b[1]
        - result[1] = imag part = a[0]*b[1] + a[1]*b[0]
    """
    real_part = a__ri[0] * b__ri[0] - a__ri[1] * b__ri[1]
    imag_part = a__ri[0] * b__ri[1] + a__ri[1] * b__ri[0]

    return jnp.stack([real_part, imag_part], axis=0)
