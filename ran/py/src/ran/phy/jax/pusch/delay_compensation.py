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

"""Delay compensation utilities for channel estimation."""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=["forward"])
def delay_compensate(
    h__ri_sc: jax.Array,
    delay_samples: float,
    forward: bool = True,
) -> jax.Array:
    """Apply or undo delay compensation in frequency domain (real-valued).

    This implementation uses real-valued arithmetic for TensorRT compatibility.

    Parameters
    ----------
    h__ri_sc : jax.Array
        Channel estimate with stacked real/imag on axis 0, shape (2, ..., n_sc).
    delay_samples : float
        Delay in samples to compensate. If 0.0, becomes identity operation.
    forward : bool, optional
        If True, apply compensation (negative phase shift).
        If False, undo compensation (positive phase shift).

    Returns
    -------
    jax.Array
        Channel estimate with delay compensation applied or undone.
        Same shape as input (2, ..., n_sc).

    Notes
    -----
    Forward: h_shifted[k] = h[k] * exp(-j * 2π * k * delay / n_sc)
    Reverse: h[k] = h_shifted[k] * exp(j * 2π * k * delay / n_sc)

    Complex multiplication (a + jb) * (c + jd) = (ac - bd) + j(ad + bc) is
    computed using only real arithmetic.
    """
    n_sc = h__ri_sc.shape[-1]
    k = jnp.arange(n_sc)

    # Phase shift direction
    sign = -1.0 if forward else 1.0
    phase = sign * 2.0 * jnp.pi * k * delay_samples / n_sc

    # Compute cos and sin (real-valued)
    cos_phase = jnp.cos(phase)  # (n_sc,)
    sin_phase = jnp.sin(phase)  # (n_sc,)

    # Extract real and imaginary parts
    h_real = h__ri_sc[0]  # (..., n_sc)
    h_imag = h__ri_sc[1]  # (..., n_sc)

    # Complex multiplication: (h_real + j*h_imag) * (cos_phase + j*sin_phase)
    # Real part: h_real * cos_phase - h_imag * sin_phase
    # Imag part: h_real * sin_phase + h_imag * cos_phase
    h_shifted_real = h_real * cos_phase - h_imag * sin_phase
    h_shifted_imag = h_real * sin_phase + h_imag * cos_phase

    return jnp.stack([h_shifted_real, h_shifted_imag], axis=0)


@functools.partial(jax.jit, static_argnames=["forward"])
def delay_compensate_complex(
    h__sc: jax.Array,
    delay_samples: float,
    forward: bool = True,
) -> jax.Array:
    """Apply or undo delay compensation in frequency domain (complex-valued).

    This is a complex-valued convenience wrapper for delay_compensate.

    Parameters
    ----------
    h__sc : jax.Array
        Channel estimate in frequency domain, shape (n_sc,) or (..., n_sc).
        Complex-valued array.
    delay_samples : float
        Delay in samples to compensate. If 0.0, becomes identity operation.
    forward : bool, optional
        If True, apply compensation (negative phase shift).
        If False, undo compensation (positive phase shift).

    Returns
    -------
    jax.Array
        Channel estimate with delay compensation applied or undone.
        Same shape as input, complex-valued.

    Notes
    -----
    Forward: h_shifted[k] = h[k] * exp(-j * 2π * k * delay / n_sc)
    Reverse: h[k] = h_shifted[k] * exp(j * 2π * k * delay / n_sc)
    """
    n_sc = h__sc.shape[-1]
    k = jnp.arange(n_sc)
    sign = -1.0 if forward else 1.0
    phase = sign * 2.0 * jnp.pi * k * delay_samples / n_sc
    phase_shift = jnp.exp(1j * phase)

    return h__sc * phase_shift
