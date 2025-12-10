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

"""AWGN Channel."""

import jax
import jax.numpy as jnp


def awgn(rng: jax.Array, H: jax.Array, snr_db: float) -> jax.Array:
    """Add AWGN to channel (JAX version).

    Args:
        rng: JAX PRNG key
        H: Channel with shape (n_sc, n_sym, n_ant) complex
        snr_db: SNR in dB

    Returns:
        Noisy channel
    """
    # Calculate noise power
    signal_power = jnp.mean(jnp.abs(H) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate complex noise
    rng_real, rng_imag = jax.random.split(rng)
    noise_real = jax.random.normal(rng_real, H.shape)
    noise_imag = jax.random.normal(rng_imag, H.shape)
    noise = jnp.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

    return H + noise
