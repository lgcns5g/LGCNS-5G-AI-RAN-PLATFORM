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

from dataclasses import dataclass
from typing import Optional

from jax import Array, numpy as jnp

from ran.trt_plugins.fft import fft_2048, ifft_2048

from .delay_compensation import delay_compensate


@dataclass(frozen=True)
class WeightedThresholdFilterConfig:
    """Configuration for weighted threshold filter.

    Frozen dataclass that is hashable for use with JAX static_argnum.

    Attributes
    ----------
    fft_size : int
        FFT size for delay domain processing.
    delay_compensation_samples : float
        Delay compensation in samples.
    decay_rate : float
        Exponential decay rate for delay-dependent threshold weighting.
        Higher values suppress late taps more aggressively.
    k_sigma : float
        Statistical threshold multiplier (number of standard deviations above mean).
        Higher values result in less aggressive filtering.
    """

    fft_size: int = 2048
    delay_compensation_samples: float = 0.0
    decay_rate: float = 0.01
    k_sigma: float = 3.0


def weighted_threshold_filter(
    h_noisy__ri_port_dsym_rxant_dsc: Array,
    n_dmrs_sc: int,
    config: Optional[WeightedThresholdFilterConfig] = None,
) -> Array:
    """FFT-based channel denoising using weighted statistical thresholding.

    Denoises frequency-domain channel estimates by transforming to delay domain,
    applying delay-weighted statistical thresholding to suppress noise while
    preserving signal taps, and transforming back to frequency domain.

    Algorithm
    ---------
    1. Apply delay compensation using frequency-domain phase shift
    2. Zero-pad input to FFT size samples
    3. Transform to delay domain via IFFT
    4. Estimate noise statistics (mean, std) from tail samples
    5. Compute base threshold: mean + k_sigma * std
    6. Apply exponential delay weighting: threshold *= exp(decay_rate * delay)
    7. Zero out taps below weighted threshold
    8. Transform back to frequency domain via FFT
    9. Extract original signal portion

    The weighted threshold increases exponentially with delay, suppressing
    late taps more aggressively than early taps to balance noise reduction
    with signal preservation.

    Parameters
    ----------
    h_noisy__ri_port_dsym_rxant_dsc : Array
        Noisy frequency-domain channel estimates with stacked real/imag,
        shape (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    n_dmrs_sc : int
        Number of DMRS subcarriers (static, compile-time constant).
    config : WeightedThresholdFilterConfig | None, optional
        Configuration for weighted threshold filter. If None, uses default values.

    Returns
    -------
    Array
        Denoised channel estimates with shape
        (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    """
    # Use default config if not provided
    if config is None:
        config = WeightedThresholdFilterConfig()

    # Reshape for batched FFT-based denoising
    # Extract other dimensions from shape (these are not used in slicing)
    n_dmrs_port = h_noisy__ri_port_dsym_rxant_dsc.shape[1]
    n_dmrs_syms = h_noisy__ri_port_dsym_rxant_dsc.shape[2]
    n_rxant = h_noisy__ri_port_dsym_rxant_dsc.shape[3]

    h_noisy__ri_batch_sc = h_noisy__ri_port_dsym_rxant_dsc.reshape(
        2, n_dmrs_port * n_dmrs_syms * n_rxant, n_dmrs_sc
    )

    n_f = n_dmrs_sc  # Use static parameter instead of extracting from shape

    # Apply delay compensation (forward)
    h_noisy__ri_batch_sc = delay_compensate(
        h_noisy__ri_batch_sc,
        delay_samples=config.delay_compensation_samples,
        forward=True,
    )

    # -----------------------------------------------------------------------
    # IFFT with zero padding
    # -----------------------------------------------------------------------

    if n_f > config.fft_size:
        error_msg = f"n_f={n_f} exceeds fft_size={config.fft_size}"
        raise ValueError(error_msg)

    pad_length = config.fft_size - n_f
    h_noisy__ri_batch_sc = jnp.pad(
        h_noisy__ri_batch_sc, ((0, 0), (0, 0), (0, pad_length)), mode="constant", constant_values=0
    )

    # Convert to delay domain using IFFT-2048
    h_delay_real__batch_sc, h_delay_imag__batch_sc = ifft_2048(
        h_noisy__ri_batch_sc[0].astype(jnp.float32), h_noisy__ri_batch_sc[1].astype(jnp.float32)
    )
    # Scale to maintain power
    h_delay_real__batch_sc = h_delay_real__batch_sc * config.fft_size
    h_delay_imag__batch_sc = h_delay_imag__batch_sc * config.fft_size

    # Repack into stacked format
    h_delay__ri_batch_sc = jnp.stack(
        [h_delay_real__batch_sc, h_delay_imag__batch_sc],
        axis=0,
    )

    # -----------------------------------------------------------------------
    # Zero out taps below threshold (defined by estimated noise power)
    # -----------------------------------------------------------------------

    # Compute power profile
    # Power = real^2 + imag^2
    h_power__batch_sc = h_delay__ri_batch_sc[0] ** 2 + h_delay__ri_batch_sc[1] ** 2

    # Estimate noise power from tail samples
    noise_start_static = int(2 * config.fft_size / 3)
    noise_samples_batch = h_power__batch_sc[:, noise_start_static : config.fft_size]

    # Compute noise statistics
    noise_mean__batch = jnp.mean(noise_samples_batch, axis=1, keepdims=True)
    noise_std__batch = jnp.std(noise_samples_batch, axis=1, keepdims=True)

    # Statistical threshold: mean + k*std
    threshold__batch = noise_mean__batch + config.k_sigma * noise_std__batch

    # Create delay-dependent scaling factor
    delay_indices = jnp.arange(config.fft_size)
    delay_penalty = jnp.exp(config.decay_rate * delay_indices)

    # Apply to threshold (broadcast across batch dimension)
    threshold__batch = threshold__batch * delay_penalty[None, :]

    # Zero out taps below threshold
    h_est_delay__ri_batch_sc = jnp.where(
        h_power__batch_sc < threshold__batch, 0.0, h_delay__ri_batch_sc
    )

    # -----------------------------------------------------------------------
    # FFT, convert to float16, and remove padding
    # -----------------------------------------------------------------------

    # Convert back to frequency domain using FFT-2048 (batched along axis 1)
    # Need to divide by 2048 to undo the IFFT scaling
    h_est__ri_batch_sc_real, h_est__ri_batch_sc_imag = fft_2048(
        h_est_delay__ri_batch_sc[0], h_est_delay__ri_batch_sc[1]
    )
    # Undo IFFT scaling
    h_est__ri_batch_sc_real = (h_est__ri_batch_sc_real / config.fft_size).astype(jnp.float16)
    h_est__ri_batch_sc_imag = (h_est__ri_batch_sc_imag / config.fft_size).astype(jnp.float16)

    h_est__ri_batch_sc = jnp.stack(
        [h_est__ri_batch_sc_real, h_est__ri_batch_sc_imag],
        axis=0,
    )

    # Extract the original signal portion (first n_f samples)
    h_est__ri_batch_sc = h_est__ri_batch_sc[:, :, :n_f]

    # Undo delay compensation (reverse)
    h_est__ri_batch_sc = delay_compensate(
        h_est__ri_batch_sc,
        delay_samples=config.delay_compensation_samples,
        forward=False,
    )

    # Reshape back to original shape and return
    return h_est__ri_batch_sc.reshape(2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc)
