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
class FreeEnergyFilterConfig:
    """Configuration for free energy filter.

    Frozen dataclass that is hashable for use with JAX static_argnum.

    Attributes
    ----------
    fft_size : int
        FFT size for delay domain processing.
    alpha : float
        Window selection parameter controlling aggressiveness of filtering.
        Higher values result in more aggressive filtering (shorter windows).
    tau_min : int
        Minimum allowed window length.
    tau_max_absolute : int
        Maximum allowed window length.
    delay_compensation_samples : float
        Delay compensation in samples.
    """

    fft_size: int = 2048
    alpha: float = 2.0
    tau_min: int = 0
    tau_max_absolute: int = 1024
    delay_compensation_samples: float = 50.0


def free_energy_filter(
    h_noisy__ri_port_dsym_rxant_dsc: Array,
    n_dmrs_sc: int,
    config: Optional[FreeEnergyFilterConfig] = None,
) -> Array:
    """FFT-based channel denoising using free energy window selection.

    Denoises frequency-domain channel estimates by transforming to delay domain,
    selecting optimal window length that balances signal capture vs noise
    accumulation, and transforming back to frequency domain.

    Algorithm
    ---------
    1. Apply delay compensation using frequency-domain phase shift
    2. Zero-pad input to FFT size samples
    3. Transform to delay domain via IFFT
    4. Estimate noise power from tail samples
    5. Find optimal tau* minimizing: objective(tau) = -E(tau) + alpha * lambda * tau
    6. Apply hard cutoff at tau* in delay domain
    7. Transform back to frequency domain via FFT
    8. Extract original signal portion

    The optimization balances signal energy capture (-E(tau)) vs noise
    accumulation (+alpha * lambda * tau). Alpha controls window size:
    higher = more aggressive filtering.

    Parameters
    ----------
    h_noisy__ri_port_dsym_rxant_dsc : Array
        Noisy frequency-domain channel estimates with stacked real/imag,
        shape (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    n_dmrs_sc : int
        Number of DMRS subcarriers (static, compile-time constant).
    config : FreeEnergyFilterConfig | None, optional
        Configuration for free energy filter. If None, uses default values.

    Returns
    -------
    Array
        Denoised channel estimates with shape
        (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    """
    # Use default config if not provided
    if config is None:
        config = FreeEnergyFilterConfig()

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
    # IFFT with zero padding and convert to float32
    # -----------------------------------------------------------------------

    if n_f > config.fft_size:
        error_msg = f"n_f={n_f} exceeds fft_size={config.fft_size}"
        raise ValueError(error_msg)

    pad_length = config.fft_size - n_f
    h_noisy__ri_batch_sc = jnp.pad(
        h_noisy__ri_batch_sc, ((0, 0), (0, 0), (0, pad_length)), mode="constant", constant_values=0
    )

    # Convert to delay domain using IFFT-2048
    # We convert to float32 because the current algorithm is not stable with float16.
    h_delay_real__batch_sc, h_delay_imag__batch_sc = ifft_2048(
        h_noisy__ri_batch_sc[0].astype(jnp.float32), h_noisy__ri_batch_sc[1].astype(jnp.float32)
    )
    # Scale to maintain power
    h_delay_real__batch_sc = h_delay_real__batch_sc * 2048
    h_delay_imag__batch_sc = h_delay_imag__batch_sc * 2048

    # Repack into stacked format
    h_delay__ri_batch_sc = jnp.stack(
        [h_delay_real__batch_sc, h_delay_imag__batch_sc],
        axis=0,
    )

    # Compute power profile
    # Power = real^2 + imag^2
    h_power__batch_sc = h_delay__ri_batch_sc[0] ** 2 + h_delay__ri_batch_sc[1] ** 2

    # Estimate noise power from tail samples
    noise_start_static = int(2 * config.fft_size / 3)
    noise_samples_batch = h_power__batch_sc[:, noise_start_static : config.fft_size]
    lambda_noise_linear__batch = jnp.mean(noise_samples_batch, axis=1, keepdims=True)

    # Compute cumulative power for E(tau) = sum_{n=0}^{tau} |h[n]|^2
    cumsum_power__batch_tau = jnp.cumsum(h_power__batch_sc[:, : config.tau_max_absolute], axis=1)

    # -----------------------------------------------------------------------
    # Hard window selection
    # -----------------------------------------------------------------------

    # Compute objective with fixed alpha
    tau_range = jnp.arange(1, config.tau_max_absolute + 1)
    penalty__batch_tau = config.alpha * lambda_noise_linear__batch * tau_range
    objective__batch_tau = -cumsum_power__batch_tau + penalty__batch_tau
    valid_mask = tau_range >= config.tau_min
    objective_masked__batch_tau = jnp.where(valid_mask, objective__batch_tau, 1e10)

    # Find tau that minimizes the objective for each batch element
    # Shape: (batch,)
    tau_opt__batch = (
        jnp.argmin(objective_masked__batch_tau, axis=1) + 1
    )  # +1 because tau_range starts at 1

    # Apply hard cutoff at tau_opt in the full delay domain
    window_indices = jnp.arange(config.fft_size)  # Shape: (fft_size,)
    # Broadcasting: (batch, 1) compared with (fft_size,) -> (batch, fft_size)
    mask__batch_sc = jnp.where(window_indices < tau_opt__batch[:, None], 1.0, 0.0)

    # Apply mask to both real and imag components
    # Shape: (2, batch, fft_size) * (batch, fft_size) -> (2, batch, fft_size)
    h_est_delay__ri_batch_sc = h_delay__ri_batch_sc * mask__batch_sc[None, :, :]

    # -----------------------------------------------------------------------
    # FFT, convert to float16, and remove padding
    # -----------------------------------------------------------------------

    # Convert back to frequency domain using FFT-2048 (batched along axis 1)
    # Need to divide by 2048 to undo the IFFT scaling
    h_est__ri_batch_sc_real, h_est__ri_batch_sc_imag = fft_2048(
        h_est_delay__ri_batch_sc[0], h_est_delay__ri_batch_sc[1]
    )
    # Undo IFFT scaling and convert to float16
    h_est__ri_batch_sc_real = (h_est__ri_batch_sc_real / 2048).astype(jnp.float16)
    h_est__ri_batch_sc_imag = (h_est__ri_batch_sc_imag / 2048).astype(jnp.float16)

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

    # Reshape back to original shape
    h_est__ri_port_dsym_rxant_dsc = h_est__ri_batch_sc.reshape(
        2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc
    )

    return h_est__ri_port_dsym_rxant_dsc
