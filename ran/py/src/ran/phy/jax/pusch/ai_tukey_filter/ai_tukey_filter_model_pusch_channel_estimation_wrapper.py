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
from pathlib import Path

import jax
import yaml
from flax import serialization
from jax import Array, numpy as jnp

from ran.phy.jax.pusch.ai_tukey_filter.ai_tukey_filter_model import create_model
from ran.phy.jax.pusch.delay_compensation import delay_compensate
from ran.trt_plugins.fft import fft_2048, ifft_2048

# Model cache (loaded once per config)
_MODEL_CACHE = {}

# ----------------------------------------------------------------------------
# AI Tukey Filter Configuration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AITukeyFilterConfig:
    """Configuration for AI Tukey filter model.

    Frozen dataclass that is hashable for use with JAX static_argnum.

    Attributes:
        model_dir: Path to model directory containing model_params.flax and model_config.yaml
        compressed_len: Compressed representation length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        max_tau: Maximum delay span
        input_subsample_factor: Input subsampling factor
        fft_size: FFT size for delay domain processing
        delay_compensation_samples: Delay compensation in samples
    """

    model_dir: str
    compressed_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_tau: int = 1024
    input_subsample_factor: int = 64
    fft_size: int = 2048
    delay_compensation_samples: float = 50.0


def load_model_config_from_yaml(model_dir: str | Path, fft_size: int = 2048) -> AITukeyFilterConfig:
    """Load AI Tukey filter configuration from saved model directory.

    Reads the model_config.yaml file from a trained model checkpoint directory
    and constructs an AITukeyFilterConfig with the saved hyperparameters.

    Args:
        model_dir: Path to model directory containing model_config.yaml and model_params.flax
        fft_size: FFT size for delay domain processing (must match value used during compilation)

    Returns:
        AITukeyFilterConfig with parameters loaded from YAML file

    Raises:
        FileNotFoundError: If model_config.yaml does not exist in model_dir

    Example:
        >>> model_path = Path("out/ai_tukey_filter_tutorial_training/checkpoints")
        >>> config = load_model_config_from_yaml(model_path, fft_size=2048)
        >>> print(config.delay_compensation_samples)
        0.0
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "model_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model configuration not found at {config_path}. "
            f"Ensure the model has been trained and saved correctly."
        )

    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    model_cfg = yaml_config["model_config"]
    train_cfg = yaml_config["training_config"]

    return AITukeyFilterConfig(
        model_dir=str(model_dir),
        compressed_len=model_cfg["compressed_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_tau=model_cfg["max_tau"],
        input_subsample_factor=model_cfg["input_subsample_factor"],
        fft_size=fft_size,
        delay_compensation_samples=train_cfg["delay_compensation_samples"],
    )


def _load_tukey_predictor_model(config: AITukeyFilterConfig) -> tuple[object, object, bool]:
    """Load trained Tukey predictor model using FLAX bytes serialization.

    Args:
        config: AI Tukey filter configuration (should be loaded via load_model_config_from_yaml)

    Returns:
        Tuple of (model, params, success_flag)
    """
    # Determine model directory
    model_dir = Path(config.model_dir)
    if not model_dir.exists():
        return None, None, False

    # Check for model params file
    params_path = model_dir / "model_params.flax"
    if not params_path.exists():
        print(f"Error: Model params file not found at {params_path}")
        return None, None, False

    # Create model using provided configuration
    model = create_model(
        compressed_len=config.compressed_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout_rate=0.0,
        max_tau=config.max_tau,
        input_subsample_factor=config.input_subsample_factor,
    )

    # Create empty params structure for deserialization
    dummy_rng = jax.random.PRNGKey(0)
    dummy_cumsum = jnp.zeros((1, config.max_tau // config.input_subsample_factor))
    dummy_noise_db = jnp.zeros((1, 1))
    dummy_energy_db = jnp.zeros((1, 1))
    dummy_params_dict = model.init(dummy_rng, dummy_cumsum, dummy_noise_db, dummy_energy_db)
    empty_params = dummy_params_dict["params"]

    # Load params from FLAX bytes format
    with open(params_path, "rb") as f:
        bytes_data = f.read()
    params = serialization.from_bytes(empty_params, bytes_data)

    print(f"✓ Loaded Tukey predictor model from {model_dir.name}")
    print(
        f"  Model config: {config.compressed_len=}, {config.d_model=}, {config.n_heads=}, {config.n_layers=}"
    )
    print(f"  Delay compensation: {config.delay_compensation_samples} samples")
    print(f"  Effective input length: {config.max_tau // config.input_subsample_factor}")

    return model, params, True


def _get_or_load_model(config: AITukeyFilterConfig) -> tuple[object, object, bool]:
    """Get model from cache or load it.

    Args:
        config: AI Tukey filter configuration

    Returns:
        Tuple of (model, params, success_flag)
    """
    # Use config as cache key (frozen dataclass is hashable)
    if config not in _MODEL_CACHE:
        _MODEL_CACHE[config] = _load_tukey_predictor_model(config)

    return _MODEL_CACHE[config]


def tukey_window_impl(tau__batch: Array, alpha__batch: Array, fft_size: int = 2048) -> Array:
    """
    Compute differentiable Tukey window (vectorized over batch).

    Args:
        tau__batch: Window lengths (n_batch, 1)
        alpha__batch: Taper parameters [0, 1] (n_batch, 1)
        fft_size: FFT size for window generation

    Returns:
        window__batch_sc: Tukey windows (n_batch, fft_size)
    """
    indices = jnp.arange(fft_size, dtype=jnp.float32)
    indices_batch = indices[None, :]

    # Clamp alpha to avoid division by zero
    alpha_clamped = jnp.maximum(alpha__batch, 1e-6)

    # Compute boundaries
    left_boundary = alpha_clamped * tau__batch / 2.0  # (n_batch, 1)
    right_boundary = tau__batch * (1.0 - alpha_clamped / 2.0)  # (n_batch, 1)

    # Left taper: 0.5 * (1 + cos(pi * (2*n/(alpha*tau) - 1)))
    left_arg = jnp.pi * (2.0 * indices_batch / (alpha_clamped * tau__batch) - 1.0)
    left_taper = 0.5 * (1.0 + jnp.cos(left_arg))

    # Right taper: 0.5 * (1 + cos(pi * (2*n/(alpha*tau) - 2/alpha + 1)))
    right_arg = jnp.pi * (
        2.0 * indices_batch / (alpha_clamped * tau__batch) - 2.0 / alpha_clamped + 1.0
    )
    right_taper = 0.5 * (1.0 + jnp.cos(right_arg))

    # Smooth transitions using sigmoid for differentiability
    steepness = 10.0

    # Create smooth masks for each region
    in_left = jax.nn.sigmoid(steepness * (left_boundary - indices_batch))
    in_middle = jax.nn.sigmoid(steepness * (indices_batch - left_boundary)) * jax.nn.sigmoid(
        steepness * (right_boundary - indices_batch)
    )
    in_right = jax.nn.sigmoid(steepness * (indices_batch - right_boundary)) * jax.nn.sigmoid(
        steepness * (tau__batch - indices_batch)
    )

    # Combine regions
    window = left_taper * in_left + 1.0 * in_middle + right_taper * in_right

    return window  # (n_batch, fft_size)


def ai_tukey_filter(
    h_noisy__ri_port_dsym_rxant_dsc: Array,
    config: AITukeyFilterConfig,
    n_dmrs_sc: int,  # Static: number of DMRS subcarriers
) -> Array:
    """NN-based channel denoising using learned Tukey window prediction.

    Uses a trained transformer model to predict optimal Tukey window parameters
    (tau and alpha) for delay-domain filtering based on cumulative power profile,
    noise estimate, and total energy.

    Algorithm:
        1. Apply delay compensation using frequency-domain phase shift
        2. Zero-pad input to fft_size samples
        3. Transform to delay domain via fft_size-point IFFT
        4. Compute power profile and estimate noise from tail samples
        5. Extract features: normalized cumulative power, noise power (dB), total energy (dB)
        6. Use trained transformer to predict optimal (tau, alpha)
        7. Apply Tukey window with predicted parameters
        8. Transform back to frequency domain via fft_size-point FFT
        9. Extract original signal portion
        10. Undo delay compensation

    Args:
        h_noisy__ri_port_dsym_rxant_dsc: Noisy frequency-domain channel estimates with
            stacked real/imag, shape (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc)
        config: AI Tukey filter configuration
        n_dmrs_sc: Number of DMRS subcarriers (static, compile-time constant)

    Returns:
        Denoised channel estimates, shape (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc)

    Raises:
        RuntimeError: If Tukey predictor model is not available
    """
    # Load model from cache or initialize it
    model, params, model_available = _get_or_load_model(config)

    if not model_available:
        raise RuntimeError(
            f"Tukey predictor model not available at {config.model_dir}. "
            "Please ensure model_params.flax and model_config.yaml exist in the model "
            "directory."
        )

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
        error_msg = f"n_f={n_f} exceeds config.fft_size={config.fft_size}"
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
    h_delay_real__batch_sc = h_delay_real__batch_sc * config.fft_size
    h_delay_imag__batch_sc = h_delay_imag__batch_sc * config.fft_size

    # Repack into stacked format
    h_delay__ri_batch_sc = jnp.stack([h_delay_real__batch_sc, h_delay_imag__batch_sc], axis=0)

    # Compute power profile
    # Power = real^2 + imag^2
    h_power__batch_sc = h_delay__ri_batch_sc[0] ** 2 + h_delay__ri_batch_sc[1] ** 2

    # Estimate noise power from tail samples
    noise_start_static = int(2 * config.fft_size / 3)
    noise_samples_batch = h_power__batch_sc[:, noise_start_static : config.fft_size]
    lambda_noise_linear__batch = jnp.mean(noise_samples_batch, axis=1, keepdims=True)

    # Compute cumulative power
    cumsum_power__batch_tau = jnp.cumsum(h_power__batch_sc[:, : config.max_tau], axis=1)

    # -----------------------------------------------------------------------
    # Get Tukey window params from AI model
    # -----------------------------------------------------------------------

    # Features for AI model
    total_energy = cumsum_power__batch_tau[:, -1:]
    cumsum_power_norm = cumsum_power__batch_tau / (total_energy + 1e-10)

    # Convert to dB
    lambda_noise_db = 10.0 * jnp.log10(lambda_noise_linear__batch + 1e-10)
    total_energy_db = 10.0 * jnp.log10(total_energy + 1e-10)

    # Predict optimal tau and alpha
    tau_pred__batch, alpha_pred__batch = model.apply(  # type: ignore[attr-defined]
        {"params": params},
        cumsum_power_norm,
        lambda_noise_db,
        total_energy_db,
        training=False,
    )

    # -----------------------------------------------------------------------
    # Apply Tukey window with predicted parameters
    # -----------------------------------------------------------------------

    mask__batch_sc = tukey_window_impl(tau_pred__batch, alpha_pred__batch, config.fft_size)

    # Apply mask to both real and imag components
    h_est_delay__ri_batch_sc = h_delay__ri_batch_sc * mask__batch_sc[None, :, :]

    # -----------------------------------------------------------------------
    # FFT, convert to float16, and remove padding
    # -----------------------------------------------------------------------

    # Convert back to frequency domain using FFT-2048 (batched along axis 1)
    # Need to divide by config.fft_size to undo the IFFT scaling
    h_est__ri_batch_sc_real, h_est__ri_batch_sc_imag = fft_2048(
        h_est_delay__ri_batch_sc[0], h_est_delay__ri_batch_sc[1]
    )
    # Undo IFFT scaling and convert to float16
    h_est__ri_batch_sc_real = (h_est__ri_batch_sc_real / config.fft_size).astype(jnp.float16)
    h_est__ri_batch_sc_imag = (h_est__ri_batch_sc_imag / config.fft_size).astype(jnp.float16)

    h_est__ri_batch_sc = jnp.stack(
        [h_est__ri_batch_sc_real, h_est__ri_batch_sc_imag],
        axis=0,
    )

    # Extract original signal portion
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
