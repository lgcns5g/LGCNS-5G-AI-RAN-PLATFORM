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
End-to-end training for Tukey window predictor model.

Trains a transformer to predict optimal (tau, alpha) for Tukey window
by directly minimizing channel estimation MSE.
"""

import functools
import os
import shutil
from collections.abc import Callable
from pathlib import Path

# Limit JAX GPU memory pre-allocation to prevent OOM issues
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
import yaml
from flax import serialization
from flax.training import train_state
from safetensors.numpy import load_file
from tqdm import tqdm

from ran.phy.jax.pusch.channel_estimation import dmrs_matched_filter
from ran.phy.jax.pusch.delay_compensation import delay_compensate_complex

# Import DMRS and channel utilities
from ran.phy.jax.utils.awgn import awgn
from ran.trt_plugins.dmrs import (
    apply_dmrs_to_channel,
    dmrs_3276,
    extract_raw_dmrs_type1,
    gen_transmitted_dmrs_with_occ,
)

from ran.phy.jax.pusch.ai_tukey_filter import (
    create_model,
    count_parameters,
    TrainConfig,
    tukey_window_impl,
)


@functools.partial(
    jax.jit,
    static_argnames=[
        "dmrs_port_nums",
        "tau_max",
        "n_prb",
        "start_prb",
        "energy",
        "delay_compensation",
        "fft_size",
    ],
)
def _compute_features(
    rng: jax.Array,
    H__sc_sym_rxant: jax.Array,
    snr_db: float,
    x_dmrs__port_sym_dsc: jax.Array,
    dmrs_sc_idxs: jax.Array,
    dmrs_sym_idxs: jax.Array,
    dmrs_port_nums: tuple[int, ...],
    tau_max: int,
    n_prb: int,
    start_prb: int,
    energy: float,
    delay_compensation: float,
    fft_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Complete pipeline: apply DMRS, add noise, extract DMRS, compute features.

    Follows the exact flow from channel_estimation.py.

    Parameters
    ----------
    rng : jax.Array
        JAX PRNG key for noise generation.
    H__sc_sym_rxant : jax.Array
        Clean channel with shape (n_sc, n_sym, n_rxant), complex-valued.
    snr_db : float
        Signal-to-noise ratio in dB.
    x_dmrs__port_sym_dsc : jax.Array
        Transmitted DMRS signal with shape (n_port, n_dmrs_sym, n_dmrs_sc).
    dmrs_sc_idxs : jax.Array
        DMRS subcarrier indices.
    dmrs_sym_idxs : jax.Array
        DMRS symbol indices.
    dmrs_port_nums : tuple[int, ...]
        DMRS port numbers (compile-time constant).
    tau_max : int
        Maximum tau for feature computation (compile-time constant).
    n_prb : int
        Number of PRBs (compile-time constant).
    start_prb : int
        Starting PRB (compile-time constant).
    energy : float
        DMRS energy (compile-time constant).
    delay_compensation : float
        Delay compensation samples (compile-time constant).
    fft_size : int
        FFT size for delay domain processing (compile-time constant).

    Returns
    -------
    cumsum_power_norm__rxant_tau : jax.Array
        Normalized cumulative power [0, 1] with shape (n_rxant, tau_max).
    lambda_noise_db__rxant : jax.Array
        Noise estimate in dB with shape (n_rxant,).
    total_energy_db__rxant : jax.Array
        Total energy in dB with shape (n_rxant,).
    h_noisy__ri_rxant_dsc : jax.Array
        Noisy DMRS channel estimate with shape (2, n_rxant, n_dmrs_sc).
    h_clean__ri_rxant_dsc : jax.Array
        Clean DMRS channel with shape (2, n_rxant, n_dmrs_sc).
    """
    # Apply transmitted DMRS: y = h * x * sqrt(energy)
    h_tx__sc_sym_rxant = apply_dmrs_to_channel(
        H__sc_sym_rxant,
        x_dmrs__port_sym_dsc,
        dmrs_sc_idxs,
        dmrs_sym_idxs,
        energy,
    )

    # AWGN noise
    h_noisy__sc_sym_rxant = awgn(rng, h_tx__sc_sym_rxant, snr_db)

    # Convert to real format needed for TensorRT: (2, n_sym, n_rxant, n_sc)
    h_noisy__ri_sym_rxant_sc = jnp.stack(
        [
            h_noisy__sc_sym_rxant.real.transpose(1, 2, 0),
            h_noisy__sc_sym_rxant.imag.transpose(1, 2, 0),
        ],
        axis=0,
    )

    # Extract raw DMRS
    y_dmrs__ri_dsym_rxant_dsc, _, _ = extract_raw_dmrs_type1(
        xtf__ri_sym_rxant_sc=h_noisy__ri_sym_rxant_sc,
        dmrs_sym_idxs=dmrs_sym_idxs.astype(jnp.int32),
        n_prb=n_prb,
        start_prb=start_prb,
        dmrs_port_nums=dmrs_port_nums,
    )

    # Matched filter: h_ls = y * conj(x) / sqrt(energy)
    x_dmrs__ri_port_dsym_dsc = jnp.stack(
        [x_dmrs__port_sym_dsc.real, x_dmrs__port_sym_dsc.imag], axis=0
    )

    h_ls__ri_port_dsym_rxant_dsc = dmrs_matched_filter(
        y_dmrs__ri_dsym_rxant_dsc=y_dmrs__ri_dsym_rxant_dsc,
        x_dmrs__ri_port_dsym_dsc=x_dmrs__ri_port_dsym_dsc,
        energy=energy,
    )

    # Extract first DMRS symbol, first port, ALL antennas: (n_rxant, n_dmrs_sc) complex
    h_noisy__rxant_dsc = (
        h_ls__ri_port_dsym_rxant_dsc[0, 0, 0, :, :]
        + 1j * h_ls__ri_port_dsym_rxant_dsc[1, 0, 0, :, :]
    )

    # Clean reference (without DMRS, first DMRS symbol, ALL antennas)
    dmrs_sym = dmrs_sym_idxs[0]
    h_clean__dsc_rxant = H__sc_sym_rxant[dmrs_sc_idxs, dmrs_sym, :]
    h_clean__rxant_dsc = h_clean__dsc_rxant.T

    # Compute features from noisy DMRS estimate per antenna
    n_dmrs_sc = h_noisy__rxant_dsc.shape[1]

    # Apply delay compensation (CRITICAL: must match inference!)
    h_shifted__rxant_dsc = delay_compensate_complex(
        h_noisy__rxant_dsc, delay_compensation, forward=True
    )

    # Zero-pad and transform to delay domain (vectorized over rxant)
    n_fft = fft_size
    pad_length = n_fft - n_dmrs_sc
    h_padded__rxant_fft = jnp.pad(
        h_shifted__rxant_dsc,
        ((0, 0), (0, pad_length)),
        mode="constant",
        constant_values=0,
    )
    h_delay__rxant_tau = jnp.fft.ifft(h_padded__rxant_fft, axis=1) * n_fft
    h_power__rxant_tau = jnp.abs(h_delay__rxant_tau) ** 2

    # Estimate noise from tail (per antenna)
    noise_start = int(2 * n_fft / 3)
    lambda_noise__rxant = jnp.mean(h_power__rxant_tau[:, noise_start:], axis=1)

    # Compute cumulative power (per antenna)
    cumsum_power__rxant_tau = jnp.cumsum(h_power__rxant_tau[:, :tau_max], axis=1)
    total_energy__rxant = cumsum_power__rxant_tau[:, -1]
    cumsum_power_norm__rxant_tau = cumsum_power__rxant_tau / (total_energy__rxant[:, None] + 1e-10)

    # Convert to dB
    lambda_noise_db__rxant = 10.0 * jnp.log10(lambda_noise__rxant + 1e-10)
    total_energy_db__rxant = 10.0 * jnp.log10(total_energy__rxant + 1e-10)

    # Convert to real/imag format for output
    h_noisy__ri_rxant_dsc = jnp.stack([h_noisy__rxant_dsc.real, h_noisy__rxant_dsc.imag], axis=0)
    h_clean__ri_rxant_dsc = jnp.stack([h_clean__rxant_dsc.real, h_clean__rxant_dsc.imag], axis=0)

    return (
        cumsum_power_norm__rxant_tau,
        lambda_noise_db__rxant,
        total_energy_db__rxant,
        h_noisy__ri_rxant_dsc,
        h_clean__ri_rxant_dsc,
    )


@functools.partial(jax.jit, static_argnames=["delay_compensation", "fft_size"])
def _apply_tukey_denoising(
    h_noisy__dsc: jax.Array,
    tau_pred: float,
    alpha_pred: float,
    delay_compensation: float,
    fft_size: int,
) -> jax.Array:
    """Apply Tukey window denoising to DMRS channel estimate.

    Parameters
    ----------
    h_noisy__dsc : jax.Array
        Noisy DMRS channel estimate with shape (n_dmrs_sc,), complex-valued.
    tau_pred : float
        Predicted tau value.
    alpha_pred : float
        Predicted alpha value in range [0, 1].
    delay_compensation : float
        Delay compensation samples (compile-time constant).
    fft_size : int
        FFT size for delay domain processing (compile-time constant).

    Returns
    -------
    jax.Array
        Denoised DMRS channel with shape (n_dmrs_sc,), complex-valued.
    """
    n_dmrs_sc = h_noisy__dsc.shape[0]

    # Apply delay compensation
    h_shifted__dsc = delay_compensate_complex(h_noisy__dsc, delay_compensation, forward=True)

    # Zero-pad and transform to delay domain
    n_fft = fft_size
    pad_length = n_fft - n_dmrs_sc
    h_padded__fft = jnp.pad(h_shifted__dsc, ((0, pad_length),), mode="constant", constant_values=0)
    h_delay__tau = jnp.fft.ifft(h_padded__fft) * n_fft

    # Apply Tukey window using shared implementation
    # tukey_window_impl expects batch dimensions: (n_batch, 1) -> (n_batch, fft_size)
    tau__batch = jnp.array([[tau_pred]], dtype=jnp.float32)
    alpha__batch = jnp.array([[alpha_pred]], dtype=jnp.float32)
    window__batch_fft = tukey_window_impl(tau__batch, alpha__batch, fft_size=n_fft)
    window__fft = window__batch_fft[0]  # Extract single window from batch
    h_filtered__tau = h_delay__tau * window__fft

    # Transform back to frequency domain
    h_denoised__fft = jnp.fft.fft(h_filtered__tau) / n_fft
    h_denoised_shifted__dsc = h_denoised__fft[:n_dmrs_sc]

    # Undo delay compensation
    h_denoised__dsc = delay_compensate_complex(
        h_denoised_shifted__dsc, delay_compensation, forward=False
    )

    return h_denoised__dsc


def _save_config(output_dir: Path, cfg: TrainConfig) -> Path:
    """Save model and training configuration to YAML.

    Parameters
    ----------
    output_dir : Path
        Directory to save configuration file.
    cfg : TrainConfig
        Training configuration object.

    Returns
    -------
    Path
        Path to saved configuration file.
    """
    config_dict = {
        "model_config": {
            "compressed_len": cfg.model.compressed_len,
            "d_model": cfg.model.d_model,
            "n_heads": cfg.model.n_heads,
            "n_layers": cfg.model.n_layers,
            "max_tau": cfg.model.tau_max,
            "input_subsample_factor": cfg.model.input_subsample_factor,
        },
        "training_config": {
            "learning_rate": float(cfg.training.learning_rate),
            "batch_size": cfg.training.batch_size,
            "num_epochs": cfg.training.num_epochs,
            "warmup_epochs": cfg.training.warmup_epochs,
            "gradient_clip": float(cfg.training.gradient_clip),
            "delay_compensation_samples": float(cfg.model.delay_compensation_samples),
        },
        "data_config": {
            "n_prb": cfg.channel.n_prb,
            "n_sc": cfg.channel.n_sc,
            "n_dmrs_sc": cfg.channel.n_dmrs_sc,
            "dmrs_stride": cfg.dmrs_extraction.stride,
            "dmrs_offset": cfg.dmrs_extraction.offset,
        },
    }

    config_path = output_dir / "model_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return config_path


def _create_train_state(
    rng: jax.Array, learning_rate: float, cfg: TrainConfig
) -> train_state.TrainState:
    """Create initial training state.

    Parameters
    ----------
    rng : jax.Array
        JAX PRNG key for initialization.
    learning_rate : float
        Initial learning rate.
    cfg : TrainConfig
        Training configuration object.

    Returns
    -------
    train_state.TrainState
        Initialized training state with model and optimizer.
    """
    model = create_model(
        compressed_len=cfg.model.compressed_len,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout_rate=0.1,
        max_tau=cfg.model.tau_max,
        input_subsample_factor=cfg.model.input_subsample_factor,
    )

    # Initialize with dummy inputs
    dummy_cumsum_norm = jax.random.uniform(rng, (1, cfg.model.tau_max))
    dummy_noise_db = jax.random.uniform(rng, (1, 1)) * 40 - 20
    dummy_energy_db = jax.random.uniform(rng, (1, 1)) * 80 + 20

    variables = model.init(rng, dummy_cumsum_norm, dummy_noise_db, dummy_energy_db, training=False)

    # Create optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.training.gradient_clip),
        optax.adam(learning_rate),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def _get_learning_rate(epoch: int, base_lr: float, warmup_epochs: int) -> float:
    """Compute learning rate with linear warmup.

    Parameters
    ----------
    epoch : int
        Current epoch number (0-indexed).
    base_lr : float
        Base learning rate.
    warmup_epochs : int
        Number of warmup epochs.

    Returns
    -------
    float
        Learning rate for the current epoch.
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


@functools.partial(jax.jit, static_argnames=["delay_compensation", "fft_size"])
def _train_step(
    state: train_state.TrainState,
    cumsum_power_norm__batch: jax.Array,
    lambda_noise_db__batch: jax.Array,
    total_energy_db__batch: jax.Array,
    h_noisy__batch_ri_dsc: jax.Array,
    h_clean__batch_ri_dsc: jax.Array,
    rng: jax.Array,
    delay_compensation: float,
    fft_size: int,
) -> tuple[train_state.TrainState, dict]:
    """Single training step - predicts tau/alpha and applies denoising.

    Parameters
    ----------
    state : train_state.TrainState
        Current training state with model parameters.
    cumsum_power_norm__batch : jax.Array
        Normalized cumulative power with shape (batch_size, tau_max).
    lambda_noise_db__batch : jax.Array
        Noise estimate in dB with shape (batch_size, 1).
    total_energy_db__batch : jax.Array
        Total energy in dB with shape (batch_size, 1).
    h_noisy__batch_ri_dsc : jax.Array
        Noisy DMRS with shape (2, batch_size, n_dmrs_sc).
    h_clean__batch_ri_dsc : jax.Array
        Clean DMRS with shape (2, batch_size, n_dmrs_sc).
    rng : jax.Array
        JAX PRNG key for dropout.
    delay_compensation : float
        Delay compensation samples (compile-time constant).
    fft_size : int
        FFT size for delay domain processing (compile-time constant).

    Returns
    -------
    state : train_state.TrainState
        Updated training state.
    metrics : dict
        Dictionary of training metrics (loss, tau_mean, tau_std, etc.).
    """

    # Convert from (2, batch, n_dmrs_sc) real to (batch, n_dmrs_sc) complex
    h_noisy__batch_dsc = h_noisy__batch_ri_dsc[0] + 1j * h_noisy__batch_ri_dsc[1]
    h_clean__batch_dsc = h_clean__batch_ri_dsc[0] + 1j * h_clean__batch_ri_dsc[1]

    def loss_fn(params):  # type: ignore[no-untyped-def]
        # Predict tau and alpha from features
        tau_pred__batch, alpha_pred__batch = state.apply_fn(
            {"params": params},
            cumsum_power_norm__batch,
            lambda_noise_db__batch,
            total_energy_db__batch,
            training=True,
            rngs={"dropout": rng},
        )  # (batch, 1), (batch, 1)

        # Apply denoising with predicted parameters
        def denoise_and_compute_mse(  # type: ignore[no-untyped-def]
            tau_scalar, alpha_scalar, h_noisy__dsc, h_clean__dsc
        ):
            h_denoised__dsc = _apply_tukey_denoising(
                h_noisy__dsc,
                tau_scalar,
                alpha_scalar,
                delay_compensation=delay_compensation,
                fft_size=fft_size,
            )
            # Normalize MSE by signal power
            signal_power = jnp.mean(jnp.abs(h_clean__dsc) ** 2)
            mse = jnp.mean(jnp.abs(h_denoised__dsc - h_clean__dsc) ** 2) / signal_power
            return mse

        mse__batch = jax.vmap(denoise_and_compute_mse)(
            tau_pred__batch[:, 0], alpha_pred__batch[:, 0], h_noisy__batch_dsc, h_clean__batch_dsc
        )

        loss = jnp.mean(mse__batch)

        return loss, {
            "loss": loss,
            "tau_mean": jnp.mean(tau_pred__batch),
            "tau_std": jnp.std(tau_pred__batch),
            "alpha_mean": jnp.mean(alpha_pred__batch),
            "alpha_std": jnp.std(alpha_pred__batch),
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    # Add gradient norm
    grad_norm = optax.global_norm(grads)
    metrics["grad_norm"] = grad_norm

    state = state.apply_gradients(grads=grads)

    return state, metrics


@functools.partial(jax.jit, static_argnames=["n"])
def _shuffle_indices(rng: jax.Array, n: int) -> jax.Array:
    """Shuffle indices using JAX random permutation.

    Parameters
    ----------
    rng : jax.Array
        JAX PRNG key.
    n : int
        Number of indices to shuffle (compile-time constant).

    Returns
    -------
    jax.Array
        Shuffled indices with shape (n,).
    """
    return jax.random.permutation(rng, n)


@functools.partial(jax.jit, static_argnames=["delay_compensation", "fft_size"])
def _eval_step(
    state: train_state.TrainState,
    cumsum_power_norm__batch: jax.Array,
    lambda_noise_db__batch: jax.Array,
    total_energy_db__batch: jax.Array,
    h_noisy__batch_ri_dsc: jax.Array,
    h_clean__batch_ri_dsc: jax.Array,
    delay_compensation: float,
    fft_size: int,
) -> dict:
    """Single evaluation step - predicts tau/alpha and applies denoising.

    Parameters
    ----------
    state : train_state.TrainState
        Current training state with model parameters.
    cumsum_power_norm__batch : jax.Array
        Normalized cumulative power with shape (batch_size, tau_max).
    lambda_noise_db__batch : jax.Array
        Noise estimate in dB with shape (batch_size, 1).
    total_energy_db__batch : jax.Array
        Total energy in dB with shape (batch_size, 1).
    h_noisy__batch_ri_dsc : jax.Array
        Noisy DMRS with shape (2, batch_size, n_dmrs_sc).
    h_clean__batch_ri_dsc : jax.Array
        Clean DMRS with shape (2, batch_size, n_dmrs_sc).
    delay_compensation : float
        Delay compensation samples (compile-time constant).
    fft_size : int
        FFT size for delay domain processing (compile-time constant).

    Returns
    -------
    dict
        Dictionary of evaluation metrics (loss, tau_mean, tau_std, etc.).
    """

    # Convert from (2, batch, n_dmrs_sc) real to (batch, n_dmrs_sc) complex
    h_noisy__batch_dsc = h_noisy__batch_ri_dsc[0] + 1j * h_noisy__batch_ri_dsc[1]
    h_clean__batch_dsc = h_clean__batch_ri_dsc[0] + 1j * h_clean__batch_ri_dsc[1]

    # Predict tau and alpha
    tau_pred__batch, alpha_pred__batch = state.apply_fn(
        {"params": state.params},
        cumsum_power_norm__batch,
        lambda_noise_db__batch,
        total_energy_db__batch,
        training=False,
    )

    def denoise_and_compute_mse(  # type: ignore[no-untyped-def]
        tau_scalar, alpha_scalar, h_noisy__dsc, h_clean__dsc
    ):
        h_denoised__dsc = _apply_tukey_denoising(
            h_noisy__dsc,
            tau_scalar,
            alpha_scalar,
            delay_compensation=delay_compensation,
            fft_size=fft_size,
        )
        # Normalize MSE by signal power
        signal_power = jnp.mean(jnp.abs(h_clean__dsc) ** 2)
        mse = jnp.mean(jnp.abs(h_denoised__dsc - h_clean__dsc) ** 2) / signal_power
        return mse

    mse__batch = jax.vmap(denoise_and_compute_mse)(
        tau_pred__batch[:, 0], alpha_pred__batch[:, 0], h_noisy__batch_dsc, h_clean__batch_dsc
    )

    return {
        "loss": jnp.mean(mse__batch),
        "tau_mean": jnp.mean(tau_pred__batch),
        "tau_std": jnp.std(tau_pred__batch),
        "alpha_mean": jnp.mean(alpha_pred__batch),
        "alpha_std": jnp.std(alpha_pred__batch),
    }


def load_checkpoint(
    checkpoint_dir: Path,
    prefix: str = "model_params_epoch_",
    cfg: TrainConfig | None = None,
) -> train_state.TrainState:
    """Load trained model parameters using FLAX serialization.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoints.
    prefix : str, optional
        Checkpoint name prefix, by default "model_params_epoch_".
    cfg : TrainConfig | None, optional
        Training configuration (required for creating state structure).

    Returns
    -------
    train_state.TrainState
        Training state with loaded parameters.

    Raises
    ------
    FileNotFoundError
        If checkpoint directory or files not found.
    ValueError
        If cfg parameter is not provided.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find best checkpoint by looking for metadata
    best_checkpoint = None
    best_val_loss = float("inf")

    for ckpt_path in checkpoint_dir.glob(f"{prefix}*.flax"):
        # Look for metadata file
        metadata_path = ckpt_path.with_suffix("").with_name(f"{ckpt_path.stem}_metadata.yaml")
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
                val_loss = metadata.get("val_loss", float("inf"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = ckpt_path

    if best_checkpoint is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Load config if not provided
    if cfg is None:
        config_path = checkpoint_dir / "model_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        raise ValueError(
            "cfg parameter required for loading checkpoint. "
            "Load TrainConfig from checkpoint directory first."
        )

    # Create empty state with same structure
    rng = jax.random.PRNGKey(0)
    empty_state = _create_train_state(rng, cfg.training.learning_rate, cfg)

    # Load params using FLAX serialization
    with open(best_checkpoint, "rb") as f:
        bytes_data = f.read()
    restored_params = serialization.from_bytes(empty_state.params, bytes_data)

    # Create new state with restored parameters
    restored_state = empty_state.replace(params=restored_params)

    print(f"Loaded checkpoint from {best_checkpoint}")
    return restored_state


def _load_channel_data(cfg: TrainConfig) -> tuple[dict, dict]:
    """Load clean channel data from safetensors.

    DMRS transmission and noise are applied on-the-fly during training.

    Parameters
    ----------
    cfg : TrainConfig
        Training configuration.

    Returns
    -------
    train_data : dict
        Training dataset dictionary.
    val_data : dict
        Validation dataset dictionary.

    Raises
    ------
    FileNotFoundError
        If dataset files not found.
    """
    dataset_dir = cfg.paths.dataset_path
    train_path = dataset_dir / "train_data.safetensors"
    val_path = dataset_dir / "val_data.safetensors"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Dataset not found in {dataset_dir}.")

    def load_and_reconstruct(path):  # type: ignore[no-untyped-def]
        """Load from safetensors and reconstruct complex arrays."""
        data_raw = load_file(str(path))
        data = {}

        # Reconstruct complex arrays from real/imag components
        processed_keys = set()
        for key in data_raw.keys():
            if key.endswith(".real"):
                base_key = key[:-5]
                if f"{base_key}.imag" in data_raw:
                    data[base_key] = data_raw[key] + 1j * data_raw[f"{base_key}.imag"]
                    processed_keys.add(key)
                    processed_keys.add(f"{base_key}.imag")
            elif key not in processed_keys:
                data[key] = data_raw[key]

        return data

    train_data = load_and_reconstruct(train_path)
    val_data = load_and_reconstruct(val_path)

    return train_data, val_data


def train_ai_tukey_filter_model(
    verbose: bool = True,
    config_path: Path | None = None,
    config: TrainConfig | None = None,
    epoch_callback: Callable | None = None,
) -> dict:
    """Train AI Tukey filter model.

    Parameters
    ----------
    verbose : bool, optional
        If True, print progress information, by default True.
    config_path : Path | None, optional
        Path to training configuration YAML file. If None and config is None,
        uses default config in the same directory as this script.
        Mutually exclusive with config parameter.
    config : TrainConfig | None, optional
        Training configuration object. If provided, config_path is ignored.
        Allows for programmatic configuration from notebooks.
    epoch_callback : callable | None, optional
        Optional callback function called after each epoch with signature:
        callback(epoch, train_loss, val_loss, history). If provided, allows
        for custom actions like plotting or logging after each epoch.

    Returns
    -------
    dict
        Dictionary containing training history, best validation loss,
        best epoch, number of parameters, and output directories.

    Raises
    ------
    ValueError
        If both config_path and config are provided.
    """
    if verbose:
        print("AI Tukey Filter Training")
        print("-" * 40)

    # Load configuration
    if config is not None:
        if config_path is not None:
            raise ValueError("Cannot specify both config_path and config parameters")
        cfg = config
    else:
        if config_path is None:
            config_path = Path(__file__).parent / "train_ai_tukey_filter_model_config.yaml"
        cfg = TrainConfig.from_yaml(config_path)

    # Load dataset configuration (single source of truth)
    dataset_config_path = cfg.paths.dataset_path / "dataset_config.yaml"
    if not dataset_config_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_config_path}\n"
            f"Run ai_tukey_filter_cdl_training_dataset.py first"
        )

    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)

    cfg.channel.n_prb = dataset_config["num_prb"]

    if verbose:
        print(f"Dataset: {cfg.paths.dataset_path}")
        print(f"PRBs: {cfg.channel.n_prb} → {cfg.channel.n_dmrs_sc} DMRS subcarriers")

    # Pre-compute DMRS
    r_dmrs__ri_sym_cdm_dsc, _ = dmrs_3276(
        slot_number=cfg.dmrs_config.slot_number,
        n_dmrs_id=cfg.dmrs_config.n_dmrs_id,
    )

    port_idx_array = np.array(cfg.dmrs_config.port_idx, dtype=np.int32)
    scids_tuple = tuple(int(s) for s in cfg.dmrs_config.vec_scid)

    x_dmrs__ri_port_sym_sc = gen_transmitted_dmrs_with_occ(
        r_dmrs__ri_sym_cdm_dsc=r_dmrs__ri_sym_cdm_dsc,
        dmrs_port_nums=jnp.asarray(port_idx_array),
        scids=scids_tuple,
        dmrs_sym_idxs=tuple(cfg.dmrs_config.dmrs_idx),
        n_dmrs_sc=cfg.channel.n_dmrs_sc,
    )
    x_dmrs__port_sym_sc = x_dmrs__ri_port_sym_sc[0] + 1j * x_dmrs__ri_port_sym_sc[1]
    x_dmrs__port_sym_dsc = np.ascontiguousarray(x_dmrs__port_sym_sc, dtype=np.complex64)

    grid_cfg = (cfg.dmrs_config.port_idx[0] & 0b010) >> 1
    dmrs_base = 12 * cfg.dmrs_config.start_prb
    dmrs_sc_idxs: npt.NDArray[np.int_] = dmrs_base + 2 * np.arange(cfg.channel.n_dmrs_sc) + grid_cfg

    # Convert to JAX arrays
    x_dmrs__port_sym_dsc_jax = jnp.array(x_dmrs__port_sym_dsc, dtype=jnp.complex64)
    dmrs_sc_idxs_jax = jnp.array(dmrs_sc_idxs, dtype=jnp.int32)
    dmrs_sym_idxs_jax = jnp.array(cfg.dmrs_config.dmrs_idx, dtype=jnp.int32)
    dmrs_port_nums_tuple = tuple(cfg.dmrs_config.port_idx)

    rng = jax.random.PRNGKey(cfg.training.seed)

    # Load data
    train_data, val_data = _load_channel_data(cfg)
    n_train = len(train_data["H__sc_sym_rxant"])
    n_val = len(val_data["H__sc_sym_rxant"])

    if verbose:
        print(f"Train: {n_train}, Val: {n_val}")
        print(f"SNR range: [{cfg.training.snr_min_db}, {cfg.training.snr_max_db}] dB")

    # Generate random SNRs for each sample
    rng_train = np.random.default_rng(cfg.training.seed)
    rng_val = np.random.default_rng(cfg.training.seed + 1)
    train_data["snrs"] = rng_train.uniform(
        cfg.training.snr_min_db, cfg.training.snr_max_db, n_train
    )
    val_data["snrs"] = rng_val.uniform(cfg.training.snr_min_db, cfg.training.snr_max_db, n_val)

    # Create model
    rng, init_rng = jax.random.split(rng)
    state = _create_train_state(init_rng, cfg.training.learning_rate, cfg)
    n_params = count_parameters(state.params)

    if verbose:
        print(f"Model: {n_params:,} params (~{n_params * 4 / 1024:.1f} KB)")
        print(f"Output: {cfg.paths.output_path}")

    # Save configuration
    cfg.paths.output_path.mkdir(parents=True, exist_ok=True)

    # Create checkpoint directory
    checkpoint_dir = cfg.paths.checkpoint_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_config(checkpoint_dir, cfg)

    history: dict = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_paths = []  # Track checkpoint files for cleanup

    for epoch in range(1, cfg.training.num_epochs + 1):
        current_lr = _get_learning_rate(
            epoch - 1, cfg.training.learning_rate, cfg.training.warmup_epochs
        )
        if epoch <= cfg.training.warmup_epochs:
            new_tx = optax.chain(
                optax.clip_by_global_norm(cfg.training.gradient_clip),
                optax.adam(current_lr),
            )
            state = state.replace(tx=new_tx, opt_state=new_tx.init(state.params))

        # Shuffle train
        rng, shuffle_rng = jax.random.split(rng)
        indices = _shuffle_indices(shuffle_rng, n_train)

        train_metrics = []
        n_batches_train = (n_train + cfg.training.batch_size - 1) // cfg.training.batch_size
        for i in tqdm(
            range(0, n_train, cfg.training.batch_size),
            desc=f"Epoch {epoch}/{cfg.training.num_epochs} [Train]",
            total=n_batches_train,
            unit="batch",
            disable=not verbose,
        ):
            batch_idx = indices[i : min(i + cfg.training.batch_size, n_train)]

            # Compute features on-the-fly for this batch
            batch_cumsum_list = []
            batch_noise_db_list = []
            batch_energy_db_list = []
            batch_h_noisy_list = []
            batch_h_clean_list = []

            for idx in batch_idx:
                rng, sample_rng = jax.random.split(rng)
                idx_int = int(idx)  # Convert JAX scalar to Python int
                H_jax = jnp.array(train_data["H__sc_sym_rxant"][idx_int])
                (
                    cumsum__rxant_tau,
                    noise_db__rxant,
                    energy_db__rxant,
                    h_noisy__ri_rxant_dsc,
                    h_clean__ri_rxant_dsc,
                ) = _compute_features(
                    sample_rng,
                    H_jax,
                    train_data["snrs"][idx_int],
                    x_dmrs__port_sym_dsc_jax,
                    dmrs_sc_idxs_jax,
                    dmrs_sym_idxs_jax,
                    dmrs_port_nums_tuple,
                    tau_max=cfg.model.tau_max,
                    n_prb=cfg.channel.n_prb,
                    start_prb=cfg.dmrs_config.start_prb,
                    energy=cfg.dmrs_config.energy,
                    delay_compensation=cfg.model.delay_compensation_samples,
                    fft_size=cfg.model.fft_size,
                )
                # Flatten antenna dimension into lists
                n_rxant = cumsum__rxant_tau.shape[0]
                for rxant in range(n_rxant):
                    batch_cumsum_list.append(cumsum__rxant_tau[rxant])
                    batch_noise_db_list.append(noise_db__rxant[rxant])
                    batch_energy_db_list.append(energy_db__rxant[rxant])
                    batch_h_noisy_list.append(h_noisy__ri_rxant_dsc[:, rxant, :])
                    batch_h_clean_list.append(h_clean__ri_rxant_dsc[:, rxant, :])

            # Stack into batch arrays with antenna dim absorbed into batch
            cumsum_batch = jnp.stack(batch_cumsum_list)
            noise_db_batch = jnp.array(batch_noise_db_list)[:, None]
            energy_db_batch = jnp.array(batch_energy_db_list)[:, None]
            h_noisy__batch_ri_dsc = jnp.stack(
                batch_h_noisy_list, axis=1
            )  # (2, batch_size * n_rxant, n_dmrs_sc)
            h_clean__batch_ri_dsc = jnp.stack(
                batch_h_clean_list, axis=1
            )  # (2, batch_size * n_rxant, n_dmrs_sc)

            rng, step_rng = jax.random.split(rng)
            state, metrics = _train_step(
                state,
                cumsum_batch,
                noise_db_batch,
                energy_db_batch,
                h_noisy__batch_ri_dsc,
                h_clean__batch_ri_dsc,
                step_rng,
                delay_compensation=cfg.model.delay_compensation_samples,
                fft_size=cfg.model.fft_size,
            )
            train_metrics.append(metrics)

        train_loss = float(np.mean([m["loss"] for m in train_metrics]))

        # Validate
        val_metrics = []
        n_batches_val = (n_val + cfg.training.batch_size - 1) // cfg.training.batch_size
        for i in tqdm(
            range(0, n_val, cfg.training.batch_size),
            desc=f"Epoch {epoch}/{cfg.training.num_epochs} [Val]",
            total=n_batches_val,
            unit="batch",
            disable=not verbose,
        ):
            batch_end = min(i + cfg.training.batch_size, n_val)

            # Compute features on-the-fly for validation batch
            batch_cumsum_list = []
            batch_noise_db_list = []
            batch_energy_db_list = []
            batch_h_noisy_list = []
            batch_h_clean_list = []

            for idx in range(i, batch_end):
                rng, sample_rng = jax.random.split(rng)
                H_jax = jnp.array(val_data["H__sc_sym_rxant"][idx])
                (
                    cumsum__rxant_tau,
                    noise_db__rxant,
                    energy_db__rxant,
                    h_noisy__ri_rxant_dsc,
                    h_clean__ri_rxant_dsc,
                ) = _compute_features(
                    sample_rng,
                    H_jax,
                    val_data["snrs"][idx],
                    x_dmrs__port_sym_dsc_jax,
                    dmrs_sc_idxs_jax,
                    dmrs_sym_idxs_jax,
                    dmrs_port_nums_tuple,
                    tau_max=cfg.model.tau_max,
                    n_prb=cfg.channel.n_prb,
                    start_prb=cfg.dmrs_config.start_prb,
                    energy=cfg.dmrs_config.energy,
                    delay_compensation=cfg.model.delay_compensation_samples,
                    fft_size=cfg.model.fft_size,
                )
                # Flatten antenna dimension into lists
                n_rxant = cumsum__rxant_tau.shape[0]
                for rxant in range(n_rxant):
                    batch_cumsum_list.append(cumsum__rxant_tau[rxant])
                    batch_noise_db_list.append(noise_db__rxant[rxant])
                    batch_energy_db_list.append(energy_db__rxant[rxant])
                    batch_h_noisy_list.append(h_noisy__ri_rxant_dsc[:, rxant, :])
                    batch_h_clean_list.append(h_clean__ri_rxant_dsc[:, rxant, :])

            # Stack into batch arrays with antenna dim absorbed into batch
            cumsum_batch = jnp.stack(batch_cumsum_list)
            noise_db_batch = jnp.array(batch_noise_db_list)[:, None]
            energy_db_batch = jnp.array(batch_energy_db_list)[:, None]
            h_noisy__batch_ri_dsc = jnp.stack(batch_h_noisy_list, axis=1)
            h_clean__batch_ri_dsc = jnp.stack(batch_h_clean_list, axis=1)

            metrics = _eval_step(
                state,
                cumsum_batch,
                noise_db_batch,
                energy_db_batch,
                h_noisy__batch_ri_dsc,
                h_clean__batch_ri_dsc,
                delay_compensation=cfg.model.delay_compensation_samples,
                fft_size=cfg.model.fft_size,
            )
            val_metrics.append(metrics)

        val_loss = float(np.mean([m["loss"] for m in val_metrics]))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch}/{cfg.training.num_epochs}: "
                f"train={train_loss:.4f}, val={val_loss:.4f}"
            )

        # Call epoch callback if provided
        if epoch_callback is not None:
            epoch_callback(epoch, train_loss, val_loss, history)

        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Save params using FLAX bytes serialization
            checkpoint_path = checkpoint_dir / f"model_params_epoch_{epoch:03d}.flax"
            with open(checkpoint_path, "wb") as f:
                f.write(serialization.to_bytes(state.params))

            # Save metadata
            metadata = {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "train_loss": float(train_loss),
            }
            metadata_path = checkpoint_dir / f"model_params_epoch_{epoch:03d}_metadata.yaml"
            with open(metadata_path, "w", encoding="utf-8") as f:
                yaml.dump(metadata, f)

            checkpoint_paths.append((epoch, val_loss, checkpoint_path))

            # Keep only best 3 checkpoints
            if len(checkpoint_paths) > 3:
                checkpoint_paths.sort(key=lambda x: x[1])  # Sort by val_loss
                _, _, old_path = checkpoint_paths.pop()  # Remove worst checkpoint
                if old_path.exists():
                    old_path.unlink()
                    # Also delete the metadata file
                    old_metadata = old_path.parent / f"{old_path.stem}_metadata.yaml"
                    if old_metadata.exists():
                        old_metadata.unlink()

            if verbose:
                print(f"  → Saved best model (epoch {epoch})")

    if verbose:
        print(f"Training complete: best_val_loss={best_val_loss:.4f}")
        print(f"Best checkpoint: epoch {best_epoch}")

    # Copy best model to standard filename for easy loading
    best_checkpoint_path = checkpoint_dir / f"model_params_epoch_{best_epoch:03d}.flax"
    standard_model_path = checkpoint_dir / "model_params.flax"

    if best_checkpoint_path.exists():
        shutil.copy2(best_checkpoint_path, standard_model_path)
        if verbose:
            print(f"Copied best model to: {standard_model_path.name}")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "output_dir": cfg.paths.output_path,
        "checkpoint_dir": checkpoint_dir,
    }


if __name__ == "__main__":
    train_ai_tukey_filter_model()
