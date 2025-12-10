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
Lightweight transformer for predicting Tukey window parameters (tau, alpha).

The model takes 3 features per tau position:
1. cumsum_power_norm (0-1 CDF): energy accumulation shape
2. lambda_noise_db (dB): noise level estimate
3. total_energy_db (dB): absolute scale reference

Outputs:
- tau: window length [1, max_tau]
- alpha: Tukey taper parameter [0, 1] (0=rectangular, 1=Hann)
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class TukeyPredictorModel(nn.Module):
    """
    Predicts Tukey window params (tau, alpha) from cumulative power curve + noise info.

    Architecture:
        1. Subsample cumulative power curve for efficiency
        2. Broadcast 3 features to subsampled sequence length
        3. Add positional encoding
        4. Transformer layers to analyze curve shape
        5. Global pooling + dual regression heads for tau and alpha

    Attributes:
        compressed_len: Downsampled sequence length (default: 64)
        d_model: Model embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout_rate: Dropout rate for training
        max_tau: Maximum tau value (for output scaling)
        input_subsample_factor: Factor to subsample cumsum_power_norm (1=no subsample,
        32=every 32nd, 64=every 64th)
    """

    compressed_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout_rate: float = 0.1
    max_tau: int = 1024
    input_subsample_factor: int = 1

    def setup(self) -> None:
        """Initialize model layers."""
        # Project 3 input features to d_model
        self.input_projection = nn.Dense(self.d_model)

        # Positional encoding
        self.pos_encoding = self.param(
            "pos_encoding", nn.initializers.normal(stddev=0.02), (self.compressed_len, self.d_model)
        )

        # Transformer encoder layers
        self.transformer_layers = [
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.n_layers)
        ]

        # Regression head for tau (bias initialized to predict tau≈100)
        self.tau_head = nn.Sequential(
            [
                nn.Dense(64),
                nn.relu,
                nn.Dense(32),
                nn.relu,
                nn.Dense(1, bias_init=nn.initializers.constant(-2.2)),
            ]
        )

        # Regression head for alpha (bias initialized to predict alpha≈0.3)
        self.alpha_head = nn.Sequential(
            [
                nn.Dense(64),
                nn.relu,
                nn.Dense(32),
                nn.relu,
                nn.Dense(1, bias_init=nn.initializers.constant(-0.85)),
            ]
        )

    def __call__(
        self,
        cumsum_power_norm__batch_tau: jax.Array,  # (batch, tau_max) - CDF [0, 1]
        lambda_noise_db__batch: jax.Array,  # (batch, 1) - Noise in dB
        total_energy_db__batch: jax.Array,  # (batch, 1) - Total energy in dB
        training: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Predict Tukey window parameters from energy curve, noise level, and scale.

        Args:
            cumsum_power_norm__batch_tau: Normalized cumulative power [0, 1] (batch, tau_max)
            lambda_noise_db__batch: Noise estimate in dB (batch, 1)
            total_energy_db__batch: Total energy in dB (batch, 1)
            training: Whether in training mode

        Returns:
            tau_pred: Predicted tau values (batch, 1), range [1, max_tau]
            alpha_pred: Predicted alpha values (batch, 1), range [0, 1]
        """
        batch_size = cumsum_power_norm__batch_tau.shape[0]

        # Subsample cumulative power curve for efficiency
        # Take every Nth sample (e.g., [0, 32, 64, ...] for factor=32)
        if self.input_subsample_factor > 1:
            cumsum_power_subsampled = cumsum_power_norm__batch_tau[
                :, :: self.input_subsample_factor
            ]
        else:
            cumsum_power_subsampled = cumsum_power_norm__batch_tau

        seq_len = cumsum_power_subsampled.shape[1]

        # Stack 3 features: [CDF, noise_dB, energy_dB]
        # Broadcast scalars to subsampled sequence length
        noise_db_broadcast = jnp.broadcast_to(
            lambda_noise_db__batch[:, :, None], (batch_size, seq_len, 1)
        )
        energy_db_broadcast = jnp.broadcast_to(
            total_energy_db__batch[:, :, None], (batch_size, seq_len, 1)
        )

        # Concatenate: (batch, seq_len, 3)
        x = jnp.concatenate(
            [cumsum_power_subsampled[:, :, None], noise_db_broadcast, energy_db_broadcast], axis=-1
        )

        # Project to d_model: (batch, seq_len, 3) → (batch, seq_len, d_model)
        x = self.input_projection(x)

        # Downsample to compressed_len if needed: (batch, seq_len, d_model) → (batch, compressed_len, d_model)
        if seq_len > self.compressed_len:
            # Average pool with stride (seq_len / compressed_len)
            pool_size = seq_len // self.compressed_len
            # Truncate to make sequence evenly divisible
            truncated_len = self.compressed_len * pool_size
            x = x[:, :truncated_len, :]
            x = x.reshape(batch_size, self.compressed_len, pool_size, self.d_model)
            x = jnp.mean(x, axis=2)
        elif seq_len < self.compressed_len:
            # Pad if subsampled sequence is shorter than compressed_len
            pad_len = self.compressed_len - seq_len
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)), mode="edge")
        # else: seq_len == compressed_len, no change needed

        # Add positional encoding
        x = x + self.pos_encoding[None, :, :]

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Global average pooling
        x = jnp.mean(x, axis=1)

        # Dual regression heads
        tau_pred_raw = self.tau_head(x)
        alpha_pred_raw = self.alpha_head(x)

        # Sigmoid scaled to [1, max_tau] for tau
        tau_pred = 1 + (self.max_tau - 1) * jax.nn.sigmoid(tau_pred_raw)

        # Sigmoid scaled to [0, 1] for alpha
        alpha_pred = jax.nn.sigmoid(alpha_pred_raw)

        return tau_pred, alpha_pred


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        # Multi-head self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, x)

        # Add & Norm
        x = nn.LayerNorm()(x + attn_out)

        # Feed-forward
        ff_out = nn.Sequential(
            [
                nn.Dense(self.d_model * 4),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate, deterministic=not training),
                nn.Dense(self.d_model),
                nn.Dropout(rate=self.dropout_rate, deterministic=not training),
            ]
        )(x)

        # Add & Norm
        x = nn.LayerNorm()(x + ff_out)

        return x


def create_model(
    compressed_len: int = 64,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout_rate: float = 0.1,
    max_tau: int = 1024,
    input_subsample_factor: int = 1,
) -> TukeyPredictorModel:
    """
    Create Tukey predictor model.

    Args:
        compressed_len: Final sequence length after pooling
        d_model: Model embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout_rate: Dropout rate for training
        max_tau: Maximum tau value
        input_subsample_factor: Factor to subsample cumsum_power_norm (1=no subsample,
        32=every 32nd, 64=every 64th)

    Returns:
        TukeyPredictorModel instance
    """
    return TukeyPredictorModel(
        compressed_len=compressed_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        max_tau=max_tau,
        input_subsample_factor=input_subsample_factor,
    )


def count_parameters(params: Any) -> int:
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
