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

"""Dataset generation using Sionna CDL models.

This module synthesizes frequency-domain channel responses with 3GPP CDL
(Clustered Delay Line) models via Sionna and saves them directly to safetensors format.

Outputs:
  - train_data.safetensors: Training channels
  - val_data.safetensors: Validation channels
  - test_data.safetensors: Test channels
  - Each contains H__sc_sym_rxant: complex64 array of shape (N, num_sc, 14, 4)

The CDL model supports configurable antenna arrays for both BS (Base Station)
and UE (User Equipment) with customizable polarization and antenna patterns.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from safetensors.numpy import save_file
from tqdm import tqdm

from .sionna_cdl_config import SionnaCDLConfig

logger = logging.getLogger(__name__)

# Lazy import of Sionna to avoid TensorFlow loading until needed
if TYPE_CHECKING:
    pass


def _generate_channels(
    gen: Any,  # phy.channel.generate_ofdm_channel.GenerateOFDMChannel
    total: int,
    config: SionnaCDLConfig,
) -> np.ndarray:
    """Generate channel realizations from Sionna.

    Args:
        gen: Sionna channel generator
        total: Total number of channels to generate
        config: Configuration object

    Returns:
        Array of shape (total, num_sc, 14, 4) with complex64 dtype
    """
    H: np.ndarray = np.zeros((total, config.num_sc, 14, 4), dtype=np.complex64)
    filled = 0

    with tqdm(total=total, desc="Generating", unit="channels") as pbar:
        while filled < total:
            batch_size = min(config.batch_tf, total - filled)

            # Generate batch: [B, num_tx_ant, num_rx_ant, num_streams_per_tx,
            #                  num_streams_per_rx, num_symbols, num_sc]
            H_tf = gen(batch_size)
            H_np = H_tf.numpy()  # Shape: [B, 1, 4, 1, 1, 14, num_sc]

            for k in range(batch_size):
                # Extract: [1, 4, 1, 1, 14, num_sc] → [4, 14, num_sc]
                h_channel = H_np[k, 0, :, 0, 0, :, : config.num_sc]

                # Transpose to (num_sc, 14, 4)
                h_channel = h_channel.transpose(2, 1, 0)

                H[filled] = h_channel
                filled += 1
                pbar.update(1)

                if filled >= total:
                    break

    return H


def _save_to_safetensors(H: np.ndarray, path: Path) -> None:
    """Save channel data to safetensors format.

    Handles complex arrays by saving real and imaginary parts separately.

    Args:
        H: Channel array of shape (N, num_sc, 14, 4) with complex64 dtype
        path: Output file path
    """
    tensors: dict[str, np.ndarray] = {
        "H__sc_sym_rxant.real": H.real.astype(np.float32),
        "H__sc_sym_rxant.imag": H.imag.astype(np.float32),
    }
    save_file(tensors, str(path))

    # Log size info
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("  %s: %d channels, %.1f MB", path.name, H.shape[0], size_mb)


def gen_cdl_channels(
    config: SionnaCDLConfig,
    validation_frac: float = 0.15,
    out_dir: str = "/opt/nvidia/aerial-framework/out/sionna_dataset",
) -> tuple[int, int, int]:
    """Generate CDL channel dataset using Sionna and save to safetensors.

    This function generates channel realizations, splits them into train/val/test,
    and saves directly to safetensors format, eliminating intermediate files.

    Args:
        config: Configuration object containing all channel generation parameters.
        validation_frac: Fraction of training data for validation.
        out_dir: Output directory for generated files.

    Returns:
        Tuple of (num_train, num_val, num_test) samples.
    """
    # Lazy import Sionna/TensorFlow only when actually generating
    from sionna import phy  # type: ignore[import-untyped]

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Setup Sionna components
    rg = phy.ofdm.ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=config.num_sc,
        subcarrier_spacing=30e3,
        num_guard_carriers=(0, 0),
        dc_null=False,
        pilot_pattern="empty",
        num_tx=1,
        num_streams_per_tx=1,
    )

    # Create antenna arrays
    bs_array = phy.channel.tr38901.antenna.AntennaArray(
        num_rows=config.bs_num_rows,
        num_cols=config.bs_num_cols,
        polarization=config.bs_polarization,
        polarization_type=config.bs_polarization_type,
        antenna_pattern=config.bs_antenna_pattern,
        carrier_frequency=config.fc_ghz * 1e9,
    )

    ue_array = phy.channel.tr38901.antenna.AntennaArray(
        num_rows=config.ue_num_rows,
        num_cols=config.ue_num_cols,
        polarization=config.ue_polarization,
        polarization_type=config.ue_polarization_type,
        antenna_pattern=config.ue_antenna_pattern,
        carrier_frequency=config.fc_ghz * 1e9,
    )

    # Create CDL channel model
    channel_model = phy.channel.tr38901.cdl.CDL(
        model=config.tdl_model,
        delay_spread=config.delay_spread_ns * 1e-9,
        carrier_frequency=config.fc_ghz * 1e9,
        ut_array=ue_array,
        bs_array=bs_array,
        direction=config.direction,
        min_speed=config.speed_min,
        max_speed=config.speed_max,
    )

    gen = phy.channel.generate_ofdm_channel.GenerateOFDMChannel(
        channel_model=channel_model, resource_grid=rg
    )

    # Generate training data
    logger.info("Generating %d training channels...", config.train_total)
    H_train_all = _generate_channels(gen, config.train_total, config)

    # Generate test data
    logger.info("Generating %d test channels...", config.test_total)
    H_test = _generate_channels(gen, config.test_total, config)

    # Split training data into train/val
    if config.train_total < 2:
        raise ValueError(
            f"train_total must be at least 2 to allow train/val split, got {config.train_total}"
        )

    rng = np.random.default_rng(config.prng_seed)
    n_val = max(1, int(validation_frac * config.train_total))
    # Ensure at least 1 training sample remains
    n_val = min(n_val, config.train_total - 1)
    n_train = config.train_total - n_val

    indices = rng.permutation(config.train_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    H_train = H_train_all[train_indices]
    H_val = H_train_all[val_indices]

    logger.info("  Split: %d train, %d val, %d test", n_train, n_val, config.test_total)

    # Save to safetensors
    logger.info("Saving datasets to safetensors...")
    _save_to_safetensors(H_train, out_path / "train_data.safetensors")
    _save_to_safetensors(H_val, out_path / "val_data.safetensors")
    _save_to_safetensors(H_test, out_path / "test_data.safetensors")

    logger.info("Datasets saved to %s", out_dir)

    return n_train, n_val, config.test_total
