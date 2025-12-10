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

"""Dataset class for Sionna channel data."""

import glob
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from safetensors.numpy import load_file
from tqdm import tqdm

from .sionna_cdl_config import SionnaCDLConfig

# Lazy import of Sionna to avoid TensorFlow loading when only reading datasets
if TYPE_CHECKING:
    pass


class SionnaChannelDataset:
    """Loads true channel (H) from `.npz` or `.safetensors` files."""

    def __init__(self, paths: list[str], num_sc: int, num_symbols: int = 14, num_rx: int = 4):
        """Initialize dataset from shard files.

        Args:
            paths: List of paths to .npz or .safetensors files.
            num_sc: Number of subcarriers.
            num_symbols: Number of OFDM symbols.
            num_rx: Number of RX antennas.
        """
        self.samples: list[np.ndarray] = []
        self.num_sc = num_sc
        self.num_symbols = num_symbols
        self.num_rx = num_rx
        self.expected_shape = (num_sc, num_symbols, num_rx)

        for p in sorted(paths):
            if p.endswith(".safetensors"):
                # Load from safetensors format
                data_raw = load_file(p)

                # Reconstruct complex array from real/imag components
                h_key = "H__sc_sym_rxant"
                if f"{h_key}.real" in data_raw and f"{h_key}.imag" in data_raw:
                    H = data_raw[f"{h_key}.real"] + 1j * data_raw[f"{h_key}.imag"]
                else:
                    error_msg = f"Expected keys '{h_key}.real' and '{h_key}.imag' in {p}"
                    raise KeyError(error_msg)
            else:
                # Load from npz format (legacy)
                with np.load(p) as z:
                    H = z["H"]  # True channel

            expected_shape = (H.shape[0], num_sc, num_symbols, num_rx)
            if H.shape != expected_shape:
                error_msg = f"Expected {expected_shape}, got {H.shape}"
                raise ValueError(error_msg)
            for i in range(H.shape[0]):
                # Channels are in correct 3D format: (num_sc, 14, 4)
                h_channel = H[i].copy()  # Shape: (num_sc, 14, 4)
                self.samples.append(h_channel)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.samples[i]

    def get_batch(self, indices: list[int]) -> np.ndarray:
        """Get a batch of samples as NumPy arrays.

        Args:
            indices: List of sample indices.

        Returns
        -------
            NumPy array of shape (batch_size, num_sc, num_symbols, num_rx).
        """
        batch_h = []
        for idx in indices:
            batch_h.append(self.samples[idx])
        return np.array(batch_h)

    @classmethod
    def from_samples(
        cls, samples: list[np.ndarray], num_sc: int, num_symbols: int = 14, num_rx: int = 4
    ) -> "SionnaChannelDataset":
        """Create dataset from pre-loaded samples.

        Args:
            samples: List of channel samples.
            num_sc: Number of subcarriers.
            num_symbols: Number of OFDM symbols.
            num_rx: Number of RX antennas.

        Returns
        -------
            SionnaChannelDataset instance with the provided samples.
        """
        instance = cls.__new__(cls)
        instance.samples = samples
        instance.num_sc = num_sc
        instance.num_symbols = num_symbols
        instance.num_rx = num_rx
        instance.expected_shape = (num_sc, num_symbols, num_rx)
        return instance


def setup_datasets(
    train_glob: str,
    test_glob: str,
    num_sc: int,
    validation_frac: float,
    prng_seed: int = 0,
    num_symbols: int = 14,
    num_rx: int = 4,
) -> tuple[SionnaChannelDataset, SionnaChannelDataset, SionnaChannelDataset]:
    """Setup train, validation, and test datasets.

    Args:
        train_glob: Glob pattern for training data files
        test_glob: Glob pattern for test data files
        num_sc: Number of subcarriers
        validation_frac: Fraction of training data to use for validation
        prng_seed: Random seed for dataset splitting
        num_symbols: Number of OFDM symbols
        num_rx: Number of RX antennas

    Returns
    -------
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Set random seed
    rng = np.random.default_rng(prng_seed)

    # Load datasets
    train_paths = sorted(glob.glob(train_glob))
    if not train_paths:
        error_msg = f"No train shards found: {train_glob}"
        raise FileNotFoundError(error_msg)
    ds_train = SionnaChannelDataset(train_paths, num_sc, num_symbols, num_rx)

    test_paths = sorted(glob.glob(test_glob))
    if not test_paths:
        error_msg = f"No test shards found: {test_glob}"
        raise FileNotFoundError(error_msg)
    test_dataset = SionnaChannelDataset(test_paths, num_sc, num_symbols, num_rx)

    # Split dataset into train and validation
    train_indices, val_indices = split_ids(len(ds_train), validation_frac, rng)

    # Create train dataset from split samples
    train_samples = [ds_train.samples[i] for i in train_indices]
    train_dataset = SionnaChannelDataset.from_samples(train_samples, num_sc, num_symbols, num_rx)

    # Create validation dataset from split samples
    val_samples = [ds_train.samples[i] for i in val_indices]
    val_dataset = SionnaChannelDataset.from_samples(val_samples, num_sc, num_symbols, num_rx)

    return train_dataset, val_dataset, test_dataset


def gen_split(
    split_name: str,
    config: SionnaCDLConfig,
    out_dir: str,
    gen: Any,  # phy.channel.generate_ofdm_channel.GenerateOFDMChannel
) -> None:
    """Generate a simplified dataset split and save sharded `.npz` files.

    Args:
        split_name: Name of the split prefix (e.g., "train", "test").
        config: Configuration object containing all parameters.
        out_dir: Output directory for sharded files.
        gen: Sionna channel generator object.

    Produces files named `{split_name}_{idx:03d}.npz` each containing arrays
    H as described in the module docstring.
    """
    total = config.train_total if split_name == "train" else config.test_total
    num_shards = (total + config.shard_size - 1) // config.shard_size

    for shard_idx in tqdm(range(num_shards), desc=f"Generating {split_name}", unit="shards"):
        this = min(config.shard_size, total - shard_idx * config.shard_size)
        # Store 3D channels: (batch_size, num_subcarriers, num_symbols, num_rx)
        H: npt.NDArray[np.complex64] = np.zeros((this, config.num_sc, 14, 4), dtype=np.complex64)
        filled = 0

        while filled < this:
            b = min(config.batch_tf, this - filled)

            # tf.complex64 [B, num_tx_ant, num_rx_ant, num_streams_per_tx,
            # num_streams_per_rx, num_symbols, num_sc]
            H_tf = gen(b)

            # Convert to numpy: Shape [B, 1, 4, 1, 1, 14, num_sc]
            H_np = H_tf.numpy()

            # Progress indicator for large batches
            if b > 100:
                print(f"  Processing batch of {b} samples...")

            for k in range(b):
                # Extract channel for this sample: [1, 4, 1, 1, 14, num_sc]
                # We want to get rid of singleton dimensions and get (4, 14, num_sc)
                h_true_3d = H_np[k, 0, :, 0, 0, :, : config.num_sc]

                # Transpose to desired format: (subcarriers x symbols x rx_antennas)
                h_true_final = h_true_3d.transpose(2, 1, 0)

                # Store channel
                H[filled] = h_true_final
                filled += 1
                if filled >= this:
                    break

        out_path = os.path.join(out_dir, f"{split_name}_{shard_idx:03d}.npz")
        np.savez_compressed(
            out_path,
            H=H,
            meta=dict(NUM_PRB=config.num_prb, MODEL_TYPE=config.model_type, MODEL=config.tdl_model),  # type: ignore[arg-type]
        )


def split_ids(
    n: int, validation_frac: float, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """Split dataset indices into train and validation sets.

    Args:
        n: Total number of samples.
        validation_frac: Fraction of samples to use for validation.
        rng: NumPy random generator.

    Returns
    -------
        Tuple of (train_indices, val_indices).

    Raises:
        ValueError: If dataset size is too small to split (n < 2).
    """
    if n < 2:
        error_msg = f"Dataset too small to split: n={n}. Need at least 2 samples."
        raise ValueError(error_msg)

    n_val = max(1, int(validation_frac * n))

    # Ensure at least 1 training sample remains
    if n_val >= n:
        n_val = n - 1

    idx = rng.permutation(n)
    return idx[n_val:].tolist(), idx[:n_val].tolist()
