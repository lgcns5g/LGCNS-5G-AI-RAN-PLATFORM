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

"""Generate CDL channel training data for AI channel estimation.

Uses Sionna to generate CDL channel realizations and saves directly to safetensors
format with train/val/test splits. AWGN noise and DMRS transmission are applied
during training for maximum flexibility.

Output:
  - train_data.safetensors: Training channels
  - val_data.safetensors: Validation channels
  - test_data.safetensors: Test channels
  - Each contains H__sc_sym_rxant: complex64 array of shape (N, num_sc, 14, 4)
"""

import shutil
from pathlib import Path

import yaml
from safetensors.numpy import safe_open

from ran.datasets import gen_cdl_channels
from ran.datasets.sionna_cdl_config import SionnaCDLConfig
from ran.utils import load_config


def gen_ai_tukey_filter_cdl_training_dataset(
    cfg_path: Path | None = None,
    config_dict: dict | None = None,
    make_new_dataset: bool = True,
) -> tuple[Path, int, int, int]:
    """Generate CDL channel dataset.

    Args:
        cfg_path: Path to configuration YAML file. Mutually exclusive with config_dict.
        config_dict: Configuration dictionary. If provided, cfg_path is ignored.
                    Allows for programmatic configuration from notebooks.
        make_new_dataset: If True (default), erase existing dataset and regenerate.
                         If False, skip generation if dataset already exists.

    Returns:
        Tuple of (output_dir, n_train, n_val, n_test):
            - output_dir: Path to output directory containing generated data
            - n_train: Number of training samples
            - n_val: Number of validation samples
            - n_test: Number of test samples

    Raises:
        ValueError: If both cfg_path and config_dict are provided, or if neither is provided.
    """
    if config_dict is not None:
        if cfg_path is not None:
            raise ValueError("Cannot specify both cfg_path and config_dict parameters")
        original_config = config_dict.copy()
    elif cfg_path is not None:
        config_dict = load_config(cfg_path=cfg_path)
        original_config = config_dict.copy()
    else:
        raise ValueError("Must specify either cfg_path or config_dict parameter")

    output_dir = Path(config_dict.pop("output_dir"))
    validation_frac = config_dict.pop("validation_frac", 0.15)

    # Create config object
    config = SionnaCDLConfig(**config_dict)

    # Check if dataset already exists
    train_path = output_dir / "train_data.safetensors"
    val_path = output_dir / "val_data.safetensors"
    test_path = output_dir / "test_data.safetensors"
    dataset_exists = train_path.exists() and val_path.exists() and test_path.exists()

    if dataset_exists and not make_new_dataset:
        print(f"✓ Dataset already exists at: {output_dir}")
        print("  Skipping generation (set make_new_dataset=True to regenerate)")

        # Load existing dataset to get sample counts
        with safe_open(train_path, framework="numpy") as f:
            n_train = f.get_tensor("H__sc_sym_rxant.real").shape[0]
        with safe_open(val_path, framework="numpy") as f:
            n_val = f.get_tensor("H__sc_sym_rxant.real").shape[0]
        with safe_open(test_path, framework="numpy") as f:
            n_test = f.get_tensor("H__sc_sym_rxant.real").shape[0]

        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {n_val}")
        print(f"  Test samples: {n_test}")

        return output_dir, n_train, n_val, n_test

    # Erase existing directory if make_new_dataset=True and it exists
    if make_new_dataset and output_dir.exists():
        print(f"Erasing existing dataset directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = output_dir / "dataset_config.yaml"
    with open(config_output_path, "w") as f:
        yaml.dump(original_config, f, default_flow_style=False, sort_keys=False)

    n_train, n_val, n_test = gen_cdl_channels(
        config, validation_frac=validation_frac, out_dir=str(output_dir)
    )

    return output_dir, n_train, n_val, n_test
