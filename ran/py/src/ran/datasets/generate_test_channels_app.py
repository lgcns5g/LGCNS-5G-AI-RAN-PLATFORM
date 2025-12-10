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

"""Generate CDL channel test data for channel estimation testing.

Uses Sionna to generate CDL channel realizations and saves directly to safetensors
format with train/val/test splits.

Output:
  - train_data.safetensors: Training channels (small subset for validation)
  - test_data.safetensors: Test channels (main test dataset)
  - Each contains H__sc_sym_rxant: complex64 array of shape (N, num_sc, 14, 4)

Usage:
    python generate_test_channels.py [--config CONFIG_PATH] [--num-samples NUM]
"""

import argparse
import logging
import os
from pathlib import Path

# Force CPU-only mode for this test data generation app
# CPU is faster for small-medium datasets
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Limit JAX CPU memory allocation
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.05")

import yaml

from ran.datasets import gen_cdl_channels
from ran.datasets.sionna_cdl_config import SionnaCDLConfig
from ran.utils import load_config

logger = logging.getLogger(__name__)


def generate_test_channel_dataset(
    cfg_path: Path,
    num_samples: int | None = None,
) -> tuple[Path, int, int, int]:
    """Generate CDL channel dataset for testing.

    Args:
        cfg_path: Path to configuration YAML file
        num_samples: Optional override for total number of test samples.
                    If provided, updates the config's train_total and test_total.

    Returns:
        Tuple of (output_dir, n_train, n_val, n_test):
            - output_dir: Path to output directory containing generated data
            - n_train: Number of training samples
            - n_val: Number of validation samples
            - n_test: Number of test samples
    """
    config_dict = load_config(cfg_path=cfg_path)
    output_dir = Path(config_dict.pop("output_dir"))
    validation_frac = config_dict.pop("validation_frac", 0.15)

    # Override sample counts if provided
    if num_samples is not None:
        # Small train set for validation, most samples go to test
        config_dict["train_total"] = max(32, num_samples // 10)
        config_dict["test_total"] = num_samples

    # Create config object
    config = SionnaCDLConfig(**config_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = output_dir / "dataset_config.yaml"
    with open(config_output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("Generating channel dataset:")
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Train samples: %d", config.train_total)
    logger.info("  Test samples: %d", config.test_total)
    logger.info("  Validation fraction: %s", validation_frac)
    logger.info("  Random seed: %d", config.prng_seed)

    n_train, n_val, n_test = gen_cdl_channels(
        config, validation_frac=validation_frac, out_dir=str(output_dir)
    )

    return output_dir, n_train, n_val, n_test


def main() -> None:
    """Main entry point for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate CDL channel test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "generate_test_channels_app.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override total number of test samples (train set will be ~10%% of this)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {args.config}\n"
            f"Please create it or specify a different path with --config"
        )

    output_dir, n_train, n_val, n_test = generate_test_channel_dataset(
        cfg_path=args.config,
        num_samples=args.num_samples,
    )

    logger.info("=" * 80)
    logger.info("Dataset generation completed successfully!")
    logger.info("=" * 80)
    logger.info("Generated:")
    logger.info("  - %d train samples", n_train)
    logger.info("  - %d validation samples", n_val)
    logger.info("  - %d test samples", n_test)
    logger.info("")
    logger.info("Output location: %s", output_dir)
    logger.info("  - %s", output_dir / "train_data.safetensors")
    logger.info("  - %s", output_dir / "test_data.safetensors")
    logger.info("")
    logger.info("You can now run the channel estimation tests:")
    logger.info("  pytest test_channel_estimation.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
