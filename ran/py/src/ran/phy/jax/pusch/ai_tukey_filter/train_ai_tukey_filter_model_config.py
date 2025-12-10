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

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class DebugConfig:
    """Debug settings."""

    disable_jit: bool = False


@dataclass
class PathsConfig:
    """Output directory settings."""

    output_dir: str
    checkpoint_dir: str
    dataset_dir: str

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.output_path / self.checkpoint_dir

    @property
    def dataset_path(self) -> Path:
        return Path(self.dataset_dir)


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    tau_max: int
    compressed_len: int
    d_model: int
    n_heads: int
    n_layers: int
    input_subsample_factor: int
    fft_size: int
    delay_compensation_samples: float


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_epochs: int
    gradient_clip: float
    seed: int
    snr_min_db: float
    snr_max_db: float


@dataclass
class ChannelConfig:
    """Channel parameters."""

    n_prb: int = 0  # Loaded from dataset_config.yaml

    @property
    def n_sc(self) -> int:
        """Total number of subcarriers."""
        return 12 * self.n_prb

    @property
    def n_dmrs_sc(self) -> int:
        """Number of DMRS subcarriers (CDM group 0 only)."""
        return 6 * self.n_prb


@dataclass
class DMRSExtractionConfig:
    """DMRS extraction pattern."""

    stride: int
    offset: int


@dataclass
class DMRSConfig:
    """DMRS configuration."""

    n_dmrs_id: int
    slot_number: int
    n_t: int
    port_idx: list[int]
    vec_scid: list[int]
    dmrs_idx: list[int]
    start_prb: int
    energy: float


@dataclass
class TrainConfig:
    """Complete training configuration."""

    debug: DebugConfig
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    channel: ChannelConfig
    dmrs_extraction: DMRSExtractionConfig
    dmrs_config: DMRSConfig

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "TrainConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            debug=DebugConfig(**data["debug"]),
            paths=PathsConfig(**data["paths"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"]),
            channel=ChannelConfig(**data["channel"]),
            dmrs_extraction=DMRSExtractionConfig(**data["dmrs_extraction"]),
            dmrs_config=DMRSConfig(**data["dmrs_config"]),
        )

    def save_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to the output YAML file
        """
        config_dict = {
            "debug": asdict(self.debug),
            "paths": asdict(self.paths),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "channel": asdict(self.channel),
            "dmrs_extraction": asdict(self.dmrs_extraction),
            "dmrs_config": asdict(self.dmrs_config),
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
