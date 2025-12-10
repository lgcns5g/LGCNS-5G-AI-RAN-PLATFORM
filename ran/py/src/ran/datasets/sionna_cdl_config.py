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


@dataclass
class SionnaCDLConfig:
    """Configuration class for CDL channel generation parameters.

    Supports CDL (Clustered Delay Line) models with configurable antenna arrays.
    """

    # Basic dataset configuration
    num_sc_per_prb: int = 12
    num_prb: int = 273
    train_total: int = 20
    test_total: int = 50
    shard_size: int = 50
    batch_tf: int = 256
    prng_seed: int = 0

    # CDL model configuration
    model_type: str = "CDL"  # CDL (Clustered Delay Line)
    tdl_model: str = "C"  # Model variant (A/B/C/D/E)
    delay_spread_ns: float = 100.0
    fc_ghz: float = 3.5
    speed_min: float = 0.0
    speed_max: float = 30.0
    direction: str = "uplink"

    # BS (Base Station) antenna array configuration
    bs_num_rows: int = 4
    bs_num_cols: int = 1
    bs_polarization: str = "single"
    bs_polarization_type: str = "V"
    bs_antenna_pattern: str = "38.901"

    # UE (User Equipment) antenna array configuration
    ue_num_rows: int = 1
    ue_num_cols: int = 1
    ue_polarization: str = "single"
    ue_polarization_type: str = "V"
    ue_antenna_pattern: str = "38.901"

    def __post_init__(self) -> None:
        """Initialize derived parameters."""
        # Derived parameters
        self.num_sc = self.num_prb * self.num_sc_per_prb
        self.nmax_sc = self.num_sc  # Alias for compatibility
