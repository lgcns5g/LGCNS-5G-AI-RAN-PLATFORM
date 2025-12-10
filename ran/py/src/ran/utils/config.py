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

"""Configuration utilities."""

from pathlib import Path

import yaml


def load_config(cfg_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        cfg_path: Path to config file.

    Returns:
        Configuration dictionary
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
