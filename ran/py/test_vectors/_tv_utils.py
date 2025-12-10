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

"""Test-vector utilities."""

import logging
from pathlib import Path
from typing import ClassVar

from ran.utils import hdf5_load

# Shared config
logger = logging.getLogger(__name__)
TV_ROOT = Path(__file__).resolve().parents[2] / "test_data"


class TvLoader:
    """Load test vectors from H5 files.

    Caches previously loaded test vectors (H5 files) in-process to avoid reading the
    same file multiple times across tests. The cache is keyed by absolute file path.
    """

    _cache: ClassVar[dict[str, dict]] = {}

    @classmethod
    def load(cls, filename: str) -> dict:
        path = TV_ROOT / filename
        key = str(path)
        if key not in cls._cache:
            cls._cache[key] = hdf5_load(path)
        return cls._cache[key]
