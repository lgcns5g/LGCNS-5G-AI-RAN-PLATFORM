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

"""Version resolution - finds and reads root VERSION file."""

from pathlib import Path


def _find_version() -> str:
    """Walk up directory tree to find VERSION file."""
    current = Path(__file__).parent
    while current != current.parent:
        version_file = current / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip()
        current = current.parent
    raise FileNotFoundError("VERSION file not found")


__version__ = _find_version()
