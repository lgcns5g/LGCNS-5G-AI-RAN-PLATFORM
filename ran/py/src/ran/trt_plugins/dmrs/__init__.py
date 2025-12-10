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

"""DMRS optimization module for PUSCH."""

from .dmrs_trt_plugin import (
    dmrs_3276,
    get_dmrs_trt_plugin,
)
from .extract_raw_dmrs import extract_raw_dmrs_type1
from .transmitted_dmrs import (
    apply_dmrs_to_channel,
    gen_transmitted_dmrs_with_occ,
)

__all__ = [
    "apply_dmrs_to_channel",
    "dmrs_3276",
    "extract_raw_dmrs_type1",
    "gen_transmitted_dmrs_with_occ",
    "get_dmrs_trt_plugin",
]
