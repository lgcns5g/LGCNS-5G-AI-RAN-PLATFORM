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

"""Global TensorRT plugin manager for RAN package."""

import logging
from typing import Any

from ran.trt_plugins.manager.trt_plugin_manager import TensorRTPluginManager

logger = logging.getLogger(__name__)

# Global plugin manager instance
_global_trt_plugin_manager = TensorRTPluginManager()


def global_trt_plugin_manager_create_plugin(
    name: str, fields: dict[str, Any] | None = None
) -> object | None:
    """Create a TensorRT plugin.

    Args:
        name: Name of the plugin to create.

    Returns
    -------
        Plugin instance if successful, None otherwise.
    """
    if fields is None:
        fields = {}

    return _global_trt_plugin_manager.create_plugin(name, fields)
