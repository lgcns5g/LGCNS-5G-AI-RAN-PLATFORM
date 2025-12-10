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

"""AI Tukey Filter package."""

from .ai_tukey_filter_cdl_training_dataset import (
    gen_ai_tukey_filter_cdl_training_dataset,
)
from .ai_tukey_filter_model import (
    create_model,
    count_parameters,
)
from .ai_tukey_filter_model_pusch_channel_estimation_wrapper import (
    ai_tukey_filter,
    AITukeyFilterConfig,
    load_model_config_from_yaml,
    tukey_window_impl,
)
from .pretrained_models.ai_tukey_filter_pretrained_models import (
    DEFAULT_PRETRAINED_AI_TUKEY_FILTER,
    get_pretrained_ai_tukey_filter_path,
    list_pretrained_ai_tukey_filters,
)
from .train_ai_tukey_filter_model_config import (
    TrainConfig,
)

# Note: train_ai_tukey_filter_model is NOT imported here to avoid circular imports
# Import it directly: from ran.phy.jax.pusch.ai_tukey_filter.train_ai_tukey_filter_model import train_ai_tukey_filter_model

__all__ = [
    "ai_tukey_filter",
    "AITukeyFilterConfig",
    "create_model",
    "count_parameters",
    "DEFAULT_PRETRAINED_AI_TUKEY_FILTER",
    "gen_ai_tukey_filter_cdl_training_dataset",
    "get_pretrained_ai_tukey_filter_path",
    "list_pretrained_ai_tukey_filters",
    "load_model_config_from_yaml",
    "TrainConfig",
    "tukey_window_impl",
]
