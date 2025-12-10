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

"""Utilities for accessing pretrained AI Tukey filter models."""

from pathlib import Path
from typing import Optional

# Pretrained models directory (relative to this file)
_PRETRAINED_MODELS_DIR = Path(__file__).parent
# Default model
DEFAULT_PRETRAINED_AI_TUKEY_FILTER = "ai_tukey_filter_20251201"


def _find_model_path(model_name: str) -> Path | None:
    """Find model path by searching in pretrained_models directory tree.

    Searches for a model directory containing required files in both the root
    directory and any subdirectories.

    Parameters
    ----------
    model_name : str
        Model name to search for

    Returns
    -------
    Path | None
        Path to model directory if found, None otherwise
    """
    if not _PRETRAINED_MODELS_DIR.exists():
        return None

    # Search using glob pattern to find model directories anywhere in the tree
    # Matches: pretrained_models/model_name/ or pretrained_models/*/model_name/
    for pattern in [model_name, f"*/{model_name}"]:
        matches = list(_PRETRAINED_MODELS_DIR.glob(pattern))
        for path in matches:
            if (
                path.is_dir()
                and (path / "model_params.flax").exists()
                and (path / "model_config.yaml").exists()
            ):
                return path

    return None


def list_pretrained_ai_tukey_filters() -> list[str]:
    """List available pretrained AI Tukey filter models.

    Searches for models in the pretrained_models directory tree.
    Returns only model names (directory structure is transparent to users).

    Returns
    -------
    list[str]
        List of available pretrained model names
    """
    if not _PRETRAINED_MODELS_DIR.exists():
        return []

    models = set()

    # Search for all directories containing required model files
    # Pattern: **/model_name/ (any depth)
    for params_file in _PRETRAINED_MODELS_DIR.rglob("model_params.flax"):
        model_dir = params_file.parent
        config_file = model_dir / "model_config.yaml"

        # Validate both required files exist
        if config_file.exists():
            # Exclude __pycache__ and similar directories
            if "__pycache__" not in model_dir.parts:
                models.add(model_dir.name)

    return sorted(models)


def get_pretrained_ai_tukey_filter_path(model_name: Optional[str] = None) -> Path:
    """Get the path to a pretrained AI Tukey filter model.

    Automatically searches for models in the pretrained_models directory tree,
    including any subdirectories. The directory structure is transparent to users.

    Parameters
    ----------
    model_name : str | None, optional
        Model name. If None, returns the default model.
        Available models: list_pretrained_ai_tukey_filters()

    Returns
    -------
    Path
        Absolute path to the model directory containing model_params.flax and model_config.yaml

    Raises
    ------
    FileNotFoundError
        If the specified model does not exist or if no default model is available

    Examples
    --------
    >>> from ran.phy.jax.pusch.ai_tukey_filter.pretrained_models import (
    ...     get_pretrained_ai_tukey_filter_path
    ... )
    >>> model_dir = get_pretrained_ai_tukey_filter_path()
    >>> params_path = model_dir / "model_params.flax"
    """
    if model_name is None:
        if DEFAULT_PRETRAINED_AI_TUKEY_FILTER is None:
            available = list_pretrained_ai_tukey_filters()
            raise FileNotFoundError(
                "No default pretrained model is configured. "
                f"Available models: {available if available else 'none'}. "
                "Please specify a model_name explicitly."
            )
        model_name = DEFAULT_PRETRAINED_AI_TUKEY_FILTER

    # Search for model in directory tree (transparent to directory structure)
    model_path = _find_model_path(model_name)

    if model_path is None:
        available = list_pretrained_ai_tukey_filters()
        raise FileNotFoundError(
            f"Pretrained model '{model_name}' not found. "
            f"Available models: {available if available else 'none'}"
        )

    return model_path.resolve()


__all__ = [
    "DEFAULT_PRETRAINED_AI_TUKEY_FILTER",
    "get_pretrained_ai_tukey_filter_path",
    "list_pretrained_ai_tukey_filters",
]
