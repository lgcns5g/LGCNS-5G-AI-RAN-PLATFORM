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

"""Pytest configuration for RAN Python tests."""

import logging
import os
from pathlib import Path

import pytest

# Handle the case where python-dotenv is optional.
# Tests run during wheel testing won't have it installed.
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


def pytest_configure(config: object) -> None:
    """Configure pytest environment before tests run."""
    # Load environment variables from .env.python (must happen before any imports that need them)
    # Priority 1: Check for explicit env file path from CMake/ctest
    env_file_path = os.environ.get("RAN_ENV_PYTHON_FILE")
    if env_file_path:
        env_file = Path(env_file_path)
    else:
        # Priority 2: Default to source directory .env.python (synced by fixture or manual)
        env_file = Path(__file__).parent.parent / ".env.python"

    if env_file.exists() and DOTENV_AVAILABLE:
        load_dotenv(dotenv_path=env_file)

    # Silence noisy JAX debug logs
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src").setLevel(logging.WARNING)
    logging.getLogger("jax._src.dispatch").setLevel(logging.WARNING)
    logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.WARNING)
    logging.getLogger("jax._src.compiler").setLevel(logging.WARNING)
    logging.getLogger("mlir_tensorrt.compiler").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    logging.getLogger("jax._src.cache_key").setLevel(logging.WARNING)
    logging.getLogger("jax._src.compilation_cache").setLevel(logging.WARNING)

    # Silence noisy h5py debug logs
    logging.getLogger("h5py").setLevel(logging.WARNING)

    # Silence noisy matplotlib debug logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)


def pytest_sessionstart(session: object) -> None:
    """Log environment setup after pytest logging is initialized."""
    # Check same priority order as pytest_configure
    env_file_path = os.environ.get("RAN_ENV_PYTHON_FILE")
    if env_file_path:
        env_file = Path(env_file_path)
        source = "RAN_ENV_PYTHON_FILE environment variable"
    else:
        env_file = Path(__file__).parent.parent / ".env.python"
        source = "source directory (synced by fixture)"

    if env_file.exists():
        if DOTENV_AVAILABLE:
            logger.info(f"Loaded environment from: {env_file} (via {source})")

            # Log non-sensitive variables from whitelist
            safe_vars = {
                "RAN_TRT_PLUGIN_DSO_PATH",
                "RAN_TRT_ENGINE_PATH",
                "RAN_PYTEST_BUILD_DIR",
                "MLIR_TRT_COMPILER_PATH",
                "ENABLE_MLIR_TRT",
            }
            logger.info("Environment file contents:")
            for line in env_file.read_text().splitlines():
                if line and not line.startswith("#"):
                    var_name = line.split("=")[0]
                    if var_name in safe_vars:
                        logger.info(f"  {line}")
                    else:
                        logger.info(f"  {var_name}=<value set, but not logged here>")
        else:
            logger.warning(
                f".env.python file exists at {env_file} but python-dotenv is not installed. "
                "Environment variables must be set manually or tests may fail."
            )
    else:
        logger.info(
            f"No .env.python file found at {env_file} (via {source}). "
            "Environment variables must be set manually if needed. "
            "Run CMake configure to auto-generate this file, or set RAN_ENV_PYTHON_FILE to point to one."
        )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Ignore phy/jax test collection when ENABLE_MLIR_TRT=OFF.

    This hook runs before test collection, preventing import errors
    when required dependencies (mlir_tensorrt) are not installed.
    """
    enable_mlir_trt = os.environ.get("ENABLE_MLIR_TRT", "OFF").upper()

    if enable_mlir_trt == "OFF":
        # Check if path is in phy/jax directory
        parts = collection_path.parts
        if "jax" in parts and "phy" in parts:
            logger.info(f"Ignoring {collection_path} (ENABLE_MLIR_TRT=OFF)")
            return True

    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection to skip mlir_trt tests when ENABLE_MLIR_TRT=OFF.

    This is a secondary check for tests marked with @pytest.mark.mlir_trt
    that aren't in phy/jax directories.
    """
    enable_mlir_trt = os.environ.get("ENABLE_MLIR_TRT", "OFF").upper()

    if enable_mlir_trt == "OFF":
        skip_mlir_trt = pytest.mark.skip(reason="Skipping mlir_trt tests: ENABLE_MLIR_TRT=OFF")
        mlir_trt_skipped_count = 0

        for item in items:
            # Check if test has mlir_trt marker
            if "mlir_trt" in item.keywords:
                item.add_marker(skip_mlir_trt)
                mlir_trt_skipped_count += 1

        if mlir_trt_skipped_count > 0:
            logger.info(
                f"ENABLE_MLIR_TRT=OFF: Skipping {mlir_trt_skipped_count} mlir_trt tests. "
                "To enable these tests, build with -DENABLE_MLIR_TRT=ON"
            )
