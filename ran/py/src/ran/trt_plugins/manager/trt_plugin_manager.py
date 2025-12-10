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

"""TensorRT plugin manager for RAN package."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import tensorrt as trt

from ran.trt_plugins.manager._trt_plugin_field_map import _convert_type_to_trt

logger = logging.getLogger(__name__)


def _load_env_file_if_needed() -> None:
    """Load .env.python file if RAN_TRT_PLUGIN_DSO_PATH is not set.

    Priority 1: Use RAN_ENV_PYTHON_FILE if set
    Priority 2: Search upward from the current file's directory for .env.python
    Only loads if environment variables are not already set.
    """
    if os.environ.get("RAN_TRT_PLUGIN_DSO_PATH"):
        return  # Already set, nothing to do

    # Priority 1: Check for explicit env file path
    env_file_path = os.environ.get("RAN_ENV_PYTHON_FILE")
    env_file: Path | None = None
    if env_file_path:
        env_file = Path(env_file_path)
        logger.debug(f"Using RAN_ENV_PYTHON_FILE: {env_file}")
        if not env_file.exists():
            logger.warning(
                f"RAN_ENV_PYTHON_FILE points to non-existent file: {env_file}. "
                "Falling back to searching parent directories."
            )
            env_file = None  # Fall through to search

    # Priority 2: Search upward from current file for .env.python
    if not env_file:
        logger.debug("Searching for .env.python in parent directories")
        current_dir = Path(__file__).resolve().parent
        while current_dir != current_dir.parent:  # Stop at filesystem root
            candidate = current_dir / ".env.python"
            logger.debug(f"Checking: {candidate}")
            if candidate.exists():
                env_file = candidate
                logger.debug(f"Found .env.python at: {env_file}")
                break
            current_dir = current_dir.parent

    if not env_file:
        logger.debug(
            "No .env.python file found. "
            "Environment variables must be set manually or tests may fail."
        )
        return

    # Load the .env file using python-dotenv
    logger.info(f"Auto-loading environment from: {env_file}")
    load_dotenv(dotenv_path=env_file, override=False)


def get_ran_trt_plugin_dso_path() -> str:
    """Get the RAN TensorRT plugin DSO path from environment variable.

    This function returns the path to libran_trt_plugin.so from the
    RAN_TRT_PLUGIN_DSO_PATH environment variable. When running tests via CMake
    targets (py_ran_test, py_ran_wheel_test), this environment variable is
    automatically set by CMake.

    For interactive/manual usage, attempts to auto-load from .env.python file.

    Returns
    -------
        Absolute path to libran_trt_plugin.so

    Raises
    ------
        EnvironmentError: If RAN_TRT_PLUGIN_DSO_PATH is not set
        FileNotFoundError: If the DSO file does not exist at the specified path
    """
    _load_env_file_if_needed()
    env_dso_path = os.environ.get("RAN_TRT_PLUGIN_DSO_PATH")

    if not env_dso_path:
        raise EnvironmentError(
            "RAN_TRT_PLUGIN_DSO_PATH environment variable not set. "
            "This should be set automatically when running tests via CMake targets "
            "For manual testing, set it to the path of libran_trt_plugin.so."
        )

    dso_path = Path(env_dso_path)

    if not dso_path.exists():
        raise FileNotFoundError(
            f"TensorRT plugin DSO not found at path specified by "
            f"RAN_TRT_PLUGIN_DSO_PATH: {dso_path}"
        )

    return str(dso_path.absolute())


def get_ran_trt_engine_path() -> str:
    """Get the RAN TensorRT engine directory path from environment variable.

    This function returns the path to the TensorRT engine directory from the
    RAN_TRT_ENGINE_PATH environment variable. When running tests via CMake
    targets (py_ran_test, py_ran_wheel_test), this environment variable is
    automatically set by CMake.

    For interactive/manual usage, attempts to auto-load from .env.python file.

    Returns
    -------
        Absolute path to TensorRT engine directory

    Raises
    ------
        EnvironmentError: If RAN_TRT_ENGINE_PATH is not set
    """
    _load_env_file_if_needed()
    env_engine_path = os.environ.get("RAN_TRT_ENGINE_PATH")

    if not env_engine_path:
        raise EnvironmentError(
            "RAN_TRT_ENGINE_PATH environment variable not set. "
            "This should be set automatically when running tests via CMake targets. "
            "For manual testing, set it to the directory where TensorRT engines are stored."
        )

    return str(Path(env_engine_path).absolute())


def get_ran_pytest_build_dir() -> str:
    """Get the RAN pytest build directory path from environment variable.

    This function returns the path to the pytest build directory from the
    RAN_PYTEST_BUILD_DIR environment variable. When running tests via CMake
    targets, this environment variable is automatically set by CMake and points
    to the ran/py build directory (e.g., out/build/clang-release/ran/py).

    For interactive/manual usage, attempts to auto-load from .env.python file.

    Returns
    -------
        Absolute path to pytest build directory

    Raises
    ------
        EnvironmentError: If RAN_PYTEST_BUILD_DIR is not set
    """
    _load_env_file_if_needed()
    env_build_dir = os.environ.get("RAN_PYTEST_BUILD_DIR")

    if not env_build_dir:
        raise EnvironmentError(
            "RAN_PYTEST_BUILD_DIR environment variable not set. "
            "This should be set automatically when running tests via CMake targets. "
            "For manual testing, set it to the pytest build directory."
        )

    return str(Path(env_build_dir).absolute())


def should_skip_engine_generation(required_engines: list[str]) -> bool:
    """Check if TRT engine generation should be skipped.

    Checks if SKIP_TRT_ENGINE_GENERATION=1 and all required engines exist
    in RAN_TRT_ENGINE_PATH. Returns True if engines should be skipped (already exist),
    False if engines should be regenerated.

    Args:
        required_engines: List of engine filenames (e.g., ["dmrs_test.trtengine"])

    Returns:
        True if engine generation should be skipped (all engines exist),
        False if engines should be regenerated
    """
    _load_env_file_if_needed()
    skip_generation = os.getenv("SKIP_TRT_ENGINE_GENERATION", "0") == "1"

    if not skip_generation:
        return False

    engine_dir = Path(get_ran_trt_engine_path())
    all_engines_exist = all((engine_dir / engine_name).exists() for engine_name in required_engines)

    if all_engines_exist:
        logger.info(f"Skipping engine generation - all engines exist in {engine_dir}")
        logger.info("Set SKIP_TRT_ENGINE_GENERATION=0 to force regeneration")
        return True

    logger.info("Engines not found, proceeding with generation...")
    return False


def copy_trt_engine_for_cpp_tests(
    source_dir: str | Path,
    dest_engine_name: str,
    *,
    required: bool = True,
) -> Path:
    """Copy TensorRT engine from build directory to RAN_TRT_ENGINE_PATH for C++ tests.

    Args:
        source_dir: Directory containing tensorrt_cluster_engine_data.trtengine
        dest_engine_name: Destination filename (e.g., "dmrs_test.trtengine")
        required: If True, raises FileNotFoundError if source missing

    Returns:
        Path to destination engine file

    Raises:
        FileNotFoundError: If required=True and source doesn't exist
    """
    # Source: generated engine in build directory
    engine_source = Path(source_dir) / "tensorrt_cluster_engine_data.trtengine"

    # Destination: central engine location expected by C++ tests
    engine_dest_dir = Path(get_ran_trt_engine_path())
    engine_dest_dir.mkdir(parents=True, exist_ok=True)
    engine_dest = engine_dest_dir / dest_engine_name

    # Copy engine file if it exists
    if engine_source.exists():
        shutil.copy2(engine_source, engine_dest)
        logger.info(f"Copied TensorRT engine to {engine_dest}")
    elif required:
        raise FileNotFoundError(
            f"TensorRT engine not found at {engine_source}. Engine file is required for C++ tests."
        )
    else:
        logger.warning(f"TensorRT engine not found at {engine_source}")

    return engine_dest


def copy_test_data_for_cpp_tests(
    source_dir: str | Path,
    test_vector_subdir: str,
    file_patterns: list[str],
) -> Path:
    """Copy test data files from build directory to RAN_TRT_ENGINE_PATH/test_vectors for C++ tests.

    Args:
        source_dir: Directory containing the test data files
        test_vector_subdir: Subdirectory name under test_vectors/ (e.g., "ai_tukey_filter")
        file_patterns: List of file patterns to copy (e.g., ["*.bin", "*_meta.txt"])

    Returns:
        Path to destination test_vectors subdirectory
    """
    # Destination: test_vectors subdirectory under RAN_TRT_ENGINE_PATH
    engine_path = Path(get_ran_trt_engine_path())
    test_vectors_dir = engine_path / "test_vectors" / test_vector_subdir
    test_vectors_dir.mkdir(parents=True, exist_ok=True)

    # Copy all matching files
    source_path = Path(source_dir)
    files_copied = 0
    for pattern in file_patterns:
        for source_file in source_path.glob(pattern):
            if source_file.is_file():
                dest_file = test_vectors_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                files_copied += 1
                logger.debug(f"Copied {source_file.name} to {test_vectors_dir}")

    if files_copied > 0:
        logger.info(f"Copied {files_copied} test data files to {test_vectors_dir}")
    else:
        logger.warning(f"No files matched patterns {file_patterns} in {source_path}")

    return test_vectors_dir


class TensorRTPluginManager:
    """Manager for RAN TensorRT plugins."""

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self._trt_plugin_lib_path: Path | None = None
        self._trt_plugin_lib_loaded = False

    def load_plugin_library(self) -> bool:
        """Load the TensorRT plugin library from RAN_TRT_PLUGIN_DSO_PATH.

        The library path is obtained from the RAN_TRT_PLUGIN_DSO_PATH environment
        variable via config.get_ran_trt_plugin_dso_path(). When running tests via
        CMake targets, this environment variable is automatically set by CMake.

        Returns
        -------
            True if library loaded successfully.

        Raises
        ------
            EnvironmentError: If RAN_TRT_PLUGIN_DSO_PATH is not set.
            FileNotFoundError: If the DSO file does not exist at the specified path.
            RuntimeError: If plugin initialization fails.
        """
        if self._trt_plugin_lib_loaded:
            return True

        # Get library path from config
        lib_path = Path(get_ran_trt_plugin_dso_path())

        logger.info(f"Loading TensorRT plugin library: {lib_path}")

        # Load the library
        plugin_lib = ctypes.CDLL(str(lib_path))

        self._trt_plugin_lib_path = lib_path

        # Initialize TensorRT plugins (standard ones)
        trt.init_libnvinfer_plugins(None, "")

        # Initialize our custom RAN plugins
        if not hasattr(plugin_lib, "init_ran_plugins"):
            err_msg = f"init_ran_plugins function not found in {lib_path}"
            raise RuntimeError(err_msg)

        init_func = plugin_lib.init_ran_plugins
        init_func.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        init_func.restype = ctypes.c_bool

        # Default to empty namespace for convenience
        result = init_func(None, b"")
        if not result:
            err_msg = "Failed to initialize custom TensorRT plugins."
            raise RuntimeError(err_msg)

        self._trt_plugin_lib_loaded = True
        logger.info(f"TensorRT plugins loaded successfully from {lib_path}")
        return True

    def create_plugin(
        self, plugin_name: str, fields: dict[str, Any] | None = None
    ) -> object | None:
        """Create a plugin instance.

        Args:
            plugin_name: Name of the plugin to create (C++ TensorRT plugin name).
            fields: Dictionary of plugin fields to pass to the plugin creator.

        Returns
        -------
            Plugin instance if successful, None otherwise.
        """
        if fields is None:
            fields = {}

        # Load plugins if not already loaded
        if not self._trt_plugin_lib_loaded:
            self.load_plugin_library()

        plugin_registry = trt.get_plugin_registry()
        creators = plugin_registry.all_creators

        plugin_creator = None
        for creator in creators:
            if creator.name == plugin_name:
                plugin_creator = creator
                break

        if plugin_creator is None:
            err_msg = f"Plugin creator '{plugin_name}' not found in TensorRT registry."
            logger.warning(err_msg)
            return None

        # Create plugin field collection
        field_collection = None
        if fields is None:
            # No fields provided - create empty collection
            field_collection = trt.PluginFieldCollection([])
        else:
            # Convert Python dict to TensorRT plugin fields
            plugin_fields = []
            for field_name, field_value in fields.items():
                trt_type = _convert_type_to_trt(field_value)
                plugin_field = trt.PluginField(field_name, field_value, trt_type)
                plugin_fields.append(plugin_field)

            field_collection = trt.PluginFieldCollection(plugin_fields)

        return plugin_creator.create_plugin(plugin_name, field_collection, trt.TensorRTPhase.BUILD)
