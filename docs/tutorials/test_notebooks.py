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

"""Pytest tests for notebooks using nbclient with locked kernel startup.

This module replaces pytest-nbmake with direct nbclient calls, allowing us to:
1. Serialize kernel startup (prevents ZMQ port conflicts)
2. Execute notebooks in parallel (after kernel starts)

See: https://github.com/jupyter/jupyter_client/issues/487
"""

import logging
import os
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import nbformat
import pytest
from filelock import FileLock
from nbclient import NotebookClient

# Set up logging with timestamps
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class KernelStartupLock:
    """File-based lock to serialize Jupyter kernel startup.

    Prevents port conflicts when starting multiple kernels in parallel.
    Only the kernel startup phase is locked (~2s), execution runs in parallel.
    """

    def __init__(self, notebook_name: str | None = None) -> None:
        """Initialize lock with 60s timeout.

        Args:
            notebook_name: Name of notebook (for logging)
        """
        lock_path = Path(tempfile.gettempdir()) / "jupyter_kernel_startup.lock"
        self.lock = FileLock(str(lock_path), timeout=60)
        self.notebook_name = notebook_name or "unknown"
        self.acquire_time = None

    def __enter__(self):
        """Acquire lock."""
        logger.info("[%s] Requesting kernel startup lock", self.notebook_name)
        start_wait = time.time()
        result = self.lock.__enter__()
        wait_time = time.time() - start_wait
        self.acquire_time = time.time()
        logger.info(
            "[%s] Lock acquired (waited %.2fs) - starting kernel",
            self.notebook_name,
            wait_time,
        )
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        hold_time = time.time() - self.acquire_time if self.acquire_time else 0
        logger.info(
            "[%s] Lock released (held %.2fs) - kernel startup complete",
            self.notebook_name,
            hold_time,
        )
        return self.lock.__exit__(exc_type, exc_val, exc_tb)


def discover_notebooks() -> list[Path]:
    """Discover all notebooks in the generated directory.

    Returns
    -------
        List of notebook paths, sorted by name
    """
    generated_dir = Path(__file__).parent / "generated"
    if not generated_dir.exists():
        return []
    return sorted(generated_dir.glob("*.ipynb"))


def execute_notebook_with_lock(notebook_path: Path) -> None:
    """Execute a notebook with locked kernel startup.

    Args:
        notebook_path: Path to the notebook

    Raises
    ------
        Exception: If notebook execution fails

    Notes
    -----
        Timeout is read from NOTEBOOK_TIMEOUT environment variable (default: 420 seconds)
    """
    notebook_name = notebook_path.stem

    # Get timeout from environment variable (set by setup_python_env.py)
    timeout = int(os.environ.get("NOTEBOOK_TIMEOUT", "420"))

    logger.info("[%s] Starting notebook execution", notebook_name)
    start_total = time.time()

    # Load notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Create executor
    executor = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=timeout,
        resources={"metadata": {"path": notebook_path.parent}},
        record_timing=True,
    )

    try:
        # Create kernel manager
        executor.create_kernel_manager()

        # Start kernel with lock (serialized, ~2 seconds per kernel)
        kernel_start = time.time()
        with KernelStartupLock(notebook_name):
            executor.start_new_kernel()
            executor.start_new_kernel_client()
        kernel_time = time.time() - kernel_start
        logger.info("[%s] Kernel started in %.2fs", notebook_name, kernel_time)

        # Execute notebook in parallel (lock released!)
        logger.info("[%s] Executing notebook cells (parallel)...", notebook_name)
        exec_start = time.time()
        executor.execute()
        exec_time = time.time() - exec_start
        logger.info("[%s] Execution completed in %.2fs", notebook_name, exec_time)

        # Save executed notebook (like --overwrite flag in nbmake)
        with open(notebook_path, "w") as f:
            nbformat.write(nb, f)

        total_time = time.time() - start_total
        logger.info(
            "[%s] Test PASSED - Total: %.2fs (kernel: %.2fs, execution: %.2fs)",
            notebook_name,
            total_time,
            kernel_time,
            exec_time,
        )

    finally:
        # Always cleanup kernel, suppress any cleanup errors
        with suppress(Exception):
            executor.shutdown_kernel()


# Parametrize test with all discovered notebooks
@pytest.mark.parametrize("notebook_path", discover_notebooks(), ids=lambda p: p.stem)
def test_notebook(notebook_path: Path) -> None:
    """Test a single notebook execution.

    This test:
    1. Loads the notebook
    2. Starts kernel (with lock to prevent port conflicts)
    3. Executes all cells (in parallel)
    4. Saves the executed notebook

    Args:
        notebook_path: Path to notebook to test
    """
    execute_notebook_with_lock(notebook_path)

