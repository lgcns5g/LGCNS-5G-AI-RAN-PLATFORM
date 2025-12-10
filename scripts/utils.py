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

"""Shared utilities for Python scripts."""

from __future__ import annotations

import logging
import subprocess
import sys


def run_command(
    cmd: str | list[str],
    logger: logging.Logger,
    cwd: str | None = None,
    *,
    shell: bool = False,
    stream_output: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and exit on failure.

    Args:
        cmd: Command to run as string or list of arguments
        logger: Logger instance
        cwd: Working directory
        shell: Whether to use shell. Defaults to False for security.
               Only set to True when shell features (pipes, redirection) are needed.
        stream_output: If True, stream output to console instead of capturing it
        env: Optional environment variables dict to pass to subprocess
    """
    # Format command for logging
    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd
        if not shell:
            msg = (
                "run_command received a string but shell=False. "
                "For security, pass commands as a list to prevent shell injection, "
                "or set shell=True if shell features (pipes, redirects) are needed."
            )
            raise RuntimeError(msg)

    logger.debug(f"Running: {cmd_str}")

    # Run command with conditional output capture
    result = subprocess.run(
        cmd,
        check=False,
        shell=shell,
        cwd=cwd,
        env=env,
        capture_output=not stream_output,
        text=not stream_output,
    )

    # Log captured output (DEBUG on success, ERROR on failure)
    if not stream_output:
        log_level = logger.error if result.returncode != 0 else logger.debug
        if result.stdout:
            log_level(f"Command stdout: {result.stdout}")
        if result.stderr:
            log_level(f"Command stderr: {result.stderr}")

    # Exit on failure
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd_str} with return code {result.returncode}")
        sys.exit(1)
