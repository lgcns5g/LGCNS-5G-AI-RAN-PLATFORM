#!/usr/bin/env python3  # noqa: EXE001
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

"""
Shared git utilities for repository file discovery.

This module provides common utilities for discovering tracked files in the git repository,
used by various code quality and compliance checking tools.
"""

import re
import subprocess
import sys
from pathlib import Path


def get_tracked_files(
    patterns: list[str] | None = None, extensions: list[str] | None = None, cwd: str | None = None
) -> list[Path]:
    """
    Get git-tracked files matching specified patterns or extensions.

    Args:
        patterns: List of glob patterns to match (e.g., ["framework/**/*.h", "ran/**/*.cpp"])
        extensions: List of file extensions to match (e.g., ["cpp", "h", "py"])
        cwd: Working directory for git command (defaults to current directory)

    Returns
    -------
        List of Path objects for matching tracked files

    Raises
    ------
        SystemExit: If git command fails

    Notes
    -----
        - At least one of patterns or extensions must be provided
        - If extensions is provided, it also includes CMakeLists.txt files
        - Results are deduplicated if both patterns and extensions match the same file
    """
    if not patterns and not extensions:
        msg = "Either patterns or extensions must be provided"
        raise ValueError(msg)

    all_files = set()

    # Process glob patterns if provided
    if patterns:
        for pattern in patterns:
            try:
                cmd = ["git", "ls-files", pattern]
                if cwd:
                    result = subprocess.run(  # noqa: S603 - git is a trusted command
                        cmd, capture_output=True, text=True, check=True, cwd=cwd
                    )
                else:
                    result = subprocess.run(  # noqa: S603 - git is a trusted command
                        cmd, capture_output=True, text=True, check=True
                    )

                files = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                all_files.update(files)
            except subprocess.CalledProcessError:
                sys.exit(2)

    # Process extensions if provided
    if extensions:
        try:
            cmd = ["git", "ls-files"]
            if cwd:
                result = subprocess.run(  # noqa: S603 - git is a trusted command
                    cmd, capture_output=True, text=True, check=True, cwd=cwd
                )
            else:
                result = subprocess.run(  # noqa: S603 - git is a trusted command
                    cmd, capture_output=True, text=True, check=True
                )

            # Build regex pattern for extensions
            ext_pattern = "|".join(re.escape(ext) for ext in extensions)
            pattern = re.compile(rf"\.({ext_pattern})$")

            for line in result.stdout.strip().split("\n"):
                # Check file extension or special filenames:
                # CMakeLists.txt, Dockerfile, compose.yaml, *.yaml.in
                # Extract filename for precise matching of special files
                filename = line.lower().split("/")[-1] if "/" in line else line.lower()

                if line and (
                    pattern.search(line)
                    or line.lower().endswith("cmakelists.txt")
                    or "/cmakelists.txt" in line.lower()
                    or filename == "dockerfile"
                    or filename.startswith("dockerfile.")
                    or line.lower().endswith("compose.yaml")
                    or line.lower().endswith(".yaml.in")
                ):
                    all_files.add(line)

        except subprocess.CalledProcessError:
            sys.exit(2)

    # Convert to Path objects and sort
    return sorted([Path(f) for f in all_files])


def get_git_root() -> Path:
    """
    Get the root directory of the git repository.

    Returns
    -------
        Path object pointing to git repository root

    Raises
    ------
        SystemExit: If not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        sys.exit(2)
