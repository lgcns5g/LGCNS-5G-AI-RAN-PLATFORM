#!/usr/bin/env python3
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
CLI script for checking and fixing copyright header compliance.

This script validates that all tracked source files have proper SPDX-compliant
copyright headers and can automatically fix violations.

Usage:
    copyright.py check [--extensions EXT...]  # Check compliance
    copyright.py fix [--extensions EXT...]    # Fix violations

Exit codes:
  0: All files are compliant (check) or successfully fixed (fix)
  1: Violations found (check mode only)
  2: Script error
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path to import sibling modules
sys.path.insert(0, str(Path(__file__).parent))

import copyright_utils
import git_utils

# Default file extensions to check/fix
# Special filenames (CMakeLists.txt, Dockerfile, compose.yaml) are also
# included by git_utils.get_tracked_files
DEFAULT_EXTENSIONS = [
    "c",
    "cc",
    "cpp",
    "cxx",
    "h",
    "hpp",
    "hxx",
    "cu",
    "cuh",
    "py",
    "sh",
    "cmake",
    "yaml",
    "yml",
]

# Third-party files to exclude from copyright checks
EXCLUDE_FILES = [
    "cmake/CPM.cmake",  # CMake Package Manager (MIT License)
    ".cmake-format.yaml",  # Third-party tool configuration
]


def main() -> None:
    """Check or fix copyright headers for SPDX compliance in tracked files."""
    parser = argparse.ArgumentParser(
        description="Check or fix copyright headers for SPDX compliance",
        epilog="Examples:\n"
        "  %(prog)s check              # Check all tracked files\n"
        "  %(prog)s fix                # Fix all violations\n"
        "  %(prog)s check --extensions py sh  # Check only Python/Shell\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mode", choices=["check", "fix"], help="Mode: check for violations or fix files"
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="File extensions to process (default: C/C++/CUDA/Python/Shell/CMake)",
    )

    args = parser.parse_args()

    # Get tracked files with specified extensions
    files = git_utils.get_tracked_files(extensions=args.extensions)

    # Filter out excluded third-party files
    exclude_set = {Path(f) for f in EXCLUDE_FILES}
    files = [f for f in files if f not in exclude_set]

    # Execute mode
    if args.mode == "check":
        results = copyright_utils.check_files(files)
        copyright_utils.print_report(results, "check")  # exits with code 1 if violations
    else:  # fix
        results = copyright_utils.fix_files(files)
        copyright_utils.print_report(results, "fix")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(2)
    except Exception:  # noqa: BLE001 - CLI script, suppress stack trace for clean UX
        sys.exit(2)
