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
CLI script for checking and fixing include guard compliance in header files.

This script validates that C/C++/CUDA header files follow the project's include
guard standards (no #pragma once, correct prefixes) and can automatically fix
violations.

Usage:
    include_guards.py check [--patterns PATTERN...]  # Check compliance
    include_guards.py fix [--patterns PATTERN...]    # Fix violations

Exit codes:
  0: All files comply with standards (check) or successfully fixed (fix)
  1: Violations found (check mode only)
  2: Script error
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path to import sibling modules
sys.path.insert(0, str(Path(__file__).parent))

import git_utils
import include_guard_utils

# Default patterns for header files
DEFAULT_PATTERNS = [
    "framework/**/*.h",
    "framework/**/*.hpp",
    "framework/**/*.cuh",
    "ran/runtime/**/*.h",
    "ran/runtime/**/*.hpp",
    "ran/runtime/**/*.cuh",
    "ran/py/**/*.h",
    "ran/py/**/*.hpp",
    "ran/py/**/*.cuh",
]


def main() -> None:
    """Check or fix include guard compliance in header files."""
    parser = argparse.ArgumentParser(
        description="Check or fix include guard compliance for header files",
        epilog="Examples:\n"
        "  %(prog)s check              # Check all tracked headers\n"
        "  %(prog)s fix                # Fix all violations\n"
        "  %(prog)s check --patterns 'framework/**/*.h'  # Check specific pattern\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mode", choices=["check", "fix"], help="Mode: check for violations or fix files"
    )

    parser.add_argument(
        "--patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="Glob patterns for header files to process",
    )

    args = parser.parse_args()

    # Get tracked header files with specified patterns
    files = git_utils.get_tracked_files(patterns=args.patterns)

    # Execute mode
    if args.mode == "check":
        violations = include_guard_utils.check_files(files)
        exit_code = include_guard_utils.print_report(violations, len(files))
        sys.exit(exit_code)
    else:  # fix
        results = include_guard_utils.fix_files(files)
        include_guard_utils.print_fix_report(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(2)
    except Exception:  # noqa: BLE001 - CLI script, suppress stack trace for clean UX
        sys.exit(2)
