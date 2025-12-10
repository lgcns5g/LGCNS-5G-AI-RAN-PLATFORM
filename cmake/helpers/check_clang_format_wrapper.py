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
Wrapper for git-clang-format.py that provides user-friendly error messages.

This script wraps the git-clang-format.py script from Format.cmake and catches
the ValueError exception it raises when files need formatting. Instead of showing
a Python stacktrace, it displays a clear, actionable error message.
"""

import sys
import subprocess
from pathlib import Path

MIN_ARGS = 2


def main() -> int:
    """
    Main entry point for the wrapper script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # First argument should be the path to git-clang-format.py
    if len(sys.argv) < MIN_ARGS:
        print("Error: git-clang-format.py path not provided", file=sys.stderr)
        return 1

    git_clang_format_path = sys.argv[1]

    # Remaining arguments are passed to git-clang-format.py
    git_clang_format_args = sys.argv[2:]

    # Verify the script exists
    if not Path(git_clang_format_path).exists():
        print(f"Error: git-clang-format.py not found at {git_clang_format_path}", file=sys.stderr)
        return 1

    # Run git-clang-format.py
    cmd = [sys.executable, git_clang_format_path, *git_clang_format_args]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        # git-clang-format.py failed (likely due to formatting issues)

        # Display user-friendly error message
        print("\n" + "=" * 80, file=sys.stderr)
        print("❌ CODE FORMATTING CHECK FAILED", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("\nSome files in the repository do not meet formatting standards.", file=sys.stderr)
        print("\n" + "-" * 80, file=sys.stderr)
        print("To fix formatting issues, run:", file=sys.stderr)
        print("\n  cmake --build <build-dir> --target fix-format", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  cmake --build out/build/clang-debug --target fix-format", file=sys.stderr)
        print("-" * 80 + "\n", file=sys.stderr)

        return 1

    except Exception as e:
        print(f"\nUnexpected error running git-clang-format.py: {e}", file=sys.stderr)
        return 1
    else:
        # Success case
        print("✓ Code formatting check passed", file=sys.stderr)
        if result.stdout:
            print(result.stdout, end="")
        return 0


if __name__ == "__main__":
    sys.exit(main())
