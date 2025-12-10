#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201
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
Copyright header management utilities for NVIDIA Aerial Framework.

This module provides functions for detecting, validating, and fixing copyright headers
in source files to ensure SPDX compliance.

Exit codes:
  0: All files are compliant
  1: Violations found (check mode) or files fixed (fix mode reports via exceptions)
  2: Script error
"""

import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path

# Constants
# Similarity threshold for fuzzy matching copyright headers
FUZZY_MATCH_SIMILARITY_THRESHOLD = 0.85

# SPDX Headers
SPDX_HEADER_C_STYLE = """/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

SPDX_HEADER_HASH_STYLE = (
    "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "#\n"
    '# Licensed under the Apache License, Version 2.0 (the "License");\n'
    "# you may not use this file except in compliance with the License.\n"
    "# You may obtain a copy of the License at\n"
    "#\n"
    "# http://www.apache.org/licenses/LICENSE-2.0\n"
    "#\n"
    "# Unless required by applicable law or agreed to in writing, software\n"
    '# distributed under the License is distributed on an "AS IS" BASIS,\n'
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License.\n"
)


# Data Classes
class CopyrightStatus(Enum):
    """Status of copyright header in a file."""

    COMPLIANT = "compliant"
    MISSING = "missing"
    WRONG_FORMAT = "wrong_format"


@dataclass
class FileResult:
    """Result of processing a single file."""

    path: Path
    status: CopyrightStatus
    old_pattern: str | None = None
    fixed: bool = False
    error: str | None = None


# Core Functions
def get_spdx_header(file_path: Path) -> str:
    """
    Get the appropriate SPDX header based on file type.

    Args:
        file_path: Path to the file

    Returns
    -------
        SPDX header string (C-style or hash-style)
    """
    name = file_path.name.lower()
    suffix = file_path.suffix.lower()

    # Hash-style comments: Python, Shell, CMake, YAML, Docker
    if (
        suffix in [".py", ".sh", ".cmake", ".yaml", ".yml"]
        or name in ["cmakelists.txt", "dockerfile", "compose.yaml"]
        or name.startswith(("cmakelists", "dockerfile"))
        or name.endswith(".yaml.in")
    ):
        return SPDX_HEADER_HASH_STYLE

    # C-style comments: C, C++, CUDA
    return SPDX_HEADER_C_STYLE


def skip_first_line(content: str) -> str:
    """Skip first line (shebang or notebook marker)."""
    lines = content.split("\n", 1)
    return lines[1] if len(lines) > 1 else ""


def skip_linter_directives(content: str) -> str:
    """Skip comment lines with linter directives."""
    while content.startswith(("# ruff:", "# noqa:", "# type:", "# pylint:")):
        content = skip_first_line(content)
    return content


def extract_shebang(content: str) -> tuple[str, str]:
    """Extract shebang line if present."""
    if content.startswith("#!"):
        lines = content.split("\n", 1)
        return lines[0] + "\n", lines[1] if len(lines) > 1 else ""
    return "", content


def extract_linter_directives(content: str) -> tuple[str, str]:
    """Extract linter directive lines after shebang."""
    directives = []
    remaining = content
    while remaining.startswith(("# ruff:", "# noqa:", "# type:", "# pylint:")):
        lines = remaining.split("\n", 1)
        directives.append(lines[0] + "\n")
        remaining = lines[1] if len(lines) > 1 else ""
    return "".join(directives), remaining


def extract_jupyter_directive(content: str) -> tuple[str, str]:
    """Extract Jupyter cell directive if present."""
    if content.startswith('# %% [raw] tags=["remove-cell"]'):
        lines = content.split("\n", 1)
        return lines[0] + "\n", lines[1] if len(lines) > 1 else ""
    return "", content


def normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy comparison by removing comment markers and extra whitespace."""
    # Remove common comment characters
    normalized = text.replace("/*", "").replace("*/", "").replace("*", "")
    normalized = normalized.replace("#", "").replace("//", "")
    # Collapse all whitespace to single spaces
    normalized = " ".join(normalized.split())
    return normalized.lower()


def fuzzy_match_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts using difflib."""
    norm1 = normalize_for_comparison(text1)
    norm2 = normalize_for_comparison(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def extract_first_comment_block(content: str) -> tuple[str, str]:
    """
    Extract first comment block from content.

    Stops at: blank line, non-comment line, or end of file.
    Returns: (comment_block, remaining_content)
    """
    content = content.lstrip()

    # C-style block comment /* ... */
    if content.startswith("/*"):
        end_idx = content.find("*/")
        if end_idx != -1:
            block = content[: end_idx + 2]
            remaining = content[end_idx + 2 :]
            return block, remaining
        return content, ""  # Unclosed comment, take all

    # Line comments (// or #)
    if content.startswith(("//", "#")):
        lines = content.split("\n")
        comment_char = "//" if content.startswith("//") else "#"
        block_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Blank line = end of block
            if not stripped:
                remaining = "\n".join(lines[i:])
                return "\n".join(block_lines), remaining

            # Non-comment line = end of block
            if not stripped.startswith(comment_char):
                remaining = "\n".join(lines[i:])
                return "\n".join(block_lines), remaining

            # Comment line = part of block
            block_lines.append(line)

        # Reached end of file
        return "\n".join(block_lines), ""

    # No comment at start
    return "", content


def _is_copyright_line(line: str) -> bool:
    """Check if a line contains copyright/license keywords."""
    line_lower = line.lower()
    copyright_keywords = [
        "copyright",
        "spdx-",
        "all rights reserved",
        "proprietary",
        "license agreement",
        "licensed under",
        "strictly prohibited",
    ]
    return any(kw in line_lower for kw in copyright_keywords)


def _find_documentation_boundary(
    lines: list[str], blank_line_idx: int, comment_char: str
) -> tuple[int, bool]:
    """
    Look ahead from blank separator to find documentation boundary.

    Returns
    -------
        Tuple of (end_line_index, found_documentation_flag)
    """
    for j in range(blank_line_idx + 1, len(lines)):
        next_stripped = lines[j].lstrip()

        # Skip additional blank comment lines (we want TRUE blank lines, not '#')
        if next_stripped in (comment_char, comment_char + " "):
            continue

        if next_stripped and next_stripped.startswith(comment_char):
            # Check if this is documentation (no copyright keywords)
            if not _is_copyright_line(next_stripped):
                # Found documentation - preserve from here (j is actual doc line index)
                return j, True
            # Found another copyright block - continue removing
            return 0, False
        if next_stripped:
            # Non-comment line found
            return j, False
    return 0, False


def remove_leading_comment_block(content: str) -> str:
    """Remove leading comment blocks (old copyright headers), recursively if multiple exist."""
    content = content.lstrip()

    # C-style block comment
    if content.startswith("/*"):
        end_idx = content.find("*/")
        if end_idx != -1:
            remaining = content[end_idx + 2 :].lstrip()
            if remaining.startswith(("/*", "//", "#")):
                return remove_leading_comment_block(remaining)
            return remaining

    # Line comments (// or #) - ENHANCED with blank-line boundary detection
    if content.startswith(("//", "#")):
        lines = content.split("\n")
        comment_char = "//" if content.startswith("//") else "#"

        end_line = 0
        found_documentation = False

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # Non-comment line: end of all comments
            if not stripped.startswith(comment_char) and stripped:
                end_line = i
                break

            # Blank comment line: check if it separates copyright from documentation
            if stripped in (comment_char, comment_char + " "):
                end_line, found_documentation = _find_documentation_boundary(lines, i, comment_char)
                if end_line > 0:
                    break

        # If end_line is still 0, file contains only copyright comments - remove all
        if end_line == 0:
            end_line = len(lines)

        remaining = "\n".join(lines[end_line:]).lstrip()
        # Only recurse if we didn't find documentation (might be another copyright block)
        if not found_documentation and remaining.startswith(("/*", "//", "#")):
            return remove_leading_comment_block(remaining)
        return remaining

    return content


def has_any_copyright_mention(content: str) -> bool:
    """Check if file mentions copyright in header section (first 1000 chars)."""
    return bool(re.search(r"Copyright\s*\(c\)", content[:1000], re.IGNORECASE))


def has_valid_spdx_header(content: str, file_path: Path) -> bool:
    """Check if content starts with valid SPDX header."""
    # Get expected header for this file type
    expected_header = get_spdx_header(file_path).strip()

    # Skip shebang, linter directives, and Jupyter directives
    check_content = content
    if content.startswith("#!"):
        _, check_content = extract_shebang(check_content)
    _, check_content = extract_linter_directives(check_content)
    _, check_content = extract_jupyter_directive(check_content)

    # Compare with expected header
    return check_content.strip().startswith(expected_header)


def check_file(file_path: Path) -> FileResult:
    """
    Check copyright status of a file (non-destructive).

    Args:
        file_path: Path to file to check

    Returns
    -------
        FileResult with status and details
    """
    try:
        content = file_path.read_text(encoding="utf-8")

        # Check for valid SPDX header (strict exact match)
        if has_valid_spdx_header(content, file_path):
            return FileResult(path=file_path, status=CopyrightStatus.COMPLIANT)

        # Not compliant - either missing or wrong format
        if has_any_copyright_mention(content):
            return FileResult(path=file_path, status=CopyrightStatus.WRONG_FORMAT)
        return FileResult(path=file_path, status=CopyrightStatus.MISSING)

    except Exception as e:  # noqa: BLE001 - Catch all file read errors gracefully
        return FileResult(path=file_path, status=CopyrightStatus.MISSING, error=str(e))


def fix_file(file_path: Path) -> FileResult:
    """
    Fix copyright header in a file using fuzzy matching (modifies file).

    Algorithm:
    1. Extract shebang, linter directives, and Jupyter directives
    2. Extract first comment block
    3. Fuzzy match against expected SPDX header
    4. If exact match (100%) → no-op
    5. If high similarity (≥FUZZY_MATCH_SIMILARITY_THRESHOLD) → replace block with correct SPDX
    6. If has copyright keywords but low similarity → replace block with correct SPDX
    7. Otherwise → prepend SPDX (first comment block replaced if present)

    Args:
        file_path: Path to file to fix

    Returns
    -------
        FileResult with status and details
    """
    try:
        content = file_path.read_text(encoding="utf-8")

        # Extract and preserve special lines
        shebang, remaining = extract_shebang(content)
        directives, remaining = extract_linter_directives(remaining)
        jupyter_directive, remaining = extract_jupyter_directive(remaining)

        # Extract first comment block
        first_block, rest_of_file = extract_first_comment_block(remaining)

        # Get expected SPDX header for this file type
        expected_spdx = get_spdx_header(file_path)

        # Fast path: Check for exact match first
        if first_block.strip() == expected_spdx.strip():
            return FileResult(path=file_path, status=CopyrightStatus.COMPLIANT, fixed=False)

        # Fuzzy match to decide what to do
        similarity = fuzzy_match_similarity(first_block, expected_spdx)

        # Check if first block looks like a copyright (has keywords)
        has_copyright_keywords = any(
            kw in first_block.lower()
            for kw in [
                "copyright",
                "spdx",
                "license",
                "all rights reserved",
                "proprietary",
                "confidential",
            ]
        )

        # Determine status based on detection
        if similarity >= FUZZY_MATCH_SIMILARITY_THRESHOLD or has_copyright_keywords:
            status = CopyrightStatus.WRONG_FORMAT
        else:
            status = CopyrightStatus.MISSING

        # Construct new content (same for all cases - replace first block with SPDX)
        new_content = (
            shebang + directives + jupyter_directive + expected_spdx + "\n" + rest_of_file.lstrip()
        )

        # Check if content actually changed
        if content == new_content:
            return FileResult(path=file_path, status=CopyrightStatus.COMPLIANT, fixed=False)

        # Write changes
        file_path.write_text(new_content, encoding="utf-8")

        return FileResult(path=file_path, status=status, fixed=True)

    except Exception as e:  # noqa: BLE001 - Catch all file I/O errors gracefully
        return FileResult(path=file_path, status=CopyrightStatus.MISSING, error=str(e), fixed=False)


# Reporting Functions
def _print_wrong_format_details(wrong_format: list[FileResult], mode: str) -> None:
    """Print detailed breakdown for wrong format files."""
    print("\n" + "-" * 80)
    print("FILES WITH WRONG FORMAT:")
    print("-" * 80)
    pattern_counts = {}
    for r in wrong_format:
        pattern_counts[r.old_pattern] = pattern_counts.get(r.old_pattern, 0) + 1

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"\n  {pattern}: {count} files")
        if mode == "check":
            # Show a few examples
            examples = [r.path for r in wrong_format if r.old_pattern == pattern][:3]
            for ex in examples:
                print(f"    - {ex}")


def _print_missing_files(missing: list[FileResult]) -> None:
    """Print missing files list."""
    print("\n" + "-" * 80)
    max_displayed = 10  # Maximum number of files to display
    print(f"FILES MISSING COPYRIGHT (showing first {max_displayed}):")
    print("-" * 80)
    for r in missing[:max_displayed]:
        print(f"  - {r.path}")
    if len(missing) > max_displayed:
        print(f"  ... and {len(missing) - max_displayed} more")


def print_report(results: list[FileResult], mode: str) -> None:
    """
    Print summary report of copyright checking/fixing.

    Args:
        results: List of FileResult objects
        mode: "check" or "fix"
    """
    print("\n" + "=" * 80)
    print(f"COPYRIGHT HEADER {'CHECK' if mode == 'check' else 'FIX'} REPORT")
    print("=" * 80)

    compliant = [r for r in results if r.status == CopyrightStatus.COMPLIANT]
    missing = [r for r in results if r.status == CopyrightStatus.MISSING]
    wrong_format = [r for r in results if r.status == CopyrightStatus.WRONG_FORMAT]
    fixed = [r for r in results if r.fixed]
    errors = [r for r in results if r.error]

    print(f"\nTotal files processed: {len(results)}")
    print(f"  ✓ Compliant:        {len(compliant)}")
    print(f"  ✗ Wrong format:     {len(wrong_format)}")
    print(f"  ✗ Missing:          {len(missing)}")
    if mode == "fix":
        print(f"  ✓ Fixed:            {len(fixed)}")
    if errors:
        print(f"  ✗ Errors:           {len(errors)}")

    # Detailed breakdown by pattern
    if wrong_format:
        _print_wrong_format_details(wrong_format, mode)

    if missing:
        _print_missing_files(missing)

    if errors:
        print("\n" + "-" * 80)
        print("ERRORS:")
        print("-" * 80)
        for r in errors:
            print(f"  - {r.path}: {r.error}")

    if mode == "fix" and fixed:
        print("\n" + "-" * 80)
        print(f"✓ Successfully fixed {len(fixed)} files")
        print("-" * 80)

    print("\n" + "=" * 80)

    # Exit with error if there are issues in check mode
    if mode == "check" and (missing or wrong_format):
        sys.exit(1)


# Public API
def check_files(files: list[Path]) -> list[FileResult]:
    """
    Check copyright compliance for a list of files.

    Args:
        files: List of Path objects to check

    Returns
    -------
        List of FileResult objects
    """
    results = []
    for file_path in files:
        result = check_file(file_path)
        results.append(result)
    return results


def fix_files(files: list[Path]) -> list[FileResult]:
    """
    Fix copyright headers for a list of files.

    Args:
        files: List of Path objects to fix

    Returns
    -------
        List of FileResult objects
    """
    results = []
    for file_path in files:
        result = fix_file(file_path)
        results.append(result)
    return results
