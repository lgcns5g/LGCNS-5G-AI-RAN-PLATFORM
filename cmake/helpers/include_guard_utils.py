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
Include guard verification utilities for NVIDIA Aerial Framework.

This module provides functions for verifying that C/C++/CUDA header files follow
the project's include guard standards:
1. All header files must use include guards (no #pragma once)
2. Include guards must use the correct prefix based on directory
3. Guard names must match the file path structure

Exit codes:
  0: All files comply with standards
  1: Violations found
  2: Script error
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Constants
# Directory-to-prefix mapping
PREFIX_MAPPING = {
    "framework/": "FRAMEWORK_",
    "ran/": "RAN_",
}

# Old prefixes that should no longer be used
OLD_PREFIXES = ["AERIAL_DSP_", "AERIAL_RAN_", "AERIAL_"]

# Patterns to scan for header files
DEFAULT_HEADER_PATTERNS = [
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


# Data Classes
class IncludeGuardStatus(Enum):
    """Status of include guard in a file."""

    COMPLIANT = "compliant"
    FIXED = "fixed"
    ERROR = "error"


@dataclass
class IncludeGuardViolation:
    """Represents a violation of include guard standards."""

    filepath: str
    issue: str
    line_num: int | None = None
    line_content: str | None = None

    def __str__(self) -> str:
        """Format violation for display."""
        msg = f"  {self.filepath}\n    Issue: {self.issue}"
        if self.line_num is not None:
            msg += f"\n    Line {self.line_num}: {self.line_content}"
        return msg


@dataclass
class IncludeGuardResult:
    """Result of fixing include guards in a file."""

    filepath: Path
    status: IncludeGuardStatus
    old_guard: str | None = None
    new_guard: str | None = None
    error: str | None = None


# Core Functions
def get_expected_prefix(filepath: str) -> str:
    """
    Get the expected prefix for include guard based on file path.

    Args:
        filepath: Relative file path from repository root

    Returns
    -------
        Expected prefix string (e.g., "FRAMEWORK_" or "RAN_")
    """
    for path_prefix, guard_prefix in PREFIX_MAPPING.items():
        if filepath.startswith(path_prefix):
            return guard_prefix
    return ""


def extract_guard_name(content: str) -> tuple[str, int] | None:
    """
    Extract the include guard name from file content.

    Args:
        content: File content as string

    Returns
    -------
        Tuple of (guard_name, line_number) if found, None otherwise
    """
    lines = content.split("\n")
    for i, line in enumerate(lines):
        # Look for #ifndef GUARD_NAME pattern
        match = re.match(r"^\s*#\s*ifndef\s+([A-Z_][A-Z0-9_]*)\s*$", line)
        if match:
            return (match.group(1), i + 1)
    return None


def verify_file(filepath: str) -> list[IncludeGuardViolation]:
    """
    Verify a single header file complies with include guard standards.

    Args:
        filepath: Path to header file to verify

    Returns
    -------
        List of IncludeGuardViolation objects (empty if file is compliant)
    """
    violations = []
    path = Path(filepath)

    if not path.exists():
        return violations  # Skip non-existent files

    try:
        content = path.read_text()
    except Exception as e:  # noqa: BLE001 - Catch all file read errors gracefully
        violations.append(IncludeGuardViolation(filepath, f"Failed to read file: {e}"))
        return violations

    lines = content.split("\n")

    # Check for #pragma once
    for i, line in enumerate(lines):
        if line.strip() == "#pragma once":
            violations.append(
                IncludeGuardViolation(
                    filepath, "Uses #pragma once instead of include guards", i + 1, line.strip()
                )
            )
            return violations  # No need to check further

    # Check for old prefixes in include guards (only check first 50 lines for guard)
    for i, line in enumerate(lines[:50]):  # Include guards are typically at the top
        match = re.match(r"^\s*#\s*ifndef\s+([A-Z_][A-Z0-9_]*)\s*$", line)
        if match:
            guard_name = match.group(1)
            if any(guard_name.startswith(old_prefix) for old_prefix in OLD_PREFIXES):
                violations.append(
                    IncludeGuardViolation(
                        filepath,
                        f"Uses old prefix ({', '.join(OLD_PREFIXES)}) in include guard",
                        i + 1,
                        line.strip(),
                    )
                )
            break  # Only check the first #ifndef (the guard)

    # Check if guard uses correct prefix
    expected_prefix = get_expected_prefix(filepath)
    if expected_prefix:
        actual_guard = extract_guard_name(content)
        if actual_guard:
            guard_name, line_num = actual_guard
            if not guard_name.startswith(expected_prefix):
                violations.append(
                    IncludeGuardViolation(
                        filepath,
                        f"Guard must start with '{expected_prefix}', found '{guard_name}'",
                        line_num,
                        f"#ifndef {guard_name}",
                    )
                )
        else:
            violations.append(
                IncludeGuardViolation(
                    filepath, "No include guard found (missing #ifndef directive)"
                )
            )

    return violations


def generate_guard_name(filepath: str) -> str:
    """
    Generate the correct include guard name for a file path.

    Args:
        filepath: Relative file path from repository root

    Returns
    -------
        Include guard name (e.g., "FRAMEWORK_CORE_TYPES_HPP")
    """
    # Get the expected prefix
    prefix = get_expected_prefix(filepath)

    # Convert path to guard name: replace / and . with _, uppercase
    # Remove the prefix directory from the path
    path_without_prefix = filepath
    for path_prefix in PREFIX_MAPPING:
        if filepath.startswith(path_prefix):
            path_without_prefix = filepath[len(path_prefix) :]
            break

    # Convert to guard name format
    guard_name = path_without_prefix.replace("/", "_").replace(".", "_").upper()

    return prefix + guard_name


def fix_file(filepath: Path) -> IncludeGuardResult:  # noqa: PLR0912, PLR0915
    """
    Fix include guards in a header file.

    Args:
        filepath: Path to header file to fix

    Returns
    -------
        IncludeGuardResult with status and details
    """
    try:
        content = filepath.read_text()
        lines = content.split("\n")

        # Check if already compliant
        violations = verify_file(str(filepath))
        if not violations:
            return IncludeGuardResult(
                filepath=filepath,
                status=IncludeGuardStatus.COMPLIANT,
            )

        # Generate correct guard name
        correct_guard = generate_guard_name(str(filepath))

        # Extract old guard if exists
        old_guard_info = extract_guard_name(content)
        old_guard = old_guard_info[0] if old_guard_info else None

        # Find where copyright header ends (if exists)
        # Look for SPDX header or old copyright patterns
        copyright_end_line = 0
        in_copyright = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Detect start of copyright block
            if "SPDX-FileCopyrightText" in stripped or "Copyright" in stripped:
                in_copyright = True
            # Detect end of C-style copyright block
            if in_copyright and "*/" in stripped:
                copyright_end_line = i + 1
                break
            # Detect end of hash-style copyright block (empty line after comments)
            if in_copyright and stripped and not stripped.startswith("#"):
                copyright_end_line = i
                break

        # Process file content: remove pragma once and include guards while preserving macros
        max_trailing_lines = 3  # Maximum non-empty lines allowed after guard endif

        cleaned_lines = []
        i = 0
        found_include_guard = False
        guard_name = None
        in_guard_header = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip #pragma once
            if stripped == "#pragma once":
                i += 1
                continue

            # Detect the first #ifndef (likely the include guard)
            if not found_include_guard and stripped.startswith("#ifndef"):
                match = re.match(r"^\s*#\s*ifndef\s+([A-Z_][A-Z0-9_]*)\s*$", stripped)
                if match:
                    guard_name = match.group(1)
                    found_include_guard = True
                    in_guard_header = True
                    i += 1
                    continue

            # Skip the #define that immediately follows the include guard #ifndef
            if in_guard_header and stripped.startswith("#define"):
                match = re.match(r"^\s*#\s*define\s+([A-Z_][A-Z0-9_]*)\s*$", stripped)
                if match and match.group(1) == guard_name:
                    in_guard_header = False
                    i += 1
                    continue
                # If #define doesn't match guard name, it's a macro - keep it
                in_guard_header = False

            # Skip the closing #endif for the include guard (at the end of file)
            if stripped.startswith("#endif") and guard_name:
                # Check if this is the final endif (near end of file)
                remaining_content = [line_text for line_text in lines[i + 1 :] if line_text.strip()]
                if len(remaining_content) <= max_trailing_lines:
                    # This is likely the closing guard endif
                    i += 1
                    continue

            cleaned_lines.append(line)
            i += 1

        # Build new content with correct guards
        new_lines = []

        # Add copyright header if it exists
        if copyright_end_line > 0:
            new_lines.extend(lines[:copyright_end_line])
            if new_lines and new_lines[-1].strip():  # Add blank line after copyright
                new_lines.append("")

        # Add include guards
        new_lines.append(f"#ifndef {correct_guard}")
        new_lines.append(f"#define {correct_guard}")
        new_lines.append("")

        # Add cleaned content (skip copyright lines if already added)
        content_start = copyright_end_line if copyright_end_line > 0 else 0
        new_lines.extend(cleaned_lines[content_start:])

        # Add closing endif
        # Remove trailing empty lines first
        while new_lines and not new_lines[-1].strip():
            new_lines.pop()

        new_lines.append("")
        new_lines.append(f"#endif  // {correct_guard}")
        new_lines.append("")

        # Write back to file
        new_content = "\n".join(new_lines)
        filepath.write_text(new_content)

        return IncludeGuardResult(
            filepath=filepath,
            status=IncludeGuardStatus.FIXED,
            old_guard=old_guard,
            new_guard=correct_guard,
        )

    except Exception as e:  # noqa: BLE001 - Catch all file I/O errors gracefully
        return IncludeGuardResult(filepath=filepath, status=IncludeGuardStatus.ERROR, error=str(e))


# Reporting Functions
def print_report(violations: list[IncludeGuardViolation], _files_checked: int) -> int:
    """
    Print summary report of include guard verification.

    Args:
        violations: List of IncludeGuardViolation objects
        _files_checked: Total number of files checked (reserved for future use)

    Returns
    -------
        Exit code (0 for success, 1 for violations)
    """
    print("=" * 70)
    print("Include Guard Verification")
    print("=" * 70)
    print()

    if violations:
        print(f"❌ VIOLATIONS FOUND: {len(violations)}\n")
        print("-" * 70)
        for violation in violations:
            print(violation)
            print()
        print("-" * 70)
        print(f"\nTotal violations: {len(violations)}")
        print(f"Files checked: {_files_checked}")
        print(f"Files with violations: {len({v.filepath for v in violations})}")
        print("\nPlease fix the violations above.")
        return 1

    print("✅ All header files comply with include guard standards!")
    print(f"\nFiles checked: {_files_checked}")
    return 0


def print_fix_report(results: list[IncludeGuardResult]) -> None:
    """
    Print summary report of include guard fixing.

    Args:
        results: List of IncludeGuardResult objects
    """
    print("\n" + "=" * 70)
    print("Include Guard Fix Report")
    print("=" * 70)

    compliant = [r for r in results if r.status == IncludeGuardStatus.COMPLIANT]
    fixed = [r for r in results if r.status == IncludeGuardStatus.FIXED]
    errors = [r for r in results if r.status == IncludeGuardStatus.ERROR]

    print(f"\nTotal files processed: {len(results)}")
    print(f"  ✓ Already compliant:  {len(compliant)}")
    print(f"  ✓ Fixed:              {len(fixed)}")
    print(f"  ✗ Errors:             {len(errors)}")

    if fixed:
        print("\n" + "-" * 70)
        print("FIXED FILES:")
        print("-" * 70)
        max_displayed = 10  # Maximum number of files to display
        for r in fixed[:max_displayed]:
            if r.old_guard:
                print(f"  {r.filepath}")
                print(f"    {r.old_guard} → {r.new_guard}")
            else:
                print(f"  {r.filepath}: Added {r.new_guard}")
        if len(fixed) > max_displayed:
            print(f"\n  ... and {len(fixed) - max_displayed} more")

    if errors:
        print("\n" + "-" * 70)
        print("ERRORS:")
        print("-" * 70)
        for r in errors:
            print(f"  - {r.filepath}: {r.error}")

    print("\n" + "=" * 70)
    if fixed:
        print(f"✓ Successfully fixed {len(fixed)} files")
    else:
        print("✓ No files needed fixing")
    print("=" * 70)


# Public API
def check_files(files: list[Path]) -> list[IncludeGuardViolation]:
    """
    Check include guard compliance for a list of header files.

    Args:
        files: List of Path objects to check

    Returns
    -------
        List of IncludeGuardViolation objects
    """
    all_violations = []
    for file_path in files:
        violations = verify_file(str(file_path))
        all_violations.extend(violations)
    return all_violations


def fix_files(files: list[Path]) -> list[IncludeGuardResult]:
    """
    Fix include guards for a list of header files.

    Args:
        files: List of Path objects to fix

    Returns
    -------
        List of IncludeGuardResult objects
    """
    results = []
    for file_path in files:
        result = fix_file(file_path)
        results.append(result)
    return results
