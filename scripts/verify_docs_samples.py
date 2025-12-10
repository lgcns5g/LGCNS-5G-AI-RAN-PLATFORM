#!/usr/bin/env -S uv run --script  # noqa: EXE001
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

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Verify API documentation code samples comply with documentation guidelines."""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Represents a rule violation in documentation samples."""

    rst_file: Path
    source_file: Path
    example_id: str
    line_start: int
    line_end: int
    rule: str
    details: str


@dataclass
class ExampleBlock:
    """Represents an example code block in a source file."""

    example_id: str
    start_line: int
    end_line: int
    content: list[str]


def find_literalinclude_directives(rst_file: Path) -> list[tuple[str, str, str, int | None]]:
    """Extract literalinclude directives from RST file.

    Returns list of tuples: (source_path, start_marker, end_marker, dedent_value)
    """
    directives = []
    content = rst_file.read_text()

    # Match literalinclude blocks with optional dedent
    pattern = (
        r"\.\.\s+literalinclude::\s+([^\n]+)\s+"
        r":language:[^\n]+\s+"
        r":start-after:\s+([^\n]+)\s+"
        r":end-before:\s+([^\n]+)"
        r"(?:\s+:dedent:\s+(\d+))?"
    )
    matches = re.finditer(pattern, content, re.MULTILINE)

    for match in matches:
        source_path = match.group(1).strip()
        start_marker = match.group(2).strip()
        end_marker = match.group(3).strip()
        dedent_value = int(match.group(4)) if match.group(4) else None
        directives.append((source_path, start_marker, end_marker, dedent_value))

    return directives


def extract_example_block(source_file: Path, start_marker: str, end_marker: str) -> ExampleBlock:
    """Extract code between example markers from source file."""
    lines = source_file.read_text().splitlines()
    start_line = None
    end_line = None

    for i, line in enumerate(lines, 1):
        if start_marker in line:
            start_line = i
        elif end_marker in line and start_line is not None:
            end_line = i
            break

    if start_line is None or end_line is None:
        msg = f"Markers not found: {start_marker} / {end_marker}"
        raise ValueError(msg)

    example_id = start_marker.replace("example-begin", "").strip()
    content = lines[start_line : end_line - 1]

    return ExampleBlock(example_id, start_line + 1, end_line - 1, content)


def check_no_assertions(block: ExampleBlock) -> list[str]:
    """Check for EXPECT/ASSERT statements inside example block."""
    violations = []
    patterns = [r"\bEXPECT_", r"\bASSERT_", r"\bASSERT_NO_THROW"]

    for line_num, line in enumerate(block.content, block.start_line):
        for pattern in patterns:
            if re.search(pattern, line):
                msg = f"Line {line_num}: Found assertion inside example block: {line.strip()}"
                violations.append(msg)

    return violations


def check_no_nolint(block: ExampleBlock) -> list[str]:
    """Check for NOLINT comments inside example block."""
    violations = []
    patterns = [r"//\s*NOLINT", r"//\s*NOLINTNEXTLINE"]

    for line_num, line in enumerate(block.content, block.start_line):
        for pattern in patterns:
            if re.search(pattern, line):
                msg = f"Line {line_num}: Found NOLINT comment inside example block: {line.strip()}"
                violations.append(msg)

    return violations


def check_no_iwyu_pragma(block: ExampleBlock) -> list[str]:
    """Check for IWYU pragma comments inside example block."""
    violations = []
    pattern = r"//\s*IWYU\s+pragma:"

    for line_num, line in enumerate(block.content, block.start_line):
        if re.search(pattern, line):
            msg = f"Line {line_num}: Found IWYU pragma inside example block: {line.strip()}"
            violations.append(msg)

    return violations


def check_dedent_matches_indentation(block: ExampleBlock, dedent_value: int | None) -> list[str]:
    """Check that RST dedent value matches actual code indentation."""
    violations = []

    # Calculate minimum indentation across all non-empty lines
    min_indent = float("inf")
    for line in block.content:
        if line.strip():  # Skip empty lines
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)

    # If all lines were empty, no violation
    if min_indent == float("inf"):
        return violations

    actual_indent = int(min_indent)

    if dedent_value is None:
        if actual_indent > 0:
            msg = (
                f"Code has {actual_indent} spaces of indentation but RST lacks :dedent: directive. "
                f"Add ':dedent: {actual_indent}' to the literalinclude directive."
            )
            violations.append(msg)
    elif dedent_value != actual_indent:
        msg = (
            f"RST specifies ':dedent: {dedent_value}' but code has "
            f"{actual_indent} spaces of indentation. "
            f"Update to ':dedent: {actual_indent}'."
        )
        violations.append(msg)

    return violations


def verify_documentation(docs_dir: Path) -> list[Violation]:
    """Verify all RST documentation in the specified directory."""
    violations = []
    rst_files = list(docs_dir.rglob("*.rst"))

    logger.info(f"Found {len(rst_files)} RST files to verify")

    for rst_file in rst_files:
        logger.debug(f"Checking: {rst_file.relative_to(docs_dir)}")

        # Find and verify literalinclude directives
        try:
            directives = find_literalinclude_directives(rst_file)

            for source_path, start_marker, end_marker, dedent_value in directives:
                # Resolve relative path from RST file location
                source_file = (rst_file.parent / source_path).resolve()

                if not source_file.exists():
                    violations.append(
                        Violation(
                            rst_file=rst_file,
                            source_file=Path(source_path),
                            example_id=start_marker,
                            line_start=0,
                            line_end=0,
                            rule="File not found",
                            details=f"Referenced source file does not exist: {source_path}",
                        )
                    )
                    continue

                # Extract and verify example block
                try:
                    block = extract_example_block(source_file, start_marker, end_marker)

                    # Run all checks
                    checks = [
                        ("No EXPECT/ASSERT inside examples", check_no_assertions),
                        ("No NOLINT inside examples", check_no_nolint),
                        ("No IWYU pragma inside examples", check_no_iwyu_pragma),
                    ]

                    for rule_name, check_func in checks:
                        issues = check_func(block)
                        violations.extend(
                            Violation(
                                rst_file=rst_file,
                                source_file=source_file,
                                example_id=block.example_id,
                                line_start=block.start_line,
                                line_end=block.end_line,
                                rule=rule_name,
                                details=issue,
                            )
                            for issue in issues
                        )

                    # Check dedent matches indentation
                    dedent_issues = check_dedent_matches_indentation(block, dedent_value)
                    violations.extend(
                        Violation(
                            rst_file=rst_file,
                            source_file=source_file,
                            example_id=block.example_id,
                            line_start=block.start_line,
                            line_end=block.end_line,
                            rule="Dedent matches indentation",
                            details=issue,
                        )
                        for issue in dedent_issues
                    )

                except ValueError as e:
                    violations.append(
                        Violation(
                            rst_file=rst_file,
                            source_file=source_file,
                            example_id=start_marker,
                            line_start=0,
                            line_end=0,
                            rule="Example extraction failed",
                            details=str(e),
                        )
                    )

        except Exception:
            logger.exception(f"Error processing {rst_file}")

    return violations


def print_report(violations: list[Violation]) -> None:
    """Print verification report."""
    if not violations:
        logger.info("✓ All documentation samples verified successfully!")
        return

    logger.warning(f"Found {len(violations)} violation(s):")

    # Group by RST file
    by_rst = {}
    for v in violations:
        if v.rst_file not in by_rst:
            by_rst[v.rst_file] = []
        by_rst[v.rst_file].append(v)

    for rst_file, file_violations in by_rst.items():
        logger.warning(f"\n{rst_file}")
        logger.warning("=" * 80)

        for v in file_violations:
            logger.warning(f"\n  Rule: {v.rule}")
            if v.example_id != "N/A":
                logger.warning(f"  Example: {v.example_id}")
                logger.warning(f"  Source: {v.source_file}")
                logger.warning(f"  Lines: {v.line_start}-{v.line_end}")
            logger.warning(f"  Details: {v.details}")


def main() -> None:
    """Run documentation verification."""
    parser = argparse.ArgumentParser(description="Verify API documentation code samples")
    parser.add_argument(
        "docs_dir",
        type=Path,
        nargs="?",
        default=Path("docs/api"),
        help="Documentation directory to verify (default: docs/api)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    if not args.docs_dir.exists():
        logger.error(f"Directory not found: {args.docs_dir}")
        sys.exit(1)

    violations = verify_documentation(args.docs_dir)
    print_report(violations)

    sys.exit(0 if not violations else 1)


if __name__ == "__main__":
    main()
