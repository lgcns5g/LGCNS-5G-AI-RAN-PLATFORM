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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///

"""Generate SVG files from Mermaid diagram source files using mermaid.ink API."""

from __future__ import annotations

import base64
import logging
import sys
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def generate_svg_from_mermaid(mermaid_file: Path, output_file: Path) -> None:
    """Generate SVG from mermaid file using mermaid.ink API.

    Args:
        mermaid_file: Path to input .mermaid file
        output_file: Path to output .svg file
    """
    logger.info("Generating %s from %s...", output_file.name, mermaid_file.name)

    # Read mermaid source
    mermaid_code = mermaid_file.read_text()

    # Encode to base64 for API
    encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()

    # Call mermaid.ink API
    url = f"https://mermaid.ink/svg/{encoded}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to generate SVG from %s", mermaid_file.name)
        sys.exit(1)

    # Write SVG output
    output_file.write_bytes(response.content)
    logger.info("✓ Generated %s", output_file)


def main() -> None:
    """Generate all mermaid diagrams in docs/figures/src/."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    project_root = Path(__file__).parent.parent
    src_dir = project_root / "docs" / "figures" / "src"
    output_dir = project_root / "docs" / "figures" / "generated"

    if not src_dir.exists():
        logger.error("Source directory not found: %s", src_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .mermaid files
    mermaid_files = sorted(src_dir.glob("*.mermaid"))

    if not mermaid_files:
        logger.warning("No .mermaid files found in %s", src_dir)
        return

    logger.info("Found %d mermaid diagram(s)", len(mermaid_files))

    # Generate SVGs
    for mermaid_file in mermaid_files:
        output_file = output_dir / f"{mermaid_file.stem}.svg"
        generate_svg_from_mermaid(mermaid_file, output_file)

    logger.info("✓ Successfully generated %d SVG(s)", len(mermaid_files))


if __name__ == "__main__":
    main()
