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
CLI script for clearing metadata from figure files (XML and SVG).

This removes host and agent attributes that can cause noise in diffs.
"""

import re
import sys
from pathlib import Path


def _process_file(fpath: Path, patterns: list[str]) -> bool:
    """Process file and return True if updated."""
    content = fpath.read_text(encoding="utf-8")
    new_content = content
    for pat in patterns:
        new_content = re.sub(pat, "", new_content)
    if content != new_content:
        print(f"Updating {fpath}")
        fpath.write_text(new_content, encoding="utf-8")
        return True
    return False

def _process_files(files: list[Path], patterns: list[str]) -> int:
    """Process files and return the number of files updated."""
    return sum(1 for fpath in files if _process_file(fpath, patterns))


def clear_metadata() -> None:
    """Clear metadata from XML and SVG figure files."""
    # 1) XML files: docs/figures/src/*.xml
    xml_files = list(Path("docs/figures/src").glob("*.xml"))

    # 2) SVG files: docs/figures/generated/*.svg
    svg_files = list(Path("docs/figures/generated").glob("*.svg"))

    xml_patterns = [r' host="[^"]*"', r' agent="[^"]*"']
    svg_patterns = [r" host=&quot;[^&]*&quot;", r" agent=&quot;[^&]*&quot;"]

    n_xml_files = _process_files(xml_files, xml_patterns)
    n_svg_files = _process_files(svg_files, svg_patterns)

    print(f"Finished. Updated {n_xml_files + n_svg_files} files.")


if __name__ == "__main__":
    try:
        clear_metadata()
    except KeyboardInterrupt:
        sys.exit(2)
    except Exception:
        sys.exit(1)
