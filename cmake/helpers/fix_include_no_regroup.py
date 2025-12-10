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
Wrapper for fix_include that prevents consolidation of include groups.

This script reads the fix_include tool and patches it in memory to treat all #includes
as barriers, preventing group consolidation. This preserves blank line groupings between
include sections while still allowing add/remove operations.

Usage:
    python3 fix_include_no_regroup.py [fix_include args]

This wrapper reads IWYU output from stdin, just like fix_include.
"""

import shutil
import sys
from pathlib import Path

# Find fix_include in PATH
fix_include_path = shutil.which("fix_include")
if not fix_include_path:
    print("Error: fix_include not found in PATH", file=sys.stderr)
    sys.exit(1)

# Read the original fix_include script
try:
    with Path(fix_include_path).open("r", encoding="utf-8") as f:
        script_content = f.read()
except FileNotFoundError:
    print(f"Error: fix_include not found at {fix_include_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error reading fix_include: {e}", file=sys.stderr)
    sys.exit(1)

# Patch the _BARRIER_INCLUDES regex to match ALL includes
# Original: _BARRIER_INCLUDES = re.compile(r'^\s*#\s*include\s+(<linux/)')
# Patched:  _BARRIER_INCLUDES = re.compile(r'^\s*#\s*include\s+')
#
# This makes every #include act as a "barrier", preventing the consolidation
# of include groups that are separated by blank lines.
script_content = script_content.replace(
    "_BARRIER_INCLUDES = re.compile(r'^\\s*#\\s*include\\s+(<linux/)')",
    "_BARRIER_INCLUDES = re.compile(r'^\\s*#\\s*include\\s+')",
)

# Execute the patched script
exec(compile(script_content, fix_include_path, "exec"))
