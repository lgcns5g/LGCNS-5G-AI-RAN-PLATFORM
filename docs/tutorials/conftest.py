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

"""Pytest configuration for tutorial notebooks."""

import sys

import nbformat


def pytest_collection_modifyitems(items):
    """Modify test items to add output display hook."""
    for item in items:
        # Check if this is a parametrized notebook test
        is_notebook_test = (
            item.name.startswith("test_notebook[")
            and hasattr(item, "callspec")
            and "notebook_path" in item.callspec.params
        )

        if is_notebook_test:
            # Store original runtest
            original_runtest = item.runtest

            def runtest_with_output(self=item, _original=original_runtest):
                """Run test and display notebook outputs afterward."""
                # Execute the test
                _original()

                # After successful execution, display outputs
                try:
                    # Extract notebook path from test parameter
                    nb_path = self.callspec.params["notebook_path"]

                    # Read the notebook (now executed and saved)
                    with open(nb_path) as f:
                        nb = nbformat.read(f, as_version=4)

                    print(f"\n{'=' * 70}")
                    print(f"Notebook cell outputs: {nb_path.name}")
                    print("=" * 70)

                    for i, cell in enumerate(nb.cells, start=1):
                        if cell.cell_type == "code":
                            outputs = cell.get("outputs", [])

                            if outputs:
                                # Print cell number header
                                print(f"\n[Cell {i}]")

                                # Show all outputs
                                for output in outputs:
                                    output_type = output.get("output_type")

                                    if output_type == "stream":
                                        text = output.get("text", "")
                                        if isinstance(text, list):
                                            text = "".join(text)
                                        sys.stdout.write(text)
                                    elif output_type == "execute_result":
                                        data = output.get("data", {})
                                        if "text/plain" in data:
                                            text = data["text/plain"]
                                            if isinstance(text, list):
                                                text = "".join(text)
                                            print(text)
                                    elif output_type == "error":
                                        ename = output.get("ename", "Error")
                                        evalue = output.get("evalue", "")
                                        print(f"ERROR: {ename}: {evalue}")

                    print(f"\n{'=' * 70}\n")

                except Exception as e:  # noqa: BLE001
                    print(f"\nWarning: Could not display notebook outputs: {e}\n")

            # Replace the runtest method
            item.runtest = runtest_with_output
