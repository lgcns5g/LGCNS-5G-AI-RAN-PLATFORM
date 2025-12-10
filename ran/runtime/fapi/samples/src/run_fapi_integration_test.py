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
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///

"""Integration test runner for fapi_sample with test_mac.

This script orchestrates running test_mac (MAC layer) and fapi_sample (PHY layer)
together, validating the FAPI interface between them.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


class WarningErrorHandler(logging.Handler):
    """Custom logging handler to track WARNING and ERROR messages."""

    def __init__(self) -> None:
        super().__init__()
        self.has_warnings = False
        self.has_errors = False

    def emit(self, record: logging.LogRecord) -> None:
        """Track WARNING and ERROR level messages."""
        if record.levelno >= logging.ERROR:
            self.has_errors = True
        elif record.levelno >= logging.WARNING:
            self.has_warnings = True

    def has_warning_or_error(self) -> bool:
        """Return True if any WARNING or ERROR messages were logged."""
        return self.has_warnings or self.has_errors


def get_runtime_config() -> dict[str, Any]:
    """Get runtime configuration from environment variables with defaults.

    Environment variables (optional - defaults used if not set):
    - TEST_CELLS: Number of cells (integer, default: 1)
    - TEST_VECTOR: Test vector HDF5 filename (default: TVnr_7201_gNB_FAPI_s0.h5)

    Note: TEST_SLOTS is not used by this script. The number of slots is automatically
    determined from the launch pattern cycle length to minimize capture file size.

    Returns
    -------
    dict[str, Any]
        Configuration with keys: cell_count, cells, pattern, test_vector

    """
    # Read from environment with sensible defaults
    cell_count_str = os.getenv("TEST_CELLS", "1")
    test_vector = os.getenv("TEST_VECTOR", "TVnr_7201_gNB_FAPI_s0.h5")

    try:
        cell_count = int(cell_count_str)
    except ValueError as e:
        msg = f"TEST_CELLS must be an integer, got: {cell_count_str}"
        raise RuntimeError(msg) from e

    if cell_count < 1:
        msg = f"TEST_CELLS must be >= 1, got: {cell_count}"
        raise RuntimeError(msg)

    # Maximum cells supported by FAPI (FapiState::DEFAULT_MAX_CELLS)
    max_cells_supported = 20

    if cell_count > max_cells_supported:
        cell_word = "cell" if max_cells_supported == 1 else "cells"
        msg = (
            f"TEST_CELLS exceeds maximum supported cells. "
            f"Got: {cell_count}, Maximum: {max_cells_supported}. "
            f"The FAPI layer supports up to {max_cells_supported} {cell_word} "
            f"(DEFAULT_MAX_CELLS in fapi_state.hpp)."
        )
        raise RuntimeError(msg)

    # Derive values
    cells = f"{cell_count}C"  # e.g., "1C", "2C", "4C"
    pattern = "_fapi_sample"  # Fixed test pattern name

    return {
        "cell_count": cell_count,
        "cells": cells,
        "pattern": pattern,
        "test_vector": test_vector,
    }


def validate_test_vector(test_vector: str, cubb_home: Path) -> None:
    """Validate that test vector file exists.

    Parameters
    ----------
    test_vector : str
        Test vector filename (e.g., 'TVnr_7204_gNB_FAPI_s0.h5')
    cubb_home : Path
        CUBB_HOME directory path

    Raises
    ------
    RuntimeError
        If test vector file doesn't exist or is not an HDF5 file

    """
    # Check file extension
    if not test_vector.endswith(".h5"):
        msg = f"TEST_VECTOR must be an HDF5 file (.h5), got: {test_vector}"
        raise RuntimeError(msg)

    # Check file exists in test_data directory
    # CUBB_HOME contains testVectors with symlinks to source test data
    # This matches the pattern used in generate_launch_pattern
    test_data_dir = cubb_home / "testVectors"
    test_vector_path = test_data_dir / test_vector

    if not test_vector_path.exists():
        # List available test vectors for helpful error message
        available_vectors = []
        if test_data_dir.exists():
            available_vectors = sorted([f.name for f in test_data_dir.glob("*.h5")])

        msg = f"TEST_VECTOR file not found: {test_vector}\nExpected location: {test_vector_path}\n"
        if available_vectors:
            msg += "Available test vectors:\n"
            for tv in available_vectors:
                msg += f"  - {tv}\n"
        else:
            msg += f"Test data directory: {test_data_dir}\n"

        raise RuntimeError(msg)


def generate_launch_pattern(
    template_path: str,
    cubb_home: str,
    config: dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Generate launch pattern YAML from template with runtime cell count and test vector.

    Parameters
    ----------
    template_path : str
        Path to launch_pattern_fapi_sample.yaml.in template
    cubb_home : str
        CUBB_HOME directory path
    config : dict[str, Any]
        Runtime configuration from get_runtime_config()
        Must contain: cell_count, cells, pattern, test_vector
    logger : logging.Logger
        Logger instance

    Returns
    -------
    str
        Path to generated launch pattern file

    """
    cell_count = config["cell_count"]
    cells = config["cells"]
    pattern = config["pattern"]
    test_vector = config["test_vector"]

    logger.info(f"Generating launch pattern for {cell_count} cells using {test_vector}")

    # Read template
    template = Path(template_path)
    template_content = template.read_text()

    # Build Cell_Configs array - one entry per cell, all using same test vector
    cell_configs = [test_vector for _ in range(cell_count)]
    cell_configs_yaml = yaml.safe_dump(cell_configs, default_flow_style=True).strip()

    logger.info(f"Cell_Configs: {cell_configs}")

    # Build slot_config - defines cell configuration for slots with data
    slot_config = [
        {"cell_index": cell_idx, "channels": [test_vector]} for cell_idx in range(cell_count)
    ]

    # Convert slot_config to YAML format (indented properly for the template)
    slot_config_yaml = yaml.safe_dump(slot_config, default_flow_style=False, sort_keys=False)
    # Add proper indentation for YAML nesting (2 spaces for config: field)
    slot_config_yaml = "\n".join(
        "    " + line if line.strip() else line for line in slot_config_yaml.strip().split("\n")
    )

    logger.info(f"slot_config length: {len(slot_config)}")

    # Substitute placeholders in template
    output_content = template_content.replace("@CELL_CONFIGS@", cell_configs_yaml)
    output_content = output_content.replace("@SLOT_CONFIG@", "\n" + slot_config_yaml)

    # Write generated launch pattern to expected location
    # Note: test_mac expects "launch_pattern_" + pattern + "_" + cells where pattern has leading _
    # So with pattern="_fapi_sample" and cells="1C", we get "launch_pattern__fapi_sample_1C.yaml"
    output_filename = f"launch_pattern_{pattern}_{cells}.yaml"
    output_path = Path(cubb_home) / "testVectors" / "multi-cell" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_content)

    logger.info(f"Generated launch pattern: {output_path}")
    logger.info(f"  - Cell count: {cell_count}")
    logger.info(f"  - Test vector: {test_vector}")
    logger.info(f"  - Cell_Configs length: {len(cell_configs)}")

    return str(output_path)


def get_launch_pattern_cycle_length(launch_pattern_path: str, logger: logging.Logger) -> int:
    """Parse launch pattern YAML and return cycle length (max slot + 1).

    Parameters
    ----------
    launch_pattern_path : str
        Path to generated launch pattern YAML file
    logger : logging.Logger
        Logger instance

    Returns
    -------
    int
        Cycle length in slots (e.g., 20 for a pattern with slots 0-19)

    Raises
    ------
    RuntimeError
        If launch pattern file doesn't exist or cannot be parsed

    """
    pattern_file = Path(launch_pattern_path)
    if not pattern_file.exists():
        msg = f"Launch pattern file not found: {launch_pattern_path}"
        raise RuntimeError(msg)

    with pattern_file.open() as f:
        pattern = yaml.safe_load(f)

    sched = pattern.get("SCHED", [])
    if not sched:
        msg = "Empty SCHED section in launch pattern"
        raise RuntimeError(msg)

    max_slot = max(entry.get("slot", 0) for entry in sched if isinstance(entry, dict))
    cycle_length = max_slot + 1

    logger.info(
        f"Detected launch pattern cycle length: {cycle_length} slots (max slot: {max_slot})"
    )
    return cycle_length


def generate_test_mac_config(
    template_path: str,
    cubb_home: str,
    test_slots: int,
    logger: logging.Logger,
) -> str:
    """Generate test_mac config YAML from template with specified test_slots.

    Parameters
    ----------
    template_path : str
        Path to test_mac_fapi_sample.yaml.in template
    cubb_home : str
        CUBB_HOME directory path
    test_slots : int
        Number of slots to run (0 = indefinite)
    logger : logging.Logger
        Logger instance

    Returns
    -------
    str
        Path to generated test_mac config file

    """
    logger.info(f"Generating test_mac config with test_slots={test_slots}")

    # Read template
    template = Path(template_path)
    template_content = template.read_text()

    # Substitute placeholder
    output_content = template_content.replace("@TEST_SLOTS@", str(test_slots))

    # Write generated config to expected location
    output_filename = "test_mac_fapi_sample.yaml"
    output_path = Path(cubb_home) / "cuPHY-CP" / "testMAC" / "testMAC" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_content)

    logger.info(f"Generated test_mac config: {output_path}")
    logger.info(f"  - Test slots: {test_slots}")

    return str(output_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments

    """
    parser = argparse.ArgumentParser(
        description="Integration test runner for fapi_sample with test_mac"
    )
    parser.add_argument("test_mac_path", help="Path to test_mac executable")
    parser.add_argument(
        "test_mac_output_dir", help="Output directory for test_mac (where capture file is written)"
    )
    parser.add_argument("fapi_sample_path", help="Path to fapi_sample executable")
    parser.add_argument(
        "test_mac_config",
        help="Filename of test_mac config YAML (must be in $CUBB_HOME/cuPHY-CP/testMAC/testMAC/)",
    )
    parser.add_argument(
        "launch_pattern_template",
        help="Path to launch_pattern_fapi_sample.yaml.in template file",
    )
    parser.add_argument(
        "test_mac_config_template",
        help="Path to test_mac_fapi_sample.yaml.in template file",
    )
    parser.add_argument(
        "--overwrite-capture",
        action="store_true",
        help="Force regeneration of FAPI capture file even if it already exists",
    )
    return parser.parse_args()


def cleanup_test_mac(test_mac_proc: subprocess.Popen, logger: logging.Logger) -> None:
    """Clean up test_mac process by sending shutdown signals.

    Parameters
    ----------
    test_mac_proc : subprocess.Popen
        The test_mac process to clean up
    logger : logging.Logger
        Logger for status messages

    """
    if test_mac_proc and test_mac_proc.poll() is None:
        logger.info("Sending SIGUSR1 to test_mac for clean shutdown...")
        try:
            # Send SIGUSR1 twice because the NVIPC library's ipc_epoll_wait()
            # automatically retries when interrupted by a signal (EINTR).
            # First signal: sets run_flag, but epoll_wait retries immediately
            # Second signal: triggers exit(1) in handler since flag is already set
            test_mac_proc.send_signal(signal.SIGUSR1)
            time.sleep(0.5)  # Brief delay to ensure first signal is processed
            test_mac_proc.send_signal(signal.SIGUSR1)
            test_mac_proc.wait(timeout=5)
            logger.info("test_mac exited cleanly")
        except subprocess.TimeoutExpired:
            logger.warning("test_mac didn't exit after SIGUSR1, killing...")
            test_mac_proc.kill()
            test_mac_proc.wait()


def run_test_mac(
    args: argparse.Namespace, config: dict[str, Any], logger: logging.Logger
) -> subprocess.Popen:
    """Launch test_mac process in background.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    config : dict[str, Any]
        Runtime configuration from get_runtime_config()
    logger : logging.Logger
        Logger for status messages

    Returns
    -------
    subprocess.Popen
        The running test_mac process

    """
    pattern = config["pattern"]
    cells = config["cells"]

    test_mac_cmd = [
        args.test_mac_path,
        pattern,
        cells,
        "--config",
        args.test_mac_config,
    ]

    logger.info(f"Starting test_mac: {' '.join(test_mac_cmd)}")
    logger.info(f"Working directory: {args.test_mac_output_dir}")
    logger.info(f"Cell configuration: {cells} ({config['cell_count']} cells)")
    return subprocess.Popen(test_mac_cmd, cwd=args.test_mac_output_dir)  # noqa: S603


def run_fapi_sample(
    args: argparse.Namespace, cubb_home: str, config: dict[str, Any], logger: logging.Logger
) -> int:
    """Launch fapi_sample and wait for it to complete.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cubb_home : str
        Path to CUBB_HOME directory
    config : dict[str, Any]
        Runtime configuration from get_runtime_config()
    logger : logging.Logger
        Logger for status messages

    Returns
    -------
    int
        Exit code from fapi_sample

    """
    pattern = config["pattern"]
    cells = config["cells"]

    # fapi_sample needs full paths (it doesn't use CUBB_HOME path resolution)
    test_mac_config_full_path = f"{cubb_home}/cuPHY-CP/testMAC/testMAC/{args.test_mac_config}"
    # Note: filename has double underscore due to pattern having leading underscore
    launch_pattern_filename = f"launch_pattern_{pattern}_{cells}.yaml"
    launch_pattern_file = f"{cubb_home}/testVectors/multi-cell/{launch_pattern_filename}"

    # Construct capture filename dynamically based on runtime TEST_CELLS
    capture_file_path = f"{args.test_mac_output_dir}/fapi_capture{pattern}_{cells}.fapi"

    fapi_cmd = [
        args.fapi_sample_path,
        "--validate",
        "--launch_pattern_file",
        launch_pattern_file,
        "--test_mac_config_file",
        test_mac_config_full_path,
        "--capture_file",
        capture_file_path,
    ]

    logger.info(f"Starting fapi_sample: {' '.join(fapi_cmd)}")
    logger.info(f"Capture file: {capture_file_path}")
    fapi_proc = subprocess.run(fapi_cmd, timeout=60, check=False)  # noqa: S603
    fapi_exit_code = fapi_proc.returncode

    logger.info(f"fapi_sample exited with code: {fapi_exit_code}")
    return fapi_exit_code


def main() -> int:  # noqa: PLR0911, PLR0915, PLR0912
    """Run integration test with test_mac and fapi_sample.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)

    """
    logger = logging.getLogger(__name__)

    # Track if any warnings or errors occurred
    warning_error_handler = WarningErrorHandler()

    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add the warning/error handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(warning_error_handler)

    # Get runtime configuration from environment
    try:
        config = get_runtime_config()
    except RuntimeError:
        logger.exception("Failed to get runtime configuration")
        return 1

    logger.info(f"Runtime configuration: {config}")

    # Verify CUBB_HOME is set (required for test_mac to find files)
    cubb_home = os.getenv("CUBB_HOME")
    if not cubb_home:
        logger.error("CUBB_HOME environment variable not set")
        return 1

    logger.info(f"CUBB_HOME: {cubb_home}")

    # Validate test vector exists
    try:
        validate_test_vector(config["test_vector"], Path(cubb_home))
    except RuntimeError:
        logger.exception("Test vector validation failed")
        return 1

    # Verify template files exist
    launch_pattern_template = Path(args.launch_pattern_template)
    test_mac_config_template = Path(args.test_mac_config_template)

    if not launch_pattern_template.exists():
        logger.error(f"Launch pattern template not found: {launch_pattern_template}")
        return 1

    if not test_mac_config_template.exists():
        logger.error(f"Test MAC config template not found: {test_mac_config_template}")
        return 1

    # Generate launch pattern YAML with runtime cell count
    # (must be done first to parse cycle length)
    try:
        launch_pattern_path = generate_launch_pattern(
            str(launch_pattern_template), cubb_home, config, logger
        )
    except Exception:
        logger.exception("Failed to generate launch pattern")
        return 1

    # Parse launch pattern to determine cycle length
    try:
        cycle_length = get_launch_pattern_cycle_length(launch_pattern_path, logger)
    except RuntimeError:
        logger.exception("Failed to parse launch pattern cycle length")
        return 1

    # Always use one pattern cycle to minimize capture file size
    # fapi_sample runs with --validate and --capture_file, generating the capture file
    # Using one cycle provides sufficient data while preventing unbounded file growth
    logger.info(f"Configuring test_mac to run one pattern cycle ({cycle_length} slots)")

    # Generate test_mac config YAML with cycle length
    try:
        generate_test_mac_config(str(test_mac_config_template), cubb_home, cycle_length, logger)
    except Exception:
        logger.exception("Failed to generate test_mac config")
        return 1

    # Check if capture file already exists (unless regeneration is forced)
    # Note: capture filename also has pattern with leading underscore
    capture_file_path = (
        f"{args.test_mac_output_dir}/fapi_capture{config['pattern']}_{config['cells']}.fapi"
    )
    if not args.overwrite_capture and Path(capture_file_path).exists():
        logger.info(f"FAPI capture file already exists: {capture_file_path}")
        logger.info("Skipping test execution (use --overwrite-capture to force regeneration)")
        return 0

    test_mac_proc = None
    fapi_exit_code = 1  # Default to failure

    try:
        # Launch test_mac in background
        test_mac_proc = run_test_mac(args, config, logger)

        # Give test_mac time to initialize NVIPC
        time.sleep(8)

        # Launch fapi_sample and wait for it to complete
        fapi_exit_code = run_fapi_sample(args, cubb_home, config, logger)

    except subprocess.TimeoutExpired:
        logger.exception("fapi_sample timed out")
        fapi_exit_code = 1

    except (OSError, subprocess.SubprocessError):
        logger.exception("Failed to run integration test")
        fapi_exit_code = 1

    finally:
        # Clean up test_mac
        cleanup_test_mac(test_mac_proc, logger)

    # Exit with error if fapi_sample failed or if any warnings/errors occurred
    if fapi_exit_code != 0:
        logger.error(f"Integration test failed with exit code: {fapi_exit_code}")
        return fapi_exit_code

    if warning_error_handler.has_warning_or_error():
        logger.error(
            "Integration test completed with warnings or errors - exiting with error code 1"
        )
        return 1

    logger.info("Integration test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
