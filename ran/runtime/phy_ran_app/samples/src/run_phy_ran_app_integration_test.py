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

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "scapy>=2.6.1",
#     "orjson>=3.8.0",
#     "pyyaml>=6.0",
# ]
# ///

"""Integration test runner for phy_ran_app with test_mac and ru_emulator.

This script orchestrates running all three processes:
- test_mac (MAC layer, FAPI secondary)
- phy_ran_app (PHY layer, FAPI primary + Fronthaul)
- ru_emulator (RU layer, O-RAN Fronthaul)

"""

import argparse
import copy
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Import from fronthaul tools for interface detection
# Resolve path relative to this script's location
_fronthaul_tools = (
    Path(__file__).resolve().parent.parent.parent.parent / "fronthaul" / "tools" / "src"
)
if not _fronthaul_tools.exists():
    raise ImportError(f"Fronthaul tools not found at: {_fronthaul_tools}")
sys.path.insert(0, str(_fronthaul_tools))
from detect_loopback import (  # noqa: E402
    check_and_set_permissions,
    detect_all_loopback_pairs,
    get_mellanox_ethernet_interfaces,
)


def is_arm_architecture() -> bool:
    """Check if running on ARM architecture (aarch64/arm64)."""
    machine = platform.machine().lower()
    return machine in ("aarch64", "arm64")


def has_isolated_cpus() -> bool:
    """Check if isolated CPUs are configured for real-time operation."""
    try:
        isolated_file = Path("/sys/devices/system/cpu/isolated")
        if not isolated_file.exists():
            return False

        content = isolated_file.read_text().strip()
        # Empty or "0" means no isolated CPUs
        return bool(content and content != "0")
    except (OSError, FileNotFoundError):
        return False


class WarningErrorHandler(logging.Handler):
    """Track WARNING and ERROR level messages."""

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


@dataclass
class RuConfigParams:
    """Parameters for RU configuration generation."""

    template_path: str
    output_path: str
    ru_pci: str
    ru_mac: str
    du_mac: str


def get_runtime_config() -> dict[str, Any]:
    """Get runtime configuration from environment variables with defaults.

    Environment variables (optional - defaults used if not set):
    - TEST_CELLS: Number of cells (integer, default: 1)
    - TEST_SLOTS: Number of slots to run (integer, default: 100)
    - TEST_VECTOR: Test vector HDF5 filename (default: TVnr_7201_gNB_FAPI_s0.h5)

    Returns
    -------
    dict[str, Any]
        Configuration with keys: cell_count, cells, pattern, test_slots, test_vector

    """
    # Read from environment with sensible defaults
    cell_count_str = os.getenv("TEST_CELLS", "1")
    test_slots_str = os.getenv("TEST_SLOTS", "100")
    test_vector = os.getenv("TEST_VECTOR", "TVnr_7201_gNB_FAPI_s0.h5")

    try:
        cell_count = int(cell_count_str)
    except ValueError as e:
        msg = f"TEST_CELLS must be an integer, got: {cell_count_str}"
        raise RuntimeError(msg) from e

    try:
        test_slots = int(test_slots_str)
    except ValueError as e:
        msg = f"TEST_SLOTS must be an integer, got: {test_slots_str}"
        raise RuntimeError(msg) from e

    if cell_count < 1:
        msg = f"TEST_CELLS must be >= 1, got: {cell_count}"
        raise RuntimeError(msg)

    # Maximum cells supported by TensorRT engine in phy_ran_app
    max_cells_supported = 1

    if cell_count > max_cells_supported:
        cell_word = "cell" if max_cells_supported == 1 else "cells"
        msg = (
            f"TEST_CELLS exceeds maximum supported cells. "
            f"Got: {cell_count}, Maximum: {max_cells_supported}. "
            f"The phy_ran_app TensorRT engine currently supports "
            f"only {max_cells_supported} {cell_word}."
        )
        raise RuntimeError(msg)

    if test_slots < 0:
        msg = f"TEST_SLOTS must be >= 0, got: {test_slots}"
        raise RuntimeError(msg)

    # Derive values
    cells = f"{cell_count}C"  # e.g., "1C", "2C", "4C"
    pattern = "_fapi_sample"  # Fixed test pattern name

    return {
        "cell_count": cell_count,
        "cells": cells,
        "pattern": pattern,
        "test_slots": test_slots,
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

    # Check file exists in testVectors directory
    test_vectors_dir = cubb_home / "testVectors"
    test_vector_path = test_vectors_dir / test_vector

    if not test_vector_path.exists():
        # List available test vectors for helpful error message
        available_vectors = []
        if test_vectors_dir.exists():
            available_vectors = sorted([f.name for f in test_vectors_dir.glob("*.h5")])

        msg = f"TEST_VECTOR file not found: {test_vector}\nExpected location: {test_vector_path}\n"
        if available_vectors:
            msg += "Available test vectors:\n"
            for tv in available_vectors:
                msg += f"  - {tv}\n"
        else:
            msg += f"Test vectors directory: {test_vectors_dir}\n"

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
    # Note: Filename matches what test_mac expects in its working directory
    output_filename = "test_mac_phy_ran_app.yaml"
    output_path = Path(cubb_home) / "cuPHY-CP" / "testMAC" / "testMAC" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_content)

    logger.info(f"Generated test_mac config: {output_path}")
    logger.info(f"  - Test slots: {test_slots}")

    return str(output_path)


def generate_ru_config_with_cells(
    params: RuConfigParams, cell_count: int, logger: logging.Logger
) -> None:
    """Generate RU emulator config from template with runtime cell count.

    Parameters
    ----------
    params : RuConfigParams
        Configuration parameters for RU emulator
    cell_count : int
        Number of cells to configure
    logger : logging.Logger
        Logger instance

    Raises
    ------
    RuntimeError
        If template processing fails

    """
    try:
        template = Path(params.template_path)
        template_content = template.read_text()

        # Substitute MAC addresses and PCIe address
        config_content = template_content.replace("@RU_PCIE_ADDR_SHORT@", params.ru_pci)
        config_content = config_content.replace("@RU_MAC_ADDRESS@", params.ru_mac)
        config_content = config_content.replace("@DU_MAC_ADDRESS@", params.du_mac)

        # Parse YAML to modify cell_configs section
        config_data = yaml.safe_load(config_content)

        # Get reference cell config (first cell from template)
        ru_config = config_data.get("ru_emulator", {})
        template_cells = ru_config.get("cell_configs", [])

        if not template_cells:
            msg = "Template must contain at least one cell configuration"
            raise RuntimeError(msg)

        reference_cell = template_cells[0]

        # Generate cell configurations based on cell_count
        cell_configs = []
        for cell_idx in range(cell_count):
            cell_config = copy.deepcopy(reference_cell)
            cell_config["name"] = f"Cell{cell_idx + 1}"
            # Assign VLAN IDs starting from 2 (Cell1=VLAN 2, Cell2=VLAN 3, etc.)
            cell_config["vlan"] = 2 + cell_idx
            cell_configs.append(cell_config)

        # Update config with generated cells
        ru_config["cell_configs"] = cell_configs
        config_data["ru_emulator"] = ru_config

        # Write output
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated RU config: {output_path}")
        logger.info(f"  - Cell count: {cell_count}")
        logger.info(f"  - VLAN IDs: {[c['vlan'] for c in cell_configs]}")

    except (OSError, FileNotFoundError, yaml.YAMLError) as e:
        msg = f"Failed to generate RU config: {e}"
        raise RuntimeError(msg) from e


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments

    """
    parser = argparse.ArgumentParser(
        description="Integration test runner for phy_ran_app with test_mac and ru_emulator"
    )
    # test_mac arguments (from fapi_sample pattern)
    parser.add_argument("test_mac_path", help="Path to test_mac executable")
    parser.add_argument("test_mac_workdir", help="Working directory for test_mac")
    # ru_emulator arguments (from fronthaul_app pattern)
    parser.add_argument("ru_emulator_path", help="Path to ru_emulator executable")
    parser.add_argument("ru_emulator_workdir", help="Working directory for ru_emulator")
    # phy_ran_app arguments (replaces fapi_sample + fronthaul_app)
    parser.add_argument("phy_ran_app_path", help="Path to phy_ran_app executable")
    # Template paths for runtime configuration
    parser.add_argument(
        "launch_pattern_template",
        help="Path to launch_pattern_fapi_sample.yaml.in template file",
    )
    parser.add_argument(
        "test_mac_config_template",
        help="Path to test_mac_fapi_sample.yaml.in template file",
    )
    parser.add_argument("ru_emulator_config_template", help="RU emulator config YAML template path")
    # Optional interface overrides
    parser.add_argument(
        "--du-interface",
        help="DU-side interface (auto-detect if not specified)",
    )
    parser.add_argument(
        "--ru-interface",
        help="RU-side interface (auto-detect if not specified)",
    )
    return parser.parse_args()


# Interface Detection (from fronthaul_app pattern)
def detect_or_use_interfaces(args: argparse.Namespace, logger: logging.Logger) -> tuple[str, str]:
    """Detect loopback pair or use manually specified interfaces.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    logger : logging.Logger
        Logger for status messages

    Returns
    -------
    tuple[str, str]
        DU interface name, RU interface name

    """
    if args.du_interface and args.ru_interface:
        logger.info(
            "Using manually specified interfaces: %s <-> %s",
            args.du_interface,
            args.ru_interface,
        )
        return args.du_interface, args.ru_interface

    logger.info("Auto-detecting loopback-connected interfaces...")
    interfaces = get_mellanox_ethernet_interfaces()
    if not interfaces:
        msg = "No Mellanox Ethernet interfaces found"
        raise RuntimeError(msg)

    loopback_pairs = detect_all_loopback_pairs(interfaces, timeout=1.0, logger=logger)
    if not loopback_pairs:
        msg = "No loopback pairs detected. Please specify --du-interface and --ru-interface"
        raise RuntimeError(msg)

    pair = loopback_pairs[0]
    logger.info("Detected loopback pair: %s <-> %s", pair.interface_a, pair.interface_b)
    return pair.interface_a, pair.interface_b


def get_interface_info(iface_name: str) -> tuple[str, str, str]:
    """Get PCIe address and MAC address from interface name.

    Parameters
    ----------
    iface_name : str
        Network interface name

    Returns
    -------
    tuple[str, str, str]
        PCIe address (full form), PCIe address (short form without 0000: prefix), MAC address

    Raises
    ------
    RuntimeError
        If the interface doesn't exist or is inaccessible

    """
    try:
        device_path = Path(f"/sys/class/net/{iface_name}/device").resolve()
        pci_addr = device_path.name
        pci_short = pci_addr.replace("0000:", "")

        mac_addr = Path(f"/sys/class/net/{iface_name}/address").read_text().strip()
        return pci_addr, pci_short, mac_addr  # noqa: TRY300
    except (OSError, FileNotFoundError) as e:
        msg = f"Failed to get info for interface {iface_name}: {e}"
        raise RuntimeError(msg) from e


# Process Management (merged from both patterns)
def cleanup_test_mac(test_mac_proc: subprocess.Popen | None, logger: logging.Logger) -> None:
    """Clean up test_mac process by sending shutdown signals (from fapi_sample pattern).

    Parameters
    ----------
    test_mac_proc : subprocess.Popen | None
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


def cleanup_ru_emulator(proc: subprocess.Popen | None, logger: logging.Logger) -> None:
    """Send shutdown signal to RU emulator (from fronthaul_app pattern).

    Parameters
    ----------
    proc : subprocess.Popen | None
        The ru_emulator process to clean up, or None if not started
    logger : logging.Logger
        Logger for status messages

    """
    if proc and proc.poll() is None:
        logger.info("Sending SIGTERM to ru_emulator...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
            logger.info("ru_emulator exited cleanly")
        except subprocess.TimeoutExpired:
            logger.warning("ru_emulator didn't exit, killing...")
            proc.kill()
            proc.wait()


def run_test_mac(
    args: argparse.Namespace, config: dict[str, Any], logger: logging.Logger
) -> subprocess.Popen:
    """Launch test_mac process in background (from fapi_sample pattern).

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
        "test_mac_phy_ran_app.yaml",  # Generated from template by generate_test_mac_config()
    ]

    logger.info("Starting test_mac: %s", " ".join(test_mac_cmd))
    logger.info("Working directory: %s", args.test_mac_workdir)
    logger.info("Config file: test_mac_phy_ran_app.yaml")
    return subprocess.Popen(test_mac_cmd, cwd=args.test_mac_workdir)  # noqa: S603


def run_ru_emulator(
    args: argparse.Namespace,
    cubb_home: str,
    config: dict[str, Any],
    config_path: str,
    logger: logging.Logger,
) -> subprocess.Popen:
    """Launch RU emulator in background (from fronthaul_app pattern).

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cubb_home : str
        Path to CUBB_HOME directory
    config : dict[str, Any]
        Runtime configuration from get_runtime_config()
    config_path : str
        Path to RU emulator config file
    logger : logging.Logger
        Logger for status messages

    Returns
    -------
    subprocess.Popen
        The running ru_emulator process

    """
    pattern = config["pattern"]
    cells = config["cells"]

    # RU emulator constructs launch pattern filename from positional args as:
    # "launch_pattern" + "_" + arg1 + "_" + arg2 + ... + ".yaml"
    # To match fapi_sample's "launch_pattern__fapi_sample_2C.yaml",
    # pass single arg: "_fapi_sample_2C" (pattern + "_" + cells)
    launch_pattern_arg = f"{pattern}_{cells}"  # e.g., "_fapi_sample_2C"

    cmd = [
        args.ru_emulator_path,
        launch_pattern_arg,  # e.g., "_fapi_sample_2C"
        "--channels",
        "PUSCH",
        "--config",
        Path(config_path).name,  # Config file basename (ru_emulator looks in its workdir)
        "--tv",
        f"{cubb_home}/testVectors/",
        "--lp",
        f"{cubb_home}/testVectors/multi-cell/",
    ]

    logger.info("Starting ru_emulator: %s", " ".join(cmd))
    logger.info("Working directory: %s", args.ru_emulator_workdir)
    logger.info("Config: %s", config_path)
    return subprocess.Popen(cmd, cwd=args.ru_emulator_workdir)  # noqa: S603


def run_phy_ran_app(
    args: argparse.Namespace,
    config: dict[str, Any],
    nic_pci: str,
    config_path: str,
    logger: logging.Logger,
) -> int:
    """Launch phy_ran_app and wait for completion (merged pattern).

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    config : dict[str, Any]
        Runtime configuration from get_runtime_config()
    nic_pci : str
        NIC PCIe address (full form, e.g., "0000:aa:00.1")
    config_path : str
        Path to phy_ran_app config file (fronthaul YAML)
    logger : logging.Logger
        Logger for status messages

    Returns
    -------
    int
        Exit code from phy_ran_app

    """
    cells = config["cells"]
    test_slots = config["test_slots"]

    # phy_ran_app combines both FAPI (from test_mac) and Fronthaul (to ru_emulator)
    cmd = [
        args.phy_ran_app_path,
        "--nic",
        nic_pci,
        "--config",
        config_path,
        "--expected-cells",
        cells.replace("C", ""),  # "2C" -> "2"
    ]

    # Only add --slots if not running indefinitely (0 = unlimited, omit flag entirely)
    if test_slots > 0:
        cmd.extend(["--slots", str(test_slots)])

    # Only add --validate flag if both ARM architecture and isolated CPUs
    # Validation requires ARM with real-time system configuration
    is_arm = is_arm_architecture()
    isolated_cpus = has_isolated_cpus()

    if is_arm and isolated_cpus:
        cmd.append("--validate")
        logger.info("ARM architecture with isolated CPUs detected - enabling validation")
    else:
        logger.info(
            f"Validation disabled - ARM: {is_arm}, Isolated CPUs: {isolated_cpus} (both required)"
        )

    # Calculate timeout based on test_slots: (slots * 0.01) + 60s overhead (from fronthaul pattern)
    # If test_slots is 0 (indefinite), use no timeout
    if test_slots == 0:
        subprocess_timeout = None
        timeout_msg = "no timeout (indefinite run)"
    else:
        subprocess_timeout = int(test_slots * 0.01) + 60
        timeout_msg = f"{subprocess_timeout}s"

    logger.info("Starting phy_ran_app: %s", " ".join(cmd))
    logger.info(f"Test slots: {test_slots}, timeout: {timeout_msg}")
    proc = subprocess.run(cmd, timeout=subprocess_timeout, check=False)  # noqa: S603
    logger.info("phy_ran_app exited with code: %d", proc.returncode)
    return proc.returncode


# Main Entry Point (merged from both patterns)
def main() -> int:  # noqa: PLR0915
    """Run integration test with test_mac, phy_ran_app, and ru_emulator.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)

    """
    logger = logging.getLogger(__name__)
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

    # Get runtime configuration from environment variables
    config = get_runtime_config()
    logger.info("Runtime configuration:")
    logger.info(f"  - Cells: {config['cells']} (count: {config['cell_count']})")
    logger.info(f"  - Pattern: {config['pattern']}")
    logger.info(f"  - Test slots: {config['test_slots']}")
    logger.info(f"  - Test vector: {config['test_vector']}")

    # Verify CUBB_HOME is set (required by test_mac and ru_emulator)
    cubb_home = os.getenv("CUBB_HOME")
    if not cubb_home:
        logger.error("CUBB_HOME environment variable not set")
        return 1

    logger.info("CUBB_HOME: %s", cubb_home)

    # Validate test vector exists
    try:
        validate_test_vector(config["test_vector"], Path(cubb_home))
    except RuntimeError:
        logger.exception("Test vector validation failed")
        return 1

    # Check and set CAP_NET_RAW capability for loopback detection (from fronthaul pattern)
    if not check_and_set_permissions(logger):
        logger.error("Failed to set CAP_NET_RAW capability - loopback detection will fail")
        logger.error("Please run: sudo setcap cap_net_raw=eip $(which python3)")
        return 1

    ru_proc = None
    test_mac_proc = None
    phy_ran_app_exit_code = 1  # Default to failure

    try:
        # Detect or use specified interfaces (from fronthaul pattern)
        du_iface, ru_iface = detect_or_use_interfaces(args, logger)

        # Get interface details (full PCI, short PCI, MAC)
        du_pci_full, _, du_mac = get_interface_info(du_iface)
        ru_pci_full, ru_pci_short, ru_mac = get_interface_info(ru_iface)

        logger.info("DU interface: %s (PCIe: %s, MAC: %s)", du_iface, du_pci_full, du_mac)
        logger.info("RU interface: %s (PCIe: %s, MAC: %s)", ru_iface, ru_pci_full, ru_mac)

        # Generate YAML configurations from templates BEFORE launching processes
        logger.info("Generating configurations from templates...")

        # 1. Generate launch pattern YAML
        generate_launch_pattern(args.launch_pattern_template, cubb_home, config, logger)

        # 2. Generate test_mac config YAML
        generate_test_mac_config(
            args.test_mac_config_template, cubb_home, config["test_slots"], logger
        )

        # 3. Generate RU emulator config from template (with multi-cell support)
        config_dir = Path(cubb_home) / "cuPHY-CP" / "ru-emulator" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        ru_config_path = config_dir / "ru_emulator_config.yaml"
        generate_ru_config_with_cells(
            RuConfigParams(
                template_path=args.ru_emulator_config_template,
                output_path=str(ru_config_path),
                ru_pci=ru_pci_short,
                ru_mac=ru_mac,
                du_mac=du_mac,
            ),
            config["cell_count"],
            logger,
        )

        # Launch ru_emulator in background (from fronthaul pattern)
        ru_proc = run_ru_emulator(args, cubb_home, config, str(ru_config_path), logger)

        # Wait for ru_emulator initialization (from fronthaul pattern)
        logger.info("Waiting for ru_emulator initialization...")
        time.sleep(8)

        # Launch test_mac in background (from fapi_sample pattern)
        test_mac_proc = run_test_mac(args, config, logger)

        # Wait for test_mac NVIPC initialization (from fapi_sample pattern)
        logger.info("Waiting for test_mac initialization...")
        time.sleep(8)

        # Launch phy_ran_app and wait for completion (replaces both fapi_sample and fronthaul_app)
        # Use the SAME generated RU config as fronthaul_app does (EXACT pattern copy)
        phy_ran_app_exit_code = run_phy_ran_app(
            args, config, du_pci_full, str(ru_config_path), logger
        )

    except subprocess.TimeoutExpired:
        logger.exception("phy_ran_app timed out")
        phy_ran_app_exit_code = 1

    except (OSError, subprocess.SubprocessError):
        logger.exception("Failed to run integration test")
        phy_ran_app_exit_code = 1

    except RuntimeError:
        logger.exception("Configuration error")
        phy_ran_app_exit_code = 1

    finally:
        # Clean up processes in reverse order (from both patterns)
        cleanup_test_mac(test_mac_proc, logger)
        cleanup_ru_emulator(ru_proc, logger)

    # Exit with error if phy_ran_app failed or if any warnings/errors occurred
    if phy_ran_app_exit_code != 0:
        logger.error("Integration test failed with exit code: %d", phy_ran_app_exit_code)
        return phy_ran_app_exit_code

    if warning_error_handler.has_warning_or_error():
        logger.error(
            "Integration test completed with warnings or errors - exiting with error code 1"
        )
        return 1

    logger.info("Integration test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
