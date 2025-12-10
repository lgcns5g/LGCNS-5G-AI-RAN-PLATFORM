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
#     "scapy>=2.6.1",
#     "orjson>=3.8.0",
#     "pyyaml>=6.0",
# ]
# ///

"""Integration test runner for fronthaul_app with ru_emulator.

This script orchestrates running ru_emulator (RU layer) and fronthaul_app (DU layer)
together, validating the O-RAN fronthaul interface between them.
"""

import argparse
import copy
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from detect_loopback import (
    check_and_set_permissions,
    detect_all_loopback_pairs,
    get_mellanox_ethernet_interfaces,
)


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


def get_runtime_config() -> dict[str, Any]:
    """Get runtime configuration from environment variables with defaults.

    Environment variables (optional - defaults used if not set):
    - TEST_CELLS: Number of cells (integer, default: 1)
    - TEST_SLOTS: Number of test slots (integer, default: 100)

    Returns
    -------
    dict[str, Any]
        Configuration with keys: cell_count, cells, pattern, test_slots

    """
    # Read from environment with sensible defaults
    cell_count_str = os.getenv("TEST_CELLS", "1")
    test_slots_str = os.getenv("TEST_SLOTS", "100")

    try:
        cell_count = int(cell_count_str)
    except ValueError as e:
        msg = f"TEST_CELLS must be an integer, got: {cell_count_str}"
        raise RuntimeError(msg) from e

    if cell_count < 1:
        msg = f"TEST_CELLS must be >= 1, got: {cell_count}"
        raise RuntimeError(msg)

    # Maximum cells supported by this integration test
    max_cells_supported = 1

    if cell_count > max_cells_supported:
        cell_word = "cell" if max_cells_supported == 1 else "cells"
        msg = (
            f"TEST_CELLS exceeds maximum supported by this integration test. "
            f"Got: {cell_count}, Maximum: {max_cells_supported}. "
            f"This fronthaul integration test supports up to {max_cells_supported} {cell_word}."
        )
        raise RuntimeError(msg)

    try:
        test_slots = int(test_slots_str)
    except ValueError as e:
        msg = f"TEST_SLOTS must be an integer, got: {test_slots_str}"
        raise RuntimeError(msg) from e

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
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments

    """
    parser = argparse.ArgumentParser(
        description="Integration test runner for fronthaul_app with ru_emulator"
    )
    parser.add_argument("ru_emulator_path", help="Path to ru_emulator executable")
    parser.add_argument("ru_emulator_workdir", help="Working directory for ru_emulator")
    parser.add_argument("ru_emulator_config_template", help="RU emulator config YAML template path")
    parser.add_argument("fronthaul_app_path", help="Path to fronthaul_app executable")
    parser.add_argument(
        "fapi_capture_dir",
        help="Directory containing FAPI capture file (script constructs filename)",
    )
    parser.add_argument(
        "--du-interface",
        help="DU-side interface (auto-detect if not specified)",
    )
    parser.add_argument(
        "--ru-interface",
        help="RU-side interface (auto-detect if not specified)",
    )
    return parser.parse_args()


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
            f"Using manually specified interfaces: {args.du_interface} <-> {args.ru_interface}"
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
    logger.info(f"Detected loopback pair: {pair.interface_a} <-> {pair.interface_b}")
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


def generate_ru_config(params: RuConfigParams, logger: logging.Logger) -> None:
    """Generate RU emulator config from template with substitutions.

    Parameters
    ----------
    params : RuConfigParams
        Configuration parameters including template path, output path, and MAC/PCI addresses
    logger : logging.Logger
        Logger for status messages

    """
    try:
        template = Path(params.template_path)
        config_text = template.read_text()
    except (OSError, FileNotFoundError) as e:
        msg = f"Failed to read template file {params.template_path}: {e}"
        raise RuntimeError(msg) from e

    # Simple string substitution
    config_text = config_text.replace("@RU_PCIE_ADDR_SHORT@", params.ru_pci)
    config_text = config_text.replace("@RU_MAC_ADDRESS@", params.ru_mac)
    config_text = config_text.replace("@DU_MAC_ADDRESS@", params.du_mac)

    try:
        output = Path(params.output_path)
        output.write_text(config_text)
    except OSError as e:
        msg = f"Failed to write config file {params.output_path}: {e}"
        raise RuntimeError(msg) from e

    logger.info(f"Generated RU config: {params.output_path}")


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
            raise RuntimeError(msg)  # noqa: TRY301

        reference_cell = template_cells[0]

        # Generate cell configs based on cell_count
        cell_configs = []
        for cell_idx in range(cell_count):
            # Deep copy reference cell and update name and VLAN
            cell_config = copy.deepcopy(reference_cell)
            cell_config["name"] = f"Cell{cell_idx + 1}"
            cell_config["vlan"] = 2 + cell_idx  # VLAN 2, 3, 4, ...
            cell_configs.append(cell_config)

        # Update config with generated cells
        ru_config["cell_configs"] = cell_configs
        config_data["ru_emulator"] = ru_config

        # Write final config
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated RU config with {cell_count} cells: {params.output_path}")

    except Exception as e:
        msg = f"Failed to generate RU emulator config: {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e


def run_ru_emulator(
    args: argparse.Namespace,
    cubb_home: str,
    config: dict[str, Any],
    config_path: str,
    logger: logging.Logger,
) -> subprocess.Popen:
    """Launch RU emulator in background.

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

    launch_pattern_arg = f"{pattern}_{cells}"

    cmd = [
        args.ru_emulator_path,
        launch_pattern_arg,
        "--channels",
        "PUSCH",
        "--config",
        Path(config_path).name,
        "--tv",
        f"{cubb_home}/testVectors/",
        "--lp",
        f"{cubb_home}/testVectors/multi-cell/",
    ]
    logger.info(f"Starting ru_emulator: {' '.join(cmd)}")
    logger.info(f"Cell configuration: {cells} ({config['cell_count']} cells)")
    logger.info(f"Working directory: {args.ru_emulator_workdir}")
    return subprocess.Popen(cmd, cwd=args.ru_emulator_workdir)  # noqa: S603


def run_fronthaul_app(
    args: argparse.Namespace,
    config: dict[str, Any],
    nic_pci: str,
    config_path: str,
    fapi_capture_file: str,
    logger: logging.Logger,
) -> int:
    """Launch fronthaul_app and wait for completion.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing fronthaul_app_path
    config : dict[str, Any]
        Runtime configuration from get_runtime_config()
    nic_pci : str
        DU-side NIC PCIe address (full form)
    config_path : str
        Path to ru_emulator_config.yaml file
    fapi_capture_file : str
        Path to FAPI capture file
    logger : logging.Logger
        Logger for status messages

    Returns
    -------
    int
        Exit code from fronthaul_app

    """
    # Build command with config file (new YAML-based interface)
    # Use test_slots from runtime config (environment variable)
    cmd = [
        args.fronthaul_app_path,
        "--nic",
        nic_pci,
        "--config",
        config_path,
        "--fapi-file",
        fapi_capture_file,
    ]

    # Only add --slots if not running indefinitely (0 = unlimited, omit flag entirely)
    if config["test_slots"] > 0:
        cmd.extend(["--slots", str(config["test_slots"])])

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

    # Calculate timeout based on test_slots: (slots * 0.01) + 60s overhead
    # If test_slots is 0 (indefinite), use no timeout
    test_slots = config["test_slots"]
    if test_slots == 0:
        subprocess_timeout = None
        timeout_msg = "no timeout (indefinite run)"
    else:
        subprocess_timeout = int(test_slots * 0.01) + 60
        timeout_msg = f"{subprocess_timeout}s"

    logger.info(f"Starting fronthaul_app: {' '.join(cmd)}")
    logger.info(f"Test slots: {test_slots}, timeout: {timeout_msg}")
    proc = subprocess.run(cmd, timeout=subprocess_timeout, check=False)  # noqa: S603
    logger.info(f"fronthaul_app exited with code: {proc.returncode}")
    return proc.returncode


def cleanup_ru_emulator(proc: subprocess.Popen | None, logger: logging.Logger) -> None:
    """Send shutdown signal to RU emulator.

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


def main() -> int:  # noqa: PLR0911, PLR0915
    """Run integration test with ru_emulator and fronthaul_app.

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

    # Get runtime configuration from environment
    try:
        config = get_runtime_config()
    except RuntimeError:
        logger.exception("Failed to get runtime configuration")
        return 1

    logger.info(f"Runtime configuration: {config}")

    # Verify CUBB_HOME is set (needed for ru_emulator only)
    cubb_home = os.getenv("CUBB_HOME")
    if not cubb_home:
        logger.error("CUBB_HOME environment variable not set")
        return 1

    logger.info(f"CUBB_HOME: {cubb_home}")

    # Construct FAPI capture filename dynamically based on runtime TEST_CELLS
    fapi_capture_file = (
        f"{args.fapi_capture_dir}/fapi_capture{config['pattern']}_{config['cells']}.fapi"
    )

    # Verify FAPI capture file exists (it should have been generated with matching cell count)
    if not Path(fapi_capture_file).exists():
        logger.error(f"FAPI capture file not found: {fapi_capture_file}")
        logger.error(f"Expected cell count: {config['cells']} ({config['cell_count']} cells)")
        logger.error("Run FAPI integration test first with matching TEST_CELLS")
        return 1

    logger.info(f"Using FAPI capture file: {fapi_capture_file}")

    # Check and set CAP_NET_RAW capability for loopback detection
    if not check_and_set_permissions(logger):
        logger.error("Failed to set CAP_NET_RAW capability - loopback detection will fail")
        logger.error("Please run: sudo setcap cap_net_raw=eip $(which python3)")
        return 1

    ru_proc = None
    fronthaul_exit_code = 1  # Default to failure

    try:
        # Detect or use specified interfaces
        du_iface, ru_iface = detect_or_use_interfaces(args, logger)

        # Get interface details (full PCI, short PCI, MAC)
        du_pci_full, _, du_mac = get_interface_info(du_iface)
        ru_pci_full, ru_pci_short, ru_mac = get_interface_info(ru_iface)

        logger.info(f"DU interface: {du_iface} (PCIe: {du_pci_full}, MAC: {du_mac})")
        logger.info(f"RU interface: {ru_iface} (PCIe: {ru_pci_full}, MAC: {ru_mac})")

        # Generate RU emulator config with runtime cell count
        config_dir = Path(cubb_home) / "cuPHY-CP" / "ru-emulator" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "ru_emulator_config.yaml"

        generate_ru_config_with_cells(
            RuConfigParams(
                template_path=args.ru_emulator_config_template,
                output_path=str(config_path),
                ru_pci=ru_pci_short,
                ru_mac=ru_mac,
                du_mac=du_mac,
            ),
            config["cell_count"],
            logger,
        )

        # Launch ru_emulator in background
        ru_proc = run_ru_emulator(args, cubb_home, config, str(config_path), logger)

        # Wait for ru_emulator initialization
        logger.info("Waiting for ru_emulator initialization...")
        time.sleep(8)

        # Launch fronthaul_app and wait for completion
        fronthaul_exit_code = run_fronthaul_app(
            args, config, du_pci_full, str(config_path), fapi_capture_file, logger
        )

    except subprocess.TimeoutExpired:
        logger.exception("fronthaul_app timed out")
        fronthaul_exit_code = 1

    except (OSError, subprocess.SubprocessError):
        logger.exception("Failed to run integration test")
        fronthaul_exit_code = 1

    except RuntimeError:
        logger.exception("Configuration error")
        fronthaul_exit_code = 1

    finally:
        # Clean up ru_emulator
        cleanup_ru_emulator(ru_proc, logger)

    # Exit with error if fronthaul_app failed or if any warnings/errors occurred
    if fronthaul_exit_code != 0:
        logger.error(f"Integration test failed with exit code: {fronthaul_exit_code}")
        return fronthaul_exit_code

    if warning_error_handler.has_warning_or_error():
        logger.error(
            "Integration test completed with warnings or errors - exiting with error code 1"
        )
        return 1

    logger.info("Integration test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
