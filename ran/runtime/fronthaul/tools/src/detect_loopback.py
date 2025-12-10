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
# ]
# ///

"""Mellanox Ethernet loopback cable detection tool.

Detection Process:
    1. Discovers Mellanox Ethernet interfaces via /sys/class/net by filtering
       for vendor ID 0x15b3 and Ethernet type (1)
    2. Tests all interface pairs for bidirectional loopback connectivity
    3. For each pair, crafts unique broadcast packets with random payloads
    4. Sends packets from interface A while sniffing on interface B
    5. Verifies reception and repeats in reverse direction (B -> A)
    6. Marks pair as loopback if both directions succeed
    7. Outputs results in JSON format with detected pairs and test metadata

Requires CAP_NET_RAW capability for raw socket access.
"""

import argparse
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import orjson
    from scapy.all import Ether, Raw, conf, sendp, sniff
except ImportError:
    logger = logging.getLogger(__name__)
    logger.exception("Missing required dependency")
    sys.exit(1)

# Mellanox PCI vendor ID
MELLANOX_VENDOR_ID = "0x15b3"
ETHERNET_TYPE = 1


class WarningErrorHandler(logging.Handler):
    """Track WARNING and ERROR level messages."""

    def __init__(self) -> None:
        super().__init__()
        self.has_issues = False

    def emit(self, record: logging.LogRecord) -> None:
        """Mark when warnings or errors occur."""
        if record.levelno >= logging.WARNING:
            self.has_issues = True


@dataclass
class InterfaceInfo:
    """Network interface information."""

    name: str
    mac: str
    pci: str


@dataclass
class LoopbackPair:
    """Detected loopback pair information."""

    interface_a: str
    interface_b: str
    bidirectional: bool


def get_mellanox_ethernet_interfaces() -> list[InterfaceInfo]:
    """Discover Mellanox Ethernet interfaces (excluding InfiniBand)."""
    interfaces = []
    sys_net_path = Path("/sys/class/net")

    if not sys_net_path.exists():
        return interfaces

    for iface_path in sys_net_path.iterdir():
        iface = iface_path.name

        # Skip loopback and virtual interfaces
        if iface in ["lo", "docker0"] or iface.startswith(("veth", "virbr", "br-")):
            continue

        try:
            # Check vendor ID (Mellanox = 0x15b3)
            vendor_id = (iface_path / "device" / "vendor").read_text().strip()
            if vendor_id != MELLANOX_VENDOR_ID:
                continue

            # Check interface type (1 = Ethernet, 32 = InfiniBand)
            if_type = int((iface_path / "type").read_text().strip())
            if if_type != ETHERNET_TYPE:
                continue

            # Get MAC address and PCI address
            mac = (iface_path / "address").read_text().strip()
            pci = (iface_path / "device").resolve().name

            interfaces.append(InterfaceInfo(name=iface, mac=mac, pci=pci))
        except (OSError, ValueError):
            continue

    return interfaces


def test_loopback_direction(
    iface_send: str, iface_recv: str, timeout: float, logger: logging.Logger
) -> bool:
    """Test if packets sent from iface_send are received on iface_recv."""
    payload = f"LOOPBACK_TEST_{secrets.token_hex(8)}_{iface_send}_to_{iface_recv}"
    logger.debug(f"  Testing {iface_send} -> {iface_recv}...")

    # Get source MAC address
    try:
        src_mac = Path(f"/sys/class/net/{iface_send}/address").read_text().strip()
    except OSError:
        # INFO log level because some interfaces may fail, just need to detect one loopback pair.
        logger.info(f"    Could not read MAC for {iface_send}")
        return False

    # Craft broadcast packet with unique payload
    packet = Ether(src=src_mac, dst="ff:ff:ff:ff:ff:ff") / Raw(load=payload.encode())

    # Use synchronous sniff in a thread for reliable packet capture
    packets_received = []
    sniffer_error = []

    def sniff_thread() -> None:
        try:
            packets_received.extend(
                sniff(iface=iface_recv, promisc=True, timeout=timeout, store=True)
            )
        except (OSError, PermissionError) as e:
            sniffer_error.append(e)

    # Start sniffer thread
    sniffer = threading.Thread(target=sniff_thread)
    sniffer.start()

    # Small delay to ensure sniffer is ready
    time.sleep(0.1)

    # Send packet
    try:
        sendp(packet, iface=iface_send, verbose=False)
    except (OSError, RuntimeError) as e:
        # INFO log level because some interfaces may fail, just need to detect one loopback pair.
        logger.info(f"    Error sending packet from {iface_send}: {e}")
        sniffer.join()
        return False

    # Wait for sniffer to finish
    sniffer.join()

    # Check for sniffer errors
    if sniffer_error:
        # INFO log level because some interfaces may fail, just need to detect one loopback pair.
        logger.info(f"    Error starting sniffer on {iface_recv}: {sniffer_error[0]}")
        return False

    # Check if our packet was received
    for pkt in packets_received:
        if Raw in pkt and payload.encode() in bytes(pkt[Raw]):
            logger.debug("    ✓ Received")
            return True

    logger.debug("    ✗ Not received")
    return False


def test_loopback_pair(
    iface_a: str, iface_b: str, timeout: float, logger: logging.Logger
) -> LoopbackPair | None:
    """Test bidirectional loopback connection between two interfaces."""
    logger.debug(f"\nTesting pair: {iface_a} <-> {iface_b}")

    # Test A -> B
    a_to_b = test_loopback_direction(iface_a, iface_b, timeout, logger)

    if not a_to_b:
        return None

    # Test B -> A
    b_to_a = test_loopback_direction(iface_b, iface_a, timeout, logger)

    if not b_to_a:
        return None

    # Both directions work - bidirectional loopback detected
    return LoopbackPair(
        interface_a=iface_a,
        interface_b=iface_b,
        bidirectional=True,
    )


def detect_all_loopback_pairs(
    interfaces: list[InterfaceInfo],
    timeout: float,
    logger: logging.Logger,
    *,
    detect_all: bool = False,
) -> list[LoopbackPair]:
    """Test all interface pairs and detect loopback connections."""
    loopback_pairs = []
    tested_pairs = set()

    n = len(interfaces)
    for i in range(n):
        for j in range(i + 1, n):
            iface_a = interfaces[i].name
            iface_b = interfaces[j].name

            # Track tested pairs to avoid duplicates
            pair_key = tuple(sorted([iface_a, iface_b]))
            if pair_key in tested_pairs:
                continue
            tested_pairs.add(pair_key)

            # Test the pair
            result = test_loopback_pair(iface_a, iface_b, timeout, logger)
            if result:
                loopback_pairs.append(result)
                # Stop after first detection unless --detect-all is specified
                if not detect_all:
                    logger.info(
                        "First loopback pair detected, stopping "
                        "(use --detect-all to find all pairs)"
                    )
                    return loopback_pairs

    return loopback_pairs


def generate_output(
    interfaces: list[InterfaceInfo],
    loopback_pairs: list[LoopbackPair],
    timeout: float,
    test_duration: float,
) -> dict[str, Any]:
    """Generate JSON output structure."""
    # Find unconnected interfaces
    connected_ifaces = set()
    for pair in loopback_pairs:
        connected_ifaces.add(pair.interface_a)
        connected_ifaces.add(pair.interface_b)

    unconnected = [iface.name for iface in interfaces if iface.name not in connected_ifaces]

    # Calculate total pairs tested
    n = len(interfaces)
    total_pairs = n * (n - 1) // 2

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "test_parameters": {"timeout_seconds": timeout, "packet_size": 64},
        "interfaces_tested": [
            {"name": iface.name, "mac": iface.mac, "pci": iface.pci} for iface in interfaces
        ],
        "loopback_pairs": [
            {
                "interface_a": pair.interface_a,
                "interface_b": pair.interface_b,
                "bidirectional": pair.bidirectional,
            }
            for pair in loopback_pairs
        ],
        "unconnected_interfaces": unconnected,
        "test_summary": {
            "total_interfaces": len(interfaces),
            "total_pairs_tested": total_pairs,
            "loopback_pairs_found": len(loopback_pairs),
            "test_duration_seconds": round(test_duration, 2),
        },
    }


def write_json_output(
    data: dict[str, Any], output_path: str | None, logger: logging.Logger
) -> None:
    """Write JSON output to file or stdout."""
    json_bytes = orjson.dumps(data, option=orjson.OPT_INDENT_2)

    if output_path:
        try:
            Path(output_path).write_bytes(json_bytes)
            logger.debug(f"Output written to {output_path}")
        except OSError:
            logger.exception("Failed to write output file")
    else:
        sys.stdout.buffer.write(json_bytes)
        sys.stdout.buffer.write(b"\n")


def check_and_set_permissions(logger: logging.Logger) -> bool:
    """Check if we have CAP_NET_RAW and attempt to set it if missing."""
    try:
        # Try to create a raw socket (requires CAP_NET_RAW)
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
        sock.close()
    except PermissionError:
        logger.info("Missing CAP_NET_RAW capability, attempting to set it...")
    except OSError as e:
        logger.warning(f"Error checking CAP_NET_RAW capability: {e}")
        return True
    else:
        return True

    try:
        real_python = Path(sys.executable).resolve()

        # Check if setcap is available
        setcap = shutil.which("setcap")
        if not setcap:
            logger.warning("setcap not found - cannot set CAP_NET_RAW capability")
            logger.error("Please run: sudo setcap cap_net_raw=eip $(which python3)")
            return False

        logger.info(f"Setting CAP_NET_RAW on {real_python}")

        # Try to set capability
        result = subprocess.run(  # noqa: S603
            [setcap, "cap_net_raw=eip", str(real_python)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to set capability: {result.stderr}")
            logger.error("Please run: sudo setcap cap_net_raw=eip $(which python3)")
            return False

        logger.info("Successfully set CAP_NET_RAW capability")
        logger.info("Re-executing script with new capabilities...")

        # Re-execute the script to pick up the new capability
        os.execv(sys.executable, [sys.executable, *sys.argv])  # noqa: S606

    except (OSError, subprocess.SubprocessError):
        logger.exception(
            "Error setting capability - please run: sudo setcap cap_net_raw=eip $(which python3)"
        )
        return False


def main() -> int:
    """Detect Mellanox loopback pairs and output results."""
    logger = logging.getLogger(__name__)
    warning_error_handler = WarningErrorHandler()

    parser = argparse.ArgumentParser(
        description="Detect Mellanox Ethernet interfaces connected via loopback cables"
    )

    parser.add_argument("--output", metavar="FILE", help="Output JSON file path (default: stdout)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Per-pair test timeout (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List interfaces without sending packets",
    )
    parser.add_argument(
        "--detect-all",
        action="store_true",
        help="Detect all loopback pairs (default: stop after first detection)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add the warning/error handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(warning_error_handler)

    # Disable Scapy warnings
    conf.verb = 0

    # Discover Mellanox Ethernet interfaces
    logger.info("Discovering Mellanox Ethernet interfaces...")

    interfaces = get_mellanox_ethernet_interfaces()

    if not interfaces:
        logger.error("No Mellanox Ethernet interfaces found")
        return 1

    logger.info(f"Found {len(interfaces)} Mellanox Ethernet interface(s)")
    for iface in interfaces:
        logger.debug(f"  - {iface.name} ({iface.mac}) @ {iface.pci}")

    # Dry-run mode: just output interface info
    if args.dry_run:
        logger.info("Dry-run mode: listing interfaces only")
        output = {
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "dry-run",
            "interfaces_discovered": [
                {"name": iface.name, "mac": iface.mac, "pci": iface.pci} for iface in interfaces
            ],
            "total_interfaces": len(interfaces),
        }
        write_json_output(output, args.output, logger)
        return 0 if not warning_error_handler.has_issues else 1

    # Check and set permissions for raw socket access
    if not check_and_set_permissions(logger):
        return 1

    # Run loopback detection
    mode_msg = "all pairs" if args.detect_all else "first pair only"
    logger.info(
        f"Starting loopback detection (timeout: {args.timeout}s per pair, mode: {mode_msg})..."
    )

    start_time = time.time()
    loopback_pairs = detect_all_loopback_pairs(
        interfaces, args.timeout, logger, detect_all=args.detect_all
    )
    test_duration = time.time() - start_time

    # Generate output
    output = generate_output(interfaces, loopback_pairs, args.timeout, test_duration)
    write_json_output(output, args.output, logger)

    # Print summary
    logger.info(f"Detection complete in {test_duration:.2f}s")
    logger.info(f"Loopback pairs found: {len(loopback_pairs)}")
    for pair in loopback_pairs:
        logger.info(f"  • {pair.interface_a} <-> {pair.interface_b}")

    if len(loopback_pairs) == 0:
        logger.error("No loopback pairs detected")

    return 1 if warning_error_handler.has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
