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

"""Build TRT engines and plugins, copy to runtime directory with architecture naming."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Set up logging with simple format."""
    logger = logging.getLogger("build_trt_engines")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("==> %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_cpu_arch() -> str:
    """Get standardized CPU architecture name."""
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if arch in ("aarch64", "arm64"):
        return "aarch64"
    raise RuntimeError(f"Unsupported CPU architecture: {arch}")


def get_gpu_arch(logger: logging.Logger) -> str:
    """Detect GPU architecture using nvidia-smi (uses first GPU)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error("Failed to detect GPU architecture using nvidia-smi")
        sys.exit(1)

    # Use only the first GPU's compute capability
    compute_cap = result.stdout.strip().split("\n")[0].replace(".", "")
    return f"sm{compute_cap}"


def run_cmd(cmd: list[str], logger: logging.Logger) -> None:
    """Run command and exit on failure."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        sys.exit(1)


def main() -> None:
    """Build TRT engines and copy artifacts with architecture naming."""
    logger = setup_logging()

    repo_root = Path(__file__).parent.parent
    cpu_arch = get_cpu_arch()
    gpu_arch = get_gpu_arch(logger)
    preset = "clang-release"
    build_dir = repo_root / f"out/build/{preset}"

    logger.info(f"CPU: {cpu_arch}, GPU: {gpu_arch}")

    # Configure and build
    run_cmd(
        ["cmake", "--preset", preset, "-DENABLE_CLANG_TIDY=OFF"],
        logger,
    )
    run_cmd(
        [
            "cmake",
            "--build",
            str(build_dir),
            "-t",
            "ran_trt_plugin",
            "py_ran_setup",
            "pusch_tests",
            "py_ran_test_pusch_inner_receiver_fixture",
        ],
        logger,
    )

    # Verify tests pass before copying artifacts
    logger.info("Running PUSCH tests to verify build...")
    run_cmd(
        ["ctest", "--preset", preset, "-R", "pusch_tests"],
        logger,
    )
    dest_dir = repo_root / "ran/runtime/pusch/engine"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy TRT plugin library
    plugin_src = build_dir / "ran/py/libran_trt_plugin.so"
    plugin_dst = dest_dir / f"libran_trt_plugin.{cpu_arch}.{gpu_arch}.so"
    logger.info(f"Copying {plugin_src.name} -> {plugin_dst.name}")
    shutil.copy2(plugin_src, plugin_dst)

    # Copy free energy filter engine
    engine_src = (
        build_dir
        / "ran/py/tests/phy/jax/pusch/pusch_inner_receiver"
        / "free_energy_filter/tensorrt_cluster_engine_data.trtengine"
    )
    engine_name = f"pusch_inner_receiver_free_energy_filter.{cpu_arch}.{gpu_arch}.trtengine"
    engine_dst = dest_dir / engine_name
    logger.info(f"Copying {engine_src.name} -> {engine_dst.name}")
    shutil.copy2(engine_src, engine_dst)

    logger.info("Build and copy complete!")


if __name__ == "__main__":
    main()
