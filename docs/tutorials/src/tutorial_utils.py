# %% [raw] tags=["remove-cell"]
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

# %% [markdown]
# # Tutorial Utilities
#
# Shared helper functions for Aerial Framework tutorials.

# %%
"""Shared utilities for Aerial Framework tutorials."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def load_ran_env_file() -> bool:
    """Load .env.python file to set environment variables.

    Priority 1: Use RAN_ENV_PYTHON_FILE if set
    Priority 2: Search for .env.python in ran/py/

    This loads environment variables like MLIR_TRT_COMPILER_PATH that are
    needed for MLIR-TensorRT compilation.

    Returns
    -------
        True if env file was found and loaded, False otherwise.
    """
    # Priority 1: Check for explicit env file path
    env_file_path = os.environ.get("RAN_ENV_PYTHON_FILE")
    env_file: Path | None = None

    if env_file_path:
        env_file = Path(env_file_path)
        if not env_file.exists():
            print(f"⚠️  RAN_ENV_PYTHON_FILE points to non-existent file: {env_file}")
            env_file = None

    # Priority 2: Default to source directory
    if not env_file:
        project_root = get_project_root()
        env_file = project_root / "ran" / "py" / ".env.python"

    if not env_file.exists():
        print(f"⚠️  .env.python file not found at: {env_file}")
        return False

    # Load the .env file using python-dotenv
    load_dotenv(dotenv_path=env_file, override=False)
    return True


def show_output(result: subprocess.CompletedProcess, lines: int = 5):
    """Show last few lines of stdout/stderr."""
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-lines:]:
            print(f"  {line}")
    if result.returncode != 0 and result.stderr:
        print("Error output:")
        for line in result.stderr.strip().split("\n")[-lines:]:
            print(f"  {line}")


def is_running_in_docker() -> bool:
    """Check if running inside Docker container."""
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read() or "buildkit" in f.read()
    except (FileNotFoundError, PermissionError):
        return False


def get_project_root() -> Path:
    """Find project root by searching for CMakePresets.json."""
    if "AERIAL_FRAMEWORK_ROOT" in os.environ:
        return Path(os.environ["AERIAL_FRAMEWORK_ROOT"])

    current = Path.cwd()
    while current != current.parent:
        if (current / "CMakePresets.json").exists():
            return current
        current = current.parent

    print("❌ Could not locate project root: CMakePresets.json not found")
    sys.exit(1)


def run_container_command(cmd: str, container_name: str = "", **kwargs):
    """Run command in container (if already inside container, run directly)."""
    # Set default arguments
    run_kwargs = {"capture_output": True, "text": True, "check": False}
    run_kwargs.update(kwargs)

    in_docker = is_running_in_docker()
    if in_docker:
        full_cmd = cmd
    else:
        if not container_name:
            msg = "container_name required when not running in Docker"
            raise ValueError(msg)
        run_kwargs.pop("cwd", None)  # cwd is not needed when outside container
        # Use bash -l to run commands in login shell to verify proper environment setup
        full_cmd = f"docker exec {container_name} bash -l -c {shlex.quote(cmd)}"

    print(f"Running: {full_cmd}")
    return subprocess.run(shlex.split(full_cmd), **run_kwargs)


def get_container_env(var_name: str, container_name: str = "", default: str = "") -> str:
    """Get environment variable value from inside container."""
    in_docker = is_running_in_docker()
    if in_docker:
        return os.getenv(var_name, default)

    if not container_name:
        return default

    result = subprocess.run(
        ["docker", "exec", container_name, "bash", "-c", f"echo ${var_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        value = result.stdout.strip()
        return value if value else default
    return default


def check_container_running(container_name: str) -> None:
    """Check if Docker container is running and exit with instructions if not.

    Args:
        container_name: Name of the container to check
    """
    if is_running_in_docker():
        return  # Already inside container, no check needed

    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0 or result.stdout.strip() != "true":
        print(f"\n❌ Container '{container_name}' is not running.")
        print("\nPlease refer to the Getting Started tutorial to start the container.")
        sys.exit(1)


def check_mlir_trt_enabled() -> bool:
    """Check if MLIR-TRT is enabled via RAN_ENV_PYTHON_FILE or .env.python."""
    # Priority 1: Check for explicit env file path
    env_file_path = os.environ.get("RAN_ENV_PYTHON_FILE")
    if env_file_path:
        env_file = Path(env_file_path)
    else:
        # Priority 2: Default to source directory
        project_root = get_project_root()
        env_file = project_root / "ran" / "py" / ".env.python"

    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("ENABLE_MLIR_TRT="):
                    return line.split("=", 1)[1].strip() == "ON"

    return False


def require_mlir_trt() -> None:
    """Check if MLIR-TRT is enabled and exit with instructions if not."""
    if not check_mlir_trt_enabled():
        print("\n⚠️  MLIR-TensorRT is not enabled.")
        print("This tutorial requires MLIR-TensorRT support.")
        print("Please configure CMake with -DENABLE_MLIR_TRT=ON")
        print("or set RAN_ENV_PYTHON_FILE to point to a build with MLIR-TRT enabled.")
        print("\nSkipping notebook execution.")
        sys.exit(0)


def configure_cmake(build_dir: Path, preset: str | None = None) -> None:
    """Ensure CMake is configured for the build directory."""
    if preset is None:
        preset = build_dir.name

    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        return

    project_root = get_project_root()
    print(f"Configuring CMake with preset {preset}...")

    result = subprocess.run(
        ["cmake", "--preset", preset, "-DENABLE_CLANG_TIDY=OFF", "-DENABLE_IWYU=OFF"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        msg = f"Failed to configure CMake:\n{result.stderr}"
        raise RuntimeError(msg)

    print("✓ CMake configured")


def build_cmake_target(build_dir: Path, targets: str | list[str]) -> None:
    """Build one or more CMake targets."""
    if isinstance(targets, str):
        targets = [targets]

    target_names = ", ".join(targets)
    print(f"Building {target_names}...")

    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", *targets],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        msg = f"Failed to build {target_names}:\n{result.stderr}"
        raise RuntimeError(msg)

    print(f"✓ {target_names} ready")


def check_network_devices(container_name: str) -> bool:
    """Check if networking devices are available inside the container and print status."""
    dev_vfio = get_container_env("DEV_VFIO", container_name, "/dev/null")
    dev_infiniband = get_container_env("DEV_INFINIBAND", container_name, "/dev/null")
    dev_gdrdrv = get_container_env("DEV_GDRDRV", container_name, "/dev/null")
    dev_hugepages = get_container_env("DEV_HUGEPAGES", container_name, "/dev/null")

    has_networking = (
        dev_vfio != "/dev/null"
        and dev_infiniband != "/dev/null"
        and dev_gdrdrv != "/dev/null"
        and dev_hugepages != "/dev/null"
    )

    if has_networking:
        print("✅ Networking devices detected")
    else:
        print("⏭️ Skipping network test (networking devices not available)")
        print("   This test requires NIC hardware (BF3) configured in loopback mode.")
        print("   To enable: ensure hardware devices are present on host and restart container.")
        print("   Required environment variables:")
        print("     DEV_VFIO, DEV_INFINIBAND, DEV_GDRDRV, DEV_HUGEPAGES")

    return has_networking


def parse_benchmark_output(stdout: str, benchmark_pattern: str = "bm_") -> list[str]:
    """Parse CTest benchmark output and extract formatted results.

    Extracts benchmark table from ctest output, removing CTest line prefixes
    (e.g., "67: ") for cleaner display.

    Parameters
    ----------
    stdout : str
        Raw stdout from ctest benchmark command
    benchmark_pattern : str, optional
        Pattern to match benchmark names (default: "bm_")

    Returns
    -------
        List of formatted benchmark result lines. Empty if parsing fails.
    """
    lines = stdout.split("\n")
    benchmark_lines = []
    in_benchmark_section = False

    for line in lines:
        # Remove CTest line prefix (e.g., "67: ") for cleaner output
        clean_line = re.sub(r"^\s*\d+:\s*", "", line)

        # Look for start of benchmark table (the header with dashes)
        if re.search(r"^-{50,}", clean_line):
            in_benchmark_section = True
            benchmark_lines.append(clean_line)
        # If we're in the benchmark section, capture the content
        elif in_benchmark_section:
            # Capture benchmark header, data rows, or stop at test status line
            if re.search(rf"(Benchmark\s+Time|{re.escape(benchmark_pattern)})", clean_line):
                benchmark_lines.append(clean_line)
            elif re.search(r"Test #\d+:.*ran\.phy_bench.*Passed", line):
                benchmark_lines.append(line)
                break  # Stop after the final test status

    return benchmark_lines


def check_nsys_profile(build_dir: Path, profile_name: str) -> None:
    """Check for nsys profile file and display info or debug message.

    Parameters
    ----------
    build_dir : Path
        Build directory containing nsys_results/
    profile_name : str
        Expected profile filename (without .nsys-rep extension)
    """
    nsys_dir = build_dir / "nsys_results"
    expected_file = nsys_dir / f"{profile_name}.nsys-rep"

    # Check for the expected file
    if expected_file.exists():
        file_size_mb = expected_file.stat().st_size / (1024 * 1024)
        print(f"  Profile: {expected_file}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  View with: nsys-ui {expected_file}")
    else:
        # List all .nsys-rep files in the directory for debugging
        print(f"⚠️  Expected profile file not found: {expected_file}")
        if nsys_dir.exists():
            nsys_files = list(nsys_dir.glob("*.nsys-rep"))
            if nsys_files:
                print(f"  Available profile files in {nsys_dir}:")
                for f in nsys_files:
                    print(f"    - {f.name}")
            else:
                print(f"  No .nsys-rep files found in {nsys_dir}")
        else:
            print(f"  Directory does not exist: {nsys_dir}")
