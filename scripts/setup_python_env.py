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
#     "tomli>=2.0.0; python_version < '3.11'",
# ]
# ///

"""Python development environment manager."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport

from utils import run_command

# Color constants for terminal output
RED = "\033[0;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
YELLOW = "\033[0;33m"
NC = "\033[0m"  # No Color

# MLIR-TensorRT Configuration
MLIR_TRT_VERSION = "0.4.2.dev20251112"
MLIR_TRT_CUDA_VERSION_DEFAULT = "12.9"
MLIR_TRT_TENSORRT_VERSION_DEFAULT = "10.12"
MLIR_TRT_BASE_URL_DEFAULT = (
    "https://github.com/NVIDIA/TensorRT-Incubator/releases/"
    f"download/mlir-tensorrt-v{MLIR_TRT_VERSION.replace('.dev', 'dev')}"
)
MLIR_TRT_WHEELS_VERSION_DEFAULT = f"{MLIR_TRT_VERSION}+cuda12.trt1012"
MLIR_TRT_DOWNLOAD_DIR_DEFAULT = "ran/py/mlir-trt-downloads"  # Relative to project root


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": f"{BLUE}",
        "INFO": f"{BLUE}",
        "WARNING": f"{YELLOW}",
        "ERROR": f"{RED}",
        "CRITICAL": f"{RED}",
    }

    SYMBOLS: ClassVar[dict[str, str]] = {
        "DEBUG": "•",
        "INFO": "==>",
        "WARNING": "⚠",
        "ERROR": "✗",
        "CRITICAL": "✗",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color and symbol."""
        # Get the color and symbol for this log level
        color = self.COLORS.get(record.levelname, "")
        symbol = self.SYMBOLS.get(record.levelname, "")

        # Format the message
        message = super().format(record)

        # Add color and symbol
        if color and symbol:
            return f"{color}{symbol} {message}{NC}"
        return message


def setup_logging(*, verbose: bool = False) -> logging.Logger:
    """Set up logging with colored output."""
    # Set log level based on verbosity
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("setup_python_env")
    logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create formatter with timestamp including milliseconds
    formatter = ColoredFormatter("%(asctime)s.%(msecs)03d - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def build_download_url(base_url: str, filename: str) -> str:
    """Build download URL from base URL and filename.

    Args:
        base_url: Base URL for download (e.g., GitHub release URL)
        filename: URL-encoded filename to append

    Returns
    -------
        Complete download URL
    """
    return f"{base_url}/{filename}"


def download_wheels(
    base_url: str,
    wheel_names: list[str],
    output_dir: str,
    logger: logging.Logger,
) -> bool:
    """Download wheels using curl.

    Only downloads wheels that don't already exist in output_dir (caching behavior).

    Args:
        base_url: Base URL for wheel downloads
        wheel_names: List of wheel filenames to download
        output_dir: Directory to save wheels
        logger: Logger instance

    Returns
    -------
        True if all downloads successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wheels_to_download = [w for w in wheel_names if not (output_path / w).exists()]
    if not wheels_to_download:
        logger.info("All wheels already exist")
        return True

    for wheel in wheels_to_download:
        logger.info(f"Downloading {wheel}...")

        # URL-encode the wheel filename (handles '+' and other special characters)
        encoded_wheel = quote(wheel, safe="")
        wheel_url = build_download_url(base_url, encoded_wheel)

        # Build curl command
        curl_cmd = ["curl", "-Lf", "-o", f"{output_dir}/{wheel}", wheel_url]

        # Use subprocess.run directly to handle errors without exiting
        logger.debug(f"Running: {' '.join(curl_cmd)}")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=False)  # noqa: S603

        if result.returncode != 0:
            logger.error(f"Failed to download {wheel}")
            if result.stderr:
                logger.debug(f"curl stderr: {result.stderr}")
            return False

    return True


def download_and_extract_tarball(
    base_url: str,
    logger: logging.Logger,
    cache_dir: str,
    extract_dir: str,
    cuda_version: str,
    tensorrt_version: str,
) -> str:
    """Download and extract MLIR-TensorRT tarball.

    Args:
        base_url: Base URL for downloads
        logger: Logger instance
        cache_dir: Directory to cache downloaded tarball
        extract_dir: Directory to extract tarball
        cuda_version: CUDA version string (e.g., "12.9")
        tensorrt_version: TensorRT version string (e.g., "10.12")

    Returns
    -------
        Path to extracted directory

    Raises
    ------
        SystemExit: If download or extraction fails
    """
    # Map architecture to tarball naming convention
    arch = platform.machine()
    # Note: aarch64 tarballs use "arm64-sbsa-linux" format for CUDA 12.x
    if arch == "x86_64":
        arch_name = "x86_64"
        platform_suffix = "linux"
    else:
        arch_name = "arm64"
        # CUDA 12.x uses "sbsa-linux", CUDA 13.x uses just "linux"
        platform_suffix = "sbsa-linux" if cuda_version.startswith("12.") else "linux"

    # Construct tarball filename dynamically
    # GitHub release uses "mlir-tensorrt-compiler-" prefix for tarballs
    tarball_name = (
        f"mlir-tensorrt-compiler-{arch_name}-{platform_suffix}-cuda{cuda_version}-tensorrt{tensorrt_version}.tar.gz"
    )

    # Setup paths
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cached_tarball = cache_path / tarball_name

    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    compiler_path = extract_path / "mlir-tensorrt" / "bin" / "mlir-tensorrt-compiler"
    if compiler_path.exists():
        logger.info(f"MLIR-TensorRT tarball already extracted at: {extract_path}")
        return str(extract_path)

    # Download tarball if not cached
    if not cached_tarball.exists():
        logger.info(f"Downloading MLIR-TensorRT tarball for {arch}...")

        # URL-encode the tarball filename (handles special characters)
        encoded_tarball = quote(tarball_name, safe="")
        tarball_url = build_download_url(base_url, encoded_tarball)

        # Build curl command
        curl_cmd = ["curl", "-Lf", "-o", str(cached_tarball), tarball_url]

        # Download using run_command (handles errors and logging)
        run_command(curl_cmd, logger)
        logger.info(f"Tarball cached at: {cached_tarball}")
    else:
        logger.info(f"Using cached tarball: {cached_tarball}")

    # Extract tarball (secure method for Python 3.12+)
    # GitHub release format: extracts directly to ./mlir-tensorrt/ (no wrapper directory)
    logger.info(f"Extracting tarball to: {extract_path}")
    try:
        with tarfile.open(cached_tarball, "r:gz") as tar:
            tar.extractall(path=extract_path, filter="data")
    except tarfile.TarError:
        logger.exception("Failed to extract tarball")
        sys.exit(1)

    logger.info("Tarball extracted successfully")

    # Verify extraction
    if not compiler_path.exists():
        logger.error(f"Compiler not found at: {compiler_path}")
        sys.exit(1)

    logger.info("MLIR-TensorRT tarball extracted successfully!")
    return str(extract_path)


def download_mlir_trt_wheels(
    base_url: str,
    logger: logging.Logger,
    version: str,
    output_dir: str,
) -> None:
    """Download MLIR-TensorRT wheels for all supported architectures.

    Args:
        base_url: Base URL for downloads
        logger: Logger instance
        version: Version string in PEP 427 format (e.g., "0.4.2.dev20251112+cuda12.trt1012")
        output_dir: Output directory for wheels

    Notes
    -----
        Downloads wheels for both x86_64 and aarch64 to enable cross-platform lock files.
        Version string must be in PEP 427 format with '+' for local version identifiers.
    """
    if not base_url or not version:
        logger.error("Base URL and version are required for MLIR-TensorRT wheel downloads")
        sys.exit(1)

    logger.info("Setting up MLIR-TensorRT wheels for all architectures...")

    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Download wheels for all supported architectures (for cross-platform lock files)
    # Note: GitHub release includes runtime and compiler
    all_wheels = []
    for arch in ["x86_64", "aarch64"]:
        suffix = f"{version}-{py_ver}-{py_ver}-linux_{arch}.whl"
        wheels = [f"mlir_tensorrt_{pkg}-{suffix}" for pkg in ["runtime", "compiler"]]
        all_wheels.extend(wheels)
        logger.info(f"Downloading wheels for linux_{arch}...")

    # Download wheels
    if not download_wheels(base_url, all_wheels, output_dir, logger):
        logger.error("Failed to download MLIR-TensorRT wheels")
        sys.exit(1)

    logger.info("MLIR-TensorRT wheels ready for all architectures!")


def download_mlir_trt_wheels_standalone(
    logger: logging.Logger,
    *,
    base_url: str = MLIR_TRT_BASE_URL_DEFAULT,
    wheels_version: str = MLIR_TRT_WHEELS_VERSION_DEFAULT,
    download_dir: str = MLIR_TRT_DOWNLOAD_DIR_DEFAULT,
) -> None:
    """Download MLIR-TensorRT wheels only.

    Args:
        logger: Logger instance
        base_url: Base URL for downloads (defaults to MLIR_TRT_BASE_URL_DEFAULT)
        wheels_version: Wheel version (defaults to MLIR_TRT_WHEELS_VERSION_DEFAULT)
        download_dir: Download directory (defaults to MLIR_TRT_DOWNLOAD_DIR_DEFAULT)
    """
    logger.info("Downloading MLIR-TensorRT wheels...")
    logger.info(f"  Version: {wheels_version}")
    logger.info(f"  Download dir: {download_dir}")

    download_mlir_trt_wheels(base_url, logger, wheels_version, download_dir)


def download_mlir_trt_tarball_standalone(
    logger: logging.Logger,
    *,
    extract_dir: str,
    base_url: str = MLIR_TRT_BASE_URL_DEFAULT,
    cuda_version: str = MLIR_TRT_CUDA_VERSION_DEFAULT,
    tensorrt_version: str = MLIR_TRT_TENSORRT_VERSION_DEFAULT,
    download_dir: str = MLIR_TRT_DOWNLOAD_DIR_DEFAULT,
) -> str:
    """Download and extract MLIR-TensorRT tarball only.

    Args:
        logger: Logger instance
        extract_dir: Extraction directory (REQUIRED - from CMAKE_BINARY_DIR)
        base_url: Base URL (defaults to MLIR_TRT_BASE_URL_DEFAULT)
        cuda_version: CUDA version (defaults to MLIR_TRT_CUDA_VERSION_DEFAULT)
        tensorrt_version: TensorRT version (defaults to MLIR_TRT_TENSORRT_VERSION_DEFAULT)
        download_dir: Cache directory (defaults to MLIR_TRT_DOWNLOAD_DIR_DEFAULT)

    Returns
    -------
        Path to extracted directory
    """
    logger.info("Downloading and extracting MLIR-TensorRT tarball...")
    logger.info(f"  CUDA: {cuda_version}, TensorRT: {tensorrt_version}")
    logger.info(f"  Cache dir: {download_dir}")
    logger.info(f"  Extract dir: {extract_dir}")

    return download_and_extract_tarball(
        base_url, logger, download_dir, extract_dir, cuda_version, tensorrt_version
    )


def setup_mlir_tensorrt(
    logger: logging.Logger,
    *,
    tarball_extract_dir: str | None = None,
) -> None:
    """Set up MLIR-TensorRT: download wheels and tarball.

    Downloads MLIR-TensorRT artifacts to support PHY JAX:
    - Wheels: Downloaded for ALL architectures (x86_64, aarch64) to enable cross-platform lock files
    - Tarball: Downloaded for CURRENT architecture only and extracted to tarball_extract_dir

    All version configuration comes from module constants at top of file.
    Only tarball_extract_dir is required (build-specific path from CMake).

    Args:
        logger: Logger instance
        tarball_extract_dir: Where to extract tarball (from CMAKE_BINARY_DIR)

    Notes
    -----
        Wheels are downloaded but NOT installed. Installation is handled by py_ran_setup
        when ENABLE_MLIR_TRT is ON (which includes mlir_trt_wheels extra automatically).
        Multi-architecture wheels enable a single uv.lock file to work across platforms.
        Downloads from public GitHub releases (no authentication required).
        Caches downloads to avoid re-downloading.
    """
    if not tarball_extract_dir:
        logger.error("tarball_extract_dir is required for MLIR-TensorRT setup")
        sys.exit(1)

    # Resolve download directory relative to project root
    project_root = Path(__file__).parent.parent
    download_dir = str(project_root / MLIR_TRT_DOWNLOAD_DIR_DEFAULT)

    logger.info("=" * 80)
    logger.info("Setting up MLIR-TensorRT")
    logger.info(f"  Architecture: {platform.machine()}")
    logger.info(f"  Wheel version: {MLIR_TRT_WHEELS_VERSION_DEFAULT}")
    logger.info(f"  CUDA version: {MLIR_TRT_CUDA_VERSION_DEFAULT}")
    logger.info(f"  TensorRT version: {MLIR_TRT_TENSORRT_VERSION_DEFAULT}")
    logger.info(f"  Download directory: {download_dir}")
    logger.info(f"  Extract directory: {tarball_extract_dir}")
    logger.info("=" * 80)

    # Validate architecture
    arch = platform.machine()
    if arch not in ("x86_64", "aarch64"):
        logger.error(f"Unsupported architecture: {arch}. Supported: x86_64, aarch64")
        sys.exit(1)

    # Download wheels (uses resolved download_dir)
    download_mlir_trt_wheels_standalone(logger, download_dir=download_dir)

    # Download and extract tarball (uses resolved download_dir)
    download_mlir_trt_tarball_standalone(
        logger, extract_dir=tarball_extract_dir, download_dir=download_dir
    )

    logger.info("MLIR-TensorRT setup complete!")
    logger.info("Note: Wheels downloaded but not installed. Run py_ran_setup to install them.")


def find_package_root(test_dir: Path) -> Path:
    """Find package root by walking up directory tree to find pyproject.toml.

    Args:
        test_dir: Test directory to start searching from

    Returns
    -------
        Path to package root containing pyproject.toml

    Raises
    ------
        ValueError: If no pyproject.toml found in parent directories
    """
    current = test_dir.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise ValueError(f"No pyproject.toml found in parent directories of {test_dir}")


def discover_python_packages(project_root: Path) -> list[Path]:
    """Discover Python packages by finding all pyproject.toml files."""
    python_packages = []
    
    # Find all pyproject.toml files, excluding build directories and venvs
    for pyproject_file in project_root.rglob("pyproject.toml"):
        # Skip files in build directories, venvs, and hidden directories
        if any(part.startswith(".") or part in ("out", "build", ".venv", "htmlcov") 
               for part in pyproject_file.parts):
            continue
        
        # The package directory is the parent of pyproject.toml
        package_dir = pyproject_file.parent
        python_packages.append(package_dir)
    
    return sorted(python_packages)


def setup_environment(
    package_dirs: list[Path], logger: logging.Logger, extras: list[str] | None = None
) -> None:
    """Set up the Python environment for each package.

    Args:
        package_dirs: List of package directories or subdirectories to setup
        logger: Logger instance
        extras: List of extra dependency groups to install (default: ['dev'])
    """
    if not extras:
        extras = ["dev"]

    logger.info("Setting up Python environment...")
    logger.info(f"Installing extras: {', '.join(extras)}")

    # Resolve package directories (handle both package roots and subdirectories)
    resolved_package_dirs = []
    for pkg_dir in package_dirs:
        pkg_dir_resolved = pkg_dir.resolve()

        # Check if this is a package root or a subdirectory
        if (pkg_dir_resolved / "pyproject.toml").exists():
            # Already a package root
            resolved_package_dirs.append(pkg_dir_resolved)
        else:
            # Subdirectory - find package root
            try:
                package_root = find_package_root(pkg_dir_resolved)
                logger.debug(f"Found package root {package_root} for {pkg_dir_resolved}")
                # Avoid duplicates if multiple subdirs point to same package
                if package_root not in resolved_package_dirs:
                    resolved_package_dirs.append(package_root)
            except ValueError:
                logger.exception(
                    f"Package directory {pkg_dir_resolved} does not contain pyproject.toml"
                )
                sys.exit(1)

    # Install each Python package in development mode
    for package_dir in resolved_package_dirs:
        package_name = package_dir.parent.name
        logger.debug(f"Setting up {package_name} package...")

        # Create/sync virtual environment for this package with specified extras
        sync_cmd = ["uv", "sync"]
        for extra in extras:
            sync_cmd.extend(["--extra", extra])
        run_command(sync_cmd, logger, cwd=package_dir, stream_output=True)

    logger.info("Python environment setup complete!")


def load_pyproject_toml(package_dir: Path) -> dict | None:
    """Load and parse pyproject.toml file.

    Args:
        package_dir: Package directory containing pyproject.toml

    Returns
    -------
        Parsed TOML data as dictionary, or None if file doesn't exist or is invalid

    Notes
    -----
        This function uses lenient error handling - returns None on any error.
        Callers should decide whether to treat None as an error or continue gracefully.
    """
    pyproject_path = package_dir / "pyproject.toml"

    if not pyproject_path.exists():
        return None

    try:
        with pyproject_path.open("rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return None


def get_package_name_from_pyproject(package_dir: Path) -> str:
    """Read the package name from pyproject.toml.

    Args:
        package_dir: Package directory containing pyproject.toml

    Returns
    -------
        Package name from [project].name section

    Raises
    ------
        FileNotFoundError: If pyproject.toml doesn't exist or is invalid
        ValueError: If [project].name is not defined
    """
    pyproject_data = load_pyproject_toml(package_dir)

    if pyproject_data is None:
        raise FileNotFoundError(f"pyproject.toml not found or invalid in {package_dir}")

    # Get package name from project section
    package_name = pyproject_data.get("project", {}).get("name", "").strip()

    if not package_name:
        msg = f"No package name found in [project].name in {package_dir}/pyproject.toml"
        raise ValueError(msg)

    return package_name


def get_source_files_from_pyproject(package_dir: Path) -> list[str]:
    """Read source files from pyproject.toml exactly as specified.

    Args:
        package_dir: Package directory containing pyproject.toml

    Returns
    -------
        List of source file patterns from [tool.hatch.build.targets.sdist].include

    Raises
    ------
        FileNotFoundError: If pyproject.toml doesn't exist or is invalid
        ValueError: If no source files are specified in the build config
    """
    pyproject_data = load_pyproject_toml(package_dir)

    if pyproject_data is None:
        raise FileNotFoundError(f"pyproject.toml not found or invalid in {package_dir}")

    # Get files from hatch build config exactly as specified
    include_files = (
        pyproject_data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("sdist", {})
        .get("include", [])
    )

    if not include_files:
        msg = (
            f"No files specified in [tool.hatch.build.targets.sdist].include "
            f"in {package_dir}/pyproject.toml"
        )
        raise ValueError(msg)

    return include_files


def get_package_dirs(package_dir: Path, build_dir: Path, package_name: str) -> tuple[Path, Path]:
    """Get working directory and dist directory for a package."""
    if build_dir:
        # Out-of-tree build
        work_dir = build_dir / "python" / package_name
        dist_dir = work_dir / "dist"
    else:
        # In-tree build
        work_dir = package_dir
        dist_dir = package_dir / "dist"

    return work_dir, dist_dir


def setup_build_directory(
    package_dir: Path, build_dir: Path, package_name: str, logger: logging.Logger
) -> Path:
    """Set up build directory for out-of-tree builds and return working directory."""
    work_dir, _ = get_package_dirs(package_dir, build_dir, package_name)

    if build_dir:
        # Out-of-tree build: copy source to build directory
        work_dir.mkdir(parents=True, exist_ok=True)

        # Get source files from pyproject.toml
        source_files = get_source_files_from_pyproject(package_dir)

        logger.debug(f"Copying source files: {source_files}")

        for file_name in source_files:
            src_path = package_dir / file_name
            dst_path = work_dir / file_name

            if src_path.exists():
                if src_path.is_dir():
                    if dst_path.exists():
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    # Ensure parent directory exists before copying
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)

        # Copy find-links directories from pyproject.toml for uv validation during wheel build
        pyproject_data = load_pyproject_toml(package_dir)
        if pyproject_data:
            find_links = pyproject_data.get("tool", {}).get("uv", {}).get("find-links", [])
            for link in find_links:
                src = package_dir / link
                dst = work_dir / link
                if src.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    logger.debug(f"Copied find-links directory: {link}")
                else:
                    dst.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created empty find-links stub: {link}")

    return work_dir


def wheel_build(
    package_dirs: list[Path], logger: logging.Logger, build_dir: Path | None = None
) -> None:
    """Build wheels for packages using uv."""
    logger.info("Building wheels...")

    # Find root VERSION file
    current = Path(__file__).parent
    version_file = None
    while current != current.parent:
        if (current / "VERSION").exists():
            version_file = current / "VERSION"
            break
        current = current.parent

    for package_dir in package_dirs:
        package_name = package_dir.parent.name
        logger.debug(f"Building wheel for {package_name}...")

        work_dir = setup_build_directory(package_dir, build_dir, package_name, logger)

        # Copy VERSION to build directory root so __version__.py can find it
        if version_file:
            shutil.copy(version_file, work_dir / "VERSION")
            logger.debug(f"Copied VERSION to {work_dir / 'VERSION'}")

        cmd = ["uv", "build", "--wheel", "--out-dir", "dist"]
        run_command(cmd, logger, cwd=work_dir)

    logger.info("Wheel building complete!")


def wheel_install(
    package_dirs: list[Path],
    logger: logging.Logger,
    build_dir: Path | None = None,
) -> None:
    """Install built wheels."""
    logger.info("Installing wheels...")

    for package_dir in package_dirs:
        package_name = package_dir.parent.name
        logger.debug(f"Installing wheel for {package_name}...")

        _, dist_dir = get_package_dirs(package_dir, build_dir, package_name)

        if not dist_dir.exists():
            logger.warning(f"No dist directory found at {dist_dir}")
            continue

        wheel_files = list(dist_dir.glob("*.whl"))
        if not wheel_files:
            logger.warning(f"No wheel files found in {dist_dir}")
            continue

        latest_wheel = max(wheel_files, key=lambda x: x.stat().st_mtime)

        logger.debug(f"Installing {latest_wheel.name}...")
        cmd = ["uv", "pip", "install", str(latest_wheel)]
        run_command(cmd, logger, cwd=str(package_dir))


def wheel_test(
    package_dirs: list[Path],
    logger: logging.Logger,
    build_dir: Path | None = None,
) -> None:
    """Test built wheels in isolated environments.

    Args:
        package_dirs: List of package directories to test
        logger: Logger instance
        build_dir: Build directory for output files
    """
    logger.info("Testing wheels...")

    for package_dir in package_dirs:
        package_name = package_dir.parent.name
        logger.debug(f"Testing wheel for {package_name}...")

        # Get the actual Python package name from pyproject.toml
        try:
            pyproject_package_name = get_package_name_from_pyproject(package_dir)
            import_name = pyproject_package_name.replace("-", "_")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not get package name from pyproject.toml: {e}")
            import_name = package_name.replace("-", "_")

        work_dir, dist_dir = get_package_dirs(package_dir, build_dir, package_name)
        test_source_dir = work_dir / "tests" if build_dir else package_dir / "tests"

        wheel_files = list(dist_dir.glob("*.whl"))
        if not wheel_files:
            logger.warning(f"No wheel found for {package_name}, skipping test...")
            continue

        latest_wheel = max(wheel_files, key=lambda p: p.stat().st_mtime)

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            logger.info(f"Creating wheel test environment for {package_name}...")
            cmd = ["uv", "venv", "test_env"]
            run_command(cmd, logger, cwd=temp_path)

            logger.info(f"Installing wheel {latest_wheel.name}...")
            python_path = f"{temp_path}/test_env/bin/python"
            cmd = ["uv", "pip", "install", str(latest_wheel.absolute()), "--python", python_path]
            run_command(cmd, logger, cwd=temp_path)

            # Test basic import
            logger.debug(f"Testing basic import of {import_name}...")

            python_path = f"{temp_path}/test_env/bin/python"

            try:
                run_command(
                    [
                        python_path,
                        "-c",
                        f"import {import_name}; print('Import successful')",
                    ],
                    logger,
                    cwd=temp_path,
                )
            except SystemExit:
                logger.exception(f"Basic import test failed for {package_name}")
                continue

            # Run tests if they exist
            if test_source_dir.exists():
                logger.debug("Running tests from wheel...")
                shutil.copytree(test_source_dir, temp_path / "tests")

                cmd = [
                    "uv", "pip", "install", "pytest", "--python", f"{temp_path}/test_env/bin/python"
                ]
                run_command(cmd, logger, cwd=temp_path)

                try:
                    cmd = [
                        f"{temp_path}/test_env/bin/python", "-m", "pytest", "tests/", "-v",
                        "--color=yes"
                    ]

                    # Explicitly pass environment to isolated venv
                    env = os.environ.copy()
                    run_command(
                        cmd, logger, cwd=temp_path, stream_output=True, env=env
                    )
                except SystemExit:
                    logger.exception(f"Wheel tests failed for {package_name}")
                    continue

        logger.info(f"Wheel test passed for {package_name}")

    logger.info("All wheel tests completed!")


def test(
    package_dirs: list[Path],
    logger: logging.Logger,
    build_dir: Path | None = None,  # noqa: PT028
    extras: list[str] | None = None,  # noqa: PT028
) -> None:
    """Run tests with coverage.

    Args:
        package_dirs: List of package directories OR test directories to test
        logger: Logger instance
        build_dir: Build directory for output files (for coverage artifacts only)
        extras: List of extra dependency groups to install (default: ['dev'])
    """
    # Setup environment first
    if not extras:
        extras = ["dev"]
    setup_environment(package_dirs, logger, extras)

    logger.info("Running tests...")

    for pkg_dir in package_dirs:
        # Check if this is a package root or a test directory
        if (pkg_dir / "pyproject.toml").exists():
            # Full package test with coverage
            package_root = pkg_dir
            package_name = package_root.parent.name
            logger.debug(f"Testing {package_name} package...")

            # Get package name for coverage
            try:
                pyproject_package_name = get_package_name_from_pyproject(package_root)
                coverage_package = pyproject_package_name.replace("-", "_")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Could not get package name from pyproject.toml: {e}")
                coverage_package = package_name.replace("-", "_")

            # Run all tests in the package (no specific paths)
            test_paths = ["tests/"]
        else:
            # Test directory - find package root and run pytest on specific directory
            try:
                package_root = find_package_root(pkg_dir)
                test_paths = [str(pkg_dir.relative_to(package_root))]
                logger.debug(f"Testing {test_paths[0]} in package {package_root.name}...")
                coverage_package = None  # No coverage for granular test runs
                package_name = None
            except ValueError:
                logger.exception(f"Could not find package root for test directory {pkg_dir}")
                raise

        # Build pytest command
        cmd = ["uv", "run", "pytest", "-sv", "--log-cli-level=DEBUG", "--color=yes"]

        # Coverage control flag - set to True to enable coverage reporting
        enable_coverage = False

        # Add coverage if enabled and this is a full package test with build_dir
        env = os.environ.copy()
        if enable_coverage and coverage_package and build_dir:
            work_dir, _ = get_package_dirs(package_root, build_dir, package_name)
            work_dir.mkdir(parents=True, exist_ok=True)
            coverage_file = work_dir / ".coverage"
            htmlcov_dir = work_dir / "htmlcov"

            cmd.extend(
                [
                    f"--cov={coverage_package}",
                    "--cov-report=term-missing",
                    f"--cov-report=html:{htmlcov_dir}",
                ]
            )
            env["COVERAGE_FILE"] = str(coverage_file)

        # Add test paths
        cmd.extend(test_paths)

        # Run pytest from package root
        run_command(cmd, logger, cwd=str(package_root), stream_output=True, env=env)

    logger.info("All tests passed!")


# Command definitions for linting tools (base commands without directories)
LINT_COMMANDS = {
    "black": (
        ["uv", "run", "black", "--check"],
        "Checking code formatting",
    ),
    "isort": (
        ["uv", "run", "isort", "--check-only"],
        "Checking import sorting",
    ),
    "flake8": (["uv", "run", "flake8"], "Running flake8"),
    "pylint": (["uv", "run", "pylint"], "Running pylint"),
    "mypy": (["uv", "run", "mypy"], "Running type checker"),
    "ruff_check": (
        ["uv", "run", "ruff", "check"],
        "Running ruff check",
    ),
    "ruff_fix": (
        ["uv", "run", "ruff", "check", "--fix"],
        "Running ruff fix",
    ),
}


def run_doc8(
    package_dirs: list[Path],
    logger: logging.Logger,
    project_root: Path | None = None,
) -> None:
    """Run doc8 documentation linter."""
    logger.info("Running doc8...")

    if not project_root:
        msg = "project_root is required for doc8"
        raise ValueError(msg)

    if not package_dirs:
        logger.error("No Python packages supplied/discovered - cannot run doc8")
        sys.exit(1)

    package_dir = package_dirs[0]
    logger.debug("Running doc8 documentation linter...")
    docs_path = str(project_root / "docs")
    # Execute from the package dir so uv picks up its .venv.
    # Use list form + shell=False for safety
    cmd = ["uv", "run", "doc8", "--max-line-length", "100", "--ignore-path", ".venv", docs_path]
    run_command(cmd, logger, cwd=str(package_dir), shell=False)

    logger.info("doc8 completed!")


def run_single_lint_tool(
    package_dirs: list[Path],
    tool_name: str,
    logger: logging.Logger,
) -> None:
    """Run a single linting tool."""
    # Check if it's a regular tool or special tool
    if tool_name in LINT_COMMANDS:
        base_command, description = LINT_COMMANDS[tool_name]
        logger.info(f"Running {tool_name}...")

        for package_dir in package_dirs:
            package_name = package_dir.parent.name
            logger.debug(f"{description} for {package_name} package...")

            source_dirs = get_source_dirs_from_pyproject(package_dir)
            command = base_command + source_dirs
            run_command(command, logger, cwd=str(package_dir))

    else:
        raise ValueError(f"Unknown lint tool: {tool_name}")

    logger.info(f"{tool_name} completed!")


def lint(
    package_dirs: list[Path],
    logger: logging.Logger,
    project_root: Path | None = None,
) -> None:
    """Run the default lint pipeline.

    Delegates to run_single_lint_tool for extensibility.
    """
    logger.info("Running lint checks (ruff + mypy + doc8)...")

    # Run ruff check
    run_single_lint_tool(package_dirs, "ruff_check", logger)

    # Run mypy
    run_single_lint_tool(package_dirs, "mypy", logger)

    # Run doc8
    run_doc8(package_dirs, logger, project_root)

    logger.info("All lint checks passed!")


def fix_lint(
    package_dirs: list[Path],
    logger: logging.Logger,
) -> None:
    """Run ruff fix and mypy (auto-fix what can be fixed)."""
    logger.info("Running lint fixes (ruff + mypy)...")

    # Run ruff fix
    run_single_lint_tool(package_dirs, "ruff_fix", logger)

    # Run mypy (can't fix, but check after fixes)
    run_single_lint_tool(package_dirs, "mypy", logger)

    logger.info("Lint fixes completed!")


def fix_format(package_dirs: list[Path], logger: logging.Logger) -> None:
    """Format Python code using ruff."""
    logger.info("Formatting Python code with ruff...")

    for package_dir in package_dirs:
        package_name = package_dir.parent.name
        logger.debug(f"Formatting {package_name} package...")

        source_dirs = get_source_dirs_from_pyproject(package_dir)
        run_command(["uv", "run", "ruff", "format", *source_dirs], logger, cwd=package_dir)

    logger.info("Code formatting complete!")


def get_source_dirs_from_pyproject(package_dir: Path) -> list[str]:
    """Get source directories from pyproject.toml [tool.hatch.build.sources].

    Plus tests/ if exists.
    """
    pyproject_data = load_pyproject_toml(package_dir)

    if pyproject_data is None:
        raise FileNotFoundError(f"pyproject.toml not found or invalid in {package_dir}")

    hatch_sources = (
        pyproject_data.get("tool", {}).get("hatch", {}).get("build", {}).get("sources", {})
    )
    source_dirs = [
        f"{src}/" if not src.endswith("/") else src
        for src in hatch_sources
        if (package_dir / src).is_dir()
    ]

    # Fallback to src/ if nothing found
    if not source_dirs:
        source_dirs = ["src/"]

    # Add tests/ if exists and not already included
    if (package_dir / "tests").is_dir() and "tests/" not in source_dirs:
        source_dirs.append("tests/")

    return source_dirs


def check_format(package_dirs: list[Path], logger: logging.Logger) -> None:
    """Check if code formatting is up to date using ruff."""
    logger.info("Checking code formatting with ruff...")

    for package_dir in package_dirs:
        package_name = package_dir.parent.name
        logger.debug(f"Checking formatting for {package_name} package...")

        source_dirs = get_source_dirs_from_pyproject(package_dir)
        run_command(
            ["uv", "run", "ruff", "format", "--check", *source_dirs],
            logger,
            cwd=str(package_dir),
        )

    logger.info("Code formatting check completed!")


def validate_figures(project_root: Path, logger: logging.Logger) -> None:
    """Validate that all .drawio.xml files have corresponding .svg exports."""
    logger.info("Validating figure files...")

    figures_src = project_root / "docs" / "figures" / "src"
    figures_generated = project_root / "docs" / "figures" / "generated"

    if not figures_src.exists():
        logger.debug("No figures/src directory found, skipping validation")
        return

    # Find all .drawio.xml files
    xml_files = list(figures_src.glob("*.drawio.xml"))

    if not xml_files:
        logger.debug("No .drawio.xml files found in figures/src")
        return

    missing_svgs = []

    for xml_file in xml_files:
        # Expected SVG name: replace .drawio.xml with .drawio.svg
        svg_name = xml_file.name.replace(".drawio.xml", ".drawio.svg")
        svg_path = figures_generated / svg_name

        if not svg_path.exists():
            missing_svgs.append({"xml": xml_file.name, "expected_svg": svg_name})
        else:
            logger.debug(f"✓ Found {svg_name} for {xml_file.name}")

    if missing_svgs:
        logger.error("Missing SVG exports for the following figures:")
        for missing in missing_svgs:
            logger.error(f"  Source: figures/src/{missing['xml']}")
            logger.error(f"  Missing: figures/generated/{missing['expected_svg']}")

        logger.error("")
        logger.error("To fix this:")
        logger.error("1. Open the .drawio.xml file in draw.io")
        logger.error("2. Export as SVG to figures/generated/ with the expected filename")
        logger.error(
            "3. Ensure the SVG filename matches: filename.drawio.xml → filename.drawio.svg"
        )

        sys.exit(1)

    logger.info(f"✓ All {len(xml_files)} figure(s) have corresponding SVG exports")


def _validate_sphinx_prerequisites(
    package_dirs: list[Path],
    build_dir: Path | None,
    logger: logging.Logger,
    operation: str,
) -> None:
    """Validate prerequisites for Sphinx operations.

    Args:
        package_dirs: List of package directories
        build_dir: Build directory path
        logger: Logger instance
        operation: Name of the operation (for error messages)
    """
    if not package_dirs:
        error_msg = f"No package directories provided for {operation}"
        logger.error(error_msg)
        sys.exit(1)

    if not build_dir:
        error_msg = f"Build directory is required for {operation}"
        logger.error(error_msg)
        sys.exit(1)

    if not build_dir.exists():
        logger.error(f"Build directory does not exist: {build_dir}")
        sys.exit(1)


def sphinx_docs(
    project_root: Path,
    package_dirs: list[Path],
    logger: logging.Logger,
    build_dir: Path | None = None,
    *,
    warnings_as_errors: bool = True,
) -> None:
    """Build Sphinx HTML documentation."""
    logger.info("Building Sphinx HTML documentation...")

    # Validate figures before building docs
    validate_figures(project_root, logger)

    # Validate prerequisites
    _validate_sphinx_prerequisites(package_dirs, build_dir, logger, "documentation build")

    # Use the first package directory as the working directory for sphinx
    package_dir = package_dirs[0]

    # Set up paths based on build_dir
    source_dir = project_root / "docs"
    output_dir = build_dir / "docs" / "sphinx"
    doxygen_xml_dir = build_dir / "docs" / "doxygen" / "xml"

    logger.info("Building documentation with:")
    logger.info(f"  Source dir: {source_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Doxygen XML dir: {doxygen_xml_dir}")
    logger.info(f"  Package env: {package_dir}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build sphinx HTML documentation with optional warnings as errors and parallel jobs
    sphinx_flags = ["-b", "html"]
    if warnings_as_errors:
        sphinx_flags.extend(["-W", "--keep-going"])
        logger.info("Documentation warnings will be treated as errors")

    # Use 4 parallel jobs for faster builds
    sphinx_flags.extend(["-j", "4"])
    logger.info("Building documentation with 4 parallel jobs")

    breathe_config = f"breathe_projects.aerial-framework={doxygen_xml_dir}"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "sphinx",
        *sphinx_flags,
        "-D",
        breathe_config,
        str(source_dir),
        str(output_dir),
    ]

    # Set locale to avoid locale.Error in Sphinx
    env = os.environ.copy()
    env["LC_ALL"] = "C.utf8"

    run_command(cmd, logger, cwd=str(package_dir), env=env)

    logger.info("Documentation built successfully!")
    logger.info(f"Documentation available at: {output_dir}/index.html")


def sphinx_linkcheck(
    project_root: Path,
    package_dirs: list[Path],
    logger: logging.Logger,
    build_dir: Path | None = None,
) -> None:
    """Run Sphinx linkcheck to validate external links."""
    logger.info("Checking external links...")

    # Validate prerequisites
    _validate_sphinx_prerequisites(package_dirs, build_dir, logger, "linkcheck")

    # Use the first package directory as the working directory for sphinx
    package_dir = package_dirs[0]

    # Set up paths based on build_dir
    source_dir = project_root / "docs"
    doxygen_xml_dir = build_dir / "docs" / "doxygen" / "xml"
    linkcheck_dir = build_dir / "docs" / "linkcheck"
    linkcheck_dir.mkdir(parents=True, exist_ok=True)

    breathe_config = f"breathe_projects.aerial-framework={doxygen_xml_dir}"
    cmd = [
        "uv", "run", "python", "-m", "sphinx", "-b", "linkcheck", "-D", breathe_config,
        str(source_dir), str(linkcheck_dir)
    ]

    # Set locale to avoid locale.Error in Sphinx
    env = os.environ.copy()
    env["LC_ALL"] = "C.utf8"

    # Run linkcheck and capture output to filter for broken links
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, cwd=str(package_dir), env=env
    )

    # Display only broken links as warnings (in yellow)
    broken_links = []
    for line in result.stdout.split("\n"):
        if "broken" in line.lower():
            logger.warning(line)
            broken_links.append(line)

    # Show summary
    if broken_links:
        logger.warning(f"Found {len(broken_links)} broken external link(s)")
        logger.warning(f"Detailed report available at: {linkcheck_dir}/output.txt")
        logger.warning("Note: Broken external links do not fail the build")
    else:
        logger.info("All external links are valid!")

    logger.info(f"Detailed report available at: {linkcheck_dir}/output.txt")


def _filter_notebook_cells(
    notebook_path: Path,
    logger: logging.Logger,
    *,
    clear_execution_count: bool = True,
) -> int:
    """Remove cells tagged with 'remove-cell' and clear outputs.

    Returns
    -------
        Number of cells removed
    """
    # Retry to handle race condition where jupytext subprocess may have exited
    # but file buffers haven't been fully flushed to disk yet
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with notebook_path.open() as f:
                notebook = json.load(f)
            break
        except (json.JSONDecodeError, FileNotFoundError):
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.exception(
                    f"Failed to read notebook {notebook_path} after {max_retries} retries. "
                    f"File may be corrupted or jupytext conversion failed."
                )
                raise

    original_count = len(notebook["cells"])
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if "remove-cell" not in cell.get("metadata", {}).get("tags", [])
    ]

    # Clear outputs of selected cells and execution counts from all cells
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            if clear_execution_count:
                cell["execution_count"] = None
            if "keep-output" not in cell.get("metadata", {}).get("tags", []):
                cell["outputs"] = []

    removed = original_count - len(notebook["cells"])

    # Ensure notebook has kernel metadata for syntax highlighting
    if "metadata" not in notebook:
        notebook["metadata"] = {}

    if "kernelspec" not in notebook["metadata"]:
        notebook["metadata"]["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }

    if "language_info" not in notebook["metadata"]:
        notebook["metadata"]["language_info"] = {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0",
        }

    # Save notebook (either cells were removed or outputs were cleared)
    with notebook_path.open("w") as f:
        json.dump(notebook, f, indent=1)

    return removed


def _setup_notebook_extras_and_download_mlir(
    logger: logging.Logger,
    extras: list[str] | None = None,
) -> list[str]:
    """Auto-detect extras and download MLIR wheels for notebook conversion.

    Args:
        logger: Logger instance
        extras: Extra dependency groups (auto-detects if None)

    Returns
    -------
        List of extras to use for environment setup
    """
    # Auto-detect extras if not provided
    if not extras:
        logger.info("Auto-detecting extras for notebook conversion...")

        # Try to download MLIR wheels (fast if cached, fails gracefully)
        project_root = Path(__file__).parent.parent
        download_dir = str(project_root / MLIR_TRT_DOWNLOAD_DIR_DEFAULT)

        try:
            logger.info("Attempting to download MLIR-TensorRT wheels...")
            download_mlir_trt_wheels_standalone(logger, download_dir=download_dir)
            extras = ["dev", "ran_mlir_trt"]
            logger.info("Using extras: dev, ran_mlir_trt")
        except (SystemExit, Exception):  # noqa: BLE001
            logger.warning("MLIR-TensorRT wheels not available, using base configuration")
            extras = ["dev", "ran_base"]
            logger.info("Using extras: dev, ran_base")
    elif "ran_mlir_trt" in extras:
        # Explicitly specified extras - download MLIR wheels if needed
        logger.info("Downloading MLIR-TensorRT wheels for ran_mlir_trt extra...")
        project_root = Path(__file__).parent.parent
        download_dir = str(project_root / MLIR_TRT_DOWNLOAD_DIR_DEFAULT)
        download_mlir_trt_wheels_standalone(logger, download_dir=download_dir)

    return extras


def _convert_single_notebook(
    py_file: Path,
    output_file: Path,
    venv_dir: Path,
    logger: logging.Logger,
) -> tuple[str, bool]:
    """Convert single .py file to .ipynb notebook."""
    try:
        # Use --update to preserve outputs from existing notebooks (for keep-output cells)
        cmd = [
            "uv", "run", "jupytext", "--to", "ipynb", "--update",
            "--output", str(output_file), str(py_file),
        ]
        run_command(cmd, logger, cwd=str(venv_dir))

        # Filter out cells tagged with "remove-cell"
        removed = _filter_notebook_cells(output_file, logger)
        if removed > 0:
            logger.debug(f"Removed {removed} cell(s) from {output_file.name}")

    except (subprocess.CalledProcessError, OSError):
        logger.exception(f"Failed to convert {py_file.name}")
        return (py_file.name, False)

    return (py_file.name, True)


def jupytext_convert(
    package_dirs: list[Path],
    logger: logging.Logger,
    extras: list[str] | None = None,
) -> None:
    """Convert .py files to .ipynb notebooks using jupytext.

    Automatically downloads MLIR-TensorRT wheels and sets up environment
    with appropriate extras when not specified.

    Args:
        package_dirs: List of package directories
        logger: Logger instance
        extras: Extra dependency groups to install (auto-detects if None)
    """
    # Setup extras and download MLIR wheels if needed
    extras = _setup_notebook_extras_and_download_mlir(logger, extras)

    # Setup environment with detected/specified extras
    setup_environment(package_dirs, logger, extras)

    logger.info("Converting notebooks...")

    for package_dir in package_dirs:
        package_dir_resolved = package_dir.resolve()

        # Find src/ directory (first check tutorials/src, then fall back to src)
        src_dir = package_dir_resolved / "tutorials" / "src"
        if not src_dir.exists():
            src_dir = package_dir_resolved / "src"
            if not src_dir.exists():
                logger.warning(f"No tutorials/src/ or src/ directory in {package_dir_resolved}")
                continue

        # Output goes next to src/
        generated_dir = src_dir.parent / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        # Note: We preserve existing notebooks to retain outputs from keep-output cells.
        # The --update flag in jupytext updates code while preserving outputs.

        # Find venv by walking up to pyproject.toml
        venv_dir = package_dir_resolved
        while venv_dir != venv_dir.parent and not (venv_dir / "pyproject.toml").exists():
            venv_dir = venv_dir.parent

        # Copy tutorial_utils.py to generated/ if it exists (but don't convert to notebook)
        utils_file = src_dir / "tutorial_utils.py"
        if utils_file.exists():
            utils_dest = generated_dir / "tutorial_utils.py"
            shutil.copy2(utils_file, utils_dest)
            logger.info(f"Copied tutorial_utils.py to {generated_dir}")

        # Collect all .py files to convert (excluding tutorial_utils.py)
        py_files = [
            py_file for py_file in src_dir.glob("*.py")
            if py_file.name != "tutorial_utils.py"
        ]

        if not py_files:
            continue

        # Convert notebooks in parallel
        logger.info(f"Converting {len(py_files)} notebook(s) in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(py_files)) as executor:
            futures = [
                executor.submit(
                    _convert_single_notebook,
                    py_file,
                    generated_dir / py_file.with_suffix(".ipynb").name,
                    venv_dir,
                    logger,
                )
                for py_file in py_files
            ]

            # Wait for all conversions to complete and collect results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Report results
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful

        if failed > 0:
            logger.error(f"Conversion completed with {failed} failure(s)")
            sys.exit(1)

    logger.info("Notebook conversion complete!")


def notebook_test(
    package_dirs: list[Path],
    notebook_name: str | None,
    logger: logging.Logger,
    timeout: int = 300,
) -> None:
    """Test a single notebook using nbclient with locked kernel startup.

    Uses nbclient directly instead of pytest-nbmake to allow fine-grained control
    over kernel startup. A file lock serializes kernel startup across parallel tests,
    preventing ZMQ port conflicts (https://github.com/jupyter/jupyter_client/issues/487).

    Args:
        package_dirs: List of package directories (must contain exactly one directory)
        notebook_name: Name of the notebook (without .ipynb extension)
        logger: Logger instance
        timeout: Timeout for notebook execution in seconds
    """
    if not notebook_name:
        logger.error("--notebook-name is required for notebook_test command")
        sys.exit(1)

    if len(package_dirs) != 1:
        logger.error("notebook_test requires exactly one package directory")
        sys.exit(1)

    package_dir = package_dirs[0]
    package_dir_resolved = package_dir.resolve()

    # Find generated/ directory (first try tutorials/generated, then fall back to generated)
    generated_dir = package_dir_resolved / "tutorials" / "generated"
    if not generated_dir.exists():
        generated_dir = package_dir_resolved / "generated"

    # Check if notebook exists
    notebook_path = generated_dir / f"{notebook_name}.ipynb"
    if not notebook_path.exists():
        logger.error(f"Notebook {notebook_path} does not exist")
        logger.error("Please run the jupytext_convert target first to generate notebooks")
        sys.exit(1)

    logger.info(f"Testing notebook: {notebook_name}")

    # Find venv by walking up to pyproject.toml
    venv_dir = package_dir_resolved
    while venv_dir != venv_dir.parent and not (venv_dir / "pyproject.toml").exists():
        venv_dir = venv_dir.parent

    # Run pytest on the specific notebook test
    # Uses test_notebooks.py which implements nbclient with locked kernel startup
    cmd = [
        "uv", "run", "--extra", "dev", "pytest", "tutorials/test_notebooks.py",
        "-k", f"test_notebook[{notebook_name}]", "-v", "--tb=short", "--capture=no"
    ]

    # Set environment variables for test
    env = os.environ.copy()
    env["SKIP_NOTEBOOK_CTESTS"] = "1"
    env["NOTEBOOK_TIMEOUT"] = str(timeout)

    try:
        run_command(cmd, logger, cwd=str(venv_dir), stream_output=True, env=env)
        logger.info(f"✓ {notebook_name} passed")
    except subprocess.CalledProcessError:
        logger.exception(f"✗ {notebook_name} failed")
        sys.exit(1)

    # Filter notebook outputs after test
    _filter_notebook_cells(notebook_path, logger, clear_execution_count=False)

    logger.info(f"Notebook {notebook_name} testing complete!")


def main() -> None:  # noqa: PLR0915, PLR0912
    """Run the Python development environment manager."""
    parser = argparse.ArgumentParser(
        description="Python development environment manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup ran/py                        # Set up specific package
  %(prog)s test ran/py                         # Run tests for specific package
  %(prog)s docs ran/py --build-dir build       # Build documentation
  %(prog)s test ran/py --build-dir build       # Run tests with build directory
  %(prog)s wheel_build ran/py                  # Build wheels for all packages
  %(prog)s all ran/py                          # Run everything on specific package
        """,
    )

    # Command choices
    all_commands = [
        "setup",
        "test",
        "lint",
        "fix_lint",
        "check_format",
        "fix_format",
        "wheel_build",
        "wheel_install",
        "wheel_test",
        "docs",
        "sphinx_docs",
        "sphinx_linkcheck",
        "jupytext_convert",
        "notebook_test",
        "setup_mlir_trt",
        "all",
        "black",
        "isort",
        "flake8",
        "pylint",
        "mypy",
        "ruff_check",
        "ruff_fix",
        "doc8",
    ]

    parser.add_argument(
        "command",
        choices=all_commands,
        help="Command to run",
    )

    parser.add_argument(
        "packages",
        nargs="*",
        help="Python package directories (default: auto-discover all packages)",
    )

    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory for output files (default: package directory)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (show debug messages)",
    )

    parser.add_argument(
        "--notebook-timeout",
        type=int,
        default=420,
        help="Timeout for notebook execution in seconds (default: 420)",
    )

    parser.add_argument(
        "--notebook-name",
        type=str,
        help="Name of notebook to test (without .ipynb extension) for notebook_test command",
    )

    parser.add_argument(
        "--extras",
        nargs="+",
        default=None,
        help="Extras to install (default: dev, or auto-detect for jupytext_convert)",
    )

    parser.add_argument(
        "--mlir-tarball-extract-dir",
        required=False,
        help="Extract directory for MLIR-TensorRT tarball (required for setup_mlir_trt command)",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(verbose=args.verbose)

    project_root = Path(__file__).parent.parent

    # Determine package directories
    if args.packages:
        package_dirs = [Path(pkg) for pkg in args.packages]
        # Validate that directories exist
        for pkg_dir in package_dirs:
            if not pkg_dir.exists():
                logger.error(f"Package directory {pkg_dir} does not exist")
                sys.exit(1)
            # Only check for pyproject.toml if not running notebook or test commands
            # (jupytext_convert/notebook_test just need src/ directory,
            #  test and setup commands support both package roots and subdirectories)
            notebook_commands = ["jupytext_convert", "notebook_test"]
            if (
                args.command not in [*notebook_commands, "test", "setup"]
                and not (pkg_dir / "pyproject.toml").exists()
            ):
                logger.error(f"Package directory {pkg_dir} does not contain pyproject.toml")
                sys.exit(1)
    else:
        # Auto-discover packages
        package_dirs = discover_python_packages(project_root)
        if not package_dirs:
            logger.error("No Python packages found in project directory")
            sys.exit(1)

    logger.debug(f"Working with packages: {[str(p) for p in package_dirs]}")

    # Handle commands
    if args.command == "all":
        setup_environment(package_dirs, logger, extras=args.extras)
        fix_format(package_dirs, logger)
        test(package_dirs, logger, args.build_dir)
        lint(package_dirs, logger, project_root)
        wheel_build(package_dirs, logger, args.build_dir)
        wheel_install(package_dirs, logger, args.build_dir)
        wheel_test(package_dirs, logger, args.build_dir)
    elif args.command == "setup":
        setup_environment(package_dirs, logger, extras=args.extras)
    elif args.command == "test":
        test(package_dirs, logger, args.build_dir, extras=args.extras)
    elif args.command == "lint":
        lint(package_dirs, logger, project_root)
    elif args.command == "fix_lint":
        fix_lint(package_dirs, logger)
    elif args.command in LINT_COMMANDS:
        run_single_lint_tool(package_dirs, args.command, logger)
    elif args.command == "doc8":
        run_doc8(package_dirs, logger, project_root)
    elif args.command == "check_format":
        check_format(package_dirs, logger)
    elif args.command == "fix_format":
        fix_format(package_dirs, logger)
    elif args.command == "wheel_build":
        wheel_build(package_dirs, logger, args.build_dir)
    elif args.command == "wheel_install":
        wheel_install(package_dirs, logger, args.build_dir)
    elif args.command == "wheel_test":
        wheel_test(package_dirs, logger, args.build_dir)
    elif args.command == "sphinx_docs":
        sphinx_docs(project_root, package_dirs, logger, args.build_dir)
    elif args.command == "sphinx_linkcheck":
        sphinx_linkcheck(project_root, package_dirs, logger, args.build_dir)
    elif args.command == "docs":
        # Run both sphinx_docs and sphinx_linkcheck sequentially for backwards compatibility
        sphinx_docs(project_root, package_dirs, logger, args.build_dir)
        sphinx_linkcheck(project_root, package_dirs, logger, args.build_dir)
    elif args.command == "jupytext_convert":
        jupytext_convert(package_dirs, logger, extras=args.extras)
    elif args.command == "notebook_test":
        notebook_test(package_dirs, args.notebook_name, logger, args.notebook_timeout)
    elif args.command == "setup_mlir_trt":
        setup_mlir_tensorrt(
            logger,
            tarball_extract_dir=args.mlir_tarball_extract_dir,
        )

    logger.info("All operations completed successfully!")


if __name__ == "__main__":
    main()
