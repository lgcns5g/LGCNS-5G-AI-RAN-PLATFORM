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

"""Wrap mlir-tensorrt-compiler subprocess calls."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_compiler_env() -> dict[str, str]:
    """Build environment for mlir-tensorrt-compiler subprocess.

    The mlir-tensorrt-compiler binary needs to find libtvm_ffi.so at runtime.
    This function adds the TVM library path to LD_LIBRARY_PATH for the subprocess
    without modifying the parent process environment.

    Returns
    -------
        Environment dict with TVM library path added to LD_LIBRARY_PATH
    """
    env = os.environ.copy()

    # Add TVM library path for libtvm_ffi.so
    venv_path = Path(sys.executable).parent.parent
    tvm_lib_path = (
        venv_path
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "tvm_ffi"
        / "lib"
    )

    if tvm_lib_path.exists():
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{tvm_lib_path}:{current_ld}" if current_ld else str(tvm_lib_path)
        logger.debug(f"Added TVM library path to subprocess environment: {tvm_lib_path}")
    else:
        logger.warning(f"TVM library path not found: {tvm_lib_path}")

    return env


def _post_process_emitc_cpp(output_dir: Path) -> None:
    """Post-process generated C++ code to fix EmitC backend issues.

    This function fixes the EmitC backend issue where void main() is generated:
    It changes 'void main(...)' to 'void _main(...)' to avoid returning void from main
    and multiple main functions

    Args:
        output_dir: Directory containing the generated output.cpp file
    """
    output_cpp = output_dir / "output.cpp"
    if not output_cpp.exists():
        logger.warning(f"Expected output.cpp not found in {output_dir}")
        return

    # Read the generated C++ file
    with output_cpp.open("r") as f:
        content = f.read()

    # Apply fixes
    original_content = content

    # Fix: Change 'void main(...)' to 'void _main(...)'
    content = content.replace("void main(", "void _main(")

    # Only write if changes were made
    if content != original_content:
        with output_cpp.open("w") as f:
            f.write(content)
        logger.info(f"Post-processed {output_cpp} to fix EmitC backend issues")
    else:
        logger.info(f"No changes needed for {output_cpp}")


def mlir_tensorrt_compiler(
    mlir_filepath: Path,
    output_dir: Path,
    *,
    entrypoint: str = "main",
    host_target: str = "emitc",
    mlir_tensorrt_compilation_flags: list[str] | None = None,
    mlir_trt_compiler: str | None = None,
    timeout: int = 180,
) -> bool:
    """
    Wrap mlir-tensorrt-compiler tool.

    Args:
        mlir_filepath: Path to the input MLIR file
        output_dir: Directory to output the compiled artifacts
        entrypoint: The entry point function name (default: "main")
        host_target: The host target for compilation (default: "emitc").
            Available options:
            - "emitc": Compile host code to C++ (default)
            - "executor": Compile host code to MLIR-TRT interpretable executable
            - "llvm": Compile host code to LLVM IR
        mlir_tensorrt_compilation_flags: Optional list of compilation flags.
            Supported flags: tensorrt-builder-opt-level, tensorrt-workspace-memory-pool-limit, etc.
        mlir_trt_compiler: Path to mlir-tensorrt-compiler binary.
                          If None, checks MLIR_TRT_COMPILER_PATH env var, then searches PATH.
        timeout: Timeout in seconds for compilation (default: 180).
                Prevents hanging processes that can leak GPU memory.

    Returns
    -------
        True if compilation was successful, False otherwise

    Raises
    ------
        FileNotFoundError: If mlir-tensorrt-compiler is not found
        RuntimeError: If compilation fails or times out
    """
    if mlir_tensorrt_compilation_flags is None:
        mlir_tensorrt_compilation_flags = []

    # --------------------------------
    # Find the MLIR-TensorRT compiler binary
    # --------------------------------

    if mlir_trt_compiler is None:
        # Priority 1: Check environment variable (set by CMake)
        mlir_trt_compiler = os.getenv("MLIR_TRT_COMPILER_PATH")

        if mlir_trt_compiler:
            logger.debug(f"Using MLIR-TensorRT compiler from env: {mlir_trt_compiler}")
        else:
            # Priority 2: Try to find mlir-tensorrt-compiler in PATH
            mlir_trt_compiler = shutil.which("mlir-tensorrt-compiler")

            if mlir_trt_compiler:
                logger.debug(f"Found MLIR-TensorRT compiler in PATH: {mlir_trt_compiler}")

    if not mlir_trt_compiler:
        error_msg = (
            "mlir-tensorrt-compiler not found. "
            "Either set MLIR_TRT_COMPILER_PATH environment variable, "
            "add it to PATH, or pass mlir_trt_compiler parameter explicitly."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------
    # Build the command for the mlir-tensorrt-compiler tool
    # --------------------------------

    cmd = [mlir_trt_compiler]

    # Build the options string for --opts flag
    # Remove '--' prefix from flags if present (CLI tool expects flags without --)
    cleaned_flags = [flag.removeprefix("--") for flag in mlir_tensorrt_compilation_flags]

    # Add the host-target and entrypoint flags
    cleaned_flags.extend([f"host-target={host_target}", f"entrypoint={entrypoint}"])

    # Join all flags into a single string for --opts
    opts_str = " ".join(cleaned_flags)
    cmd.extend(["--opts", opts_str])

    # Add the input MLIR file and output directory
    cmd.extend([str(mlir_filepath), "-o", str(output_dir)])

    logger.info(f"Running mlir-tensorrt-compiler: {' '.join(cmd)}")
    logger.info(f"Compilation timeout: {timeout}s")

    # cmd contains only hardcoded paths and compiler flags, no user input
    # ruff: noqa: S603
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_build_compiler_env(),
        )
    except FileNotFoundError as e:
        error_msg = f"mlir-tensorrt-compiler not found: {e}"
        logger.exception(error_msg)
        raise FileNotFoundError(error_msg) from e
    except subprocess.TimeoutExpired:
        error_msg = f"MLIR-TensorRT compilation timed out after {timeout}s"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if result.returncode != 0:
        error_msg = f"MLIR-TensorRT compilation failed with return code {result.returncode}"
        if result.stdout:
            logger.error(f"Compiler stdout:\n{result.stdout}")
            error_msg += f"\nCompiler stdout:\n{result.stdout}"
        if result.stderr:
            logger.error(f"Compiler stderr:\n{result.stderr}")
            error_msg += f"\nCompiler stderr:\n{result.stderr}"

        logger.exception(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"MLIR-TensorRT compilation successful: {result.stdout}")

    # Post-process generated C++ code to fix EmitC backend issues
    if host_target == "emitc":
        _post_process_emitc_cpp(output_dir)

    # Force filesystem sync to ensure compiler output files are visible
    # Prevents race conditions in parallel compilation where subprocess may exit
    # before OS flushes .trtengine and other artifacts to disk
    os.sync()

    return True
