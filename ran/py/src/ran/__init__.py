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

"""RAN runtime Python package."""

import ctypes
import sys
from pathlib import Path

from .__version__ import __version__ as __version__


def _preload_tvm_ffi() -> bool:
    """Pre-load TVM FFI library for MLIR-TensorRT Python bindings.

    The mlir-tensorrt Python package needs to load libtvm_ffi.so at import time.
    Pre-loading with RTLD_GLOBAL makes it available to the MLIR-TRT bindings.

    Note: The mlir-tensorrt-compiler subprocess gets its library path via
    environment in mlir_trt_compiler_wrapper._build_compiler_env().

    Returns
    -------
        True if library was found and preloaded, False otherwise.
    """
    venv_path = Path(sys.executable).parent.parent
    tvm_lib_path = (
        venv_path
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "tvm_ffi"
        / "lib"
    )

    if not tvm_lib_path.exists():
        return False

    # Pre-load with RTLD_GLOBAL so mlir-tensorrt Python bindings can find it
    tvm_lib_file = tvm_lib_path / "libtvm_ffi.so"
    if tvm_lib_file.exists():
        ctypes.CDLL(str(tvm_lib_file), mode=ctypes.RTLD_GLOBAL)
        return True

    return False


# Preload TVM FFI library on import
_preload_tvm_ffi()

from . import (  # noqa: E402
    phy as phy,
    utils as utils,
)

# Note: datasets is not imported by default to avoid always importing Sionna/TensorFlow
# Users must explicitly import: from ran.datasets import ...
