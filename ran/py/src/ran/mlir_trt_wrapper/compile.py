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

"""Compiler for StableHLO MLIR modules to GPU executable blobs."""

import logging
import os
from pathlib import Path

import mlir_tensorrt.compiler.api as compiler  # type: ignore
from mlir_tensorrt.compiler import ir

from ran import mlir_trt_wrapper as mtw

logger = logging.getLogger(__name__)


def compile(  # noqa: A001
    stablehlo_mlir: str,
    name: str,
    export_path: Path,
    *,
    export_mlir: bool = True,
    mlir_entry_point: str = "main",
    mlir_tensorrt_compilation_flags: list[str] | None = None,
    enable_strongly_typed: bool = True,
    trt_plugin_configs: dict[str, dict[str, str | dict[str, str]]] | None = None,
    mlir_trt_compiler: str | None = None,
) -> compiler.Executable:
    """Compile StableHLO MLIR module to GPU executable.

    Args:
        stablehlo_mlir: StableHLO MLIR module as a string
        name: Function or module name
        export_path: Export path for compiled artifacts
        export_mlir: Whether to save MLIR modules to disk
        mlir_entry_point: Entry point function name
        mlir_tensorrt_compilation_flags: Additional compilation flags. If None, defaults
            to ["tensorrt-builder-opt-level=0"]. Flags should be provided without '--' prefix.
            User-provided flags are merged with defaults and mandatory flags (duplicates removed).
            Available flags: tensorrt-builder-opt-level=<0-5>, tensorrt-fp16,
            tensorrt-workspace-memory-pool-limit=<size>, tensorrt-enable-timing-cache,
            tensorrt-timing-cache-path=<path>, artifacts-dir=<path>, etc.
            Run 'mlir-tensorrt-compiler --help' for complete list
        enable_strongly_typed: Enable strongly-typed TensorRT mode (default: True).
            Prevents difficult-to-debug type-related issues. Set to False only if you
            understand the implications (expert use only).
        trt_plugin_configs: Plugin configurations mapping target names to config dicts
            with keys: dso_path, plugin_version, plugin_namespace, creator_func,
            creator_params. Transforms custom calls to TensorRT opaque plugins
        mlir_trt_compiler: Path to mlir-tensorrt-compiler binary. If None, searches PATH

    Returns
    -------
        MLIR-TensorRT executable object

    Raises
    ------
        RuntimeError: If MLIR transformation, compilation, or artifact generation fails
        FileNotFoundError: If mlir-tensorrt-compiler is not found

    Examples
    --------
        >>> import os
        >>> import jax
        >>> import jax.export
        >>> from jax import numpy as jnp
        >>> from ran import mlir_trt_wrapper as mtw
        >>> from pathlib import Path
        >>>
        >>> def my_func(x, y):
        ...     return x + y
        >>>
        >>> jit_func = jax.jit(my_func)
        >>> inputs = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        >>> exported = jax.export.export(jit_func)(*inputs)
        >>> stablehlo_mlir = exported.mlir_module()
        >>>
        >>> exe = mtw.compile(
        ...     stablehlo_mlir=stablehlo_mlir,
        ...     name="my_func",
        ...     export_path=Path("./output"),
        ... )
        >>>
        >>> # Expert use: disable strongly-typed mode
        >>> exe = mtw.compile(
        ...     stablehlo_mlir=stablehlo_mlir,
        ...     name="my_func",
        ...     export_path=Path("./output"),
        ...     enable_strongly_typed=False,
        ... )

    """
    export_path.mkdir(parents=True, exist_ok=True)

    fn_stablehlo_mlir = stablehlo_mlir

    if export_mlir:
        mlir_filepath = export_path / f"{name}.original.stablehlo.mlir"
        with mlir_filepath.open("w") as f:
            f.write(fn_stablehlo_mlir)
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Original StableHLO MLIR saved to {mlir_filepath}")

    if trt_plugin_configs is not None:
        try:
            # Transform custom calls to TensorRT plugin opaque operations
            fn_stablehlo_mlir = mtw.transform_mlir_custom_call_to_trt_plugin(
                fn_stablehlo_mlir, trt_plugin_configs
            )
            logger.info("Applied transform_mlir_custom_call_to_trt_plugin")
        except Exception as e:
            error_msg = (
                f"MLIR transformation failed for plugin configs: {list(trt_plugin_configs.keys())}"
            )
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e

    if export_mlir:
        mlir_filepath = export_path / f"{name}.opaque_plugin.stablehlo.mlir"
        with mlir_filepath.open("w") as f:
            f.write(fn_stablehlo_mlir)
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Updated StableHLO MLIR saved to {mlir_filepath}")

    try:
        with ir.Context() as context:
            m = ir.Module.parse(fn_stablehlo_mlir)

            # Configure backend priorities for TensorRT with emitc
            backends_attr = ir.ArrayAttr.get(
                [
                    ir.Attribute.parse(
                        "#plan.tensorrt_backend<disallow_shape_tensor_calculations=true, benefit=3>"
                    ),
                    ir.Attribute.parse("#plan.kernel_backend<benefit=2>"),
                    ir.Attribute.parse("#plan.host_backend<benefit=1>"),
                ]
            )
            m.operation.attributes["plan.backends"] = backends_attr

            # Build compilation flags with proper precedence and deduplication
            # 1. Start with default flags
            default_flags = ["tensorrt-builder-opt-level=0"]

            # 2. Add strongly-typed flag if enabled (default on, expert can disable)
            if enable_strongly_typed:
                default_flags.append("tensorrt-strongly-typed=true")

            # 3. User-provided flags override defaults
            if mlir_tensorrt_compilation_flags is None:
                mlir_tensorrt_compilation_flags = default_flags
            else:
                # Merge user flags with defaults, user flags take precedence
                mlir_tensorrt_compilation_flags = list(mlir_tensorrt_compilation_flags)

            # 4. Always append mandatory workaround flags for issue #39
            mandatory_workaround_flags = [
                "abi-version=0",
                "enable-v2-constant-folding=true",
            ]

            # Deduplicate flags: keep last occurrence of each flag key
            flags_dict = {}
            for flag in (
                default_flags + mlir_tensorrt_compilation_flags + mandatory_workaround_flags
            ):
                # Split on '=' to get the flag key
                flag_key = flag.split("=")[0]
                flags_dict[flag_key] = flag

            mlir_tensorrt_compilation_flags = list(flags_dict.values())

            client = compiler.CompilerClient(context)
            task = client.get_compilation_task(
                "stablehlo-to-executable",
                mlir_tensorrt_compilation_flags,
            )

            if export_mlir:
                mlir_filepath = export_path / f"{name}.stablehlo.mlir"
                final_stablehlo_mlir = m.operation
                with mlir_filepath.open("w") as f:
                    f.write(str(final_stablehlo_mlir))
                    f.flush()
                    os.fsync(f.fileno())

                logger.info(f"Updated StableHLO MLIR saved to {mlir_filepath}")

            task.run(final_stablehlo_mlir)
            mlir_trt_exe = compiler.translate_mlir_to_executable(final_stablehlo_mlir)
    except Exception as e:
        error_msg = f"MLIR-TensorRT compilation failed for '{name}'"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    try:
        compile_artifacts_file = export_path / f"{name}.bin"
        with compile_artifacts_file.open("wb") as f:
            f.write(mlir_trt_exe.serialize())
        logger.info(f"Compiled artifacts saved to: {compile_artifacts_file}")

        # Verify MLIR file exists before invoking compiler subprocess
        if not mlir_filepath.exists():
            error_msg = f"MLIR file does not exist before compilation: {mlir_filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Compile with CLI for executor target
        mtw.mlir_tensorrt_compiler(
            mlir_filepath=mlir_filepath,
            output_dir=export_path,
            entrypoint=mlir_entry_point,
            host_target="emitc",
            mlir_tensorrt_compilation_flags=mlir_tensorrt_compilation_flags,
            mlir_trt_compiler=mlir_trt_compiler,
        )
        logger.info("MLIR-TensorRT compilation successful")
    except (FileNotFoundError, RuntimeError) as e:
        # Propagate errors from mlir_tensorrt_compiler with context
        error_msg = f"Failed to generate emitc artifacts for '{name}'"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    return mlir_trt_exe
