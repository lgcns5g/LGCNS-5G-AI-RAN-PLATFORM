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

"""JAX factory function for Cholesky factorization and inversion using TensorRT plugin."""

from typing import Any, ClassVar, TypedDict, cast

import jax
import numpy as np
import tensorrt as trt
import torch
from jax import Array, numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching, mlir

from ran.trt_plugins.manager.global_trt_plugin_manager import (
    global_trt_plugin_manager_create_plugin,
)
from ran.trt_plugins.manager.trt_plugin_manager import get_ran_trt_plugin_dso_path

# Constants for dimension checks
_NDIM_2D = 2
_NDIM_3D = 3


class TrtPluginConfig(TypedDict):
    """Type definition for TensorRT plugin configuration."""

    dso_path: str
    plugin_name: str
    plugin_extended_name: str
    plugin_version: str
    plugin_namespace: str
    creator_func: str
    creator_params: dict[str, int]
    layer_name: str  # Human-readable name for nsys profiling


class CholeskyFactorInvTrtPlugin:
    """JAX cuSOLVER Cholesky factor inversion class that uses TensorRT plugin.

    This class encapsulates the TensorRT plugin functionality for computing
    the inverse of the Cholesky factor (L^{-1}) using cuSOLVERDx library,
    with TensorRT and MLIR registrations done once during initialization.

    Algorithm:
    1. Cholesky decomposition: A = L*L^H (POTRF)
    2. Solve L*X = I (TRSM with lower triangular)
    Result: X = L^{-1} (inverse of Cholesky factor)

    This is used for whitening in MMSE-IRC equalizers where you need L^{-1}
    to transform the channel matrix.
    """

    # Class-level cache for TensorRT contexts
    # Key: (matrix_size, is_complex, batch_size)  # noqa: ERA001
    # Value: (engine, context)  # noqa: ERA001
    _trt_context_cache: ClassVar[
        dict[tuple[int, bool, int], tuple[trt.ICudaEngine, trt.IExecutionContext]]
    ] = {}

    def __init__(self, matrix_size: int = 2, is_complex: bool = False, *, name: str) -> None:
        """Initialize the cuSOLVER Cholesky inversion TensorRT plugin.

        Args:
            matrix_size: Size of the square matrix (N for NxN matrix).
                        Supported values: 2, 4, 8
            is_complex: Whether to handle complex data (True) or real data (False)
            name: Name for the plugin (used during JAX export)

        Raises
        ------
            ValueError: If matrix_size is not in [2, 4, 8]
        """
        if matrix_size not in [2, 4, 8]:
            raise ValueError(
                f"Unsupported matrix size: {matrix_size}. Supported sizes are: 2, 4, 8"
            )
        self.matrix_size = matrix_size
        self.is_complex = is_complex
        self.__name__ = name

        # Get plugin configuration. This configuration is needed for loading the plugin
        # and lowering the plugin as a custom call to TensorRT.
        data_type_suffix = "complex" if is_complex else "real"
        self.trt_plugin_config: TrtPluginConfig = {
            "dso_path": get_ran_trt_plugin_dso_path(),
            "plugin_name": "CholeskyFactorInv",  # Must match C++ plugin name
            # Unique Python identifier including data type
            "plugin_extended_name": f"CholeskyFactorInv_{matrix_size}_{data_type_suffix}",
            "plugin_version": "1",
            "plugin_namespace": "",
            "creator_func": "get_cholesky_factor_inv_creator",
            "creator_params": {
                "matrix_size": matrix_size,
                "is_complex": int(is_complex),
            },
            "layer_name": f"CholeskyFactorInv_Size{matrix_size}x{matrix_size}_{data_type_suffix}",
        }

        # Register TensorRT plugin
        self._register_and_create_tensorrt_plugin()

        # Create the JAX primitive and register MLIR components
        # Each unique matrix_size and data type needs its own primitive name
        primitive_name = f"cholesky_factor_inv_trt_plugin_{matrix_size}_{data_type_suffix}"
        self.primitive = Primitive(primitive_name)
        # Always return tuple: complex returns (real, imag), real returns (output,)
        self.primitive.multiple_results = True
        self._register_mlir_components()

    def _register_and_create_tensorrt_plugin(self) -> None:
        """Register the TensorRT plugin library and create a plugin instance.

        We use plugin_extended_name for Python-side registration to allow multiple
        instances with different parameters, while plugin_name must match the C++
        TensorRT plugin name.
        """
        # Plugin library is automatically loaded on first create_plugin call
        # Create plugin instance using the actual C++ plugin name
        # Pass matrix_size and is_complex parameters to the plugin creator
        self.plugin = global_trt_plugin_manager_create_plugin(
            self.trt_plugin_config["plugin_name"],
            {
                "matrix_size": np.int32(self.matrix_size),
                "is_complex": np.int32(int(self.is_complex)),
            },
        )

    def _register_mlir_components(self) -> None:
        """Register MLIR components for the primitive.

        Each instance gets its own primitive with unique registrations.
        The primitive's impl/abstract/lowering closures capture this instance's
        matrix_size and plugin object.
        """
        self.primitive.def_impl(self._cusolver_cholesky_inv_trt_plugin_impl)
        self.primitive.def_abstract_eval(self._cusolver_cholesky_inv_trt_plugin_abstract)
        mlir.register_lowering(self.primitive, self._cusolver_cholesky_inv_trt_plugin_lowering)
        batching.primitive_batchers[self.primitive] = (
            self._cusolver_cholesky_inv_trt_plugin_batching
        )

    def _get_or_build_trt_context(
        self,
        batch_size: int,
    ) -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
        """Get or build TensorRT context for given batch size.

        This method caches TensorRT engines and contexts to avoid repeated
        expensive build operations. The cache key is (matrix_size, is_complex, batch_size).

        Args:
            batch_size: Batch size for the TensorRT network

        Returns
        -------
            Tuple of (engine, context) for execution
        """
        cache_key = (self.matrix_size, self.is_complex, batch_size)

        # Check cache first
        if cache_key in self._trt_context_cache:
            return self._trt_context_cache[cache_key]

        # Build new context
        input_shape = (batch_size, self.matrix_size, self.matrix_size)

        trt_logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()

        # Add input tensors
        input_real_tensor = network.add_input(
            name="covariance_real",
            dtype=trt.float32,
            shape=input_shape,
        )

        if self.is_complex:
            input_imag_tensor = network.add_input(
                name="covariance_imag",
                dtype=trt.float32,
                shape=input_shape,
            )
            inputs = [input_real_tensor, input_imag_tensor]
        else:
            inputs = [input_real_tensor]

        # Add plugin layer
        plugin_layer = network.add_plugin_v3(
            inputs=inputs,
            shape_inputs=[],
            plugin=self.plugin,
        )
        plugin_layer.name = self.trt_plugin_config["layer_name"]

        # Mark outputs
        output_real_tensor = plugin_layer.get_output(0)
        output_real_tensor.name = "l_inv_real"
        network.mark_output(output_real_tensor)

        if self.is_complex:
            output_imag_tensor = plugin_layer.get_output(1)
            output_imag_tensor.name = "l_inv_imag"
            network.mark_output(output_imag_tensor)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Create runtime and context
        runtime = trt.Runtime(trt_logger)
        engine_obj = runtime.deserialize_cuda_engine(serialized_engine)
        if engine_obj is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        context = engine_obj.create_execution_context()

        # Cache and return
        self._trt_context_cache[cache_key] = (engine_obj, context)
        return engine_obj, context

    def _cusolver_cholesky_inv_trt_plugin_abstract(
        self,
        *args: Array,
    ) -> tuple[ShapedArray, ...]:
        """Evaluate shapes and dtypes.

        Return the shape and dtype of the result when tracing the primitive.

        Args:
            args: For real: (covariance_real,)
                  For complex: (covariance_real, covariance_imag)

        Returns
        -------
            Always returns tuple of ShapedArray
            For real data: (ShapedArray,) - single float32 output
            For complex data: (ShapedArray, ShapedArray) - (real, imag) float32 outputs
        """
        # Get shape from first argument (real part for complex, or single array for real)
        first_arg = args[0]

        if first_arg.ndim == _NDIM_2D:
            # Single matrix - [matrix_size, matrix_size]
            shape: tuple[int, ...] = (self.matrix_size, self.matrix_size)
        elif first_arg.ndim == _NDIM_3D:
            # Batched matrices - [batch_size, matrix_size, matrix_size]
            shape = (int(first_arg.shape[0]), self.matrix_size, self.matrix_size)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {first_arg.ndim}D")

        # Always return tuple of float32 arrays
        if self.is_complex:
            # Complex: return (real, imag)
            return (
                jax.core.ShapedArray(shape, jnp.float32),
                jax.core.ShapedArray(shape, jnp.float32),
            )
        # Real: return (output,) - single element tuple
        return (jax.core.ShapedArray(shape, jnp.float32),)

    def _cusolver_cholesky_inv_trt_plugin_lowering(
        self,
        ctx: mlir.ir.Context,  # noqa: ARG002
        *operands: mlir.ir.BlockArgument,
    ) -> list[Any]:
        """Implement cuSOLVER Cholesky inversion TensorRT plugin using MLIR.

        Lowering rule for the cuSOLVER TensorRT plugin to a StableHLO MLIR custom
        call. This rule is used to compile the primitive with MLIR-TensorRT.

        Args:
            ctx: MLIR context
            operands: For real: (covariance_real,)
                     For complex: (covariance_real, covariance_imag)
                     All operands are float32 tensors

        Returns
        -------
            List of MLIR operation results
            For real: [output]
            For complex: [output_real, output_imag]
        """
        # Get the input shape from first operand to determine batch size
        input_type = operands[0].type
        if not isinstance(input_type, mlir.ir.RankedTensorType):
            err_msg = "Input must be a ranked tensor"
            raise TypeError(err_msg)

        if len(input_type.shape) == _NDIM_2D:
            # Single matrix - [matrix_size, matrix_size]
            output_shape = [self.matrix_size, self.matrix_size]
        elif len(input_type.shape) == _NDIM_3D:
            # Batched matrices - [batch_size, matrix_size, matrix_size]
            batch_size = input_type.shape[0]
            output_shape = [batch_size, self.matrix_size, self.matrix_size]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {len(input_type.shape)}D")

        output_type = mlir.ir.RankedTensorType.get(output_shape, mlir.ir.F32Type.get())

        # For complex: 2 inputs, 2 outputs
        # For real: 1 input, 1 output
        if self.is_complex:
            result_types = [output_type, output_type]  # (real, imag)
        else:
            result_types = [output_type]

        # Use backend_config to encode the matrix size and data type
        backend_config = f"{self.matrix_size}_{'complex' if self.is_complex else 'real'}"

        plugin_op = mlir.custom_call(
            call_target_name="tensorrt_cholesky_inv_plugin",
            api_version=2,
            result_types=result_types,
            operands=list(operands),
            backend_config=backend_config,
            has_side_effect=False,
        )

        # Cast to Any: OpResultList type varies with MLIR_TRT config, causing conditional type errors
        return list(cast(Any, plugin_op.results))

    def _cusolver_cholesky_inv_trt_plugin_impl(self, *args: Array) -> tuple[Array, ...]:
        """Perform Cholesky factor inversion using cuSOLVERDx.

        Implementation rule for use with CUDA-backend. Note that this function
        uses Torch tensors to interface with the TensorRT Python API, therefore, it
        cannot be JIT compiled directly (JIT compilation is done in the lowering rule).

        This implementation uses a cache to avoid rebuilding TensorRT networks and
        engines for repeated calls with the same configuration (matrix_size, is_complex,
        batch_size), making it much faster for debugging purposes.

        Args:
            args: For real: (covariance_real,)
                  For complex: (covariance_real, covariance_imag)
                  Can be 2D [matrix_size, matrix_size] or 3D [batch_size, matrix_size, matrix_size]
                  All inputs are float32

        Returns
        -------
            Always returns tuple
            For real: (output_real,)
            For complex: (output_real, output_imag)
        """
        # Extract inputs
        if self.is_complex:
            if len(args) != 2:
                raise ValueError(f"Complex plugin expects 2 inputs, got {len(args)}")
            covariance_real_np = np.array(args[0], dtype=np.float32)
            covariance_imag_np = np.array(args[1], dtype=np.float32)
        else:
            if len(args) != 1:
                raise ValueError(f"Real plugin expects 1 input, got {len(args)}")
            covariance_real_np = np.array(args[0], dtype=np.float32)
            covariance_imag_np = None

        # Determine if we have a batch dimension
        if covariance_real_np.ndim == _NDIM_2D:
            # Single matrix - add batch dimension
            covariance_real_np = covariance_real_np[np.newaxis, :, :]
            if covariance_imag_np is not None:
                covariance_imag_np = covariance_imag_np[np.newaxis, :, :]
            batch_size = 1
            squeeze_output = True
        elif covariance_real_np.ndim == _NDIM_3D:
            batch_size = covariance_real_np.shape[0]
            squeeze_output = False
        else:
            raise ValueError(f"Expected 2D or 3D input, got {covariance_real_np.ndim}D")

        # Convert numpy arrays to torch tensors
        covariance_real_torch = torch.from_numpy(covariance_real_np).cuda()
        if self.is_complex:
            covariance_imag_torch = torch.from_numpy(covariance_imag_np).cuda()

        # Get or build cached TensorRT context
        _, context = self._get_or_build_trt_context(batch_size)

        # Allocate output buffers
        output_real_torch = torch.empty_like(covariance_real_torch)
        if self.is_complex:
            output_imag_torch = torch.empty_like(covariance_imag_torch)
            bindings = [
                covariance_real_torch.data_ptr(),
                covariance_imag_torch.data_ptr(),
                output_real_torch.data_ptr(),
                output_imag_torch.data_ptr(),
            ]
        else:
            bindings = [covariance_real_torch.data_ptr(), output_real_torch.data_ptr()]

        # Execute (this is now the only expensive operation after first call)
        context.execute_v2(bindings)

        # Convert outputs back to numpy
        output_real_np = output_real_torch.cpu().numpy()
        if self.is_complex:
            output_imag_np = output_imag_torch.cpu().numpy()

        # Remove batch dimension if it was added
        if squeeze_output:
            output_real_np = output_real_np.squeeze(0)
            if self.is_complex:
                output_imag_np = output_imag_np.squeeze(0)

        # Always return tuple
        if self.is_complex:
            return (jnp.array(output_real_np), jnp.array(output_imag_np))
        return (jnp.array(output_real_np),)

    def _cusolver_cholesky_inv_trt_plugin_batching(
        self,
        batched_args: tuple[Array, ...],
        batch_dims: tuple[int | None, ...],
    ) -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
        """Handle batching for the cuSOLVER Cholesky inversion primitive.

        Args:
            batched_args: For real: (covariance_real,)
                         For complex: (covariance_real, covariance_imag)
            batch_dims: Tuple indicating which dimension is the batch dimension
                       for each input (None if not batched)

        Returns
        -------
            Always returns ((outputs...), (batch_dims...))
            For real: ((output_real,), (batch_dim,))
            For complex: ((output_real, output_imag), (batch_dim, batch_dim))
        """
        if self.is_complex:
            # Complex: expect 2 inputs
            covariance_real, covariance_imag = batched_args
            batch_dim_real, batch_dim_imag = batch_dims

            if batch_dim_real is None and batch_dim_imag is None:
                # No batching
                result = self.primitive.bind(covariance_real, covariance_imag)
                return (result, (None, None))

            # Move batch dimensions to front if needed
            if batch_dim_real is not None and batch_dim_real != 0:
                covariance_real = jnp.moveaxis(covariance_real, batch_dim_real, 0)
            if batch_dim_imag is not None and batch_dim_imag != 0:
                covariance_imag = jnp.moveaxis(covariance_imag, batch_dim_imag, 0)

            # Apply primitive
            result = self.primitive.bind(covariance_real, covariance_imag)
            return (result, (0, 0))
        # Real: expect 1 input
        (covariance_real,) = batched_args
        (batch_dim_real,) = batch_dims

        if batch_dim_real is None:
            # No batching
            result = self.primitive.bind(covariance_real)
            return (result, (None,))

        # Move batch dimension to front if needed
        if batch_dim_real != 0:
            covariance_real = jnp.moveaxis(covariance_real, batch_dim_real, 0)

        # Apply primitive
        result = self.primitive.bind(covariance_real)
        return (result, (0,))

    def __call__(
        self, covariance_real: Array, covariance_imag: Array | None = None
    ) -> Array | tuple[Array, Array]:
        """Compute inverse of Cholesky factor using cuSOLVERDx.

        Computes L^{-1} where A = L*L^H (Cholesky decomposition).

        Args:
            covariance_real: Real part of input covariance matrices (or full matrix for real case),
                           shape (..., matrix_size, matrix_size), dtype float32
            covariance_imag: Imaginary part of input covariance matrices (for complex case only),
                           shape (..., matrix_size, matrix_size), dtype float32

        Returns
        -------
            For real data: Array with same shape as input (single output)
            For complex data: Tuple of (real_part, imag_part) arrays

        Raises
        ------
            ValueError: If input matrix size doesn't match plugin configuration,
                       or if inputs don't match the is_complex setting

        Example:
            >>> # Real case
            >>> plugin_real = CholeskyFactorInvTrtPlugin(matrix_size=2, is_complex=False, name="cholesky_real")
            >>> cov_real = jnp.array([[[4.0, 2.0], [2.0, 3.0]]])  # 1x2x2
            >>> l_inv = plugin_real(cov_real)

            >>> # Complex case
            >>> plugin_complex = CholeskyFactorInvTrtPlugin(matrix_size=2, is_complex=True, name="cholesky_complex")
            >>> cov = jnp.array([[[4.0, 2.0], [2.0, 3.0]]], dtype=jnp.complex64)
            >>> cov_real, cov_imag = jnp.real(cov), jnp.imag(cov)
            >>> l_inv_real, l_inv_imag = plugin_complex(cov_real, cov_imag)
            >>> l_inv = l_inv_real + 1j * l_inv_imag
        """
        # Validate input shape
        if (
            covariance_real.shape[-2] != self.matrix_size
            or covariance_real.shape[-1] != self.matrix_size
        ):
            raise ValueError(
                f"Input matrix size {covariance_real.shape[-2:]} does not match "
                f"plugin matrix size ({self.matrix_size}, {self.matrix_size})"
            )

        # Check if inputs match plugin configuration
        if self.is_complex:
            if covariance_imag is None:
                raise ValueError(
                    "Complex plugin requires both covariance_real and covariance_imag arguments"
                )
            if covariance_imag.shape != covariance_real.shape:
                raise ValueError(
                    f"Real and imaginary parts must have same shape: "
                    f"{covariance_real.shape} vs {covariance_imag.shape}"
                )
            # Call primitive with both inputs, returns tuple (real, imag)
            return self.primitive.bind(covariance_real, covariance_imag)
        if covariance_imag is not None:
            raise ValueError(
                "Real plugin expects only covariance_real argument (covariance_imag should be None)"
            )
        # Call primitive with single input, returns (output,) tuple - extract single element
        result = self.primitive.bind(covariance_real)
        return result[0]

    def get_config(self) -> TrtPluginConfig:
        """Get the TensorRT plugin configuration.

        Returns
        -------
            TensorRT plugin configuration dictionary
        """
        return self.trt_plugin_config.copy()


def cholesky_factor_inv(
    covariance_real: Array,
    covariance_imag: Array | None = None,
) -> Array | tuple[Array, Array]:
    """Compute inverse of Cholesky factor via cuSOLVERDx TensorRT plugin.

    Computes L^{-1} where A = L*L^H (Cholesky decomposition).
    This is a convenience function that creates a plugin instance and applies it.

    Args:
        covariance_real: Real part of input covariance matrices (or full matrix for real case),
                        shape (..., matrix_size, matrix_size), dtype float32
        covariance_imag: Imaginary part of input covariance matrices (for complex case only),
                        shape (..., matrix_size, matrix_size), dtype float32

    Returns
    -------
        For real data: Array with same shape as input (single output)
        For complex data: Tuple of (real_part, imag_part) arrays

    Raises
    ------
        ValueError: If input matrix is not square, or matrix size not in [2, 4, 8]

    Example:
        >>> # Real case
        >>> cov_real = jnp.array([[[4.0, 2.0], [2.0, 3.0]]])  # 1x2x2
        >>> l_inv = cholesky_factor_inv(cov_real)

        >>> # Complex case
        >>> cov = jnp.array([[[4.0 + 1j, 2.0 - 1j], [2.0 + 1j, 3.0]]])
        >>> cov_real, cov_imag = jnp.real(cov), jnp.imag(cov)
        >>> l_inv_real, l_inv_imag = cholesky_factor_inv(cov_real, cov_imag)
    """
    # Check that covariance is a square matrix
    if covariance_real.shape[-2] != covariance_real.shape[-1]:
        err_msg = f"Input covariance matrix must be square, got {covariance_real.shape[-2:]}"
        raise ValueError(err_msg)

    matrix_size = covariance_real.shape[-1]
    is_complex = covariance_imag is not None

    data_type_suffix = "complex" if is_complex else "real"
    plugin = CholeskyFactorInvTrtPlugin(
        matrix_size=matrix_size,
        is_complex=is_complex,
        name=f"cholesky_inv_{matrix_size}x{matrix_size}_{data_type_suffix}",
    )

    if is_complex:
        return plugin(covariance_real, covariance_imag)
    return plugin(covariance_real)


# Module-level singletons for commonly used matrix sizes
# These are created once at module import time and reused across the application

#: Singleton Cholesky inversion plugin for 2x2 matrices.
#: Usage: cholesky_inv_2x2(covariance) -> l_inv
cholesky_inv_2x2 = CholeskyFactorInvTrtPlugin(
    matrix_size=2, is_complex=True, name="cholesky_inv_2x2"
)

#: Singleton Cholesky inversion plugin for 4x4 matrices.
#: Usage: cholesky_inv_4x4(covariance) -> l_inv
cholesky_inv_4x4 = CholeskyFactorInvTrtPlugin(
    matrix_size=4, is_complex=True, name="cholesky_inv_4x4"
)

#: Singleton Cholesky inversion plugin for 8x8 matrices.
#: Usage: cholesky_inv_8x8(covariance) -> l_inv
cholesky_inv_8x8 = CholeskyFactorInvTrtPlugin(
    matrix_size=8, is_complex=True, name="cholesky_inv_8x8"
)

__all__ = [
    "CholeskyFactorInvTrtPlugin",
    "cholesky_factor_inv",
    "cholesky_inv_2x2",
    "cholesky_inv_4x4",
    "cholesky_inv_8x8",
]
