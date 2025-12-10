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

"""JAX factory function for JAX FFT function that uses TensorRT plugin."""

from collections.abc import Callable
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
_NDIM_1D = 1
_NDIM_2D = 2


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


class FftTrtPlugin:
    """JAX FFT class that uses TensorRT plugin.

    This class encapsulates the TensorRT plugin functionality for performing
    FFT computation using cuFFTDx library, with TensorRT and MLIR registrations
    done once during initialization.
    """

    # Class-level cache for TensorRT contexts
    # Key: (fft_size, direction, batch_size)  # noqa: ERA001
    # Value: (engine, context)  # noqa: ERA001
    _trt_context_cache: ClassVar[
        dict[tuple[int, int, int], tuple[trt.ICudaEngine, trt.IExecutionContext]]
    ] = {}

    def __init__(self, fft_size: int, direction: int = 0, *, name: str) -> None:
        """Initialize the FFT TensorRT plugin.

        Args:
            fft_size: Size of the FFT to compute
            direction: FFT direction (0=forward, 1=inverse)
            name: Name for the plugin (used during JAX export)
        """
        self.fft_size = fft_size
        self.direction = direction
        self.__name__ = name

        # Get plugin configuration. This configuration is needed for loading the plugin
        # and lowering the plugin as a custom call to TensorRT.
        direction_str = "ifft" if direction == 1 else "fft"
        self.trt_plugin_config: TrtPluginConfig = {
            "dso_path": get_ran_trt_plugin_dso_path(),
            "plugin_name": "FftTrt",  # Must match C++ plugin name
            # Unique Python identifier
            "plugin_extended_name": f"FftTrt_{direction_str}_{fft_size}",
            "plugin_version": "1",
            "plugin_namespace": "",
            "creator_func": "get_fft_trt_creator",
            "creator_params": {"fft_size": fft_size, "direction": direction},
            "layer_name": f"FFT_{direction_str}_Size{fft_size}",
        }

        # Register TensorRT plugin
        self._register_and_create_tensorrt_plugin()

        # Create the JAX primitive and register MLIR components
        # Each unique (fft_size, direction) combination needs its own primitive name
        direction_str = "ifft" if direction == 1 else "fft"
        primitive_name = f"fft_{direction_str}_trt_plugin_{fft_size}"
        self.primitive = Primitive(primitive_name)
        self.primitive.multiple_results = (
            True  # This primitive returns multiple results (real, imag)
        )
        self._register_mlir_components()

    def _register_and_create_tensorrt_plugin(self) -> None:
        """Register the TensorRT plugin library and create a plugin instance.

        We use plugin_extended_name for Python-side registration to allow multiple
        instances with different parameters, while plugin_name must match the C++
        TensorRT plugin name.
        """
        # Plugin library is automatically loaded on first create_plugin call
        # Create plugin instance using the actual C++ plugin name
        # Pass fft_size and direction parameters to the plugin creator
        self.plugin = global_trt_plugin_manager_create_plugin(
            self.trt_plugin_config["plugin_name"],
            {
                "fft_size": np.int32(self.fft_size),
                "direction": np.int32(self.direction),
            },
        )

    def _register_mlir_components(self) -> None:
        """Register MLIR components for the primitive.

        Each instance gets its own primitive with unique registrations.
        The primitive's impl/abstract/lowering closures capture this instance's
        fft_size, direction, and plugin object.
        """
        self.primitive.def_impl(self._cufft_trt_plugin_impl)
        self.primitive.def_abstract_eval(self._cufft_trt_plugin_abstract)
        mlir.register_lowering(self.primitive, self._cufft_trt_plugin_lowering)
        batching.primitive_batchers[self.primitive] = self._cufft_trt_plugin_batching

    def _cufft_trt_plugin_abstract(
        self,
        x_real: Array,
        x_imag: Array,
    ) -> tuple[ShapedArray, ShapedArray]:
        """Evaluate shapes and dtypes.

        Return the shape and dtype of the result when tracing the primitive.

        Args:
            x_real: Real part of input signal - can be 1D [fft_size] or
                2D [batch_size, fft_size]
            x_imag: Imaginary part of input signal - same shape as x_real

        Returns
        -------
            Tuple of (real, imag) ShapedArrays with same shape as input float32 dtype.
        """
        if x_real.ndim == _NDIM_1D:
            # Single input - [fft_size]
            return (
                jax.core.ShapedArray((self.fft_size,), jnp.float32),
                jax.core.ShapedArray((self.fft_size,), jnp.float32),
            )
        if x_real.ndim == _NDIM_2D:
            # Batched input - [batch_size, fft_size]
            return (
                jax.core.ShapedArray((x_real.shape[0], self.fft_size), jnp.float32),
                jax.core.ShapedArray((x_imag.shape[0], self.fft_size), jnp.float32),
            )
        raise ValueError(f"Expected 1D or 2D input, got {x_real.ndim}D")

    def _cufft_trt_plugin_lowering(
        self,
        ctx: mlir.ir.Context,  # noqa: ARG002
        x_real: mlir.ir.BlockArgument,
        x_imag: mlir.ir.BlockArgument,
    ) -> list[Any]:
        """Implement cuFFT TensorRT plugin using MLIR.

        Lowering rule for the cuFFT TensorRT plugin to a StableHLO MLIR custom
        call. This rule is used to compile the primitive with MLIR-TensorRT.

        Args:
            ctx: MLIR context
            x_real: Real part MLIR block argument
            x_imag: Imaginary part MLIR block argument

        Returns
        -------
            List of MLIR operation results (real, imag)
        """
        # Get the input shape to determine batch size
        input_type = x_real.type
        if not isinstance(input_type, mlir.ir.RankedTensorType):
            err_msg = "Input must be a ranked tensor"
            raise TypeError(err_msg)

        if len(input_type.shape) == _NDIM_1D:
            # Single input - [fft_size]
            output_real_type = mlir.ir.RankedTensorType.get([self.fft_size], mlir.ir.F32Type.get())
            output_imag_type = mlir.ir.RankedTensorType.get([self.fft_size], mlir.ir.F32Type.get())
        elif len(input_type.shape) == _NDIM_2D:
            # Batched input - [batch_size, fft_size]
            batch_size = input_type.shape[0]
            output_real_type = mlir.ir.RankedTensorType.get(
                [batch_size, self.fft_size], mlir.ir.F32Type.get()
            )
            output_imag_type = mlir.ir.RankedTensorType.get(
                [batch_size, self.fft_size], mlir.ir.F32Type.get()
            )
        else:
            raise ValueError(f"Expected 1D or 2D input, got {len(input_type.shape)}D")

        # Use backend_config to encode the FFT direction so it can be extracted
        # during MLIR transformation to select the correct plugin config
        direction_str = "forward" if self.direction == 0 else "inverse"
        plugin_op = mlir.custom_call(
            call_target_name="tensorrt_fft_plugin",
            api_version=2,
            result_types=[output_real_type, output_imag_type],
            operands=[x_real, x_imag],
            backend_config=direction_str,
            has_side_effect=False,
        )

        # Cast to Any: OpResultList type varies with MLIR_TRT config, causing conditional type errors
        return list(cast(Any, plugin_op.results))

    def _get_or_build_trt_context(
        self, batch_size: int
    ) -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
        """Get or build cached TensorRT engine and context.

        This method caches TensorRT engines and contexts to avoid repeated
        expensive build operations. The cache key is (fft_size, direction, batch_size).

        Args:
            batch_size: Batch size for the TensorRT network

        Returns
        -------
            Tuple of (engine, context) for execution
        """
        cache_key = (self.fft_size, self.direction, batch_size)

        # Check cache first
        if cache_key in self._trt_context_cache:
            return self._trt_context_cache[cache_key]

        # Build new context
        trt_logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()

        # Enable optimizations
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

        # Add plugin to network with separate real and imaginary inputs/outputs
        input_real_tensor = network.add_input(
            name="input_real", dtype=trt.float32, shape=[batch_size, self.fft_size]
        )
        input_imag_tensor = network.add_input(
            name="input_imag", dtype=trt.float32, shape=[batch_size, self.fft_size]
        )

        plugin_layer = network.add_plugin_v3(
            inputs=[input_real_tensor, input_imag_tensor],
            shape_inputs=[],
            plugin=self.plugin,
        )
        plugin_layer.name = self.trt_plugin_config["layer_name"]

        # Mark outputs
        output_real_tensor = plugin_layer.get_output(0)
        output_real_tensor.name = "output_real"
        network.mark_output(output_real_tensor)

        output_imag_tensor = plugin_layer.get_output(1)
        output_imag_tensor.name = "output_imag"
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

    def _cufft_trt_plugin_impl(self, x_real: Array, x_imag: Array) -> tuple[Array, Array]:
        """Perform FFT computation using cuFFT.

        Implementation rule for use with CUDA-backend. Note that this function
        uses Torch tensors to interface with the TensorRT Python API, therefore, it
        cannot be JIT compiled directly (JIT compilation is done in the lowering rule).

        This implementation uses a cache to avoid rebuilding TensorRT networks and
        engines for repeated calls with the same configuration (fft_size, direction,
        batch_size), making it much faster for debugging purposes.

        Args:
            x_real: Real part of input signal - can be 1D [fft_size] or
                2D [batch_size, fft_size]
            x_imag: Imaginary part of input signal - same shape as x_real

        Returns
        -------
            Tuple of (real, imag) parts of FFT result with same shape as input.
        """
        # Convert JAX arrays to numpy
        x_real_np = np.array(x_real, dtype=np.float32)
        x_imag_np = np.array(x_imag, dtype=np.float32)

        # Determine if input is single or batched
        if x_real_np.ndim == 1:
            # Single input - [fft_size]
            batch_size = 1
            # Reshape to 2D for consistent processing
            x_real_np = x_real_np.reshape(1, self.fft_size)
            x_imag_np = x_imag_np.reshape(1, self.fft_size)
            squeeze_output = True
        else:
            # Batched input - [batch_size, fft_size]
            batch_size = x_real_np.shape[0]
            squeeze_output = False

        # Convert to torch tensors for TensorRT
        x_real_tensor = torch.from_numpy(x_real_np).cuda()
        x_imag_tensor = torch.from_numpy(x_imag_np).cuda()

        # Get or build cached TensorRT context
        _, context = self._get_or_build_trt_context(batch_size)

        # Allocate output tensors
        output_real = torch.empty_like(x_real_tensor)
        output_imag = torch.empty_like(x_imag_tensor)

        # Execute (this is now the only expensive operation after first call)
        context.execute_v2(
            bindings=[
                x_real_tensor.data_ptr(),
                x_imag_tensor.data_ptr(),
                output_real.data_ptr(),
                output_imag.data_ptr(),
            ]
        )

        # Convert back to numpy
        output_real_np = output_real.cpu().numpy()
        output_imag_np = output_imag.cpu().numpy()

        # Remove batch dimension if it was added
        if squeeze_output:
            output_real_np = output_real_np.squeeze(0)
            output_imag_np = output_imag_np.squeeze(0)

        # Convert to JAX arrays and return
        return jnp.array(output_real_np), jnp.array(output_imag_np)

    def _cufft_trt_plugin_batching(
        self,
        batched_args: tuple[Array, Array],
        batched_dims: tuple[int | None, int | None],
    ) -> tuple[tuple[Array, Array], tuple[int | None, int | None]]:
        """Batching rule for the cuFFT TensorRT plugin.

        This function defines how the primitive should behave when batched.
        For batched inputs, it calls the primitive directly since the implementation
        supports batching internally.

        Args:
            batched_args: Tuple containing the batched real and imaginary arrays
            batched_dims: Tuple indicating which dimensions are batched for each array

        Returns
        -------
            Tuple of ((batched_real_output, batched_imag_output),
                (output_batch_dim, output_batch_dim))
        """
        # Extract the batched inputs and their batch dimensions
        x_real_batched, x_real_batch_dim = batched_args[0], batched_dims[0]
        x_imag_batched, x_imag_batch_dim = batched_args[1], batched_dims[1]

        # If inputs are not batched, just call the primitive normally
        if x_real_batch_dim is None and x_imag_batch_dim is None:
            result_real, result_imag = self.primitive.bind(x_real_batched, x_imag_batched)
            return ((result_real, result_imag), (None, None))

        # For batched inputs, call the primitive directly since it supports batching
        # internally.
        result_real, result_imag = self.primitive.bind(x_real_batched, x_imag_batched)

        # The output shape should be (batch_size, fft_size) for 2D input
        # or (fft_size,) for 1D input
        if x_real_batched.ndim == _NDIM_2D:
            return ((result_real, result_imag), (0, 0))  # batch dimension at the front
        return ((result_real, result_imag), (None, None))  # no batch dimension

    def __call__(self, x_real: Array, x_imag: Array) -> tuple[Array, Array]:
        """Call the cuFFT TensorRT plugin primitive.

        Args:
            x_real: Real part of input signal - can be 1D [fft_size] or
                2D [batch_size, fft_size]
            x_imag: Imaginary part of input signal - same shape as x_real

        Returns
        -------
            Tuple of (real, imag) parts of FFT result with same shape as input.
        """
        return self.primitive.bind(x_real, x_imag)

    def get_config(self) -> TrtPluginConfig:
        """Get the TensorRT plugin configuration.

        Returns
        -------
            TensorRT plugin configuration dictionary
        """
        return self.trt_plugin_config.copy()


def get_fft_trt_plugin(fft_size: int, direction: int = 0) -> tuple[FftTrtPlugin, TrtPluginConfig]:
    """Get JAX cuFFT class that uses TensorRT plugin.

    Factory function for JAX cuFFT function that uses TensorRT plugin.

    Args:
        fft_size: Size of the FFT to compute
        direction: FFT direction (0=forward, 1=inverse)

    Returns
    -------
        cufft_trt_plugin: JAX cuFFT function that uses TensorRT plugin
        trt_plugin_config: TensorRT plugin configuration
    """
    # Create an instance of the class
    direction_str = "ifft" if direction == 1 else "fft"
    plugin_instance = FftTrtPlugin(fft_size, direction, name=f"{direction_str}_{fft_size}")

    # Return the callable and config for backward compatibility
    return plugin_instance, plugin_instance.get_config()


def get_fft_jax(
    fft_size: int, direction: int = 0
) -> tuple[Callable[[Array, Array], tuple[Array, Array]], TrtPluginConfig]:
    """Get JAX cuFFT function that uses TensorRT plugin.

    Factory function for JAX cuFFT function that uses TensorRT plugin.
    This version works with separate real and imaginary parts to avoid complex
    types in TensorRT.

    Args:
        fft_size: Size of the FFT to compute
        direction: FFT direction (0=forward, 1=inverse)

    Returns
    -------
        cufft_jax: JAX cuFFT function that takes (real, imag) and returns (real, imag)
        trt_plugin_config: TensorRT plugin configuration
    """
    fft_trt_plugin, trt_plugin_config = get_fft_trt_plugin(fft_size=fft_size, direction=direction)

    def fft_jax(x_real: Array, x_imag: Array) -> tuple[Array, Array]:
        """
        Perform FFT computation using cuFFT library.

        Args:
            x_real: Real part of input signal - can be 1D [fft_size] or
                2D [batch_size, fft_size]
            x_imag: Imaginary part of input signal - same shape as x_real

        Returns
        -------
            Tuple of (real, imag) parts of FFT result with same shape as input.
        """
        return fft_trt_plugin(x_real, x_imag)

    return fft_jax, trt_plugin_config


# Module-level singletons for commonly used FFT sizes
# These are created once at module import time and reused across the application

#: Singleton FFT-128 plugin instance (forward direction).
#: Usage: fft_128(x_real, x_imag) -> (y_real, y_imag)
fft_128 = FftTrtPlugin(fft_size=128, direction=0, name="fft_128")

#: Singleton IFFT-128 plugin instance (inverse direction).
#: Usage: ifft_128(x_real, x_imag) -> (y_real, y_imag)
ifft_128 = FftTrtPlugin(fft_size=128, direction=1, name="ifft_128")

#: Singleton FFT-2048 plugin instance (forward direction).
#: Usage: fft_2048(x_real, x_imag) -> (y_real, y_imag)
fft_2048 = FftTrtPlugin(fft_size=2048, direction=0, name="fft_2048")

#: Singleton IFFT-2048 plugin instance (inverse direction).
#: Usage: ifft_2048(x_real, x_imag) -> (y_real, y_imag)
ifft_2048 = FftTrtPlugin(fft_size=2048, direction=1, name="ifft_2048")

#: Singleton FFT-4096 plugin instance (forward direction).
#: Usage: fft_4096(x_real, x_imag) -> (y_real, y_imag)
fft_4096 = FftTrtPlugin(fft_size=4096, direction=0, name="fft_4096")

#: Singleton IFFT-4096 plugin instance (inverse direction).
#: Usage: ifft_4096(x_real, x_imag) -> (y_real, y_imag)
ifft_4096 = FftTrtPlugin(fft_size=4096, direction=1, name="ifft_4096")
