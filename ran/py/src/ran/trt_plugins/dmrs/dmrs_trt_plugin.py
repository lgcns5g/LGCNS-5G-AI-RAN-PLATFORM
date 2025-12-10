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

"""JAX DMRS functions using TensorRT plugin.

This module provides DMRS sequence generation using a fused TensorRT CUDA kernel that
computes both gold sequences and frequency scrambling in a single GPU kernel.

Usage Examples
--------------
Using singleton instances (recommended for common sequence lengths):
    >>> from ran.trt_plugins.dmrs import dmrs_3276
    >>> # Generate DMRS symbols and scrambling sequences
    >>> # Plugin is pre-configured with sequence_length=3276, n_t=14
    >>> r_dmrs, scr_seq = dmrs_3276(slot_number=0, n_dmrs_id=0)
    >>> # r_dmrs: float32 array, shape (2, 14, 2, 1638) - [real, imag]
    >>> # scr_seq: int32 array, shape (14, 2, 3276) - binary gold sequence

Using factory function for custom sequence lengths:
    >>> from ran.trt_plugins.dmrs import get_dmrs_trt_plugin
    >>> # Create plugin with custom parameters
    >>> dmrs_1200, config = get_dmrs_trt_plugin(sequence_length=1200, n_t=14)
    >>> r_dmrs, scr_seq = dmrs_1200(slot_number=0, n_dmrs_id=0)
"""

from typing import ClassVar, TypedDict

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


class TrtPluginConfig(TypedDict):
    """Type definition for TensorRT plugin configuration.

    Attributes
    ----------
    dso_path : str
        Path to the dynamic shared object (plugin library).
    plugin_name : str
        Name of the TensorRT plugin.
    plugin_version : str
        Version string of the plugin.
    plugin_namespace : str
        Namespace for the plugin.
    creator_func : str
        Name of the creator function.
    creator_params : dict[str, int]
        Parameters passed to the creator function.
    layer_name : str
        Human-readable name for nsys profiling.
    """

    dso_path: str
    plugin_name: str
    plugin_version: str
    plugin_namespace: str
    creator_func: str
    creator_params: dict[str, int]
    layer_name: str


class DMRSTrtPlugin:
    """JAX DMRS class that uses TensorRT plugin.

    This class encapsulates the TensorRT plugin functionality for generating
    DMRS sequences, with TensorRT and MLIR registrations done once during
    initialization.
    """

    # Class-level cache for TensorRT contexts
    # Key: (sequence_length, batch_size)  # noqa: ERA001
    # Value: (engine, context)  # noqa: ERA001
    _trt_context_cache: ClassVar[
        dict[tuple[int, int], tuple[trt.ICudaEngine, trt.IExecutionContext]]
    ] = {}

    def __init__(self, sequence_length: int, name: str, n_t: int = 14) -> None:
        """Initialize the DMRS TensorRT plugin.

        Parameters
        ----------
        sequence_length : int
            Length of DMRS sequence per port (must be even).
        name : str
            Name for the plugin (used during JAX export).
        n_t : int, optional
            Number of OFDM symbols per slot (default: 14).
        """
        self.sequence_length = sequence_length
        self.n_t = n_t
        self.__name__ = name

        # Configure TensorRT plugin parameters
        # This configuration is needed for loading the plugin and lowering as custom call
        self.trt_plugin_config: TrtPluginConfig = {
            "dso_path": get_ran_trt_plugin_dso_path(),
            "plugin_name": "DmrsTrt",
            "plugin_version": "1",
            "plugin_namespace": "",
            "creator_func": "get_dmrs_trt_creator",
            "creator_params": {"sequence_length": sequence_length, "n_t": n_t},
            "layer_name": f"DMRS_Gen_SeqLen{sequence_length}_Nt{n_t}",
        }

        # Create TensorRT plugin instance
        # Plugin library is automatically loaded on first create_plugin call
        self.plugin = global_trt_plugin_manager_create_plugin(
            self.trt_plugin_config["plugin_name"],
            {"sequence_length": np.int32(self.sequence_length), "n_t": np.int32(self.n_t)},
        )

        # Create JAX primitive and register all components
        # multiple_results=True is required because primitive returns a tuple of two outputs
        self.primitive = Primitive("dmrs_trt_plugin")
        self.primitive.multiple_results = True
        self.primitive.def_impl(self._dmrs_trt_plugin_impl)
        self.primitive.def_abstract_eval(self._dmrs_trt_plugin_abstract)
        mlir.register_lowering(self.primitive, self._dmrs_trt_plugin_lowering)
        batching.primitive_batchers[self.primitive] = self._dmrs_trt_plugin_batching

    def _dmrs_trt_plugin_abstract(
        self,
        slot_number: Array,
        n_dmrs_id: Array,
    ) -> tuple[ShapedArray, ShapedArray]:
        """Evaluate shapes and dtypes.

        Return the shapes and dtypes of the two results when tracing the primitive.

        The plugin generates DMRS sequences for all n_t OFDM symbols and both
        n_scid ports (0, 1) using compile-time constant n_t. It returns both
        complex DMRS values and binary gold sequences.

        Parameters
        ----------
        slot_number : Array
            Slot number (scalar).
        n_dmrs_id : Array
            DMRS identity (scalar).

        Returns
        -------
        tuple[ShapedArray, ShapedArray]
            Tuple of two ShapedArrays:
            - Complex DMRS: shape (2, n_t, 2, sequence_length/2), dtype float32
            - Binary sequence: shape (n_t, 2, sequence_length), dtype int32
        """
        # Verify inputs are scalars
        if slot_number.ndim != 0 or n_dmrs_id.ndim != 0:
            raise ValueError("slot_number and n_dmrs_id must be scalars")

        # Output 0: Complex DMRS shape (2, n_t, 2, sequence_length/2)
        complex_shape = jax.core.ShapedArray(
            (2, self.n_t, 2, self.sequence_length // 2), jnp.float32
        )
        # Output 1: Binary sequence shape (n_t, 2, sequence_length)
        binary_shape = jax.core.ShapedArray((self.n_t, 2, self.sequence_length), jnp.int32)

        return (complex_shape, binary_shape)

    def _dmrs_trt_plugin_lowering(
        self,
        ctx: mlir.ir.Context,  # noqa: ARG002
        slot_number: mlir.ir.BlockArgument,
        n_dmrs_id: mlir.ir.BlockArgument,
    ) -> list[mlir.ir.OpResult]:
        """Lower DMRS primitive to TensorRT plugin custom call.

        Lowering rule for the DMRS TensorRT plugin to a StableHLO MLIR custom
        call. This rule is used to compile the primitive with MLIR-TensorRT.

        The TensorRT plugin uses compile-time constant n_t and generates sequences
        for all n_t symbols and both n_scid ports.

        Parameters
        ----------
        ctx : mlir.ir.Context
            MLIR context.
        slot_number : mlir.ir.BlockArgument
            Slot number tensor (scalar).
        n_dmrs_id : mlir.ir.BlockArgument
            DMRS identity tensor (scalar).

        Returns
        -------
        list[mlir.ir.OpResult]
            List of MLIR operation results.
        """
        # Verify inputs are scalars
        slot_number_type = slot_number.type
        n_dmrs_id_type = n_dmrs_id.type
        if not isinstance(slot_number_type, mlir.ir.RankedTensorType) or not isinstance(
            n_dmrs_id_type, mlir.ir.RankedTensorType
        ):
            err_msg = "Inputs must be ranked tensors"
            raise TypeError(err_msg)

        if len(slot_number_type.shape) != 0 or len(n_dmrs_id_type.shape) != 0:
            err_msg = "Inputs must be scalar tensors"
            raise ValueError(err_msg)

        # Output 0: Complex DMRS shape (2, n_t, 2, sequence_length/2)
        output_complex_type = mlir.ir.RankedTensorType.get(
            [2, self.n_t, 2, self.sequence_length // 2], mlir.ir.F32Type.get()
        )
        # Output 1: Binary sequence shape (n_t, 2, sequence_length)
        output_binary_type = mlir.ir.RankedTensorType.get(
            [self.n_t, 2, self.sequence_length], mlir.ir.IntegerType.get_signless(32)
        )

        # Reshape scalar inputs to [1] so we can concatenate them
        # Can't concatenate rank-0 tensors, need rank-1 [1] shape
        target_type = mlir.ir.RankedTensorType.get([1], mlir.ir.IntegerType.get_signless(32))
        slot_number_reshaped = mlir.hlo.reshape(result=target_type, operand=slot_number)
        n_dmrs_id_reshaped = mlir.hlo.reshape(result=target_type, operand=n_dmrs_id)

        # Concatenate inputs: [1] + [1] = [2]
        concatenated = mlir.hlo.concatenate([slot_number_reshaped, n_dmrs_id_reshaped], dimension=0)

        plugin_op = mlir.custom_call(
            call_target_name="tensorrt_dmrs_plugin",
            api_version=2,
            result_types=[output_complex_type, output_binary_type],
            operands=[concatenated],
            backend_config="",
            has_side_effect=False,
        )

        # Return both outputs
        return [plugin_op.results[0], plugin_op.results[1]]

    def _get_or_build_trt_context(
        self, batch_size: int
    ) -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
        """Get or build cached TensorRT engine and context.

        This method caches TensorRT engines and contexts to avoid repeated
        expensive build operations. The cache key is (sequence_length, n_t).

        Parameters
        ----------
        batch_size : int
            Batch size for the TensorRT network (should be 1 for scalar inputs).

        Returns
        -------
        trt.ICudaEngine
            TensorRT engine for execution.
        trt.IExecutionContext
            TensorRT execution context.
        """
        cache_key = (self.sequence_length, self.n_t)

        # Check cache first
        # Note: If plugin outputs change, cache should be cleared
        if cache_key in self._trt_context_cache:
            return self._trt_context_cache[cache_key]

        # Build new context
        trt_logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()

        # Enable optimizations for reduced kernel launch latency
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

        # Add plugin to network with concatenated input shape
        # Input: [slot_number, n_dmrs_id] (2 scalars)
        input_tensor = network.add_input(name="concatenated_input", dtype=trt.int32, shape=[2])
        plugin_layer = network.add_plugin_v3(
            inputs=[input_tensor], shape_inputs=[], plugin=self.plugin
        )
        plugin_layer.name = self.trt_plugin_config["layer_name"]

        # Mark both outputs
        output_complex = plugin_layer.get_output(0)
        output_complex.name = "dmrs_complex"
        network.mark_output(output_complex)

        output_binary = plugin_layer.get_output(1)
        output_binary.name = "dmrs_binary"
        network.mark_output(output_binary)

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

    def _dmrs_trt_plugin_impl(
        self,
        slot_number: Array,
        n_dmrs_id: Array,
    ) -> tuple[Array, Array]:
        """Generate DMRS sequences using TensorRT plugin.

        Implementation rule for use with the CUDA-backend. Note that this
        function uses Torch tensors to interface with the TensorRT Python API,
        therefore, it cannot be JIT compiled directly (JIT compilation is done
        in the lowering rule).

        This implementation uses a cache to avoid rebuilding TensorRT networks
        and engines for repeated calls with the same configuration.

        Parameters
        ----------
        slot_number : Array
            Slot number (scalar).
        n_dmrs_id : Array
            DMRS identity (scalar).

        Returns
        -------
        tuple[Array, Array]
            Tuple of two arrays:
            - Complex DMRS: shape (2, n_t, 2, sequence_length/2), dtype float32
            - Binary sequence: shape (n_t, 2, sequence_length), dtype int32
        """
        # Verify inputs are scalars
        if slot_number.ndim != 0 or n_dmrs_id.ndim != 0:
            raise ValueError("slot_number and n_dmrs_id must be scalars")

        # Convert JAX arrays to numpy scalars
        slot_number_np = np.array(slot_number, dtype=np.int32).reshape(1)
        n_dmrs_id_np = np.array(n_dmrs_id, dtype=np.int32).reshape(1)

        # Transfer to GPU
        slot_number_trt = torch.tensor(slot_number_np, dtype=torch.int32, device="cuda")
        n_dmrs_id_trt = torch.tensor(n_dmrs_id_np, dtype=torch.int32, device="cuda")

        # Concatenate inputs: [slot_number, n_dmrs_id]
        concatenated_input = torch.cat([slot_number_trt, n_dmrs_id_trt], dim=0)

        # Get or build cached TensorRT context (batch_size=1 since we have scalar inputs)
        _, context = self._get_or_build_trt_context(batch_size=1)

        # Allocate output tensors
        # Output 0: Complex DMRS shape (2, n_t, 2, sequence_length/2)
        dmrs_complex_trt = torch.empty(
            (2, self.n_t, 2, self.sequence_length // 2), dtype=torch.float32, device="cuda"
        )
        # Output 1: Binary sequence shape (n_t, 2, sequence_length)
        dmrs_binary_trt = torch.empty(
            (self.n_t, 2, self.sequence_length), dtype=torch.int32, device="cuda"
        )

        # Execute with three bindings: input, output0, output1
        context.execute_v2(
            bindings=[
                concatenated_input.data_ptr(),
                dmrs_complex_trt.data_ptr(),
                dmrs_binary_trt.data_ptr(),
            ]
        )

        # Convert back to numpy and then JAX arrays
        complex_np = dmrs_complex_trt.cpu().numpy()
        binary_np = dmrs_binary_trt.cpu().numpy()

        return (jnp.array(complex_np, dtype=jnp.float32), jnp.array(binary_np, dtype=jnp.int32))

    def _dmrs_trt_plugin_batching(
        self,
        batched_args: tuple[Array, Array],
        batched_dims: tuple[int | None, int | None],
    ) -> tuple[tuple[Array, Array], tuple[int | None, int | None]]:
        """Batching rule for DMRS primitive.

        Batching is not supported. The kernel already processes all n_t symbols
        and n_scid ports internally. For multiple slots, batch at application level.

        Parameters
        ----------
        batched_args : tuple[Array, Array]
            Tuple containing (slot_number, n_dmrs_id).
        batched_dims : tuple[int | None, int | None]
            Tuple indicating which dimension is batched for each argument.

        Returns
        -------
        tuple[Array, Array]
            Output arrays (complex DMRS, binary sequence).
        tuple[int | None, int | None]
            Output batch dimensions (None, None).

        Raises
        ------
        ValueError
            If any input is batched (vmap not supported).
        """
        # Check if any input is batched
        if any(dim is not None for dim in batched_dims):
            raise ValueError(
                "DMRS plugin does not support batched inputs (vmap). "
                "The kernel already processes all n_t symbols and 2 ports internally. "
                "For multiple slot_numbers or n_dmrs_ids, use a loop or jax.lax.scan."
            )

        # No batching - call primitive normally
        slot_number, n_dmrs_id = batched_args
        result1, result2 = self.primitive.bind(slot_number, n_dmrs_id)
        return (result1, result2), (None, None)

    def __call__(
        self,
        slot_number: int,
        n_dmrs_id: int,
    ) -> tuple[Array, Array]:
        """Generate DMRS symbols and scrambling sequences.

        High-level DMRS generation interface. The fused kernel computes both
        gold sequences and frequency scrambling on GPU. The sequence_length and
        n_t are compile-time constants set during plugin initialization.

        Parameters
        ----------
        slot_number : int
            Slot number.
        n_dmrs_id : int
            DMRS identity.

        Returns
        -------
        r_dmrs__ri_sym_cdm_sc : Array
            JAX array with stacked real/imag components with shape
            (2, n_t, 2, sequence_length//2) where first dimension is [real, imag].
        scr_seq__sym_cdm_sc : Array
            Integer JAX array of shape (n_t, 2, sequence_length).

        Examples
        --------
        >>> from ran.trt_plugins.dmrs import dmrs_3276
        >>> # Plugin configured with sequence_length=3276, n_t=14
        >>> r_dmrs, scr_seq = dmrs_3276(slot_number=0, n_dmrs_id=0)
        >>> # r_dmrs[0] contains real parts, r_dmrs[1] contains imaginary parts
        >>> r_dmrs.shape  # (2, 14, 2, 1638)
        >>> scr_seq.shape  # (14, 2, 3276)
        """
        # Generate DMRS sequences using the fused TensorRT kernel
        # The kernel computes both complex DMRS and binary gold sequences
        return self.primitive.bind(
            jnp.array(slot_number, dtype=jnp.int32), jnp.array(n_dmrs_id, dtype=jnp.int32)
        )

    def get_config(self) -> TrtPluginConfig:
        """Get the TensorRT plugin configuration.

        Returns
        -------
        TrtPluginConfig
            TensorRT plugin configuration dictionary.
        """
        return self.trt_plugin_config.copy()


def get_dmrs_trt_plugin(
    sequence_length: int,
    n_t: int = 14,
) -> tuple[DMRSTrtPlugin, TrtPluginConfig]:
    """Get JAX DMRS class that uses TensorRT plugin.

    Factory function for JAX DMRS function that uses TensorRT plugin.

    Parameters
    ----------
    sequence_length : int
        Length of DMRS sequence per port (must be even).
    n_t : int, optional
        Number of OFDM symbols per slot (default: 14).

    Returns
    -------
    DMRSTrtPlugin
        JAX DMRS function that uses TensorRT plugin.
    TrtPluginConfig
        TensorRT plugin configuration.
    """
    # Create an instance of the class
    plugin_instance = DMRSTrtPlugin(
        sequence_length, name=f"dmrs_seqlen_{sequence_length}_nt_{n_t}", n_t=n_t
    )

    # Return the callable and config for backward compatibility
    return plugin_instance, plugin_instance.get_config()


# Module-level singletons for commonly used DMRS sequence lengths
# These are created once at module import time and reused across the application

#: Singleton DMRS plugin instance for sequence length 3276 (273 PRBs, 100 MHz BW).
#: Usage: dmrs_3276(slot_number, n_dmrs_id)
dmrs_3276 = DMRSTrtPlugin(sequence_length=3276, n_t=14, name="dmrs_3276")
