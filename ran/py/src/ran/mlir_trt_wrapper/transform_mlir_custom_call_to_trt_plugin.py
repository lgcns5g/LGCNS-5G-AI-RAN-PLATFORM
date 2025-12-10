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

"""MLIR transformation utilities for converting custom calls to TensorRT plugins."""

import re


class MLIRTransformationError(Exception):
    """Raised when MLIR transformation fails."""


def transform_mlir_custom_call_to_trt_plugin(  # noqa: PLR0915
    mlir_string: str,
    plugin_configs: dict[str, dict[str, str | dict[str, str]]],
) -> str:
    """Transform stablehlo.custom_call operations to tensorrt.opaque_plugin operations.

    This function parses MLIR code containing stablehlo.custom_call operations and
    transforms them into equivalent tensorrt.opaque_plugin operations. This is a
    temporary solution until the functionality is added to the MLIR TensorRT.

    The regex pattern matches MLIR custom call operations with the following structure:
    %result_var = stablehlo.custom_call @target_name(operands) {attributes} :
    (input_types) -> output_types

    Args:
        mlir_string: The MLIR string from exported.mlir_module()
        plugin_configs: Dictionary mapping target names to their plugin configurations.
            Each target name (e.g., "tensorrt_dmrs_plugin") maps to a configuration
            dict. The key can optionally include a suffix matching the backend_config
            attribute (e.g., "tensorrt_cufft_plugin_forward") to provide different
            configurations for the same target name with different backend_config values.

            Configuration dict keys:
            - dso_path: Path to the plugin DSO file (required)
            - plugin_version: Version string for the plugin (optional, default: "1")
            - plugin_namespace: Namespace for the plugin (optional, default: "")
            - creator_func: Creator function name (optional, auto-generated if not
            provided)
            - creator_params: Dictionary of creator parameters (optional, defaults to
            {"dummy_param": 0} if empty)

            When a custom_call has a backend_config attribute, the lookup order is:
            1. Try "{target_name}_{backend_config}" (e.g., "tensorrt_cufft_plugin_forward")
            2. Fall back to "{target_name}" if the suffixed key doesn't exist

    Returns
    -------
        Transformed MLIR string with tensorrt.opaque_plugin operations

    Raises
    ------
        MLIRTransformationError: If the transformation fails, input is invalid, or
            a custom_call operation is found without a corresponding config

    Examples
    --------
        Single plugin:
        >>> mlir_input = '''
        ... %4 = stablehlo.custom_call @tensorrt_sequential_sum_plugin(%3)
        ...     {api_version = 2 : i32} : (tensor<30000xf32>) -> tensor<30000xf32>
        ... '''
        >>> configs = {
        ...     "tensorrt_sequential_sum_plugin": {
        ...         "dso_path": "/path/to/plugin.so",
        ...         "creator_params": {"i32_param": 10},
        ...     }
        ... }
        >>> result = transform_mlir_custom_call_to_trt_plugin(mlir_input, configs)

        Multiple plugins:
        >>> mlir_input = '''
        ... %4 = stablehlo.custom_call @tensorrt_dmrs_plugin(%3)
        ...     {api_version = 2 : i32} : (tensor<100xf32>) -> tensor<100xf32>
        ... %5 = stablehlo.custom_call @tensorrt_cufft_plugin(%4)
        ...     {api_version = 2 : i32} : (tensor<100xf32>) -> tensor<100xf32>
        ... '''
        >>> configs = {
        ...     "tensorrt_dmrs_plugin": {
        ...         "dso_path": "/path/to/dmrs.so",
        ...         "creator_params": {"sequence_length": 3276},
        ...     },
        ...     "tensorrt_cufft_plugin": {
        ...         "dso_path": "/path/to/cufft.so",
        ...         "creator_params": {"fft_size": 2048},
        ...     },
        ... }
        >>> result = transform_mlir_custom_call_to_trt_plugin(mlir_input, configs)

        Multiple plugins with backend_config for disambiguation:
        >>> mlir_input = '''
        ... %4 = stablehlo.custom_call @tensorrt_cufft_plugin(%3)
        ...     {api_version = 2 : i32, backend_config = "forward"} :
        ...     (tensor<100xf32>) -> tensor<100xf32>
        ... %5 = stablehlo.custom_call @tensorrt_cufft_plugin(%4)
        ...     {api_version = 2 : i32, backend_config = "inverse"} :
        ...     (tensor<100xf32>) -> tensor<100xf32>
        ... '''
        >>> configs = {
        ...     "tensorrt_cufft_plugin_forward": {
        ...         "dso_path": "/path/to/cufft.so",
        ...         "creator_params": {
        ...             "fft_size": 2048,
        ...             "direction": 0,
        ...         },
        ...     },
        ...     "tensorrt_cufft_plugin_inverse": {
        ...         "dso_path": "/path/to/cufft.so",
        ...         "creator_params": {
        ...             "fft_size": 2048,
        ...             "direction": 1,
        ...         },
        ...     },
        ... }
        >>> result = transform_mlir_custom_call_to_trt_plugin(mlir_input, configs)
    """
    # Regex pattern to match stablehlo.custom_call operations
    # Groups: 1=result_var, 2=target_name, 3=operands, 4=attributes, 5=input_types,
    # 6=output_types
    custom_call_pattern = (
        r"%(\w+(?::\d+)?) = stablehlo\.custom_call @(\w+)\(([^)]+)\) "
        r"\{([^}]*)\} : \(([^)]+)\) -> ([^)]+)"
    )

    def _validate_plugin_config(plugin_config: dict) -> None:
        """Validate that required plugin configuration is present."""
        dso_path = plugin_config.get("dso_path")
        if not dso_path:
            msg = "dso_path is required in plugin_config"
            raise MLIRTransformationError(msg)

    def replace_custom_call(match: re.Match[str]) -> str:
        """Replace a single custom call match with tensorrt.opaque_plugin operation."""
        try:
            result_var = match.group(1)
            target_name = match.group(2)
            operands = match.group(3)
            attributes = match.group(4)
            input_types = match.group(5)
            output_types = match.group(6)

            # Extract backend_config from attributes if present
            backend_config = None
            backend_config_match = re.search(r'backend_config\s*=\s*"([^"]+)"', attributes)
            if backend_config_match:
                backend_config = backend_config_match.group(1)

            # Build lookup key: first try target_name with backend_config suffix,
            # then fall back to just target_name
            config_key = target_name
            if backend_config:
                # Try with suffix first (e.g., "tensorrt_cufft_plugin_forward")
                config_key_with_suffix = f"{target_name}_{backend_config}"
                if config_key_with_suffix in plugin_configs:
                    config_key = config_key_with_suffix

            # Look up the configuration for this specific plugin
            plugin_config = plugin_configs.get(config_key)
            if plugin_config is None:
                msg = (
                    f"No configuration found for plugin '{config_key}' "
                    f"(target_name='{target_name}', backend_config='{backend_config}'). "
                    f"Available plugins: {list(plugin_configs.keys())}"
                )
                raise MLIRTransformationError(msg)  # noqa: TRY301

            # Extract plugin name from target name (remove common prefixes/suffixes)
            plugin_name = _extract_plugin_name(target_name)

            # Validate plugin configuration
            _validate_plugin_config(plugin_config)

            # Get plugin configuration
            dso_path_raw = plugin_config.get("dso_path")
            dso_path = dso_path_raw if isinstance(dso_path_raw, str) else ""

            plugin_version_raw = plugin_config.get("plugin_version", "1")
            plugin_version = plugin_version_raw if isinstance(plugin_version_raw, str) else "1"

            plugin_namespace_raw = plugin_config.get("plugin_namespace", "")
            plugin_namespace = plugin_namespace_raw if isinstance(plugin_namespace_raw, str) else ""

            creator_func_raw = plugin_config.get(
                "creator_func", f"get{_capitalize_first(plugin_name)}Creator"
            )
            creator_func = (
                creator_func_raw
                if isinstance(creator_func_raw, str)
                else f"get{_capitalize_first(plugin_name)}Creator"
            )
            creator_params_raw = plugin_config.get("creator_params", {})
            creator_params: dict[str, str | int | float | bool] = dict(
                creator_params_raw if isinstance(creator_params_raw, dict) else {}
            )

            # Get optional layer_name for better profiling visibility
            layer_name_raw = plugin_config.get("layer_name", "")
            layer_name = layer_name_raw if isinstance(layer_name_raw, str) else ""

            # Build creator_params string with default if empty
            if not creator_params:
                # Provide a default parameter to avoid compiler issues with empty
                # creator_params
                creator_params = {"dummy_param": "0"}
            params_str = _build_creator_params_string(creator_params)

            # Create the tensorrt.opaque_plugin operation
            replacement = _build_tensorrt_plugin_operation(
                result_var,
                dso_path,
                plugin_name,
                plugin_version,
                plugin_namespace,
                creator_func,
                params_str,
                operands,
                input_types,
                output_types,
                layer_name,
            )

        except Exception as e:
            msg = f"Failed to transform custom call: {e}"
            raise MLIRTransformationError(msg) from e
        else:
            return replacement

    # Perform the transformation
    try:
        transformed_mlir = re.sub(custom_call_pattern, replace_custom_call, mlir_string)
    except Exception as e:
        msg = f"Regex substitution failed: {e}"
        raise MLIRTransformationError(msg) from e
    else:
        return transformed_mlir


def _extract_plugin_name(target_name: str) -> str:
    """Extract clean plugin name from target name.

    Args:
        target_name: The target name from the custom call

    Returns
    -------
        Clean plugin name with common prefixes/suffixes removed
    """
    # Handle specific plugin name mappings
    if target_name == "tensorrt_fft_plugin":
        return "FftTrt"
    if target_name == "tensorrt_dmrs_plugin":
        return "DmrsTrt"
    if target_name == "tensorrt_sequential_sum_plugin":
        return "SequentialSum"
    if target_name == "tensorrt_cholesky_factor_inv_plugin":
        return "CholeskyFactorInv"

    # Remove common prefixes and suffixes for other plugins
    return target_name.replace("tensorrt_", "").replace("_plugin", "")


def _capitalize_first(text: str) -> str:
    """Capitalize the first letter of a string.

    Args:
        text: Input string

    Returns
    -------
        String with first letter capitalized
    """
    return text[0].upper() + text[1:] if text else ""


def _build_creator_params_string(
    creator_params: dict[str, str | int | float | bool],
) -> str:
    """Build MLIR string representation of creator parameters.

    Args:
        creator_params: Dictionary of parameter names to values

    Returns
    -------
        MLIR-formatted parameter string
    """
    if not creator_params:
        return ""

    param_parts = []
    for key, value in creator_params.items():
        # Handle different data types properly
        if isinstance(value, int):
            # For TensorRT plugins, integer parameters should be i32 to match
            # the expected PluginFieldType::kINT32 in the C++ plugin code
            param_parts.append(f"{key} = {value} : i32")
        elif isinstance(value, float):
            param_parts.append(f"{key} = {value}")
        elif isinstance(value, bool):
            param_parts.append(f"{key} = {str(value).lower()}")
        else:
            # String values need quotes
            param_parts.append(f'{key} = "{value}"')

    return ",\n        ".join(param_parts)


def _build_tensorrt_plugin_operation(  # noqa: PLR0913
    result_var: str,
    dso_path: str,
    plugin_name: str,
    plugin_version: str,
    plugin_namespace: str,
    creator_func: str,
    params_str: str,
    operands: str,
    input_types: str,
    output_types: str,
    layer_name: str = "",
) -> str:
    """Build the complete tensorrt.opaque_plugin operation string.

    Args:
        result_var: Result variable name
        dso_path: Path to the plugin DSO
        plugin_name: Name of the plugin
        plugin_version: Version of the plugin
        plugin_namespace: Namespace for the plugin
        creator_func: Creator function name
        params_str: Formatted parameter string
        operands: Input operands
        input_types: Input type specification
        output_types: Output type specification
        layer_name: Optional human-readable layer name for profiling

    Returns
    -------
        Complete MLIR tensorrt.opaque_plugin operation string
    """
    # Build the creator_params section if parameters exist
    creator_params_section = ""
    if params_str:
        creator_params_section = f",\n      creator_params = {{\n        {params_str}\n      }}"

    # Build the layer_name section if provided
    layer_name_section = ""
    if layer_name:
        layer_name_section = f',\n      layer_name = "{layer_name}"'

    # Construct the complete operation
    return f"""%{result_var} = tensorrt.opaque_plugin {{
      dso_path = "{dso_path}",
      plugin_name = "{plugin_name}",
      plugin_version = "{plugin_version}",
      plugin_namespace = "{plugin_namespace}",
      creator_func = "{creator_func}"{creator_params_section}{layer_name_section}
    }} ({operands}) : ({input_types}) -> {output_types}"""
