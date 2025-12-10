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

"""
Test function for debugging DMRS generation, extraction, and FFT channel estimation.

Note: CDL channel datasets must be pre-generated using generate_cdl_test_data.py
before running this test.
"""

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields, replace
import logging
from pathlib import Path

import yaml


# Limit JAX GPU memory pre-allocation to prevent OOM issues
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

# Choose JAX backend
os.environ["JAX_PLATFORMS"] = "cuda"  # "cpu, cuda, or mlir_tensorrt"

from typing import Any

import pytest
import jax
import numpy as np
import numpy.typing as npt
from jax import export
from matplotlib import pyplot as plt
from tqdm import tqdm

from ran import mlir_trt_wrapper as mtw
from ran.phy import numpy as phy

import test_vectors as tvs

from ran.datasets import setup_datasets

from ran.phy.jax.pusch import ai_tukey_filter
from ran.phy.jax.pusch.channel_estimation import (
    channel_estimator,
    ChannelEstimatorDynamicInputs,
    ChannelEstimatorStaticInputs,
    ChannelEstimatorOutputs,
)
from ran.phy.jax.pusch.free_energy_filter import FreeEnergyFilterConfig
from ran.phy.jax.pusch.identity_filter import IdentityFilterConfig
from ran.phy.jax.pusch.weighted_threshold_filter import WeightedThresholdFilterConfig

from ran.trt_plugins.dmrs import (
    apply_dmrs_to_channel,
    dmrs_3276,
    gen_transmitted_dmrs_with_occ,
)

# TensorRT plugin configurations for compilation
from ran.trt_plugins.fft import (
    fft_2048,
    ifft_2048,
)

# TensorRT plugin manager utilities
from ran.trt_plugins.manager.trt_plugin_manager import (
    copy_trt_engine_for_cpp_tests,
    copy_test_data_for_cpp_tests,
    get_ran_pytest_build_dir,
    should_skip_engine_generation,
)

# Check if engines already exist (evaluated at module import time)
_skip_engine_gen = should_skip_engine_generation(
    [
        # "channel_estimator_ai_tukey_filter.trtengine",
        "channel_estimator_free_energy_filter.trtengine",
        "channel_estimator_weighted_threshold_filter.trtengine",
        "channel_estimator_identity_filter.trtengine",
    ]
)

# Skip all tests in this module if engines already exist
pytestmark = pytest.mark.skipif(
    _skip_engine_gen,
    reason="TRT engines already exist and SKIP_TRT_ENGINE_GENERATION=1",
)

# --------------------------------
# Test function for debugging
# --------------------------------


def test_channel_estimation() -> None:
    """
    Test channel estimation pipeline

    The test compares test vector outputs, NumPy reference, and compiled executable.
    """

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    # --------------------------------
    # Test parameters
    # --------------------------------

    # TV_NAME = "TVnr_7201_cuPhyMax.h5"
    TV_NAME = "TVnr_7204_cuPhyMax.h5"

    mlir_tensorrt_compilation_flags = [
        "tensorrt-builder-opt-level=0",  # 0 (debug), 5 (release)
        "tensorrt-workspace-memory-pool-limit=1GiB",
    ]

    # Skip validation and CDL testing to save time - only compile and copy TRT engines.
    # Set to False to enable full validation against reference outputs and CDL dataset performance testing.
    skip_validation = True

    # Sionna test dataset parameters (minimal for fast testing)
    validation_frac = 0.1  # Fraction of dataset to use for validation
    prng_seed = 42  # Random number generator seed
    test_snrs = np.arange(-10, 10, 2)  # Test SNRs in dB (reduced range for speed)

    # Define channel filter methods to test
    channel_filter_methods = [
        # "ai_tukey_filter",
        "free_energy_filter",
        "weighted_threshold_filter",
        "identity_filter",
    ]

    # --------------------------------
    # Build directory
    # --------------------------------

    ran_pytest_build_dir = get_ran_pytest_build_dir()
    build_dir = (
        Path(ran_pytest_build_dir) / "tests" / "phy" / "jax" / "pusch" / "channel_estimation"
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------
    # Generate CDL test dataset
    # --------------------------------

    logger.info("Generating CDL test dataset...")

    # Paths for dataset generation
    aerial_root = os.environ.get("AERIAL_FRAMEWORK_ROOT", "/opt/nvidia/aerial-framework")
    data_gen_script_path = (
        Path(aerial_root)
        / "ran"
        / "py"
        / "src"
        / "ran"
        / "datasets"
        / "generate_test_channels_app.py"
    )
    data_gen_config_template = Path(__file__).parent / "channel_estimation_test_cdl_params.yaml"

    # Resolve output directory (use build dir)
    dataset_dir = Path(ran_pytest_build_dir) / "sionna_dataset_test"
    train_path = dataset_dir / "train_data.safetensors"
    test_path = dataset_dir / "test_data.safetensors"

    # Generate dataset if it doesn't exist
    if not train_path.exists() or not test_path.exists():
        logger.info(f"Dataset not found at {dataset_dir}, generating...")
        logger.info(f"Using generation script: {data_gen_script_path}")
        logger.info(f"Using config template: {data_gen_config_template}")

        # Create temporary config with resolved output_dir
        with open(data_gen_config_template, "r") as f:
            config = yaml.safe_load(f)

        # Update output_dir with actual resolved path
        config["output_dir"] = str(dataset_dir)

        # Write temporary config
        temp_config_path = build_dir / "channel_estimation_test_cdl_params_temp.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated temporary config at: {temp_config_path}")
        logger.info(f"Output directory: {dataset_dir}")

        # Run dataset generation script
        result = subprocess.run(
            [sys.executable, str(data_gen_script_path), "--config", str(temp_config_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            error_msg = f"Dataset generation failed with return code {result.returncode}\n"
            if result.stderr:
                error_msg += f"Error output:\n{result.stderr}\n"
            if result.stdout:
                error_msg += f"Standard output:\n{result.stdout}\n"
            error_msg += "\nTo generate dataset manually, run:\n"
            error_msg += f"  python {data_gen_script_path} --config {temp_config_path}\n"
            pytest.fail(error_msg)

        logger.info("Dataset generation completed successfully")
    else:
        logger.info(f"Using existing dataset at {dataset_dir}")

    # Verify dataset files exist
    if not train_path.exists() or not test_path.exists():
        pytest.fail(
            f"CDL test datasets not found after generation!\n"
            f"Expected files:\n"
            f"  - {train_path}\n"
            f"  - {test_path}\n"
        )

    # --------------------------------
    # Test vector
    # --------------------------------

    tv = tvs.TvLoader.load(TV_NAME)

    # DMRS and channel estimation kwargs
    dmrs_kwargs = tvs.build_dmrs_kwargs(tv)
    ls_kwargs = tvs.build_ls_kwargs(tv)

    # Transmission settings from test vector
    n_f = np.int32(dmrs_kwargs["n_f"])  # Number of subcarriers
    n_prb = np.int32(ls_kwargs["n_prb"])  # Number of PRBs
    start_prb = np.int32(ls_kwargs["start_prb"])  # Starting PRB index for allocation
    n_t = np.int32(tv["Nt"][0][0])  # Number of OFDM symbols per slot
    nl = np.int32(ls_kwargs["nl"])  # Number of layers
    port_idx = tuple[int](port for port in ls_kwargs["port_idx"])  # DMRS port numbers
    vec_scid = tuple[int](scid for scid in ls_kwargs["vec_scid"])  # Scrambling id, 0 or 1
    dmrs_idx = tuple(dmrs_idx for dmrs_idx in ls_kwargs["sym_idx_dmrs"])
    n_dmrs_id = np.int32(dmrs_kwargs["n_dmrs_id"])
    slot_number = np.int32(dmrs_kwargs["slot_number"])
    energy = 2.0  # np.int32(np.float32(tv['energy'])[0])
    apply_cov_shrinkage = True

    # Layer to UE mapping (single UE for now)
    n_ue = np.int32(1)
    layer2ue = tuple([0] * int(nl))

    # Channel from test vector - downcast from complex128 to complex64
    # Preserve F-contiguous (column-major) layout
    xtf__sc_sym_rxant = np.array(ls_kwargs["xtf"], dtype=np.complex64, order="F")
    print("\nInput xtf__sc_sym_rxant:")
    print(f"  Shape: {xtf__sc_sym_rxant.shape}")
    print(f"  Dtype: {xtf__sc_sym_rxant.dtype}")
    print(f"  Strides: {xtf__sc_sym_rxant.strides}")
    print(f"  Flags: {xtf__sc_sym_rxant.flags}")
    print(f"  Flatten[0:5]: {xtf__sc_sym_rxant.flatten('K')[0:5]}")

    # Split complex into real/imag on first axis
    xtf__ri_sc_sym_rxant = np.ascontiguousarray(
        np.stack([xtf__sc_sym_rxant.real, xtf__sc_sym_rxant.imag], axis=0), dtype=np.float32
    )

    # Convert to row-major for TensorRT engine (maintain data layout).
    xtf__rxant_sym_sc_ri = np.ascontiguousarray(
        np.einsum("abcd->dcba", xtf__ri_sc_sym_rxant, order="C"), dtype=np.float32
    )
    print("\nInput xtf__rxant_sym_sc_ri:")
    print(f"  Shape: {xtf__rxant_sym_sc_ri.shape}")
    print(f"  Dtype: {xtf__rxant_sym_sc_ri.dtype}")
    print(f"  Strides: {xtf__rxant_sym_sc_ri.strides}")
    print(f"  Flags: {xtf__rxant_sym_sc_ri.flags}")
    print(f"  Flatten[0:10]: {xtf__rxant_sym_sc_ri.flatten('K')[0:10]}")
    print(f"  xtf__rxant_sym_sc_ri[0,0,0,:]: {xtf__rxant_sym_sc_ri[0, 0, 0, :]}")

    # Transpose to expected format for channel estimator and convert to float16
    xtf__ri_sym_rxant_sc = np.ascontiguousarray(
        np.einsum("abcd->dbac", xtf__rxant_sym_sc_ri), dtype=np.float16
    )

    # ================================================================================================
    # Compile TensorRT executables for each channel filter method
    # ================================================================================================

    # Dynamic inputs (same for all methods)
    dynamic_inputs = ChannelEstimatorDynamicInputs(
        xtf__ri_sym_rxant_sc=xtf__ri_sym_rxant_sc,
    )
    dynamic_inputs_pos = dynamic_inputs.to_tuple()

    # TensorRT plugin configuration (same for all methods)
    trt_plugin_configs = {
        "tensorrt_dmrs_plugin": dmrs_3276.get_config(),
        "tensorrt_fft_plugin_forward": fft_2048.get_config(),
        "tensorrt_fft_plugin_inverse": ifft_2048.get_config(),
    }

    # Storage for executables and reference outputs for each method
    executables = {}
    outputs_ref_dict = {}

    print("\n" + "=" * 80)
    print("Compiling TensorRT executables for each channel filter method...")
    print("=" * 80)

    # Store MLIR and build directories for parallel compilation
    mlir_modules = {}
    build_dirs = {}

    # Sequential: Setup, export to MLIR for each method
    for method in channel_filter_methods:
        print(f"\n{'=' * 80}")
        print(f"Channel filter method: {method}")
        print(f"{'=' * 80}")

        # Create method-specific build directory for artifacts
        method_build_dir = build_dir / method
        if method_build_dir.exists():
            print(f"  Cleaning existing directory: {method_build_dir}")
            shutil.rmtree(method_build_dir)
        method_build_dir.mkdir(parents=True, exist_ok=True)
        build_dirs[method] = method_build_dir

        # Save input data for C++ benchmark
        print("  Saving input data for C++ benchmark...")
        xtf_file = method_build_dir / "xtf_input.bin"
        xtf_meta_file = method_build_dir / "xtf_input_meta.txt"
        xtf__ri_sym_rxant_sc.tofile(xtf_file)
        with open(xtf_meta_file, "w") as f:
            f.write("# xtf input data metadata\n")
            f.write(f"shape: {xtf__ri_sym_rxant_sc.shape}\n")
            f.write(f"dtype: {xtf__ri_sym_rxant_sc.dtype}\n")
            f.write(f"size_bytes: {xtf__ri_sym_rxant_sc.nbytes}\n")

        # Create channel filter config if needed
        channel_filter_config: (
            ai_tukey_filter.AITukeyFilterConfig
            | FreeEnergyFilterConfig
            | WeightedThresholdFilterConfig
            | IdentityFilterConfig
            | None
        ) = None
        if method == "ai_tukey_filter":
            model_path = ai_tukey_filter.get_pretrained_ai_tukey_filter_path()
            channel_filter_config = ai_tukey_filter.AITukeyFilterConfig(
                model_dir=str(model_path),
                fft_size=2048,
            )
            print(f"  Using pretrained model: {model_path.name}")
            print(f"  Model directory: {model_path}")

        elif method == "free_energy_filter":
            # Create config with explicit values to test static args compilation
            channel_filter_config = FreeEnergyFilterConfig(
                fft_size=2048,
                alpha=2.0,
                tau_min=0,
                tau_max_absolute=1024,
                delay_compensation_samples=50.0,
            )
            print(f"  Using Free Energy filter config: {channel_filter_config}")
        elif method == "weighted_threshold_filter":
            # Create config with explicit values to test static args compilation
            channel_filter_config = WeightedThresholdFilterConfig(
                fft_size=2048,
                delay_compensation_samples=50.0,
                decay_rate=0.01,
                k_sigma=3.0,
            )
            print(f"  Using Weighted Threshold filter config: {channel_filter_config}")
        elif method == "identity_filter":
            # Create config with explicit values to test static args compilation
            channel_filter_config = IdentityFilterConfig(
                fft_size=2048,
                delay_compensation_samples=50.0,
            )
            print(f"  Using Identity filter config: {channel_filter_config}")

        # Create static inputs for this method
        static_inputs = ChannelEstimatorStaticInputs(
            slot_number=int(slot_number),
            n_dmrs_id=int(n_dmrs_id),
            rww_regularizer_val=1e-8,
            start_prb=int(start_prb),
            scids=vec_scid,
            apply_cov_shrinkage=apply_cov_shrinkage,
            channel_filter_method=method,
            dmrs_sym_idxs=dmrs_idx,
            dmrs_port_nums=port_idx,
            layer2ue=layer2ue,
            n_prb=int(n_prb),
            n_ue=int(n_ue),
            n_f=int(n_f),
            n_t=int(n_t),
            energy=energy,
            channel_filter_config=channel_filter_config,
        )

        # Convert to tuples for positional function call
        static_inputs_pos = static_inputs.to_tuple()
        all_inputs = dynamic_inputs_pos + static_inputs_pos
        static_argnums = tuple(range(len(dynamic_inputs_pos), len(all_inputs)))

        # Get reference outputs using cpython runtime
        print("  Computing reference outputs...")
        outputs_ref_tuple = channel_estimator(*all_inputs)
        outputs_ref = ChannelEstimatorOutputs.from_tuple(outputs_ref_tuple)
        outputs_ref_dict[method] = outputs_ref

        # Clear JAX cache to avoid conflicts between methods
        jax.clear_caches()

        # JIT and export to StableHLO MLIR
        print("  Exporting to StableHLO MLIR...")
        jit_channel_estimator = jax.jit(channel_estimator, static_argnums=static_argnums)

        exported = jax.export.export(
            jit_channel_estimator,
            disabled_checks=[
                export.DisabledSafetyCheck.custom_call("tensorrt_dmrs_plugin"),
                export.DisabledSafetyCheck.custom_call("tensorrt_fft_plugin"),
            ],
        )(*all_inputs)

        stablehlo_mlir = exported.mlir_module()
        mlir_modules[method] = stablehlo_mlir

        with open(method_build_dir / "stablehlo_mlir.mlir", "w") as f:
            f.write(stablehlo_mlir)

    # Parallel: Compile all MLIR modules to TensorRT engines
    print("\n" + "=" * 80)
    print(f"Compiling {len(channel_filter_methods)} TensorRT engines in parallel...")
    print("=" * 80)

    def compile_trt_engine(method: str) -> tuple[str, object, float]:
        """Compile MLIR to TensorRT engine for a single method."""
        start_time = time.time()
        method_build_dir = build_dirs[method]
        stablehlo_mlir = mlir_modules[method]

        # Create method-specific compilation flags with artifacts directory
        method_compilation_flags = mlir_tensorrt_compilation_flags + [
            f"artifacts-dir={method_build_dir}",
        ]

        exe = mtw.compile(
            stablehlo_mlir=stablehlo_mlir,
            name=f"channel_estimator_{method}",
            export_path=method_build_dir,
            mlir_entry_point="main",
            mlir_tensorrt_compilation_flags=method_compilation_flags,
            trt_plugin_configs=trt_plugin_configs,  # type: ignore[arg-type]
        )

        compilation_time = time.time() - start_time
        print(f"  Compiled {method} in {compilation_time:.2f}s")
        return method, exe, compilation_time

    parallel_start = time.time()
    with ThreadPoolExecutor(max_workers=len(channel_filter_methods)) as executor:
        future_to_method = {
            executor.submit(compile_trt_engine, method): method for method in channel_filter_methods
        }

        for future in as_completed(future_to_method):
            method = future_to_method[future]
            try:
                result_method, exe, comp_time = future.result()
                executables[result_method] = exe
            except Exception as exc:
                print(f"  Compilation failed for {method}: {exc}")
                raise

    parallel_total = time.time() - parallel_start
    print(f"Parallel compilation completed in {parallel_total:.2f}s")

    # Sequential: Post-compilation steps for each method
    for method in channel_filter_methods:
        print(f"\n  Post-processing {method}...")
        method_build_dir = build_dirs[method]
        exe = executables[method]

        # Copy TensorRT engine to location expected by C++ benchmark
        engine_dest = copy_trt_engine_for_cpp_tests(
            method_build_dir, f"channel_estimator_{method}.trtengine"
        )
        print(f"  Copied TensorRT engine to {engine_dest}")

        # Copy test input data files for C++ benchmark
        test_data_dest = copy_test_data_for_cpp_tests(
            method_build_dir, f"pusch_channel_estimation/{method}", ["*.bin", "*_meta.txt"]
        )
        print(f"    Copied test data files to {test_data_dest}")

    # Skip validation if flag is set
    if skip_validation:
        print("\n" + "=" * 80)
        print("✓ TensorRT engine compilation and export completed successfully!")
        print(f"  Compiled engines for: {', '.join(channel_filter_methods)}")
        print("  Skipped validation and CDL dataset testing to save time (skip_validation=True)")
        print("=" * 80)
    else:
        # Validation for each method
        for method in channel_filter_methods:
            print(f"\n  Validating {method}...")
            method_build_dir = build_dirs[method]
            exe = executables[method]
            outputs_ref = outputs_ref_dict[method]

            # Verify correctness against reference
            print("    Verifying correctness...")
            outputs_tuple = tuple(np.zeros_like(o) for o in outputs_ref.to_tuple())
            mtw.execute(
                exe=exe,
                inputs=dynamic_inputs_pos,
                outputs=outputs_tuple,
                sync_stream=True,
                mlir_entry_point="main",
            )
            outputs_trt = ChannelEstimatorOutputs.from_tuple(outputs_tuple)

            all_passed = True
            for field in fields(ChannelEstimatorOutputs):
                field_name = field.name
                ref_val = getattr(outputs_ref, field_name)
                trt_val = getattr(outputs_trt, field_name)
                max_diff: np.floating[Any] = np.max(np.abs(trt_val - ref_val))
                if max_diff > 1e-2:
                    all_passed = False
                    print(f"    ✗ {field_name}: FAILED (max diff = {max_diff:.6f})")
                else:
                    print(f"    ✓ {field_name}: passed (max diff = {max_diff:.6f})")

                # Assert each field matches with tolerance
                np.testing.assert_allclose(
                    trt_val,
                    ref_val,
                    rtol=1e-2,
                    atol=1e-2,
                    err_msg=f"{method}: {field_name} mismatch between TRT and reference",
                )

            if all_passed:
                print(f"  ✓ Test passed for {method}")
            else:
                print(f"  ✗ Test FAILED for {method}")

    # Return early if validation is disabled
    if skip_validation:
        return

    # --------------------------------
    # Test with CDL dataset
    # --------------------------------

    logger.info("\n" + "=" * 60)
    logger.info("CDL dataset performance testing with MLIR-TensorRT runtime...")
    logger.info("=" * 60)

    # Setup test dataset (ignore train and val datasets)
    # num_sc for 273 PRBs = 273 * 12 = 3276
    _, _, test_dataset = setup_datasets(
        train_glob=str(train_path),
        test_glob=str(test_path),
        num_sc=3276,  # 273 PRBs * 12 subcarriers
        validation_frac=validation_frac,
        prng_seed=prng_seed,
    )

    print(f"Loaded {len(test_dataset)} test channel samples")

    # Generate DMRS (use same DMRS for all channel samples below)
    r_dmrs__ri_sym_cdm_dsc, _ = dmrs_3276(slot_number=int(slot_number), n_dmrs_id=int(n_dmrs_id))

    # Get the transmitted DMRS with frequency and time OCC applied
    n_dmrs_sc = int(n_prb) * 6  # Type 1 DMRS: 6 subcarriers per PRB
    x_dmrs__ri_port_sym_sc = gen_transmitted_dmrs_with_occ(
        r_dmrs__ri_sym_cdm_dsc=r_dmrs__ri_sym_cdm_dsc,
        dmrs_port_nums=port_idx
        if isinstance(port_idx, (list, tuple, np.ndarray))
        else np.array([port_idx]),
        scids=(int(vec_scid),) if isinstance(vec_scid, (int, np.integer)) else tuple(vec_scid),
        dmrs_sym_idxs=dmrs_idx,
        n_dmrs_sc=n_dmrs_sc,
    )
    x_dmrs_real__port_sym_sc = x_dmrs__ri_port_sym_sc[0]
    x_dmrs_imag__port_sym_sc = x_dmrs__ri_port_sym_sc[1]
    x_dmrs__port_sym_dsc = np.ascontiguousarray(
        x_dmrs_real__port_sym_sc + 1j * x_dmrs_imag__port_sym_sc, dtype=np.complex64
    )

    # Determine the Type 1 DMRS grid configuration (even=0 or odd=1 subcarriers)
    grid_cfg = (
        port_idx[0] if isinstance(port_idx, (list, tuple, np.ndarray)) else port_idx
    ) & 0b010
    grid_cfg = grid_cfg >> 1  # 0 for even, 1 for odd

    # ================================================================================================
    # Test loop - Execute all channel filter methods
    # ================================================================================================

    # Storage for MSE results for each method
    mse_db_legacy__sample: list[float] = []
    snr__sample: list[float] = []
    mse_db_methods: dict[str, list[float]] = {method: [] for method in channel_filter_methods}

    # Edge vs Center tracking
    mse_db_methods_edge_left: dict[str, list[float]] = {
        method: [] for method in channel_filter_methods
    }
    mse_db_methods_center: dict[str, list[float]] = {
        method: [] for method in channel_filter_methods
    }
    mse_db_methods_edge_right: dict[str, list[float]] = {
        method: [] for method in channel_filter_methods
    }
    mse_db_legacy_edge_left__sample: list[float] = []
    mse_db_legacy_center__sample: list[float] = []
    mse_db_legacy_edge_right__sample: list[float] = []

    for test_idx in tqdm(range(len(test_dataset)), desc="Testing all methods", unit="samples"):
        # Test dataset loading and processing
        H__sc_sym_rxant = test_dataset[test_idx]

        # Extract the true channel at the DMRS positions
        H_true__sc_sym_rxant = np.ascontiguousarray(H__sc_sym_rxant, dtype=np.complex64)

        # Apply DMRS to the true channel (transmit: y = h * x)
        # n_dmrs_sc already defined above
        dmrs_base = 12 * start_prb
        dmrs_sc_idxs = dmrs_base + 2 * np.arange(n_dmrs_sc) + grid_cfg

        # Use shared function to apply DMRS to channel
        H_dmrs__sc_sym_rxant = np.array(
            apply_dmrs_to_channel(
                jax.numpy.asarray(H__sc_sym_rxant),
                jax.numpy.asarray(x_dmrs__port_sym_dsc),
                jax.numpy.asarray(dmrs_sc_idxs),
                jax.numpy.asarray(dmrs_idx),
                energy,
            )
        )

        # Add noise to the channel with DMRS (randomly pick an SNR from test_snrs)
        _snr: float = float(test_snrs[test_idx % len(test_snrs)])
        snr__sample.append(_snr)
        H_dmrs_noisy__sc_sym_rxant = phy.utils.awgn(H_dmrs__sc_sym_rxant, _snr)

        # Legacy delay-domain channel estimator
        # NOTE: channel_est_dd operates on FULL resource grid, not just allocation

        # Extract FULL grid DMRS symbols from received grid
        y_dmrs_full = H_dmrs_noisy__sc_sym_rxant[:, dmrs_idx, :]  # (n_f, n_dmrs_sym, n_ant)

        # Prepare transmitted DMRS: expand to full grid with zeros on non-DMRS subcarriers
        x_dmrs_full: npt.NDArray[np.complex64] = np.zeros(
            (n_f, len(dmrs_idx), nl), dtype=np.complex64
        )
        # DMRS tones are at even (grid_cfg=0) or odd (grid_cfg=1) subcarriers within the allocation
        for port_idx_iter in range(nl):
            for sym_idx in range(len(dmrs_idx)):
                x_dmrs_full[dmrs_sc_idxs, sym_idx, port_idx_iter] = x_dmrs__port_sym_dsc[
                    port_idx_iter, sym_idx, :
                ]

        # Reshape y_dmrs to include layer dimension (broadcast across layers)
        # Function expects (n_f, n_sym, nl, n_ant) but we have (n_f, n_sym, n_ant)
        # For single-layer case, expand the layer dimension
        y_dmrs_full_with_layers = np.expand_dims(y_dmrs_full, axis=2)  # (n_f, n_dmrs_sym, 1, n_ant)
        y_dmrs_full_with_layers = np.repeat(
            y_dmrs_full_with_layers, nl, axis=2
        )  # (n_f, n_dmrs_sym, nl, n_ant)

        # Call channel estimator on FULL grid (convert to complex128 as required)
        h_full = phy.pusch.channel_estimation.channel_est_dd(
            x_dmrs=x_dmrs_full.astype(np.complex128),
            y_dmrs=y_dmrs_full_with_layers.astype(np.complex128),
        )  # Returns: (n_f, nl, n_ant)

        # Expand to full grid format (replicate channel estimate across all symbols)
        H_legacy_est__sc_sym_rxant = np.zeros_like(H_dmrs_noisy__sc_sym_rxant)
        for sym in range(H_legacy_est__sc_sym_rxant.shape[1]):
            H_legacy_est__sc_sym_rxant[:, sym, :] = h_full[:, 0, 0]  # Using layer 0

        # Prepare channel for MLIR-TensorRT runtime
        H_dmrs_noisy__ri_sc_sym_rxant = np.stack(
            [H_dmrs_noisy__sc_sym_rxant.real, H_dmrs_noisy__sc_sym_rxant.imag], axis=0
        ).astype(np.float32)
        H_dmrs_noisy__ri_sym_rxant_sc = np.einsum("abcd->acdb", H_dmrs_noisy__ri_sc_sym_rxant)
        H_dmrs_noisy__ri_sym_rxant_sc = np.ascontiguousarray(
            H_dmrs_noisy__ri_sym_rxant_sc, dtype=np.float16
        )

        # Setup dynamic inputs for MLIR-TensorRT runtime
        dynamic_inputs_runtime = replace(
            dynamic_inputs,
            xtf__ri_sym_rxant_sc=H_dmrs_noisy__ri_sym_rxant_sc,
        )
        runtime_inputs = dynamic_inputs_runtime.to_tuple()

        # Execute all channel filter methods
        H_est_methods = {}
        for method in channel_filter_methods:
            # Get executable for this method
            exe = executables[method]
            outputs_ref = outputs_ref_dict[method]

            # Setup output arrays
            outputs_runtime_tuple = tuple(np.zeros_like(o) for o in outputs_ref.to_tuple())

            # Execute
            mtw.execute(
                exe=exe,
                inputs=runtime_inputs,
                outputs=outputs_runtime_tuple,
                sync_stream=True,
                mlir_entry_point="main",
            )

            # Extract channel estimates
            outputs_runtime = ChannelEstimatorOutputs.from_tuple(outputs_runtime_tuple)
            h_interp__ri_port_rxant_sc = outputs_runtime.h_interp__ri_port_rxant_sc
            H_est__port_rxant_sc = (
                h_interp__ri_port_rxant_sc[0] + 1j * h_interp__ri_port_rxant_sc[1]
            )
            H_est_full__sc_sym_rxant = np.transpose(H_est__port_rxant_sc, (2, 0, 1))
            H_est_methods[method] = H_est_full__sc_sym_rxant

        # --------------------------------
        # Plot channel estimates for debugging (first 10 samples only)
        # --------------------------------

        if test_idx < 5:
            taps_to_plot = 200
            linewidth = 1.0

            plt.figure(figsize=(12, 6))

            # Plot noisy CIR
            h_dmrs_noisy__tau = jax.numpy.fft.ifft(H_dmrs_noisy__sc_sym_rxant[:, dmrs_idx[0], 0])
            plt.plot(
                10 * np.log10(np.maximum(np.abs(h_dmrs_noisy__tau), 1e-10))[:taps_to_plot],
                label="Noisy CIR",
                linewidth=linewidth,
                marker="o",
                markersize=0.5,
                markevery=50,
                alpha=0.6,
            )

            # Plot legacy MMSE
            h_legacy_est__tau = jax.numpy.fft.ifft(H_legacy_est__sc_sym_rxant[:, 0, 0])
            plt.plot(
                10 * np.log10(np.maximum(np.abs(h_legacy_est__tau), 1e-10))[:taps_to_plot],
                label="Legacy MMSE",
                linewidth=linewidth,
                marker="^",
                markersize=0.5,
                markevery=50,
                alpha=0.8,
            )

            # Plot all channel filter methods
            markers = ["d", "s", "*", "v", "<", ">"]
            for i, method in enumerate(channel_filter_methods):
                h_est__tau = jax.numpy.fft.ifft(H_est_methods[method][:, 0, 0])
                plt.plot(
                    10 * np.log10(np.maximum(np.abs(h_est__tau), 1e-10))[:taps_to_plot],
                    label=method.replace("_", " ").title(),
                    linewidth=linewidth,
                    marker=markers[i % len(markers)],
                    markersize=0.5,
                    markevery=50,
                    alpha=0.8,
                )

            # Plot true channel
            h_true__tau = jax.numpy.fft.ifft(H_true__sc_sym_rxant[:, dmrs_idx[0], 0])
            plt.plot(
                10 * np.log10(np.maximum(np.abs(h_true__tau), 1e-10))[:taps_to_plot],
                label="True CIR",
                linewidth=linewidth + 0.5,
                linestyle="--",
                alpha=0.9,
            )

            plt.xlabel("Delay Tap")
            plt.ylabel("Magnitude (dB)")
            plt.title(f"Channel Impulse Response Comparison (Sample {test_idx}, SNR={_snr:.1f} dB)")
            plt.legend(loc="best", fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.savefig(
                build_dir / f"h_est_cir_comparison_{test_idx}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # --------------------------------
        # Compute MSE for all methods
        # --------------------------------

        # Get true channel
        sc_idx = np.arange(0, n_f, 1)
        H_true = H_true__sc_sym_rxant[sc_idx, dmrs_idx[0], 0]
        H_legacy_est = H_legacy_est__sc_sym_rxant[sc_idx, 0, 0]

        # Compute overall MSE for legacy MMSE
        mse_legacy = np.mean(np.abs(H_legacy_est - H_true) ** 2 / (np.mean(np.abs(H_true) ** 2)))
        mse_db_legacy__sample.append(10 * np.log10(mse_legacy))

        # Compute overall MSE for all filter methods
        for method in channel_filter_methods:
            H_est = H_est_methods[method][sc_idx, 0, 0]
            mse = np.mean(np.abs(H_est - H_true) ** 2 / (np.mean(np.abs(H_true) ** 2)))
            mse_db_methods[method].append(10 * np.log10(mse))

        # --------------------------------
        # Edge vs Center Analysis (for Gibbs effect)
        # --------------------------------
        edge_size = 200
        edge_left_idx = np.arange(0, edge_size)
        edge_right_idx = np.arange(n_f - edge_size, n_f)
        center_idx = np.arange(edge_size, n_f - edge_size)

        def region_mse(
            H_est: npt.NDArray[np.complexfloating[Any, Any]],
            H_true: npt.NDArray[np.complexfloating[Any, Any]],
            region_idx: npt.NDArray[np.signedinteger[Any]],
        ) -> np.floating[Any]:
            error = np.abs(H_est[region_idx] - H_true[region_idx]) ** 2
            signal_power = np.mean(np.abs(H_true) ** 2)
            return np.mean(error) / signal_power

        # Legacy MMSE edge/center MSE
        mse_legacy_edge_left = region_mse(H_legacy_est, H_true, edge_left_idx)
        mse_legacy_edge_right = region_mse(H_legacy_est, H_true, edge_right_idx)
        mse_legacy_center = region_mse(H_legacy_est, H_true, center_idx)
        mse_db_legacy_edge_left__sample.append(10 * np.log10(mse_legacy_edge_left))
        mse_db_legacy_center__sample.append(10 * np.log10(mse_legacy_center))
        mse_db_legacy_edge_right__sample.append(10 * np.log10(mse_legacy_edge_right))

        # All filter methods edge/center MSE
        for method in channel_filter_methods:
            H_est = H_est_methods[method][sc_idx, 0, 0]
            mse_edge_left = region_mse(H_est, H_true, edge_left_idx)
            mse_edge_right = region_mse(H_est, H_true, edge_right_idx)
            mse_center = region_mse(H_est, H_true, center_idx)
            mse_db_methods_edge_left[method].append(10 * np.log10(mse_edge_left))
            mse_db_methods_center[method].append(10 * np.log10(mse_center))
            mse_db_methods_edge_right[method].append(10 * np.log10(mse_edge_right))

    # ================================================================================================
    # Post-processing: Plot MSE vs SNR for all methods
    # ================================================================================================

    print("\n" + "=" * 80)
    print("Generating comparison plots...")
    print("=" * 80)

    # Convert to numpy arrays
    snr_array = np.array(snr__sample)
    mse_db_legacy_array = np.array(mse_db_legacy__sample)

    # Plot MSE vs SNR comparison
    plt.figure(figsize=(12, 7))

    # Colors for different methods
    colors = ["red", "green", "orange", "purple", "brown", "pink"]
    markers_plot = ["o", "s", "^", "v", "d", "<"]

    # Plot legacy MMSE
    plt.scatter(
        snr_array,
        mse_db_legacy_array,
        label="Legacy MMSE",
        alpha=0.4,
        s=15,
        color="blue",
        marker="x",
    )

    # Plot all filter methods
    for i, method in enumerate(channel_filter_methods):
        mse_db_array = np.array(mse_db_methods[method])
        plt.scatter(
            snr_array,
            mse_db_array,
            label=method.replace("_", " ").title(),
            alpha=0.4,
            s=15,
            color=colors[i % len(colors)],
            marker=markers_plot[i % len(markers_plot)],
        )

    # Add best-fit curves
    snr_range = np.linspace(snr_array.min(), snr_array.max(), 100)

    # Legacy MMSE fit
    legacy_coeffs = np.polyfit(snr_array, mse_db_legacy_array, deg=4)
    legacy_fit = np.polyval(legacy_coeffs, snr_range)
    plt.plot(snr_range, legacy_fit, color="blue", linewidth=2.5, linestyle="--", alpha=0.8)

    # Filter methods fits
    for i, method in enumerate(channel_filter_methods):
        mse_db_array = np.array(mse_db_methods[method])
        coeffs = np.polyfit(snr_array, mse_db_array, deg=4)
        fit = np.polyval(coeffs, snr_range)
        plt.plot(
            snr_range, fit, color=colors[i % len(colors)], linewidth=2.5, linestyle="--", alpha=0.8
        )

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Normalized MSE (dB)", fontsize=12)
    plt.title("Channel Estimation MSE vs SNR - Method Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(build_dir / "mse_comparison_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  ✓ Saved MSE comparison plot")

    # ============================================================================
    # Edge vs Center MSE Comparison (for Gibbs effect analysis) - All Methods
    # ============================================================================

    # Convert to numpy arrays
    mse_db_legacy_edge_left_array = np.array(mse_db_legacy_edge_left__sample)
    mse_db_legacy_center_array = np.array(mse_db_legacy_center__sample)
    mse_db_legacy_edge_right_array = np.array(mse_db_legacy_edge_right__sample)

    # Create figure with 3 subplots (one for each region)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Edge vs Center MSE - Gibbs Effect Comparison (All Methods)", fontsize=15, fontweight="bold"
    )

    region_names = ["Edge Left (PRBs 0-33)", "Center (PRBs 34-239)", "Edge Right (PRBs 240-273)"]
    region_data_legacy = [
        mse_db_legacy_edge_left_array,
        mse_db_legacy_center_array,
        mse_db_legacy_edge_right_array,
    ]
    region_data_methods = [
        mse_db_methods_edge_left,
        mse_db_methods_center,
        mse_db_methods_edge_right,
    ]

    for ax_idx, (ax, title) in enumerate(zip(axes, region_names)):
        # Plot legacy MMSE
        ax.scatter(
            snr_array,
            region_data_legacy[ax_idx],
            alpha=0.4,
            s=12,
            color="blue",
            marker="x",
            label="Legacy MMSE",
        )
        legacy_coeffs = np.polyfit(snr_array, region_data_legacy[ax_idx], deg=4)
        legacy_fit = np.polyval(legacy_coeffs, snr_range)
        ax.plot(snr_range, legacy_fit, color="blue", linewidth=2.5, linestyle="--", alpha=0.8)

        # Plot all filter methods
        for i, method in enumerate(channel_filter_methods):
            method_data = np.array(region_data_methods[ax_idx][method])
            ax.scatter(
                snr_array,
                method_data,
                alpha=0.4,
                s=12,
                color=colors[i % len(colors)],
                marker=markers_plot[i % len(markers_plot)],
                label=method.replace("_", " ").title(),
            )

            # Best fit line
            method_coeffs = np.polyfit(snr_array, method_data, deg=4)
            method_fit = np.polyval(method_coeffs, snr_range)
            ax.plot(
                snr_range,
                method_fit,
                color=colors[i % len(colors)],
                linewidth=2.5,
                linestyle="--",
                alpha=0.8,
            )

        ax.set_xlabel("SNR (dB)", fontsize=11)
        ax.set_ylabel("Normalized MSE (dB)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(build_dir / "mse_edge_vs_center_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved edge vs center comparison plot")

    # ============================================================================
    # Print summary statistics
    # ============================================================================

    print("\n" + "=" * 80)
    print("Performance Summary (averaged across all samples):")
    print("=" * 80)

    # Overall MSE
    print(f"\n{'Method':<30} {'Overall MSE (dB)':>15}")
    print("-" * 80)
    print(f"{'Legacy MMSE':<30} {np.mean(mse_db_legacy_array):>15.2f}")
    for method in channel_filter_methods:
        mse_avg = np.mean(mse_db_methods[method])
        improvement = np.mean(mse_db_legacy_array) - mse_avg
        print(f"{method.replace('_', ' ').title():<30} {mse_avg:>15.2f}  (Δ={improvement:+.2f} dB)")

    # Edge vs center analysis
    print(
        f"\n{'Method':<30} {'Edge-L (dB)':>12} {'Center (dB)':>12} {'Edge-R (dB)':>12} {'Avg Penalty':>12}"
    )
    print("-" * 80)

    # Legacy MMSE
    legacy_edge_penalty = (
        np.mean(mse_db_legacy_edge_left_array)
        - np.mean(mse_db_legacy_center_array)
        + np.mean(mse_db_legacy_edge_right_array)
        - np.mean(mse_db_legacy_center_array)
    ) / 2
    print(
        f"{'Legacy MMSE':<30} "
        f"{np.mean(mse_db_legacy_edge_left_array):>12.2f} "
        f"{np.mean(mse_db_legacy_center_array):>12.2f} "
        f"{np.mean(mse_db_legacy_edge_right_array):>12.2f} "
        f"{legacy_edge_penalty:>12.2f}"
    )

    # All filter methods
    for method in channel_filter_methods:
        edge_left_avg = np.mean(mse_db_methods_edge_left[method])
        center_avg = np.mean(mse_db_methods_center[method])
        edge_right_avg = np.mean(mse_db_methods_edge_right[method])
        edge_penalty = ((edge_left_avg - center_avg) + (edge_right_avg - center_avg)) / 2

        print(
            f"{method.replace('_', ' ').title():<30} "
            f"{edge_left_avg:>12.2f} "
            f"{center_avg:>12.2f} "
            f"{edge_right_avg:>12.2f} "
            f"{edge_penalty:>12.2f}"
        )

    print("\n" + "=" * 80)
    print("Testing completed successfully!")
    print("=" * 80)
