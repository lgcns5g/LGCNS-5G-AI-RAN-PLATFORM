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
Test function for phy.jax.pusch.pusch_inner_rx pipeline including de-rate-matcher,
de-scrambler, codeblock concatenation, and LDPC decoder.
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path

# Choose JAX backend
os.environ["JAX_PLATFORMS"] = "cuda"  # "cpu, cuda, or mlir_tensorrt"

# Limit JAX GPU memory pre-allocation to prevent OOM issues
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

import jax
import numpy as np
import pytest
from jax import export

# Test vectors
import test_vectors as tvs

# 5G reference model outer receiver
from ran.phy.numpy.pusch.outer_receiver import (
    OuterRxParams,
    pusch_outer_rx,
)

# Optimized PUSCH Rx inner reveiver using JAX and MLIR-TensorRT
from ran import mlir_trt_wrapper as mtw
from ran.phy.jax.pusch.pusch_inner_receiver import (
    pusch_inner_rx,
    PuschInnerRxDynamicInputs,
    PuschInnerRxStaticInputs,
    PuschInnerRxOutputs,
)

# TensorRT plugin configurations (needed for MLIR-TensorRT compiler)
from ran.trt_plugins.dmrs import dmrs_3276
from ran.trt_plugins.fft import fft_2048, ifft_2048
from ran.trt_plugins.cholesky_factor_inv import cholesky_inv_4x4
from ran.phy.jax.pusch import ai_tukey_filter
from ran.phy.jax.pusch.free_energy_filter import FreeEnergyFilterConfig
from ran.phy.jax.pusch.weighted_threshold_filter import WeightedThresholdFilterConfig

# TensorRT plugin manager utilities
from ran.trt_plugins.manager.trt_plugin_manager import (
    copy_trt_engine_for_cpp_tests,
    copy_test_data_for_cpp_tests,
    get_ran_pytest_build_dir,
    should_skip_engine_generation,
)

# Configure logger
logger = logging.getLogger(__name__)

# Check if engines already exist (evaluated at module import time)
_skip_engine_gen = should_skip_engine_generation(
    [
        # "pusch_inner_receiver_ai_tukey_filter.trtengine",
        "pusch_inner_receiver_free_energy_filter.trtengine",
        "pusch_inner_receiver_weighted_threshold_filter.trtengine",
    ]
)

# Skip all tests in this module if engines already exist
pytestmark = pytest.mark.skipif(
    _skip_engine_gen,
    reason="TRT engines already exist and SKIP_TRT_ENGINE_GENERATION=1",
)

# -----------------------------------------------------------------------------
# Test inner receiver with 5G reference model outer receiver
# -----------------------------------------------------------------------------


def test_inner_receiver() -> None:
    """
    Test the optimized PUSCH Rx pipeline inner receiver

    The inner receiver includes DMRS generation, channel estimation,
    covariance estimation, noise variance, RSRP, SINR estimation, equalization, soft
    demapper. The optimized inner receiver is compiled to a TensorRT engine and tested
    end-to-end with the 5G reference model outer receiver.
    """

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Suppress verbose JAX debug logs
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src.dispatch").setLevel(logging.WARNING)
    logging.getLogger("jax._src.compiler").setLevel(logging.WARNING)

    # --------------------------------
    # Test parameters
    # --------------------------------

    TV_NAME = "TVnr_7204_cuPhyMax.h5"  # TVnr_7201_cuPhyMax.h5

    mlir_tensorrt_compilation_flags = [
        "tensorrt-builder-opt-level=0",  # 0 (debug), 5 (release)
        "tensorrt-workspace-memory-pool-limit=50MiB",
    ]

    # Define channel filter methods to test
    channel_filter_methods = [
        # "ai_tukey_filter",
        "free_energy_filter",
        "weighted_threshold_filter",
    ]

    # --------------------------------
    # Setup build directory
    # --------------------------------

    ran_pytest_build_dir = get_ran_pytest_build_dir()
    build_dir = (
        Path(ran_pytest_build_dir) / "tests" / "phy" / "jax" / "pusch" / "pusch_inner_receiver"
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------------
    # 5G reference model test vector
    # -----------------------------------------------------------------------------

    tv = tvs.TvLoader.load(TV_NAME)

    # Extract DMRS settings from the test vector
    dmrs_kwargs = tvs.build_dmrs_kwargs(tv)

    n_t = np.int32(tv["Nt"][0][0])  # Number of OFDM symbols per slot
    n_f = np.int32(dmrs_kwargs["n_f"])  # Number of sub-carriers
    n_dmrs_id = np.int32(dmrs_kwargs["n_dmrs_id"])
    slot_number = np.int32(dmrs_kwargs["slot_number"])

    # Extract channel estimation settings from the test vector
    ls_kwargs = tvs.build_ls_kwargs(tv)

    n_prb = np.int32(ls_kwargs["n_prb"])  # Number of PRBs
    # Start PRB index for PUSCH allocation
    start_prb = np.int32(ls_kwargs["start_prb"])
    nl = np.int32(ls_kwargs["nl"])  # Number of layers
    port_idx = tuple(port for port in ls_kwargs["port_idx"])  # DMRS port numbers
    vec_scid = tuple(scid for scid in ls_kwargs["vec_scid"])  # Scrambling id
    dmrs_idx = tuple(dmrs_idx for dmrs_idx in ls_kwargs["sym_idx_dmrs"])
    energy = 2.0
    apply_cov_shrinkage = True

    # Soft demapper
    qam_bits = int(tv["qam"].item())
    n_ue = np.int32(1)
    layer2ue = tuple([0] * int(nl))

    # Data symbol indices: all symbols except DMRS symbols
    all_symbols = set(range(n_t))
    dmrs_symbols = set(dmrs_idx)
    data_sym_idxs = tuple(sorted(all_symbols - dmrs_symbols))

    # Reference model outer receiver parameters
    outer_rx_params = OuterRxParams(
        # Descramble
        n_id=int(tv["N_id"].item()),
        n_rnti=int(tv["n_rnti"].item()),
        # Derate match
        bgn=int(tv["BGN"].item()),
        c=int(tv["C"].item()),
        k=int(tv["K"].item()),
        f=int(tv["F"].item()),
        k_prime=int(tv["K_prime"].item()),
        zc=int(tv["Zc"].item()),
        nl=int(nl),
        rv_idx=int(tv["rvIdx"].item()),
        nref=int(tv["Nref"].item()),
        g=int(tv["G"].item()),
        # LDPC decode
        i_ls=int(tv["i_LS"].item()),
        max_num_itr_cbs=int(tv["maxNumItr_CBs"].flatten()[0].item()),
        # CRC decode
        crc_name=np.asarray(tv["CRC"]).tobytes().decode("ascii"),
    )

    # Get channel from test vector: Downcast from complex128 to complex64 and preserve
    # column-major layout.
    xtf__sc_sym_rxant = np.array(ls_kwargs["xtf"], dtype=np.complex64, order="F")

    # Split complex into real/imag on first axis and cast to float16
    xtf__ri_sc_sym_rxant = np.ascontiguousarray(
        np.stack([xtf__sc_sym_rxant.real, xtf__sc_sym_rxant.imag], axis=0), dtype=np.float16
    )

    # Convert to row-major for TensorRT engine (maintain data layout).
    xtf__rxant_sym_sc_ri = np.ascontiguousarray(
        np.einsum("abcd->dcba", xtf__ri_sc_sym_rxant, order="C")
    )

    # -----------------------------------------------------------------------------
    # Compile TensorRT executables for each channel filter method
    # -----------------------------------------------------------------------------

    # Dynamic inputs (same for all methods)
    dynamic_inputs = PuschInnerRxDynamicInputs(
        xtf__rxant_sym_sc_ri=xtf__rxant_sym_sc_ri,
    )
    dynamic_inputs_pos = dynamic_inputs.to_tuple()

    # TensorRT plugin configuration (same for all methods)
    trt_plugin_configs = {
        "tensorrt_dmrs_plugin": dmrs_3276.get_config(),
        "tensorrt_fft_plugin_forward": fft_2048.get_config(),
        "tensorrt_fft_plugin_inverse": ifft_2048.get_config(),
        "tensorrt_cholesky_inv_plugin": cholesky_inv_4x4.get_config(),
    }

    # Dictionaries to store MLIR-TrT executables and reference outputs for each
    # channel filter method
    executables = {}
    outputs_ref_dict = {}

    logger.info("\n" + "=" * 80)
    logger.info("Compiling TensorRT executables for each channel filter method...")
    logger.info("=" * 80)

    # Store MLIR and build directories for parallel compilation
    mlir_modules = {}
    build_dirs = {}

    # Sequential: Setup, export to MLIR for each method
    for method in channel_filter_methods:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Channel filter method: {method}")
        logger.info(f"{'=' * 40}")

        # Create build directory
        method_build_dir = build_dir / method
        if method_build_dir.exists():
            logger.info(f"  Cleaning existing directory: {method_build_dir}")
            shutil.rmtree(method_build_dir)
        method_build_dir.mkdir(parents=True, exist_ok=True)
        build_dirs[method] = method_build_dir

        # Save input data to build directory for C++ driver
        logger.info("  Saving input data for C++ driver...")
        xtf_file = method_build_dir / "xtf_input.bin"
        xtf_meta_file = method_build_dir / "xtf_input_meta.txt"
        xtf__rxant_sym_sc_ri.tofile(xtf_file)

        with open(xtf_meta_file, "w") as f:
            f.write("# xtf input data metadata\n")
            f.write(f"shape: {xtf__rxant_sym_sc_ri.shape}\n")
            f.write(f"dtype: {xtf__rxant_sym_sc_ri.dtype}\n")
            f.write("format: Row-major (n_rxant, n_sym, n_sc, real/imag)\n")
            f.write(f"n_ri: {xtf__rxant_sym_sc_ri.shape[0]}\n")
            f.write(f"n_sc: {xtf__rxant_sym_sc_ri.shape[1]}\n")
            f.write(f"n_sym: {xtf__rxant_sym_sc_ri.shape[2]}\n")
            f.write(f"n_rxant: {xtf__rxant_sym_sc_ri.shape[3]}\n")
            f.write(f"size_bytes: {xtf__rxant_sym_sc_ri.nbytes}\n")

        # Create channel filter config if needed
        channel_filter_config: (
            ai_tukey_filter.AITukeyFilterConfig
            | FreeEnergyFilterConfig
            | WeightedThresholdFilterConfig
            | None
        ) = None

        if method == "ai_tukey_filter":
            # Create AI Tukey filter configuration using trained model
            model_path = ai_tukey_filter.get_pretrained_ai_tukey_filter_path()

            # Model hyperparameters loaded from model_config.yaml in model_dir
            channel_filter_config = ai_tukey_filter.AITukeyFilterConfig(
                model_dir=str(model_path),
                fft_size=2048,
            )
            logger.info(f"  Using pretrained model: {model_path.name}")
            logger.info(f"  Model directory: {model_path}")
        elif method == "free_energy_filter":
            # Create config with explicit values to test static args compilation
            channel_filter_config = FreeEnergyFilterConfig(
                fft_size=2048,  # Only 2048 supported for now.
                alpha=2.0,
                tau_min=0,
                tau_max_absolute=1024,
                delay_compensation_samples=50.0,
            )
            logger.info(f"  Using Free Energy filter config: {channel_filter_config}")
        elif method == "weighted_threshold_filter":
            # Create config with explicit values to test static args compilation
            channel_filter_config = WeightedThresholdFilterConfig(
                fft_size=2048,
                delay_compensation_samples=50.0,
                decay_rate=0.01,
                k_sigma=3.0,
            )
            logger.info(f"  Using Weighted Threshold filter config: {channel_filter_config}")

        # Create static inputs for this method
        static_inputs = PuschInnerRxStaticInputs(
            slot_number=int(slot_number),
            n_dmrs_id=int(n_dmrs_id),
            rww_regularizer_val=1e-8,
            start_prb=int(start_prb),
            nl_offset=0,
            scids=vec_scid,
            apply_cov_shrinkage=apply_cov_shrinkage,
            channel_filter_method=method,
            qam_bits=qam_bits,
            dmrs_sym_idxs=dmrs_idx,
            data_sym_idxs=data_sym_idxs,
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

        # Get reference outputs using CPU, CUDA, or MLIR-TensorRT backends
        logger.info("  Computing reference outputs...")
        outputs_ref_tuple = pusch_inner_rx(*all_inputs)
        outputs_ref = PuschInnerRxOutputs.from_tuple(outputs_ref_tuple)
        outputs_ref_dict[method] = outputs_ref

        # Clear JAX cache to avoid conflicts between methods
        # jax.clear_caches()

        # JIT and export to StableHLO MLIR
        logger.info("  Exporting to StableHLO MLIR...")
        jit_pusch_inner_rx = jax.jit(pusch_inner_rx, static_argnums=static_argnums)

        exported = jax.export.export(
            jit_pusch_inner_rx,
            disabled_checks=[
                export.DisabledSafetyCheck.custom_call("tensorrt_dmrs_plugin"),
                export.DisabledSafetyCheck.custom_call("tensorrt_fft_plugin"),
                export.DisabledSafetyCheck.custom_call("tensorrt_cholesky_inv_plugin"),
            ],
        )(*all_inputs)

        stablehlo_mlir = exported.mlir_module()
        mlir_modules[method] = stablehlo_mlir

        with open(method_build_dir / "stablehlo_mlir.mlir", "w") as f:
            f.write(stablehlo_mlir)

    # Parallel: Compile all MLIR modules to TensorRT engines
    logger.info("\n" + "=" * 80)
    logger.info(f"Compiling {len(channel_filter_methods)} TensorRT engines in parallel...")
    logger.info("=" * 80)

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
            name=f"pusch_inner_rx_{method}",
            export_path=method_build_dir,
            mlir_entry_point="main",
            mlir_tensorrt_compilation_flags=method_compilation_flags,
            trt_plugin_configs=trt_plugin_configs,  # type: ignore[arg-type]
        )

        compilation_time = time.time() - start_time
        logger.info(f"  Compiled {method} in {compilation_time:.2f}s")
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
                logger.error(f"  Compilation failed for {method}: {exc}")
                raise

    parallel_total = time.time() - parallel_start
    logger.info(f"Parallel compilation completed in {parallel_total:.2f}s")

    # Sequential: Verification and post-processing for each method
    for method in channel_filter_methods:
        logger.info(f"\n  Verifying and post-processing {method}...")
        method_build_dir = build_dirs[method]
        exe = executables[method]
        outputs_ref = outputs_ref_dict[method]

        # Verify correctness against reference
        logger.info("    Verifying correctness...")
        outputs_tuple = tuple(np.zeros_like(o) for o in outputs_ref.to_tuple())

        mtw.execute(
            exe=exe,
            inputs=dynamic_inputs_pos,
            outputs=outputs_tuple,
            sync_stream=True,
            mlir_entry_point="main",
        )
        outputs_trt = PuschInnerRxOutputs.from_tuple(outputs_tuple)

        # Log max diff for informational purposes (outliers expected)
        for field in fields(PuschInnerRxOutputs):
            field_name = field.name
            ref_val = getattr(outputs_ref, field_name)
            trt_val = getattr(outputs_trt, field_name)
            max_diff: float = float(np.max(np.abs(trt_val - ref_val)))
            mean_diff: float = float(np.mean(np.abs(trt_val - ref_val)))
            logger.info(
                f"    {field_name}: max error = {max_diff:.6f}, mean error = {mean_diff:.6f}"
            )

            # Determine tolerance based on field type
            # Looser tolerance for LLRs since only signs matter for decoding
            rtol, atol = (  # (0.5, 5.0)
                (3.0, 200.0) if field_name == "llr__time_allocfreq_layer_qambit" else (1e-2, 1e-2)
            )

            # Assert field matches with appropriate tolerance
            np.testing.assert_allclose(
                trt_val,
                ref_val,
                rtol=rtol,
                atol=atol,
                err_msg=f"{method}: {field_name} mismatch between TRT and reference",
            )

        logger.info(f"    Compilation and execution completed for {method}")

        # Save reference outputs for C++ driver validation
        logger.info("    Saving reference outputs for C++ driver...")

        # Save LLR output
        llr_file = method_build_dir / "llr_output.bin"
        llr_meta_file = method_build_dir / "llr_output_meta.txt"
        outputs_trt.llr__time_allocfreq_layer_qambit.tofile(llr_file)
        with open(llr_meta_file, "w") as f:
            llr_out = outputs_trt.llr__time_allocfreq_layer_qambit
            f.write("# LLR output data metadata\n")
            f.write(f"shape: {llr_out.shape}\n")
            f.write(f"dtype: {llr_out.dtype}\n")
            f.write("format: column-major (n_datasym, n_allocsc, n_layer, qam_bits)\n")
            f.write(f"n_datasym: {llr_out.shape[0]}\n")
            f.write(f"n_allocsc: {llr_out.shape[1]}\n")
            f.write(f"n_layer: {llr_out.shape[2]}\n")
            f.write(f"qam_bits: {llr_out.shape[3]}\n")
            f.write(f"size_bytes: {llr_out.nbytes}\n")

        # Save post-EQ noise variance (averaged across symbols for C++ driver)
        post_eq_noise_var_db_avg = np.mean(outputs_trt.post_eq_noise_var_db__ue_sym)
        noise_file = method_build_dir / "post_eq_noise_var_db_output.bin"
        noise_meta_file = method_build_dir / "post_eq_noise_var_db_output_meta.txt"
        np.array([post_eq_noise_var_db_avg], dtype=np.float32).tofile(noise_file)
        with open(noise_meta_file, "w") as f:
            f.write("# Post-EQ noise variance output (averaged)\n")
            f.write("shape: (1,)\n")
            f.write("dtype: float32\n")
            f.write(f"value_db: {post_eq_noise_var_db_avg:.6f}\n")
            noise_var_shape = outputs_trt.post_eq_noise_var_db__ue_sym.shape
            f.write(f"original_shape: {noise_var_shape}\n")

        # Save post-EQ SINR (averaged across symbols for C++ driver)
        post_eq_sinr_db_avg = np.mean(outputs_trt.post_eq_sinr_db__ue_sym)
        sinr_file = method_build_dir / "post_eq_sinr_db_output.bin"
        sinr_meta_file = method_build_dir / "post_eq_sinr_db_output_meta.txt"
        np.array([post_eq_sinr_db_avg], dtype=np.float32).tofile(sinr_file)
        with open(sinr_meta_file, "w") as f:
            f.write("# Post-EQ SINR output (averaged)\n")
            f.write("shape: (1,)\n")
            f.write("dtype: float32\n")
            f.write(f"value_db: {post_eq_sinr_db_avg:.6f}\n")
            sinr_shape = outputs_trt.post_eq_sinr_db__ue_sym.shape
            f.write(f"original_shape: {sinr_shape}\n")

        # --------------------------------
        # Copy TensorRT engine and test data to location expected by C++ test
        # --------------------------------

        engine_dest = copy_trt_engine_for_cpp_tests(
            method_build_dir, f"pusch_inner_receiver_{method}.trtengine"
        )
        logger.info(f"  Copied TensorRT engine to {engine_dest}")

        # Copy test input/output data files for C++ validation
        test_data_dest = copy_test_data_for_cpp_tests(
            method_build_dir, f"pusch_inner_receiver/{method}", ["*.bin", "*_meta.txt"]
        )
        logger.info(f"  Copied test data files to {test_data_dest}")

    # --------------------------------
    # Outer receiver verification for all methods
    # --------------------------------

    outer_rx_results = {}  # Store results for each method

    for method in channel_filter_methods:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Outer Receiver Verification: {method}")
        logger.info(f"{'=' * 80}")

        exe = executables[method]
        outputs_ref = outputs_ref_dict[method]
        method_build_dir = build_dir / method

        # Re-execute TRT for this method
        logger.info("  Executing TRT compiled model...")
        outputs_tuple = tuple(np.zeros_like(o) for o in outputs_ref.to_tuple())
        mtw.execute(
            exe=exe,
            inputs=dynamic_inputs_pos,
            outputs=outputs_tuple,
            sync_stream=True,
            mlir_entry_point="main",
        )
        outputs_trt = PuschInnerRxOutputs.from_tuple(outputs_tuple)

        # --------------------------------
        # Post equalization performance metrics
        # --------------------------------

        logger.info("\n  Post-EQ Performance Metrics:")
        logger.info(f"  {'-' * 60}")

        if "postEqNoiseVardB" in tv:
            post_noise_tv = np.atleast_1d(np.squeeze(tv["postEqNoiseVardB"]))
            post_noise_trt = np.atleast_1d(np.squeeze(outputs_trt.post_eq_noise_var_db__ue_sym))
            post_noise_ref = np.atleast_1d(np.squeeze(outputs_ref.post_eq_noise_var_db__ue_sym))

            logger.info("    Noise Variance (dB):")
            logger.info(f"      TV:  {np.mean(post_noise_tv):.3f} dB")
            noise_trt_diff = np.mean(post_noise_trt - post_noise_tv)
            logger.info(
                f"      TRT: {np.mean(post_noise_trt):.3f} dB (diff: {noise_trt_diff:.3f} dB)"
            )
            noise_ref_diff = np.mean(post_noise_ref - post_noise_tv)
            logger.info(
                f"      Ref: {np.mean(post_noise_ref):.3f} dB (diff: {noise_ref_diff:.3f} dB)"
            )

            # Assert noise variance matches test vector within tolerance
            for output_val, output_name in [(post_noise_trt, "TRT"), (post_noise_ref, "Reference")]:
                np.testing.assert_allclose(
                    output_val,
                    post_noise_tv,
                    rtol=0.1,
                    atol=2.0,
                    err_msg=f"{method}: {output_name} post-EQ noise variance mismatch with test vector",
                )

        if "postEqSinrdB" in tv:
            post_sinr_tv = np.atleast_1d(np.squeeze(tv["postEqSinrdB"]))
            post_sinr_trt = np.atleast_1d(np.squeeze(outputs_trt.post_eq_sinr_db__ue_sym))
            post_sinr_ref = np.atleast_1d(np.squeeze(outputs_ref.post_eq_sinr_db__ue_sym))

            logger.info("    SINR (dB):")
            logger.info(f"      TV:  {np.mean(post_sinr_tv):.3f} dB")
            sinr_trt_diff = np.mean(post_sinr_trt - post_sinr_tv)
            logger.info(
                f"      TRT: {np.mean(post_sinr_trt):.3f} dB (diff: {sinr_trt_diff:.3f} dB)"
            )
            sinr_ref_diff = np.mean(post_sinr_ref - post_sinr_tv)
            logger.info(
                f"      Ref: {np.mean(post_sinr_ref):.3f} dB (diff: {sinr_ref_diff:.3f} dB)"
            )

            # Assert SINR matches test vector within tolerance
            for output_val, output_name in [(post_sinr_trt, "TRT"), (post_sinr_ref, "Reference")]:
                np.testing.assert_allclose(
                    output_val,
                    post_sinr_tv,
                    rtol=0.1,
                    atol=2.0,
                    err_msg=f"{method}: {output_name} post-EQ SINR mismatch with test vector",
                )

        # --------------------------------
        # Outer receiver verification
        # --------------------------------

        logger.info("\n  Outer Receiver Verification:")
        logger.info(f"  {'-' * 60}")

        # Apply outer receiver processing to TRT output
        outer_rx_outputs_trt = pusch_outer_rx(
            llr__time_allocfreq_layer_qambit=outputs_trt.llr__time_allocfreq_layer_qambit.astype(
                np.float32
            ),
            outer_rx_params=outer_rx_params,
            qam_bits=qam_bits,
        )

        # Apply outer_rx processing to reference output
        outer_rx_outputs_ref = pusch_outer_rx(
            llr__time_allocfreq_layer_qambit=outputs_ref.llr__time_allocfreq_layer_qambit,
            outer_rx_params=outer_rx_params,
            qam_bits=qam_bits,
        )

        outer_rx_trt_passed = outer_rx_outputs_trt.tb_err == 0
        outer_rx_ref_passed = outer_rx_outputs_ref.tb_err == 0

        logger.info(
            f"    TRT: TB error = {outer_rx_outputs_trt.tb_err}, "
            f"CB errors = {outer_rx_outputs_trt.cb_err}"
        )
        logger.info(
            f"    Ref: TB error = {outer_rx_outputs_ref.tb_err}, "
            f"CB errors = {outer_rx_outputs_ref.cb_err}"
        )

        # Assert outer receiver passes for both TRT and reference
        for passed, output, output_name in [
            (outer_rx_trt_passed, outer_rx_outputs_trt, "TRT"),
            (outer_rx_ref_passed, outer_rx_outputs_ref, "Reference"),
        ]:
            assert passed, (
                f"{method}: {output_name} outer receiver failed with TB error = {output.tb_err}"
            )

        # Compare with test vector if available
        if "tbErr" in tv:
            tv_tb_err = int(tv["tbErr"].item())
            logger.info(f"    TV:  TB error = {tv_tb_err}")

            if outer_rx_outputs_trt.tb_err == tv_tb_err:
                logger.info("    ✓ TRT TB error matches TV")
            else:
                logger.warning(
                    f"    ✗ TRT TB error MISMATCH (TRT={outer_rx_outputs_trt.tb_err}, TV={tv_tb_err})"
                )

            # Assert TB error matches test vector
            for output, output_name in [
                (outer_rx_outputs_trt, "TRT"),
                (outer_rx_outputs_ref, "Reference"),
            ]:
                assert output.tb_err == tv_tb_err, (
                    f"{method}: {output_name} TB error ({output.tb_err}) does not match test vector ({tv_tb_err})"
                )

        # Store results for summary
        outer_rx_results[method] = {
            "outer_rx_trt_passed": outer_rx_trt_passed,
            "outer_rx_ref_passed": outer_rx_ref_passed,
            "outputs_trt": outputs_trt,
            "outputs_ref": outputs_ref,
            "outer_rx_trt": outer_rx_outputs_trt,
            "outer_rx_ref": outer_rx_outputs_ref,
        }

        # Method summary
        logger.info("\n  Method Summary:")
        outer_trt_status = "PASSED" if outer_rx_trt_passed else "FAILED"
        logger.info(f"    Outer Receiver TRT: {outer_trt_status}")
        outer_ref_status = "PASSED" if outer_rx_ref_passed else "FAILED"
        logger.info(f"    Outer Receiver Ref: {outer_ref_status}")

    # --------------------------------
    # LLR Comparison Summary (using first channel filter method)
    # --------------------------------

    first_method = channel_filter_methods[0]
    first_result = outer_rx_results[first_method]
    outputs_ref = first_result["outputs_ref"]  # type: ignore[assignment]
    outputs_trt = first_result["outputs_trt"]  # type: ignore[assignment]

    logger.info("\n" + "=" * 60)
    logger.info(f"LLR Comparison Summary ({first_method})")
    logger.info("=" * 60)

    # Helper function to compare LLRs
    def compare_llrs(
        llr1: np.ndarray,
        llr2: np.ndarray,
        name1: str,
        name2: str,
        show_sign_flip_pattern: bool = False,
    ) -> None:
        diff = llr1 - llr2
        abs_diff = np.abs(diff)
        sign_flip_mask = np.sign(llr1) != np.sign(llr2)
        sign_flips: int = int(np.sum(sign_flip_mask))

        logger.info(f"\n{name1} vs {name2}:")
        logger.info(f"  Max diff: {np.max(abs_diff):.3f}")
        logger.info(f"  Mean diff: {np.mean(diff):.3f}")
        logger.info(f"  Sign flips: {sign_flips} ({100 * sign_flips / llr1.size:.2f}%)")

        # Show examples of largest differences
        if np.max(abs_diff) > 1.0:
            large_diff_idx = np.argsort(abs_diff.flatten())[-3:]  # Top 3
            logger.debug("  Examples of largest diffs:")
            for i in large_diff_idx[::-1]:  # Reverse to show largest first
                loc = np.unravel_index(i, llr1.shape)
                logger.debug(
                    f"    {loc}: {name1}={llr1[loc]:.3f}, {name2}={llr2[loc]:.3f}, "
                    f"diff={diff[loc]:.3f}"
                )

        # Show sign flip pattern if requested
        if show_sign_flip_pattern and sign_flips > 0:
            logger.debug("  Sign flip examples (first 10):")
            sign_flip_indices = np.where(sign_flip_mask.flatten())[0][:10]
            for i in sign_flip_indices:
                loc = np.unravel_index(i, llr1.shape)
                logger.debug(f"    Idx {i}: {name1}={llr1[loc]:.3f}, {name2}={llr2[loc]:.3f}")

    # Get TV LLRs if available
    if "LLRseq" in tv:
        tv_llr_flat = tv["LLRseq"][:, 0]  # Flatten to 1D

        # Convert column-major outputs to row-major for comparison with TV
        # Row-major format: (qam_bits, layer, freq, time)
        # Input: (time, freq, layer, qambit) -> Output: (qambit, layer, freq, time)
        ref_llr_row = np.einsum("abcd->dcba", outputs_ref.llr__time_allocfreq_layer_qambit)
        trt_llr_row = np.einsum("abcd->dcba", outputs_trt.llr__time_allocfreq_layer_qambit)

        # Flatten for comparison with TV (Fortran order to match TV)
        ref_llr_flat = ref_llr_row.flatten(order="F")
        trt_llr_flat = trt_llr_row.flatten(order="F")

        # Ensure same length
        min_len = min(len(tv_llr_flat), len(ref_llr_flat), len(trt_llr_flat))
        tv_llr_flat = tv_llr_flat[:min_len]
        ref_llr_flat = ref_llr_flat[:min_len]
        trt_llr_flat = trt_llr_flat[:min_len]

        # 1. TRT vs TV
        compare_llrs(trt_llr_flat, tv_llr_flat, "TRT", "TV", show_sign_flip_pattern=True)

        # 2. TRT vs Ref
        compare_llrs(trt_llr_flat, ref_llr_flat, "TRT", "Ref", show_sign_flip_pattern=True)

        # 3. Ref vs TV
        compare_llrs(ref_llr_flat, tv_llr_flat, "Ref", "TV", show_sign_flip_pattern=True)

        # Additional pattern analysis: check if sign flips correlate with bit positions
        logger.info("\n" + "=" * 60)
        logger.info("Sign Flip Pattern Analysis (Ref vs TV)")
        logger.info("=" * 60)

        # Reshape to see bit-level pattern (use qam_bits from outer scope)
        # ref_llr_flat and tv_llr_flat already have same length (min_len)
        samples_per_bit = min_len // qam_bits

        # Validate divisibility and warn if data will be truncated
        if min_len % qam_bits != 0:
            truncated_samples = min_len % qam_bits
            logger.warning(
                f"  min_len ({min_len}) not evenly divisible by qam_bits ({qam_bits}). "
                f"Truncating {truncated_samples} samples for bit-level analysis."
            )

        ref_by_bit = ref_llr_flat[: samples_per_bit * qam_bits].reshape(qam_bits, samples_per_bit)
        tv_by_bit = tv_llr_flat[: samples_per_bit * qam_bits].reshape(qam_bits, samples_per_bit)

        for bit_idx in range(qam_bits):
            bit_sign_flips: int = int(np.sum((ref_by_bit[bit_idx] * tv_by_bit[bit_idx]) < 0))
            total_for_bit = ref_by_bit[bit_idx].size
            pct = 100 * bit_sign_flips / total_for_bit
            logger.info(f"  Bit {bit_idx}: {bit_sign_flips} sign flips ({pct:.1f}%)")

            # Show a few examples for this bit
            flip_mask = (ref_by_bit[bit_idx] * tv_by_bit[bit_idx]) < 0
            flip_indices = np.where(flip_mask)[0][:3]
            if len(flip_indices) > 0:
                logger.debug("    Examples:")
                for idx in flip_indices:
                    logger.debug(
                        f"      Ref={ref_by_bit[bit_idx, idx]:.3f}, TV={tv_by_bit[bit_idx, idx]:.3f}"
                    )
    else:
        # Just compare TRT vs Ref if no TV available
        ref_llr_flat = outputs_ref.llr__time_allocfreq_layer_qambit.flatten()
        trt_llr_flat = outputs_trt.llr__time_allocfreq_layer_qambit.flatten()
        compare_llrs(trt_llr_flat, ref_llr_flat, "TRT", "Ref")

    # -----------------------------------------------------------------------------
    # Final Summary - All Methods
    # -----------------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("Final Summary - All Methods")
    logger.info("=" * 80)

    all_methods_passed = True
    for method in channel_filter_methods:
        result = outer_rx_results[method]
        outer_rx_trt_passed = result["outer_rx_trt_passed"]  # type: ignore[assignment]
        outer_rx_ref_passed = result["outer_rx_ref_passed"]  # type: ignore[assignment]

        logger.info(f"\n{method}:")
        logger.info(f"  Outer Receiver TRT: {'PASSED' if outer_rx_trt_passed else 'FAILED'}")
        logger.info(f"  Outer Receiver Ref: {'PASSED' if outer_rx_ref_passed else 'FAILED'}")

        # Update overall pass status
        if not (outer_rx_trt_passed and outer_rx_ref_passed):
            all_methods_passed = False

    logger.info("\n" + "=" * 80)
    if all_methods_passed:
        logger.info("SUCCESS: All methods passed full end-to-end pipeline verification!")
    else:
        logger.error("FAILURE: Some methods failed verification")
    logger.info("=" * 80)

    # Final assertion to ensure all methods passed
    assert all_methods_passed, "Some channel filter methods failed end-to-end verification"
