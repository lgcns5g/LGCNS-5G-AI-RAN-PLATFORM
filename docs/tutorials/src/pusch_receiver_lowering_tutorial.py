# %% [raw] tags=["remove-cell"]
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

# %% [markdown] tags=["keep-output"]
# # 4. PUSCH Receiver Lowering Tutorial
#
# ## Step 1: Introduction
#
#    ![High-level overview of the PUSCH inner receiver](
#     ../../figures/generated/pusch_inner_receiver_phy_v2.drawio.svg
#     )
#
# This tutorial demonstrates how to compile and benchmark the example
# PUSCH inner receiver pipeline shown in the figure above.
# On a high-level, the workflow will be the
# same as the [MLIR-TensorRT Tutorial](mlir_trt_tutorial.ipynb);
# that is,
# - We define a JIT-able JAX function that implements the whole PUSCH inner
#   receiver pipeline.
# - We export the function to StableHLO.
# - We compile the function using MLIR-TensorRT and TensorRT.
#
# We, therefore, recommend that you take a look at the
# [MLIR-TensorRT Tutorial](mlir_trt_tutorial.ipynb) first.
#
# Our example inner receiver is bootstrapped from
# NVIDIA's existing cuPHY implementation, which is
# based on the
# [5G NR FAPI specification](https://www.smallcellforum.org/technology/5g-fapi-standard/).
# The FAPI specification has many parameters
# with strict typing requirements, so we use Python
# dataclasses to manage the input and output parameters. We
# will show you how to distinguish between static (compile-time) inputs
# and dynamic (runtime) inputs. Static arguments get included in
# StableHLO as constants and then baked into the generated TensorRT
# engine by the TensorRT compiler.
#
# We have tried to keep the example inner receiver implementation simple
# and modular. It is composed of the following blocks:
# 1. **DMRS generation**: Generate DMRS (including Gold Sequence
#    and QPSK mapping).
# 2. **DMRS FOCC and TOCC**: Apply frequency and time orthogonal cover codes
#    to the DMRS.
# 3. **DMRS extraction**: Extract DMRS from the received resource grid
#    (select and copy-out DMRS REs from the full OFDM resource grid).
# 4. **LS channel estimation**: Here we simply multiply the received DMRS
#    by the conjugate of the corresponding DMRS (matched filter approximation).
# 5. **Channel filter**: Estimate the wireless channel for each DMRS symbol.
#    Here we batch a simple FFT-based threshold filter over DMRS symbols
#    and Rx antennas. It is straightforward to implement and inject other
#    channel filters (including AI-based approaches), see the
#    [Channel Filter Design Tutorial](pusch_channel_estimation_lowering_tutorial.ipynb).
# 6. **Interference+noise covariance estimation**: Compute the
#    interference+noise covariance matrix.
# 7. **Compute MMSE-IRC weights**: Compute MIMO MMSE-IRC equalizer weights
#    using Cholesky factorization (including covariance matrix shrinkage).
# 8. **MIMO equalization**: Apply the MMSE-IRC equalizer weights to the
#    received signal.
# 9. **Soft demapping**: Generate LLRs for the LDPC decoder.
# 10. **Post-equalization metrics**: Compute post-equalization metrics
#     (noise variance and SINR).
#
# To verify that the compiled TensorRT engine works correctly, we need to
# use a suitable PUSCH outer receiver (e.g., LDPC decoder).
# In this notebook, we use the NumPy reference PUSCH outer receiver. In
# the Aerial Framework runtime, we, of course, combine the TensorRT engine
# with the optimized cuPHY CUDA outer receiver.
#
# In this tutorial, we will show you how to profile the compiled TensorRT engine
# using [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems).
# That is, you will see how to get immediate feedback on the
# compute needs of your algorithms. The example
# inner receiver implementation currently takes ~200 microseconds.
#
# We will need to use operations that TensorRT does not natively
# support (e.g., IFFT/FFT and Cholesky factorization). TensorRT does not
# support complex data types natively, so the entire inner receiver has
# been implemented in float16/float32.
#
# Finally, we have implemented DMRS as a custom plugin to illustrate how
# you can write stable algorithms as optimized CUDA kernels and call
# them from Python code or TensorRT engines.
#
# **Time:** ~30 minutes
#

# %% [markdown] tags=["keep-output"]
# ## Step 2: Prerequisites and Environment Setup
#
# This tutorial requires:
# - Running inside the Aerial Framework Docker container
# - Completion of the [Getting Started](getting_started.ipynb) tutorial
# - Completion of the [MLIR-TensorRT Tutorial](mlir_trt_tutorial.ipynb) tutorial
#
# %% [markdown] tags=["keep-output"]
# ## Step 3: Configure CMake and Build Required Targets
#
# **RAN Python Environment Setup:** This tutorial requires the
# RAN Python environment with MLIR-TensorRT support. First time
# setup may take several minutes to download and install the
# required Python packages. The setup involves two key steps:
#
# 1. **CMake Configuration:** Configure the build system with MLIR-TensorRT enabled
# 2. **Build Targets:** Build *py_ran_setup* (Python dependencies) and
#    *pusch_inner_receiver_bench* (benchmark).
#

# %% tags=["keep-output"]
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tutorial_utils import (
    build_cmake_target,
    check_nsys_profile,
    configure_cmake,
    get_project_root,
    is_running_in_docker,
    load_ran_env_file,
    parse_benchmark_output,
    require_mlir_trt,
)

# Ensure running inside Docker container
if not is_running_in_docker():
    msg = (
        "This notebook must be run inside the Docker container. "
        "Please refer to the Getting Started tutorial for instructions."
    )
    raise RuntimeError(msg)

PROJECT_ROOT = get_project_root()
RAN_PY_PATH = PROJECT_ROOT / "ran" / "py"
PRESET = "gcc-release"
RAN_BUILD_DIR = PROJECT_ROOT / "out" / "build" / PRESET

# Configure CMake if needed
configure_cmake(RAN_BUILD_DIR, PRESET)

# Build required targets (first time may take a few minutes)
try:
    build_cmake_target(
        RAN_BUILD_DIR, ["py_ran_setup", "pusch_inner_receiver_bench", "sync_env_python"]
    )
except RuntimeError as e:
    print(f"\n✗ Build failed: {e}\n")
    print("To view full build output and fix issues, enter the container:")
    print("  docker exec -it aerial-framework-base-$USER bash -l")
    print(
        f"  cmake --build out/build/{PRESET} "
        f"--target py_ran_setup sync_env_python pusch_inner_receiver_bench"
    )
    sys.exit(1)

# Load environment variables from .env.python (includes MLIR_TRT_COMPILER_PATH)
load_ran_env_file()

# Check if MLIR-TensorRT is enabled
require_mlir_trt()

print(f"\nBuild directory: {RAN_BUILD_DIR}")
print("✅ Step 3 complete: CMake configured and targets built")

# %% [markdown] tags=["keep-output"]
# ## Step 4: Import Dependencies
#

# %% tags=["keep-output"]
# TensorRT enables lazy loading of CUDA modules (improves loading time)
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# Available JAX backends are: cpu, cuda, and mlir_tensorrt
# We use cuda for this tutorial (and use a separate call to ahead-of-time compile
# PUSCH inner receiver to TensorRT engines).
os.environ["JAX_PLATFORMS"] = "cuda"

# Limit JAX GPU memory pre-allocation to prevent OOM issues in CI tests.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

# Third-party imports
import jax
import numpy as np

# FAPI-based cuPHY reference test vectors
import test_vectors as tvs
from jax import export

# Aerial Framework imports
from ran import mlir_trt_wrapper as mtw
from ran.phy.jax.pusch.pusch_inner_receiver import (
    PuschInnerRxDynamicInputs,
    PuschInnerRxOutputs,
    PuschInnerRxStaticInputs,
    pusch_inner_rx,
)
from ran.phy.jax.pusch.weighted_threshold_filter import WeightedThresholdFilterConfig

# Reference NumPy PUSCH outer receiver (for verification)
from ran.phy.numpy.pusch.outer_receiver import OuterRxParams, pusch_outer_rx

# Custom TensorRT plugins
from ran.trt_plugins.cholesky_factor_inv import cholesky_inv_4x4
from ran.trt_plugins.dmrs import dmrs_3276
from ran.trt_plugins.fft import fft_2048, ifft_2048

# TensorRT plugin manager (for copying test data and TensorRT engines for C++ benchmarks)
from ran.trt_plugins.manager.trt_plugin_manager import (
    copy_test_data_for_cpp_tests,
    copy_trt_engine_for_cpp_tests,
)

print("✅ All imports successful!")

# Set up TensorRT engine directory
build_dir = RAN_BUILD_DIR / "ran" / "py" / "trt_engines"
build_dir.mkdir(parents=True, exist_ok=True)
os.environ["RAN_TRT_ENGINE_PATH"] = str(build_dir)
print(f"Tutorial build directory: {build_dir}")

print("✅ Step 4 complete: Python dependencies verified and imported")


# %% [markdown] tags=["keep-output"]
# ## Step 5: Configure PUSCH Parameters
#
# Get PUSCH parameters from the reference cuPHY test vector. This example uses:
# - 273 PRBs (100 MHz @ 30 kHz subcarrier spacing)
# - 4 receive antennas
# - Single layer, single UE
# - One DMRS symbol (symbol 2)
# - 256-QAM modulation
#

# %% tags=["keep-output"]
# Test vector
TV_NAME = "TVnr_7204_cuPhyMax.h5"  # TVnr_7201_cuPhyMax.h5, TVnr_7204_cuPhyMax.h5

# Load test vector
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
start_prb = np.int32(ls_kwargs["start_prb"])  # Start PRB index
nl = np.int32(ls_kwargs["nl"])  # Number of layers
port_idx = tuple(port for port in ls_kwargs["port_idx"])  # DMRS port numbers
vec_scid = tuple(scid for scid in ls_kwargs["vec_scid"])  # Scrambling id
dmrs_idx = tuple(dmrs_idx for dmrs_idx in ls_kwargs["sym_idx_dmrs"])
energy = 2.0  # DMRS energy scaling factor
apply_cov_shrinkage = True  # Apply shrinkage to covariance matrix estimate

# Soft demapper
qam_bits = int(tv["qam"].item())
n_ue = np.int32(1)
layer2ue = tuple([0] * int(nl))

# Data symbol indices: all symbols except DMRS symbols
all_symbols = set(range(n_t))
dmrs_symbols = set(dmrs_idx)
data_sym_idxs = tuple(sorted(all_symbols - dmrs_symbols))

# PUSCH outer receiver parameters (needed to run the Python
# LDPC backend for verification)
outer_rx_params = OuterRxParams(
    n_id=int(tv["N_id"].item()),
    n_rnti=int(tv["n_rnti"].item()),
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
    i_ls=int(tv["i_LS"].item()),
    max_num_itr_cbs=int(tv["maxNumItr_CBs"].flatten()[0].item()),
    crc_name=np.asarray(tv["CRC"]).tobytes().decode("ascii"),
)

# %% [markdown]
# The noisy channel test vector reference has column-major
# layout and dtype complex128. The Aerial Framework fronthaul kernel produces
# float16 IQ in column major layout with real and imaginary values interleaved.
# The TensorRT engine works only with row-major layout and float.
# We, therefore, need to be a little careful with how the test noisy channel tensor's
# data is arranged for the TensorRT engine.
#
# In the next cell, we downcast from complex128 to float16 and split the real and
# imaginary components. We then arrange the buffer to have the same data layout
# as the Aerial Framework C++ pipeline, but we use the contiguous row-major
# data type needed by TensorRT.

# %% tags=["keep-output"]
# Downcast from complex128 to complex64 and preserve column-major layout.
xtf__sc_sym_rxant = np.array(ls_kwargs["xtf"], dtype=np.complex64, order="F")

# Split complex into real/imag on first axis and cast to float16
xtf__ri_sc_sym_rxant = np.ascontiguousarray(
    np.stack([xtf__sc_sym_rxant.real, xtf__sc_sym_rxant.imag], axis=0), dtype=np.float16
)

# Convert to row-major for TensorRT engine (maintain data layout).
xtf__rxant_sym_sc_ri = np.ascontiguousarray(
    np.einsum("abcd->dcba", xtf__ri_sc_sym_rxant, order="C")
)

print("✅ Step 5 complete: PUSCH parameters configured")


# %% [markdown] tags=["keep-output"]
# ## Step 6: Compile the Example PUSCH Inner Receiver
#
# We are now almost ready to compile the PUSCH inner receiver.
# The remaining compilation steps are:
# 1. Create dynamic inputs (inputs that change at runtime).
# 2. Create static inputs (inputs that are constant for the
#    entire execution and can be baked into the TensorRT engine).
# 3. Export JAX PUSCH inner receiver function to StableHLO
# 4. Compile StableHLO using MLIR-TensorRT and TensorRT compilers.

# Note: For simplicity, we have chosen to make all parameters static
# (except for the noisy channel). This is a simplified example for
# demonstration purposes.

# %% tags=["keep-output"]

# Dynamic inputs
dynamic_inputs = PuschInnerRxDynamicInputs(
    xtf__rxant_sym_sc_ri=xtf__rxant_sym_sc_ri,
)
dynamic_inputs_pos = dynamic_inputs.to_tuple()

# Channel filter to compile in this tutorial
filter_name = "weighted_threshold_filter"

# Create method-specific build directory for compilation artifacts
method_build_dir = build_dir / "pusch_inner_receiver" / filter_name
if method_build_dir.exists():
    print(f"  Cleaning existing directory: {method_build_dir}")
    shutil.rmtree(method_build_dir)
method_build_dir.mkdir(parents=True, exist_ok=True)

# Ensure parent build directory exists (for other engines from fixture tests)
build_dir.mkdir(parents=True, exist_ok=True)

# Save input data for Aerial Framework C++ benchmarks
xtf_file = method_build_dir / "xtf_input.bin"
xtf_meta_file = method_build_dir / "xtf_input_meta.txt"
xtf__rxant_sym_sc_ri.tofile(xtf_file)
with open(xtf_meta_file, "w") as f:
    f.write("# xtf input data metadata\n")
    f.write(f"shape: {xtf__rxant_sym_sc_ri.shape}\n")
    f.write(f"dtype: {xtf__rxant_sym_sc_ri.dtype}\n")
    f.write("format: Row-major (n_rxant, n_sym, n_sc, real/imag)\n")
    f.write(f"size_bytes: {xtf__rxant_sym_sc_ri.nbytes}\n")

# Channel filter configuration (simple FFT-based weighted threshold filter)
channel_filter_config = WeightedThresholdFilterConfig(
    fft_size=2048,
    delay_compensation_samples=50.0,
    decay_rate=0.01,
    k_sigma=3.0,
)

# Create static inputs
static_inputs = PuschInnerRxStaticInputs(
    slot_number=int(slot_number),
    n_dmrs_id=int(n_dmrs_id),
    rww_regularizer_val=1e-8,
    start_prb=int(start_prb),
    nl_offset=0,
    scids=vec_scid,
    apply_cov_shrinkage=apply_cov_shrinkage,
    channel_filter_method="weighted_threshold_filter",
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

# Export the PUSCH inner receiver to StableHLO MLIR.
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

# Compile the PUSCH inner receiver (as StableHLO) to a TensorRT engine.

mlir_tensorrt_compilation_flags = [
    "tensorrt-builder-opt-level=0",
    "tensorrt-workspace-memory-pool-limit=50MiB",
    f"artifacts-dir={method_build_dir}",
]

trt_plugin_configs = {
    "tensorrt_dmrs_plugin": dmrs_3276.get_config(),
    "tensorrt_fft_plugin_forward": fft_2048.get_config(),
    "tensorrt_fft_plugin_inverse": ifft_2048.get_config(),
    "tensorrt_cholesky_inv_plugin": cholesky_inv_4x4.get_config(),
}

exe = mtw.compile(
    stablehlo_mlir=stablehlo_mlir,
    name="pusch_inner_rx",
    export_path=method_build_dir,
    mlir_entry_point="main",
    mlir_tensorrt_compilation_flags=mlir_tensorrt_compilation_flags,
    trt_plugin_configs=trt_plugin_configs,
)

# Copy TensorRT engine and test data for C++ tests
engine_dest = copy_trt_engine_for_cpp_tests(
    method_build_dir, f"pusch_inner_receiver_{filter_name}.trtengine"
)
print(f"  Copied TensorRT engine to {engine_dest}")

print("✅ Step 6 complete: PUSCH inner receiver compiled to TensorRT")


# %% [markdown] tags=["keep-output"]
# ## Step 7: Verify the Compiled PUSCH Inner Receiver
#
# The next cell demonstrates how to execute the compiled PUSCH inner receiver
# using the MLIR-TensorRT backend. We will also execute the JAX function using
# JAX's CUDA backend to compare against the TensorRT results.


# %% tags=["keep-output"]

# Reference outputs: Execute the PUSCH inner receiver using the
# JAX CUDA backend.
outputs_ref_tuple = pusch_inner_rx(*all_inputs)
outputs_ref = PuschInnerRxOutputs.from_tuple(outputs_ref_tuple)

# TensorRT outputs: Execute the compiled PUSCH inner receiver
# using the MLIR-TensorRT backend.
outputs_tuple = tuple(np.zeros_like(o) for o in outputs_ref.to_tuple())
mtw.execute(
    exe=exe,
    inputs=dynamic_inputs_pos,
    outputs=outputs_tuple,
    sync_stream=True,
    mlir_entry_point="main",
)
outputs_trt = PuschInnerRxOutputs.from_tuple(outputs_tuple)

# Save reference outputs for C++ benchmark
llr_file = method_build_dir / "llr_output.bin"
outputs_trt.llr__time_allocfreq_layer_qambit.tofile(llr_file)

# Copy test data to the C++ benchmark directory.
test_data_dest = copy_test_data_for_cpp_tests(
    method_build_dir, f"pusch_inner_receiver/{filter_name}", ["*.bin", "*_meta.txt"]
)
print(f"  Copied test data to {test_data_dest}")

print("✅ Step 7 complete: PUSCH inner receiver executed with JAX and TensorRT backends")


# %% [markdown] tags=["keep-output"]
# ## Step 8: Verify with the Full PUSCH Receiver Pipeline
#
# The next cell verifies the outputs of the compiled TensorRT engine with the
# full PUSCH receiver pipeline (inner and outer receivers).

# %% tags=["keep-output"]

# Reference LDPC outputs: Apply PUSCH outer receiver processing to reference output
outer_rx_outputs_ref = pusch_outer_rx(
    llr__time_allocfreq_layer_qambit=outputs_ref.llr__time_allocfreq_layer_qambit.astype(
        np.float32
    ),
    outer_rx_params=outer_rx_params,
    qam_bits=qam_bits,
)

# TensorRT LDPC outputs: Apply PUSCH outer receiver processing to TensorRT output
outer_rx_outputs_trt = pusch_outer_rx(
    llr__time_allocfreq_layer_qambit=outputs_trt.llr__time_allocfreq_layer_qambit.astype(
        np.float32
    ),
    outer_rx_params=outer_rx_params,
    qam_bits=qam_bits,
)

# Check transport block and code block errors
outer_rx_trt_passed = outer_rx_outputs_trt.tb_err == 0
outer_rx_ref_passed = outer_rx_outputs_ref.tb_err == 0

# Compare with test vector
tv_tb_err = int(tv["tbErr"].item())
if outer_rx_outputs_trt.tb_err == tv_tb_err:
    print("  ✓ TRT TB error matches test vector")
else:
    print(f"  ✗ TRT TB error mismatch (TRT={outer_rx_outputs_trt.tb_err}, TV={tv_tb_err})")

# Final verification status
if outer_rx_trt_passed and outer_rx_ref_passed:
    print("  ✓ Outer receiver verification PASSED!")
else:
    print("  ✗ Outer receiver verification FAILED")

print("✅ Step 8 complete: Output verification completed")


# %% [markdown] tags=["keep-output"]
# ## Step 9: Run C++ Benchmarks via CTest
#
# The compiled TensorRT engines can be benchmarked using CTest.
# The benchmark measures end-to-end latency including:
# - Host-to-device memory transfer
# - TensorRT engine execution
# - Device-to-host memory transfer

# %% tags=["keep-output"]
# Get the TensorRT engine path where files were copied
ran_trt_engine_path = Path(os.environ["RAN_TRT_ENGINE_PATH"])

print(f"Build directory: {RAN_BUILD_DIR}")
print(f"TensorRT engines directory: {ran_trt_engine_path}")
print(
    f"Engine will be loaded from: {ran_trt_engine_path}/"
    "pusch_inner_receiver_weighted_threshold.trtengine"
)
print(f"Test data will be loaded from: {ran_trt_engine_path}/test_vectors/weighted_threshold/\n")


print(f"\n{'=' * 80}")
print("Benchmarking")
print(f"{'=' * 80}")

# Map filter method names to ctest filter names (remove '_filter' suffix for ctest)
ctest_filter_name = filter_name.replace("_filter", "")

# Run ctest for this specific filter
ctest_cmd = [
    "ctest",
    "--preset",
    PRESET,
    "-R",
    f"ran.phy_bench.pusch_inner_receiver_bench.{ctest_filter_name}",
    "-V",
]

print(f"Running: {' '.join(ctest_cmd)}\n")

result = subprocess.run(ctest_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

if result.returncode == 0:
    print("✓ Benchmark complete!\n")

    # Parse and display benchmark results table
    benchmark_lines = parse_benchmark_output(result.stdout, "bm_pusch_inner_receiver")

    if benchmark_lines:
        print("\n".join(benchmark_lines))
    else:
        # Fallback: show last part of output if parsing failed
        print("Benchmark output:")
        print("\n".join(result.stdout.split("\n")[-20:]))
else:
    print("✗ Benchmark failed")
    print(result.stdout[-500:])
    print(result.stderr[-500:])
    sys.exit(1)

print("✅ Step 9 complete: C++ benchmarks executed")

# %% [markdown] tags=["keep-output"]
# ## Step 10: NVIDIA Nsight Systems Profiling
#
# Profile the TensorRT engines with NVIDIA Nsight Systems to analyze:
# - GPU kernel execution timeline
# - Memory transfer operations
# - TensorRT layer fusion effectiveness
# - Potential bottlenecks
#
# Both stream and graph execution modes are profiled for comparison.

# %% tags=["keep-output"]
# Run nsys profiling via ctest
# Note: ctest creates one profile file that gets overwritten by each mode
nsys_cmd = [
    "ctest",
    "--preset",
    PRESET,
    "-R",
    f"ran.phy_nsys.pusch_inner_receiver_bench.{ctest_filter_name}",
    "-V",
]

print(f"Running: {' '.join(nsys_cmd)}")
print("This will profile both stream and graph modes...")
print("This may take a few minutes...")

result = subprocess.run(nsys_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

if result.returncode == 0:
    print("✓ Profiling complete!")
    check_nsys_profile(RAN_BUILD_DIR, f"pusch_inner_receiver_bench_{ctest_filter_name}")
else:
    print("✗ Profiling failed")
    print(result.stdout[-500:])
    sys.exit(1)

print("✅ Step 10 complete: NVIDIA Nsight Systems profiling completed")

# %% [markdown] tags=["keep-output"]
# ## Step 11: Summary and Next Steps
#
# **What we accomplished:**
#
# 1. ✅ Compiled PUSCH inner receiver to TensorRT
# 2. ✅ Verified correctness of compiled engines against JAX CUDA backend
# 3. ✅ Benchmarked latency with CTest
# 4. ✅ Profiled with NVIDIA Nsight Systems
#
#
# **Next steps:**
#
# - Review Nsight Systems profiles to identify optimization opportunities
# - Explore the [AI Tukey Filter Training](ai_tukey_filter_training_tutorial.ipynb)
# tutorial to train a custom AI Tukey filter for channel estimation.


# %% tags=["keep-output"]
print("✅ Tutorial complete!")
