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

# %% [markdown]
# # 9. Top Level PHY RAN Application
#
# This notebook provides a PHY RAN App integration test where we emulate the surrounding components
# (MAC Layer and Radio Unit) to isolate and test the PHY layer performance.
#
# **Key Features:**
# 1. **Background**: Overview of MAC & RU Emulators and PHY RAN App.
# 2. **System Overview**: Architectural and communication highlights.
# 3. **Setup and Build**: Build of components for the integration test.
# 4. **Test and Results**: Running the test and interpreting output/results.
#
# **Prerequisites:**
# - GH200 server with BF3 NIC configured in loopback mode
# - CPU core isolation for real-time performance
# - Execution inside the Docker container
#
# **Time:** ~10 minutes

# %% [markdown]
# ## Step 1: Understand System Architecture
#
# To test the PUSCH receiver pipeline from previous tutorials, we need to emulate the surrounding
# components:
# - **testMAC:** Emulates the MAC, sending scheduling commands (FAPI) via Shared
#   Memory (nvIPC).
# - **phy_ran_app:** The *System Under Test*. It runs the full PUSCH (Uplink) receiver pipeline.
# - **ru_emulator:** Emulates the Radio Unit, exchanging IQ samples via Fronthaul (DPDK/DOCA).
#
# A simplified 5G network deployment may look like:
#
# ![Real System Architecture](../../figures/generated/real_world_deployment_arch.drawio.svg)
#
# We want to simulate these components using the same hardware as that used in real deployments.
# In this tutorial, we use a GH200 server and BF3 NIC configured in loopback mode. The software
# processes are executed inside a container as shown below:
#
# ![PHY App Container Architecture](../../figures/generated/phy_app_container.drawio.svg)

# %% [markdown]
# ## Step 2: Explore Communication Flow (FAPI and Fronthaul)
#
# The interaction between these components is driven by strict timing and protocol standards. Below
# is a simplified diagram of the FAPI and Fronthaul data flow:
#
# ![Communication Flow](../../figures/generated/comms_phy_ran_app.drawio.svg)
#
# ### Communication Interfaces
#
# 1.  **Northbound (MAC-PHY):**
#     *   **Protocol:** FAPI (Functional Application Platform Interface)
#     *   **Transport:** nvIPC (Shared Memory)
#     *   **Messages:** `UL_TTI_REQUEST` (Schedule this slot), `SLOT.indication` (Time tick)
#
# 2.  **Southbound (RU):**
#     *   **Protocol:** O-RAN Fronthaul (7.2 Split)
#     *   **Transport:** Ethernet (DPDK for Control, DOCA GPUNetIO for Data)
#     *   **Messages:** C-Plane (Control info), U-Plane (IQ Data)
#
# ### Data Flow
#
# The `phy_ran_app` doesn't just forward messages. It translates them:
# *   Receives a **FAPI Request** from the MAC ("Decode user X at freq Y").
# *   Translates this into **C-Plane Messages** for the Radio ("Prepare to receive on PRB Z").
# *   Waits for **U-Plane Data** from the Radio (IQ samples).
# *   Runs the **Signal Processing Pipeline** (PUSCH).
# *   Sends the transport block and CRC back to the MAC
#
# And the testMAC subsequently compares the received bits with the expected bits to verify the
# correctness of the PUSCH receiver pipeline. It additionally checks for CRC pass/fail and other
# measurements like RSSI and SINR.

# %% [markdown]
# ## Step 3: Review PUSCH Pipeline
#
# The phy_ran_app runs the PUSCH receiver, which contains the Python-lowered inner receiver and
# CUDA-based outer receiver from earlier tutorials. Below is a simplified diagram of the PUSCH
# receiver pipeline:
#
# ![Simple PUSCH Receiver Pipeline](../../figures/generated/simple_pusch_rx.drawio.svg)
#
# %% [markdown]
# ## Step 4: Configure
#
# We will now configure the build system for `phy_ran_app`. This process uses CMake with
# a release preset to ensure optimal performance for real-time execution.

# %% tags=["keep-output"]
import os

# Import shared tutorial utilities from tutorial_utils.py
from tutorial_utils import (
    build_cmake_target,
    check_container_running,
    check_network_devices,
    configure_cmake,
    get_project_root,
    is_running_in_docker,
    run_container_command,
    show_output,
)

IN_DOCKER = is_running_in_docker()
PROJECT_ROOT = get_project_root()
CONTAINER_NAME = f"aerial-framework-base-{os.environ.get('USER', 'default')}"

# Ensure the container is running
print(f"Project root: {PROJECT_ROOT}")
if IN_DOCKER:
    print("✅ Running inside Docker container")
else:
    print(f"Running on host, will use container: {CONTAINER_NAME}")
    check_container_running(CONTAINER_NAME)
    print(f"✅ Container '{CONTAINER_NAME}' is running")
print("✅ Step 4a complete: Environment setup verified")

# %% [markdown]
# **Configure CMake:**

# %% tags=["keep-output"]
# Configure CMake with preset
preset = "gcc-release"
print(f"Configuring {preset}...")

configure_cmake(PROJECT_ROOT / f"out/build/{preset}", preset=preset)
print("✅ Step 4b complete: CMake configured")
# %% [markdown]
# ## Step 5: Build the Target
#
# Now we compile the application and its dependencies.
#
# We build `phy_ran_app` and its dependencies, which includes `testMAC` and `ru_emulator`.

# %% tags=["keep-output"]
print(f"Building phy_ran_app with {preset} preset...")

build_cmake_target(PROJECT_ROOT / f"out/build/{preset}", "phy_ran_app")
print("✅ Step 5 complete: PHY RAN app built")
# %% [markdown]
# ## Step 6: Run Integration Test
#
# The integration test launches the full trio: `ru_emulator` (background), `testMAC`
# (background), and `phy_ran_app` (foreground).
#
# It validates that:
# 1.  Connections are established.
# 2.  Packets flow from RU to DU.
# 3.  The pipeline processes data without errors.
# 4.  Timing constraints are met.


# %% tags=["keep-output"]
def _print_integration_summary(log: str) -> None:
    checks = {
        "CRC PASS found": "crc=PASS" in log,
        "ru_emulator exited cleanly": "INFO: ru_emulator exited cleanly" in log,
        "test_mac exited cleanly": "INFO: test_mac exited cleanly" in log,
        "phy_ran_app exit code 0": "INFO: phy_ran_app exited with code: 0" in log,
        "Integration test completed": "INFO: Integration test completed successfully" in log,
        "ctest 100% tests passed": "100% tests passed" in log,
    }

    overall_ok = all(checks.values())
    line = "=" * 60

    print(line)
    print("PHY RAN APP INTEGRATION SUMMARY")
    print(line)
    for label, ok in checks.items():
        print(f"{'✅' if ok else '❌'} {label}")
    print(line)
    print("✅ OVERALL: PASS" if overall_ok else "❌ OVERALL: FAIL")
    print(line)


# Check networking availability
if check_network_devices(CONTAINER_NAME):
    print("Running phy_ran_app integration test...")

    # Run the integration test (default: 100 slots)
    cmd = f"ctest --preset {preset} -R phy_ran_app.integration_test"
    result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

    show_output(result, lines=20)
    _print_integration_summary(result.stdout)
    success = result.returncode == 0
    print("✅ Integration test passed" if success else "⚠️  Integration test failed")
    print("✅ Step 6 complete: Integration test executed")

# %% [markdown]
# ### Understanding the Output
#
# *   **On-time Packets:** Indicates that the `phy_ran_app` is processing data fast enough
#     to keep up with the 5G slot schedule.
# *   **Late Packets:** If these appear, it often indicates system tuning issues
#     (e.g., CPU isolation not configured, debug build used instead of release).
# *   **CRC Pass/Fail:** The test vector contains known data. A "Pass" means the PUSCH
#     pipeline successfully recovered the original bits.
#
# For more information on the integration test, see the
# [Real-Time Applications](../../developer_guide/real_time_apps.rst) documentation.

# %% [markdown]
# ## Next Steps
#
# Now that you have verified the core PHY application:
#
# *   **Performance Profiling:** Use Nsight Systems to visualize the GPU kernel execution
#     timeline.
# *   **Custom Test Vectors:** Try running with different `TEST_VECTOR` files to test
#     different MCS (Modulation and Coding Schemes).
# *   **Test Slots:** Configure `TEST_SLOTS=<num>` to control how many slots are tested.
#     See [PHY RAN App API Reference](../../api/ran/runtime/phy_ran_app.rst) for more details.
# *   **CTest Timeout for Long Tests:** For long-running tests with large `TEST_SLOTS` values,
#     CTest's default timeout (1500s) may be insufficient. Use `--timeout` to increase it:
#     `TEST_SLOTS=200000 ctest --preset gcc-release --timeout 3000 -R phy_ran_app.integration_test`
