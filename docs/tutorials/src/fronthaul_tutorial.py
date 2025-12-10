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
# # 8. Fronthaul Uplink Processing
#
# This tutorial demonstrates the Aerial Framework fronthaul application for O-RAN uplink processing.
# The fronthaul application implements the DU (Distributed Unit) side of the O-RAN fronthaul
# interface, processing both C-Plane (Control Plane) and U-Plane (User Plane) traffic with GPU
# acceleration.
#
# **Key Features:**
# - C-Plane transmission via DPDK with accurate scheduling
# - U-Plane reception via DOCA GPUNetIO with GPU kernel processing
# - Real-time task scheduling with timed triggers
#
# **Prerequisites:**
# - GH200 server with BF3 NIC configured in loopback mode
# - PTP/PHC2SYS time synchronization
# - CPU core isolation for real-time performance
# - Completed Getting Started tutorial
#
# **Time:** ~15 minutes

# %% [markdown]
# ## 1. Real-Time System Setup
#
# The fronthaul application requires a properly configured real-time system for deterministic
# performance. This section covers the essential setup steps for a GH200 server with BF3 NIC.

# %% [markdown]
# ### Hardware Configuration
#
# **Required Hardware:**
# - NVIDIA GH200 Grace Hopper server
# - NVIDIA BlueField-3 (BF3) NIC with two ports
# - Direct Attach Copper (DAC) cable (at least 100 GbE recommended) connecting the two BF3 ports in
#   a loopback configuration
#
# **Loopback Configuration:**
# ```
# BF3 NIC Port 0 (DU side) <--[DAC Cable]--> BF3 NIC Port 1 (RU side)
# ```
#
# This loopback configuration allows testing the complete fronthaul stack without external
# radio hardware. The DU side runs `fronthaul_app` while the RU side runs `ru_emulator`.

# %% [markdown]
# ### Real-Time System Configuration
#
# For fronthaul testing, the system must be configured with real-time capabilities.
# Follow the complete setup guide in the **NVIDIA Aerial CUDA-Accelerated RAN Installation Guide**:
#
# [Installing Tools and Drivers on Grace Hopper Systems](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/install_guide/installing_tools_gh.html)
#
# **Key sections for fronthaul:**
#
# 1. **Configure the Network Interfaces**
#    - Setup BF3 NIC interfaces (aerial00, aerial01) with appropriate MTU size
#    - Configure static IP addresses and bring interfaces up for fronthaul traffic
#
# 2. **Time Synchronization (PTP/PHC2SYS)**
#    - Configure PTP daemon for NIC hardware clock synchronization
#    - Setup PHC2SYS to synchronize system clock to NIC
#
# 3. **CPU Core Isolation**
#    - Configure kernel boot parameters: `isolcpus`, `nohz_full`, `rcu_nocbs`
#    - Prevents OS interference with fronthaul processing
#
# 4. **Hugepages Configuration**
#    - Required for DPDK memory allocation
#
# 5. **IOMMU and VFIO Setup**
#    - Enables direct NIC access from user space
#    - Required for DPDK and DOCA GPUNetIO

# %% [markdown]
# ## 2. O-RAN Fronthaul Overview
#
# The O-RAN fronthaul interface connects the Radio Unit (RU) and Distributed Unit (DU),
# separating radio functions from baseband processing.

# %% [markdown]
# ### Fronthaul Control and Data Interfaces
#
# The O-RAN fronthaul specification separates control and data interfaces:
#
# | Aspect | C-Plane (Control) | U-Plane (User Data) |
# |--------|-------------------|---------------------|
# | **Purpose** | Scheduling and configuration | IQ sample data transfer |
# | **Direction** | DU → RU | RU → DU (uplink) |
# | **Content** | Slot timing, PRB allocation, beam config | Compressed IQ samples, PRB mapping |
# | **Timing Window** | T1a (250-500 μs before slot) | Ta4 (200-400 μs after slot) |
# | **Processing** | CPU (DPDK) | GPU kernel (DOCA GPUNetIO) |
# | **Data Rate** | Low (control messages) | High (IQ data streams) |
# | **Implementation** | Packet transmission | GPU-direct packet reception |
#
# **Message Flow:**
# ```
# C-Plane: DU (fronthaul_app) --[scheduling]--> RU (ru_emulator)
# U-Plane: RU (ru_emulator) --[IQ samples]--> DU (fronthaul_app)
# ```
#
# **Key Design Decisions:**
# - **C-Plane uses DPDK:** CPU handles low-rate control with precise timing
# - **U-Plane uses DOCA GPUNetIO:** GPU receives high-rate data directly from NIC
# - **Separation:** Allows independent optimization of control and data paths

# %% [markdown]
# ### Data Flow Diagram
#
# The following diagram illustrates the complete O-RAN fronthaul data flow:
#
# ![O-RAN Fronthaul Data Flow](../../figures/generated/fronthaul_dataflow.drawio.svg)

# %% [markdown]
# ## Network Environment and Libraries
#
# The fronthaul application uses DPDK for C-plane processing and DOCA GPUNetIO for U-plane
# processing.

# %% [markdown]
# ### Network Library Selection
#
# The fronthaul application uses different network libraries optimized for each interface:
#
# | Library | Plane | Rationale |
# |---------|-------|-----------|
# | **DPDK** | C-Plane TX | Kernel bypass, precise timing, efficient for low-rate control |
# | **DOCA GPUNetIO** | U-Plane RX | GPU-direct DMA, eliminates CPU bottleneck for high-rate data |
#
# **Configuration Pattern:**
#
# Both libraries share a common configuration approach with plane-specific parameters:
#
# ```cpp
# // Common network configuration structure (from fronthaul_app_utils.cpp)
# struct NetworkConfig {
#     std::string nic_pcie_addr;      // NIC PCIe address
#     std::uint32_t gpu_device_id;    // GPU for processing
#     std::uint32_t core_id;          // CPU core (DPDK) or unused (DOCA)
#     std::uint32_t mtu_size;         // Maximum transmission unit
#
#     // Plane-specific queue configuration
#     std::uint16_t queue_size;       // TX queue (DPDK) or RX queue (DOCA)
#     bool gpu_direct;                // DOCA only: enable GPU-direct DMA
#     std::uint32_t num_buffers;      // Ping-pong buffering (DOCA)
# };
# ```
#
# **Key Components by Plane:**
#
# - **C-Plane (DPDK):**
#   - TX Queue: Holds packets ready for transmission
#   - Memory Pool: Pre-allocated buffers for zero-copy operation
#
# - **U-Plane (DOCA GPUNetIO):**
#   - RX Queue: GPU-accessible incoming packet queue
#   - GPU Semaphore: Signals packet arrival to GPU kernel
#   - Ping-Pong Buffers: Dual buffers for overlapped receive/process
#
# **Resources:**
# - [DOCA GPUNetIO Programming Guide](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html)
# - Additional DOCA GPUNetIO examples can be found in `framework/net/samples`

# %% [markdown]
# ## 4. Functional API (FAPI) Capture and C-Plane Packet Preparation
#
# The fronthaul application replays FAPI (Functional API - Small Cell Forum API) messages captured
# from testMAC, converting them to O-RAN C-Plane packets with accurate send scheduling.

# %% [markdown]
# ### FAPI Capture for TestMAC Separation
#
# **Purpose:** Decouple fronthaul testing from full PHY stack
#
# **Workflow:**
# 1. Run `testMAC` with FAPI capture enabled
# 2. TestMAC generates FAPI messages for uplink scheduling
# 3. Messages saved to `.fapi` file
# 4. `fronthaul_app` replays messages in real-time
#
# %% [markdown]
# **FAPI File Replay:**
#
# ```cpp
# // Create FAPI replay from capture file (from fronthaul_app.cpp)
# rf::FapiFileReplay fapi_replay(
#     fapi_file_path,
#     fh_config.numerology.slots_per_subframe
# );
#
# RT_LOGC_INFO(
#     rf::FronthaulApp::App,
#     "Loaded {} requests from {} cells",
#     fapi_replay.get_total_request_count(),
#     fapi_replay.get_cell_count()
# );
# ```

# %% [markdown]
# **FAPI Message Structure:**
# - **UL_TTI.request:** Uplink scheduling per slot
# - **Cell ID:** Which cell to configure
# - **Slot number:** Absolute slot index
# - **PUSCH PDUs:** Physical Uplink Shared Channel configuration

# %% [markdown]
# ### C-Plane Packet Preparation
#
# **Conversion Process:** FAPI → O-RAN C-Plane
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: C-Plane packet creation function</i></summary>
#
# ```cpp
# // C-Plane packet creation function (from fronthaul_app_utils.cpp)
# std::function<void()> make_process_cplane_func(
#         ran::fronthaul::Fronthaul &fronthaul,
#         ran::fapi::FapiFileReplay &fapi_replay,
#         bool &is_first_slot,
#         const std::chrono::nanoseconds t0,
#         const std::chrono::nanoseconds tai_offset) {
#
#     return [&fronthaul, &fapi_replay, &is_first_slot, t0, tai_offset]() {
#         // Advance to next slot (skip on first call to avoid skipping slot 0)
#         if (!is_first_slot) {
#             fapi_replay.advance_slot();
#         } else {
#             is_first_slot = false;
#         }
#
#         // Get current slot
#         const std::uint64_t absolute_slot = fapi_replay.get_current_absolute_slot();
#
#         // Process each cell for current slot
#         for (const auto cell_id : fapi_replay.get_cell_ids()) {
#             // Get request for current slot (returns std::nullopt if no match)
#             const auto request_opt = fapi_replay.get_request_for_current_slot(cell_id);
#
#             if (!request_opt) {
#                 continue;  // No UL data for this cell/slot
#             }
#
#             // Send C-Plane for this cell
#             const auto &req_info = request_opt.value();
#             fronthaul.send_ul_cplane(
#                 *req_info.request, req_info.body_len, cell_id, absolute_slot, t0, tai_offset);
#         }
#     };
# }
# ```
# </details>

# %% [markdown]
# ### Accurate Send Time Scheduling
#
# **Challenge:** C-Plane packets must arrive at RU within T1a window
#
# **Solution:** Calculate exact transmission time based on slot timing
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: Packet send time calculation</i></summary>
#
# ```cpp
# // Packet send time calculation (from fronthaul.cpp)
# PacketSendTimeResult calculate_packet_send_time(const PacketSendTimeParams &params) {
#     PacketSendTimeResult result{};
#
#     // Calculate expected start time for this slot
#     const auto iabsolute_slot = static_cast<std::int64_t>(params.absolute_slot);
#     const auto islot_ahead = static_cast<std::int64_t>(params.slot_ahead);
#     const auto iabsolute_slot_ahead = iabsolute_slot - islot_ahead;
#     result.expected_start = params.t0 + params.slot_period * iabsolute_slot_ahead;
#
#     // Calculate threshold: (slot_period * slot_ahead) - t1a_max_cp_ul
#     result.threshold = params.slot_period * islot_ahead - params.t1a_max_cp_ul;
#
#     // Calculate time delta and check threshold
#     result.time_delta = params.actual_start - result.expected_start;
#     result.exceeds_threshold = (result.time_delta > std::chrono::nanoseconds{0}) &&
#                                (result.time_delta > result.threshold);
#
#     // Calculate transmission time with TAI offset
#     result.start_tx = result.expected_start + result.threshold + params.tai_offset;
#
#     return result;
# }
# ```
# </details>
#
# **Setting Timestamp on Packets:**
#
# After calculating the send time, the timestamp is set on all packet mbufs to enable
# accurate hardware-based transmission scheduling:
#
# ```cpp
# // Set timestamp on packets (from oran/cplane_message.cpp: prepare_cplane_message)
# if (info.tx_window_start > last_packet_ts) {
#     // Set timestamp on all packets (including fragments)
#     for (std::uint16_t pkt_idx = 0; pkt_idx < packet_num; ++pkt_idx) {
#         buffers[pkt_idx].set_timestamp(info.tx_window_start);
#     }
#     last_packet_ts = info.tx_window_start;
# }
# ```
#
# **How It Works:**
# - `tx_window_start` contains the calculated send time in nanoseconds
# - `set_timestamp()` writes the timestamp to the mbuf's `ol_flags` and `timestamp` fields
# - The NIC's hardware timestamping capability uses this timestamp to transmit at the
#   exact time
# - Timestamp is only updated if it's newer than the last packet (avoids going backwards)

# %% [markdown]
# **Timing Parameters:**
# - **t0:** Time of SFN 0, subframe 0, slot 0 (reference point)
# - **tai_offset:** TAI (International Atomic Time) offset from GPS
# - **slot_ahead:** How many slots ahead we're starting processing
# - **t1a_max_cp_ul:** Maximum advance time for C-Plane (e.g., 500 μs)

# %% [markdown]
# ## 5. DOCA GPUNetIO Pipeline and Order Kernel
#
# The U-Plane processing uses GPU kernels to receive and process packets directly
# from the NIC without CPU involvement.

# %% [markdown]
# ### Order Kernel Pipeline Architecture
#
# **Purpose:** Receive U-Plane packets and reorder IQ samples for PHY processing
#
# **Pipeline Stages:**
# 1. **Packet Reception:** GPU kernel polls DOCA RX queue
# 2. **Header Parsing:** Extract O-RAN headers (PRB index, symbol, compression)
# 3. **Decompression:** Decompress BFP IQ samples
# 4. **Reordering:** Place samples in correct PRB/symbol positions
# 5. **Output:** Contiguous IQ buffer ready for channel estimation

# %% [markdown]
# **Ping-Pong Buffering:**
# - Two GPU buffers alternate between receive and process
# - While slot N processes, slot N+1 receives
# - Enables overlapped computation and communication

# %% [markdown]
# **Pipeline Configuration:**
#
# ```cpp
# // Order kernel pipeline configuration (from order_kernel_pipeline.hpp)
# struct OrderKernelPipelineConfig {
#     // Network configuration
#     UPlaneNetworkConfig network_config{};
#
#     // Pipeline parameters
#     std::uint32_t num_prbs{273};           // Number of PRBs
#     std::uint32_t num_symbols{14};         // OFDM symbols per slot
#     std::uint32_t num_antenna_ports{4};    // Antenna ports
#
#     // Buffer configuration (ping-pong)
#     std::uint32_t num_buffers{2};          // Dual buffering
#
#     // Timing windows
#     std::uint64_t ta4_min_ns{200000};      // Ta4 min (200 μs)
#     std::uint64_t ta4_max_ns{400000};      // Ta4 max (400 μs)
#
#     // Kernel launch parameters
#     std::uint32_t threads_per_block{320};  // CUDA threads
#     std::uint32_t blocks_per_grid{1};      // Single CTA for polling
# };
# ```

# %% [markdown]
# ### Order Kernel Implementation
#
# **File:** `ran/runtime/fronthaul/lib/src/oran_order_kernels.cu`

# %% [markdown]
# **Key Kernel Functions:**
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: Order kernel implementation</i></summary>
#
# ```cpp
# // Order kernel entry point (simplified from oran_order_kernels.cu)
# __global__ void order_kernel_pingpong(
#     doca_gpu_eth_rxq *rxq,              // DOCA RX queue
#     doca_gpu_semaphore_gpu *sem_gpu,    // Packet arrival semaphore
#     uint8_t *output_iq_buffer,          // Output IQ samples
#     OrderKernelDescriptor *descriptor,  // Configuration
#     uint32_t *packet_count,             // Statistics
#     uint64_t *earliest_timestamp        // Timing info
# ) {
#     // Single CTA polls for packets and processes
#     __shared__ uint32_t packets_received;
#     __shared__ uint64_t ta4_deadline;
#
#     if (threadIdx.x == 0) {
#         packets_received = 0;
#         ta4_deadline = get_slot_start_time() + descriptor->ta4_max_ns;
#     }
#     __syncthreads();
#
#     // Poll for packets until Ta4 deadline
#     while (get_current_time() < ta4_deadline) {
#         // Wait for packet arrival signal
#         if (doca_gpu_dev_sem_get_status(sem_gpu) > 0) {
#             // Receive packet batch
#             uint32_t num_pkts = 0;
#             doca_gpu_dev_eth_rxq_receive_block(rxq, &num_pkts, ...);
#
#             // Process each packet in parallel (across threads)
#             for (uint32_t pkt_idx = threadIdx.x;
#                  pkt_idx < num_pkts;
#                  pkt_idx += blockDim.x) {
#
#                 // Parse O-RAN headers
#                 auto *oran_hdr = parse_oran_header(packet[pkt_idx]);
#                 uint32_t prb_start = oran_hdr->prb_start;
#                 uint32_t symbol_id = oran_hdr->symbol_id;
#
#                 // Decompress IQ samples (BFP)
#                 decompress_bfp(
#                     oran_hdr->iq_data,
#                     output_iq_buffer + get_output_offset(prb_start, symbol_id)
#                 );
#             }
#
#             packets_received += num_pkts;
#         }
#     }
#
#     // Write statistics
#     if (threadIdx.x == 0) {
#         *packet_count = packets_received;
#     }
# }
# ```
# </details>

# %% [markdown]
# **Kernel Characteristics:**
# - **Single CTA:** One CUDA Thread Block handles all packets for a slot
# - **Polling:** Continuously checks for packet arrival (low latency)
# - **Parallel Processing:** Threads process different packets simultaneously
# - **Deadline-Driven:** Stops at Ta4 timeout if packets missing

# %% [markdown]
# ## 6. Task Definition and Timed Trigger
#
# The fronthaul application uses the Aerial Framework task system for real-time
# slot-based scheduling.

# %% [markdown]
# ### Task Graph Definition
#
# The fronthaul processing uses a sequential two-task pipeline:
#
# | Task | Function | Processor | Purpose |
# |------|----------|-----------|---------|
# | **process_cplane** | Send C-Plane packets | CPU (DPDK) | Transmit scheduling info to RU |
# | **process_uplane** | Receive U-Plane packets | GPU kernel | Receive and process IQ data |
#
# **Task Dependency:** `process_cplane` → `process_uplane` (sequential execution)
#
# **Why Sequential:**
# - C-Plane must be sent before U-Plane can be received
# - RU processes C-Plane to determine what U-Plane to send back
# - Dependency ensures correct ordering without explicit synchronization
#
# **Task Graph Creation:**
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: Task graph construction</i></summary>
#
# ```cpp
# // Create task graph with sequential C-Plane and U-Plane processing (from fronthaul_app.cpp)
# adspt::TaskGraph graph("fronthaul_processing");
#
# bool is_first_slot = true;
# auto cplane_task = graph.register_task("process_cplane")
#                        .function(fronthaul_app::make_process_cplane_func(
#                            fronthaul, fapi_replay, is_first_slot, t0, tai_offset))
#                        .add();
#
# graph.register_task("process_uplane")
#     .depends_on(cplane_task)  // U-Plane waits for C-Plane completion
#     .function(fronthaul_app::make_process_uplane_func(fronthaul, fapi_replay))
#     .add();
#
# graph.build();
# ```
# </details>

# %% [markdown]
# ### Real-Time Scheduling Configuration
#
# The fronthaul application uses two real-time cores for deterministic processing:
#
# | Core | Role | Purpose |
# |------|------|---------|
# | **7** | Trigger | Fires every slot period (500 μs), schedules task graph |
# | **8** | Worker | Executes process_cplane → process_uplane sequentially |
#
# Both cores run at real-time priority 95 (SCHED_FIFO) and are isolated from OS interference.
#
# **Configuration Code:**
#
# <details>
# <summary><b>Click to expand:</b> <i>Example: Real-time scheduler and trigger setup</i></summary>
#
# ```cpp
# // Real-time scheduler and trigger configuration (from fronthaul_app.cpp)
# static constexpr int RT_PRIORITY = 95;
# static constexpr std::uint32_t MONITOR_CORE = 0;
# static constexpr std::uint32_t WORKER_CORE = 8;
# static constexpr std::uint32_t TRIGGER_CORE = 7;
#
# // Create task scheduler with pinned RT worker
# auto scheduler =
#     adspt::TaskScheduler::create()
#         .workers(adspt::WorkersConfig{{
#             adspt::WorkerConfig::create_pinned_rt(WORKER_CORE, RT_PRIORITY)
#         }})
#         .monitor_core(MONITOR_CORE)
#         .build();
#
# // Create timed trigger for slot-based execution
# auto trigger = adspt::TimedTrigger::create(
#                    [&scheduler, &graph]() {
#                        scheduler.schedule(graph);  // Execute task graph
#                    },
#                    std::chrono::nanoseconds{slot_period_ns})  // 500 μs for 30 kHz SCS
#                .pin_to_core(TRIGGER_CORE)
#                .with_stats_core(MONITOR_CORE)
#                .with_rt_priority(RT_PRIORITY)
#                .enable_statistics()
#                .max_triggers(num_slots)  // Optional: limit number of slots
#                .build();
#
# // Start trigger at calculated SFN 0 time
# trigger.start(adspt::Nanos{start_time_ns});
# ```
# </details>
#
# **Key Characteristics:**
# - **Slot-aligned execution:** Trigger fires at precise slot boundaries (SFN 0 alignment)
# - **High-resolution timing:** < 1 μs jitter using high-resolution timer
# - **Real-time priority:** Both trigger and worker run at SCHED_FIFO priority 95
# - **Core isolation:** Cores 7 and 8 are isolated from OS (no interrupts/context switches)
#
# **Timing Diagram:**
#
# - The timed trigger runs on core 7 and is responsible for scheduling the task graph every slot.
# - The scheduling consists of simply putting the task graph on the worker queue, and the actual
#   execution of the task graph happens on worker core 8.
#
# The following diagram illustrates the task execution timeline:
#
# ![Task Execution and Timed Trigger](../../figures/generated/fronthaul_task_timeline.drawio.svg)

# %% [markdown]
# ## 7. Running the Fronthaul Test
#
# This section demonstrates how to build and run the fronthaul integration test
# using CMake and CTest.

# %% [markdown]
# ### Build the Fronthaul Application

# %%
import os
import sys

# Import shared tutorial utilities from tutorial_utils.py (in the same directory)
# Contains helper functions for Docker container interaction and project navigation
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

# %%
IN_DOCKER = is_running_in_docker()
PROJECT_ROOT = get_project_root()
CONTAINER_NAME = f"aerial-framework-base-{os.environ.get('USER', 'default')}"

print(f"Project root: {PROJECT_ROOT}")
if IN_DOCKER:
    print("✅ Running inside Docker container")
else:
    print(f"Running on host, will use container: {CONTAINER_NAME}")
    check_container_running(CONTAINER_NAME)
    print(f"✅ Container '{CONTAINER_NAME}' is running")
print("✅ Step 7a complete: Environment setup verified")

# %% [markdown]
# **Configure CMake preset:**

# %%
# Configure CMake with preset
preset = "gcc-release"
build_dir = PROJECT_ROOT / "out" / "build" / preset

configure_cmake(build_dir, preset)
print("✅ Step 7b complete: CMake configured")

# %% [markdown]
# **Build fronthaul and FAPI targets:**
#
# Build `fronthaul_all` and `fapi_all` targets to compile all required components for testing.

# %%
# Build fronthaul_all and fapi_all targets
try:
    build_cmake_target(build_dir, ["fronthaul_all", "fapi_all"])
except RuntimeError as e:
    print(f"❌ Build failed: {e}")
    print("\nNote: Error message shows last few lines of output.")
    print("If build fails, enter the container to run commands manually and view full logs:")
    print("  docker exec -it aerial-framework-base-$USER bash -l")
    print(f"  cmake --build out/build/{preset} --target fronthaul_all fapi_all")
    sys.exit(1)
print("✅ Step 7c complete: Fronthaul application built")

# %% [markdown]
# ### Running the Integration Test
#
# **Run the test with default parameters:**
# ```bash
# ctest --preset gcc-release -R fronthaul_app.integration_test
# ```
#
# **Environment Variables for Test Configuration:**
#
# The fronthaul integration test uses environment variables to configure test parameters:
#
# | Variable | Default | Purpose |
# |----------|---------|---------|
# | **TEST_CELLS** | 1 | Number of cells |
# | **TEST_SLOTS** | 100 | Test duration (500 μs/slot @ 30 kHz SCS) |
# | **TEST_VECTOR** | TVnr_7201_gNB_FAPI_s0.h5 | Test vector for FAPI generation |
#
# **Note:** Currently, only `TEST_CELLS=1` is supported.
#
# **Usage Examples with Custom Parameters:**
# ```bash
# # Test for 200 slots
# TEST_SLOTS=200 ctest --preset gcc-release -R fronthaul_app.integration_test
# ```
#
# **Note:** TEST_VECTOR affects FAPI capture generation (testMAC), not fronthaul_app directly.
# The fronthaul test uses the generated FAPI capture files.

# %% [markdown]
# ### FAPI Capture File
#
# The integration test constructs FAPI capture filenames based on TEST_CELLS:
#
# **Pattern:** `fapi_capture_fapi_sample_${TEST_CELLS}C.fapi`
#
# **Example:**
# - `TEST_CELLS=1` → `fapi_capture_fapi_sample_1C.fapi`
#
# **Location:** `${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/testMAC/testMAC/`
#
# **Generation:** FAPI capture files are generated by running the FAPI integration test first:
# ```bash
# # Generate FAPI capture
# ctest --preset gcc-release -R fapi_sample.integration_test
#
# # Then run fronthaul test
# ctest --preset gcc-release -R fronthaul_app.integration_test
# ```

# %% [markdown]
# ### Integration Test
#
# The integration test uses `run_fronthaul_integration_test.py` to coordinate both sides:
#
# | Component | Role | Launch Mode | Key Arguments |
# |-----------|------|-------------|---------------|
# | **ru_emulator** | RU side | Background | `--channels PUSCH --config ru_emulator_config.yaml` |
# | **fronthaul_app** | DU side | Foreground | `--nic <pcie> --config <yaml> --fapi-file <fapi>` |
#
# **Script Location:** `ran/runtime/fronthaul/tools/src/run_fronthaul_integration_test.py`
#
# **RU Config Template:** `ran/runtime/fronthaul/tools/config/ru_emulator_config.yaml.in`
#
# **Template Substitutions:**
# - `@RU_PCIE_ADDR_SHORT@` → RU-side NIC PCIe address (e.g., `17:00.1`)
# - `@RU_MAC_ADDRESS@` → RU-side NIC MAC address
# - `@DU_MAC_ADDRESS@` → DU-side NIC MAC address
#
# **Generated Config:**
# `${CMAKE_BINARY_DIR}/aerial_sdk/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator_config.yaml`
#
# **Key Configuration Sections:**
# - **Network:** PCIe address, MAC addresses, VLANs
# - **Cells:** Per-cell configuration (name, VLAN, timing)
# - **Timing:** T1a, Ta4 windows, slot period
# - **GPS:** Alpha/beta parameters for time synchronization

# %% [markdown]
# ### Running the Integration Test with CTest
#
# **Note:** This test requires NIC hardware (BF3 configured in loopback) and real-time system setup.
# The test will be skipped if networking devices are not available.
#

# %%
if os.environ.get("SKIP_NOTEBOOK_CTESTS", "").lower() not in ("1", "true", "yes"):
    # Check if networking devices are available inside container
    if check_network_devices(CONTAINER_NAME):
        print("Running fronthaul integration test (default: 1 cell, 100 slots)...")

        cmd = f"ctest --preset {preset} -R fronthaul_app.integration_test"
        result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

        if result.returncode == 0:
            print("✅ Integration test passed")
        else:
            print("⚠️  Integration test failed")
            print("\nNote: This cell displays only the last few lines of output.")
            print("If test fails, enter the container to run commands manually and view full logs:")
            print("  docker exec -it aerial-framework-base-$USER bash -l")
            print(f"  ctest --preset {preset} -R fronthaul_app.integration_test")

        print("\nTest output (last few lines):")
        show_output(result, lines=20)
        print("✅ Step 7d complete: Integration test executed")
else:
    print("⏭️  Skipping fronthaul ctests (SKIP_NOTEBOOK_CTESTS set)")
    print("✅ Step 7d complete: Integration test skipped (SKIP_NOTEBOOK_CTESTS set)")

# %% [markdown]
# **Run with a different configuration:**

# ```bash
# # Example: Run with a different test vector and 20000 slots
# TEST_VECTOR=TVnr_7204_gNB_FAPI_s0.h5 TEST_SLOTS=20000 ctest --preset gcc-release \
#     -R fronthaul_app.integration_test
# ```

# %% [markdown]
# ### Test Output and Verification
#
# **Expected Output:**
# ```
# The following tests passed:
#  	fapi_sample.fixture_setup
#  	fronthaul_app.integration_test
#
#  100% tests passed, 0 tests failed out of 2
#
#  Label Time Summary:
#  integration     =  24.57 sec*proc (2 tests)
#  requires_nic    =  24.06 sec*proc (1 test)
#
#  Total Test time (real) =  24.58 sec
# ```
#
# **Key Metrics to Check:**
# - **Packets sent:** Should match expected count (cells x slots x packets_per_slot)
# - **Packets received:** Should match RU emulator transmission
# - **Timing violations:** Should be zero (all packets within T1a/Ta4 windows)
# - **GPU kernel timeouts:** Should be zero (all U-Plane packets received)

# %% [markdown]
# ## Next Steps
#
# - Explore PHY processing pipelines (PUSCH receiver tutorial)
# - Profile GPU kernel performance with Nsight Systems
# - Integrate with full RAN stack (testMAC + fronthaul + PHY)
#
# ## Resources
#
# - [Aerial CUDA-Accelerated RAN Documentation](https://docs.nvidia.com/aerial/cuda-accelerated-ran/)
# - [DOCA GPUNetIO Programming Guide](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html)
# - Framework networking: `framework/net/`
# - Fronthaul application: `ran/runtime/fronthaul/`
#
# ## Troubleshooting
#
# **1. Loopback Cable Issues:**
# - **Loopback not detected:** Verify DAC cable is properly connected to both BF3 ports; check cable
#   supports 100 GbE; run loopback detection tool:
#   `ctest --preset gcc-release -R fronthaul_tools.detect_loopback`
# - **Wrong interfaces detected:** Manually specify interfaces with `--du-interface` and
#   `--ru-interface` flags in `run_fronthaul_integration_test.py` script
# - **Link down:** Check interface status with `ip link show`; bring up interfaces:
#   `sudo ip link set <interface> up`
#
# **2. Time Synchronization Issues:**
# - **Timing violations between fronthaul_app and ru_emulator:** Check PTP and PHC2SYS are running
#   outside of the container:
#   ```bash
#   # Check PTP daemon status
#   ps aux | grep ptp4l
#
#   # Check PHC2SYS status
#   ps aux | grep phc2sys
#
#   # Check PTP status
#   sudo systemctl status ptp4l.service
#
#   # Make sure NTP is turned off
#   sudo timedatectl set-ntp false
#   timedatectl
#   ```
# - **T1a/Ta4 window violations:** Check fronthaul_app logs for timing warnings; increase window
#   sizes in `ru_emulator_config.yaml` if needed
#
# **3. CPU Core Isolation Issues:**
# - **Tasks intermittently stalled:** Verify worker and timing cores 7, 8 are isolated:
#   ```bash
#   cat /sys/devices/system/cpu/isolated
#   ```
# - **Missing U-Plane packets:** Check for OS interference on isolated cores; verify kernel boot
#   parameters include `isolcpus=7,8 nohz_full=7,8 rcu_nocbs=7,8`; reboot if parameters
#   are missing
#
# **4. FAPI Capture File Issues:**
# - **FAPI file not found:** Generate FAPI capture first:
#   `ctest --preset gcc-release -R fapi_sample.integration_test`
# - **File path errors:** Check FAPI_CAPTURE_DIR environment variable points to correct directory
#
# **5. Build and Test Issues:**
# - **Build fails:** Ensure all dependencies installed; check `README.md` for build prerequisites;
#   clean build: `rm -rf out/build/<preset>`
# - **Configuration or build fails:** If CMake configure or build steps fail, enter the container
#   to run commands manually and view complete logs:
#   - Enter container: `docker exec -it aerial-framework-base-$USER bash -l`
#   - Configure: `cmake --preset gcc-release -DENABLE_CLANG_TIDY=OFF`
#   - Build: `cmake --build out/build/gcc-release --target fronthaul_all fapi_all`
# - **View full build/test output:** If the build or tests fail, the notebook displays only
#   the last few lines. Enter the container to run commands manually and view complete logs:
#   - Enter container: `docker exec -it aerial-framework-base-$USER bash -l`
#   - Build: `cmake --build out/build/gcc-release --target fronthaul_all fapi_all`
#   - Test: `ctest --preset gcc-release -R fronthaul_app.integration_test`
# - **Test timeout:** CTest has a default timeout of 1500 seconds. For long-running tests
#   with large `TEST_SLOTS` values, you may need to increase the timeout:
#   ```bash
#   TEST_SLOTS=200000 ctest --preset gcc-release --timeout 3000 -R fronthaul_app.integration_test
#   ```
#   For hardware issues (NIC, GPU), check that devices are accessible and properly configured.
# - **NIC not accessible:** Verify VFIO/IOMMU setup; check hugepages allocation:
#   `grep Huge /proc/meminfo`; ensure user has permissions for `/dev/vfio`
# - **Networking devices not detected:** Ensure DEV_VFIO, DEV_INFINIBAND, DEV_GDRDRV, and
#   DEV_HUGEPAGES environment variables are set and point to actual devices (not /dev/null).
#   These are set by the container setup script `container/setup_container.sh`.
#
# **6. GPU Kernel Issues:**
# - **Order kernel timeouts:** Check GPU is accessible: `nvidia-smi`; verify CUDA version
#   compatibility; check for GPU memory issues
# - **Packet processing errors:** Review fronthaul_app logs for error messages; verify
#   U-Plane configuration matches RU emulator settings
#
# See `README.md` and the [Aerial CUDA-Accelerated RAN Installation
# Guide](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/install_guide/installing_tools_gh.html)
# for more details.
