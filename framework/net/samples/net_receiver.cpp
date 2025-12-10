/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file net_receiver.cpp
 * @brief Network receiver sample application
 */

#include <atomic> // for std::atomic_bool
#include <chrono> // for std::chrono::milliseconds
#include <compare>
#include <csignal> // for signal, SIGINT, SIGTERM
#include <cstdint> // for uint32_t
#include <cstdlib> // for EXIT_SUCCESS, EXIT_FAILURE
#include <exception>
#include <format>   // for format
#include <iostream> // for cerr
#include <memory>
#include <string>
#include <thread> // for std::this_thread::sleep_for

#include <driver_types.h>
#include <quill/LogMacros.h>
#include <tl/expected.hpp>

#include <cuda_runtime.h> // for cudaStream_t, cudaStreamCreateWithFlags

#include "gpunetio_kernels.hpp" // for launch_gpunetio_receiver_kernel
#include "log/components.hpp"
#include "log/rt_log_macros.hpp"       // for RT_LOGC_*
#include "memory/unique_ptr_utils.hpp" // for make_unique_device, make_unique_pinned
#include "net/doca_rxq.hpp"
#include "net/env.hpp"     // for Env, EnvConfig
#include "net/net_log.hpp" // for Net component
#include "net/nic.hpp"
#include "net_samples.hpp" // for utility functions
#include "utils/core_log.hpp"
#include "utils/cuda_stream.hpp" // for CudaStream

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic_bool g_force_quit{false};

void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        g_force_quit = true;
    }
}
} // namespace

/**
 * @brief Main entry point for network receiver application
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return 0 on success, non-zero on error
 */
int main(int argc, const char **argv) {
    using namespace framework::net;
    using framework::memory::make_unique_device;
    using framework::memory::make_unique_pinned;

    try {
        setup_logging();

        const auto args = parse_arguments(NetSample::Receiver, argc, argv);
        if (!args.has_value()) {
            // Empty error string means --help or --version was shown (success)
            // Non-empty error string means parse error (failure)
            if (!args.error().empty()) {
                RT_LOGC_ERROR(Net::NetGeneral, "{}", args.error());
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        signal(SIGINT, signal_handler);  // NOLINT(cert-err33-c)
        signal(SIGTERM, signal_handler); // NOLINT(cert-err33-c)

        const auto config = create_net_env_config(NetSample::Receiver, *args);
        const Env env{config};

        // example-begin net-receiver-rxq-usage-1
        // Get DOCA RX queue parameters for GPU kernel
        const auto *doca_rxq = env.nic().doca_rx_queue(0).params();
        if (doca_rxq == nullptr) {
            RT_LOGC_ERROR(Net::NetDoca, "Failed to get DOCA RX queue structure");
            return EXIT_FAILURE;
        }

        auto gpu_exit_condition = make_unique_device<uint32_t>(1);
        auto cpu_exit_condition = make_unique_pinned<uint32_t>(1);

        *cpu_exit_condition = 0;
        if (!cuda_memcpy_host_to_device(
                    gpu_exit_condition.get(), cpu_exit_condition.get(), sizeof(uint32_t))) {
            return EXIT_FAILURE;
        }

        // Launch GPU kernel with DOCA RX queue
        const framework::utils::CudaStream stream;
        const auto result =
                launch_gpunetio_receiver_kernel(stream.get(), *doca_rxq, gpu_exit_condition.get());
        if (result != 0) {
            RT_LOGC_ERROR(Net::NetGpu, "Failed to launch receiver kernel: {}", result);
            return EXIT_FAILURE;
        }
        // example-end net-receiver-rxq-usage-1

        const auto start_time = std::chrono::steady_clock::now();
        const bool has_timeout = args->timeout_seconds > 0;
        const auto timeout_duration = std::chrono::seconds(args->timeout_seconds);
        bool timed_out = false;

        RT_LOGC_INFO(
                Net::NetGpu,
                "Receiver kernel launched, waiting for packet or Ctrl+C or timeout...");

        while (!g_force_quit) {
            // Break on completion or error
            if (const auto cuda_status = cudaStreamQuery(stream.get());
                cuda_status != cudaErrorNotReady) {
                if (cuda_status != cudaSuccess) {
                    RT_LOGC_ERROR(
                            Net::NetGpu,
                            "CUDA stream query failed: {}",
                            cudaGetErrorString(cuda_status));
                    return EXIT_FAILURE;
                }
                break; // Success
            }

            if (has_timeout) {
                const auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed >= timeout_duration) {
                    RT_LOGC_INFO(
                            Net::NetGpu,
                            "Timeout reached after {}s - stopping",
                            args->timeout_seconds);
                    timed_out = true;
                    break;
                }
            }

            // Brief sleep to avoid busy waiting
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(10ms);
        }

        if (g_force_quit) {
            RT_LOGC_INFO(Net::NetGpu, "Received termination signal");
        }

        if (g_force_quit || timed_out) {
            RT_LOGC_INFO(Net::NetGpu, "Stopping kernel");
            *cpu_exit_condition = 1;
            if (!cuda_memcpy_host_to_device(
                        gpu_exit_condition.get(), cpu_exit_condition.get(), sizeof(uint32_t))) {
                return EXIT_FAILURE;
            }
        } else {
            RT_LOGC_INFO(Net::NetGpu, "Kernel completed - packet received successfully");
        }

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << std::format("Unhandled exception: {}\n", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return EXIT_FAILURE;
    }
}
