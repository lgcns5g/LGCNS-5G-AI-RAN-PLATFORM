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
 * @file net_sender.cpp
 * @brief Network sender sample application
 */

#include <cstdlib> // for EXIT_SUCCESS, EXIT_FAILURE
#include <exception>
#include <format>   // for format
#include <iostream> // for cerr
#include <string>
#include <system_error> // for std::error_code

#include <driver_types.h>
#include <quill/LogMacros.h>
#include <tl/expected.hpp>

// clang-format off
// example-begin net-sender-includes-1
#include "net/dpdk_types.hpp"
#include "net/env.hpp"
#include "net/nic.hpp"
// example-end net-sender-includes-1
// clang-format on

#include "gpunetio_kernels.hpp"  // for launch_gpunetio_sender_kernel
#include "log/rt_log_macros.hpp" // for RT_LOGC_*
#include "net/doca_txq.hpp"
#include "net/net_log.hpp"       // for Net component
#include "net_samples.hpp"       // for utility functions
#include "utils/cuda_stream.hpp" // for CudaStream

/**
 * @brief Main entry point for network sender application
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return 0 on success, non-zero on error
 */
int main(int argc, const char **argv) {
    using namespace framework::net;

    try {
        setup_logging();

        const auto args = parse_arguments(NetSample::Sender, argc, argv);
        if (!args.has_value()) {
            // Empty error string means --help or --version was shown (success)
            // Non-empty error string means parse error (failure)
            if (!args.error().empty()) {
                RT_LOGC_ERROR(Net::NetGeneral, "{}", args.error());
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        const auto config = create_net_env_config(NetSample::Sender, *args);
        const Env env{config};

        if (args->cpu_only) {
            if (const auto result = send_dpdk_message(env, args->mac_addr); result) {
                RT_LOGC_ERROR(
                        Net::NetDpdk, "Failed to send DPDK message: {}", get_error_name(result));
                return EXIT_FAILURE;
            }
        } else {
            // example-begin net-sender-txq-usage-1
            // Get DOCA TX queue parameters for GPU kernel
            const auto *doca_txq = env.nic().doca_tx_queue(0).params();
            if (doca_txq == nullptr) {
                RT_LOGC_ERROR(Net::NetDoca, "Failed to get DOCA TX queue structure");
                return EXIT_FAILURE;
            }

            // Launch GPU kernel with DOCA TX queue
            const framework::utils::CudaStream stream;
            const auto result = launch_gpunetio_sender_kernel(stream.get(), *doca_txq);
            if (result != 0) {
                RT_LOGC_ERROR(Net::NetGpu, "Failed to launch sender kernel: {}", result);
                return EXIT_FAILURE;
            }
            // example-end net-sender-txq-usage-1
        }

        RT_LOGC_INFO(Net::NetGpu, "Packet sent successfully!");

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << std::format("Unhandled exception: {}\n", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return EXIT_FAILURE;
    }
}
