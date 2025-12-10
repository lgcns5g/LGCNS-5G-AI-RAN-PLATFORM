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

#include <format>
#include <memory>
#include <string>

#include <doca_ctx.h>
#include <doca_error.h>
#include <quill/LogMacros.h>

#include "log/rt_log_macros.hpp"
#include "net/details/doca_utils.hpp"
#include "net/gpu.hpp"
#include "net/net_log.hpp"

namespace framework::net {

void Gpu::GpuDeleter::operator()(doca_gpu *gpu) const noexcept {
    if (gpu != nullptr) {
        const doca_error_t result = doca_close_cuda_device(gpu);
        if (result != DOCA_SUCCESS) {
            RT_LOGC_ERROR(
                    Net::NetGpu, "Failed to close CUDA device: {}", doca_error_get_descr(result));
        }
    }
}

Gpu::Gpu(const GpuDeviceId gpu_device_id)
        : pci_bus_id_(doca_device_id_to_pci_bus_id(gpu_device_id.value())) {
    // Get PCI bus ID from CUDA device ID
    if (pci_bus_id_.empty()) {
        log_and_throw(
                Net::NetGpu, "Failed to get PCI bus ID for GPU device {}", gpu_device_id.value());
    }

    // Create the raw doca_gpu object
    doca_gpu *raw_gpu = nullptr;
    const doca_error_t result = doca_open_cuda_device(pci_bus_id_, &raw_gpu);
    if (result != DOCA_SUCCESS) {
        log_and_throw(
                Net::NetGpu, "Failed to create DOCA GPU device: {}", doca_error_get_descr(result));
    }

    if (raw_gpu == nullptr) {
        log_and_throw(Net::NetGpu, "DOCA GPU device creation returned null pointer");
    }

    // Transfer ownership to unique_ptr
    gpu_dev_.reset(raw_gpu);
}

const std::string &Gpu::pci_bus_id() const noexcept { return pci_bus_id_; }

doca_gpu *Gpu::get() const noexcept { return gpu_dev_.get(); }

} // namespace framework::net
