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

#include <any>
#include <array>   // for array
#include <atomic>  // for atomic, memory_order_relaxed
#include <cstddef> // for size_t, ptrdiff_t
#include <cstdint> // for uint32_t, uint16_t, uint8_t
#include <format>
#include <functional> // for function
#include <memory>
#include <optional>
#include <span>
#include <stdexcept> // for invalid_argument
#include <string>    // for operator==, operator+
#include <unordered_map>
#include <utility>
#include <vector>

#include <driver_types.h> // for cudaMemcpyKind, CUstream_st
#include <quill/LogMacros.h>
#include <wise_enum_detail.h> // for optional_type

#include <wise_enum.h> // for from_string

#include <cuda_runtime_api.h>

#include "driver/driver.hpp"
#include "driver/driver_log.hpp"
#include "driver/pusch_pipeline_context.hpp" // for PuschPipelineContext
#include "log/components.hpp"
#include "log/rt_log_macros.hpp"
#include "pipeline/types.hpp" // for PortInfo, DeviceTensor
#include "pusch/pusch_defines.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "ran_common.hpp" // for NUM_CELLS_SUPPORTED
#include "tensor/data_types.hpp"
#include "tensor/tensor_info.hpp" // for TensorInfo
#include "utils/error_macros.hpp"

namespace ran::driver {

namespace pipeline = framework::pipeline;
namespace common = ran::common;
namespace pusch = ran::pusch;

Driver::Driver() {
    // Register Driver logging components
    framework::log::register_component<DriverComponent>(
            {{DriverComponent::Driver, framework::log::LogLevel::Info},
             {DriverComponent::PuschPipelineContext, framework::log::LogLevel::Info}});
}

void Driver::create_pusch_pipeline(
        const common::PhyParams &phy_params,
        const std::string &execution_mode,
        UlIndicationCallback ul_indication_callback,
        std::span<const pipeline::PortInfo> order_kernel_outputs) {
    // Register UL indication callback
    set_ul_indication_callback(std::move(ul_indication_callback));

    // Convert string to ExecutionMode
    const auto mode_opt = wise_enum::from_string<pipeline::ExecutionMode>(execution_mode);
    if (!mode_opt.has_value()) {
        RT_LOGEC_ERROR(
                DriverComponent::Driver,
                DriverEvent::CreatePuschPipeline,
                "Invalid execution mode: '{}'. Valid modes: 'Stream', 'Graph'",
                execution_mode);
        throw std::invalid_argument("Invalid execution mode: " + execution_mode);
    }

    // Validate Order Kernel outputs were provided
    if (order_kernel_outputs.empty()) {
        RT_LOGEC_ERROR(
                DriverComponent::Driver,
                DriverEvent::CreatePuschPipeline,
                "Order Kernel outputs not provided - ensure Fronthaul U-Plane is initialized");
        throw std::runtime_error("Order Kernel outputs not provided");
    }

    // Validate first output has valid tensors
    if (order_kernel_outputs[0].tensors.empty()) {
        RT_LOGEC_ERROR(
                DriverComponent::Driver,
                DriverEvent::CreatePuschPipeline,
                "Order Kernel output has no tensors");
        throw std::runtime_error("Order Kernel output has no tensors");
    }

    RT_LOGC_INFO(
            DriverComponent::Driver,
            "Order Kernel outputs received - buffer ptr={}, size={} elements",
            order_kernel_outputs[0].tensors[0].device_ptr,
            order_kernel_outputs[0].tensors[0].tensor_info.get_total_elements());

    // Validate buffer sizes match between Order Kernel and expected PUSCH input
    const auto &order_kernel_tensor_info = order_kernel_outputs[0].tensors[0].tensor_info;
    const std::size_t order_kernel_size_bytes = order_kernel_tensor_info.get_total_elements() *
                                                framework::tensor::get_nv_type_storage_element_size(
                                                        order_kernel_tensor_info.get_type());

    // Expected PUSCH input size: [num_rx_ant, OFDM_symbols, subcarriers, real/imag]
    const framework::tensor::TensorInfo expected_tensor_info{
            framework::tensor::NvDataType::TensorR16F,
            {phy_params.num_rx_ant,
             common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX,
             static_cast<std::size_t>(phy_params.num_prb) * common::NUM_SUBCARRIERS_PER_PRB,
             common::REAL_IMAG_INTERLEAVED}};
    const std::size_t expected_pusch_size_bytes =
            expected_tensor_info.get_total_elements() *
            framework::tensor::get_nv_type_storage_element_size(expected_tensor_info.get_type());

    if (order_kernel_size_bytes != expected_pusch_size_bytes) {
        const std::string error_msg = std::format(
                "Buffer size mismatch: Order Kernel={} bytes, PUSCH expects={} bytes",
                order_kernel_size_bytes,
                expected_pusch_size_bytes);
        RT_LOGEC_ERROR(DriverComponent::Driver, DriverEvent::CreatePuschPipeline, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOGC_INFO(
            DriverComponent::Driver,
            "Buffer sizes validated successfully (Order Kernel={} bytes [{}], PUSCH expected={} "
            "bytes [{}])",
            order_kernel_size_bytes,
            framework::tensor::nv_get_data_type_string(order_kernel_tensor_info.get_type()),
            expected_pusch_size_bytes,
            framework::tensor::nv_get_data_type_string(expected_tensor_info.get_type()));

    // Pass physical layer parameters and Order Kernel outputs to create PUSCH pipeline
    // Convert span to vector for compatibility with existing interface
    const std::vector<pipeline::PortInfo> order_kernel_outputs_vec(
            order_kernel_outputs.begin(), order_kernel_outputs.end());
    pusch_pipeline_context.create_pusch_pipeline(
            phy_params, mode_opt.value(), order_kernel_outputs_vec);
}

void Driver::set_ul_indication_callback(UlIndicationCallback callback) {
    ul_indication_callback_ = std::move(callback);
    RT_LOGC_INFO(DriverComponent::Driver, "UL indication callback registered");
}

bool Driver::process_slot_response(
        const std::size_t slot,
        const std::uint16_t cell_id,
        const std::uint32_t active_cell_bitmap) {
    // Validate cell_id to prevent undefined behavior in bit shift
    if (cell_id >= common::NUM_CELLS_SUPPORTED) {
        RT_LOGEC_ERROR(
                DriverComponent::Driver,
                DriverEvent::SlotResponseReceived,
                "Slot {}: Invalid cell_id {} (max: {})",
                slot,
                cell_id,
                common::NUM_CELLS_SUPPORTED);
        return false;
    }

    // Set bit for this cell in the slot response bitmap (lock-free atomic OR)
    const std::uint32_t prev_bitmap =
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            slot_ready[slot].slot_rsp_rcvd.fetch_or(1U << cell_id, std::memory_order_relaxed);
    const std::uint32_t current_bitmap = prev_bitmap | (1U << cell_id);

    RT_LOGEC_DEBUG(
            DriverComponent::Driver,
            DriverEvent::SlotResponseReceived,
            "Slot {}: Slot response from cell_id={}, slot_rsp_rcvd=0x{:X}, "
            "active_cell_bitmap=0x{:X}",
            slot,
            cell_id,
            current_bitmap,
            active_cell_bitmap);

    // Check if all active cells have responded
    if (current_bitmap == active_cell_bitmap) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        slot_ready[slot].is_completed.store(true, std::memory_order_relaxed);
        RT_LOGEC_INFO(
                DriverComponent::Driver,
                DriverEvent::SlotResponseReceived,
                "Slot {} is completed - all active cells have responded (bitmap=0x{:X})",
                slot,
                active_cell_bitmap);

        return true;
    }

    return false;
}

void Driver::reset_slot_status(const std::size_t slot) {
    // Lock-free atomic reset
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    slot_ready[slot].slot_rsp_rcvd.store(0, std::memory_order_relaxed);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    slot_ready[slot].is_completed.store(false, std::memory_order_relaxed);

    RT_LOGEC_DEBUG(
            DriverComponent::Driver,
            DriverEvent::ResetSlotStatus,
            "Slot {}: Reset slot status",
            slot);
}

void Driver::send_ul_indication(const std::size_t slot) {
    // Retrieve the slot resources
    const PuschSlotResources &slot_resources = pusch_pipeline_context.get_slot_resources(slot);

    // Check if host buffers and pipeline are allocated (load atomic values)
    const std::ptrdiff_t host_buffers_idx =
            slot_resources.host_buffers_index.load(std::memory_order_relaxed);
    const std::ptrdiff_t pipeline_idx =
            slot_resources.pipeline_index.load(std::memory_order_relaxed);
    if (host_buffers_idx < 0 || pipeline_idx < 0) {
        RT_LOGC_ERROR(
                DriverComponent::Driver, "Slot {}: No host buffers or pipeline allocated", slot);
        return;
    }

    // Get external outputs from pipeline resource
    const std::vector<pipeline::PortInfo> &external_outputs =
            pusch_pipeline_context.get_external_outputs_by_index(
                    static_cast<std::size_t>(pipeline_idx));

    RT_LOGC_INFO(
            DriverComponent::Driver,
            "Send UL indications: slot={} num_outputs={}",
            slot,
            external_outputs.size());

    PuschHostOutput &host_output = pusch_pipeline_context.get_host_output_by_index(
            static_cast<std::size_t>(host_buffers_idx));

    cudaStream_t stream =
            pusch_pipeline_context.get_stream_by_index(static_cast<std::size_t>(pipeline_idx));
    if (stream == nullptr) {
        RT_LOGC_ERROR(DriverComponent::Driver, "Slot {}: No stream allocated", slot);
        return;
    }

    for (std::size_t i = 0; i < external_outputs.size(); ++i) {
        const auto &port = external_outputs[i];
        RT_LOGC_DEBUG(
                DriverComponent::Driver,
                "Output port[{}]: name={} num_tensors={}",
                i,
                port.name,
                port.tensors.size());

        if (port.name == "tb_crcs") {
            if (port.tensors.empty()) {
                RT_LOGC_ERROR(DriverComponent::Driver, "Port '{}' has no tensors", port.name);
                continue;
            }
            FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                    host_output.pusch_outputs.tb_crcs.data(),
                    port.tensors[0].device_ptr,
                    host_output.pusch_outputs.tb_crcs.size() * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost,
                    stream));
        } else if (port.name == "post_eq_noise_var_db") {
            if (port.tensors.empty()) {
                RT_LOGC_ERROR(DriverComponent::Driver, "Port '{}' has no tensors", port.name);
                continue;
            }
            FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                    host_output.pusch_outputs.post_eq_noise_var_db.data(),
                    port.tensors[0].device_ptr,
                    host_output.pusch_outputs.post_eq_noise_var_db.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream));
        } else if (port.name == "post_eq_sinr_db") {
            if (port.tensors.empty()) {
                RT_LOGC_ERROR(DriverComponent::Driver, "Port '{}' has no tensors", port.name);
                continue;
            }
            FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                    host_output.pusch_outputs.post_eq_sinr_db.data(),
                    port.tensors[0].device_ptr,
                    host_output.pusch_outputs.post_eq_sinr_db.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream));
        } else if (port.name == "tb_payloads") {
            if (port.tensors.empty()) {
                RT_LOGC_ERROR(DriverComponent::Driver, "Port '{}' has no tensors", port.name);
                continue;
            }
            // Single cell case for now
            const std::size_t num_elements = port.tensors[0].tensor_info.get_total_elements();

            FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
                    host_output.pusch_outputs.tb_payloads[0],
                    port.tensors[0].device_ptr,
                    num_elements * sizeof(std::uint8_t),
                    cudaMemcpyDeviceToHost,
                    stream));
        } else {
            RT_LOGC_ERROR(DriverComponent::Driver, "Unknown output port: {}", port.name);
        }
    }
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));
    if (ul_indication_callback_) {
        ul_indication_callback_(slot);
    } else {
        RT_LOGC_ERROR(
                DriverComponent::Driver, "Slot {}: No UL indication callback registered", slot);
    }
}

void Driver::launch_pipelines(const std::size_t slot) {
    // Retrieve the slot resources
    const PuschSlotResources &slot_resources = pusch_pipeline_context.get_slot_resources(slot);

    // Check if host buffers are allocated (load atomic value)
    const std::ptrdiff_t host_buffers_idx =
            slot_resources.host_buffers_index.load(std::memory_order_relaxed);
    if (host_buffers_idx < 0) {
        RT_LOGEC_ERROR(
                DriverComponent::Driver,
                DriverEvent::LaunchPipelines,
                "Slot {}: No host buffers allocated",
                slot);
        return;
    }

    // Allocate pipeline resource
    auto maybe_pipeline = pusch_pipeline_context.get_pipeline_resource();
    if (!maybe_pipeline.has_value()) {
        RT_LOGEC_ERROR(
                DriverComponent::Driver,
                DriverEvent::LaunchPipelines,
                "Slot {}: Failed to allocate pipeline resource",
                slot);
        return;
    }

    // Extract pipeline resource (explicit for clang-analyzer)
    const std::size_t pipeline_index = maybe_pipeline.value().first;
    PuschPipelineResources *pipeline_res = maybe_pipeline.value().second;

    RT_LOGEC_DEBUG(
            DriverComponent::Driver,
            DriverEvent::LaunchPipelines,
            "Slot {}: Allocated pipeline resource {}",
            slot,
            pipeline_index);

    // Save pipeline index for this slot
    pusch_pipeline_context.save_pipeline_index(slot, pipeline_index);

    // Get host input using index from slot_resources (cast to size_t after validation)
    const PuschHostInput &host_input = pusch_pipeline_context.get_host_input_by_index(
            static_cast<std::size_t>(host_buffers_idx));

    // ========================================================================
    // Execute PUSCH Pipeline (Order Kernel already executed by Fronthaul)
    // ========================================================================
    // NOTE: Order Kernel is executed by Fronthaul::process_uplane() in the U-Plane task.
    // PUSCH pipeline uses pre-configured external_inputs pointing to Order Kernel's
    // output buffers (set up during create_pusch_pipeline with addresses from Fronthaul).
    // This is zero-copy: PUSCH reads directly from Order Kernel's output buffer.

    // Prepare PUSCH dynamic parameters
    const pusch::PuschDynamicParams pusch_dynamic_params =
            pusch_pipeline_context.prepare_pusch_dynamic_params(host_input.pusch_inputs);
    const pipeline::DynamicParams params{.module_specific_params = pusch_dynamic_params};

    RT_LOGEC_INFO(
            DriverComponent::Driver,
            DriverEvent::LaunchPipelines,
            "Slot {}: Calling configure_io",
            slot);
    pipeline_res->pipeline->configure_io(
            params,
            pipeline_res->external_inputs,
            pipeline_res->external_outputs,
            pipeline_res->stream);

    RT_LOGEC_INFO(
            DriverComponent::Driver,
            DriverEvent::LaunchPipelines,
            "Slot {}: Calling execute_stream",
            slot);

    pipeline_res->pipeline->execute_stream(pipeline_res->stream);

    RT_LOGEC_INFO(
            DriverComponent::Driver,
            DriverEvent::LaunchPipelines,
            "Slot {}: Stream execution completed",
            slot);

    // Release lambda - release resources allocated for this slot
    auto release_resources = [this, slot, host_buffers_idx, pipeline_index]() {
        RT_LOGC_DEBUG(
                DriverComponent::Driver,
                "Slot {}: Releasing resources (host_buffers_index={}, pipeline_index={})",
                slot,
                host_buffers_idx,
                pipeline_index);
        pusch_pipeline_context.release_host_buffers(static_cast<std::size_t>(host_buffers_idx));
        pusch_pipeline_context.release_pipeline_resource(pipeline_index);
        pusch_pipeline_context.clear_slot_resources(slot);
    };

    // Get actual output pointer from pipeline
    if (pipeline_res->external_outputs.empty() ||
        pipeline_res->external_outputs[0].tensors.empty()) {
        RT_LOGEC_ERROR(
                DriverComponent::Driver,
                DriverEvent::LaunchPipelines,
                "Slot {}: No output allocated",
                slot);
        release_resources();
        return;
    }

    send_ul_indication(slot);
    release_resources();

    RT_LOGEC_INFO(
            DriverComponent::Driver,
            DriverEvent::LaunchPipelines,
            "Slot {}: Completed pipeline execution",
            slot);
}

} // namespace ran::driver
