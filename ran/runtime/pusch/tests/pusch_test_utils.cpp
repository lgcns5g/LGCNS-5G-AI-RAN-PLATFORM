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

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <driver_types.h>
#include <quill/LogMacros.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "ldpc/crc_decoder_module.hpp"
#include "ldpc/ldpc_decoder_module.hpp"
#include "ldpc/ldpc_derate_match_module.hpp"
#include "ldpc/ldpc_params.hpp"
#include "log/components.hpp"
#include "log/rt_log_macros.hpp"
#include "memory/unique_ptr_utils.hpp"
#include "pusch/inner_rx_module.hpp"
#include "pusch_test_utils.hpp"
#include "ran_common.hpp"
#include "tensor/data_types.hpp"
#include "utils/core_log.hpp"
#include "utils/error_macros.hpp"

namespace ran::pusch {

namespace {
namespace pipeline = framework::pipeline;
namespace tensor = framework::tensor;
} // namespace

std::vector<framework::memory::UniqueDevicePtr<std::byte>> prepare_pusch_inputs(
        std::vector<pipeline::PortInfo> &inputs,
        const ran::common::PhyParams &phy_params,
        const ran::aerial_tv::CuphyPuschTestVector &test_vector,
        cudaStream_t stream) {
    std::vector<framework::memory::UniqueDevicePtr<std::byte>> input_device_ptrs;

    const tensor::TensorInfo tensor_info{
            tensor::NvDataType::TensorR16F,
            {phy_params.num_rx_ant,
             ran::common::OFDM_SYMBOLS_NORMAL_CYCLIC_PREFIX,
             static_cast<std::size_t>(phy_params.num_prb) * ran::common::NUM_SUBCARRIERS_PER_PRB,
             ran::common::REAL_IMAG_INTERLEAVED}};

    // Allocate memory for RX buffer
    const std::size_t size_bytes = tensor_info.get_total_elements() * sizeof(__half);
    input_device_ptrs.push_back(framework::memory::make_unique_device<std::byte>(size_bytes));
    void *d_ptr = input_device_ptrs.back().get();
    RT_LOG_INFO("Allocated {} bytes at device ptr {}", size_bytes, d_ptr);

    const pipeline::DeviceTensor device_tensor{.device_ptr = d_ptr, .tensor_info = tensor_info};
    const pipeline::PortInfo port_info{.name = "xtf", .tensors = {device_tensor}};

    inputs.push_back(port_info);

    // Read inputs from the test vector and copy to device using stream-aware async copy
    const auto xtf_array = test_vector.read_complex_array<__half>("X_tf_fp16");

    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
            inputs[0].tensors[0].device_ptr,
            xtf_array.data.data(),
            xtf_array.data.size() * sizeof(__half),
            cudaMemcpyHostToDevice,
            stream));

    // Synchronize stream to ensure input data is ready before returning
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    return input_device_ptrs;
}

template <typename T>
std::vector<T> tensor_to_host_vector(
        const tensor::TensorInfo &tensor_info, const void *device_ptr, cudaStream_t stream) {
    const std::size_t num_elements = tensor_info.get_total_elements();
    std::vector<T> host_data(num_elements);

    // Use async copy on the same stream as GPU work for proper synchronization
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaMemcpyAsync(
            host_data.data(),
            device_ptr,
            num_elements * sizeof(T),
            cudaMemcpyDeviceToHost,
            stream));

    // Synchronize stream to ensure async copy completes before returning
    FRAMEWORK_CUDA_RUNTIME_CHECK_THROW(cudaStreamSynchronize(stream));

    return host_data;
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_TENSOR_TO_HOST_VECTOR(T)                                                       \
    template std::vector<T> tensor_to_host_vector<T>(                                              \
            const tensor::TensorInfo &tensor_info, const void *device_ptr, cudaStream_t stream);

// Explicit template instantiations
INSTANTIATE_TENSOR_TO_HOST_VECTOR(float)
INSTANTIATE_TENSOR_TO_HOST_VECTOR(__half)

#undef INSTANTIATE_TENSOR_TO_HOST_VECTOR

pipeline::PipelineSpec create_pusch_pipeline_spec(
        const std::string &instance_id,
        const ran::common::PhyParams &phy_params,
        const pipeline::ExecutionMode execution_mode) {
    pipeline::PipelineSpec spec;
    spec.pipeline_name = "PuschPipeline";
    spec.execution_mode = execution_mode;

    // InnerRx module configuration
    const ran::pusch::InnerRxModule::StaticParams inner_rx_params{
            .phy_params = phy_params, .execution_mode = spec.execution_mode};

    const pipeline::ModuleCreationInfo inner_rx_module_spec{
            .module_type = "inner_rx_module",
            .instance_id = std::format("inner_rx_{}", instance_id),
            .init_params = std::any(inner_rx_params)};

    spec.modules.emplace_back(inner_rx_module_spec);

    // LDPC Derate Match Module configuration
    const ran::ldpc::LdpcDerateMatchModule::StaticParams ldpc_derate_match_params{
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_rm_llrs_per_cb = ran::ldpc::MAX_NUM_RM_LLRS_PER_CB,
            .max_num_ue_grps = ran::common::MAX_NUM_UE_GRPS};

    const pipeline::ModuleSpec ldpc_derate_match_module_spec{pipeline::ModuleCreationInfo{
            .module_type = "ldpc_derate_match_module",
            .instance_id = std::format("ldpc_derate_match_{}", instance_id),
            .init_params = std::any(ldpc_derate_match_params)}};

    spec.modules.emplace_back(ldpc_derate_match_module_spec);

    // LDPC Decoder Module configuration
    const ran::ldpc::LdpcDecoderModule::StaticParams ldpc_decoder_params{
            .clamp_value = ran::ldpc::LDPC_CLAMP_VALUE,
            .max_num_iterations = ran::ldpc::LDPC_MAX_ITERATIONS,
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS,
            .normalization_factor = ran::ldpc::LDPC_NORMALIZATION_FACTOR,
            .max_iterations_method = ran::ldpc::LdpcMaxIterationsMethod::Lut,
            .max_num_ldpc_het_configs = ran::ldpc::LDPC_MAX_HET_CONFIGS};

    const pipeline::ModuleSpec ldpc_decoder_module_spec{pipeline::ModuleCreationInfo{
            .module_type = "ldpc_decoder_module",
            .instance_id = std::format("ldpc_decoder_{}", instance_id),
            .init_params = std::any(ldpc_decoder_params)}};

    spec.modules.emplace_back(ldpc_decoder_module_spec);

    // CRC Decoder Module configuration
    const ran::ldpc::CrcDecoderModule::StaticParams crc_decoder_params{
            .max_num_cbs_per_tb = ran::common::MAX_NUM_CBS_PER_TB,
            .max_num_tbs = ran::common::MAX_NUM_TBS};

    const pipeline::ModuleSpec crc_decoder_module_spec{pipeline::ModuleCreationInfo{
            .module_type = "crc_decoder_module",
            .instance_id = std::format("crc_decoder_{}", instance_id),
            .init_params = std::any(crc_decoder_params)}};

    spec.modules.emplace_back(crc_decoder_module_spec);

    spec.external_inputs = {"xtf"};
    spec.external_outputs = {"post_eq_noise_var_db", "post_eq_sinr_db", "tb_crcs", "tb_payloads"};

    return spec;
}

BenchmarkStatistics compute_benchmark_statistics(const std::vector<double> &times) {
    if (times.empty()) {
        return {};
    }

    auto sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    const auto count = sorted_times.size();
    const auto min_time = sorted_times.front();
    const auto max_time = sorted_times.back();
    const auto median_time = sorted_times.at(count / 2);
    const auto p95_index =
            std::min(static_cast<std::size_t>(static_cast<double>(count) * 0.95), count - 1);
    const auto p95_time = sorted_times.at(p95_index);

    // Calculate mean and stddev
    const auto sum = std::accumulate(sorted_times.begin(), sorted_times.end(), 0.0);
    const auto mean_time = sum / static_cast<double>(count);

    double variance{0.0};
    for (const auto time : sorted_times) {
        const auto diff = time - mean_time;
        variance += diff * diff;
    }
    const auto stddev_time = std::sqrt(variance / static_cast<double>(count));

    return {min_time, max_time, mean_time, median_time, stddev_time, p95_time, count};
}

} // namespace ran::pusch
