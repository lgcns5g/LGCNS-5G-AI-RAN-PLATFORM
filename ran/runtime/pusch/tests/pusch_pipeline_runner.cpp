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
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include <driver_types.h>

#include <gsl-lite/gsl-lite.hpp>

#include "aerial_tv/aerial_tv_utils.hpp"
#include "aerial_tv/cuphy_pusch_tv.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pusch/pusch_defines.hpp"
#include "pusch/pusch_module_factories.hpp"
#include "pusch/pusch_pipeline.hpp"
#include "pusch_pipeline_runner.hpp"
#include "pusch_test_utils.hpp"
#include "ran_common.hpp"
#include "utils/cuda_stream.hpp"

namespace ran::pusch {

namespace {
namespace pipeline = framework::pipeline;
namespace utils = framework::utils;
} // namespace

PuschPipelineRunner::PuschPipelineRunner(
        const std::string &test_vector_path,
        const framework::pipeline::ExecutionMode execution_mode)
        : test_vector_{std::make_unique<ran::aerial_tv::CuphyPuschTestVector>(
                  test_vector_path.c_str())} {

    // Load test vector
    phy_params_ = ran::aerial_tv::to_phy_params(*test_vector_);

    // Create module factory
    module_factory_ = std::make_unique<PuschModuleFactory>();

    // Create pipeline spec
    const auto spec = create_pusch_pipeline_spec("benchmark_pipeline", phy_params_, execution_mode);

    // Create pipeline
    pipeline_ = std::make_unique<PuschPipeline>(
            "pusch_pipeline",
            gsl_lite::not_null<pipeline::IModuleFactory *>(module_factory_.get()),
            spec);

    // example-begin pipeline-setup-1
    // Setup pipeline
    pipeline_->setup();
    // example-end pipeline-setup-1

    // Reserve space for I/O
    external_inputs_.reserve(NUM_EXTERNAL_INPUTS);
    external_outputs_.resize(NUM_EXTERNAL_OUTPUTS);
}

void PuschPipelineRunner::configure(const utils::CudaStream &stream) {
    // Prepare inputs (first time only)
    if (input_device_ptrs_.empty()) {
        external_inputs_.clear();
        input_device_ptrs_ =
                prepare_pusch_inputs(external_inputs_, phy_params_, *test_vector_, stream.get());
    }

    // Setup dynamic params
    const auto pusch_outer_rx_params = ran::aerial_tv::to_pusch_outer_rx_params(*test_vector_);

    pipeline::DynamicParams params{};
    const PuschDynamicParams pusch_input{
            .inner_rx_params = {}, .outer_rx_params = pusch_outer_rx_params};
    params.module_specific_params = pusch_input;

    // example-begin configure-execute-1
    // Configure I/O with dynamic parameters
    pipeline_->configure_io(params, external_inputs_, external_outputs_, stream.get());

    // Warmup pipeline
    pipeline_->warmup(stream.get());
    // example-end configure-execute-1

    // Sync to ensure all async operations (TB params copy, descriptors, warmup) are complete
    if (!stream.synchronize()) {
        throw std::runtime_error("Stream synchronization failed after configure");
    }
}

void PuschPipelineRunner::warmup(cudaStream_t stream) { pipeline_->warmup(stream); }

void PuschPipelineRunner::execute_once(const utils::CudaStream &stream) {
    // example-begin execute-pipeline-1
    // Execute based on pipeline mode
    if (pipeline_->get_execution_mode() == pipeline::ExecutionMode::Stream) {
        pipeline_->execute_stream(stream.get());
    } else {
        pipeline_->execute_graph(stream.get());
    }
    // example-end execute-pipeline-1
}

pipeline::ExecutionMode PuschPipelineRunner::get_execution_mode() const {
    return pipeline_->get_execution_mode();
}

std::size_t PuschPipelineRunner::get_num_external_outputs() const {
    return pipeline_->get_num_external_outputs();
}

} // namespace ran::pusch
