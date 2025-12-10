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

#ifndef FRAMEWORK_PIPELINES_SAMPLE_PIPELINE_FACTORY_HPP
#define FRAMEWORK_PIPELINES_SAMPLE_PIPELINE_FACTORY_HPP

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include "pipeline/imodule_factory.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/ipipeline_factory.hpp"
#include "pipeline/types.hpp"

namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;

/**
 * Factory for creating SamplePipeline instances
 *
 * Supports pipeline type: "sample"
 *
 * Creates a two-module pipeline (SampleModuleA + SampleModuleB) based on
 * PipelineSpec configuration. Uses injected IModuleFactory for module creation.
 *
 * Expected PipelineSpec structure:
 * - modules[0]: { moduleType: "sample_module_a", instanceId: "module_a",
 *                 initParams: SampleModuleA::StaticParams }
 * - modules[1]: { moduleType: "sample_module_b", instanceId: "module_b",
 *                 initParams: SampleModuleB::StaticParams }
 * - connections: [ { source: "module_a.output" -> target: "module_b.input" } ]
 * - external_inputs: ["input0", "input1"]
 * - external_outputs: ["output"]
 */
class SamplePipelineFactory final : public pipeline::IPipelineFactory {
public:
    /**
     * Constructor with module factory dependency injection
     *
     * @param[in] module_factory Module factory for creating pipeline modules
     *   (non-owning pointer, must outlive this factory)
     * @throws std::invalid_argument if module_factory is nullptr
     */
    explicit SamplePipelineFactory(gsl_lite::not_null<pipeline::IModuleFactory *> module_factory);

    /**
     * Destructor
     */
    ~SamplePipelineFactory() override = default;

    // Non-copyable, non-movable
    SamplePipelineFactory(const SamplePipelineFactory &) = delete;
    SamplePipelineFactory &operator=(const SamplePipelineFactory &) = delete;
    SamplePipelineFactory(SamplePipelineFactory &&) = delete;
    SamplePipelineFactory &operator=(SamplePipelineFactory &&) = delete;

    /**
     * Create a SamplePipeline instance from specification
     *
     * Parses the PipelineSpec, creates modules using the injected module factory,
     * and constructs a configured SamplePipeline.
     *
     * @param[in] pipeline_type The type of pipeline to create (must be
     * "sample")
     * @param[in] pipeline_id Unique identifier for this pipeline instance
     * @param[in] spec Complete pipeline specification including modules and
     * connections
     * @return Unique pointer to the created pipeline
     * @throws std::invalid_argument if pipeline_type is not "sample"
     * @throws std::runtime_error if pipeline creation fails (invalid spec,
     * module creation error)
     */
    [[nodiscard]] std::unique_ptr<pipeline::IPipeline> create_pipeline(
            std::string_view pipeline_type,
            const std::string &pipeline_id,
            const pipeline::PipelineSpec &spec) override;

    /**
     * Check if a pipeline type is supported
     *
     * @param[in] pipeline_type The type of pipeline to check
     * @return true if pipeline_type is "sample", false otherwise
     */
    [[nodiscard]] bool is_pipeline_type_supported(std::string_view pipeline_type) const override;

    /**
     * Get all supported pipeline types
     *
     * @return Vector containing "sample"
     */
    [[nodiscard]] std::vector<std::string> get_supported_pipeline_types() const override;

private:
    gsl_lite::not_null<pipeline::IModuleFactory *>
            module_factory_; //!< Module factory for creating modules (non-owning)
};

} // namespace framework::pipelines::samples

#endif // FRAMEWORK_PIPELINES_SAMPLE_PIPELINE_FACTORY_HPP
