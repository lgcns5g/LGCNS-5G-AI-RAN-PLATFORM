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

#ifndef FRAMEWORK_CORE_IPIPELINE_FACTORY_HPP
#define FRAMEWORK_CORE_IPIPELINE_FACTORY_HPP

#include <any>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"

namespace framework::pipeline {

/**
 * @class IPipelineFactory
 * @brief Interface for creating pipelines dynamically.
 *
 * This interface defines the contract for factories that can create different
 * types of pipelines based on type identifiers and specifications. The factory
 * pattern enables configuration-driven pipeline construction and supports
 * extensibility through runtime registration of pipeline types.
 */
class IPipelineFactory {
public:
    /**
     * Default constructor.
     */
    IPipelineFactory() = default;

    /**
     * Virtual destructor.
     */
    virtual ~IPipelineFactory() = default;

    /**
     * Move constructor.
     */
    IPipelineFactory(IPipelineFactory &&) = default;

    /**
     * Move assignment operator.
     * @return Reference to this object
     */
    IPipelineFactory &operator=(IPipelineFactory &&) = default;

    IPipelineFactory(const IPipelineFactory &) = delete;
    IPipelineFactory &operator=(const IPipelineFactory &) = delete;

    /**
     * Create a pipeline from a specification.
     *
     * This method constructs a complete pipeline based on the provided
     * specification, which includes module definitions, connections, and
     * external I/O configuration.
     *
     * @param[in] pipeline_type The type of pipeline to create (e.g., "skeleton",
     * "multi_module")
     * @param[in] pipeline_id Unique identifier for this pipeline instance
     * @param[in] spec Complete pipeline specification including modules and
     * connections
     * @return Unique pointer to the created pipeline
     * @throws std::invalid_argument if pipeline_type is not supported
     * @throws std::runtime_error if pipeline creation fails
     */
    [[nodiscard]] virtual std::unique_ptr<IPipeline> create_pipeline(
            std::string_view pipeline_type,
            const std::string &pipeline_id,
            const PipelineSpec &spec) = 0;

    /**
     * Check if a pipeline type is supported by this factory.
     *
     * @param[in] pipeline_type The type of pipeline to check
     * @return true if the pipeline type is supported, false otherwise
     */
    [[nodiscard]] virtual bool is_pipeline_type_supported(std::string_view pipeline_type) const = 0;

    /**
     * Get all supported pipeline types.
     *
     * @return Vector of supported pipeline type identifiers
     */
    [[nodiscard]] virtual std::vector<std::string> get_supported_pipeline_types() const = 0;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IPIPELINE_FACTORY_HPP
