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

#ifndef FRAMEWORK_CORE_PIPELINE_FACTORY_HPP
#define FRAMEWORK_CORE_PIPELINE_FACTORY_HPP

#include <any>
#include <concepts>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pipeline/imodule_factory.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/ipipeline_factory.hpp"
#include "pipeline/types.hpp"
#include "utils/string_hash.hpp"

namespace framework::pipeline {

/**
 * Concept defining requirements for pipeline creator callables.
 *
 * A pipeline creator must be invocable with:
 * - IModuleFactory& (reference to module factory)
 * - const std::string& (pipeline instance ID)
 * - const std::any& (type-erased parameters)
 *
 * And must return a std::unique_ptr<IPipeline>.
 *
 * This concept enables:
 * - Better compile-time error messages
 * - Acceptance of any callable type (lambdas, functors, function pointers)
 * - Explicit documentation of requirements
 *
 * @tparam F Callable type to check
 */
template <typename F>
concept PipelineCreatorCallable =
        std::invocable<F, IModuleFactory &, const std::string &, const std::any &> &&
        std::same_as<
                std::invoke_result_t<F, IModuleFactory &, const std::string &, const std::any &>,
                std::unique_ptr<IPipeline>>;

/**
 * Concept requiring both invocability and move constructibility.
 *
 * This ensures the callable can be stored in containers and moved.
 * Used for pipeline creators that will be registered in the factory.
 *
 * @tparam F Callable type to check
 */
template <typename F>
concept StorablePipelineCreator = PipelineCreatorCallable<F> && std::move_constructible<F>;

/**
 * @class PipelineFactory
 * @brief Concrete implementation of IPipelineFactory with runtime registration
 *
 * This factory uses a registry pattern allowing pipeline types to be registered
 * at runtime. Each pipeline creator receives a reference to the module factory
 * for constructing pipeline modules.
 *
 * Example usage:
 * @code
 * ModuleFactory module_factory;
 * // ... register module types ...
 *
 * PipelineFactory pipeline_factory(module_factory);
 * pipeline_factory.register_pipeline_type("skeleton",
 *     [](IModuleFactory& mf, const std::string& id, const std::any& params) {
 *         return std::make_unique<SkeletonPipeline>(mf,
 * std::any_cast<PipelineSpec>(params));
 *     });
 *
 * auto pipeline = pipeline_factory.create_pipeline("skeleton", "pipeline_0",
 * spec);
 * @endcode
 */
class PipelineFactory final : public IPipelineFactory {
public:
    /**
     * Pipeline creator function signature.
     * Takes module factory reference, instance ID, and parameters.
     * Returns unique pointer to pipeline.
     */
    using PipelineCreator = std::function<std::unique_ptr<IPipeline>(
            IModuleFactory &, const std::string &, const std::any &)>;

    /**
     * Constructor.
     *
     * @param[in] module_factory Reference to module factory for creating pipeline
     * modules
     */
    explicit PipelineFactory(IModuleFactory &module_factory);

    /**
     * Destructor.
     */
    ~PipelineFactory() override = default;

    // Non-copyable, non-movable (due to reference member)
    PipelineFactory(const PipelineFactory &) = delete;
    PipelineFactory &operator=(const PipelineFactory &) = delete;
    PipelineFactory(PipelineFactory &&) = delete;
    PipelineFactory &operator=(PipelineFactory &&) = delete;

    /**
     * Register a pipeline type with its creator function (concept-constrained).
     *
     * Accepts any callable (lambda, function pointer, functor, std::function)
     * that satisfies the StorablePipelineCreator concept.
     *
     * This templated overload enables:
     * - Better compile-time error messages
     * - Zero-overhead for stateless lambdas (avoids std::function wrapper)
     * - Type safety enforced at compile time
     *
     * @tparam Creator Type of the callable (deduced automatically)
     * @param[in] pipeline_type Type identifier for the pipeline
     * @param[in] creator Callable that creates instances of this pipeline type
     * @throws std::invalid_argument if pipeline_type is already registered
     */
    template <StorablePipelineCreator Creator>
    void register_pipeline_type(std::string_view pipeline_type, Creator &&creator);

    /**
     * Register a pipeline type with its creator function (std::function
     * overload).
     *
     * This overload accepts std::function directly for explicit usage.
     * Prefer using the templated overload for better performance with lambdas.
     *
     * @param[in] pipeline_type Type identifier for the pipeline
     * @param[in] creator Function that creates instances of this pipeline type
     * @throws std::invalid_argument if pipeline_type is already registered
     */
    void register_pipeline_type(std::string_view pipeline_type, PipelineCreator creator);

    /**
     * Create a pipeline from a specification.
     *
     * @param[in] pipeline_type The type of pipeline to create
     * @param[in] pipeline_id Unique identifier for this pipeline instance
     * @param[in] spec Complete pipeline specification (usually PipelineSpec)
     * @return Unique pointer to the created pipeline
     * @throws std::invalid_argument if pipeline_type is not supported
     * @throws std::runtime_error if pipeline creation fails
     */
    [[nodiscard]] std::unique_ptr<IPipeline> create_pipeline(
            std::string_view pipeline_type,
            const std::string &pipeline_id,
            const PipelineSpec &spec) override;

    /**
     * Check if a pipeline type is supported by this factory.
     *
     * @param[in] pipeline_type The type of pipeline to check
     * @return true if the pipeline type is supported, false otherwise
     */
    [[nodiscard]] bool is_pipeline_type_supported(std::string_view pipeline_type) const override;

    /**
     * Get all supported pipeline types.
     *
     * @return Vector of supported pipeline type identifiers
     */
    [[nodiscard]] std::vector<std::string> get_supported_pipeline_types() const override;

private:
    // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
    IModuleFactory &module_factory_; //!< Reference to module factory
    // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)
    std::unordered_map<
            std::string,
            PipelineCreator,
            utils::TransparentStringHash,
            std::equal_to<>>
            pipeline_creators_; //!< Registry of pipeline creators by type
};

// Template method implementation (must be in header)
template <StorablePipelineCreator Creator>
void PipelineFactory::register_pipeline_type(std::string_view pipeline_type, Creator &&creator) {
    // Forward to the non-template version by wrapping in std::function
    // The concept ensures this conversion is valid and will compile
    register_pipeline_type(pipeline_type, PipelineCreator{std::forward<Creator>(creator)});
}

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_PIPELINE_FACTORY_HPP
