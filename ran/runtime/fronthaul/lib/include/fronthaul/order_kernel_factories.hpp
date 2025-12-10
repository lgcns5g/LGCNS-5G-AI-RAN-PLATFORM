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

// Factory implementations for OrderKernelModule and OrderKernelPipeline

#ifndef RAN_FRONTHAUL_ORDER_KERNEL_FACTORIES_HPP
#define RAN_FRONTHAUL_ORDER_KERNEL_FACTORIES_HPP

#include <any>
#include <memory>
#include <string>
#include <string_view>

#include "fronthaul/order_kernel_module.hpp"
#include "fronthaul/order_kernel_pipeline.hpp"
#include "net/doca_types.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/ipipeline_factory.hpp"
#include "pipeline/types.hpp"

namespace ran::fronthaul {

/**
 * Factory for creating OrderKernelModule instances
 *
 * Creates modules of type "order_kernel_module" using the factory pattern.
 */
class OrderKernelModuleFactory final : public framework::pipeline::IModuleFactory {
public:
    OrderKernelModuleFactory() = default;
    ~OrderKernelModuleFactory() override = default;

    // Non-copyable, movable
    OrderKernelModuleFactory(const OrderKernelModuleFactory &) = delete;
    OrderKernelModuleFactory &operator=(const OrderKernelModuleFactory &) = delete;

    /** Move constructor */
    OrderKernelModuleFactory(OrderKernelModuleFactory &&) = default;

    /**
     * Move assignment operator
     *
     * @return Reference to this object
     */
    OrderKernelModuleFactory &operator=(OrderKernelModuleFactory &&) = default;

    /**
     * Create an OrderKernelModule instance
     *
     * @param[in] module_type Module type identifier (must be "order_kernel_module")
     * @param[in] instance_id Unique instance identifier
     * @param[in] static_params Static parameters (OrderKernelModule::StaticParams)
     * @return Unique pointer to created module
     * @throws std::invalid_argument if module_type is not supported
     * @throws std::bad_any_cast if static_params has wrong type
     */
    [[nodiscard]] std::unique_ptr<framework::pipeline::IModule> create_module(
            std::string_view module_type,
            const std::string &instance_id,
            const std::any &static_params) override;

    /**
     * Check if factory supports the given module type
     *
     * @param[in] module_type Module type to check
     * @return true if module_type is "order_kernel_module"
     */
    [[nodiscard]] bool supports_module_type(std::string_view module_type) const override;

    /**
     * Create an OrderKernelModule instance with specific return type
     *
     * Convenience method that returns the specific module type without requiring a cast.
     * This is a non-virtual wrapper around the virtual create_module() method.
     *
     * @param[in] instance_id Unique instance identifier
     * @param[in] static_params Static parameters (OrderKernelModule::StaticParams)
     * @return Unique pointer to created OrderKernelModule
     * @throws std::bad_any_cast if static_params has wrong type
     */
    [[nodiscard]]
    std::unique_ptr<ran::fronthaul::OrderKernelModule>
    create_order_kernel_module(const std::string &instance_id, const std::any &static_params);
};

/**
 * Factory for creating OrderKernelPipeline instances
 *
 * Creates pipelines of type "order_kernel_pipeline" using the factory pattern.
 * Each pipeline gets its own dedicated OrderKernelModuleFactory instance.
 */
class OrderKernelPipelineFactory final : public framework::pipeline::IPipelineFactory {
public:
    OrderKernelPipelineFactory() = default;
    ~OrderKernelPipelineFactory() override = default;

    // Non-copyable, movable
    OrderKernelPipelineFactory(const OrderKernelPipelineFactory &) = delete;
    OrderKernelPipelineFactory &operator=(const OrderKernelPipelineFactory &) = delete;

    /** Move constructor */
    OrderKernelPipelineFactory(OrderKernelPipelineFactory &&) = default;

    /**
     * Move assignment operator
     *
     * @return Reference to this object
     */
    OrderKernelPipelineFactory &operator=(OrderKernelPipelineFactory &&) = default;

    /**
     * Create an OrderKernelPipeline instance
     *
     * @param[in] pipeline_type Pipeline type identifier (must be "order_kernel_pipeline")
     * @param[in] pipeline_id Unique pipeline identifier
     * @param[in] spec Pipeline specification
     * @return Unique pointer to created pipeline
     * @throws std::invalid_argument if pipeline_type is not supported or spec is invalid
     */
    [[nodiscard]]
    std::unique_ptr<framework::pipeline::IPipeline> create_pipeline(
            std::string_view pipeline_type,
            const std::string &pipeline_id,
            const framework::pipeline::PipelineSpec &spec) override;

    /**
     * Check if factory supports the given pipeline type
     *
     * @param[in] pipeline_type Pipeline type to check
     * @return true if pipeline_type is "order_kernel_pipeline"
     */
    [[nodiscard]] bool is_pipeline_type_supported(std::string_view pipeline_type) const override;

    /**
     * Get list of supported pipeline types
     *
     * @return Vector containing "order_kernel_pipeline"
     */
    [[nodiscard]] std::vector<std::string> get_supported_pipeline_types() const override;

    /**
     * Set DOCA RX queue parameters for pipeline creation
     *
     * Must be called before create_pipeline() to provide infrastructure handles.
     *
     * @param[in] doca_params DOCA RX queue parameters (non-owning pointer, must outlive factory)
     */
    void set_doca_params(const framework::net::DocaRxQParams *doca_params) noexcept {
        doca_rxq_params_ = doca_params;
    }
    /**
     * Create an OrderKernelPipeline instance with specific return type
     *
     * Convenience method that returns the specific pipeline type without requiring a cast.
     * This is a non-virtual wrapper around the virtual create_pipeline() method.
     *
     * @param[in] pipeline_id Unique pipeline identifier
     * @param[in] spec Pipeline specification
     * @return Unique pointer to created OrderKernelPipeline
     * @throws std::runtime_error if DOCA params not set or pipeline creation fails
     */
    [[nodiscard]]
    std::unique_ptr<ran::fronthaul::OrderKernelPipeline> create_order_kernel_pipeline(
            const std::string &pipeline_id, const framework::pipeline::PipelineSpec &spec);

private:
    const framework::net::DocaRxQParams *doca_rxq_params_{
            nullptr}; //!< DOCA RX queue parameters (non-owning)
};

} // namespace ran::fronthaul

#endif // RAN_FRONTHAUL_ORDER_KERNEL_FACTORIES_HPP
