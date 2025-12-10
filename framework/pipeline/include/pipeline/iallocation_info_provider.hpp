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

#ifndef FRAMEWORK_CORE_IALLOCATION_INFO_PROVIDER_HPP
#define FRAMEWORK_CORE_IALLOCATION_INFO_PROVIDER_HPP

#include "pipeline/types.hpp"

namespace framework::pipeline {

/**
 * @class IAllocationInfoProvider
 * @brief Interface for providing memory allocation requirements.
 *
 * This interface allows modules to specify their memory allocation needs
 * using the detailed ModuleMemoryRequirements structure that aligns with
 * cuBB's memory management patterns.
 */
class IAllocationInfoProvider {
public:
    /**
     * Virtual destructor.
     */
    virtual ~IAllocationInfoProvider() = default;

    IAllocationInfoProvider(const IAllocationInfoProvider &) = delete;
    IAllocationInfoProvider(IAllocationInfoProvider &&) = delete;
    IAllocationInfoProvider &operator=(const IAllocationInfoProvider &) = delete;
    IAllocationInfoProvider &operator=(IAllocationInfoProvider &&) = delete;

    /**
     * Get the memory requirements for a module.
     *
     * @return The module's memory requirements including static/dynamic
     * descriptors and device tensors
     */
    [[nodiscard]] virtual ModuleMemoryRequirements get_requirements() const = 0;

protected:
    /**
     * Default constructor.
     */
    IAllocationInfoProvider() = default;
};

} // namespace framework::pipeline

#endif // FRAMEWORK_CORE_IALLOCATION_INFO_PROVIDER_HPP
