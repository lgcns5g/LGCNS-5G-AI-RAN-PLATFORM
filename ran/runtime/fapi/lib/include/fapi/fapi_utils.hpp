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
 * @file fapi_utils.hpp
 * @brief Utility functions for FAPI applications
 *
 * Provides common configuration helpers for FAPI-based applications.
 */

#ifndef RAN_FAPI_UTILS_HPP
#define RAN_FAPI_UTILS_HPP

#include <string>
#include <string_view>

#include "fapi/fapi_export.hpp"

namespace ran::fapi {

/**
 * Create default NVIPC configuration YAML string
 *
 * Generates standard NVIPC configuration for PHY-MAC interface with customizable
 * prefix. The configuration includes default buffer sizes, memory pool settings,
 * and application configuration suitable for most FAPI applications.
 *
 * @param[in] prefix NVIPC shared memory prefix (e.g., "fapi_sample", "phy_ran_app")
 * @return YAML configuration string ready for FapiState initialization
 */
[[nodiscard]] FAPI_EXPORT std::string create_default_nvipc_config(std::string_view prefix);

} // namespace ran::fapi

#endif // RAN_FAPI_UTILS_HPP
