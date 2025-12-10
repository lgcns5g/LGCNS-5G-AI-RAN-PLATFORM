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

#ifndef FRAMEWORK_NET_LOG_HPP
#define FRAMEWORK_NET_LOG_HPP

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <quill/LogMacros.h>

#include "log/components.hpp"
#include "log/rt_log_macros.hpp"
#include "net/net_export.hpp"

namespace framework::net {

/**
 * @brief Declare logging components for network subsystem
 */
DECLARE_LOG_COMPONENT(Net, NetGeneral, NetGpu, NetDpdk, NetDoca);

/**
 * Log error message and throw exception
 *
 * @tparam ExceptionType Exception type to throw (defaults to std::runtime_error)
 * @tparam Args Variadic template arguments for format string
 * @param[in] component Net component for error logging
 * @param[in] format_string Format string for error message
 * @param[in] args Format string arguments
 * @throws ExceptionType with formatted error message
 */
template <typename ExceptionType = std::runtime_error, typename... Args>
[[noreturn]] void
log_and_throw(Net component, std::format_string<Args...> format_string, Args &&...args) {
    const std::string error_msg = std::format(format_string, std::forward<Args>(args)...);
    RT_LOGC_ERROR(component, "{}", error_msg);
    throw ExceptionType(error_msg);
}

} // namespace framework::net

#endif // FRAMEWORK_NET_LOG_HPP
