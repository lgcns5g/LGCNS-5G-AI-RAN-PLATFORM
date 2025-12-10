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

#ifndef FRAMEWORK_UNDERLYING_HPP
#define FRAMEWORK_UNDERLYING_HPP

#include <type_traits>

namespace framework::utils {

/**
 * Converts an enumeration to its underlying type
 *
 * This function provides the same functionality as std::to_underlying from
 * C++23 for earlier C++ standards. It safely converts an enumeration value to
 * its underlying integral type.
 *
 * @tparam Enum The enumeration type to convert
 * @param[in] e The enumeration value to convert
 * @return The enumeration value converted to its underlying type
 *
 * @note This function is equivalent to:
 *       static_cast<std::underlying_type_t<Enum>>(e)
 */
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(const Enum e) noexcept {
    static_assert(std::is_enum_v<Enum>, "to_underlying requires an enumeration type");
    return static_cast<std::underlying_type_t<Enum>>(e);
}

} // namespace framework::utils

#endif // FRAMEWORK_UNDERLYING_HPP
