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

#ifndef FRAMEWORK_MEMCPY_HELPER_HPP
#define FRAMEWORK_MEMCPY_HELPER_HPP

#include <driver_types.h>

namespace framework::memory {

// Forward declarations
struct DeviceAlloc;
struct PinnedAlloc;

/**
 * Helper template struct to determine the appropriate CUDA memory copy kind
 * based on allocator types
 *
 * This template provides a type-safe way to determine the correct
 * cudaMemcpyKind enum value based on the source and destination allocator
 * types. It is specialized for different combinations of DeviceAlloc and
 * PinnedAlloc allocators.
 *
 * @tparam TDstAlloc Destination allocator type (DeviceAlloc or PinnedAlloc)
 * @tparam TSrcAlloc Source allocator type (DeviceAlloc or PinnedAlloc)
 */
template <class TDstAlloc, class TSrcAlloc> struct MemcpyHelper;

/**
 * Specialization for device-to-device memory copy operations
 *
 * This specialization handles memory copy operations between two device memory
 * locations. It provides the appropriate cudaMemcpyKind value for
 * device-to-device transfers.
 */
template <> struct MemcpyHelper<DeviceAlloc, DeviceAlloc> final {
    static constexpr cudaMemcpyKind KIND = cudaMemcpyDeviceToDevice; //!< Memory copy kind for
                                                                     //!< device-to-device transfers
};

/**
 * Specialization for device-to-host memory copy operations
 *
 * This specialization handles memory copy operations from device memory to
 * pinned host memory. It provides the appropriate cudaMemcpyKind value for
 * device-to-host transfers.
 */
template <> struct MemcpyHelper<PinnedAlloc, DeviceAlloc> final {
    static constexpr cudaMemcpyKind KIND =
            cudaMemcpyDeviceToHost; //!< Memory copy kind for device-to-host transfers
};

/**
 * Specialization for host-to-device memory copy operations
 *
 * This specialization handles memory copy operations from pinned host memory to
 * device memory. It provides the appropriate cudaMemcpyKind value for
 * host-to-device transfers.
 */
template <> struct MemcpyHelper<DeviceAlloc, PinnedAlloc> final {
    static constexpr cudaMemcpyKind KIND =
            cudaMemcpyHostToDevice; //!< Memory copy kind for host-to-device transfers
};

/**
 * Specialization for host-to-host memory copy operations
 *
 * This specialization handles memory copy operations between two pinned host
 * memory locations. It provides the appropriate cudaMemcpyKind value for
 * host-to-host transfers.
 */
template <> struct MemcpyHelper<PinnedAlloc, PinnedAlloc> final {
    static constexpr cudaMemcpyKind KIND =
            cudaMemcpyHostToHost; //!< Memory copy kind for host-to-host transfers
};

} // namespace framework::memory

#endif // FRAMEWORK_MEMCPY_HELPER_HPP
