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
 * @file cuda_stream.hpp
 * @brief RAII wrapper for CUDA stream management
 */

#ifndef FRAMEWORK_CORE_CUDA_STREAM_HPP
#define FRAMEWORK_CORE_CUDA_STREAM_HPP

#include <cuda_runtime.h>

namespace framework::utils {

/**
 * RAII wrapper for CUDA stream management
 *
 * This class provides automatic lifetime management for cudaStream_t handles.
 * The stream is created with cudaStreamNonBlocking flag and automatically
 * synchronized and destroyed when the object goes out of scope.
 *
 * Example usage:
 * @code
 * {
 *     CudaStream stream;
 *     kernel<<<blocks, threads, 0, stream.get()>>>();
 *     stream.synchronize();  // Optional explicit sync
 * }  // Stream automatically synchronized and destroyed here
 * @endcode
 */
class CudaStream final {
public:
    /**
     * Create and initialize CUDA stream
     *
     * Creates a non-blocking CUDA stream using cudaStreamNonBlocking flag.
     *
     * @throws std::runtime_error if CUDA stream creation fails
     */
    CudaStream();

    /**
     * Synchronize and destroy CUDA stream
     *
     * Automatically synchronizes the stream before destroying it to ensure
     * all queued operations complete. Errors during cleanup are logged but
     * do not throw exceptions (destructor noexcept).
     */
    ~CudaStream();

    // Non-copyable (manages CUDA resource)
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    /**
     * Move constructor - transfer ownership of CUDA stream
     *
     * Transfers ownership of the CUDA stream from another CudaStream object.
     * The source object is left in a valid but empty state (nullptr stream).
     *
     * @param[in] other Source CudaStream to move from
     */
    CudaStream(CudaStream &&other) noexcept;

    /**
     * Move assignment operator - transfer ownership of CUDA stream
     *
     * Synchronizes and destroys the current stream (if any), then transfers
     * ownership of the CUDA stream from another CudaStream object. The source
     * object is left in a valid but empty state (nullptr stream).
     *
     * @param[in] other Source CudaStream to move from
     * @return Reference to this object
     */
    CudaStream &operator=(CudaStream &&other) noexcept;

    /**
     * Get the underlying CUDA stream handle
     *
     * @return CUDA stream handle for use with CUDA APIs
     */
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    /**
     * Synchronize the CUDA stream
     *
     * Blocks the calling CPU thread until all previously queued operations
     * on this stream have completed.
     *
     * @return true if synchronization succeeded, false on error (error is logged)
     */
    [[nodiscard]] bool synchronize() const noexcept;

private:
    cudaStream_t stream_{nullptr}; //!< CUDA stream handle
};

} // namespace framework::utils

#endif // FRAMEWORK_CORE_CUDA_STREAM_HPP
