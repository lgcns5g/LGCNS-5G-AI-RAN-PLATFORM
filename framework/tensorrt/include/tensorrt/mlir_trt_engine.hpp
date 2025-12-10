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

#ifndef FRAMEWORK_MLIR_TRT_ENGINE_HPP
#define FRAMEWORK_MLIR_TRT_ENGINE_HPP

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <driver_types.h>

#include "tensorrt/trt_engine_interface.hpp"
#include "tensorrt/trt_engine_interfaces.hpp"
#include "tensorrt/trt_engine_logger.hpp"
#include "tensorrt/trt_engine_params.hpp"
#include "utils/errors.hpp"

namespace framework::tensorrt {

/**
 * @brief Simplified TensorRT engine that mimics MLIR-TensorRT runtime patterns
 *
 * @details This class provides a streamlined TensorRT engine implementation
 * that:
 * - Eliminates batch size management (users handle batching externally)
 * - Removes internal buffer allocation (users provide pre-allocated CUDA
 * buffers)
 * - Uses constructor-based initialization (no separate init() phase)
 * - Accepts tensor dimensions and strides directly in MLIRTensorParams
 * - Supports user-provided tensor names from TrtParams
 *
 * The engine uses tensor metadata (dims/strides) provided by users in
 * MLIRTensorParams and directly interfaces with TensorRT APIs.
 */
class MLIRTrtEngine final {
public:
    /**
     * @brief Construct MLIRTrtEngine with full initialization
     *
     * @details All initialization is performed in the constructor, eliminating
     * the need for a separate init() method. The TensorRT runtime must be
     * pre-initialized and provided by the caller. Tensor shapes (dims/strides)
     * must be provided in the MLIRTensorParams. If strides are not provided
     * (last stride == 0), row-major strides are automatically computed.
     *
     * @param[in] input_tensor_prms Input tensor parameters (name, data_type,
     * rank, dims, optional strides)
     * @param[in] output_tensor_prms Output tensor parameters (name, data_type,
     * rank, dims, optional strides)
     * @param[in] tensorrt_runtime Pre-initialized TensorRT runtime (required)
     * @param[in] pre_post_trt_eng_enqueue Optional pre/post enqueue operations
     * (e.g., CUDA graph capture)
     *
     * @throws std::invalid_argument if tensorrt_runtime is nullptr, or if rank
     * is invalid (> 8)
     * @throws std::runtime_error on initialization failure
     */
    MLIRTrtEngine(
            std::vector<MLIRTensorParams> input_tensor_prms,
            std::vector<MLIRTensorParams> output_tensor_prms,
            std::unique_ptr<ITrtEngine> tensorrt_runtime,
            std::unique_ptr<IPrePostTrtEngEnqueue> pre_post_trt_eng_enqueue = nullptr);

    ~MLIRTrtEngine() = default;

    // Delete all special member functions
    MLIRTrtEngine(const MLIRTrtEngine &engine) = delete;
    MLIRTrtEngine &operator=(const MLIRTrtEngine &engine) = delete;
    MLIRTrtEngine(MLIRTrtEngine &&engine) = delete;
    MLIRTrtEngine &operator=(MLIRTrtEngine &&engine) = delete;

    /**
     * @brief Perform warmup inference to allocate TensorRT resources
     *
     * @details Runs the TensorRT engine once to ensure all internal resources
     * are allocated and avoid first-run latency. Extracts tensor shapes from
     * buffer descriptors and executes a single inference pass.
     *
     * @note Requires setup() to be called first to provide buffer pointers
     * for metadata extraction.
     *
     * @param[in] cu_stream CUDA stream for warmup operations
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc warmup(cudaStream_t cu_stream);

    /**
     * @brief Setup input and output buffer addresses
     *
     * @details Caches the provided buffer pointers for use during inference.
     * Buffers are direct pointers to CUDA memory (data pointers), not descriptor
     * pointers. Buffers must remain valid for the lifetime of inference
     * operations. No batch size parameter is needed as batching is handled
     * externally.
     *
     * @param[in] input_buffers Vector of input data buffer pointers (must match
     * input_tensor_prms size)
     * @param[in] output_buffers Vector of output data buffer pointers (must match
     * output_tensor_prms size)
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc
    setup(const std::vector<void *> &input_buffers, const std::vector<void *> &output_buffers);

    /**
     * @brief Execute inference on the configured tensors
     *
     * @details Performs the complete inference pipeline:
     * 1. Set tensor addresses in TensorRT using user-provided buffer pointers
     * 2. Set input shapes in TensorRT using dims from MLIRTensorParams
     * 3. Execute pre-enqueue operations (e.g., CUDA graph capture start)
     * 4. Run TensorRT inference
     * 5. Execute post-enqueue operations (e.g., CUDA graph capture end)
     *
     * @param[in] cu_stream CUDA stream for execution
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc run(cudaStream_t cu_stream) const;

private:
    /**
     * @brief Configure an input tensor
     * @param[in] tensor_index Index of the input tensor to process
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc configure_input_tensor(std::size_t tensor_index) const;

    /**
     * @brief Configure an output tensor
     * @param[in] tensor_index Index of the output tensor to process
     * @return utils::NvErrc::Success on success, error code on failure
     */
    [[nodiscard]]
    utils::NvErrc configure_output_tensor(std::size_t tensor_index) const;

    // Core tensor parameters (includes rank, dims, strides)
    std::vector<MLIRTensorParams> input_tensor_prms_;
    std::vector<MLIRTensorParams> output_tensor_prms_;

    // User-provided buffers (direct data pointers, no internal allocation)
    std::vector<void *> input_buffers_;
    std::vector<void *> output_buffers_;

    // Required components
    TrtLogger logger_;
    std::unique_ptr<IPrePostTrtEngEnqueue> pre_post_trt_eng_enqueue_;
    std::unique_ptr<ITrtEngine> tensorrt_runtime_;
};

} // namespace framework::tensorrt

#endif // FRAMEWORK_MLIR_TRT_ENGINE_HPP
