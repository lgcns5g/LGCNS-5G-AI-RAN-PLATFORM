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
 * @file trt_test_utils.cpp
 * @brief Implementation of TensorRT test utilities
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <sstream>

#include <NvInfer.h>

#include <cuda_runtime_api.h>

#include "trt_test_utils.hpp"

namespace ran::trt_utils {

Statistics compute_statistics(const std::vector<double> &samples) {
    Statistics stats{};

    if (samples.empty()) {
        return stats;
    }

    stats.count = samples.size();

    // Sort samples for percentile calculations
    auto sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    stats.min = sorted_samples.front();
    stats.max = sorted_samples.back();
    stats.median = sorted_samples.at(sorted_samples.size() / 2);

    static constexpr double P95_PERCENTILE = 0.95;
    const auto p95_idx = std::min(
            static_cast<std::size_t>(static_cast<double>(sorted_samples.size()) * P95_PERCENTILE),
            sorted_samples.size() - 1);
    stats.p95 = sorted_samples.at(p95_idx);

    // Calculate mean
    const auto size_double = static_cast<double>(sorted_samples.size());
    stats.mean = std::accumulate(sorted_samples.begin(), sorted_samples.end(), 0.0) / size_double;

    // Calculate standard deviation
    const double variance = std::accumulate(
                                    sorted_samples.begin(),
                                    sorted_samples.end(),
                                    0.0,
                                    [mean = stats.mean](const double acc, const double value) {
                                        return acc + (value - mean) * (value - mean);
                                    }) /
                            size_double;
    stats.stddev = std::sqrt(variance);

    return stats;
}

std::string dims_to_string(const nvinfer1::Dims &dims) {
    std::ostringstream oss;
    oss << "(";
    const std::span<const int64_t> dims_span(static_cast<const int64_t *>(dims.d), dims.nbDims);
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        oss << dims_span[i];
        if (i < dims.nbDims - 1) {
            oss << ", ";
        }
    }
    oss << ")";
    return oss.str();
}

bool dims_equal(const nvinfer1::Dims &a, const nvinfer1::Dims &b) {
    if (a.nbDims != b.nbDims) {
        return false;
    }
    const std::span<const int64_t> a_span(static_cast<const int64_t *>(a.d), a.nbDims);
    const std::span<const int64_t> b_span(static_cast<const int64_t *>(b.d), b.nbDims);
    for (int32_t i = 0; i < a.nbDims; ++i) {
        if (a_span[i] != b_span[i]) {
            return false;
        }
    }
    return true;
}

bool dims_compatible(const nvinfer1::Dims &expected, const nvinfer1::Dims &actual) {
    if (expected.nbDims != actual.nbDims) {
        return false;
    }
    const std::span<const int64_t> expected_span(
            static_cast<const int64_t *>(expected.d), expected.nbDims);
    const std::span<const int64_t> actual_span(
            static_cast<const int64_t *>(actual.d), actual.nbDims);
    for (int32_t i = 0; i < expected.nbDims; ++i) {
        // Dynamic dimension (-1) in expected matches any positive value in actual
        if (expected_span[i] == -1) {
            continue;
        }
        // Otherwise, dimensions must match exactly
        if (expected_span[i] != actual_span[i]) {
            return false;
        }
    }
    return true;
}

std::size_t compute_size_from_dims(const nvinfer1::Dims &dims) {
    if (dims.nbDims == 0) {
        return 1; // Scalar
    }
    const std::span<const int64_t> dims_span(static_cast<const int64_t *>(dims.d), dims.nbDims);
    std::size_t size = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        size *= static_cast<std::size_t>(dims_span[i]);
    }
    return size;
}

std::size_t get_element_size(const nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return sizeof(uint16_t);
    case nvinfer1::DataType::kINT8:
        return sizeof(int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(int32_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kFP8:
        return sizeof(uint8_t);
    case nvinfer1::DataType::kINT64:
        return sizeof(int64_t);
    case nvinfer1::DataType::kINT4: // Special case, packed format
    default:
        return 0;
    }
}

std::string get_trt_engine_path(StdioLogger &logger) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe) - Called once during initialization
    const char *env_path = std::getenv("RAN_TRT_ENGINE_PATH");
    if (env_path == nullptr) {
        const std::string error_msg =
                "RAN_TRT_ENGINE_PATH environment variable is not set. "
                "This variable should point to the directory containing TensorRT engine files.";
        logger.log(nvinfer1::ILogger::Severity::kERROR, error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
    return {env_path};
}

std::size_t TensorSpec::get_num_elements() const { return compute_size_from_dims(shape); }

std::size_t TensorSpec::get_bytes() const { return get_num_elements() * get_element_size(type); }

// CudaStream implementation
CudaStream::CudaStream() {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
}

CudaStream::~CudaStream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

CudaStream::CudaStream(CudaStream &&other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream &CudaStream::operator=(CudaStream &&other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() const {
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }
}

// TrtExecutor implementation
TrtExecutor::TrtExecutor(const ExecutionMode mode) : mode_(mode) {}

TrtExecutor::~TrtExecutor() {
    if (mode_ == ExecutionMode::Graph) {
        reset_graph();
    }
}

void TrtExecutor::prepare(
        nvinfer1::IExecutionContext *const context,
        cudaStream_t stream,
        const std::size_t warmup_launches) {
    if (context == nullptr) {
        throw std::runtime_error("TrtExecutor: context cannot be null");
    }

    context_ = context;

    if (mode_ == ExecutionMode::Graph) {
        // Graph mode: capture and instantiate
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            throw std::runtime_error("TrtExecutor: failed to synchronize stream before capture");
        }
        capture_graph(context, stream);

        // Perform warmup launches
        for (std::size_t i = 0; i < warmup_launches; ++i) {
            launch_graph(stream);
        }
        if (warmup_launches > 0) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                throw std::runtime_error("TrtExecutor: failed to synchronize after warmup");
            }
        }
    } else {
        // Stream mode: perform warmup launches
        for (std::size_t i = 0; i < warmup_launches; ++i) {
            if (!context_->enqueueV3(stream)) {
                throw std::runtime_error("TrtExecutor: warmup enqueueV3 failed");
            }
        }
        if (warmup_launches > 0) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                throw std::runtime_error("TrtExecutor: failed to synchronize after warmup");
            }
        }
    }
}

void TrtExecutor::execute(cudaStream_t stream) {
    if (!is_ready()) {
        throw std::runtime_error("TrtExecutor: must call prepare() first");
    }

    if (mode_ == ExecutionMode::Graph) {
        launch_graph(stream);
    } else {
        if (!context_->enqueueV3(stream)) {
            throw std::runtime_error("TrtExecutor: enqueueV3 failed");
        }
    }
}

bool TrtExecutor::is_ready() const {
    if (mode_ == ExecutionMode::Graph) {
        return context_ != nullptr && captured_ && instantiated_;
    }
    return context_ != nullptr;
}

void TrtExecutor::reset_graph() {
    if (exec_ != nullptr) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    captured_ = false;
    instantiated_ = false;
}

void TrtExecutor::capture_graph(nvinfer1::IExecutionContext *const context, cudaStream_t stream) {
    // Begin stream capture
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
        throw std::runtime_error("TrtExecutor: cudaStreamBeginCapture failed");
    }

    // Enqueue TensorRT inference
    if (!context->enqueueV3(stream)) {
        cudaStreamEndCapture(stream, &graph_); // Clean up capture
        throw std::runtime_error("TrtExecutor: enqueueV3 failed during capture");
    }

    // End capture
    if (cudaStreamEndCapture(stream, &graph_) != cudaSuccess) {
        throw std::runtime_error("TrtExecutor: cudaStreamEndCapture failed");
    }
    captured_ = true;

    // Instantiate graph
    if (cudaGraphInstantiate(&exec_, graph_, 0) != cudaSuccess) {
        captured_ = false;
        throw std::runtime_error("TrtExecutor: cudaGraphInstantiate failed");
    }
    instantiated_ = true;
}

void TrtExecutor::launch_graph(cudaStream_t stream) {
    if (!instantiated_) {
        throw std::runtime_error("TrtExecutor: graph not instantiated");
    }

    if (cudaGraphLaunch(exec_, stream) != cudaSuccess) {
        throw std::runtime_error("TrtExecutor: cudaGraphLaunch failed");
    }
}

// CudaBuffer implementation
CudaBuffer::CudaBuffer(const std::size_t size) : size_(size) {
    if (cudaMalloc(&ptr_, size) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory");
    }
}

CudaBuffer::~CudaBuffer() {
    if (ptr_ != nullptr) {
        cudaFree(ptr_);
    }
}

CudaBuffer::CudaBuffer(CudaBuffer &&other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

CudaBuffer &CudaBuffer::operator=(CudaBuffer &&other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void CudaBuffer::copy_from_host(const void *src, const std::optional<cudaStream_t> stream) {
    if (stream) {
        if (cudaMemcpyAsync(ptr_, src, size_, cudaMemcpyHostToDevice, *stream) != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device (async)");
        }
    } else {
        if (cudaMemcpy(ptr_, src, size_, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device");
        }
    }
}

void CudaBuffer::copy_to_host(void *dst, const std::optional<cudaStream_t> stream) {
    if (stream) {
        if (cudaMemcpyAsync(dst, ptr_, size_, cudaMemcpyDeviceToHost, *stream) != cudaSuccess) {
            throw std::runtime_error("Failed to copy data from device (async)");
        }
    } else {
        if (cudaMemcpy(dst, ptr_, size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Failed to copy data from device");
        }
    }
}

// TensorBinder implementation
TensorBinder &TensorBinder::bind(
        const std::string &tensor_name,
        void *buffer,
        const TensorSpec &expected_spec,
        const std::string &description) {
    bindings_.push_back({tensor_name, buffer, expected_spec, description});
    return *this;
}

bool TensorBinder::apply(
        nvinfer1::IExecutionContext *context,
        nvinfer1::ICudaEngine *engine,
        StdioLogger &logger) const {
    for (const auto &spec : bindings_) {
        // Check if tensor exists
        const int32_t num_tensors = engine->getNbIOTensors();
        bool found = false;
        for (int32_t i = 0; i < num_tensors; ++i) {
            if (std::string(engine->getIOTensorName(i)) == spec.tensor_name) {
                found = true;
                break;
            }
        }

        if (!found) {
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("Tensor '{}' not found in engine", spec.tensor_name).c_str());
            return false;
        }

        // Validate shape (engine shape may have dynamic dimensions -1)
        const auto engine_shape = engine->getTensorShape(spec.tensor_name.c_str());
        if (!dims_compatible(engine_shape, spec.expected_spec.shape)) {
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("Shape mismatch for '{}' ({})", spec.tensor_name, spec.description)
                            .c_str());
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("  Engine shape: {}", dims_to_string(engine_shape)).c_str());
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("  Tensor shape: {}", dims_to_string(spec.expected_spec.shape))
                            .c_str());
            return false;
        }

        // Validate type
        const auto actual_type = engine->getTensorDataType(spec.tensor_name.c_str());
        if (actual_type != spec.expected_spec.type) {
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("Type mismatch for '{}' ({})", spec.tensor_name, spec.description)
                            .c_str());
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("  Expected type: {}", static_cast<int>(spec.expected_spec.type))
                            .c_str());
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("  Actual type:   {}", static_cast<int>(actual_type)).c_str());
            return false;
        }

        // Bind tensor
        if (!context->setTensorAddress(spec.tensor_name.c_str(), spec.buffer)) {
            logger.log(
                    nvinfer1::ILogger::Severity::kERROR,
                    std::format("Failed to set tensor address for '{}'", spec.tensor_name).c_str());
            return false;
        }

        logger.log(
                nvinfer1::ILogger::Severity::kINFO,
                std::format(
                        "Bound '{}' to {} {}",
                        spec.tensor_name,
                        spec.description,
                        dims_to_string(engine_shape))
                        .c_str());
    }
    return true;
}

// StdioLogger implementation
void StdioLogger::log(const Severity severity, const char *msg) noexcept {
    // Filter out messages below minimum severity level
    if (severity > min_severity_) {
        return;
    }

    const char *severity_str{};
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
        severity_str = "INTERNAL_ERROR";
        break;
    case Severity::kERROR:
        severity_str = "ERROR";
        break;
    case Severity::kWARNING:
        severity_str = "WARNING";
        break;
    case Severity::kINFO:
        severity_str = "INFO";
        break;
    case Severity::kVERBOSE:
        severity_str = "VERBOSE";
        break;
    default:
        severity_str = "UNKNOWN";
        break;
    }
    std::cout << std::format("[TRT][{}] {}\n", severity_str, msg);
}

// TrtEngine implementation
TrtEngine::TrtEngine(const std::string &engine_filename, StdioLogger &logger) {
    // Construct full path from environment variable
    const std::string engine_dir = get_trt_engine_path(logger);
    const std::string full_path = engine_dir + "/" + engine_filename;

    logger.log(
            nvinfer1::ILogger::Severity::kINFO,
            std::format("Loading TensorRT engine: {}", full_path).c_str());

    // Load engine from file
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        const std::string error_msg = std::format("Failed to open engine file: {}", full_path);
        logger.log(nvinfer1::ILogger::Severity::kERROR, error_msg.c_str());
        throw std::runtime_error(error_msg);
    }

    file.seekg(0, std::ios::end);
    const std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), static_cast<std::streamsize>(size));
    file.close();

    // Print TensorRT version information
    logger.log(
            nvinfer1::ILogger::Severity::kINFO,
            std::format(
                    "TensorRT version: {}.{}.{}.{}",
                    NV_TENSORRT_MAJOR,
                    NV_TENSORRT_MINOR,
                    NV_TENSORRT_PATCH,
                    NV_TENSORRT_BUILD)
                    .c_str());

    // Create runtime and deserialize engine
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_) {
        const std::string error_msg = "Failed to create TensorRT runtime";
        logger.log(nvinfer1::ILogger::Severity::kERROR, error_msg.c_str());
        throw std::runtime_error(error_msg);
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        const std::string error_msg = "Failed to deserialize TensorRT engine";
        logger.log(nvinfer1::ILogger::Severity::kERROR, error_msg.c_str());
        throw std::runtime_error(error_msg);
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        const std::string error_msg = "Failed to create execution context";
        logger.log(nvinfer1::ILogger::Severity::kERROR, error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
}

void TrtEngine::print_engine_info(StdioLogger &logger) const {
    const int32_t num_bindings = engine_->getNbIOTensors();
    logger.log(
            nvinfer1::ILogger::Severity::kINFO,
            std::format("Engine has {} I/O tensors:", num_bindings).c_str());

    for (int32_t i = 0; i < num_bindings; ++i) {
        const char *tensor_name = engine_->getIOTensorName(i);
        const auto mode = engine_->getTensorIOMode(tensor_name);
        const auto shape = engine_->getTensorShape(tensor_name);
        const auto dtype = engine_->getTensorDataType(tensor_name);

        const std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
        logger.log(
                nvinfer1::ILogger::Severity::kINFO,
                std::format(
                        "  [{}] {} - {} - Shape: {} - Type: {}",
                        i,
                        tensor_name,
                        mode_str,
                        dims_to_string(shape),
                        static_cast<int>(dtype))
                        .c_str());
    }
}

bool set_cuda_device(const int device_id) {
    int device_count{};
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << std::format("CUDA device not available: {}\n", cudaGetErrorString(error));
        return false;
    }
    if (device_count == 0) {
        std::cerr << "No CUDA devices found\n";
        return false;
    }
    if (device_id >= device_count) {
        std::cerr << std::format("Device ID {} out of range (0-{})\n", device_id, device_count - 1);
        return false;
    }
    if (cudaSetDevice(device_id) != cudaSuccess) {
        std::cerr << std::format("Failed to set CUDA device {}\n", device_id);
        return false;
    }
    return true;
}

bool engine_exists(const std::string &engine_filename) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe) - Called once during initialization
    const char *env_path = std::getenv("RAN_TRT_ENGINE_PATH");
    if (env_path == nullptr) {
        return false;
    }
    const std::filesystem::path engine_path = std::filesystem::path(env_path) / engine_filename;
    return std::filesystem::exists(engine_path);
}

} // namespace ran::trt_utils
