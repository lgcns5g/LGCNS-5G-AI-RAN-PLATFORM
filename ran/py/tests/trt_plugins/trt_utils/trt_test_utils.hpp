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
 * @file trt_test_utils.hpp
 * @brief Utilities for TensorRT plugin testing
 *
 * Provides RAII wrappers, tensor management, and helper functions for testing
 * TensorRT plugins with CUDA.
 */

#ifndef RAN_TRT_UTILS_HPP
#define RAN_TRT_UTILS_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <NvInfer.h>
#include <driver_types.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ran::trt_utils {

// Forward declarations
class StdioLogger;

/**
 * Execution mode for TensorRT
 */
enum class ExecutionMode {
    Stream, //!< Execute using CUDA streams (default)
    Graph   //!< Execute using CUDA graphs (optimized for repeated execution)
};

/**
 * Convert ExecutionMode to string
 *
 * @param[in] mode Execution mode
 * @return String representation of the mode
 */
[[nodiscard]] inline const char *execution_mode_to_string(const ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::Stream:
        return "Stream";
    case ExecutionMode::Graph:
        return "Graph";
    default:
        return "Unknown";
    }
}

/**
 * Range specification for tensor slicing
 */
struct Range {
    int64_t start{}; //!< Start index (inclusive)
    int64_t end{};   //!< End index (exclusive)
};

/// Statistical metrics computed from a collection of samples
struct Statistics {
    double min{};        //!< Minimum value
    double max{};        //!< Maximum value
    double mean{};       //!< Arithmetic mean
    double median{};     //!< Median value (50th percentile)
    double stddev{};     //!< Standard deviation
    double p95{};        //!< 95th percentile
    std::size_t count{}; //!< Number of samples
};

/**
 * Compute statistical metrics from a collection of samples
 *
 * @param[in] samples Vector of sample values
 * @return Statistics structure containing computed metrics
 */
[[nodiscard]] Statistics compute_statistics(const std::vector<double> &samples);

/**
 * Convert Dims to string for error messages
 *
 * @param[in] dims Dimension object to convert
 * @return String representation of dimensions
 */
[[nodiscard]] std::string dims_to_string(const nvinfer1::Dims &dims);

/**
 * Compare two Dims for equality
 *
 * @param[in] a First dimension object
 * @param[in] b Second dimension object
 * @return true if dimensions are equal
 */
[[nodiscard]] bool dims_equal(const nvinfer1::Dims &a, const nvinfer1::Dims &b);

/**
 * Check if actual dims are compatible with expected dims
 *
 * Handles dynamic dimensions (-1) in comparison.
 *
 * @param[in] expected Expected dimensions
 * @param[in] actual Actual dimensions
 * @return true if dimensions are compatible
 */
[[nodiscard]] bool dims_compatible(const nvinfer1::Dims &expected, const nvinfer1::Dims &actual);

/**
 * Compute total number of elements from Dims
 *
 * @param[in] dims Dimension object
 * @return Total number of elements
 */
[[nodiscard]] std::size_t compute_size_from_dims(const nvinfer1::Dims &dims);

/**
 * Get element size in bytes for a DataType
 *
 * @param[in] type TensorRT data type
 * @return Size of one element in bytes
 */
[[nodiscard]] std::size_t get_element_size(const nvinfer1::DataType type);

/**
 * Get TensorRT engine directory path from environment variable
 *
 * @param[in] logger Logger for diagnostic messages
 * @return Path to TensorRT engine directory
 * @throws std::runtime_error if RAN_TRT_ENGINE_PATH environment variable is not set
 */
[[nodiscard]] std::string get_trt_engine_path(StdioLogger &logger);

/**
 * Tensor specification bundling shape and type
 */
struct TensorSpec {
    nvinfer1::Dims shape;    //!< Tensor dimensions
    nvinfer1::DataType type; //!< Element data type

    /**
     * Get total number of elements in tensor
     *
     * @return Total number of elements
     */
    [[nodiscard]] std::size_t get_num_elements() const;

    /**
     * Get total size in bytes
     *
     * @return Total size in bytes
     */
    [[nodiscard]] std::size_t get_bytes() const;
};

/**
 * Map C++ types to TensorRT DataType at compile time
 *
 * @tparam T C++ type to map
 */
template <typename T> struct TypeToDataType;

/// TypeToDataType specialization for float
template <> struct TypeToDataType<float> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kFLOAT; //!< TensorRT data type
};

/// TypeToDataType specialization for __half
template <> struct TypeToDataType<__half> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kHALF; //!< TensorRT data type
};

/// TypeToDataType specialization for __nv_bfloat16
template <> struct TypeToDataType<__nv_bfloat16> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kBF16; //!< TensorRT data type
};

/// TypeToDataType specialization for int32_t
template <> struct TypeToDataType<int32_t> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kINT32; //!< TensorRT data type
};

/// TypeToDataType specialization for int8_t
template <> struct TypeToDataType<int8_t> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kINT8; //!< TensorRT data type
};

/// TypeToDataType specialization for uint8_t
template <> struct TypeToDataType<uint8_t> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kUINT8; //!< TensorRT data type
};

/// TypeToDataType specialization for int64_t
template <> struct TypeToDataType<int64_t> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kINT64; //!< TensorRT data type
};

/// TypeToDataType specialization for bool
template <> struct TypeToDataType<bool> {
    static constexpr nvinfer1::DataType VALUE = nvinfer1::DataType::kBOOL; //!< TensorRT data type
};

/// Type trait to detect CUDA floating point types
template <typename T>
inline constexpr bool IS_CUDA_FLOAT_V =
        std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;

/**
 * Convert float to __half
 *
 * @param[in] val Float value to convert
 * @return Half-precision value
 */
[[nodiscard]] inline __half float_to_half(const float val) { return __float2half(val); }

/**
 * Convert __half to float
 *
 * @param[in] val Half-precision value to convert
 * @return Float value
 */
[[nodiscard]] inline float half_to_float(const __half val) { return __half2float(val); }

/**
 * Convert float to __nv_bfloat16
 *
 * @param[in] val Float value to convert
 * @return BFloat16 value
 */
[[nodiscard]] inline __nv_bfloat16 float_to_bfloat16(const float val) {
    return __float2bfloat16(val);
}

/**
 * Convert __nv_bfloat16 to float
 *
 * @param[in] val BFloat16 value to convert
 * @return Float value
 */
[[nodiscard]] inline float bfloat16_to_float(const __nv_bfloat16 val) {
    return __bfloat162float(val);
}

/**
 * Convert any supported type to float for comparison
 *
 * @tparam T Source type
 * @param[in] val Value to convert
 * @return Float representation of value
 */
template <typename T> [[nodiscard]] float to_float(const T &val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, __half>) {
        return half_to_float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return bfloat16_to_float(val);
    } else {
        return static_cast<float>(val);
    }
}

/**
 * Convert float to any supported type for initialization
 *
 * @tparam T Target type
 * @param[in] val Float value to convert
 * @return Converted value of type T
 */
template <typename T> [[nodiscard]] T from_float(const float val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, __half>) {
        return float_to_half(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return float_to_bfloat16(val);
    } else if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::round(val));
    } else {
        return static_cast<T>(val);
    }
}

/**
 * Get TensorRT DataType from C++ type
 *
 * @tparam T C++ type
 * @return Corresponding TensorRT DataType
 */
template <typename T> constexpr nvinfer1::DataType get_data_type() {
    return TypeToDataType<T>::VALUE;
}

/**
 * RAII wrapper for CUDA stream
 */
class CudaStream final {
private:
    cudaStream_t stream_{};

public:
    CudaStream();
    ~CudaStream();

    // Delete copy operations
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    /**
     * Move constructor
     *
     * @param[in,out] other Stream to move from
     */
    CudaStream(CudaStream &&other) noexcept;

    /**
     * Move assignment operator
     *
     * @param[in,out] other Stream to move from
     * @return Reference to this
     */
    CudaStream &operator=(CudaStream &&other) noexcept;

    /**
     * Get underlying CUDA stream handle
     *
     * @return CUDA stream handle
     */
    [[nodiscard]] cudaStream_t get() const { return stream_; }

    /**
     * Synchronize stream execution
     */
    void synchronize() const;
};

/**
 * Unified executor for TensorRT context supporting both stream and graph modes
 *
 * Supports two execution modes:
 * - Stream: Direct execution via enqueueV3() (lower setup overhead)
 * - Graph: CUDA graph capture and replay (lower repeated execution overhead)
 *
 * Typical usage:
 *
 * For tests (no warmup):
 * @code
 * TrtExecutor executor(mode);
 * executor.prepare(context, stream);
 * executor.execute(stream);
 * @endcode
 *
 * For benchmarks (with warmup):
 * @code
 * TrtExecutor executor(mode);
 * executor.prepare(context, stream, 1);  // 1 warmup launch
 * for (auto _ : state) {
 *     executor.execute(stream);
 * }
 * @endcode
 */
class TrtExecutor final {
private:
    ExecutionMode mode_;
    nvinfer1::IExecutionContext *context_{};

    // Graph mode state
    cudaGraph_t graph_{};
    cudaGraphExec_t exec_{};
    bool captured_{};
    bool instantiated_{};

public:
    /**
     * Construct executor with specified execution mode
     *
     * @param[in] mode Execution mode (Stream or Graph)
     */
    explicit TrtExecutor(ExecutionMode mode);

    ~TrtExecutor();

    // Delete copy and move operations
    TrtExecutor(const TrtExecutor &) = delete;
    TrtExecutor &operator=(const TrtExecutor &) = delete;
    TrtExecutor(TrtExecutor &&) = delete;
    TrtExecutor &operator=(TrtExecutor &&) = delete;

    /**
     * Prepare executor for execution with optional warmup
     *
     * Stream mode:
     * - Stores context pointer
     * - Performs warmup_launches via enqueueV3()
     *
     * Graph mode:
     * - Captures TensorRT operations into a CUDA graph
     * - Instantiates graph
     * - Performs warmup_launches of the graph
     *
     * @param[in] context TensorRT execution context
     * @param[in] stream CUDA stream for execution/capture
     * @param[in] warmup_launches Number of warmup executions (default: 0 for tests, use 1+ for
     * benchmarks)
     * @throws std::runtime_error if capture or execution fails
     */
    void
    prepare(nvinfer1::IExecutionContext *context,
            cudaStream_t stream,
            std::size_t warmup_launches = 0);

    /**
     * Execute TensorRT engine
     *
     * - Stream mode: Calls context->enqueueV3()
     * - Graph mode: Launches captured graph
     *
     * Must call prepare() first.
     *
     * @param[in] stream CUDA stream for execution
     * @throws std::runtime_error if not prepared or execution fails
     */
    void execute(cudaStream_t stream);

    /**
     * Get current execution mode
     *
     * @return Execution mode
     */
    [[nodiscard]] ExecutionMode mode() const { return mode_; }

    /**
     * Check if executor is ready for execution
     *
     * @return true if prepare() was called successfully
     */
    [[nodiscard]] bool is_ready() const;

private:
    /// Reset graph state (Graph mode only)
    void reset_graph();

    /// Capture TensorRT context into graph (Graph mode only)
    void capture_graph(nvinfer1::IExecutionContext *context, cudaStream_t stream);

    /// Launch captured graph (Graph mode only)
    void launch_graph(cudaStream_t stream);
};

/**
 * RAII wrapper for CUDA memory with optional stream support
 */
class CudaBuffer final {
private:
    void *ptr_{};
    std::size_t size_{};

public:
    /**
     * Allocate CUDA device memory
     *
     * @param[in] size Size in bytes to allocate
     */
    explicit CudaBuffer(const std::size_t size);
    ~CudaBuffer();

    // Delete copy operations
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    /**
     * Move constructor
     *
     * @param[in,out] other Buffer to move from
     */
    CudaBuffer(CudaBuffer &&other) noexcept;

    /**
     * Move assignment operator
     *
     * @param[in,out] other Buffer to move from
     * @return Reference to this
     */
    CudaBuffer &operator=(CudaBuffer &&other) noexcept;

    /**
     * Get device memory pointer
     *
     * @return Device memory pointer
     */
    [[nodiscard]] void *get() const { return ptr_; }

    /**
     * Get buffer size
     *
     * @return Size in bytes
     */
    [[nodiscard]] std::size_t size() const { return size_; }

    /**
     * Copy data from host to device
     *
     * @param[in] src Host memory pointer
     * @param[in] stream Optional CUDA stream for async copy
     */
    void copy_from_host(const void *src, const std::optional<cudaStream_t> stream = std::nullopt);

    /**
     * Copy data from device to host
     *
     * @param[out] dst Host memory pointer
     * @param[in] stream Optional CUDA stream for async copy
     */
    void copy_to_host(void *dst, const std::optional<cudaStream_t> stream = std::nullopt);
};

/**
 * RAII wrapper combining host memory, device memory, and tensor specification
 *
 * @tparam T Element type
 */
template <typename T> class CudaTensor {
private:
    TensorSpec spec_;
    std::vector<T> host_data_;
    CudaBuffer device_buffer_;
    std::string name_;

    /// Compute row-major (C-style) flat index from multi-dimensional indices
    [[nodiscard]] std::size_t
    compute_flat_index(const std::initializer_list<int64_t> indices) const;

    /// Compute row-major (C-style) flat index from vector of indices
    [[nodiscard]] std::size_t compute_flat_index(const std::vector<int64_t> &indices) const;

public:
    /**
     * Construct from shape only - type is automatically derived from T
     *
     * @param[in] shape Tensor dimensions
     * @param[in] name Optional tensor name for debugging
     */
    explicit CudaTensor(const nvinfer1::Dims &shape, std::string name = "")
            : spec_{shape, get_data_type<T>()}, host_data_(spec_.get_num_elements()),
              device_buffer_(spec_.get_bytes()), name_(std::move(name)) {}

    /**
     * Construct from full TensorSpec (validates type matches T)
     *
     * @param[in] spec Tensor specification
     * @param[in] name Optional tensor name for debugging
     * @throws std::invalid_argument if spec type doesn't match T
     */
    explicit CudaTensor(const TensorSpec &spec, std::string name = "")
            : spec_(spec), host_data_(spec.get_num_elements()), device_buffer_(spec.get_bytes()),
              name_(std::move(name)) {
        // Validate that spec type matches T
        if (spec_.type != get_data_type<T>()) {
            throw std::invalid_argument("TensorSpec type does not match template parameter T");
        }
    }

    /**
     * Destructor
     */
    ~CudaTensor() = default;

    // Delete copy operations (CUDA memory cannot be copied)
    CudaTensor(const CudaTensor &) = delete;
    CudaTensor &operator=(const CudaTensor &) = delete;

    /**
     * Move constructor
     */
    CudaTensor(CudaTensor &&) noexcept = default;

    /**
     * Move assignment operator
     *
     * @return Reference to this
     */
    CudaTensor &operator=(CudaTensor &&) noexcept = default;

    /**
     * Get tensor specification
     *
     * @return Tensor spec
     */
    [[nodiscard]] const TensorSpec &spec() const { return spec_; }

    /**
     * Get tensor name
     *
     * @return Tensor name
     */
    [[nodiscard]] const std::string &name() const { return name_; }

    /**
     * Get mutable host data
     *
     * @return Host data vector
     */
    std::vector<T> &host() { return host_data_; }

    /**
     * Get const host data
     *
     * @return Host data vector
     */
    [[nodiscard]] const std::vector<T> &host() const { return host_data_; }

    /**
     * Get mutable device pointer
     *
     * @return Device memory pointer
     */
    void *device() { return device_buffer_.get(); }

    /**
     * Get const device pointer
     *
     * @return Device memory pointer
     */
    [[nodiscard]] const void *device() const { return device_buffer_.get(); }

    /**
     * Get number of elements
     *
     * @return Total number of elements
     */
    [[nodiscard]] std::size_t size() const { return spec_.get_num_elements(); }

    /**
     * Flat index access
     *
     * @param[in] i Flat index
     * @return Reference to element
     */
    [[nodiscard]] T &operator[](const std::size_t i) { return host_data_[i]; }

    /**
     * Const flat index access
     *
     * @param[in] i Flat index
     * @return Const reference to element
     */
    [[nodiscard]] const T &operator[](const std::size_t i) const { return host_data_[i]; }

    /**
     * Multi-dimensional index access
     *
     * @param[in] indices Indices for each dimension
     * @return Reference to element
     */
    [[nodiscard]] T &operator()(const std::initializer_list<int64_t> indices) {
        return host_data_[compute_flat_index(indices)];
    }

    /**
     * Const multi-dimensional index access
     *
     * @param[in] indices Indices for each dimension
     * @return Const reference to element
     */
    [[nodiscard]] const T &operator()(const std::initializer_list<int64_t> indices) const {
        return host_data_[compute_flat_index(indices)];
    }

    /**
     * Multi-dimensional index access with vector
     *
     * @param[in] indices Indices for each dimension
     * @return Reference to element
     */
    [[nodiscard]] T &operator()(const std::vector<int64_t> &indices) {
        return host_data_[compute_flat_index(indices)];
    }

    /**
     * Const multi-dimensional index access with vector
     *
     * @param[in] indices Indices for each dimension
     * @return Const reference to element
     */
    [[nodiscard]] const T &operator()(const std::vector<int64_t> &indices) const {
        return host_data_[compute_flat_index(indices)];
    }

    /**
     * Access 2D tensor element
     *
     * @param[in] i0 Index in first dimension
     * @param[in] i1 Index in second dimension
     * @return Reference to element
     */
    [[nodiscard]] T &at(const int64_t i0, const int64_t i1) { return (*this)({i0, i1}); }

    /**
     * Const access 2D tensor element
     *
     * @param[in] i0 Index in first dimension
     * @param[in] i1 Index in second dimension
     * @return Const reference to element
     */
    [[nodiscard]] const T &at(const int64_t i0, const int64_t i1) const {
        return (*this)({i0, i1});
    }

    /**
     * Access 3D tensor element
     *
     * @param[in] i0 Index in first dimension
     * @param[in] i1 Index in second dimension
     * @param[in] i2 Index in third dimension
     * @return Reference to element
     */
    [[nodiscard]] T &at(const int64_t i0, const int64_t i1, const int64_t i2) {
        return (*this)({i0, i1, i2});
    }

    /**
     * Const access 3D tensor element
     *
     * @param[in] i0 Index in first dimension
     * @param[in] i1 Index in second dimension
     * @param[in] i2 Index in third dimension
     * @return Const reference to element
     */
    [[nodiscard]] const T &at(const int64_t i0, const int64_t i1, const int64_t i2) const {
        return (*this)({i0, i1, i2});
    }

    /**
     * Access 4D tensor element
     *
     * @param[in] i0 Index in first dimension
     * @param[in] i1 Index in second dimension
     * @param[in] i2 Index in third dimension
     * @param[in] i3 Index in fourth dimension
     * @return Reference to element
     */
    [[nodiscard]] T &at(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) {
        return (*this)({i0, i1, i2, i3});
    }

    /**
     * Const access 4D tensor element
     *
     * @param[in] i0 Index in first dimension
     * @param[in] i1 Index in second dimension
     * @param[in] i2 Index in third dimension
     * @param[in] i3 Index in fourth dimension
     * @return Const reference to element
     */
    [[nodiscard]] const T &
    at(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) const {
        return (*this)({i0, i1, i2, i3});
    }

    /**
     * Copy host data to device
     *
     * @param[in] stream Optional CUDA stream for async copy
     */
    void copy_to_device(const std::optional<cudaStream_t> stream = std::nullopt) {
        device_buffer_.copy_from_host(host_data_.data(), stream);
    }

    /**
     * Copy device data to host
     *
     * @param[in] stream Optional CUDA stream for async copy
     */
    void copy_from_device(const std::optional<cudaStream_t> stream = std::nullopt) {
        device_buffer_.copy_to_host(host_data_.data(), stream);
    }

    /**
     * Check if tensor has any non-zero values
     *
     * @return true if at least one element is non-zero
     */
    [[nodiscard]] bool has_non_zero() const;

    /**
     * Check if all tensor values are non-zero
     *
     * @return true if all elements are non-zero
     */
    [[nodiscard]] bool all_non_zero() const;

    /**
     * Count non-zero values in tensor
     *
     * @return Number of non-zero elements
     */
    [[nodiscard]] std::size_t count_non_zero() const;

    /**
     * @brief Format a slice of the tensor for printing
     * @param indices Indices where exactly one is Range (the slice dimension)
     * @param values_per_line Number of values to print per line
     * @return Formatted string with header, tab-indented wrapped values, and closing brace
     */
    [[nodiscard]] std::string
    format(const std::initializer_list<std::variant<int64_t, Range>> indices,
           const int values_per_line = 8) const;

    /**
     * @brief Format a slice of complex tensor for printing
     *
     * Assumes dimension 0 is [real, imag] and formats as complex numbers
     *
     * @param indices Indices where exactly one is Range (the slice dimension)
     *                Note: these indices are for dimensions 1..n (dim 0 is handled automatically)
     * @param values_per_line Number of complex values to print per line
     * @return Formatted string with tab-indented wrapped complex values
     */
    [[nodiscard]] std::string format_complex(
            const std::initializer_list<std::variant<int64_t, Range>> indices,
            const int values_per_line = 8) const;
};

/**
 * Helper class for binding tensors with shape/type validation
 */
class TensorBinder final {
private:
    struct BindingSpec {
        std::string tensor_name;
        void *buffer{};
        TensorSpec expected_spec;
        std::string description;
    };

    std::vector<BindingSpec> bindings_;

public:
    /**
     * Add a tensor binding with shape and type validation
     *
     * @param[in] tensor_name Name of tensor in engine
     * @param[in] buffer Device memory pointer
     * @param[in] expected_spec Expected tensor specification
     * @param[in] description Human-readable description
     * @return Reference to this for chaining
     */
    [[nodiscard]] TensorBinder &
    bind(const std::string &tensor_name,
         void *buffer,
         const TensorSpec &expected_spec,
         const std::string &description);

    /**
     * Add a tensor binding from CudaTensor
     *
     * @tparam T Tensor element type
     * @param[in] tensor_name Name of tensor in engine
     * @param[in,out] tensor CudaTensor to bind
     * @param[in] description Human-readable description
     * @return Reference to this for chaining
     */
    template <typename T>
    [[nodiscard]] TensorBinder &
    bind(const std::string &tensor_name, CudaTensor<T> &tensor, const std::string &description) {
        return bind(tensor_name, tensor.device(), tensor.spec(), description);
    }

    /**
     * Apply all bindings to the execution context with validation
     *
     * @param[in] context Execution context to bind tensors to
     * @param[in] engine CUDA engine for validation
     * @param[in] logger Logger for diagnostic messages
     * @return True if all bindings were successful, false otherwise
     */
    [[nodiscard]] bool
    apply(nvinfer1::IExecutionContext *context,
          nvinfer1::ICudaEngine *engine,
          StdioLogger &logger) const;
};

/**
 * Custom logger for TensorRT with stdout output
 */
class StdioLogger final : public nvinfer1::ILogger {
private:
    Severity min_severity_{};

public:
    /**
     * Construct logger with minimum severity level
     *
     * @param[in] min_severity Minimum severity to log
     */
    explicit StdioLogger(const Severity min_severity = Severity::kVERBOSE)
            : min_severity_(min_severity) {}

    /**
     * Log a message
     *
     * @param[in] severity Message severity
     * @param[in] msg Message text
     */
    void log(const Severity severity, const char *msg) noexcept override;
};

/**
 * RAII wrapper for TensorRT engine
 */
class TrtEngine final {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

public:
    ~TrtEngine() = default;

    /**
     * Construct TensorRT engine from engine file
     *
     * The engine file is loaded from the directory specified by the
     * RAN_TRT_ENGINE_PATH environment variable.
     *
     * @param[in] engine_filename Name of the engine file (e.g., "model.trtengine")
     * @param[in] logger Logger for diagnostic messages
     * @throws std::runtime_error if environment variable not set or file cannot be loaded
     */
    TrtEngine(const std::string &engine_filename, StdioLogger &logger);

    // Delete copy operations
    TrtEngine(const TrtEngine &) = delete;
    TrtEngine &operator=(const TrtEngine &) = delete;

    /**
     * Move constructor
     */
    TrtEngine(TrtEngine &&) noexcept = default;

    /**
     * Move assignment operator
     *
     * @return Reference to this
     */
    TrtEngine &operator=(TrtEngine &&) noexcept = default;

    /**
     * Get underlying CUDA engine
     *
     * @return CUDA engine pointer
     */
    [[nodiscard]] nvinfer1::ICudaEngine *get_engine() const { return engine_.get(); }

    /**
     * Get execution context
     *
     * @return Execution context pointer
     */
    [[nodiscard]] nvinfer1::IExecutionContext *get_context() const { return context_.get(); }

    /**
     * Print engine information
     *
     * @param[in] logger Logger for output messages
     */
    void print_engine_info(StdioLogger &logger) const;
};

/**
 * Initialize RAN TensorRT plugin library
 *
 * @param[in] logger TensorRT logger pointer
 * @param[in] lib_namespace Plugin namespace string (defaults to empty string)
 * @return true if initialization successful
 */
extern "C" bool init_ran_plugins(void *logger, const char *lib_namespace = "");

/**
 * Set CUDA device
 *
 * Checks that CUDA is available and sets the specified device as active.
 * Prints errors to std::cerr on failure.
 *
 * @param[in] device_id CUDA device ID to set
 * @return true if successful, false on error
 */
[[nodiscard]] bool set_cuda_device(int device_id);

/**
 * Check if TRT engine file exists
 *
 * @param[in] engine_filename Name of the engine file (e.g., "model.trtengine")
 * @return true if engine file exists in RAN_TRT_ENGINE_PATH directory
 */
[[nodiscard]] bool engine_exists(const std::string &engine_filename);

} // namespace ran::trt_utils

// Template implementations that must be in header
#include "trt_test_utils_impl.hpp" // IWYU pragma: keep

#endif // RAN_TRT_UTILS_HPP
