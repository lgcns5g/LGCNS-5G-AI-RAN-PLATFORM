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

#include <cassert>
#include <complex>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <cufft.h>
#include <cufftdx.hpp>

#include <cuda_runtime.h>

/**
 * @brief CUDA error checking macro
 *
 * Checks CUDA function calls and throws std::runtime_error if an error occurs.
 * This provides better error handling than silent failures.
 */
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            throw std::runtime_error(cudaGetErrorString(error));                                   \
        }                                                                                          \
    } while (0)

namespace ran::trt_plugin {

/**
 * @brief Template-based cuFFTDx FFT kernel
 *
 * This kernel uses cuFFTDx for device-side FFT computation. The template
 * parameters allow compile-time optimization while supporting runtime
 * configuration through TensorRT plugin parameters.
 *
 * @tparam FFT_SIZE Size of the FFT to compute
 * @tparam FFTS_PER_BLOCK Number of FFTs per CUDA block
 * @tparam ELEMENTS_PER_THREAD Number of elements per thread
 */
/**
 * @brief Template-based cuFFTDx FFT kernel
 *
 * Pre-compiled kernel template for specific FFT configurations.
 * Each instantiation is optimized for a specific FFT size, direction, and configuration.
 *
 * @tparam FFT_SIZE Size of the FFT to compute
 * @tparam FFT_DIRECTION Direction of the FFT (fft_direction::forward or fft_direction::inverse)
 * @tparam ELEMENTS_PER_THREAD Number of elements per thread
 */
template <int32_t FFT_SIZE, cufftdx::fft_direction FFT_DIRECTION, int32_t ELEMENTS_PER_THREAD = 8>
__global__ void cufftdx_fft_template_kernel(
        const float *input_real,
        const float *input_imag,
        const int32_t batch_size,
        float *output_real,
        float *output_imag,
        void *workspace) {
    using namespace cufftdx;

    // Define the FFT descriptor using cuFFTDx operators - this is compile-time
    using FFT =
            decltype(Size<FFT_SIZE>() + Precision<float>() + Type<fft_type::c2c>() + Direction<FFT_DIRECTION>() + FFTsPerBlock<1>() + ElementsPerThread<ELEMENTS_PER_THREAD>() + SM<700>() + Block());

    using complex_type = typename FFT::value_type;

    // Thread-local storage for FFT data
    complex_type thread_data[FFT::storage_size];

    // Shared memory for FFT operations
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    // For single FFT per block, use block index as global FFT id
    const unsigned int global_fft_id = blockIdx.x;

    if (global_fft_id >= batch_size) {
        return; // Skip if beyond batch size
    }

    // Load data from global memory to registers
    const unsigned int offset = FFT_SIZE * global_fft_id;
    const unsigned int stride = FFT::stride;
    unsigned int index = offset + threadIdx.x;

    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < FFT_SIZE) {
            // Convert separate real and imaginary data to complex
            thread_data[i] = complex_type(input_real[index], input_imag[index]);
            index += stride;
        }
    }

    // Execute FFT using cuFFTDx
    FFT().execute(thread_data, shared_mem);

    // Save results back to global memory using cuFFTDx's stride pattern
    // Apply normalization for inverse FFT (1/N scaling)
    const float normalization_factor =
            (FFT_DIRECTION == fft_direction::inverse) ? (1.0f / FFT_SIZE) : 1.0f;

    index = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < FFT_SIZE) {
            // Convert complex data back to separate real and imaginary parts
            // Apply normalization for inverse FFT to match JAX behavior
            output_real[index] = thread_data[i].real() * normalization_factor;
            output_imag[index] = thread_data[i].imag() * normalization_factor;
            index += stride;
        }
    }
}

// Explicit template instantiations for specific FFT sizes and directions
// These create separate, optimized kernels for each configuration
// Forward FFTs
template __global__ void cufftdx_fft_template_kernel<128, cufftdx::fft_direction::forward, 8>(
        const float *, const float *, const int32_t, float *, float *, void *);
template __global__ void cufftdx_fft_template_kernel<2048, cufftdx::fft_direction::forward, 8>(
        const float *, const float *, const int32_t, float *, float *, void *);
template __global__ void cufftdx_fft_template_kernel<4096, cufftdx::fft_direction::forward, 8>(
        const float *, const float *, const int32_t, float *, float *, void *);
// Inverse FFTs
template __global__ void cufftdx_fft_template_kernel<128, cufftdx::fft_direction::inverse, 8>(
        const float *, const float *, const int32_t, float *, float *, void *);
template __global__ void cufftdx_fft_template_kernel<2048, cufftdx::fft_direction::inverse, 8>(
        const float *, const float *, const int32_t, float *, float *, void *);
template __global__ void cufftdx_fft_template_kernel<4096, cufftdx::fft_direction::inverse, 8>(
        const float *, const float *, const int32_t, float *, float *, void *);

/**
 * @brief Launches the cuFFTDx computation kernel (batched)
 *
 * This function configures and launches the CUDA kernel for FFT computation
 * using cuFFTDx library. It handles kernel launch parameters and error checking.
 *
 * @param[in] input Pointer to input interleaved complex data array (GPU memory,
 * batch_size * fft_size * 2 elements)
 * @param[in] fftSize Size of the FFT to compute
 * @param[in] batchSize Number of input signals to process in parallel
 * @param[out] output Pointer to output interleaved complex data buffer (GPU memory,
 * batch_size * fft_size * 2 elements)
 * @param[in] workspace Workspace memory for FFT computation
 * @param[in] stream CUDA stream for asynchronous execution
 * @param[in] precision FFT precision (0=float, 1=double)
 * @param[in] fftType FFT type (0=c2c, 1=r2c, 2=c2r)
 * @param[in] direction FFT direction (0=forward, 1=inverse)
 * @param[in] fftsPerBlock Number of FFTs per CUDA block
 * @param[in] elementsPerThread Number of elements per thread
 *
 * @throws std::invalid_argument if fftSize <= 0
 * @throws std::invalid_argument if batchSize <= 0
 * @throws std::invalid_argument if fftSize is not supported (must be 128, 2048, or 4096)
 * @throws std::runtime_error if CUDA kernel launch fails
 *
 * @note The kernel is launched with optimized grid/block dimensions
 * @note Workspace parameter is used for cuFFTDx operations
 * @see fft_kernel for the actual kernel implementation
 */
void launch_fft_kernel(
        const float *input_real,
        const float *input_imag,
        int32_t fftSize,
        int32_t batchSize,
        float *output_real,
        float *output_imag,
        void *workspace,
        cudaStream_t stream,
        int32_t precision = 0,
        int32_t fftType = 0,
        int32_t direction = 0,
        int32_t fftsPerBlock = 1,
        int32_t elementsPerThread = 8) {
    using namespace cufftdx;

    // Validate input parameters
    if (fftSize <= 0) {
        throw std::invalid_argument(std::format("Invalid FFT size: {}", fftSize));
    }
    if (batchSize <= 0) {
        throw std::invalid_argument(std::format("Invalid batch size: {}", batchSize));
    }

    // Runtime dispatch to template kernel based on FFT size
    // Currently supports 128-point, 2048-point, and 4096-point FFT
    // Additional FFT sizes can be added by including template specializations
    // and corresponding runtime dispatch cases

    // Configure kernel launch parameters for cuFFTDx
    // Use cuFFTDx's recommended configuration for 128-point FFT
    constexpr int32_t ffts_per_block = 1;

    // For 128-point FFT, cuFFTDx typically uses 64 threads per block
    // This is based on cuFFTDx's internal thread mapping
    const int32_t threads_per_block = 64; // cuFFTDx's recommended thread count for 128-point FFT
    const int32_t num_blocks = batchSize; // One FFT per block

    dim3 block(threads_per_block, ffts_per_block);
    dim3 grid(num_blocks);

    // Calculate shared memory size for cuFFTDx
    // For 128-point FFT with 64 threads, cuFFTDx needs more shared memory
    const int32_t shared_mem_size = 8192; // Increased shared memory for cuFFTDx

    // Determine FFT direction from parameter (0=forward, 1=inverse)
    const bool is_inverse = (direction == 1);

    if (fftSize == 128) {
        if (is_inverse) {
            cufftdx_fft_template_kernel<128, fft_direction::inverse, 8>
                    <<<grid, block, shared_mem_size, stream>>>(
                            input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else {
            cufftdx_fft_template_kernel<128, fft_direction::forward, 8>
                    <<<grid, block, shared_mem_size, stream>>>(
                            input_real, input_imag, batchSize, output_real, output_imag, workspace);
        }
    } else if (fftSize == 2048) {
        // For 2048-point FFT, cuFFTDx typically uses 256 threads per block
        // This is based on cuFFTDx's internal thread mapping for larger FFTs
        const int32_t threads_per_block_2048 = 256;
        const int32_t shared_mem_size_2048 = 32768; // Increased shared memory for 2048-point FFT

        dim3 block_2048(threads_per_block_2048, ffts_per_block);
        dim3 grid_2048(batchSize);
        if (is_inverse) {
            cufftdx_fft_template_kernel<2048, fft_direction::inverse, 8>
                    <<<grid_2048, block_2048, shared_mem_size_2048, stream>>>(
                            input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else {
            cufftdx_fft_template_kernel<2048, fft_direction::forward, 8>
                    <<<grid_2048, block_2048, shared_mem_size_2048, stream>>>(
                            input_real, input_imag, batchSize, output_real, output_imag, workspace);
        }
    } else if (fftSize == 4096) {
        // For 4096-point FFT, cuFFTDx typically uses 512 threads per block
        // This is based on cuFFTDx's internal thread mapping for larger FFTs
        const int32_t threads_per_block_4096 = 512;
        const int32_t shared_mem_size_4096 = 65536; // Increased shared memory for 4096-point FFT

        dim3 block_4096(threads_per_block_4096, ffts_per_block);
        dim3 grid_4096(batchSize);
        if (is_inverse) {
            cufftdx_fft_template_kernel<4096, fft_direction::inverse, 8>
                    <<<grid_4096, block_4096, shared_mem_size_4096, stream>>>(
                            input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else {
            cufftdx_fft_template_kernel<4096, fft_direction::forward, 8>
                    <<<grid_4096, block_4096, shared_mem_size_4096, stream>>>(
                            input_real, input_imag, batchSize, output_real, output_imag, workspace);
        }
    } else {
        throw std::invalid_argument(
                std::format("Unsupported FFT size: {} (supported: 128, 2048, 4096)", fftSize));
    }

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ran::trt_plugin
