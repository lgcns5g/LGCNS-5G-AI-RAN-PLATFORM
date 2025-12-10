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
#include <memory>
#include <stdexcept>

#include <cusolverdx.hpp>

#include <cuda_runtime.h>

/**
 * @brief CUDA error checking macro
 *
 * Checks CUDA function calls and throws std::runtime_error if an error occurs.
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
 * @brief Template-based cuSOLVERDx Cholesky factor inversion kernel
 *
 * This kernel uses cuSOLVERDx for device-side fused Cholesky decomposition and
 * L inverse.
 *
 * Algorithm:
 * 1. Cholesky decomposition: A = L*L^H (POTRF)
 * 2. Zero out upper triangle of L (POTRF only fills lower triangle)
 * 3. Solve L*X = I column-by-column (TRSM with lower triangular)
 * Result: X = L^{-1} (inverse of Cholesky factor)
 *
 * Note: The kernel computes L^{-1}, which can be used for whitening in
 * MMSE-IRC equalizers.
 *
 * Memory Layout:
 * - Input/output: row-major
 * - cuSOLVERDx: row-major
 * - Row-major indexing: A[row * lda + col]
 *
 * Design Decisions:
 * - Use row-major Arrangement to avoid transpose overhead when using with AI frameworks
 * - TRSM solves column-by-column at a time (incl. in row-major layout)
 * - Zero upper triangle after POTRF (only lower triangle is filled, see docs)
 * - Architecture set to SM<900> for Hopper GPU
 * - CRITICAL: BlockDim<32, 1, 1>() must be specified in BOTH POTRF and TRSM operator
 *   declarations AND must match the actual kernel launch block size. Mismatch can cause
 *   incorrect results due to thread synchronization issues.
 * - Always launch with 32 threads (cuSOLVERDx uses at most 32 threads for these sizes)
 *
 * @tparam MATRIX_SIZE Size of the square matrix (N for NxN matrix)
 * @tparam IS_COMPLEX true for complex data, false for real data
 */
template <int32_t MATRIX_SIZE, bool IS_COMPLEX>
__global__ void cholesky_factor_inv_template_kernel(
        const float *input_real,
        const float *input_imag,
        const int32_t batch_size,
        float *output_real,
        float *output_imag,
        void *workspace) {
    using namespace cusolverdx;

    // Cholesky decomposition descriptor using cuSOLVERDx operators (row-major).
    // For complex: Precision<float>() + Type<type::complex>()
    // For real: Precision<float>() only
    // IMPORTANT: BlockDim must match actual kernel launch block size (32 threads)
    using POTRF = std::
            conditional_t<IS_COMPLEX, decltype(Size<MATRIX_SIZE>() + Precision<float>() + Type<type::complex>() + Function<function::potrf>() + FillMode<fill_mode::lower>() + Arrangement<arrangement::row_major>() + SM<900>() + Block() + BlockDim<32, 1, 1>()), decltype(Size<MATRIX_SIZE>() + Precision<float>() + Function<function::potrf>() + FillMode<fill_mode::lower>() + Arrangement<arrangement::row_major>() + SM<900>() + Block() + BlockDim<32, 1, 1>())>;

    // Lower triangular solve descriptor for L^{-1} (row-major).
    // TRSM solves L*X = I where L is lower triangular Cholesky factor.
    // General TRSM parameters: op(A)*X = B where
    //   - A is K×K triangular matrix (side::left)
    //   - B is M×N matrix (M rows, N columns = number of RHS)
    //   - For side::left: M must equal K
    // Size<K, M, N> template parameters:
    //   - K = dimension of square triangular matrix A
    //   - M = number of rows in B (must equal K for side::left)
    //   - N = number of columns in B (number of right-hand sides)
    // IMPORTANT: BlockDim must match actual kernel launch block size (32 threads)
    using TRSM = std::
            conditional_t<IS_COMPLEX, decltype(Size<MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE>() + Precision<float>() + Type<type::complex>() + Function<function::trsm>() + Side<side::left>() + FillMode<fill_mode::lower>() + TransposeMode<transpose::non_transposed>() + Diag<diag::non_unit>() + Arrangement<arrangement::row_major, arrangement::row_major>() + LeadingDimension<MATRIX_SIZE, MATRIX_SIZE>() + SM<900>() + Block() + BlockDim<32, 1, 1>()), decltype(Size<MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE>() + Precision<float>() + Function<function::trsm>() + Side<side::left>() + FillMode<fill_mode::lower>() + TransposeMode<transpose::non_transposed>() + Diag<diag::non_unit>() + Arrangement<arrangement::row_major, arrangement::row_major>() + LeadingDimension<MATRIX_SIZE, MATRIX_SIZE>() + SM<900>() + Block() + BlockDim<32, 1, 1>())>;

    using value_type = typename POTRF::a_data_type;

    // Verify thread requirements at compile time (only for POTRF)
    static_assert(
            POTRF::block_dim.x * POTRF::block_dim.y * POTRF::block_dim.z <= 1024,
            "POTRF requires more threads than block size limit");

    // Shared memory for cuSOLVERDx operations (operates on matrices in shared memory)
    extern __shared__ __align__(16) char shared_mem_raw[];
    value_type *shared_mem = reinterpret_cast<value_type *>(shared_mem_raw);

    // Shared memory for info flag (must be in shared memory, not registers)
    // Only one thread writes to this, but all threads need to read the same value
    __shared__ int shared_info;

    // For one matrix per block, use block index as global matrix id
    const unsigned int global_matrix_id = blockIdx.x;
    if (global_matrix_id >= batch_size) {
        return; // Skip if beyond batch size
    }

    // Get thread ID within block
    const unsigned int tid = threadIdx.x;
    const unsigned int num_matrix_elements = MATRIX_SIZE * MATRIX_SIZE;

    // Pointers to matrices in shared memory
    // Use TRSM sizes as they account for the extended B matrix
    value_type *A = shared_mem;
    value_type *B = shared_mem + TRSM::a_size;

    // Load input matrix from global memory to shared memory (matrix A)
    // Both input and A are row-major (element at row r, col c: A[r * lda + c])
    const unsigned int offset = num_matrix_elements * global_matrix_id;
    for (unsigned int i = tid; i < num_matrix_elements; i += blockDim.x) {
        if constexpr (IS_COMPLEX) {
            // Complex: convert separate real/imag arrays to cuSOLVERDx complex type
            A[i] = value_type(input_real[offset + i], input_imag[offset + i]);
        } else {
            // Real: direct copy
            A[i] = input_real[offset + i];
        }
    }

    // Initialize B matrix as identity (MATRIX_SIZE × MATRIX_SIZE, row-major layout)
    for (unsigned int i = tid; i < num_matrix_elements; i += blockDim.x) {
        const unsigned int row = i / MATRIX_SIZE;
        const unsigned int col = i % MATRIX_SIZE;
        if constexpr (IS_COMPLEX) {
            B[i] = (row == col) ? value_type(1.0f, 0.0f) : value_type(0.0f, 0.0f);
        } else {
            B[i] = (row == col) ? 1.0f : 0.0f;
        }
    }
    __syncthreads();

    // Execute Cholesky decomposition using cuSOLVERDx (A = L*L^H)
    // Info must be in shared memory so all threads see the same value
    if (tid == 0) {
        shared_info = 0; // Initialize
    }
    __syncthreads();

    POTRF().execute(A, &shared_info);
    __syncthreads();

    // Check for positive definiteness
    // All threads read from shared memory, so they all see the same value
    if (shared_info != 0) {
        // Matrix is not positive definite - set output to zero
        for (unsigned int i = tid; i < num_matrix_elements; i += blockDim.x) {
            output_real[offset + i] = 0.0f;
            if constexpr (IS_COMPLEX) {
                output_imag[offset + i] = 0.0f;
            }
        }
        return;
    }

    // For lower fill mode, only the diagonal and lower triangular part of A is processed,
    // the upper part of the matrix is untouched. We need to zero out the upper part of A.
    for (unsigned int i = tid; i < num_matrix_elements; i += blockDim.x) {
        const unsigned int row = i / MATRIX_SIZE;
        const unsigned int col = i % MATRIX_SIZE;
        if (col > row) {
            if constexpr (IS_COMPLEX) {
                A[i] = value_type(0.0f, 0.0f);
            } else {
                A[i] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Solve L*X = I to get X = L^{-1}
    // B is M x N (MATRIX_SIZE x MATRIX_SIZE) identity matrix
    // For row-major: ldb = N (leading dimension of B)
    TRSM().execute(A, MATRIX_SIZE, B, MATRIX_SIZE);
    __syncthreads();

    // Save L^{-1} back to global memory
    for (unsigned int i = tid; i < num_matrix_elements; i += blockDim.x) {
        if constexpr (IS_COMPLEX) {
            // Complex: split cuSOLVERDx complex type to separate real/imag arrays
            output_real[offset + i] = B[i].real();
            output_imag[offset + i] = B[i].imag();
        } else {
            // Real: direct copy
            output_real[offset + i] = B[i];
        }
    }
}

// Explicit template instantiations for specific matrix sizes and data types
// These create separate, optimized kernels for each configuration

// Real float instantiations (IS_COMPLEX=false)
template __global__ void cholesky_factor_inv_template_kernel<2, false>(
        const float *, const float *, const int32_t, float *, float *, void *);

template __global__ void cholesky_factor_inv_template_kernel<4, false>(
        const float *, const float *, const int32_t, float *, float *, void *);

template __global__ void cholesky_factor_inv_template_kernel<8, false>(
        const float *, const float *, const int32_t, float *, float *, void *);

// Complex float instantiations (IS_COMPLEX=true)
template __global__ void cholesky_factor_inv_template_kernel<2, true>(
        const float *, const float *, const int32_t, float *, float *, void *);

template __global__ void cholesky_factor_inv_template_kernel<4, true>(
        const float *, const float *, const int32_t, float *, float *, void *);

template __global__ void cholesky_factor_inv_template_kernel<8, true>(
        const float *, const float *, const int32_t, float *, float *, void *);

/**
 * @brief Launches the cuSOLVERDx Cholesky inversion kernel (batched)
 *
 * This function configures and launches the CUDA kernel for Cholesky
 * decomposition followed by matrix inversion using cuSOLVERDx library.
 * It handles kernel launch parameters and error checking.
 *
 * Supports both real and complex data types:
 * - For REAL data (is_complex=false):
 *   * input_real = the real input data
 *   * input_imag = nullptr (unused)
 *   * output_real = the real output data
 *   * output_imag = nullptr (unused)
 *
 * - For COMPLEX data (is_complex=true):
 *   * input_real = real part of complex input
 *   * input_imag = imaginary part of complex input
 *   * output_real = real part of complex output
 *   * output_imag = imaginary part of complex output
 *
 * TensorRT doesn't support complex types, so complex data is split into
 * separate real and imaginary arrays at the interface level.
 *
 * @param[in] input_real Real data (if is_complex=false) or real part (if is_complex=true)
 * @param[in] input_imag Imaginary part (if is_complex=true) or nullptr (if is_complex=false)
 * @param[in] matrixSize Size of each square matrix (N for NxN)
 * @param[in] batchSize Total number of matrices to process
 * @param[out] output_real Real output (if is_complex=false) or real part (if is_complex=true)
 * @param[out] output_imag Imaginary part (if is_complex=true) or nullptr (if is_complex=false)
 * @param[in] workspace Workspace memory for cuSOLVERDx computation
 * @param[in] stream CUDA stream for asynchronous execution
 * @param[in] is_complex false for real data, true for complex data
 *
 * @throws std::invalid_argument if matrixSize <= 0
 * @throws std::invalid_argument if batchSize <= 0
 * @throws std::invalid_argument if matrixSize is not supported (must be 2, 4, or 8)
 * @throws std::runtime_error if CUDA kernel launch fails
 */
void launch_cholesky_factor_inv_kernel(
        const float *input_real,
        const float *input_imag,
        int32_t matrixSize,
        int32_t batchSize,
        float *output_real,
        float *output_imag,
        void *workspace,
        cudaStream_t stream,
        bool is_complex = false) {
    using namespace cusolverdx;

    // Validate input parameters
    if (matrixSize <= 0) {
        throw std::invalid_argument("Cholesky matrix size must be positive");
    }
    if (batchSize <= 0) {
        throw std::invalid_argument("Cholesky batch size must be positive");
    }

    // Runtime dispatch to template kernel based on matrix size
    dim3 grid(batchSize); // One matrix per block

    // IMPORTANT: Always use 32 threads to match BlockDim<32, 1, 1> in operator declarations
    // cuSOLVERDx uses at most 32 threads for these matrix sizes, and the BlockDim template
    // parameter must match the actual kernel launch configuration
    dim3 block(32, 1, 1);

    // Calculate shared memory size for cuSOLVERDx
    // cuSOLVERDx provides size requirements via a_size and b_size traits
    // For complex, each element is commondx::detail::complex<float> (2 floats)
    const int32_t element_size = is_complex ? 2 * sizeof(float) : sizeof(float);
    const int32_t matrix_elements = matrixSize * matrixSize;
    // Allocate space for both A and B matrices (both MATRIX_SIZE × MATRIX_SIZE)
    // with padding for alignment and any internal workspace cuSOLVERDx might need
    const int32_t shared_mem_size = std::max(16384, 2 * matrix_elements * element_size + 4096);

    // Dispatch to correct template based on matrix size and data type
    if (is_complex) {
        // Complex data type dispatch (IS_COMPLEX=true)
        if (matrixSize == 2) {
            cholesky_factor_inv_template_kernel<2, true><<<grid, block, shared_mem_size, stream>>>(
                    input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else if (matrixSize == 4) {
            cholesky_factor_inv_template_kernel<4, true><<<grid, block, shared_mem_size, stream>>>(
                    input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else if (matrixSize == 8) {
            cholesky_factor_inv_template_kernel<8, true><<<grid, block, shared_mem_size, stream>>>(
                    input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else {
            throw std::invalid_argument("Unsupported complex matrix size (supported: 2, 4, 8)");
        }
    } else {
        // Real data type dispatch (IS_COMPLEX=false)
        if (matrixSize == 2) {
            cholesky_factor_inv_template_kernel<2, false><<<grid, block, shared_mem_size, stream>>>(
                    input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else if (matrixSize == 4) {
            cholesky_factor_inv_template_kernel<4, false><<<grid, block, shared_mem_size, stream>>>(
                    input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else if (matrixSize == 8) {
            cholesky_factor_inv_template_kernel<8, false><<<grid, block, shared_mem_size, stream>>>(
                    input_real, input_imag, batchSize, output_real, output_imag, workspace);
        } else {
            throw std::invalid_argument("Unsupported real matrix size (supported: 2, 4, 8)");
        }
    }

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ran::trt_plugin
