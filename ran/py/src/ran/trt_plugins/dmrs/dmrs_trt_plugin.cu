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
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include "GOLD_1_SEQ_LUT.h"
#include "GOLD_2_32_P_LUT.h"

// Constants for DMRS generation
#define WORD_SIZE 32
#define POLY_2 0x8000000F // Polynomial 2 for DMRS sequence generation

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

// Galois LFSR for Gold sequence generation
// Computes 31 bits using the galois mask 0xF. Note that this is a device-only
// function due to the use of __brev().
__device__ inline uint32_t galois31MaskLFSRWord(uint32_t state) {
    // We start with the following recurrence relations for the state (s) and
    // result (r):
    //              /  s(b-1,i)                                        b > 3
    //   s(b,i+1) = |  s(b-1,i) ^ s(30,i)                              b = 1, 2, 3
    //              \  s(30,i)                                         b = 0
    //
    //              /  r(b,i)                                          b != i
    //   r(b,i+1) = |
    //              \  s(30,i)                                         otherwise
    // The second branch of r does not include r(i,0) because r(i,0) is zero for
    // all i. We do not actually need to compute the full new state, but we will
    // need to solve for some state bits to fully compute the result.
    //
    // The bits of the result (res) are thus as follows:
    //   r(0,32) = r(0,31) = ... = r(0,1) = s(30,0)
    //   r(1,32) = r(1,2) = s(30,1) = s(29,0)
    //   r(2,32) = r(2,3) = s(30,2) = s(28,0)
    //   ...
    //   r(27,32) = r(27,28) = s(30,27) = s(3,0)
    //   r(28,32) = r(28,29) = s(30,28) = s(3,1) = s(2,0) ^ s(30,0)
    //   r(29,32) = r(29,30) = s(30,29) = s(3,2) = s(2,1) ^ s(30,1) = s(1,0) ^
    //   s(30,0) ^ s(29,0) r(30,32) = r(30,31) = s(30,30) = s(3,3) = s(2,2) ^
    //   s(30,2) = s(1,1) ^ s(30,1) ^ s(30,2)
    //            = s(0,0) ^ s(30,0) ^ s(29,0) ^ s(28,0)
    //   r(31,32) = 0 because we only generate 31 output bits (n=31)
    const uint32_t rev_state = __brev(state);
    const uint32_t res =
            ((rev_state >> 1) & 0xFFFFFFF) |
            // bit 28 - s(2,0) ^ s(30,0)
            (((state & 0x4) << 26) ^ ((state & 0x40000000) >> 2)) |
            // bit 29 - s(1,0) ^ s(30,0) ^ s(29,0)
            (((state & 0x2) << 28) ^ ((state & 0x40000000) >> 1) ^ (state & 0x20000000)) |
            // bit 30 - s(0,0) ^ s(30,0) ^ s(29,0) ^ s(28,0)
            (((state & 0x1) << 30) ^ ((state & 0x40000000) >> 0) ^ ((state & 0x20000000) << 1) ^
             ((state & 0x10000000) << 2));
    // bit 31 is 0
    return res;
}

/**
 * @brief Multiply by polynomial 2 using high 31 bits
 *
 * @param[in] state Input state
 * @return Result of polynomial multiplication
 */
__device__ __forceinline__ uint32_t polyBMulHigh31(uint32_t a) {
    uint32_t prodHi = (a >> 30) ^ (a >> 29) ^ (a >> 28) ^ a;
    return prodHi;
}

/**
 * @brief Modular polynomial multiplication using lookup table
 *
 * @param[in] state Input state
 * @param[in] lut_val Lookup table value
 * @param[in] poly Polynomial value
 * @return Result of modular polynomial multiplication
 */
__device__ __forceinline__ uint32_t mulModPoly31LUT(uint32_t a, uint32_t b, uint32_t poly) {
    uint32_t prod = 0;
    // a moduloe POLY_2, 31 BITs
    uint32_t crc = a ^ (a >= POLY_2) * POLY_2;
#pragma unroll
    for (int i = 0; i < 31; i++) {
        prod ^= (crc & 1) * b;
        b = (b << 1) ^ (b & (1 << (30)) ? poly : 0);
        crc >>= 1;
    }

    return prod;
}

/**
 * @brief Generate Fibonacci LFSR output (32 bits)
 *
 * @param[in] fstate Input state
 * @return LFSR output (32 bits)
 */
__device__ __forceinline__ uint32_t fibonacciLFSR2_1bit(uint32_t state) {
    uint32_t res = state;
    // x^{31} + x^3 + x^2 + x + 1
    uint32_t bit = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
    bit = bit & 1;
    state >>= 1;
    state ^= (bit << 30);
    res ^= (state >> 30) << 31;
    return res;
}

/**
 * @brief DMRS sequence generation exactly matching the reference implementation
 *
 * This function implements the exact algorithm from the reference
 * implementation using the seed2 parameter and lookup tables correctly.
 *
 * @param[in] seed2 Initialization seed for DMRS sequence (c_init value)
 * @param[in] n Offset in the sequence (in bits)
 * @return 32-bit DMRS sequence value
 */
__device__ __forceinline__ uint32_t gold32(uint32_t seed2, uint32_t n) {
    uint32_t prod2;

    // Reverse 31 bits of the seed
    uint32_t state2 = __brev(seed2) >> 1; // reverse 31 bits

    // Multiply by polynomial 2
    state2 = polyBMulHigh31(state2);

    // Compute modular polynomial multiplication using lookup table
    prod2 = mulModPoly31LUT(state2, GOLD_2_32_P_LUT[(n) / WORD_SIZE], POLY_2);

    // Apply Galois field mask
    uint32_t fstate2 = galois31MaskLFSRWord(prod2);

    // Generate Fibonacci LFSR output
    uint32_t output2 = fibonacciLFSR2_1bit(fstate2);

    // XOR with precomputed DMRS sequence 1
    return GOLD_1_SEQ_LUT[n / WORD_SIZE] ^ output2;
}

// Helper function to compute c_init for a given combination of parameters
__device__ __forceinline__ int32_t
compute_c_init(int32_t slot_number, int32_t n_t, int32_t n_dmrs_id, int32_t t, int32_t n_scid) {
    // c_init = ((1 << 17) * (slot_number * n_t + t) * (2 * n_dmrs_id + 1) + 2 *
    // n_dmrs_id + n_scid) % ((1 << 31) - 1)
    int64_t temp = (1LL << 17) * (slot_number * n_t + t) * (2 * n_dmrs_id + 1);
    temp += 2 * n_dmrs_id + n_scid;
    return static_cast<int32_t>(temp % ((1LL << 31) - 1));
}

/**
 * DMRS kernel that generates sequences for all n_t symbols and n_scid ports
 *
 * Generates both complex DMRS values (with frequency scrambling applied) and
 * binary gold sequences.
 *
 * @param[in] input_params GPU pointer to [slot_number, n_dmrs_id]
 * @param[in] sequence_length Sequence length per port
 * @param[in] n_t Number of OFDM symbols per slot
 * @param[out] r_dmrs_ri_sym_cdm_sc Complex DMRS output (2, n_t, 2, sequence_length/2)
 * @param[out] scr_seq_sym_ri_sc Binary gold sequence output (n_t, 2, sequence_length)
 */
__global__ void dmrs_kernel(
        const int32_t *input_params,
        const int32_t sequence_length,
        const int32_t n_t,
        float *r_dmrs_ri_sym_cdm_sc,
        int32_t *scr_seq_sym_ri_sc) {

    // Read parameters from GPU memory
    const int32_t slot_number = input_params[0];
    const int32_t n_dmrs_id = input_params[1];

    // Each block processes all sequences for one (t, n_scid) combination
    // Total sequences: n_t * 2 (for n_scid=0,1)
    const int32_t total_sequences = n_t * 2;

    // Calculate how many 32-bit chunks we need per sequence
    const int32_t chunks_per_sequence = (sequence_length + 31) / 32;

    // Constant for frequency scrambling: 1/sqrt(2)
    constexpr float inv_sqrt2 = 0.70710678118f;

    // Half-length for complex output (pairs become single complex values)
    const int32_t half_length = sequence_length / 2;

    // Use stride-based loop pattern
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_sequences * chunks_per_sequence;
         i += gridDim.x * blockDim.x) {

        // Calculate which sequence and which chunk within that sequence
        const int32_t seq_idx = i / chunks_per_sequence;
        const int32_t chunk_in_seq = i % chunks_per_sequence;

        if (seq_idx >= total_sequences) {
            continue;
        }

        // Map seq_idx to (t, n_scid)
        // Layout: t=1,scid=0; t=1,scid=1; t=2,scid=0; t=2,scid=1; ...
        const int32_t t = (seq_idx / 2) + 1; // 1-based symbol index per 3GPP spec
        const int32_t n_scid = seq_idx % 2;

        // Compute c_init using OFDM symbol index t (1-based per 3GPP)
        const int32_t c_init_val = compute_c_init(slot_number, n_t, n_dmrs_id, t, n_scid);

        // Generate 32 bits of DMRS sequence
        const uint32_t gold_index = chunk_in_seq;
        const uint32_t gold_chunk = gold32(c_init_val, gold_index << 5);

        // Calculate output offset for 3D binary layout: (n_t, 2, sequence_length)
        // t is 1-based for c_init, but output array uses 0-based indexing
        const int32_t binary_offset = ((t - 1) * 2 + n_scid) * sequence_length + chunk_in_seq * 32;

        // Write 32 bits to outputs, handling the last chunk which might be partial
        const int32_t bits_to_write = min(32, sequence_length - chunk_in_seq * 32);

        for (int32_t bit = 0; bit < bits_to_write; ++bit) {
            const int32_t seq_pos = chunk_in_seq * 32 + bit;
            const int32_t bit_val = (gold_chunk >> bit) & 1;

            // Write to binary output: shape (n_t, 2, sequence_length)
            scr_seq_sym_ri_sc[binary_offset + bit] = bit_val;

            // Compute frequency scrambling (complex DMRS symbol)
            // Formula: (1 - 2*bit) / sqrt(2)
            const float scrambled_val = (1.0f - 2.0f * bit_val) * inv_sqrt2;

            // Even positions -> real part, odd positions -> imag part
            // Complex output shape: (2, n_t, 2, sequence_length//2)
            // where dim0: [0]=real, [1]=imag
            const int32_t complex_idx = seq_pos / 2; // Index into half-length array
            const bool is_even = (seq_pos % 2) == 0;

            // Calculate offset in complex output array
            // Layout: r_dmrs_ri_sym_cdm_sc[ri, t-1, n_scid, complex_idx]
            // Note: Arithmetic overflow not a concern for 5G NR use cases:
            // Max n_t=14, max sequence_length=3276 -> max offset ~92K (within int32_t)
            const int32_t ri_dim = is_even ? 0 : 1;                           // 0=real, 1=imag
            const int32_t complex_offset = ri_dim * (n_t * 2 * half_length) + // ri dimension
                                           (t - 1) * (2 * half_length) +      // t dimension
                                           n_scid * half_length +             // n_scid dimension
                                           complex_idx;                       // frequency index

            r_dmrs_ri_sym_cdm_sc[complex_offset] = scrambled_val;
        }
    }
}

/**
 * Launches DMRS sequence generation kernel
 *
 * @param[in] input_params GPU pointer to [slot_number, n_dmrs_id]
 * @param[in] sequence_length Sequence length per port
 * @param[in] n_t Number of OFDM symbols per slot
 * @param[out] r_dmrs_ri_sym_cdm_sc Complex DMRS output (GPU memory, 2 x n_t x 2 x
 * sequence_length/2)
 * @param[out] scr_seq_sym_ri_sc Binary gold sequence output (GPU memory, n_t x 2 x sequence_length)
 * @param[in] stream CUDA stream for asynchronous execution
 */
void launch_dmrs_kernel(
        const int32_t *input_params,
        const int32_t sequence_length,
        const int32_t n_t,
        float *r_dmrs_ri_sym_cdm_sc,
        int32_t *scr_seq_sym_ri_sc,
        cudaStream_t stream) {

    // Validate input parameters
    if (input_params == nullptr) {
        throw std::invalid_argument("DMRS input_params must not be null");
    }
    if (r_dmrs_ri_sym_cdm_sc == nullptr) {
        throw std::invalid_argument("DMRS r_dmrs_ri_sym_cdm_sc must not be null");
    }
    if (scr_seq_sym_ri_sc == nullptr) {
        throw std::invalid_argument("DMRS scr_seq_sym_ri_sc must not be null");
    }
    if (sequence_length <= 0) {
        throw std::invalid_argument("DMRS sequence length must be positive");
    }
    if (sequence_length % 2 != 0) {
        throw std::invalid_argument("DMRS sequence length must be even");
    }
    if (n_t <= 0) {
        throw std::invalid_argument("DMRS n_t must be positive");
    }

    // Configure kernel launch parameters
    // Total sequences: n_t * 2 (for n_scid=0,1)
    const int32_t total_sequences = n_t * 2;
    const int32_t chunks_per_sequence = (sequence_length + 31) / 32;
    const int32_t total_elements = chunks_per_sequence * total_sequences;

    // Use 256 threads per block
    constexpr int32_t threads_per_block = 256;
    const int32_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    dim3 block(threads_per_block);
    dim3 grid(num_blocks);

    dmrs_kernel<<<grid, block, 0, stream>>>(
            input_params, sequence_length, n_t, r_dmrs_ri_sym_cdm_sc, scr_seq_sym_ri_sc);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ran::trt_plugin
