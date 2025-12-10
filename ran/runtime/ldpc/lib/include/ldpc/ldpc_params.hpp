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

#ifndef RAN_LDPC_PARAMS_HPP
#define RAN_LDPC_PARAMS_HPP

#include <cstdint>
#include <iosfwd>
#include <optional>

namespace ran::ldpc {

inline constexpr float LDPC_CLAMP_VALUE = 32.0F; //!< Clamp value for LLRs
inline constexpr std::size_t LDPC_MAX_ITERATIONS =
        10; //!< Maximum number of LDPC decoding iterations
inline constexpr float LDPC_NORMALIZATION_FACTOR = 0.8125F; //!< Normalization factor
inline constexpr std::size_t LDPC_MAX_HET_CONFIGS =
        32; //!< Maximum number of heterogeneous LDPC configurations
inline constexpr std::size_t MAX_NUM_RM_LLRS_PER_CB =
        26112; //!< Maximum number of rate matching LLRs per CB

/**
 * LDPC encoding/decoding parameters container that computes derived LDPC
 * parameters.
 *
 * This class holds the fundamental LDPC encoding/decoding parameters and
 * provides direct access to all computed derived parameters such as code block
 * segmentation, LDPC base graph selection, and other encoding/decoding
 * parameters.
 */
class LdpcParams {
public:
    /**
     * Construct LDPC encoding/decoding parameters
     *
     * All derived parameters are computed automatically upon construction,
     * except: If rate_matching_length or redundancy_version are not provided,
     * num_parity_nodes is not computed.
     *
     * @param[in] transport_block_size Size of the transport block in bits without
     * CRC.
     * @param[in] code_rate Code rate (0.0 to 1.0)
     * @param[in] rate_matching_length Rate matching length. This is the number of LLRs in the rate
     * matching block.
     * @param[in] redundancy_version Redundancy version (0-3)
     */
    explicit LdpcParams(
            std::uint32_t transport_block_size,
            float code_rate,
            std::optional<std::uint32_t> rate_matching_length,
            std::optional<std::uint8_t> redundancy_version);

    /**
     * Get transport block size
     *
     * @return Transport block size in bits without CRC
     */
    [[nodiscard]] std::uint32_t transport_block_size() const { return transport_block_size_; }

    /**
     * Get code rate
     *
     * @return Code rate (0.0 to 1.0)
     */
    [[nodiscard]] float code_rate() const { return code_rate_; }

    /**
     * Get rate matching length
     *
     * @return Rate matching length. This is the number of LLRs in the rate matching block.
     */
    [[nodiscard]] std::uint32_t rate_matching_length() const { return rate_matching_length_; }

    /**
     * Get redundancy version
     *
     * @return Redundancy version (0-3)
     */
    [[nodiscard]] std::uint8_t redundancy_version() const { return redundancy_version_; }

    /**
     * Get LBRM enabled flag
     *
     * @return LBRM enabled (0 or 1) - Not supported yet
     */
    [[nodiscard]] std::uint8_t lbrm_enabled() const { return lbrm_enabled_; }

    /**
     * Get LDPC base graph
     *
     * @return LDPC base graph (1 or 2)
     */
    [[nodiscard]] std::uint8_t base_graph() const { return base_graph_; }

    /**
     * Get number of code blocks
     *
     * @return Number of code blocks, C in 38.212
     */
    [[nodiscard]] std::uint32_t num_code_blocks() const { return num_code_blocks_; }

    /**
     * Get number of information nodes
     *
     * @return Number of information nodes
     */
    [[nodiscard]] std::uint32_t num_info_nodes() const { return num_info_nodes_; }

    /**
     * Get K' value
     *
     * @return K' in 38.212
     */
    [[nodiscard]] std::uint32_t k_prime() const { return k_prime_; }

    /**
     * Get LDPC lifting size
     *
     * @return LDPC lifting size, Z in 38.212
     */
    [[nodiscard]] std::uint32_t lifting_size() const { return lifting_size_; }

    /**
     * Get number of code block info bits
     *
     * @return Number of code block info bits, K in 38.212
     */
    [[nodiscard]] std::uint32_t num_code_block_info_bits() const {
        return num_code_block_info_bits_;
    }

    /**
     * Get number of filler bits
     *
     * @return Number of filler bits, F in 38.212
     */
    [[nodiscard]] std::uint32_t num_filler_bits() const { return num_filler_bits_; }

    /**
     * Get circular buffer size
     *
     * @return Circular buffer size for rate matching
     */
    [[nodiscard]] std::uint32_t circular_buffer_size() const { return circular_buffer_size_; }

    /**
     * Get circular buffer size padded
     *
     * @return Circular buffer size for rate matching padded
     */
    [[nodiscard]] std::uint32_t circular_buffer_size_padded() const {
        return circular_buffer_size_padded_;
    }

    /**
     * Get number of parity nodes
     *
     * @return Number of parity nodes
     * @throw std::logic_error If num_parity_nodes was not computed (when rate_matching_length or
     * redundancy_version were not provided to constructor)
     */
    [[nodiscard]] std::uint32_t num_parity_nodes() const;

    // Constants for 3GPP specifications
    static constexpr std::uint32_t MAX_CODE_BLOCK_SIZE_BG1 =
            8448; //!< Maximum code block size for base graph 1
    static constexpr std::uint32_t MAX_CODE_BLOCK_SIZE_BG2 =
            3840;                                    //!< Maximum code block size for base graph 2
    static constexpr std::uint32_t CB_CRC_SIZE = 24; //!< Code block CRC size
    static constexpr std::uint32_t TB_SIZE_THRESHOLD =
            3824; //!< Transport block size threshold for CRC size selection
    static constexpr std::uint32_t TB_CRC_SIZE_LARGE = 24; //!< CRC size for large transport blocks
    static constexpr std::uint32_t TB_CRC_SIZE_SMALL = 16; //!< CRC size for small transport blocks

    // Base graph selection constants
    static constexpr float CODE_RATE_BG_THRESHOLD_1 =
            0.25F; //!< Code rate threshold 1 for base graph selection
    static constexpr float CODE_RATE_BG_THRESHOLD_2 =
            0.67F; //!< Code rate threshold 2 for base graph selection
    static constexpr std::uint32_t TB_SIZE_BG_THRESHOLD1 =
            292; //!< Transport block size threshold 1 for base graph selection
    static constexpr std::uint32_t TB_SIZE_BG_THRESHOLD2 =
            3824; //!< Transport block size threshold 2 for base graph selection

    // Additional constants for LDPC
    static constexpr std::uint32_t MIN_PARITY_NODES = 4; //!< Minimum permitted parity node count
                                                         //!< irrespective of BG (base graph)
    static constexpr std::uint32_t MAX_PARITY_NODES_BG1 =
            46; //!< Maximum number of parity nodes for base graph 1
    static constexpr std::uint32_t MAX_PARITY_NODES_BG2 =
            42; //!< Maximum number of parity nodes for base graph 2
    static constexpr std::uint32_t UNPUNCTURED_VAR_NODES_BG1 =
            66; //!< Circular buffer nodes for base graph 1
    static constexpr std::uint32_t UNPUNCTURED_VAR_NODES_BG2 =
            50; //!< Circular buffer nodes for base graph 2

    static constexpr std::uint32_t PADDING_ALIGNMENT = 8; //!< Padding alignment for circular buffer

    static constexpr std::uint32_t INFO_NODES_BG1 = 22; //!< Information nodes for base graph 1
    static constexpr std::uint32_t INFO_NODES_BG2_MAX =
            10; //!< Information nodes for base graph 2 (maximum)
    static constexpr std::uint32_t INFO_NODES_BG2_MEDIUM =
            9; //!< Information nodes for base graph 2 (medium)
    static constexpr std::uint32_t INFO_NODES_BG2_SMALL =
            8; //!< Information nodes for base graph 2 (small)
    static constexpr std::uint32_t INFO_NODES_BG2_SMALLEST =
            6; //!< Information nodes for base graph 2 (smallest)
    static constexpr std::uint32_t TB_SIZE_THRESHOLD_KB10 =
            640; //!< Transport block size threshold for Kb 10
    static constexpr std::uint32_t TB_SIZE_THRESHOLD_KB9 =
            560; //!< Transport block size threshold for Kb 9
    static constexpr std::uint32_t TB_SIZE_THRESHOLD_KB8 =
            192; //!< Transport block size threshold for Kb 8
    static constexpr std::uint32_t MAX_LIFTING_SIZE = 384; //!< Maximum possible lifting size

    // Rate matching starting position (k0) factors for different redundancy
    // versions
    static constexpr std::uint32_t RV1_START_POS_FACTOR_BG1 = 17; //!< RV1 factor for base graph 1
    static constexpr std::uint32_t RV2_START_POS_FACTOR_BG1 = 33; //!< RV2 factor for base graph 1
    static constexpr std::uint32_t RV3_START_POS_FACTOR_BG1 = 56; //!< RV3 factor for base graph 1
    static constexpr std::uint32_t RV1_START_POS_FACTOR_BG2 = 13; //!< RV1 factor for base graph 2
    static constexpr std::uint32_t RV2_START_POS_FACTOR_BG2 = 25; //!< RV2 factor for base graph 2
    static constexpr std::uint32_t RV3_START_POS_FACTOR_BG2 = 43; //!< RV3 factor for base graph 2

    // Static methods to compute derived parameters, also standalone callable
    // functions.
    /**
     * Get the LDPC base graph (1 or 2)
     *
     * @param[in] tb_size Transport block size in bits without CRC.
     * @param[in] code_rate Code rate (0.0 to 1.0)
     * @return LDPC base graph (1 or 2)
     */
    static std::uint8_t get_base_graph(std::uint32_t tb_size, float code_rate);

    /**
     * Get the transport block CRC size
     *
     * @param[in] tb_size Transport block size in bits without CRC.
     * @return Transport block CRC size
     */
    static std::uint8_t get_tb_crc_size(std::uint32_t tb_size);

    /**
     * Get the transport block size with CRC
     *
     * @param[in] tb_size Transport block size in bits without CRC.
     * @return Transport block size with CRC
     */
    static std::uint32_t get_tb_size_with_crc(std::uint32_t tb_size);

    /**
     * Get the number of code blocks
     *
     * @param[in] tb_size Transport block size in bits without CRC.
     * @param[in] base_graph LDPC base graph (1 or 2)
     * @return Number of code blocks
     * @throw std::invalid_argument if base_graph is not 1 or 2
     */
    static std::uint32_t get_num_code_blocks(std::uint32_t tb_size, std::uint8_t base_graph);

    /**
     * Get the number of information nodes
     *
     * @param[in] tb_size Transport block size in bits without CRC.
     * @param[in] base_graph LDPC base graph (1 or 2)
     * @return Number of information nodes
     * @throw std::invalid_argument if base_graph is not 1 or 2
     */
    static std::uint32_t get_num_info_nodes(std::uint32_t tb_size, std::uint8_t base_graph);

    /**
     * Get the K' value
     *
     * @param[in] tb_size Transport block size in bits without CRC.
     * @param[in] num_code_blocks Number of code blocks
     * @return K' value
     */
    static std::uint32_t get_k_prime(std::uint32_t tb_size, std::uint32_t num_code_blocks);

    /**
     * Get the LDPC lifting size
     *
     * @param[in] tb_size Transport block size in bits without CRC.
     * @param[in] base_graph LDPC base graph (1 or 2)
     * @param[in] num_code_blocks Number of code blocks
     * @param[in] num_info_nodes Number of information nodes
     * @param[in] k_prime K' value
     * @return LDPC lifting size
     * @note Parameters num_code_blocks, num_info_nodes, and k_prime are optional
     * and will be computed if not provided.
     * @note If num_code_blocks is not provided, it will be computed using
     * get_num_code_blocks().
     * @note If num_info_nodes is not provided, it will be computed using
     * get_num_info_nodes().
     * @note If k_prime is not provided, it will be computed using get_k_prime().
     * @throw std::invalid_argument If base_graph is not 1 or 2.
     */
    static std::uint32_t get_lifting_size(
            std::uint32_t tb_size,
            std::uint8_t base_graph,
            std::optional<std::uint32_t> num_code_blocks,
            std::optional<std::uint32_t> num_info_nodes,
            std::optional<std::uint32_t> k_prime);

    /**
     * Get the number of code block info bits
     *
     * @param[in] base_graph LDPC base graph (1 or 2)
     * @param[in] lifting_size LDPC lifting size
     * @return Number of code block info bits
     * @throw std::invalid_argument If base_graph is not 1 or 2.
     */
    static std::uint32_t
    get_num_code_block_info_bits(std::uint8_t base_graph, std::uint32_t lifting_size);

private:
    std::uint32_t transport_block_size_{}; //!< Transport block size in bits without CRC
    float code_rate_{};                    //!< Code rate (0.0 to 1.0)
    std::uint32_t rate_matching_length_{}; //!< Rate matching length. This is the number of LLRs in
                                           //!< the rate matching block.
    std::uint8_t redundancy_version_{};    //!< Redundancy version (0-3)
    std::uint8_t lbrm_enabled_{0};         //!< LBRM enabled (0 or 1) - Not supported yet

    // Derived parameters
    std::uint8_t base_graph_{};                     //!< LDPC base graph (1 or 2)
    std::uint32_t num_code_blocks_{};               //!< Number of code blocks, C in 38.212
    std::uint32_t num_info_nodes_{};                //!< Number of information nodes
    std::uint32_t k_prime_{};                       //!< K' in 38.212
    std::uint32_t lifting_size_{};                  //!< LDPC lifting size, Z in 38.212
    std::uint32_t num_code_block_info_bits_{};      //!< Number of code block info bits,
                                                    //!< K in 38.212
    std::uint32_t num_filler_bits_{};               //!< Number of filler bits, F in 38.212
    std::uint32_t circular_buffer_size_{};          //!< Circular buffer size for rate matching
    std::uint32_t circular_buffer_size_padded_{};   //!< Circular buffer size for
                                                    //!< rate matching padded
    std::optional<std::uint32_t> num_parity_nodes_; //!< Number of parity nodes, computed if
                                                    //!< redundancy_version and rate_matching_length
                                                    //!< are provided

    /**
     * Compute derived LDPC parameters.
     *
     * @note This method is called by the constructor.
     */
    void compute_derived_parameters();

    /**
     * Compute the number of parity nodes.
     *
     * @note This method is called by the constructor.
     */
    void compute_num_parity_nodes();
};

} // namespace ran::ldpc

#endif // RAN_LDPC_PARAMS_HPP
