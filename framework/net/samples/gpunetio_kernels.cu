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
 * @file gpunetio_kernels.cu
 * @brief Simple CUDA kernels for GPUNetIO packet sending and receiving
 */

#include <cstdint>
#include <cstdio>

#include <doca_buf.h>
#include <doca_buf_inventory.h>

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "log/rt_log_macros.hpp"
#include "net/doca_types.hpp"
#include "net/net_log.hpp"

namespace {

static constexpr auto MIN_ETHERNET_SIZE = 14U;
static constexpr uint16_t ETYPE_VLAN = 0x8100;

/**
 * Validates DocaRxQParams before launching kernel
 */
bool is_rx_params_valid(const framework::net::DocaRxQParams &rxq) {
    static constexpr std::uint32_t MIN_PROTOCOL_SIZE =
            MIN_ETHERNET_SIZE + 4U; // Ethernet + length field
    static constexpr std::uint32_t MIN_PKT_NUM = 1U;

    if (rxq.max_pkt_size < MIN_PROTOCOL_SIZE) {
        RT_LOGC_ERROR(
                framework::net::Net::NetGpu,
                "Invalid max_pkt_size ({} bytes), minimum required is {} "
                "bytes for protocol data",
                rxq.max_pkt_size,
                MIN_PROTOCOL_SIZE);
        return false;
    }

    if (rxq.max_pkt_num < MIN_PKT_NUM) {
        RT_LOGC_ERROR(
                framework::net::Net::NetGpu,
                "Invalid max_pkt_num ({}), minimum required is {}",
                rxq.max_pkt_num,
                MIN_PKT_NUM);
        return false;
    }

    return true;
}

/**
 * Simple helper to add ASCII message to packet after Ethernet header
 */
__device__ bool add_message_to_packet(uint8_t *pkt, const uint32_t pkt_size) {
    static constexpr const char message[] = "hello doca gpunetio";
    static constexpr uint32_t message_len = sizeof(message) - 1;
    const uint32_t offset = MIN_ETHERNET_SIZE; // After Ethernet header
    const uint32_t required_size =
            offset + 4 + message_len; // Ethernet header + length field + message

    // Validate packet size is large enough to hold message
    if (pkt_size < required_size) {
        printf("GPU: Error - Packet size %u too small, need at least %u bytes\n",
               pkt_size,
               required_size);
        return false;
    }

    // Write length (big-endian, byte by byte)
    pkt[offset + 0] = (message_len >> 24) & 0xFF;
    pkt[offset + 1] = (message_len >> 16) & 0xFF;
    pkt[offset + 2] = (message_len >> 8) & 0xFF;
    pkt[offset + 3] = message_len & 0xFF;

    // Write message
    for (uint32_t i = 0; i < message_len; ++i) {
        pkt[offset + 4 + i] = message[i];
    }

    printf("GPU: Added message: '%s'\n", message);
    return true;
}

/**
 * Simple helper to read and print ASCII message from packet
 */
__device__ void print_message_from_packet(const uint8_t *pkt) {
    // Adjust payload offset for optional VLAN header
    const uint16_t outer_type = static_cast<uint16_t>((pkt[12] << 8) | pkt[13]);
    const bool has_vlan = (outer_type == ETYPE_VLAN);
    const uint32_t offset = MIN_ETHERNET_SIZE + (has_vlan ? 4U : 0U);

    // Read length (big-endian, byte by byte)
    const uint32_t message_len = (pkt[offset + 0] << 24) | (pkt[offset + 1] << 16) |
                                 (pkt[offset + 2] << 8) | pkt[offset + 3];

    static constexpr auto MAX_MESSAGE_LEN = 1024U;
    if (message_len > 0 && message_len < MAX_MESSAGE_LEN) { // Simple bounds check
        char message[MAX_MESSAGE_LEN]{};
        for (uint32_t i = 0; i < message_len; ++i) {
            message[i] = pkt[offset + 4 + i];
        }
        message[message_len] = '\0';

        printf("GPU: Received message: '%s'\n", message);
    }
}

/**
 * Print Ethernet L2 header (dst/src MAC and outer EtherType)
 */
__device__ void print_l2_header(const uint8_t *pkt) {
    const uint16_t ethertype = static_cast<uint16_t>((pkt[12] << 8) | pkt[13]);
    printf("GPU: Received packet - L2[dst=%02x:%02x:%02x:%02x:%02x:%02x "
           "src=%02x:%02x:%02x:%02x:%02x:%02x type=0x%04x]\n",
           pkt[0],
           pkt[1],
           pkt[2],
           pkt[3],
           pkt[4],
           pkt[5], // Destination MAC
           pkt[6],
           pkt[7],
           pkt[8],
           pkt[9],
           pkt[10],
           pkt[11], // Source MAC
           ethertype);
}

/**
 * Parse and print L2 outer/inner EtherTypes and VLAN TCI if present
 */
__device__ void print_l2_types_and_vlan(const uint8_t *pkt) {
    const uint16_t outer_type = static_cast<uint16_t>((pkt[12] << 8) | pkt[13]);
    if (outer_type == ETYPE_VLAN) {
        const uint16_t vlan_tci = static_cast<uint16_t>((pkt[14] << 8) | pkt[15]);
        const uint16_t inner_type = static_cast<uint16_t>((pkt[16] << 8) | pkt[17]);
        printf("GPU: L2 types - outer=0x%04x (VLAN), inner=0x%04x, vlan_tci=0x%04x (vlan_id=%u)\n",
               outer_type,
               inner_type,
               vlan_tci,
               static_cast<unsigned>(vlan_tci & 0x0FFF));
    } else {
        printf("GPU: L2 types - outer=0x%04x, inner=(not set), vlan_tci=(not set)\n", outer_type);
    }
}

/**
 * Simple CUDA kernel to send a single packet
 *
 * @param[in] eth_txq_gpu DOCA Ethernet TX queue GPU handler
 * @param[in] buf_arr_gpu DOCA buffer array GPU handle
 * @param[in] pkt_size Packet size to send
 */
__global__ void send_single_packet(
        doca_gpu_eth_txq *eth_txq_gpu, doca_gpu_buf_arr *buf_arr_gpu, const uint32_t pkt_size) {
    // Only thread 0 sends the packet
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    doca_gpu_buf *buf_ptr{};
    uint32_t position{};
    uint32_t curr_position{};
    uint32_t mask_max_position{};

    // Get the first buffer from the buffer array
    doca_gpu_dev_buf_get_buf(buf_arr_gpu, 0 /* doca_gpu_buf_idx */, &buf_ptr);

    // Get buffer address to add ASCII message
    uintptr_t buf_addr{};
    doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
    uint8_t *pkt = reinterpret_cast<uint8_t *>(buf_addr);

    // Add ASCII message to packet
    if (!add_message_to_packet(pkt, pkt_size)) {
        printf("GPU: Failed to add message to packet, aborting send\n");
        return;
    }

    // Ensure packet payload is globally visible before enqueue
    __threadfence();

    // Get current queue position
    doca_gpu_dev_eth_txq_get_info(eth_txq_gpu, &curr_position, &mask_max_position);
    position = curr_position & mask_max_position;

    printf("GPU: Sending single packet of size %u at position %u\n", pkt_size, position);

    // Enqueue the packet with notification flag
    doca_gpu_dev_eth_txq_send_enqueue_weak(
            eth_txq_gpu, buf_ptr, pkt_size, position, DOCA_GPU_SEND_FLAG_NOTIFY);

    // Commit and push the packet
    doca_gpu_dev_eth_txq_commit_weak(eth_txq_gpu, 1);
    doca_gpu_dev_eth_txq_push(eth_txq_gpu);

    // Wait for completion
    uint32_t completed{};
    doca_gpu_dev_eth_txq_wait_completion(
            eth_txq_gpu, 1 /* num_sends */, DOCA_GPU_ETH_TXQ_WAIT_FLAG_B, &completed);

    printf("GPU: Successfully sent packet, completed: %u\n", completed);
}

/**
 * Simple CUDA kernel to receive a single packet and block until received
 *
 * @param[in] eth_rxq_gpu DOCA Ethernet RX queue GPU handler
 * @param[in] exit_cond Exit condition flag
 */
__global__ void receive_single_packet(doca_gpu_eth_rxq *eth_rxq_gpu, uint32_t *exit_cond) {
    // Only thread 0 receives the packet
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    doca_error_t ret{};
    doca_gpu_buf *buf_ptr{};
    uintptr_t buf_addr{};
    uint32_t rx_pkt_num{};
    uint64_t rx_buf_idx{};

    printf("GPU: Starting single packet receive, waiting for packet...\n");

    // Block until we receive one packet or exit condition is set
    while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
        static constexpr auto MAX_RX_TIMEOUT_NS = 1000'000'000ULL;
        static constexpr auto NUM_PACKETS = 1U;
        ret = doca_gpu_dev_eth_rxq_receive_block(
                eth_rxq_gpu, NUM_PACKETS, MAX_RX_TIMEOUT_NS, &rx_pkt_num, &rx_buf_idx);

        if (ret != DOCA_SUCCESS) {
            printf("GPU: Receive error %d, setting exit condition\n", ret);
            DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
            break;
        }

        if (rx_pkt_num == 0) {
            // No packet received - continue polling
            // The receive_block function already handles timeout
            continue;
        }

        if (rx_pkt_num > 0) {
            // Get the received packet
            doca_gpu_dev_eth_rxq_get_buf(eth_rxq_gpu, rx_buf_idx, &buf_ptr);
            doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);

            uint8_t *pkt = reinterpret_cast<uint8_t *>(buf_addr);
            print_l2_header(pkt);
            // Print outer/inner EtherTypes and VLAN tag info
            print_l2_types_and_vlan(pkt);

            // Print received ASCII message
            print_message_from_packet(pkt);

            printf("GPU: Successfully received single packet, exiting\n");
            break;
        }
    }
}

} // namespace

namespace framework::net {

int launch_gpunetio_sender_kernel(cudaStream_t stream, const DocaTxQParams &txq) {
    if (txq.eth_txq_gpu == nullptr || txq.buf_arr_gpu == nullptr) {
        RT_LOGC_ERROR(Net::NetGpu, "Invalid input parameters for kernel launch");
        return -1;
    }

    if (const auto result = cudaGetLastError(); result != cudaSuccess) {
        RT_LOGC_ERROR(Net::NetGpu, "Previous CUDA error detected: {}", cudaGetErrorString(result));
        return -1;
    }

    RT_LOGC_INFO(
            Net::NetGpu, "Launching single packet send kernel with packet size {}", txq.pkt_size);

    send_single_packet<<<1, 1, 0, stream>>>(txq.eth_txq_gpu, txq.buf_arr_gpu, txq.pkt_size);

    if (const auto result = cudaGetLastError(); result != cudaSuccess) {
        RT_LOGC_ERROR(Net::NetGpu, "CUDA kernel launch failed: {}", cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

int launch_gpunetio_receiver_kernel(
        cudaStream_t stream, const DocaRxQParams &rxq, uint32_t *gpu_exit_condition) {
    if (rxq.eth_rxq_gpu == nullptr || gpu_exit_condition == nullptr) {
        RT_LOGC_ERROR(Net::NetGpu, "Invalid input parameters for receiver kernel launch");
        return -1;
    }

    if (!is_rx_params_valid(rxq)) {
        RT_LOGC_ERROR(Net::NetGpu, "DocaRxQParams validation failed");
        return -1;
    }

    if (const auto result = cudaGetLastError(); result != cudaSuccess) {
        RT_LOGC_ERROR(Net::NetGpu, "Previous CUDA error detected: {}", cudaGetErrorString(result));
        return -1;
    }

    RT_LOGC_INFO(Net::NetGpu, "Launching single packet receive kernel");

    receive_single_packet<<<1, 1, 0, stream>>>(rxq.eth_rxq_gpu, gpu_exit_condition);

    if (const auto result = cudaGetLastError(); result != cudaSuccess) {
        RT_LOGC_ERROR(
                Net::NetGpu, "CUDA receiver kernel launch failed: {}", cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

} // namespace framework::net
