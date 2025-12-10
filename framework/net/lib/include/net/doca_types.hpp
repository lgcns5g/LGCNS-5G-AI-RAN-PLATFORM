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

#ifndef FRAMEWORK_NET_DOCA_TYPES_HPP
#define FRAMEWORK_NET_DOCA_TYPES_HPP

#include <cstdint>

#include "net/net_export.hpp"

// Forward declarations for DOCA types
struct doca_buf_arr;
struct doca_ctx;
struct doca_dev;
struct doca_eth_rxq;
struct doca_eth_txq;
struct doca_flow_pipe;
struct doca_flow_pipe_entry;
struct doca_flow_port;
struct doca_gpu;
struct doca_gpu_buf_arr;
struct doca_gpu_eth_rxq;
struct doca_gpu_eth_txq;
struct doca_gpu_semaphore;
struct doca_gpu_semaphore_gpu;
struct doca_mmap;
struct doca_pe;
struct rte_flow;

namespace framework::net {

/**
 * Semaphore configuration parameters
 */
struct NET_EXPORT DocaSemItems final {
    std::uint32_t num_items{}; //!< Number of semaphore items
    std::uint32_t item_size{}; //!< Size of custom info per item
};

/**
 * Receive queues objects
 */
struct NET_EXPORT DocaRxQParams final {
    doca_gpu *gpu_dev{}; //!< GPUNetio handler associated to queues
                         //!< (read-only after init)
    doca_dev *ddev{};    //!< DOCA device handler associated to queues
                         //!< (read-only after init)

    doca_ctx *eth_rxq_ctx{};           //!< DOCA Ethernet receive queue context
    doca_eth_rxq *eth_rxq_cpu{};       //!< DOCA Ethernet receive queue CPU handler
    doca_gpu_eth_rxq *eth_rxq_gpu{};   //!< DOCA Ethernet receive queue GPU handler
    doca_gpu_semaphore *sem_cpu{};     //!< DOCA semaphore CPU handler
    doca_gpu_semaphore_gpu *sem_gpu{}; //!< DOCA semaphore GPU handler
    bool has_sem_items{};              //!< Whether semaphore configuration is set
    DocaSemItems sem_items{};   //!< Semaphore configuration (valid only if has_sem_items is true)
    doca_mmap *pkt_buff_mmap{}; //!< DOCA mmap to receive packet with DOCA
                                //!< Ethernet queue
    void *gpu_pkt_addr{};       //!< DOCA mmap GPU memory address
    void *cpu_pkt_addr{};       //!< CPU accessible memory address
    int dmabuf_fd{};            //!< GPU memory dmabuf descriptor
    std::uint16_t dpdk_queue_idx{};         //!< DPDK queue index for flow rules
    doca_flow_port *port{};                 //!< DOCA Flow port
    doca_flow_pipe *rxq_pipe{};             //!< DOCA Flow receive pipe
    doca_flow_pipe *root_pipe{};            //!< DOCA Flow root pipe
    doca_flow_pipe_entry *root_udp_entry{}; //!< DOCA Flow root entry
    rte_flow *dpdk_flow_rule{};             //!< DPDK flow rule that needs cleanup
    std::uint32_t max_pkt_size{};           //!< Max packet size to read
    std::uint32_t max_pkt_num{};            //!< Max number of RX packets in CUDA receive kernel
    std::uint16_t dpdk_port_id{};           //!< DPDK port ID for cleanup
};

/**
 * Send queues objects
 */
struct NET_EXPORT DocaTxQParams final {
    doca_gpu *gpu_dev{}; //!< GPUNetio handler associated to queues
    doca_dev *ddev{};    //!< DOCA device handler associated to queues

    doca_ctx *eth_txq_ctx{};         //!< DOCA Ethernet send queue context
    doca_eth_txq *eth_txq_cpu{};     //!< DOCA Ethernet send queue CPU handler
    doca_gpu_eth_txq *eth_txq_gpu{}; //!< DOCA Ethernet send queue GPU handler
    doca_pe *eth_txq_pe{};           //!< DOCA progress engine for TX queue
    doca_mmap *pkt_buff_mmap{};      //!< DOCA mmap to send packet with DOCA Ethernet queue
    void *gpu_pkt_addr{};            //!< DOCA mmap GPU memory address
    int dmabuf_fd{};                 //!< GPU memory dmabuf descriptor
    int txq_id{};                    //!< TX queue ID
    doca_flow_port *port{};          //!< DOCA Flow port
    doca_buf_arr *buf_arr{};         //!< DOCA buffer array object around GPU memory buffer
    doca_gpu_buf_arr *buf_arr_gpu{}; //!< DOCA buffer array GPU handle
    std::uint32_t pkt_size{};        //!< Packet size to send
    std::uint32_t num_packets{};     //!< Number of TX packets in CUDA send kernel
                                     //!< (max depdends on GPU memory)
    std::uint32_t inflight_sends{};  //!< Number of inflight sends in queue (should
                                     //!< not exceeed descriptor count)
};

} // namespace framework::net

#endif // FRAMEWORK_NET_DOCA_TYPES_HPP
