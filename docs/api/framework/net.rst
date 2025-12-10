Net
===

Overview
--------

The Net library provides a high-performance network communication framework
built on top of `DPDK <https://www.dpdk.org/>`_ and `DOCA <https://developer.nvidia.com/doca>`_.
It supports CPU-based packet processing with DPDK and GPU-accelerated networking with
DOCA GPUNetIO, which enables GPU-direct zero-copy packet processing.

Key Features
~~~~~~~~~~~~

-  **DPDK Integration**: Efficient packet processing using DPDK for CPU-based operations
-  **DOCA GPUNetIO**: GPU-direct networking with kernel-initiated operations
-  **RDMA Support**: Remote Direct Memory Access for high-bandwidth, low-latency transfers
-  **GPUDirect RDMA**: Direct data transfers between NIC and GPU memory without CPU staging
-  **Queue Management**: Flexible TX/RX queue configuration for both DPDK and DOCA
-  **Memory Pool Management**: Efficient buffer management with host-pinned memory support

GPU-Accelerated Networking
----------------------------------------

The Net library uses **DOCA GPUNetIO** for direct packet transfer between NIC and GPU memory,
and **GDRCopy** for efficient CPU access to GPU memory when needed for control operations.

DOCA GPUNetIO
~~~~~~~~~~~~~

`DOCA GPUNetIO <https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html>`_ enables
real-time GPU processing for network packets by removing the CPU from the critical path.
Instead of requiring CPU coordination for packet reception via GPUDirect RDMA and kernel
notification, DOCA GPUNetIO allows CUDA kernels to control Ethernet and RDMA communications
directly, sending and receiving packets with precise timing control.

GDRCopy
~~~~~~~

`GDRCopy <https://github.com/NVIDIA/gdrcopy>`_ is a low-latency GPU memory copy library
based on NVIDIA GPUDirect RDMA technology. It allows the CPU to directly access GPU
memory.

GDRCopy provides low-latency CPU access to GPU memory without kernel driver overhead,
avoiding the costs of standard ``cudaMemcpy`` operations. The Net library uses it for:

- Allocating GPU memory buffers accessible from CPU
- CPU initializing GPU queue structures before launching kernels
- CPU reading packet counters and status from GPU memory
- CPU-GPU synchronization primitives

This enables efficient control plane operations while keeping data plane operations
GPU-centric for maximum performance.

Quick Start
-----------

.. _1-include-required-headers-net:

1. Include Required Headers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/samples/net_sender.cpp
   :language: cpp
   :start-after: example-begin net-sender-includes-1
   :end-before: example-end net-sender-includes-1
   :dedent: 0

.. _2-configure-environment:

2. Configure the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create configuration for DPDK, NIC, and queues:

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin basic-environment-config-1
   :end-before: example-end basic-environment-config-1
   :dedent: 4

.. _3-configure-tx-queue:

3. Configure DOCA TX Queue
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure DOCA TX queue for GPU-accelerated sending:

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin doca-tx-queue-config-1
   :end-before: example-end doca-tx-queue-config-1
   :dedent: 4

.. _4-configure-rx-queue:

4. Configure DOCA RX Queue
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure DOCA RX queue for GPU-accelerated receiving:

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin doca-rx-queue-config-1
   :end-before: example-end doca-rx-queue-config-1
   :dedent: 4

.. _5-initialize-and-use-environment:

5. Initialize and Use the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin complete-environment-1
   :end-before: example-end complete-environment-1
   :dedent: 4

Environment Management
----------------------

The ``Env`` class provides RAII-based management of the entire networking environment.

Environment Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The environment performs these initialization steps:

1. Validates CUDA device count and GPU device ID
2. Validates NIC availability and configuration
3. Initializes DPDK EAL (Environment Abstraction Layer)
4. Creates and initializes GPU device
5. Creates and initializes NIC with configured queues

.. important::
   The environment can only be initialized **once** because DPDK EAL
   initialization can only be called one time per application. Create the ``Env`` object
   at application startup and reuse it throughout the application lifetime.

After configuration, initialize and use the environment:

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin complete-environment-2
   :end-before: example-end complete-environment-2
   :dedent: 4

CPU-Only Mode
~~~~~~~~~~~~~

For CPU-based packet processing without GPU acceleration:

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin cpu-only-config-1
   :end-before: example-end cpu-only-config-1
   :dedent: 4

Discovering Available NICs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin discover-nics-1
   :end-before: example-end discover-nics-1
   :dedent: 4

NIC Management
--------------

The ``Nic`` class manages network interface cards with automatic resource cleanup.

NIC Information
~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin nic-information-1
   :end-before: example-end nic-information-1
   :dedent: 4

Queue Access
~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin queue-access-1
   :end-before: example-end queue-access-1
   :dedent: 4

DOCA TX Queue Configuration
----------------------------

DOCA TX queues enable GPU-direct packet transmission using DOCA GPUNetIO.

Basic TX Queue Setup
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin doca-tx-queue-config-1
   :end-before: example-end doca-tx-queue-config-1
   :dedent: 4

TX Queue with VLAN Tagging
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin doca-tx-queue-vlan-1
   :end-before: example-end doca-tx-queue-vlan-1
   :dedent: 4

Using TX Queue in CUDA Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the sender sample application, showing how to use DOCA TX queue in GPU kernels:

.. literalinclude:: ../../../framework/net/samples/net_sender.cpp
   :language: cpp
   :start-after: example-begin net-sender-txq-usage-1
   :end-before: example-end net-sender-txq-usage-1
   :dedent: 12

DOCA RX Queue Configuration
----------------------------

DOCA RX queues enable GPU-direct packet reception with hardware flow filtering
using DOCA GPUNetIO.

Hardware Flow Filtering
~~~~~~~~~~~~~~~~~~~~~~~~

RX queues use hardware flow steering rules to filter incoming packets directly at the NIC level.
The filtering is based on:

- **MAC Address** (``sender_mac_addr``): Filters packets from a specific source MAC address
- **EtherType** (``ether_type``): Filters packets with a specific Ethernet protocol type
  (e.g., ``0x88b5`` for custom protocols)
- **VLAN Tag** (``vlan_tci``, optional): Filters packets with a specific VLAN ID when VLAN
  tagging is used

The NIC hardware performs the filtering before packets reach the GPU memory, eliminating
unnecessary packet copies and processing.

Basic RX Queue Setup
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin doca-rx-queue-config-1
   :end-before: example-end doca-rx-queue-config-1
   :dedent: 4

RX Queue with VLAN Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin doca-rx-queue-vlan-1
   :end-before: example-end doca-rx-queue-vlan-1
   :dedent: 4

Using RX Queue in CUDA Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the receiver sample application, showing how to use DOCA RX queue in GPU kernels:

.. literalinclude:: ../../../framework/net/samples/net_receiver.cpp
   :language: cpp
   :start-after: example-begin net-receiver-rxq-usage-1
   :end-before: example-end net-receiver-rxq-usage-1
   :dedent: 8

DPDK Queue Configuration
-------------------------

DPDK queues provide CPU-based packet processing without GPU acceleration.

DPDK TX Queue Setup
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin dpdk-tx-queue-config-1
   :end-before: example-end dpdk-tx-queue-config-1
   :dedent: 4

Sending Packets with DPDK
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the sender sample application:

.. literalinclude:: ../../../framework/net/samples/net_samples.cpp
   :language: cpp
   :start-after: example-begin net-samples-dpdk-send-1
   :end-before: example-end net-samples-dpdk-send-1
   :dedent: 4

Memory Pool Management
----------------------

Mempools manage packet buffer allocation for DPDK operations.

Mempool Configuration
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin mempool-config-1
   :end-before: example-end mempool-config-1
   :dedent: 4

Host-Pinned Memory
~~~~~~~~~~~~~~~~~~

For improved performance with GPU operations, use host-pinned memory. When combined
with GPU memory buffers, GDRCopy enables efficient direct CPU access to GPU memory:

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin mempool-host-pinned-1
   :end-before: example-end mempool-host-pinned-1
   :dedent: 4

Multiple Mempools
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin multiple-mempools-1
   :end-before: example-end multiple-mempools-1
   :dedent: 4

MAC Address Handling
--------------------

Type-Safe MAC Addresses
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/net/tests/net_sample_tests.cpp
   :language: cpp
   :start-after: example-begin mac-address-operations-1
   :end-before: example-end mac-address-operations-1
   :dedent: 4

--------------

For more examples, see:

-  ``framework/net/samples/net_sender.cpp`` - GPU-accelerated packet sender
-  ``framework/net/samples/net_receiver.cpp`` - GPU-accelerated packet receiver
-  ``framework/net/samples/net_samples.cpp`` - Common utilities and DPDK examples
-  ``framework/net/tests/net_sample_tests.cpp`` - Documentation examples
-  ``framework/net/tests/net_env_tests.cpp`` - Environment configuration validation

External Resources
------------------

-  `DOCA GPUNetIO Documentation <https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html>`_
-  `GDRCopy GitHub Repository <https://github.com/NVIDIA/gdrcopy>`_
-  `DOCA SDK <https://developer.nvidia.com/doca>`_
-  `DPDK Documentation <https://www.dpdk.org/>`_

API Reference
-------------

Complete C++ API documentation for the Net framework.

.. doxygennamespace:: framework::net
   :content-only:
   :members:
   :undoc-members:
