Installation
============

This guide walks you through setting up your host environment for the Aerial Framework.

Environments
------------

The Aerial Framework supports two installation environments depending on your use case:

**Development Environment** (GPU only)
   For developing code, running tests and simulations, and profiling GPU runtime performance.

**Runtime Environment** (GPU + NIC)
   For testing with real-time radio and fronthaul emulation. Requires an NVIDIA GPU,
   NIC, and real-time system configuration including CPU isolation, low-latency kernel,
   and PTP synchronization.

.. note::

   Most developers start with **Development Environment**. Only proceed with **Runtime Environment**
   if the goal is to test fronthaul functionality with a radio unit emulator.

Prerequisites (Both Environments)
---------------------------------

System Requirements
^^^^^^^^^^^^^^^^^^^

* Ubuntu 22.04 or later
* NVIDIA GPU with compute capability >= 8.0

Software Dependencies
^^^^^^^^^^^^^^^^^^^^^

Install required tools:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y python3 python-is-python3 git git-lfs curl

Docker and Container Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   If you already have Docker CE and NVIDIA Container Toolkit installed, skip the Docker and
   NVIDIA Container Toolkit sections below.

The Aerial Framework requires Docker and NVIDIA Container Toolkit for the containerized
development environment. See :doc:`docker_and_nvidia_setup` for detailed installation instructions.

Development Environment Setup
-----------------------------

1. Clone the Repository
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/NVIDIA/aerial-framework.git
   cd aerial-framework

2. Build and Test
^^^^^^^^^^^^^^^^^

Follow the :doc:`../tutorials/index` to set up the containerized development environment,
build the project, and run tests. Start with :ref:`running-tutorials`.

Runtime Environment Setup (Additional Requirements)
---------------------------------------------------

To test fronthaul functionality with a radio unit (RU) emulator, follow all configuration steps in the
`NVIDIA Aerial CUDA-Accelerated RAN Installation Guide for Grace Hopper
<https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/install_guide/installing_tools_gh.html>`_.

.. note::

   After completing the installation steps, connect the two ports on the same BF3 NIC
   with at least a 100 GbE direct attach copper (DAC) cable for loopback testing.

**PTP/PHC2SYS Clock Synchronization**

It is recommended to use ports on the same NIC for loopback. However, if using ports from
**different NICs**, both NICs must be synchronized to the system clock.

To verify synchronization when using multiple NICs:

.. code-block:: bash

   systemctl status 'phc2sys*'

You should see two active services (one for each NIC). If only one service is running, configure
the second NIC following the ACAR installation guide. See :doc:`../developer_guide/real_time_apps`
for detailed clock synchronization verification.

.. toctree::
   :hidden:

   docker_and_nvidia_setup

