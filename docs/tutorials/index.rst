Tutorials
=========

Learn how to develop with the Aerial Framework using hands-on Jupyter notebook tutorials.

.. _running-tutorials:

Running Tutorials
--------------------------

**Step 1: Setup Container (First Time Only)**

From the top-level ``aerial-framework`` directory, configure and pull/build the
Docker container (first time may take a few minutes):

.. code-block:: bash

   bash container/setup_container.sh

The setup script will:

* Check Docker and GPU requirements (compute capability >= 8.0)
* Detect optional networking configurations (InfiniBand, VFIO, GDRCopy, Hugepages)
* Create ``.env`` file with auto-detected settings
* Pull or build container image

If networking devices are missing, you'll see a warning but the script continues.
The Fronthaul and DPDK/DOCA tests are optional and require NIC hardware.

**Step 2: Start Container**

Stop any existing container with the same name, then start a new container in the background:

.. code-block:: bash

   docker stop aerial-framework-base-$USER || true # ignore error if container not already running
   docker compose -f container/compose.yaml run -d --rm --name aerial-framework-base-$USER aerial-framework-base

**Step 3: Convert Notebooks**

Convert Python source files to notebooks:

.. code-block:: bash

   docker exec aerial-framework-base-$USER bash -c "uv run ./scripts/setup_python_env.py jupytext_convert docs"

**Option 1: VS Code with Dev Containers (Recommended)**

.. important::
   In step 5, you must open ``/opt/nvidia/aerial-framework/docs``, not ``/opt/nvidia/aerial-framework/``.
   In step 7, select the **"framework-docs"** notebook kernel. Selecting the wrong kernel
   and venv will cause ``ModuleNotFoundError`` when running notebooks.

1. Install the "Dev Containers" extension in VS Code
2. If working on a remote machine, first connect via Remote-SSH extension
3. Press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on Mac) and select "Dev Containers: Attach to
   Running Container..."
4. Select ``aerial-framework-base-<your-username>`` from the list
5. Once attached, open the ``/opt/nvidia/aerial-framework/docs`` folder
6. Open any ``.ipynb`` file from ``tutorials/generated/``
7. Click the kernel selector (top right) and choose "framework-docs" (``.venv/bin/python``)
8. The notebooks will run using the docs environment which already has ``ipykernel`` installed

**Note:** First-time setup will download VS Code Server (first time may take a few minutes).

**Option 2: JupyterLab**

.. code-block:: bash

   # For local machine
   docker exec aerial-framework-base-$USER bash -c "uv run --directory docs jupyter-lab"
   # For remote machine (accessible over network)
   docker exec aerial-framework-base-$USER bash -c "uv run --directory docs jupyter-lab --ip='0.0.0.0' --no-browser"

JupyterLab will open in the ``docs`` directory. Navigate to ``tutorials/generated``
in the file browser to access the notebooks.

**For remote machines:** Copy the URL that shows the hostname (e.g., ``http://<remote-host>:8888/lab?token=...``) and open it in your local browser.

**Resources:**

* `Attach to a running container <https://code.visualstudio.com/docs/devcontainers/attach-container>`_ - VS Code documentation on attaching to containers
* `Working with Jupyter Notebooks in VS Code <https://code.visualstudio.com/docs/datascience/jupyter-notebooks>`_ - Guide for running Jupyter notebooks
* `Jupyter Kernel Management <https://code.visualstudio.com/docs/datascience/jupyter-kernel-management>`_ - Selecting Python interpreters for notebooks

1. Getting Started Guide
------------------------

:doc:`generated/getting_started`

* Setting up the Docker development container
* Configuring your environment
* Building the project
* Running tests

2. Reference PUSCH Receiver
---------------------------

:doc:`generated/pusch_receiver_tutorial`

* Installing the RAN Python package
* Loading test vector data
* Processing PUSCH inner receiver blocks (channel estimation, equalization, soft demapping)
* Processing PUSCH outer receiver blocks (descramble, derate, LDPC decoding, CRC)

3. MLIR-TensorRT
--------------------------

:doc:`generated/mlir_trt_tutorial`

* Defining a simple JAX function (FIR filter)
* Compiling to TensorRT
* Executing and verifying correctness

4. PUSCH Receiver Lowering
---------------------------------------------------

:doc:`generated/pusch_receiver_lowering_tutorial`

* Compiling the complete PUSCH inner receiver pipeline to TensorRT
* Executing with different backends (JAX CUDA and TensorRT)
* Benchmarking with NVIDIA Nsight Systems

5. AI Channel Filter Training
------------------------------------------------------

:doc:`generated/ai_tukey_filter_training_tutorial`

* Training a custom AI channel filter for channel estimation
* Evaluating the performance of the trained AI channel filter
* Benchmarking the performance of the trained AI channel filter
* Profiling the performance of the trained AI channel filter

6. PUSCH Channel Filter Lowering
------------------------------------------------------

:doc:`generated/pusch_channel_estimation_lowering_tutorial`

* Designing custom PUSCH channel estimation filters in JAX
* Compiling channel estimators to TensorRT engines with MLIR-TensorRT
* Testing channel filter performance with CDL datasets from Sionna
* GPU profiling and benchmark analysis with NVIDIA Nsight Systems

7. Running the Complete PUSCH Pipeline
------------------------------------------

:doc:`generated/pipeline_tutorial`

* Running a PUSCH processing pipeline with mixture of hand-written CUDA code and compiled
  TensorRT layers
* Running inner and outer PUSCH receiver blocks together
* Performance analysis and profiling
* Validating end-to-end results

.. _fronthaul-testing:

8. Fronthaul and RU Emulator Testing
-------------------------------------

:doc:`generated/fronthaul_tutorial`

* Real-time system setup with GH200 and BlueField-3 NIC
* O-RAN fronthaul C-Plane and U-Plane interfaces
* DPDK and DOCA GPUNetIO for network processing
* FAPI capture and C-Plane packet preparation
* GPU-accelerated U-Plane processing with order kernel
* Real-time task scheduling with timed triggers
* Running fronthaul integration tests

9. Top Level PHY RAN Application
----------------------------------

:doc:`generated/phy_ran_app_tutorial`

* Integrating fronthaul with PUSCH processing
* Testing with MAC and RU emulators
* Performance tuning and optimization



.. toctree::
   :maxdepth: 1
   :hidden:

   generated/getting_started
   generated/pusch_receiver_tutorial
   generated/mlir_trt_tutorial
   generated/pusch_receiver_lowering_tutorial
   generated/ai_tukey_filter_training_tutorial
   generated/pusch_channel_estimation_lowering_tutorial
   generated/pipeline_tutorial
   generated/fronthaul_tutorial
   generated/phy_ran_app_tutorial
