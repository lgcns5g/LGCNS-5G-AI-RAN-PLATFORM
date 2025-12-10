Container
=========

Pre-configured Docker environment with build tools, CUDA, and optional networking capabilities
(DPDK/DOCA) for Aerial Framework development.

Quick Start
-----------

Prerequisites
^^^^^^^^^^^^^

Before using the developer container, ensure Docker CE and NVIDIA Container Toolkit are installed
and that GPU access works inside containers.

See :doc:`../installation/docker_and_nvidia_setup` for installation and verification steps.

1) Setup Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the automated setup script to configure your environment:

.. code-block:: bash

   bash container/setup_container.sh

The script will:

- Check Docker and NVIDIA requirements (GPU compute capability >= 8.0)
- Create ``container/.env`` with auto-detected user/group IDs and workspace paths
- Auto-detect networking devices (/dev/vfio, /dev/infiniband, /dev/gdrdrv)
- Pull or build the container image if needed

Script Options
^^^^^^^^^^^^^^

.. code-block:: bash

   # Show usage
   bash container/setup_container.sh --help

   # Generate .env file only, skip all checks (useful for CI/automation)
   bash container/setup_container.sh --env-only

   # Regenerate .env file even if it exists
   bash container/setup_container.sh --overwrite-env

   # Set specific GPUs to use
   bash container/setup_container.sh --gpus=0,1

   # Regenerate .env file with all GPUs
   bash container/setup_container.sh --overwrite-env --gpus=all

.env File Reference
^^^^^^^^^^^^^^^^^^^

The setup script auto-generates ``container/.env`` with detected values. You can manually edit
this file to customize your configuration.

**User and GPU Settings:**

- ``USER_ID`` / ``GROUP_ID`` — User/group IDs for proper file permissions (auto-detected)
- ``NVIDIA_VISIBLE_DEVICES`` — Control GPU visibility: ``all``, ``none``, or specific GPUs
  (``0,1,2``)

**Workspace Paths:**

- ``DEV_WORKSPACE`` — Source code directory (auto-detected as repository root)
- ``BUILD_WORKSPACE`` — Build artifacts directory (default: ``${DEV_WORKSPACE}/out``)

**Device Paths (for DPDK/DOCA/networking features):**

- ``DEV_VFIO`` — VFIO device (``/dev/vfio/vfio`` or ``/dev/null`` if unavailable)
- ``DEV_INFINIBAND`` — InfiniBand device (``/dev/infiniband`` or ``/dev/null``)
- ``DEV_GDRDRV`` — GDRCopy device (``/dev/gdrdrv`` or ``/dev/null``)
- ``DEV_HUGEPAGES`` — Hugepages mount (``/dev/hugepages`` or ``/dev/null``)

**Container Registry:**

- ``REGISTRY`` — Container registry hostname (default: ``nvcr.io``)
- ``PROJECT`` — Registry project path (default: ``nvidia/aerial``)
- ``IMAGE_NAME`` — Image name (default: ``aerial-framework-base``)
- ``VERSION_TAG`` — Image version tag (default: ``latest``)

**SSH Agent (Optional):**

The container can forward your SSH agent for operations that require authentication.
The setup script detects ``SSH_AUTH_SOCK`` automatically.

If SSH agent is not detected, you'll see a warning during setup. This is optional and only
needed for operations requiring SSH authentication from within the container.

To set up SSH agent on your host (optional):

.. code-block:: bash

   # Add to ~/.bashrc for automatic SSH agent management
   cat >> ~/.bashrc << 'EOF'
   # Auto-start/reuse SSH agent
   [ -f ~/.ssh/agent.env ] && . ~/.ssh/agent.env >/dev/null
   if [ -z "$SSH_AGENT_PID" ] || ! ps -p "$SSH_AGENT_PID" >/dev/null 2>&1; then
       eval $(ssh-agent -s | tee ~/.ssh/agent.env) >/dev/null
   fi
   ssh-add -l >/dev/null 2>&1 || { ssh-add ~/.ssh/id_ed25519 2>/dev/null || ssh-add ~/.ssh/id_rsa 2>/dev/null; }
   EOF
   source ~/.bashrc

Start and Enter the Container
-----------------------------

.. code-block:: bash

   # Start container in background
   docker compose run -d --rm --name aerial-framework-base-$USER aerial-framework-base

   # Connect to the running container with login shell (shows banner)
   docker exec -it aerial-framework-base-$USER bash -l

   # When done, stop container
   docker stop aerial-framework-base-$USER

File Sharing
------------

Development Workspace
^^^^^^^^^^^^^^^^^^^^^

Set ``DEV_WORKSPACE`` in your ``container/.env`` file to mount your Aerial
Framework development directory:

.. code-block:: bash

   # Edit container/.env
   DEV_WORKSPACE=/path/to/your/dev/workspace

Your development workspace will be accessible at the same path inside the
container. The container sets the working directory to
``/opt/nvidia/aerial-framework``.

Build Workspace
^^^^^^^^^^^^^^^

Set ``BUILD_WORKSPACE`` in your ``container/.env`` file to use a separate
directory for build artifacts:

.. code-block:: bash

   # Edit container/.env
   BUILD_WORKSPACE=/path/to/build/directory

Example:

.. code-block:: bash

   DEV_WORKSPACE=/home/user/nfs/aerial-framework        # NFS source code
   BUILD_WORKSPACE=/home/user/aerial-framework-build    # Local build directory

The setup script sets the default ``BUILD_WORKSPACE`` to ``${DEV_WORKSPACE}/out``.

VS Code Integration
-------------------

Attach VS Code to the running container:

1. Install the "Dev Containers" extension in VS Code
2. Start the container (see :ref:`persistent-container`)
3. Press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on Mac) and select "Dev Containers: Attach to
   Running Container..."
4. Select ``aerial-framework-base-<your-username>`` from the list

See `VS Code Dev Containers documentation <https://code.visualstudio.com/docs/devcontainers/attach-container>`_ for details.

Basic Usage
-----------

Interactive Development
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Start a development session with login shell (uses settings from .env)
   docker compose run --rm --name aerial-framework-base-$USER aerial-framework-base bash -l

   # Inside the container, you can:
   whoami          # aerial
   pwd             # /opt/nvidia/aerial-framework
   nvcc --version
   nvidia-smi

.. _persistent-container:

Persistent Container
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Start container as a service (uses settings from .env)
   docker compose run -d --rm --name aerial-framework-base-$USER aerial-framework-base

   # Attach to the running container:
   docker exec -it aerial-framework-base-$USER bash -l

   # Stop when done:
   docker stop aerial-framework-base-$USER

Run Single Commands
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run one command and exit (uses settings from .env)
   docker compose run --rm --name aerial-framework-base-$USER aerial-framework-base nvidia-smi

   # Run multiple commands with login shell
   docker compose run --rm --name aerial-framework-base-$USER aerial-framework-base bash -l -c "
       ls /path/to/your/dev/workspace
       nvcc --version
   "

Common Commands
---------------

.. code-block:: bash

   # Build the container
   docker compose build

   # Pull from registry
   docker compose pull

   # Run interactively with login shell (uses .env settings)
   docker compose run --rm --name aerial-framework-base-$USER aerial-framework-base bash -l

   # Start persistent container (uses .env settings)
   docker compose run -d --rm --name aerial-framework-base-$USER aerial-framework-base
   docker exec -it aerial-framework-base-$USER bash -l

   # Check if container is running
   docker compose ps --all

   # Stop persistent container
   docker stop aerial-framework-base-$USER

   # Clean up everything
   docker system prune -f

Development Workflow Example
----------------------------

.. code-block:: bash

   # 1. Configure your environment
   bash container/setup_container.sh

   # 2. Start persistent development container
   docker compose -f container/compose.yaml run -d --rm --name aerial-framework-base-$USER aerial-framework-base

   # 3. Attach to it with login shell (starts in /opt/nvidia/aerial-framework working directory)
   docker exec -it aerial-framework-base-$USER bash -l

   # 4. Develop your code
   # Access your files at your configured DEV_WORKSPACE path

   # 5. If your connection drops or shell crashes, just reconnect:
   docker exec -it aerial-framework-base-$USER bash -l

   # When completely done
   docker stop aerial-framework-base-$USER

Troubleshooting
---------------

See :doc:`troubleshooting` for common container issues and solutions.

