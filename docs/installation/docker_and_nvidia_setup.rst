Docker and NVIDIA Container Toolkit Setup
===========================================

Overview
--------

The Aerial Framework requires Docker CE and NVIDIA Container Toolkit for the containerized
development environment. Docker provides container runtime capabilities, and NVIDIA Container
Toolkit enables GPU access within containers.

Docker Installation
-------------------

The full official instructions for installing Docker CE can be found here:
`Install Docker Engine on Ubuntu <https://docs.docker.com/engine/install/ubuntu/>`_.
The following instructions are one supported way of installing Docker CE.

Installation Steps
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

NVIDIA Container Toolkit Installation
--------------------------------------

The full official instructions for installing NVIDIA Container Toolkit can be found here:
`NVIDIA Container Toolkit installation guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.
The following instructions are one supported way of installing NVIDIA Container Toolkit.

Installation Steps
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker

Post-Installation
-----------------

Add your user to the docker group:

.. code-block:: bash

   sudo usermod -aG docker $USER

.. note::

   **Log out and log back in** for docker group changes to take effect. Verify installation with:

   .. code-block:: bash

      docker run --rm hello-world
      docker compose version
      docker run --rm --gpus all ubuntu nvidia-smi

The last command should display your GPU information if the installation was successful.

