# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DEVICE_MAX_CONNECTIONS=8
ENV CUDA_MODULE_LOADING=EAGER
ENV NVIDIA_DISABLE_REQUIRE=1

# Add CUDA to PATH (required for DOCA build)
ENV PATH=/usr/local/cuda/bin:$PATH

# Fix CUDA repository keyring first (before any apt operations)
# The base image has a conflicting CUDA repository configuration that needs to be fixed
# Download and install cuda-keyring directly to avoid repository metadata sync issues
# Version is pinned for stability with DOCA; override with --build-arg if needed
ARG TARGETARCH
ARG CUDA_KEYRING_VERSION=1.1-1
RUN if [ "$TARGETARCH" = "arm64" ]; then REPO_ARCH="sbsa"; else REPO_ARCH="x86_64"; fi && \
    CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${REPO_ARCH}" && \
    apt-get update && \
    apt-get install -y wget && \
    wget ${CUDA_REPO_URL}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb && \
    dpkg -i cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb && \
    rm -f cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb && \
    rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/*

# Install curl for DOCA repository setup  
RUN apt-get update && apt-get install -y curl gpg && rm -rf /var/lib/apt/lists/*

# Add DOCA repository
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then REPO_ARCH="arm64-sbsa"; else REPO_ARCH="x86_64"; fi && \
    DOCA_REPO="https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu24.04/${REPO_ARCH}" && \
    curl -fsSL ${DOCA_REPO}/GPG-KEY-Mellanox.pub | gpg --dearmor -o /usr/share/keyrings/mellanox-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/mellanox-keyring.gpg] ${DOCA_REPO} ./" > /etc/apt/sources.list.d/doca.list

# Copy package lists
COPY apt/ /tmp/apt/

# Update and install all packages from apt files
RUN apt-get update -y && \
    # Install all packages from all .txt files (strip comments and handle duplicates)
    cat /tmp/apt/*.txt | \
    sed 's/#.*//' | \
    sed '/^[[:space:]]*$/d' | \
    sort | uniq | \
    xargs -r apt-get install -y --no-install-recommends && \
    # Hold packages that have specific versions pinned
    cat /tmp/apt/*.txt | \
    sed 's/#.*//' | \
    grep '=' | \
    sed 's/=.*//' | \
    xargs -r apt-mark hold && \
    # Clean up
    rm -rf /var/lib/apt/lists/* /tmp/apt/

# Install NVIDIA MathDx (cuFFTDx) for FFT operations on CUDA GPUs
# Note: MATHDX_VERSION is the download version (25.06.1), but the extracted directory is 25.06
ARG MATHDX_VERSION=25.06.1
ENV MATHDX_ROOT=/opt/nvidia/mathdx/25.06
RUN echo "Installing MathDx package version: ${MATHDX_VERSION}" && \
    mkdir -p /opt/nvidia/mathdx && \
    curl -fsSL "https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-${MATHDX_VERSION}-cuda12.tar.gz" \
        -o /tmp/nvidia-mathdx.tar.gz && \
    tar -xzf /tmp/nvidia-mathdx.tar.gz -C /opt/nvidia/mathdx --strip-components=3 && \
    rm /tmp/nvidia-mathdx.tar.gz && \
    echo "MathDx installed to: ${MATHDX_ROOT}" && \
    ls -la ${MATHDX_ROOT}

# Create aerial user without specifying GID/UID (let system assign)
RUN addgroup aerial && \
    adduser --ingroup aerial --home /home/aerial --shell /bin/bash --disabled-password --gecos "" aerial

# Install fixuid for user ID management
RUN curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-${TARGETARCH}.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: aerial\ngroup: aerial\npaths:\n  - /home/aerial\n" > /etc/fixuid/config.yml

# Add aerial user to sudoers
RUN echo "aerial ALL = (root) NOPASSWD: ALL" >> /etc/sudoers

# Grant SETFCAP capability to setcap binary for aerial user capability management
# Required to set granular capabilities on DPDK and DOCA applications
RUN setcap cap_setfcap+eip /usr/sbin/setcap

# Create local bin directory for aerial user
RUN runuser -l aerial -c "mkdir -p /home/aerial/.local/bin"

# Add aerial user's local bin to PATH
ENV PATH="/home/aerial/.local/bin:$PATH"

# Install uv package manager for aerial user
RUN runuser -l aerial -c "curl -LsSf https://astral.sh/uv/install.sh | sh"

# Add banner script for interactive sessions
COPY --chmod=+x banner.sh /etc/profile.d/banner.sh

# Switch to aerial user
USER aerial
WORKDIR /home/aerial

# Set up entrypoint and default command
ENTRYPOINT ["fixuid"]
CMD ["bash", "-l"]
