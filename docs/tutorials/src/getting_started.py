# %% [raw] tags=["remove-cell"]
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

# %% [markdown]
# # 1. Getting Started
#
# This guide walks you through building and testing the Aerial Framework inside a
# containerized development environment.
#
# **Steps:** Verify Docker/GPU → Configure CMake → Build → Test
#
# **Prerequisites:** Container setup complete (see tutorials index), NVIDIA GPU with drivers
#
# **Time:** ~10-15 minutes for first-time build

# %% [markdown]
# ## Step 1: Setup and Verify Docker
#
# Import dependencies and verify Docker installation.
# Host-level checks are skipped if already running inside the container.

# %%
import os
import subprocess
import sys

# Import shared tutorial utilities from tutorial_utils.py (in the same directory)
# Contains helper functions for Docker container interaction and project navigation
from tutorial_utils import (
    check_network_devices,
    get_project_root,
    is_running_in_docker,
    run_container_command,
    show_output,
)

# %% [markdown]
# Check if we are running in a Docker container and verify GPU access.

# %%
IN_DOCKER = is_running_in_docker()

if IN_DOCKER:
    print("✅ Running inside Docker container")
else:
    print("Running on host - will use Docker container for builds")

# %% [markdown]
# Verify NVIDIA GPU is accessible for CUDA operations.

# %%
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
if result.returncode != 0:
    print("⚠️  nvidia-smi unavailable - GPU support may not work")
else:
    print("✅ NVIDIA GPU detected")
    print("\n".join(result.stdout.split("\n")[:25]))

# %%
PROJECT_ROOT = get_project_root()
CONTAINER_NAME = f"aerial-framework-base-{os.environ.get('USER', 'default')}"
print("✅ Step 1 complete: Environment setup and Docker verification finished")

# %% [markdown]
# ## Step 2: Configure Build
#
# Configure CMake with a preset (see available presets with `cmake --list-presets`).
# This example uses `gcc-release`. Clang-tidy is disabled to speed up the build.
#
# **Note:** This cell displays only the last few lines of output.
# If configuration fails, see the Troubleshooting section below for instructions
# on running commands manually in the container to view full logs.

# %%
preset = "gcc-release"
print(f"Configuring {preset}...")

cmd = f"cmake --preset {preset} -DENABLE_CLANG_TIDY=OFF -DENABLE_IWYU=OFF"
result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

print("✅ CMake configured" if result.returncode == 0 else "❌ Configuration failed")
show_output(result)
if result.returncode != 0:
    sys.exit(1)
print("✅ Step 2 complete: Build configured")

# %% [markdown]
# ## Step 3: Build Project
#
# Compile C++ code, CUDA kernels, and tests. Clean build takes a few minutes;
# incremental builds are faster. Artifacts go to `out/build/<preset>/`.
#
# **Note:** This cell displays only the last few lines of output.
# If build fails, see the Troubleshooting section below for instructions
# on running commands manually in the container to view full logs.

# %%
print(f"Building {preset} (first time may take a few minutes)...")

cmd = f"cmake --build out/build/{preset}"
result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

print("✅ Build successful" if result.returncode == 0 else "❌ Build failed")
show_output(result)
if result.returncode != 0:
    sys.exit(1)
print("✅ Step 3 complete: Project built successfully")

# %% [markdown]
# ## Step 4: Run Tests Without NIC (Optional)
#
# Run tests that don't require NIC hardware.
# These tests verify core functionality using GPU only.
#
# **Note:** This cell displays only the last few lines of output.
# If tests fail, see the Troubleshooting section below for instructions
# on running tests manually in the container to view full logs.

# %%
if os.environ.get("SKIP_NOTEBOOK_CTESTS", "").lower() not in ("1", "true", "yes"):
    print("Running tests without NIC requirements (may take a few minutes)...")

    cmd = f"ctest --preset {preset} -LE requires_nic"
    result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)

    print("✅ All tests passed!" if result.returncode == 0 else "⚠️  Some tests failed")
    show_output(result, lines=10)
    print("✅ Step 4 complete: Tests without NIC executed")
else:
    print("⏭️  Skipping ctests (SKIP_NOTEBOOK_CTESTS set)")
    print("✅ Step 4 complete: Tests skipped (SKIP_NOTEBOOK_CTESTS set)")

# %% [markdown]
# ## Step 5: Run Tests With NIC (Optional)
#
# Run networking and fronthaul tests if you have a DOCA GPUNetIO capable NIC
# configured in loopback mode (two ports connected via cable).
#
# These tests require networking devices (DEV_VFIO, DEV_INFINIBAND,
# DEV_GDRDRV, DEV_HUGEPAGES) in `.env`.
# Tests will be skipped if devices are not available.
#
# **Note:** This cell displays only the last few lines of output.
# If tests fail, see the Troubleshooting section below for instructions
# on running tests manually in the container to view full logs.

# %%
if os.environ.get("SKIP_NOTEBOOK_CTESTS", "").lower() not in ("1", "true", "yes"):
    has_networking = check_network_devices(CONTAINER_NAME)

    if has_networking:
        print("Running tests with NIC requirements...")

        cmd = f"ctest --preset {preset} -L requires_nic"
        result = run_container_command(cmd, CONTAINER_NAME, cwd=PROJECT_ROOT)
        print(
            "✅ NIC tests passed!"
            if result.returncode == 0
            else "⚠️  Some NIC tests failed (may need hardware)"
        )
        show_output(result, lines=10)
        print("✅ Step 5 complete: NIC tests executed")
else:
    print("⏭️  Skipping NIC ctests (SKIP_NOTEBOOK_CTESTS set)")
    print("✅ Step 5 complete: NIC tests skipped (SKIP_NOTEBOOK_CTESTS set)")

# %% [markdown]
# ## Next Steps
#
# **Development workflow:**
# - Start/restart container:
#   ```
#   docker stop aerial-framework-base-$USER
#   docker compose -f container/compose.yaml run -d --rm --name \
#     aerial-framework-base-$USER aerial-framework-base
#   ```
# - Enter container with login shell (shows banner):
#   `docker exec -it aerial-framework-base-$USER bash -l`
# - Edit code on host, rebuild: `cmake --build out/build/<preset>`
# - Run tests: `ctest --preset <preset> -R <pattern>`
#
# **Explore:** `framework/` (core libraries), `ran/`, `tests/`, `docs/`
#
# **Resources:** `README.md`, `cmake --list-presets`
#
# ## Troubleshooting
#
# **Setup Script Issues:**
# - **Docker not found:** Install Docker: https://docs.docker.com/engine/install/ubuntu/
# - **Docker daemon not running:** Start with `sudo systemctl start docker`
# - **NVIDIA container toolkit missing:** Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# - **GPU compute capability < 8.0:** Aerial Framework requires compute capability 8.0+
# - **Image pull fails:** Script will automatically build locally (takes 15-20 minutes first time)
#
# **Container Issues:**
# - **Container won't start:** Check `systemctl status docker`, verify `.env` USER_ID/GROUP_ID
# - **GPU unavailable:** Test `nvidia-smi`, check NVIDIA_VISIBLE_DEVICES in `.env`,
#   restart Docker daemon
# - **NIC tests skipped:** Ensure networking hardware is available on host;
#   setup script auto-detects devices
# - **Rebuild container:** `cd container/ && docker compose build --no-cache`
#
# **Build Issues:**
# - **Build fails:** Need 50GB+ space; clean with `rm -rf out/build/<preset>` or
#   `docker image prune -a`
# - **Permission errors:** Match USER_ID/GROUP_ID in `.env` with `id -u`/`id -g`;
#   fix: `sudo chown -R $USER:$USER out/`
# - **Configuration or build fails:** If Steps 4 or 5 fail, enter the container to run commands
#   manually and view complete logs:
#   - Enter container: `docker exec -it aerial-framework-base-$USER bash -l`
#   - Configure: `cmake --preset <preset> -DENABLE_CLANG_TIDY=OFF -DENABLE_IWYU=OFF`
#   - Build: `cmake --build out/build/<preset>`
# - **Tests fail:** Check GPU with `nvidia-smi`; run verbose:
#   `ctest --preset <preset> --verbose --rerun-failed`
# - **View full build/test output:** If the build or tests fail in Steps 4-7, enter the container to
#   run commands manually and view complete logs:
#   - Enter container: `docker exec -it aerial-framework-base-$USER bash -l`
#   - Run tests without NIC: `ctest --preset <preset> -LE requires_nic --verbose`
#   - Run tests with NIC: `ctest --preset <preset> -L requires_nic --verbose`
#   - CTest flags: `-L <label>` runs only tests with the specified label,
#     `-LE <label>` excludes tests with the label (e.g., `requires_nic` for NIC hardware tests)
#
# See `README.md` for more details.
