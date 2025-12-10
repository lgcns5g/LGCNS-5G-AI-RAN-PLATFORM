#!/usr/bin/env bash
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

set -euo pipefail -o errtrace

# Colors
G='\033[0;32m'  # Green - success
Y='\033[0;33m'  # Yellow - warnings
R='\033[0;31m'  # Red - errors
C='\033[0;36m'  # Cyan - headers
NC='\033[0m'    # No color - reset

# Script location detection
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
ENV_FILE="$SCRIPT_DIR/.env"

# Display usage information
show_help() {
    echo "Usage: setup_container.sh [OPTIONS]"
    echo ""
    echo "Automated setup script for Aerial Framework development container."
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo "  --env-only          Generate .env file only, skip all checks"
    echo "  --overwrite-env     Regenerate .env file even if it exists"
    echo "  --gpus=<devices>    Set NVIDIA_VISIBLE_DEVICES (default: all)"
    echo ""
    echo "Examples:"
    echo "  bash setup_container.sh"
    echo "  bash setup_container.sh --env-only"
    echo "  bash setup_container.sh --overwrite-env"
    echo "  bash setup_container.sh --gpus=0,1"
    echo "  bash setup_container.sh --overwrite-env --gpus=all"
}

# Check if device exists at given path
check_device_exists() {
    local device_path="$1"
    if [ -e "$device_path" ]; then
        echo "$device_path"
    else
        echo "/dev/null"
    fi
}

# Check Docker requirements
check_docker_requirements() {
    echo -e "\n${C}Checking Docker Requirements${NC}"
    
    # Check docker command exists
    if ! command -v docker &> /dev/null; then
        echo -e "${R}✗ Docker not found${NC}"
        echo -e "  Install Docker: https://docs.docker.com/engine/install/ubuntu/"
        return 1
    fi
    echo -e "${G}✓ Docker found${NC}"
    
    # Check Docker daemon running
    if ! docker run --rm hello-world &> /dev/null; then
        echo -e "${R}✗ Docker daemon not running${NC}"
        echo -e "  Start Docker: sudo systemctl start docker"
        return 1
    fi
    echo -e "${G}✓ Docker daemon running${NC}"
    
    # Check docker compose (v2 plugin style)
    if ! docker compose version &> /dev/null; then
        echo -e "${R}✗ Docker Compose plugin not found${NC}"
        echo -e "  Install: https://docs.docker.com/engine/install/ubuntu/"
        return 1
    fi
    echo -e "${G}✓ Docker Compose available${NC}"
    
    # Check nvidia-container-toolkit
    if ! docker run --rm --gpus all ubuntu nvidia-smi &> /dev/null; then
        echo -e "${R}✗ NVIDIA container toolkit not working${NC}"
        echo -e "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        echo -e "  Then restart Docker: sudo systemctl restart docker"
        return 1
    fi
    echo -e "${G}✓ NVIDIA container toolkit working${NC}"
    
    return 0
}

# Check GPU requirements
check_gpu_requirements() {
    echo -e "\n${C}Checking NVIDIA Requirements${NC}"
    
    # Check nvidia-smi exists
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${R}✗ nvidia-smi not found${NC}"
        echo -e "  Install NVIDIA drivers"
        return 1
    fi
    echo -e "${G}✓ nvidia-smi found${NC}"
    
    # Check at least one GPU detected
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    if [ -z "$gpu_count" ] || [ "$gpu_count" -eq 0 ]; then
        echo -e "${R}✗ No GPUs detected${NC}"
        return 1
    fi
    echo -e "${G}✓ Found $gpu_count GPU(s)${NC}"
    
    # Check at least one GPU has compute capability >= 8.0
    local gpu_info
    
    # Handle cases where compute_cap query is not supported by driver version
    if ! gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null) || [ -z "$gpu_info" ]; then
        echo -e "${Y}⚠️  Warning: Could not query GPU compute capability${NC}"
        echo -e "   This may occur with older drivers."
        echo -e "   Please verify manually that your GPU has compute capability >= 8.0"
        return 0
    fi
    
    local found_compatible=0
    
    while IFS= read -r line; do
        local name compute_cap major
        name=$(echo "$line" | cut -d',' -f1 | xargs)
        compute_cap=$(echo "$line" | cut -d',' -f2 | xargs)
        
        # Compare compute capability (handle both 8.0 and 8.6 formats)
        major=$(echo "$compute_cap" | cut -d'.' -f1)
        if [ "$major" -ge 8 ]; then
            echo -e "${G}✓ Compatible GPU: $name (compute capability $compute_cap)${NC}"
            found_compatible=1
        fi
    done <<< "$gpu_info"
    
    if [ "$found_compatible" -eq 0 ]; then
        echo -e "${R}✗ No GPU with compute capability >= 8.0 found${NC}"
        echo -e "  Aerial Framework requires compute capability 8.0 or higher"
        return 1
    fi
    
    return 0
}

# Check networking devices
check_networking_devices() {
    echo -e "\n${C}Checking Networking Devices${NC}"
    
    local missing_devices=()
    
    # Check device files
    local -A devices=(
        ["VFIO"]="/dev/vfio/vfio"
        ["InfiniBand"]="/dev/infiniband"
        ["GDRCopy"]="/dev/gdrdrv"
    )
    
    for name in "${!devices[@]}"; do
        if [ -e "${devices[$name]}" ]; then
            echo -e "${G}✓ $name${NC}"
        else
            echo -e "${Y}✗ $name${NC}"
            missing_devices+=("$name")
        fi
    done
    
    # Check hugepages allocation
    if grep -q "HugePages_Total:" /proc/meminfo 2>/dev/null && \
       [ "$(grep "HugePages_Total:" /proc/meminfo | awk '{print $2}')" -gt 0 ] 2>/dev/null; then
        echo -e "${G}✓ Hugepages${NC}"
    else
        echo -e "${Y}✗ Hugepages${NC}"
        missing_devices+=("Hugepages")
    fi
    
    if [ ${#missing_devices[@]} -gt 0 ]; then
        echo -e "\n${Y}Note: Some networking devices unavailable. DPDK/DOCA tests require proper hardware.${NC}"
    fi
}

# Check SSH agent
check_ssh_agent() {
    if [ -z "${SSH_AUTH_SOCK:-}" ] || [ ! -S "${SSH_AUTH_SOCK:-}" ]; then
        echo -e "\n${Y}⚠️  Optional SSH agent not detected.  Only required if you need remote access from the container.${NC}"
    fi
}

# Generate .env file
generate_env_file() {
    local overwrite_flag="$1"
    local gpu_value="$2"
    
    echo -e "\n${C}Configuring Environment${NC}"
    
    # Read VERSION file from repository root (required)
    local version_file="$REPO_ROOT/VERSION"
    
    if [ ! -f "$version_file" ]; then
        echo -e "${R}✗ VERSION file not found at $version_file${NC}"
        echo -e "  Cannot determine container version without VERSION file"
        return 1
    fi
    
    # Read version and add 'v' prefix to match container tagging convention
    local version_raw
    version_raw=$(tr -d '[:space:]' < "$version_file")
    local version_tag="v${version_raw}"
    echo -e "Using version from VERSION file: ${version_tag}"
    
    # Check if .env exists
    if [ -f "$ENV_FILE" ]; then
        # Check for version mismatch
        if grep -q "^VERSION_TAG=" "$ENV_FILE"; then
            local existing_version
            existing_version=$(grep "^VERSION_TAG=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '[:space:]')
            if [ "$existing_version" != "$version_tag" ]; then
                echo -e "${Y}⚠️  Version mismatch detected:${NC}"
                echo -e "   Current VERSION file: ${C}${version_tag}${NC}"
                echo -e "   Existing .env file:   ${C}${existing_version}${NC}"
                echo -e "   Use --overwrite-env to update .env with current VERSION"
            fi
        fi
        
        if [ "$overwrite_flag" != "true" ]; then
            echo -e "${Y}Did not update .env file. It already exists: $ENV_FILE${NC}"
            echo -e "Use --overwrite-env to regenerate it"
            return 0
        else
            echo -e "Regenerating .env file..."
        fi
    else
        echo -e "Creating .env file..."
    fi
    
    # Auto-detect values
    local user_id group_id dev_workspace build_workspace dev_vfio dev_infiniband dev_gdrdrv
    user_id=$(id -u)
    group_id=$(id -g)
    
    # Validate user/group IDs
    if ! [[ "$user_id" =~ ^[0-9]+$ ]] || ! [[ "$group_id" =~ ^[0-9]+$ ]]; then
        echo -e "${R}✗ Failed to detect valid user/group IDs${NC}"
        return 1
    fi
    
    dev_workspace="$REPO_ROOT"
    build_workspace="$REPO_ROOT/out"
    dev_vfio=$(check_device_exists /dev/vfio/vfio)
    dev_infiniband=$(check_device_exists /dev/infiniband)
    dev_gdrdrv=$(check_device_exists /dev/gdrdrv)
    
    # Check hugepages - need to be allocated (HugePages_Total > 0)
    local dev_hugepages="/dev/null"
    if grep -q "HugePages_Total:" /proc/meminfo 2>/dev/null && \
       [ "$(grep "HugePages_Total:" /proc/meminfo | awk '{print $2}')" -gt 0 ] 2>/dev/null; then
        dev_hugepages="/dev/hugepages"
    fi
    
    local nvidia_visible_devices="${gpu_value:-all}"
    
    # Registry defaults
    local registry="nvcr.io"
    local project="nvidia/aerial"
    local image_name="aerial-framework-base"
    # version_tag is set above from VERSION file
    
    # Write .env file
    cat > "$ENV_FILE" << EOF
# Auto-generated by setup_container.sh
# You can manually edit these variables as needed

# User ID mapping (required for proper file permissions)
USER_ID=$user_id
GROUP_ID=$group_id

# GPU Control - specify which GPUs to use
# On multi-GPU systems, you can change this to specific GPUs (e.g., 0,1)
# Options: all, none, or comma-separated GPU IDs (0,1,2)
NVIDIA_VISIBLE_DEVICES=$nvidia_visible_devices

# Development workspace
DEV_WORKSPACE=$dev_workspace
BUILD_WORKSPACE=$build_workspace

# Device paths (auto-detected, set to /dev/null if not available)
DEV_VFIO=$dev_vfio
DEV_INFINIBAND=$dev_infiniband
DEV_GDRDRV=$dev_gdrdrv
DEV_HUGEPAGES=$dev_hugepages

# Registry settings
REGISTRY=$registry
PROJECT=$project
IMAGE_NAME=$image_name
VERSION_TAG=$version_tag
EOF
    
    echo -e "${G}✓ Environment file created: $ENV_FILE${NC}"
    return 0
}

# Check or pull image
check_or_pull_image() {
    echo -e "\n${C}Checking Container Image${NC}"
    
    # Source .env file
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${R}✗ .env file not found${NC}"
        return 1
    fi
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    
    # Construct full image name
    # shellcheck disable=SC2153
    local full_image_name="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${VERSION_TAG}"
    
    # Try to pull the image (this checks remote and updates if newer version exists)
    echo -e "Checking for latest image: $full_image_name"
    if docker compose -f "$SCRIPT_DIR/compose.yaml" pull aerial-framework-base; then
        echo -e "${G}✓ Image is up to date${NC}"
        return 0
    fi
    
    # Pull failed, check if image exists locally
    echo -e "${Y}⚠️  Pull failed, checking for cached local image...${NC}"
    if docker images -q "$full_image_name" | grep -q .; then
        echo -e "${Y}Using cached local image: $full_image_name${NC}"
        return 0
    fi
    
    # No local image and pull failed, build locally
    echo -e "${Y}⚠️  Image not available: $full_image_name${NC}"
    echo -e "${Y}Building locally (this may take 15-20 minutes)...${NC}"
    if ! docker compose -f "$SCRIPT_DIR/compose.yaml" build aerial-framework-base; then
        echo -e "${R}✗ Image build failed${NC}"
        return 1
    fi
    echo -e "${G}✓ Image built successfully: $full_image_name${NC}"
    return 0
}

# Start container
start_container() {
    echo -e "\n${C}Starting Container${NC}"
    
    # Get container name
    local container_name="aerial-framework-base-$USER"
    
    # Check if container already running
    if docker ps -q -f name=^"${container_name}"$ | grep -q .; then
        echo -e "${G}✓ Container already running: ${container_name}${NC}"
        echo -e "\nTo enter the container:"
        echo -e "  docker exec -it ${container_name} bash -l"
        echo -e "\nTo stop the container:"
        echo -e "  docker stop ${container_name}"
        echo -e "\nTo restart the container:"
        echo -e "  docker stop ${container_name}"
        echo -e "  docker compose -f container/compose.yaml run -d --rm --name ${container_name} aerial-framework-base"
        return 0
    fi
    
    # Start container in background
    echo -e "Starting container in background..."
    if ! docker compose -f "$SCRIPT_DIR/compose.yaml" run -d --rm --name "${container_name}" aerial-framework-base; then
        echo -e "${R}✗ Failed to start container${NC}"
        return 1
    fi
    echo -e "${G}✓ Container started: ${container_name}${NC}"
    return 0
}

# Main function
main() {
    local overwrite_flag="false"
    local gpu_value=""
    local env_only_flag="false"
    
    # Parse command-line arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --help)
                show_help
                exit 0
                ;;
            --env-only)
                env_only_flag="true"
                shift
                ;;
            --overwrite-env)
                overwrite_flag="true"
                shift
                ;;
            --gpus=*)
                gpu_value="${1#*=}"
                shift
                ;;
            *)
                echo -e "${R}Error: Unknown option: $1${NC}"
                echo ""
                show_help
                exit 1
                ;;
        esac
    done
    
    # If --env-only flag is set, only generate .env file and exit
    if [ "$env_only_flag" = "true" ]; then
        generate_env_file "$overwrite_flag" "$gpu_value" || exit 1
        exit 0
    fi
    
    # Print banner
    echo -e "\n${C}Aerial Framework Container Setup${NC}"
    
    # Run setup steps
    check_docker_requirements || exit 1
    check_gpu_requirements || exit 1
    check_networking_devices
    check_ssh_agent
    generate_env_file "$overwrite_flag" "$gpu_value"
    check_or_pull_image || exit 1
    
    # Print final success message
    echo -e "\n${G}✅ Setup complete!${NC}"
    echo -e "\n${C}Next Steps:${NC}"
    echo -e "\n1. Start the container in the background:"
    echo -e "   ${C}docker compose -f container/compose.yaml run -d --rm --name aerial-framework-base-$USER aerial-framework-base${NC}"
    echo -e "\n2. Convert and Run Tutorial Notebooks:"
    echo -e "   Convert notebooks:"
    echo -e "   ${C}docker exec aerial-framework-base-$USER bash -c \"uv run ./scripts/setup_python_env.py jupytext_convert docs\"${NC}"
    echo -e "\n   Run JupyterLab (local machine):"
    echo -e "   ${C}docker exec aerial-framework-base-$USER bash -c \"uv run --directory docs jupyter-lab\"${NC}"
    echo -e "\n   Or for remote machine:"
    echo -e "   ${C}docker exec aerial-framework-base-$USER bash -c \"uv run --directory docs jupyter-lab --ip='0.0.0.0' --no-browser\"${NC}"
    echo -e "\n   Then open the displayed URL in your browser and open the notebooks in tutorials/generated."
    echo -e "\n3. Enter the container (optional, for interactive development):"
    echo -e "   ${C}docker exec -it aerial-framework-base-$USER bash -l${NC}"
    echo -e "\nRefer to the Tutorials section in the User Guide for more information."
    echo -e "\nNote: Edit container/.env to customize GPU visibility or other settings."
}

# Run main function
main "$@"

