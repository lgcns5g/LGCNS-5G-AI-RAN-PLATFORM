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

# Colors
G='\033[0;32m' Y='\033[0;33m' R='\033[0;31m' C='\033[0;36m' NC='\033[0m'

# Label width for consistent alignment
W=18

echo -e "\n${C}Entered Aerial Framework Container${NC}\n"

# Container Configuration
echo -e "${C}Container Configuration:${NC}"
[ -n "${USER_ID}" ] && printf "  %-${W}s${G}%s${NC}\n" "User ID:" "${USER_ID}" || printf "  %-${W}s${R}%s${NC}\n" "User ID:" "<not set>"
[ -n "${GROUP_ID}" ] && printf "  %-${W}s${G}%s${NC}\n" "Group ID:" "${GROUP_ID}" || printf "  %-${W}s${R}%s${NC}\n" "Group ID:" "<not set>"
[ -n "${DEV_WORKSPACE}" ] && printf "  %-${W}s${G}%s${NC}\n" "Dev workspace:" "${DEV_WORKSPACE}" || printf "  %-${W}s${R}%s${NC}\n" "Dev workspace:" "<not set>"
[ -n "${BUILD_WORKSPACE}" ] && printf "  %-${W}s${G}%s${NC}\n" "Build workspace:" "${BUILD_WORKSPACE}" || printf "  %-${W}s${R}%s${NC}\n" "Build workspace:" "<not set>"

# GPU Configuration
echo -e "\n${C}GPU Configuration:${NC}"
[ -n "${NVIDIA_VISIBLE_DEVICES}" ] && printf "  %-${W}s${G}%s${NC}\n" "Visible devices:" "${NVIDIA_VISIBLE_DEVICES}" || printf "  %-${W}s${R}%s${NC}\n" "Visible devices:" "<not set>"

# Networking Configuration
echo -e "\n${C}Networking Configuration:${NC}"
if [ -n "${DEV_VFIO}" ] && [ "${DEV_VFIO}" != "/dev/null" ]; then
    [ -e "${DEV_VFIO}" ] && printf "  %-${W}s${G}%s${NC}\n" "VFIO:" "${DEV_VFIO}" || printf "  %-${W}s${R}%s (not found)${NC}\n" "VFIO:" "${DEV_VFIO}"
else
    printf "  %-${W}s${Y}%s${NC}\n" "VFIO:" "disabled"
fi

if [ -n "${DEV_INFINIBAND}" ] && [ "${DEV_INFINIBAND}" != "/dev/null" ]; then
    [ -e "${DEV_INFINIBAND}" ] && printf "  %-${W}s${G}%s${NC}\n" "InfiniBand:" "${DEV_INFINIBAND}" || printf "  %-${W}s${R}%s (not found)${NC}\n" "InfiniBand:" "${DEV_INFINIBAND}"
else
    printf "  %-${W}s${Y}%s${NC}\n" "InfiniBand:" "disabled"
fi

if [ -n "${DEV_GDRDRV}" ] && [ "${DEV_GDRDRV}" != "/dev/null" ]; then
    [ -e "${DEV_GDRDRV}" ] && printf "  %-${W}s${G}%s${NC}\n" "GDRCopy:" "${DEV_GDRDRV}" || printf "  %-${W}s${R}%s (not found)${NC}\n" "GDRCopy:" "${DEV_GDRDRV}"
else
    printf "  %-${W}s${Y}%s${NC}\n" "GDRCopy:" "disabled"
fi

if [ -n "${DEV_HUGEPAGES}" ] && [ "${DEV_HUGEPAGES}" != "/dev/null" ]; then
    # Check if hugepages are actually allocated
    if grep -q "HugePages_Total:" /proc/meminfo 2>/dev/null && \
       [ "$(grep "HugePages_Total:" /proc/meminfo | awk '{print $2}')" -gt 0 ] 2>/dev/null; then
        printf "  %-${W}s${G}%s${NC}\n" "Hugepages:" "${DEV_HUGEPAGES}"
    else
        printf "  %-${W}s${R}%s (not allocated)${NC}\n" "Hugepages:" "${DEV_HUGEPAGES}"
    fi
else
    printf "  %-${W}s${Y}%s${NC}\n" "Hugepages:" "disabled"
fi

# SSH Configuration
echo -e "\n${C}SSH Configuration:${NC}"
[ -n "${SSH_AUTH_SOCK}" ] && [ -S "${SSH_AUTH_SOCK}" ] && printf "  %-${W}s${G}%s${NC}\n" "SSH agent:" "${SSH_AUTH_SOCK}" || \
    { [ -n "${SSH_AUTH_SOCK}" ] && printf "  %-${W}s${R}%s${NC}\n" "SSH agent:" "${SSH_AUTH_SOCK} (not found)" || printf "  %-${W}s${R}%s${NC}\n" "SSH agent:" "<not set>"; }

echo ""
