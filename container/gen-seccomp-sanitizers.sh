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

# Generate a seccomp profile by starting from Docker's default for the
# local Docker Engine version, then append sanitizer-specific allows.
#
# Usage:
#   ./gen-seccomp-sanitizers.sh [OUTPUT_JSON]
#
# Default output: container/seccomp-sanitizers.json (next to this script)
#
# Scope:
# - Development only (sanitizer runs). Do not use in production.
#
# Update cadence:
# - Regenerate whenever the Docker Engine is upgraded (profile is version-specific).
# - Otherwise, re-generate periodically to pick up default hardening.
#
# Why this script:
# - Docker does not support overlaying/extending the runtime default seccomp profile.
# - To add extra allows (e.g., for sanitizers), a full profile must be supplied.
# - This script fetches the exact default for your Engine version and appends only
#   the necessary sanitizer allowances to avoid drift.

# Resolve script directory; fall back to $0 when BASH_SOURCE is unavailable
script_path="${BASH_SOURCE[0]:-$0}"
script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
out_path="${1:-"${script_dir}/seccomp-sanitizers.json"}"

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "Error: required command '$1' not found" >&2
		exit 1
	fi
}

require_cmd curl
require_cmd jq

# Discover Docker Engine server version; fallback to 'master' if unavailable.
docker_ver=""
if command -v docker >/dev/null 2>&1; then
	set +e
	docker_ver_raw=$(docker version --format '{{.Server.Version}}' 2>/dev/null)
	status=$?
	set -e
	if [ $status -eq 0 ] && [ -n "${docker_ver_raw:-}" ]; then
		# Trim build metadata and pre-release suffixes if present
		docker_ver="${docker_ver_raw%%+*}"
		docker_ver="${docker_ver%%-*}"
	fi
fi

tag="master"
if [ -n "${docker_ver}" ]; then
	tag="v${docker_ver}"
fi

tmp_default="$(mktemp -t docker-seccomp-default.XXXXXX.json)"
cleanup() { rm -f "$tmp_default" >/dev/null 2>&1 || true; }
trap cleanup EXIT

primary_url="https://raw.githubusercontent.com/moby/moby/${tag}/profiles/seccomp/default.json"
fallback_url="https://raw.githubusercontent.com/moby/moby/master/profiles/seccomp/default.json"

echo "Fetching Docker default seccomp profile from ${primary_url}" >&2
if ! curl -fsSL "$primary_url" -o "$tmp_default"; then
	echo "Primary fetch failed; falling back to ${fallback_url}" >&2
	curl -fsSL "$fallback_url" -o "$tmp_default"
fi

# Append sanitizer allowances.
# - personality: ASLR control for sanitizers
# - process_vm_readv/process_vm_writev: cross-process memory access

jq \
	'.syscalls += [
	  {
	    "comment": "Sanitizers: allow ASLR tweaks and cross-process memory access",
	    "names": ["personality", "process_vm_readv", "process_vm_writev"],
	    "action": "SCMP_ACT_ALLOW"
	  }
	]' \
	"$tmp_default" > "${out_path}.tmp"

mv -f "${out_path}.tmp" "$out_path"

echo "Wrote ${out_path}" >&2

