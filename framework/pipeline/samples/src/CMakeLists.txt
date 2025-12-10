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

# cmake-format: off
# Sample Pipeline - Demonstration of IPipeline/IModule pattern
#
# Components:
# - SampleModuleA: TensorRT-based element-wise addition
# - SampleModuleB: CUDA kernel-based ReLU activation
# - SamplePipeline: Pipeline connecting both modules
# cmake-format: on

add_library(
    sample-pipeline STATIC
    sample_module_a.cpp sample_module_b.cpp sample_module_b_kernel.cu sample_pipeline.cpp
    sample_module_factories.cpp sample_pipeline_factory.cpp)

target_include_directories(sample-pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sample-pipeline PUBLIC framework::utils framework::memory framework::tensor
                                             framework::tensorrt framework::pipeline rt_log)

target_link_cuda(sample-pipeline)

# Add alias for consistent naming
add_library(pipelines::sample ALIAS sample-pipeline)
