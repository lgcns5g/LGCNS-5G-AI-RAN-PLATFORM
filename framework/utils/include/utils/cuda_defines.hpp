/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FRAMEWORK_CUDA_DEFINES_HPP
#define FRAMEWORK_CUDA_DEFINES_HPP

#ifdef __CUDACC__
#define CUDA_BOTH __host__ __device__
#define CUDA_BOTH_INLINE __forceinline__ __host__ __device__
#define CUDA_INLINE __forceinline__ __device__
#else
#define CUDA_BOTH
#define CUDA_INLINE
#define CUDA_BOTH_INLINE __inline__
#endif

#endif // FRAMEWORK_CUDA_DEFINES_HPP
