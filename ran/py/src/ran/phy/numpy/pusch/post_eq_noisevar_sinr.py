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

"""
Post-equalization noise variance and SINR (NumPy translation).

Implements the post-eq block from detPusch.m:
- Optionally average Ree cumulatively over DMRS positions in sub-slot processing
- For each UE, average 1./Ree over allocated REs and convert to dB
"""

import numpy as np

from ran.types import FloatArrayNP, FloatNP, IntArrayNP
from ran.utils import db


def post_eq_noisevar_sinr(
    ree: FloatArrayNP,
    layer2ue: IntArrayNP,
    n_ue: int,
) -> tuple[FloatArrayNP, FloatArrayNP]:
    """
    Compute post-equalization noise variance and SINR in dB.

    Args:
        ree: Per-layer, per-tone, per-DMRS Ree over the allocated band;
                   shape (nl, n_f_alloc, n_sym)
        layer2ue: mapping from layer index to UE index (0-based), shape (nl,)
        n_ue: number of UEs

    Returns
    -------
        post_eq_noise_var_db: shape (n_sym, n_ue)
        post_eq_sinr_db: shape (n_sym, n_ue)
    """
    _, n_f, n_sym = ree.shape
    if n_f == 0:
        zeros = np.zeros((n_sym, n_ue), dtype=FloatNP)
        return zeros, zeros

    # Vectorized aggregation across tones and layers
    alloc_ree = np.maximum(np.real(ree), np.finfo(FloatNP).tiny)
    inv_snr_mean_tones = (1.0 / alloc_ree).mean(axis=1)  # (nl, n_sym)

    # Map layers -> UEs and average per UE using bincount
    nl = inv_snr_mean_tones.shape[0]
    sum_per_ue = np.zeros((n_ue, n_sym), dtype=FloatNP)
    np.add.at(sum_per_ue, layer2ue[:nl], inv_snr_mean_tones)
    counts = np.bincount(layer2ue[:nl], minlength=n_ue).astype(FloatNP)
    nonzero = counts > 0
    if np.any(nonzero):
        sum_per_ue[nonzero, :] /= counts[nonzero, None]

    post_eq_noise_var_db = -db(sum_per_ue).T  # (n_sym, n_ue)
    post_eq_sinr_db = -post_eq_noise_var_db
    return post_eq_noise_var_db, post_eq_sinr_db


__all__ = ["post_eq_noisevar_sinr"]
