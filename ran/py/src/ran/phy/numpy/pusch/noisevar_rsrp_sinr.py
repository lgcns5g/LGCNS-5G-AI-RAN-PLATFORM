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
Noise variance, RSRP and SINR calculators (NumPy translations of MATLAB detPusch.m).

Implements helpers corresponding to the measurement block in detPusch.m:
- noise_variance_db: compute noise variance per DMRS position (dB) from tmp_noiseVar
- rsrp_db: compute RSRP per DMRS position and UE (dB) from h_est and layer2ue
- sinr_db: compute SINR per DMRS position and UE (dB) as rsrp_db - noise_variance_db
"""

import numpy as np

from ran.types import ComplexArrayNP, FloatArrayNP, FloatNP, IntArrayNP
from ran.utils import db


def noise_variance_db(
    mean_noise_var: FloatArrayNP,
) -> FloatArrayNP:
    """
    Compute noise variance per DMRS position (dB) following detPusch.m.

    Args:
        mean_noise_var: (n_prb, n_pos) linear noise variances

    Returns
    -------
        noise variance in dB, shape (n_pos,)
    """
    # Full-slot: scalar mean over all entries, then repeat per position
    return db(np.maximum(np.mean(mean_noise_var, axis=0), np.finfo(float).tiny)) + 0.5


def rsrp_db(
    h_est: ComplexArrayNP,
    layer2ue: IntArrayNP,
    n_ue: int,
) -> FloatArrayNP:
    """
    Compute RSRP per DMRS position and UE (dB) following detPusch.m.

    Args:
        h_est: (n_f, nl, n_ant, n_pos) complex channel estimates,
            with nf: number of subcarriers in frequency (12*n_prb)
        layer2ue: (nl,) int mapping from layer index to UE index (0-based)
        n_ue: number of UEs

    Returns
    -------
        rsrp in dB, shape (n_pos, n_ue)
    """
    nf, nl, n_ant, n_pos = h_est.shape

    # Sum |H|^2 across frequency and antennas -> (nl, n_pos)
    per_layer_pos = np.abs(h_est) ** 2
    per_layer_pos = per_layer_pos.sum(axis=(0, 2)).astype(FloatNP)  # (nl, n_pos)

    # Aggregate layers -> UEs: (n_ue, n_pos) then transpose to (n_pos, n_ue)
    rsrp_lin = np.zeros((n_ue, n_pos), dtype=FloatNP)
    if n_ue > 0 and nl > 0:
        np.add.at(rsrp_lin, layer2ue, per_layer_pos)
    rsrp_lin = rsrp_lin.T

    # Normalize by (nf*nAnt)
    rsrp_lin /= nf * max(n_ant, 1)

    # Always full-slot average (subslot_proc_option assumed 0)
    mean_over_pos = rsrp_lin.mean(axis=0, keepdims=True)
    rsrp_lin = np.repeat(mean_over_pos, repeats=n_pos, axis=0)

    # Convert to dB
    return db(np.maximum(rsrp_lin, np.finfo(float).tiny))


def sinr_db(
    rsrp_db: FloatArrayNP,
    noise_var_db: FloatArrayNP,
) -> FloatArrayNP:
    """
    Compute SINR per DMRS position and UE (dB) as rsrp_db - noise_var_db.

    Args:
        rsrp_db: (n_pos, n_ue) RSRP in dB
        noise_var_db: (n_pos,) noise variance in dB

    Returns
    -------
        SINR in dB, shape (n_pos, n_ue)
    """
    return rsrp_db - noise_var_db


def noise_rsrp_sinr_db(
    mean_noise_var: FloatArrayNP,
    h_est: ComplexArrayNP,
    layer2ue: IntArrayNP,
    n_ue: int,
) -> tuple[FloatArrayNP, FloatArrayNP, FloatArrayNP]:
    """
    Compute noise variance, RSRP, and SINR per DMRS position and UE (all in dB).

    This function sequentially calls noise_variance_db, rsrp_db, and sinr_db
    to compute all three metrics in one call.

    Args:
        mean_noise_var: Noise variance estimate per DMRS position, shape (n_prb, n_pos)
        h_est: Channel estimates, shape (n_f, nl, n_ant, n_pos)
        layer2ue: Maps layers to UEs (0-indexed), shape (nl,)
        n_ue: Number of UEs

    Returns
    -------
        tuple containing:
        - noise_var_db: Noise variance in dB, shape (n_pos,)
        - rsrp_db_val: RSRP in dB, shape (n_pos, n_ue)
        - sinr_db_val: SINR in dB, shape (n_pos, n_ue)
    """
    # Compute noise variance in dB
    noise_var_db = noise_variance_db(mean_noise_var)

    # Compute RSRP in dB
    rsrp_db_val = rsrp_db(h_est, layer2ue, n_ue)

    # Compute SINR in dB
    sinr_db_val = sinr_db(rsrp_db_val, noise_var_db)

    return noise_var_db, rsrp_db_val, sinr_db_val


__all__ = [
    "noise_rsrp_sinr_db",
    "noise_variance_db",
    "rsrp_db",
    "sinr_db",
]
