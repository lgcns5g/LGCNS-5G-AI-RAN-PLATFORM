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
RSSI measurements for DMRS (NumPy translation of measureRssi.m).

Implements the MATLAB logic used in detPusch.m to compute:
- rssiDb: per-symbol, per-antenna RSSI over allocated PRBs (linear->dB)
- rssiReportedDb: aggregated RSSI across symbols and antennas (dB)

Inputs follow MATLAB naming converted to snake_case and 0-based indexing
conventions are handled internally.
"""

import numpy as np

from ran.types import ComplexArrayNP, FloatArrayNP, FloatNP
from ran.utils import db


def measure_rssi(xtf_band_dmrs: ComplexArrayNP) -> tuple[FloatArrayNP, FloatNP]:
    """
    Compute DMRS RSSI metrics.

    Args
    ----
        xtf_band_dmrs: ndarray of shape (n_f_alloc, n_dmrs, n_ant)
            PRB-band slice over the DMRS symbols for all antennas.

    Returns
    -------
        rssi_db: per-symbol, per-antenna RSSI in dB, shape (n_dmrs, n_ant)
        rssi_reported_db: aggregated RSSI across symbols and antennas (dB), shape ()
    """
    # reduce over frequency for all symbols/antennas
    p_all_ta = np.sum(np.abs(xtf_band_dmrs) ** 2, axis=0)  # shape: (n_t_dmrs,n_ant)

    # cast to float64 for db()
    avg_pwr_lin = np.asarray(p_all_ta, dtype=FloatNP)

    # per-symbol, per-antenna RSSI (dB)
    rssi_db = db(avg_pwr_lin)

    # reported RSSI: mean over symbols, sum over antennas
    rssi_reported_db = db(avg_pwr_lin.mean(axis=0).sum())

    return rssi_db, FloatNP(rssi_reported_db)


__all__ = ["measure_rssi"]
