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

"""Orthogonal Cover Code (OCC) helper utilities.

Provides compact generators for the frequency-domain OCC (fOCC) pattern
across DMRS tones and the time-domain OCC (tOCC) sign vector across DMRS
symbols. These helpers centralize the alternating +/- patterns used in
multiple PHY modules.
"""

import numpy as np

from ran.types import ComplexArrayNP, ComplexNP


def focc_dmrs(n_dmrs_tones: int) -> ComplexArrayNP:
    """Generate fOCC pattern over DMRS tones: +1, -1, +1, -1, ...

    Args:
        n_dmrs_tones: Number of DMRS tones (typically 6 * n_prb)

    Returns
    -------
        Complex vector of shape (n_dmrs_tones,) with alternating +/-1
    """
    focc: ComplexArrayNP = np.ones(n_dmrs_tones, dtype=ComplexNP)
    focc[1::2] = -1
    return focc


def tocc_dmrs(n_dmrs_symbols: int) -> ComplexArrayNP:
    """Generate tOCC pattern across DMRS symbols: +1, -1, +1, -1, ...

    Args:
        n_dmrs_symbols: Number of DMRS symbols in the slot/group

    Returns
    -------
        Complex vector of shape (n_dmrs_symbols,) with alternating +/-1
    """
    tocc: ComplexArrayNP = np.ones(n_dmrs_symbols, dtype=ComplexNP)
    if n_dmrs_symbols > 1:
        tocc[1::2] = -1
    return tocc


__all__ = ["focc_dmrs", "tocc_dmrs"]
