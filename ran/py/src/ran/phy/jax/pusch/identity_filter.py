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

from dataclasses import dataclass
from typing import Optional

from jax import Array


@dataclass(frozen=True)
class IdentityFilterConfig:
    """Configuration for identity filter.

    Frozen dataclass that is hashable for use with JAX static_argnum.

    Attributes
    ----------
    fft_size : int
        FFT size for delay domain processing.
    alpha : float
        Window selection parameter controlling aggressiveness of filtering.
        Higher values result in more aggressive filtering (shorter windows).
    tau_min : int
        Minimum allowed window length.
    tau_max_absolute : int
        Maximum allowed window length.
    delay_compensation_samples : float
        Delay compensation in samples.
    """

    fft_size: int = 2048
    alpha: float = 2.0
    tau_min: int = 0
    tau_max_absolute: int = 1024
    delay_compensation_samples: float = 50.0


def identity_filter(
    h_noisy__ri_port_dsym_rxant_dsc: Array,
    n_dmrs_sc: int,
    config: Optional[IdentityFilterConfig] = None,
) -> Array:
    """Identity filter that returns the input channel estimates unchanged.

    This filter maintains the same API signature as other filters (e.g.,
    free_energy_filter) for interchangeable use, but does not use the config
    parameters since it performs no processing.

    Parameters
    ----------
    h_noisy__ri_port_dsym_rxant_dsc : Array
        Noisy frequency-domain channel estimates with stacked real/imag,
        shape (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    n_dmrs_sc : int
        Number of DMRS subcarriers (static, compile-time constant).
    config : IdentityFilterConfig | None, optional
        Configuration for identity filter. Kept for API consistency with other
        filters but not used in this implementation.

    Returns
    -------
    Array
        Input channel estimates with shape
        (2, n_dmrs_port, n_dmrs_syms, n_rxant, n_dmrs_sc).
    """
    # Maintain config parameter for API consistency with other filters
    _ = config  # Explicitly unused

    return h_noisy__ri_port_dsym_rxant_dsc
