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

"""AWGN channel."""

import numpy as np
from numpy.typing import NDArray


def awgn(H: NDArray[np.complex64], snr_db: float) -> NDArray[np.complex64]:
    """Add AWGN to channel.

    Args:
        H: Channel with shape (n_sc, n_sym, n_ant)
        snr_db: SNR in dB

    Returns
    -------
        Noisy channel
    """
    # Calculate noise power
    signal_power = np.mean(np.abs(H) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate complex noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))

    return H + noise
