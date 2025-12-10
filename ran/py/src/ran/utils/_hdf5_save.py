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

"""Saver for HDF5 files (compatible with `_hdf5_load`)."""

from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np


def _to_struct_complex(arr: np.ndarray) -> np.ndarray:
    """Convert complex array to structured {re, im} with transpose applied.

    Mirrors the inverse of `_hdf5_load.hdf5_load` which reconstructs complex
    arrays from a structured dataset with fields `re` and `im`, then applies
    `np.transpose` on load.
    """
    if not np.iscomplexobj(arr):  # pragma: no cover - defensive guard
        msg = "Expected complex ndarray for struct conversion"
        raise TypeError(msg)

    arr_t = np.transpose(arr)
    re = np.asarray(np.real(arr_t))
    im = np.asarray(np.imag(arr_t))
    dt = np.dtype([("re", re.dtype), ("im", im.dtype)])
    out = np.empty(arr_t.shape, dtype=dt)
    out["re"] = re
    out["im"] = im
    return out


def _save_array_to_hdf5(f: h5py.File, key: str, arr: np.ndarray) -> None:
    """Save an array to HDF5, handling complex and real cases with transpose."""
    if np.iscomplexobj(arr):
        f.create_dataset(key, data=_to_struct_complex(arr))
    else:
        f.create_dataset(key, data=np.transpose(arr))


def hdf5_save(filename: Path | str, data: dict[str, Any]) -> None:
    """Save a flat dict of numpy arrays/scalars to an HDF5 file.

    Parameters
    ----------
    filename : Path | str
        Output HDF5 file path
    data : dict[str, Any]
        Dictionary mapping keys to numpy arrays or scalars

    Notes
    -----
    Behavior matches the inverse of `hdf5_load`:

    - Real arrays are saved transposed
    - Complex arrays are saved as structured dataset with `re` and `im` fields, each already transposed
    - Python scalars are saved as 0-d datasets
    - Only top-level datasets are supported (no groups), matching the loader
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                _save_array_to_hdf5(f, key, value)
            elif np.isscalar(value):
                f.create_dataset(key, data=value)
            else:
                arr = np.asarray(value)
                _save_array_to_hdf5(f, key, arr)
