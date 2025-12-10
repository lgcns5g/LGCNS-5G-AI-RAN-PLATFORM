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

"""Map keys between dictionaries and trim to function signatures."""

import inspect
from typing import Any


def trim_to_signature(func: Any, data: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """Return a kwargs dict filtered to match `func`'s signature.

    - Keeps only keys that are parameters of `func` (POSITIONAL_OR_KEYWORD/KEYWORD_ONLY)
    - Does not rename or override any keys
    """
    sig = inspect.signature(func)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    return {k: v for k, v in data.items() if k in allowed}
