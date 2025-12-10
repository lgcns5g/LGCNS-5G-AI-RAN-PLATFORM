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

"""Module to register a custom call primitive."""

from collections.abc import Callable

from jax.extend.core import Primitive
from jax.interpreters import mlir
from jax.typing import ArrayLike


def register_custom_call_primitive(
    name: str,
    fn_lowering: Callable,
    fn_abstract: Callable,
) -> Callable:
    """Register a custom call primitive.

    Args:
        name: The name of the custom call primitive
        fn_lowering: The lowering function for the custom call primitive that implements
        the custom call primitive.
        fn_abstract: The abstract function for the custom call primitive that evaluates
        the shapes and dtypes of the outputs.

    Returns
    -------
        A callable that can be used to call the custom call primitive
    """
    custom_call_primitive = Primitive(name)

    def custom_call(x: ArrayLike) -> ArrayLike:
        """Wrap the primitive using .bind()."""
        return custom_call_primitive.bind(x)

    custom_call_primitive.def_abstract_eval(fn_abstract)
    mlir.register_lowering(custom_call_primitive, fn_lowering)

    return custom_call
