# Copyright 2024 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Various types used in FunMC."""

from typing import Callable, TypeAlias, TypeVar

import jaxtyping
from fun_mc import backend
import typeguard

__all__ = [
    'Array',
    'Bool',
    'BoolScalar',
    'DType',
    'Float',
    'FloatScalar',
    'Int',
    'IntScalar',
    'runtime_typed',
    'Seed',
]

Array = backend.util.Array
Seed = backend.util.Seed
DType = backend.util.DType
Float = jaxtyping.Float
Int = jaxtyping.Int
Bool = jaxtyping.Bool
BoolScalar: TypeAlias = bool | Bool[Array, '']
IntScalar: TypeAlias = int | Int[Array, '']
FloatScalar: TypeAlias = float | Float[Array, '']

F = TypeVar('F', bound=Callable)


def runtime_typed(f: F) -> F:
  """Adds runtime type checking."""
  return jaxtyping.jaxtyped(f, typechecker=typeguard.typechecked)
