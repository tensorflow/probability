# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Contains slice abstractions used in function inversion."""
from typing import Any

import dataclasses
import jax.numpy as np

from oryx.core import pytree

__all__ = [
]


@dataclasses.dataclass(frozen=True)
class Slice:
  """Represents a slice of an array along an axis."""
  start: int
  stop: int

  def __lt__(self, other: 'Slice') -> bool:
    return ((self.start > other.start and self.stop <= other.stop)
            or (self.start >= other.start and self.stop < other.stop))

  def __eq__(self, other: 'Slice') -> bool:
    return self.start == other.start and self.stop == other.stop

  def __repr__(self):
    return f'Slice({self.start}, {self.stop})'


class NDSlice(pytree.Pytree):
  """Represents a multidimensional-slice of an array.

  When we take multidimensional slices (ndslices) of arrays, we normally
  represent them as tuples of Python `slice` objects. For example, calling
  `x[0:2, 3:4]` passes `(slice(0, 2), slice(3, 4))` into the `__getitem__` of
  `x`. We are interested in maintaining the information about an ndslice (the
  tuple of slices) in addition to maintaining the value the slice corresponds to
  and the ILDJ of the value.

  For example, `NDSlice(jnp.ones(3), jnp.zeros(3), Slice(3, 6))` implies that
  there is some larger value `x` when if we take the slice `x[3:6]` we obtain
  `jnp.ones(3)` and that the ILDJ terms corresponding to that slice are
  `jnp.zeros(3)`.

  `NDSlice`s can be concatenated with each other, combining the values, ILDJs
  and the slice indices. Eventually when we concatenate enough `NDSlice`s we
  can reconstruct the original value and its ILDJ.
  """

  def __init__(self, value: Any, ildj: Any, *slices: Slice):
    self.value = value
    self.ildj = ildj
    self.slices = slices

  @property
  def ndim(self) -> int:
    return len(self.slices)

  def __eq__(self, other: 'NDSlice') -> bool:
    # Comparisons should only be for the same logical array, so we only need to
    # compare the slice indices themselves
    return self.slices == other.slices

  def __hash__(self):
    return hash(self.slices)

  def flatten(self):
    return (self.value, self.ildj), self.slices

  @classmethod
  def unflatten(cls, slices, values):
    return NDSlice(values[0], values[1], *slices)

  def __lt__(self, other: 'NDSlice') -> bool:
    """Compares two `NDSlice`s.

    A `NDSlice` is less than another any of its component slices are
    contained inside the other's slices.

    Args:
      other: A `NDSlice` object.
    Returns:
      `True` if any slice of the `NDSlice` is contained within `other` and
      `False` otherwise.
    """
    if self.ndim != other.ndim:
      raise ValueError('Cannot compare `NDSlice`s of different dimensions.')
    is_less_than = False
    for s1, s2 in zip(self.slices, other.slices):
      if s1 == s2:
        continue
      if s1 < s2:
        is_less_than = True
      if s2 < s1:
        return False
    return is_less_than

  def can_concatenate(self, other: 'NDSlice', dim: int) -> bool:
    """Determines if an `NDSlice` can be concatenated to this one.

    An `NDSlice` can be concatenated to another along a dimension `dim` so long
    as the `stop` of the first `Slice` in that dimension is the `start` of
    `other`'s `Slice` at the same dimension.

    For example, we can concatenate `NDSlice(..., Slice(0, 4))` to
    `NDSlice(..., Slice(4, 8)) along dimension 0 but we cannot concatenate
    `NDSlice(..., Slice(0, 3))` to `NDSlice(..., (4, 8))`.

    Args:
      other: `NDSlice`, an `NDSlice` object to be concatenated to this one.
      dim: `int`, an index corresponding to the dimension along which we
           concatenate.
    Returns:
      `True` if `other` can be concatenated to this `NDSlice` and `False`
       otherwise.
    """
    if self.ndim != other.ndim:
      return False
    for i, (s1, s2) in enumerate(zip(self.slices, other.slices)):
      if i == dim:
        if s1.stop != s2.start:
          return False
      else:
        if s1 != s2:
          return False
    return True

  def concatenate(self, other: 'NDSlice', dim: int) -> 'NDSlice':
    """Concatenates an `NDSlice` along a dimension.

    If `can_concatenate` evaluates to `True`, `concatenate` returns an
    `NDSlice` object that is the result of concatenating this one to `other`.
    We concatenate two `NDSlices` by merging their slices indices along
    `dim` and concatenating their values and ILDJs along `dim`.

    For example, concatenating `NDSlice(jnp.ones(2), jnp.zeros(2), Slice(0, 2))`
    to `NDSlice(jnp.ones(3), jnp.zeros(3), Slice(2, 5)) along `dim = 0` results
    in `NDSLice(jnp.ones(5), jnp.zeros(5), Slice(0, 5))`.

    Args:
      other: `NDSlice`, an `NDSlice` object to be concatenated to this one.
      dim: `int`, an index corresponding to the dimension along which we
           concatenate.
    Raises:
      ValueError if we cannot concatenate `other` to this object.
    Returns:
       Returns a new `NDSlice` object that is the result of concatenating
       `other` to this `NDSlice`.
    """
    if not self.can_concatenate(other, dim):
      raise ValueError(
          f'Cannot concatenate incompatible slices: {self}, {other}')
    new_slices = []
    for i, (s1, s2) in enumerate(zip(self.slices, other.slices)):
      if i != dim:
        new_slices.append(s1)
      else:
        new_value = np.concatenate([self.value, other.value], axis=i)
        new_ildj = np.concatenate([self.ildj, other.ildj], axis=i)
        new_slices.append(Slice(s1.start, s2.stop))
    return NDSlice(new_value, new_ildj, *new_slices)

  @classmethod
  def new(cls, value, ildj):
    slices = (Slice(0, s) for s in value.shape)
    return NDSlice(value, ildj, *slices)
