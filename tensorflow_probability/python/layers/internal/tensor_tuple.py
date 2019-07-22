# Copyright 2018 The TensorFlow Probability Authors.
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
"""`TensorTuple` a `CompositeTensor` for holding multiple `Tensor`s as one."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import type_spec  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'TensorTuple',
]


# TODO(b/133606651) Delete _to_components, _from_components, etc. once
# CompositeTensor is refactored to use _type_spec.
class TensorTuple(composite_tensor.CompositeTensor):
  """`Tensor`-like `tuple`-like for custom `Tensor` conversion masquerading."""

  def __init__(self, sequence):
    super(TensorTuple, self).__init__()
    self._sequence = tuple(tf.convert_to_tensor(value=x) for x in sequence)

  @property
  def _type_spec(self):
    return TensorTupleSpec(map(type_spec.TypeSpec.from_value, self._sequence))

  def _to_components(self):
    return self._sequence

  @classmethod
  def _from_components(cls, components):
    return cls(components)

  def _shape_invariant_to_components(self, shape=None):
    raise NotImplementedError('TensorTuple._shape_invariant_to_components')

  @property
  def _is_graph_tensor(self):
    return any(hasattr(x, 'graph') for x in self._sequence)

  def __len__(self):
    return len(self._sequence)

  def __getitem__(self, key):
    return self._sequence[key]

  def __iter__(self):
    return iter(self._sequence)

  def __repr__(self):
    return repr(self._sequence)

  def __str__(self):
    return str(self._sequence)


class TensorTupleSpec(type_spec.BatchableTypeSpec):
  """Type specification for a `TensorTuple`."""
  __slots__ = ['_specs']

  @property
  def value_type(self):
    return TensorTuple

  def __init__(self, tensor_specs):
    self._specs = tuple(tensor_specs)

  def _serialize(self):
    return (self._specs,)

  @property
  def _component_specs(self):
    return self._specs

  def _to_components(self, value):
    return value._sequence  # pylint: disable=protected-access

  def _from_components(self, tensor_list):
    return TensorTuple(tensor_list)

  def _batch(self, batch_size):
    # pylint: disable=protected-access
    return TensorTupleSpec([spec._batch(batch_size) for spec in self._specs])

  def _unbatch(self):
    # pylint: disable=protected-access
    return TensorTupleSpec([spec._unbatch() for spec in self._specs])


