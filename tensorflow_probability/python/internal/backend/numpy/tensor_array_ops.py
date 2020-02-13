# Copyright 2019 The TensorFlow Probability Authors.
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
"""Numpy implementations of `tf.TensorArray`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import numpy_array as array_ops

__all__ = [
    'TensorArray',
]


JAX_MODE = False


class TensorArray(object):
  """Stand-in for tf.TensorArray."""

  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               data=None,
               name=None):
    self._dtype = utils.numpy_dtype(dtype)
    if data is None:
      if JAX_MODE and size is not None and element_shape is not None:
        data = np.empty((size,) + tuple(element_shape), dtype=self._dtype)
      else:
        data = [None]*(0 if size is None else int(size))
    self._data = data
    self._size = size
    self._dynamic_size = dynamic_size
    self._clear_after_read = clear_after_read
    self._tensor_array_name = tensor_array_name
    self._handle = handle
    self._flow = flow
    self._infer_shape = infer_shape
    self._element_shape = element_shape
    self._colocate_with_first_write_call = colocate_with_first_write_call
    self._name = name

  @property
  def dtype(self):
    return self._dtype

  @property
  def dynamic_size(self):
    return self._dynamic_size

  @property
  def element_shape(self):
    return self._element_shape

  def size(self, name=None):  # pylint: disable=unused-argument
    return len(self._data)

  def gather(self, indices, name=None):  # pylint: disable=unused-argument
    indices = np.array(indices, dtype=np.int32)
    return array_ops.gather(np.array(self._data), indices)

  def stack(self, name=None):  # pylint: disable=unused-argument
    return np.array(self._data, dtype=self.dtype)

  def unstack(self, value, name=None):  # pylint: disable=unused-argument
    data = np.array(value, dtype=self.dtype)
    return TensorArray(self.dtype, data=data,
                       size=data.shape[0], element_shape=data.shape[1:])

  def read(self, index, name=None):  # pylint: disable=unused-argument
    index = np.array(index, dtype=np.int32)
    return self._data[index]

  def write(self, index, value, name=None):  # pylint: disable=unused-argument
    """Writes `value` at position `index`."""
    index = np.array(index, dtype=np.int32)
    value = np.array(value, dtype=self.dtype)
    if isinstance(self._data, list):
      new_data = list(self._data)
      if self.dynamic_size:
        new_data.extend([None]*(int(index) - len(new_data) + 1))
      new_data[index] = value
    elif JAX_MODE:
      import jax  # pylint: disable=g-import-not-at-top
      new_data = jax.ops.index_update(self._data, index, value)
    else:
      raise ValueError('Unexpected type: {}'.format(type(self._data)))
    return TensorArray(self.dtype, data=new_data,
                       dynamic_size=self.dynamic_size,
                       element_shape=self.element_shape,
                       infer_shape=self._infer_shape)

  def close(self, name=None):  # pylint: disable=unused-argument
    return self

  def identity(self):
    return self

  # TODO(b/143376677): Implement remaining TensorArray methods.

  @property
  def flow(self):
    raise NotImplementedError('If you need this feature, please email '
                              '`tfprobability@tensorflow.org`.')

  @property
  def handle(self):
    raise NotImplementedError('If you need this feature, please email '
                              '`tfprobability@tensorflow.org`.')

  def concat(self, name=None):
    raise NotImplementedError('If you need this feature, please email '
                              '`tfprobability@tensorflow.org`.')

  def grad(self, source, flow=None, name=None):
    raise NotImplementedError('If you need this feature, please email '
                              '`tfprobability@tensorflow.org`.')

  def scatter(self, indices, value, name=None):
    raise NotImplementedError('If you need this feature, please email '
                              '`tfprobability@tensorflow.org`.')

  def split(self, value, lengths, name=None):
    raise NotImplementedError('If you need this feature, please email '
                              '`tfprobability@tensorflow.org`.')


if JAX_MODE:
  from jax import tree_util  # pylint: disable=g-import-not-at-top

  def to_tree(val):
    vals = (val._data,)  # pylint: disable=protected-access
    aux = dict(dtype=val.dtype, element_shape=val.element_shape,
               dynamic_size=val.dynamic_size)
    return vals, aux

  def from_tree(aux, vals):
    return TensorArray(data=vals[0], **aux)

  tree_util.register_pytree_node(
      TensorArray,
      to_tree,
      from_tree
  )
