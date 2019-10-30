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

__all__ = [
    'TensorArray',
]


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
               name=None):
    self._data = [None]*(size if size else 0)
    self._dtype = utils.numpy_dtype(dtype)
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
    return np.array([self.read(i) for i in indices], dtype=self.dtype)

  def stack(self, name=None):  # pylint: disable=unused-argument
    return np.array(self._data, dtype=self.dtype)

  def unstack(self, value, name=None):  # pylint: disable=unused-argument
    self._data = [np.array(x, dtype=self.dtype) for x in value]
    return self

  def read(self, index, name=None):  # pylint: disable=unused-argument
    return self._data[int(index)]

  def write(self, index, value, name=None):  # pylint: disable=unused-argument
    index = int(index)
    if self._size is None:
      self._data.extend([None]*(index - len(self._data) + 1))
    self._data[index] = np.array(value, dtype=self.dtype)
    return self

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
