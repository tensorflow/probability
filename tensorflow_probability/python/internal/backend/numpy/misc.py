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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy.ops import is_tensor


__all__ = [
    'argsort',
    'is_tensor',
    'sort',
    'tensor_scatter_nd_add',
    'tensor_scatter_nd_sub',
    'tensor_scatter_nd_update',
    'unique',
    # 'clip_by_norm',
    # 'floormod',
    # 'meshgrid',
    # 'mod',
    # 'norm',
    # 'realdiv',
    # 'scatter_nd',
    # 'searchsorted',
    # 'strided_slice',
    # 'truediv',
    # 'truncatediv',
    # 'truncatemod',
    # 'unique_with_counts',
]


JAX_MODE = False


def _argsort(values, axis=-1, direction='ASCENDING', stable=False, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.argsort`."""
  if direction == 'ASCENDING':
    pass
  elif direction == 'DESCENDING':
    values = np.negative(values)
  else:
    raise ValueError('Unrecognized direction: {}.'.format(direction))
  return np.argsort(values, axis, kind='stable' if stable else 'quicksort')


def _sort(values, axis=-1, direction='ASCENDING', name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.sort`."""
  if direction == 'ASCENDING':
    pass
  elif direction == 'DESCENDING':
    values = np.negative(values)
  else:
    raise ValueError('Unrecognized direction: {}.'.format(direction))
  result = np.sort(values, axis, kind='stable')
  if direction == 'DESCENDING':
    return np.negative(result)
  return result


# TODO(b/140685491): Add unit-test.
def _tensor_scatter_nd_add(tensor, indices, updates, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.tensor_scatter_nd_add`."""
  indices = indices[..., 0]  # TODO(b/140685491): This is probably wrong!
  if JAX_MODE:
    import jax.ops as jaxops  # pylint: disable=g-import-not-at-top
    return jaxops.index_add(tensor, indices, updates)
  tensor[indices] += updates
  return tensor


# TODO(b/140685491): Add unit-test.
def _tensor_scatter_nd_sub(tensor, indices, updates, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.tensor_scatter_nd_sub`."""
  indices = indices[..., 0]  # TODO(b/140685491): This is probably wrong!
  if JAX_MODE:
    import jax.ops as jaxops  # pylint: disable=g-import-not-at-top
    return jaxops.index_add(tensor, indices, np.negative(updates))
  tensor[indices] -= updates
  return tensor


# TODO(b/140685491): Add unit-test.
def _tensor_scatter_nd_update(tensor, indices, updates, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.tensor_scatter_nd_update`."""
  indices = indices[..., 0]  # TODO(b/140685491): This is probably wrong!
  if JAX_MODE:
    import jax.ops as jaxops  # pylint: disable=g-import-not-at-top
    return jaxops.index_update(tensor, indices, updates)
  tensor[indices] = updates
  return tensor


_UniqueOutput = collections.namedtuple('UniqueOutput', ['y', 'idx'])


# TODO(b/140685491): Add unit-test.
def _unique(x, out_idx=tf.int32, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.unique`."""
  x = np.array(x)
  if len(x.shape) != 1:
    raise tf.errors.InvalidArgumentError('unique expects a 1D vector.')
  y, idx = np.unique(x,
                     return_index=True,
                     return_inverse=False,
                     return_counts=False,
                     axis=None)
  idx = idx.astype(utils.numpy_dtype(out_idx))
  return _UniqueOutput(y=y, idx=idx)


# --- Begin Public Functions --------------------------------------------------

argsort = utils.copy_docstring(
    tf.argsort,
    _argsort)

sort = utils.copy_docstring(
    tf.sort,
    _sort)

tensor_scatter_nd_add = utils.copy_docstring(
    tf.tensor_scatter_nd_add,
    _tensor_scatter_nd_add)

tensor_scatter_nd_sub = utils.copy_docstring(
    tf.tensor_scatter_nd_sub,
    _tensor_scatter_nd_sub)

tensor_scatter_nd_update = utils.copy_docstring(
    tf.tensor_scatter_nd_update,
    _tensor_scatter_nd_update)

unique = utils.copy_docstring(
    tf.unique,
    _unique)
