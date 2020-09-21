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

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import errors
from tensorflow_probability.python.internal.backend.numpy.numpy_math import floor
from tensorflow_probability.python.internal.backend.numpy.numpy_math import truediv
from tensorflow_probability.python.internal.backend.numpy.ops import _convert_to_tensor
from tensorflow_probability.python.internal.backend.numpy.ops import clip_by_value
from tensorflow_probability.python.internal.backend.numpy.ops import is_tensor


__all__ = [
    'argsort',
    'histogram_fixed_width',
    'histogram_fixed_width_bins',
    'is_tensor',
    'print',
    'sort',
    'tensor_scatter_nd_add',
    'tensor_scatter_nd_sub',
    'tensor_scatter_nd_update',
    'unique',
    # 'clip_by_norm',
    # 'realdiv',
    # 'scatter_nd',
    # 'strided_slice',
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
  return np.argsort(
      values, axis, kind='stable' if stable else 'quicksort').astype(np.int32)


def _histogram_fixed_width(values, value_range, nbins=100, dtype=np.int32,
                           name=None):
  """Numpy implementation of `tf.histogram_fixed_width`."""
  del name
  return np.histogram(values, bins=nbins, range=value_range)[0].astype(
      utils.numpy_dtype(dtype))


def _histogram_fixed_width_bins(values, value_range, nbins=100, dtype=np.int32,
                                name=None):
  """Numpy implementation of `tf.histogram_fixed_width_bins`."""
  del name
  nbins_float = np.array(nbins, values.dtype)
  scaled_values = truediv(
      values - value_range[0], value_range[1] - value_range[0])
  indices = floor(nbins_float * scaled_values)
  indices = clip_by_value(indices, 0, nbins_float - 1).astype(
      utils.numpy_dtype(dtype))
  return indices


def _print(*inputs, **kwargs):
  print_args = {}
  if 'output_stream' in kwargs:
    print_args['file'] = kwargs['output_stream']
  for k in ('sep', 'end'):
    if k in kwargs:
      print_args[k] = kwargs[k]
  return builtin_print(*inputs, **print_args)


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
  indices = _convert_to_tensor(indices)
  tensor = _convert_to_tensor(tensor)
  updates = _convert_to_tensor(updates)
  indices = indices[..., 0]  # TODO(b/140685491): This is probably wrong!
  if JAX_MODE:
    import jax.ops as jaxops  # pylint: disable=g-import-not-at-top
    return jaxops.index_add(tensor, indices, updates)
  tensor[indices] += updates
  return tensor


# TODO(b/140685491): Add unit-test.
def _tensor_scatter_nd_sub(tensor, indices, updates, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.tensor_scatter_nd_sub`."""
  indices = _convert_to_tensor(indices)
  tensor = _convert_to_tensor(tensor)
  updates = _convert_to_tensor(updates)
  indices = indices[..., 0]  # TODO(b/140685491): This is probably wrong!
  if JAX_MODE:
    import jax.ops as jaxops  # pylint: disable=g-import-not-at-top
    return jaxops.index_add(tensor, indices, np.negative(updates))
  tensor[indices] -= updates
  return tensor


# TODO(b/140685491): Add unit-test.
def _tensor_scatter_nd_update(tensor, indices, updates, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.tensor_scatter_nd_update`."""
  indices = _convert_to_tensor(indices)
  tensor = _convert_to_tensor(tensor)
  updates = _convert_to_tensor(updates)
  indices = indices[..., 0]  # TODO(b/140685491): This is probably wrong!
  if JAX_MODE:
    import jax.ops as jaxops  # pylint: disable=g-import-not-at-top
    return jaxops.index_update(tensor, indices, updates)
  tensor[indices] = updates
  return tensor


_UniqueOutput = collections.namedtuple('UniqueOutput', ['y', 'idx'])


# TODO(b/140685491): Add unit-test.
def _unique(x, out_idx=np.int32, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.unique`."""
  x = np.array(x)
  if len(x.shape) != 1:
    raise errors.InvalidArgumentError('unique expects a 1D vector.')
  y, idx = np.unique(x,
                     return_index=True,
                     return_inverse=False,
                     return_counts=False,
                     axis=None)
  idx = idx.astype(utils.numpy_dtype(out_idx))
  return _UniqueOutput(y=y, idx=idx)


# --- Begin Public Functions --------------------------------------------------

argsort = utils.copy_docstring(
    'tf.argsort',
    _argsort)

histogram_fixed_width = utils.copy_docstring(
    'tf.histogram_fixed_width',
    _histogram_fixed_width)

histogram_fixed_width_bins = utils.copy_docstring(
    'tf.histogram_fixed_width_bins',
    _histogram_fixed_width_bins)

builtin_print = print

# pylint: disable=redefined-builtin
print = utils.copy_docstring(
    'tf.print',
    _print)

sort = utils.copy_docstring(
    'tf.sort',
    _sort)

tensor_scatter_nd_add = utils.copy_docstring(
    'tf.tensor_scatter_nd_add',
    _tensor_scatter_nd_add)

tensor_scatter_nd_sub = utils.copy_docstring(
    'tf.tensor_scatter_nd_sub',
    _tensor_scatter_nd_sub)

tensor_scatter_nd_update = utils.copy_docstring(
    'tf.tensor_scatter_nd_update',
    _tensor_scatter_nd_update)

unique = utils.copy_docstring(
    'tf.unique',
    _unique)
