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
"""Numpy implementations of TensorFlow general top-level functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
# Dependency imports
import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import ops
from tensorflow_probability.python.internal.backend.numpy.linalg_impl import einsum
from tensorflow_probability.python.internal.backend.numpy.linalg_impl import norm
from tensorflow_probability.python.internal.backend.numpy.linalg_impl import tensordot


__all__ = [
    'concat',
    'einsum',
    'expand_dims',
    'fill',
    'gather',
    'gather_nd',
    'linspace',
    'meshgrid',
    'norm',
    'one_hot',
    'ones',
    'ones_like',
    'pad',
    'range',
    'rank',
    'reshape',
    'reverse',
    'repeat',
    'roll',
    'searchsorted',
    'shape',
    'size',
    'slice',
    'split',
    'squeeze',
    'stack',
    'tensordot',
    'tile',
    'transpose',
    'unstack',
    'where',
    'zeros',
    'zeros_like',
    # 'boolean_mask',
    # 'foldl',
    # 'foldr',
]


JAX_MODE = False


if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top


def _astuple(x):
  try:
    return tuple(x)
  except TypeError:
    return x


def _gather(  # pylint: disable=unused-argument
    params,
    indices,
    validate_indices=None,
    axis=None,
    batch_dims=0,
    name=None):
  """gather."""
  indices = ops.convert_to_tensor(indices, dtype_hint=np.int32)
  if validate_indices is not None:
    raise NotImplementedError(
        'Argument `validate_indices != None` is currently unimplemented.')
  if batch_dims < 0:
    raise NotImplementedError('Negative `batch_dims` is currently unsupported.')
  if axis is None:
    axis = batch_dims
  if axis < 0:
    axis = axis + len(params.shape)
  # NOTE: For only the numpy backend, this function could create a single result
  # ndarray and use in-place updates.  For the Jax backend, this function
  # vmaps `np.take`.
  if JAX_MODE:
    take = lambda params, indices: np.take(params, indices,  # pylint: disable=g-long-lambda
                                           axis=axis - batch_dims)
    take = functools.reduce(
        lambda g, f: f(g), [jax.vmap] * int(batch_dims),
        take
    )
    return take(params, indices)
  params = ops.convert_to_tensor(params)
  res = np.array([
      np.take(params[i], indices[i], axis=axis - batch_dims)
      for i in np.ndindex(*params.shape[:batch_dims])
  ])
  return np.reshape(
      res,
      params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis+1:])


def _args_to_matching_arrays(args_list, dtype_hint=None):
  """Converts a list to array using the first element for dtype.

  This method is used to match the behavior of `tf.concat`.

  Args:
    args_list: A list or tuple of arguments.
    dtype_hint: An optional hint used when converting the args to tensors.
  Returns:
    A list of tensors.
  """
  dtype = None
  for arg in args_list:
    if hasattr(arg, 'dtype'):
      dtype = arg.dtype
      break
  if dtype is None:
    ret = []
    for arg in args_list:
      ret.append(ops.convert_to_tensor(arg, dtype, dtype_hint=dtype_hint))
      if dtype is None:
        dtype = ret[-1].dtype
  else:
    ret = [ops.convert_to_tensor(arg, dtype) for arg in args_list]
  return ret


def _concat(values, axis, name='concat'):
  del name
  if axis is None:
    raise ValueError('None values for `axis` argument not supported.')
  if not isinstance(values, (list, tuple)):
    values = [values]
  if len(values) == 1:
    return values[0]
  values = _args_to_matching_arrays(values)
  return np.concatenate(values, axis=axis)


def _gather_nd_single(params, indices):
  idx = tuple(np.moveaxis(indices, -1, 0))
  return params[idx]


def _gather_nd(  # pylint: disable=unused-argument
    params,
    indices,
    batch_dims=0,
    name=None):
  """gather_nd."""
  indices = ops.convert_to_tensor(indices, dtype_hint=np.int32)
  if batch_dims < 0:
    raise NotImplementedError('Negative `batch_dims` is currently unsupported.')
  if not JAX_MODE and batch_dims > 0:
    raise NotImplementedError(
        '`batch_dims > 0` currently unsupported in NumPy backend.')
  gather_nd_ = _gather_nd_single
  if JAX_MODE:
    gather_nd_ = functools.reduce(
        lambda g, f: f(g), [jax.vmap] * int(batch_dims),
        gather_nd_
    )
  return gather_nd_(params, indices)


def _one_hot(  # pylint: disable=unused-argument
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None):
  """One hot."""
  if on_value is None:
    on_value = 1
  if off_value is None:
    off_value = 0
  if dtype is None:
    dtype = utils.common_dtype([on_value, off_value], np.float32)
  indices = np.array(indices)
  depth = np.array(depth)
  pred = abs(np.arange(depth, dtype=indices.dtype) -
             indices[..., np.newaxis]) > 0
  y_out = np.where(pred, np.array(off_value, dtype), np.array(on_value, dtype))
  if axis is not None:
    y_out = np.moveaxis(y_out, -1, axis)
  return y_out


def _ones_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin,unused-argument
  return np.ones_like(input, dtype=utils.numpy_dtype(dtype))


# TODO(b/136555907): Add unit-test.
def _pad(  # pylint: disable=unused-argument
    tensor,
    paddings,
    mode='CONSTANT',
    constant_values=0,
    name=None):
  return np.pad(
      tensor, paddings,
      mode=mode.lower(),
      constant_values=constant_values)


def _range(start, limit=None, delta=1, dtype=None, name='range'):  # pylint: disable=unused-argument
  dtype = utils.numpy_dtype(dtype or utils.common_dtype([start], np.int32))
  start = ops.convert_to_tensor(start, dtype=dtype)
  limit = None if limit is None else ops.convert_to_tensor(limit, dtype=dtype)
  delta = ops.convert_to_tensor(delta, dtype=dtype)
  return np.arange(start, limit, delta).astype(dtype)


def _reverse(tensor, axis, name=None):  # pylint: disable=unused-argument
  if np.array(axis).ndim == 0:
    return np.flip(tensor, axis)
  for ax in axis:
    tensor = np.flip(tensor, ax)
  return tensor


def _searchsorted(  # pylint: disable=unused-argument
    sorted_sequence,
    values,
    side='left',
    out_type=np.int32,
    name=None):
  """Find indices for insertion for list to remain sorted."""
  # JAX doesn't support searchsorted at the moment, so we do a very naive way
  # of search sorting. We also don't use np.searchsorted in the numpy backend
  # because it doesn't support batching.
  sorted_sequence = sorted_sequence[..., np.newaxis, :]
  values = values[..., :, np.newaxis]
  if side == 'left':
    is_in_right_location = sorted_sequence < values
  elif side == 'right':
    is_in_right_location = sorted_sequence <= values
  return np.sum(is_in_right_location, axis=-1).astype(out_type)


def _shape(input, out_type=np.int32, name=None):  # pylint: disable=redefined-builtin,unused-argument
  return ops.convert_to_tensor(ops.convert_to_tensor(input).shape).astype(
      out_type)


def _size(input, out_type=np.int32, name=None):  # pylint: disable=redefined-builtin, unused-argument
  return np.prod(ops.convert_to_tensor(input).shape).astype(out_type)


builtin_slice = slice  # pylint: disable=invalid-name


def _slice(input_, begin, size, name=None):  # pylint: disable=unused-argument,redefined-outer-name
  slices = tuple(
      builtin_slice(b, b + s if s != -1 else None) for b, s in zip(begin, size))
  return input_[slices]


def _split(value, num_or_size_splits, axis=0, num=None, name='split'):  # pylint: disable=unused-argument
  """Map tf.split -> np.split."""
  indices_or_sections = np.array(num_or_size_splits)
  if indices_or_sections.ndim == 1:
    if any(idx == -1 for idx in indices_or_sections):
      # Numpy parameterizes by split indices and returns nsplits+1 arrays.
      total_splits = sum(idx for idx in indices_or_sections if idx != -1)
      remainder = int(max(0, np.array(value).shape[axis] - total_splits))
      indices_or_sections = [
          idx if idx != -1 else remainder for idx in indices_or_sections
      ]
    indices_or_sections = np.cumsum(np.array(indices_or_sections))[:-1]
  return np.split(value, indices_or_sections, axis)


def _transpose(a, perm=None, conjugate=False, name='transpose'):  # pylint: disable=unused-argument
  x = np.transpose(a, perm)
  return np.conjugate(x) if conjugate else x


def _zeros_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin,unused-argument
  return np.zeros_like(input, dtype=utils.numpy_dtype(dtype))


# --- Begin Public Functions --------------------------------------------------


concat = utils.copy_docstring(
    'tf.concat',
    lambda values, axis, name='concat': _concat(values, axis))


expand_dims = utils.copy_docstring(
    'tf.expand_dims',
    lambda input, axis, name=None: np.expand_dims(input, axis))

fill = utils.copy_docstring(
    'tf.fill',
    lambda dims, value, name=None: np.full(dims, value))

gather = utils.copy_docstring(
    'tf.gather',
    _gather)

gather_nd = utils.copy_docstring(
    'tf.gather_nd',
    _gather_nd)

reverse = utils.copy_docstring('tf.reverse', _reverse)

linspace = utils.copy_docstring(
    'tf.linspace',
    lambda start, stop, num, name=None: (  # pylint: disable=g-long-lambda
        np.linspace(start, stop, num).astype(np.array(start).dtype)))

meshgrid = utils.copy_docstring(
    'tf.meshgrid',
    np.meshgrid)

norm = utils.copy_docstring(
    'tf.norm',
    norm)

one_hot = utils.copy_docstring(
    'tf.one_hot',
    _one_hot)

ones = utils.copy_docstring(
    'tf.ones',
    lambda shape, dtype=np.float32, name=None: np.ones(  # pylint: disable=g-long-lambda
        shape, utils.numpy_dtype(dtype)))

ones_like = utils.copy_docstring(
    'tf.ones_like',
    _ones_like)

pad = utils.copy_docstring(
    'tf.pad',
    _pad)

range = utils.copy_docstring(  # pylint: disable=redefined-builtin
    'tf.range',
    _range)

rank = utils.copy_docstring(
    'tf.rank',
    lambda input, name=None: np.int32(np.array(input).ndim))  # pylint: disable=redefined-builtin,g-long-lambda

repeat = utils.copy_docstring(
    'tf.repeat',
    lambda input, repeats, axis=None, name=None: np.repeat(  # pylint: disable=g-long-lambda
        input, repeats, axis=axis))

reshape = utils.copy_docstring(
    'tf.reshape',
    lambda tensor, shape, name=None: np.reshape(tensor, shape))

roll = utils.copy_docstring(
    'tf.roll',
    lambda input, shift, axis: np.roll(input, shift, axis))  # pylint: disable=unnecessary-lambda

searchsorted = utils.copy_docstring(
    'tf.searchsorted',
    _searchsorted)

shape = utils.copy_docstring(
    'tf.shape',
    _shape)

size = utils.copy_docstring(
    'tf.size',
    _size)

slice = utils.copy_docstring(  # pylint: disable=redefined-builtin
    'tf.slice', _slice)

split = utils.copy_docstring('tf.split', _split)

squeeze = utils.copy_docstring(
    'tf.squeeze',
    lambda input, axis=None, name=None: np.squeeze(input, _astuple(axis)))

stack = utils.copy_docstring(
    'tf.stack', lambda values, axis=0, name='stack': np.stack(values, axis))

tile = utils.copy_docstring(
    'tf.tile',
    lambda input, multiples, name=None: np.tile(np.array(input), multiples))

transpose = utils.copy_docstring(
    'tf.transpose',
    _transpose)

unstack = utils.copy_docstring(
    'tf.unstack',
    lambda value, num=None, axis=0, name='unstack': tuple(  # pylint: disable=g-long-lambda
        np.squeeze(x, axis=axis) for x in
        np.split(value, value.shape[axis] if num is None else num, axis)))

where = utils.copy_docstring(
    'tf1.where',
    lambda condition, x=None, y=None, name=None: np.where(condition, x, y))

zeros = utils.copy_docstring(
    'tf.zeros',
    lambda shape, dtype=np.float32, name=None: np.zeros(  # pylint: disable=g-long-lambda
        shape, utils.numpy_dtype(dtype)))

zeros_like = utils.copy_docstring(
    'tf.zeros_like',
    _zeros_like)
