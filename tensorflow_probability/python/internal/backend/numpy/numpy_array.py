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

import functools
# Dependency imports
import numpy as np
import numpy as onp  # pylint: disable=reimported

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
    'sequence_mask',
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
  params = ops.convert_to_tensor(params)
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
    if batch_dims == 0 and axis == 0:
      return params[indices]
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
    if ops.is_tensor(arg):
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
  params = ops.convert_to_tensor(params)
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


def _linspace(start, stop, num, name=None, axis=0):  # pylint: disable=unused-argument
  """Match TF behavior with np.linspace."""
  start = ops.convert_to_tensor(start)
  # Match TF weirdness arising from truediv(int32, int32) = float64
  if np.issubdtype(start.dtype, np.integer):
    start = start.astype(np.float64)
  stop = ops.convert_to_tensor(stop, dtype=start.dtype)
  if not np.issubdtype(np.array(num).dtype, np.integer):
    raise TypeError('`num` must be an integer but got {}'.format(num.dtype))
  return np.linspace(start, stop, int(num), axis=axis).astype(start.dtype)


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
  else:
    dtype = utils.numpy_dtype(dtype)
  indices = np.array(indices)
  pred = abs(np.arange(depth, dtype=indices.dtype) -
             indices[..., np.newaxis]) > 0
  y_out = np.where(pred, np.array(off_value, dtype), np.array(on_value, dtype))
  if axis is not None:
    y_out = np.moveaxis(y_out, -1, axis)
  return y_out


def _ones_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin,unused-argument
  return np.ones_like(ops.convert_to_tensor(input),
                      dtype=utils.numpy_dtype(dtype))


# TODO(b/136555907): Add unit-test.
def _pad(  # pylint: disable=unused-argument
    tensor,
    paddings,
    mode='CONSTANT',
    constant_values=0,
    name=None):
  tensor = ops.convert_to_tensor(tensor)
  constant_values = ops.convert_to_tensor(constant_values)
  return np.pad(
      tensor, paddings,
      mode=mode.lower(),
      constant_values=constant_values)


def _range(start, limit=None, delta=1, dtype=None, name='range'):  # pylint: disable=unused-argument
  """Emulates tf.range."""
  # Emulating dtype inference logic from tf.range
  dtype = utils.numpy_dtype(dtype)
  infer_dtype = lambda t: ops.convert_to_tensor(t, dtype=dtype).dtype
  # We must keep start, limit, and delta static np.array since they determine
  # the size of the result array, which JAX requires to be static.
  start = onp.array(start, dtype=infer_dtype(start))
  limit = None if limit is None else onp.array(limit, dtype=infer_dtype(limit))
  delta = onp.array(delta, dtype=infer_dtype(delta))
  if dtype is None:
    dtype_hierarchy = [np.int32, np.int64, np.float32, np.float64]
    inferred_dtype = max(
        (arg.dtype for arg in (start, limit, delta) if arg is not None),
        key=dtype_hierarchy.index)
  else:
    inferred_dtype = dtype
  return np.arange(start, limit, delta).astype(inferred_dtype)


def _reverse(tensor, axis, name=None):  # pylint: disable=unused-argument
  if np.array(axis).ndim == 0:
    return np.flip(tensor, axis)
  for ax in axis:
    tensor = np.flip(tensor, ax)
  return tensor


def _sequence_mask(lengths, maxlen=None, dtype=np.bool_, name=None):  # pylint: disable=unused-argument
  lengths = np.array(lengths, dtype=np.int32)
  if maxlen is None:
    maxlen = np.max(lengths).astype(lengths.dtype)
  return (np.arange(maxlen) < lengths[..., np.newaxis]).astype(dtype)


if JAX_MODE:
  _searchsorted_vmap_sides = {
      side: jax.vmap(functools.partial(jax.numpy.searchsorted, side=side))
      for side in ('left', 'right')
  }


def _searchsorted(  # pylint: disable=unused-argument
    sorted_sequence,
    values,
    side='left',
    out_type=np.int32,
    name=None):
  """Find indices for insertion for list to remain sorted."""
  if JAX_MODE:
    try:
      func = _searchsorted_vmap_sides[side]
    except KeyError:
      raise ValueError("'%s' is an invalid value for keyword 'side'" % side)
    sorted_sequence_2d = np.reshape(sorted_sequence,
                                    (-1, sorted_sequence.shape[-1]))
    values_2d = np.reshape(values, (-1, values.shape[-1]))
    if sorted_sequence_2d.shape[0] != values_2d.shape[0]:
      raise ValueError('Leading dim_size of both tensors must match.')
    return np.reshape(func(sorted_sequence_2d, values_2d).astype(out_type),
                      values.shape)
  # We don't use np.searchsorted in the numpy backend because it doesn't support
  # batching.
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
  return np.asarray(
      onp.prod(ops.convert_to_tensor(input).shape), dtype=out_type)


builtin_slice = slice  # pylint: disable=invalid-name


def _slice(input_, begin, size, name=None):  # pylint: disable=unused-argument,redefined-outer-name
  if JAX_MODE:
    input_ = np.asarray(input_)
    size = [dim_size - start if size == -1 else size
            for (size, start, dim_size) in zip(size, begin, input_.shape)]
    return jax.lax.dynamic_slice(input_, begin, size)
  slices = tuple(
      builtin_slice(b, b + s if s != -1 else None) for b, s in zip(begin, size))
  return input_[slices]


def _split(value, num_or_size_splits, axis=0, num=None, name='split'):  # pylint: disable=unused-argument
  """Map tf.split -> np.split."""
  if np.isscalar(num_or_size_splits):
    return np.split(value, num_or_size_splits, axis)

  indices_or_sections = onp.array(num_or_size_splits)
  if indices_or_sections.ndim == 1:
    if any(idx == -1 for idx in indices_or_sections):
      # Numpy parameterizes by split indices and returns nsplits+1 arrays.
      total_splits = sum(idx for idx in indices_or_sections if idx != -1)
      remainder = int(max(0, np.array(value).shape[axis] - total_splits))
      indices_or_sections = [
          idx if idx != -1 else remainder for idx in indices_or_sections
      ]
    indices_or_sections = onp.cumsum(onp.array(indices_or_sections))[:-1]
  return np.split(value, indices_or_sections, axis)


def _stack(values, axis=0, name='stack'):
  del name
  values = [ops.convert_to_tensor(x) for x in values]
  if values:
    return np.stack(values, axis=axis)
  else:
    if axis != 0:
      raise IndexError(f'Axis {axis} is out of range.')
    return np.zeros([0], np.float32)


def _transpose(a, perm=None, conjugate=False, name='transpose'):  # pylint: disable=unused-argument
  x = np.transpose(ops.convert_to_tensor(a), perm)
  return np.conjugate(x) if conjugate else x


def _unstack(value, num=None, axis=0, name='unstack'):
  del name
  value = ops.convert_to_tensor(value)
  if axis == 0:
    return list(value)
  return list(
      np.squeeze(x, axis=axis)
      for x in np.split(value, value.shape[axis] if num is None else num, axis))


def _where(condition, x=None, y=None, name='where'):  # pylint: disable=unused-argument
  if x is None and y is None:
    return np.stack(np.asarray(condition).nonzero(), axis=-1)
  return np.where(condition, x, y)


def _zeros_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin,unused-argument
  return np.zeros_like(input, dtype=utils.numpy_dtype(dtype))


# --- Begin Public Functions --------------------------------------------------


concat = utils.copy_docstring(
    'tf.concat',
    _concat)


expand_dims = utils.copy_docstring(
    'tf.expand_dims',
    lambda input, axis, name=None: np.expand_dims(input, axis))

fill = utils.copy_docstring(
    'tf.fill',
    lambda dims, value, name=None: np.full(dims, ops.convert_to_tensor(value)))

gather = utils.copy_docstring(
    'tf.gather',
    _gather)

gather_nd = utils.copy_docstring(
    'tf.gather_nd',
    _gather_nd)

reverse = utils.copy_docstring('tf.reverse', _reverse)

linspace = utils.copy_docstring(
    'tf.linspace',
    _linspace)

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
    lambda tensor, shape, name=None: np.reshape(  # pylint: disable=g-long-lambda
        ops.convert_to_tensor(tensor), shape))

roll = utils.copy_docstring(
    'tf.roll',
    lambda input, shift, axis: np.roll(input, shift, axis))  # pylint: disable=unnecessary-lambda

sequence_mask = utils.copy_docstring(
    'tf.sequence_mask',
    _sequence_mask)

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
    'tf.stack', _stack)

tile = utils.copy_docstring(
    'tf.tile',
    lambda input, multiples, name=None: np.tile(np.array(input), multiples))

transpose = utils.copy_docstring(
    'tf.transpose',
    _transpose)

unstack = utils.copy_docstring(
    'tf.unstack',
    _unstack)

where = utils.copy_docstring(
    'tf.where',
    _where)

zeros = utils.copy_docstring(
    'tf.zeros',
    lambda shape, dtype=np.float32, name=None: np.zeros(  # pylint: disable=g-long-lambda
        shape, utils.numpy_dtype(dtype)))

zeros_like = utils.copy_docstring(
    'tf.zeros_like',
    _zeros_like)
