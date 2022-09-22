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
"""Utility functions for dtypes."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf


__all__ = [
    'as_numpy_dtype',
    'assert_same_float_dtype',
    'base_dtype',
    'base_equal',
    'common_dtype',
    'eps',
    'is_bool',
    'is_complex',
    'is_floating',
    'is_integer',
    'is_numpy_compatible',
    'max',
    'min',
    'name',
    'real_dtype',
    'size',
]


JAX_MODE = False
NUMPY_MODE = False
SKIP_DTYPE_CHECKS = False


def is_numpy_compatible(dtype):
  """Returns if dtype has a corresponding NumPy dtype."""
  if JAX_MODE or NUMPY_MODE:
    return True
  else:
    return tf.as_dtype(dtype).is_numpy_compatible


def as_numpy_dtype(dtype):
  """Returns a `np.dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype


def base_dtype(dtype):
  """Returns a non-reference `dtype` based on this `dtype`."""
  dtype = None if dtype is None else tf.as_dtype(dtype)
  if hasattr(dtype, 'base_dtype'):
    return dtype.base_dtype
  return dtype


def base_equal(a, b):
  """Returns `True` if base dtypes are identical."""
  return base_dtype(a) == base_dtype(b)


class _NotYetSeen(object):
  """Sentinel class for uninspected arguments' dtype."""

  def __repr__(self):
    return '...'

_NOT_YET_SEEN = _NotYetSeen()


def common_dtype(args, dtype_hint=None):
  """Returns (nested) explict dtype from `args` if there is one.

  Dtypes of all args, and `dtype_hint`, must have the same nested structure if
  they are not `None`. `args` itself may be any nested structure; its
  structure is flattened and ignored.

  Args:
    args: A nested structure of objects that may have `dtype`.
    dtype_hint: Optional (nested) dtype containing defaults to use in place of
      `None`. If `dtype_hint` is not nested and the common dtype of `args` is
      nested, `dtype_hint` serves as the default for each element of the common
      nested dtype structure.

  Returns:
    dtype: The (nested) dtype common across all elements of `args`, or `None`.

  #### Examples

  Usage with non-nested dtype:

  ```python
  x = tf.ones([3, 4], dtype=tf.float64)
  y = 4.
  z = None
  common_dtype([x, y, z], dtype_hint=tf.float32)  # ==> tf.float64
  common_dtype([y, z], dtype_hint=tf.float32)     # ==> tf.float32

  # The arg to `common_dtype` can be an arbitrary nested structure; it is
  # flattened, and the common dtype of its contents is returned.
  common_dtype({'x': x, 'yz': (y, z)})
  # ==> tf.float64
  ```

  Usage with nested dtype:

  ```python
  # Define `x` and `y` as JointDistributions with the same nested dtype.
  x = tfd.JointDistributionNamed(
      {'a': tfd.Uniform(np.float64(0.), 1.),
       'b': tfd.JointDistributionSequential(
          [tfd.Normal(0., 2.), tfd.Bernoulli(0.4)])})
  x.dtype  # ==> {'a': tf.float64, 'b': [tf.float32, tf.int32]}

  y = tfd.JointDistributionNamed(
      {'a': tfd.LogitNormal(np.float64(0.), 1.),
       'b': tfd.JointDistributionSequential(
          [tfd.Normal(-1., 1.), tfd.Bernoulli(0.6)])})
  y.dtype  # ==> {'a': tf.float64, 'b': [tf.float32, tf.int32]}

  # Pack x and y into an arbitrary nested structure and pass it to
  # `common_dtype`.
  args0 = [x, y]
  common_dtype(args0)  # ==> {'a': tf.float64, 'b': [tf.float32, tf.int32]}

  # The nested structure of the argument to `common_dtype` is flattened and
  # ignored; only the nested structures of the dtypes are relevant.
  args1 = {'x': x, 'yz': {'y': y, 'z': None}}
  common_dtype(args1)  # ==> {'a': tf.float64, 'b': [tf.float32, tf.int32]}
  ```
  """

  def _unify_dtype(current, new):
    if current is not None and new is not None and current != new:
      if SKIP_DTYPE_CHECKS:
        return (np.ones([2], dtype) + np.ones([2], dt)).dtype
      raise TypeError
    return new if current is None else current

  dtype = None
  flattened_args = tf.nest.flatten(args)
  seen = [_NOT_YET_SEEN] * len(flattened_args)
  for i, a in enumerate(flattened_args):
    if hasattr(a, 'dtype') and a.dtype:
      dt = tf.nest.map_structure(
          lambda d: d if d is None else as_numpy_dtype(d), a.dtype)
      seen[i] = dt
    else:
      seen[i] = None
      continue
    if dtype is None:
      dtype = dt
    try:
      dtype = tf.nest.map_structure(_unify_dtype, dtype, dt)
    except TypeError:
      raise TypeError(
          'Found incompatible dtypes, {} and {}. Seen so far: {}'.format(
              dtype, dt, tf.nest.pack_sequence_as(args, seen))) from None
  if dtype_hint is None:
    return tf.nest.map_structure(base_dtype, dtype)
  if dtype is None:
    return tf.nest.map_structure(base_dtype, dtype_hint)
  if tf.nest.is_nested(dtype) and not tf.nest.is_nested(dtype_hint):
    dtype_hint = tf.nest.map_structure(lambda _: dtype_hint, dtype)
  return tf.nest.map_structure(
      lambda dt, h: base_dtype(h if dt is None else dt), dtype, dtype_hint)


def convert_to_dtype(tensor_or_dtype, dtype=None, dtype_hint=None):
  """Get a dtype from a list/tensor/dtype using convert_to_tensor semantics."""
  if tensor_or_dtype is None:
    return dtype or dtype_hint

  # Tensorflow dtypes need to be typechecked
  if tf.is_tensor(tensor_or_dtype):
    dt = base_dtype(tensor_or_dtype.dtype)
  elif isinstance(tensor_or_dtype, tf.DType):
    dt = base_dtype(tensor_or_dtype)
  # Numpy dtypes defer to dtype/dtype_hint
  elif isinstance(tensor_or_dtype, np.ndarray):
    dt = base_dtype(dtype or dtype_hint or tensor_or_dtype.dtype)
  elif np.issctype(tensor_or_dtype):
    dt = base_dtype(dtype or dtype_hint or tensor_or_dtype)
  else:
    # If this is a Python object, call `convert_to_tensor` and grab the dtype.
    # Note that this will add ops in graph-mode; we may want to consider
    # other ways to handle this case.
    dt = tf.convert_to_tensor(tensor_or_dtype, dtype, dtype_hint).dtype

  if not SKIP_DTYPE_CHECKS and dtype and not base_equal(dtype, dt):
    raise TypeError('Found incompatible dtypes, {} and {}.'.format(dtype, dt))
  return dt


def eps(dtype):
  """Returns the distance between 1 and the next largest representable value."""
  return np.finfo(as_numpy_dtype(dtype)).eps


def is_bool(dtype):
  """Returns whether this is a boolean data type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_bool'):
    return dtype.is_bool
  # We use `kind` because:
  # np.issubdtype(np.uint8, np.bool_) == True.
  return np.dtype(dtype).kind == 'b'


def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complexfloating)


def is_floating(dtype):
  """Returns whether this is a (non-quantized, real) floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_floating'):
    return dtype.is_floating
  return np.issubdtype(np.dtype(dtype), np.floating)


def is_integer(dtype):
  """Returns whether this is a (non-quantized) integer type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_integer') and not callable(dtype.is_integer):
    return dtype.is_integer
  return np.issubdtype(np.dtype(dtype), np.integer)


def max(dtype):  # pylint: disable=redefined-builtin
  """Returns the maximum representable value in this data type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'max') and not callable(dtype.max):
    return dtype.max
  use_finfo = is_floating(dtype) or is_complex(dtype)
  return np.finfo(dtype).max if use_finfo else np.iinfo(dtype).max


def min(dtype):  # pylint: disable=redefined-builtin
  """Returns the minimum representable value in this data type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'min') and not callable(dtype.min):
    return dtype.min
  use_finfo = is_floating(dtype) or is_complex(dtype)
  return np.finfo(dtype).min if use_finfo else np.iinfo(dtype).min


def name(dtype):
  """Returns the string name for this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'name'):
    return dtype.name
  if hasattr(dtype, '__name__'):
    return dtype.__name__
  return str(dtype)


def size(dtype):
  """Returns the number of bytes to represent this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'size') and hasattr(dtype, 'as_numpy_dtype'):
    return dtype.size
  return np.dtype(dtype).itemsize


def real_dtype(dtype):
  """Returns the dtype of the real part."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'real_dtype'):
    return dtype.real_dtype
  # TODO(jvdillon): Find a better way.
  return np.array(0, as_numpy_dtype(dtype)).real.dtype


def _assert_same_base_type(items, expected_type=None):
  r"""Asserts all items are of the same base type.

  Args:
    items: List of graph items (e.g., `Variable`, `Tensor`, `SparseTensor`,
        `Operation`, or `IndexedSlices`). Can include `None` elements, which
        will be ignored.
    expected_type: Expected type. If not specified, assert all items are
        of the same base type.

  Returns:
    Validated type, or none if neither expected_type nor items provided.

  Raises:
    ValueError: If any types do not match.
  """
  original_expected_type = expected_type
  mismatch = False
  for item in items:
    if item is not None:
      item_type = base_dtype(item.dtype)
      if expected_type is None:
        expected_type = item_type
      elif expected_type != item_type:
        mismatch = True
        break
  if mismatch:
    # Loop back through and build up an informative error message (this is very
    # slow, so we don't do it unless we found an error above).
    expected_type = original_expected_type
    original_item_str = None
    get_name = lambda x: x.name if hasattr(x, 'name') else str(x)
    for item in items:
      if item is not None:
        item_type = base_dtype(item.dtype)
        if not expected_type:
          expected_type = item_type
          original_item_str = get_name(item)
        elif expected_type != item_type:
          raise ValueError(
              '{}, type={}, must be of the same type ({}){}.'.format(
                  get_name(item),
                  item_type,
                  expected_type,
                  ((' as {}'.format(original_item_str))
                   if original_item_str else '')))
    return expected_type  # Should be unreachable
  else:
    return expected_type


def assert_same_float_dtype(tensors=None, dtype=None):
  """Validate and return float type based on `tensors` and `dtype`.

  For ops such as matrix multiplication, inputs and weights must be of the
  same float type. This function validates that all `tensors` are the same type,
  validates that type is `dtype` (if supplied), and returns the type. Type must
  be a floating point type. If neither `tensors` nor `dtype` is supplied,
  the function will return `dtypes.float32`.

  Args:
    tensors: Tensors of input values. Can include `None` elements, which will
      be ignored.
    dtype: Expected type.

  Returns:
    Validated type.

  Raises:
    ValueError: if neither `tensors` nor `dtype` is supplied, or result is not
      float, or the common type of the inputs is not a floating point type.
  """
  if tensors:
    dtype = _assert_same_base_type(tensors, dtype)
  if not dtype:
    dtype = tf.float32
  elif not is_floating(dtype):
    raise ValueError('Expected floating point type, got {}.'.format(dtype))
  return dtype
