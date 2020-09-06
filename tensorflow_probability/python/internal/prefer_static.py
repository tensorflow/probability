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
"""Operations that use static values when possible."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import decorator
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal.backend import numpy as nptf

# Try catch required to avoid breaking Probability opensource presubmits.
# TODO(amitpatankar): Remove this once tf-nightly has latest code.
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.client import pywrap_tf_session as c_api  # pylint: disable=g-direct-tensorflow-import
except ImportError:
  from tensorflow.python import pywrap_tensorflow as c_api  # pylint: disable=g-direct-tensorflow-import

from tensorflow.python.ops import control_flow_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import

JAX_MODE = False

# Enable converting TF TensorShape and Dimension into np.array. This allows TF
# code to pass TensorShapes into prefer_static functions. We can also re-use the
# nptf methods.
nptf.register_tensor_conversion_function(
    tf1.Dimension, nptf.ops._convert_dimension_to_tensor)  # pylint: disable=protected-access
nptf.register_tensor_conversion_function(
    tf.TensorShape, nptf.ops._convert_tensorshape_to_tensor)  # pylint: disable=protected-access


def _prefer_static(original_fn, static_fn):
  """Wraps original_fn, preferring to call static_fn when inputs are static."""
  original_spec = tf_inspect.getfullargspec(original_fn)
  static_spec = tf_inspect.getfullargspec(static_fn)
  if original_spec != static_spec:
    raise ValueError(
        'Arg specs do not match: original={}, static={}, fn={}'.format(
            original_spec, static_spec, original_fn))
  @decorator.decorator
  def wrap(wrapped_fn, *args, **kwargs):
    """The actual wrapper."""
    del wrapped_fn
    flat_args = tf.nest.flatten([args, kwargs])
    # N.B.: This `get_static_value` is nontrivial even in Eager mode, because
    # Keras's symbolic Tensors can exist when executing eagerly, and their
    # static values can be `None`.
    flat_args_ = [tf.get_static_value(a) for a in flat_args]
    all_static = all(arg is None or arg_ is not None
                     for arg, arg_ in zip(flat_args, flat_args_))
    if all_static:
      [args_, kwargs_] = tf.nest.pack_sequence_as([args, kwargs], flat_args_)
      return static_fn(*args_, **kwargs_)
    return original_fn(*args, **kwargs)
  return wrap(original_fn)


def _copy_docstring(original_fn, new_fn):
  """Wraps new_fn with the doc of original_fn."""
  original_spec = tf_inspect.getfullargspec(original_fn)
  new_spec = tf_inspect.getfullargspec(new_fn)
  if original_spec != new_spec:
    raise ValueError(
        'Arg specs do not match: original={}, new={}, fn={}'.format(
            original_spec, new_spec, original_fn))
  @decorator.decorator
  def wrap(wrapped_fn, *args, **kwargs):
    del wrapped_fn
    return new_fn(*args, **kwargs)
  return wrap(original_fn)


def _numpy_dtype(dtype):
  if dtype is None:
    return None
  return dtype_util.as_numpy_dtype(dtype)


def _get_static_value(pred):
  """Helper function for getting static values from maybe-tensor objects."""
  if JAX_MODE:
    try:
      return np.asarray(pred)
    except:  # JAX sometimes raises raw Exception in __array__.  # pylint: disable=bare-except
      return None
  if tf.is_tensor(pred):
    pred_value = tf.get_static_value(tf.convert_to_tensor(pred))

    # TODO(jamieas): remove the dependency on `pywrap_tensorflow`.
    # pylint: disable=protected-access
    if pred_value is None:
      pred_value = c_api.TF_TryEvaluateConstant_wrapper(pred.graph._c_graph,
                                                        pred._as_tf_output())
    # pylint: enable=protected-access
    return pred_value
  return pred


def _get_static_predicate(pred):
  """Helper function for statically evaluating predicates in `cond`."""
  pred_value = _get_static_value(pred)
  if pred_value in (0, 1, True, False):  # Accept 1/0 as valid boolean values.
    # This branch also casts np.array(False), tf.EagerTensor(True), etc.
    pred_value = bool(pred_value)
  elif pred_value is not None:
    raise TypeError('`pred` must be a Tensor, or a Python bool, or 1 or 0. '
                    'Found instead: {}'.format(pred))
  return pred_value


def _convert_to_shape_tensor_jax(value, dtype=None, dtype_hint=None, name=None):  # pylint: disable=unused-argument
  """Converts vectors and scalars of `int`-like to `ndarray`."""
  dtype = dtype_util.as_numpy_dtype(dtype or dtype_hint or np.int32)
  try:
    return np.array([int(v) for v in value], dtype=dtype)
  except:  # JAX throws raw Exception in some cases.  # pylint: disable=bare-except
    pass
  return np.array(int(value), dtype=dtype)


def smart_where(condition, x_fn, y_fn):
  """As tf.where, but only calls x_fn/y_fn when condition not statically known.

  IMPORTANT: Since this avoids executing the inoperative branch when possible,
  it will not necessarily broadcast `x_fn()` with `y_fn()`, so it is imperative
  that they return `Tensor`s which broadcast with `condition` to the same final
  shape.

  Args:
    condition: A `bool` Tensor.
    x_fn: A callable returning a `Tensor`, for locations where `condition` is
      `True`.
    y_fn: A callable returning a `Tensor`, for locations where `condition` is
      `False`.

  Returns:
    A `Tensor` equivalent to `tf.where(condition, x_fn(), y_fn())`.
  """
  cond_static = _get_static_value(condition)
  if cond_static is not None:
    if np.size(cond_static) == 1 and cond_static in (0, 1, False, True):
      return x_fn() if bool(cond_static) else y_fn()
    elif isinstance(cond_static, (np.ndarray, np.generic)):
      if np.all(cond_static):
        x = x_fn()
        return tf.broadcast_to(
            x, tf.broadcast_dynamic_shape(tf.shape(x), tf.shape(condition)))
      elif not np.any(cond_static):
        y = y_fn()
        return tf.broadcast_to(
            y, tf.broadcast_dynamic_shape(tf.shape(y), tf.shape(condition)))
  return tf.where(condition, x_fn(), y_fn())


def rank_from_shape(shape_tensor_fn, tensorshape=None):
  """Computes `rank` given a `Tensor`'s `shape`."""
  # Note: this function will implicitly interpret scalar "shapes" as length-1
  # vectors.
  if tensorshape is None:
    shape_tensor = (shape_tensor_fn() if callable(shape_tensor_fn)
                    else shape_tensor_fn)
    shape_tensor_ = tf.get_static_value(shape_tensor)
    if shape_tensor_ is not None:
      shape_tensor = np.int32(shape_tensor_)
    elif not hasattr(shape_tensor, 'shape'):
      shape_tensor = tf.convert_to_tensor(shape_tensor)
    ndims_ = tensorshape_util.num_elements(shape_tensor.shape)
    ndims_fn = lambda: tf.size(shape_tensor)
  else:
    ndims_ = tensorshape_util.rank(tensorshape)
    ndims_fn = lambda: tf.size(  # pylint: disable=g-long-lambda
        shape_tensor_fn() if callable(shape_tensor_fn) else shape_tensor_fn)
  return ndims_fn() if ndims_ is None else np.int32(ndims_)


def broadcast_shape(x_shape, y_shape):
  """Computes the shape of a broadcast.

  When both arguments are statically-known, the broadcasted shape will be
  computed statically and returned as a `TensorShape`.  Otherwise, a rank-1
  `Tensor` will be returned.

  Arguments:
    x_shape: A `TensorShape` or rank-1 integer `Tensor`.  The input `Tensor` is
      broadcast against this shape.
    y_shape: A `TensorShape` or rank-1 integer `Tensor`.  The input `Tensor` is
      broadcast against this shape.

  Returns:
    shape: A `TensorShape` or rank-1 integer `Tensor` representing the
      broadcasted shape.
  """
  x_shape_static = tf.get_static_value(x_shape)
  y_shape_static = tf.get_static_value(y_shape)
  if (x_shape_static is None) or (y_shape_static is None):
    return tf.broadcast_dynamic_shape(x_shape, y_shape)

  return tf.broadcast_static_shape(
      tf.TensorShape(x_shape_static), tf.TensorShape(y_shape_static))


def cond(pred, true_fn=None, false_fn=None, name=None):
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if not callable(true_fn):
    raise TypeError('`true_fn` must be callable.')
  if not callable(false_fn):
    raise TypeError('`false_fn` must be callable.')

  pred_value = _get_static_predicate(pred)
  if pred_value is not None:
    if pred_value:
      return true_fn()
    else:
      return false_fn()
  else:
    return tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn, name=name)


def case(pred_fn_pairs, default=None, exclusive=False, name='smart_case'):
  """Like tf.case, except attempts to statically evaluate predicates.

  If any predicate in `pred_fn_pairs` is a bool or has a constant value, the
  associated callable will be called or omitted depending on its value.
  Otherwise this functions like tf.case.

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                   callable which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  if isinstance(pred_fn_pairs, (list, tuple)):
    # We don't expect much usage of the `dict` option, esp. with unhashable
    # Tensors, but could always add another branch for that if it comes up.
    def maybe_static(pred):
      p = _get_static_predicate(pred)
      if p is None:
        return pred
      return p
    pred_fn_pairs = [(maybe_static(pred), fn) for pred, fn in pred_fn_pairs]
  return control_flow_ops._case_helper(  # pylint: disable=protected-access
      cond, pred_fn_pairs, default, exclusive, name, allow_python_preds=True)


def size0(x, name=None):
  """Returns the size of the first dimension (0 if scalar)."""
  with tf.name_scope(name or 'size0'):
    # First, ensure hasattr(x, 'shape').
    x_ = tf.get_static_value(x)
    if x_ is not None:
      x = np.array(x_)
    if not hasattr(x, 'shape'):
      x = tf.convert_to_tensor(x)
    # Next, try to read shape[0].
    ndims = tensorshape_util.rank(x.shape)
    if ndims is None or ndims == 0:
      n = ndims
    else:
      n = tf.compat.dimension_value(x.shape[0])
    if n is not None:
      return np.int32(n)
    return pad(shape(x)[:1], paddings=[[0, 1]], constant_values=0)[0]


def _ones_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  s = _shape(input)
  s_ = tf.get_static_value(s)
  if s_ is not None:
    return np.ones(s_, dtype_util.as_numpy_dtype(dtype or input.dtype))
  return tf.ones(s, dtype or s.dtype, name)
ones_like = _copy_docstring(tf.ones_like, _ones_like)


def _rank(input, name=None):  # pylint: disable=redefined-builtin,unused-argument
  if not hasattr(input, 'shape'):
    input = (tf.convert_to_tensor(input) if tf.get_static_value(input) is None
             else np.array(input))
  ndims_ = tensorshape_util.rank(getattr(input, 'shape', None))
  return tf.rank(input) if ndims_ is None else np.int32(ndims_)
rank = _copy_docstring(
    tf.rank,
    _rank)


def _setdiff1d(a, b, aminusb=True, validate_indices=True):
  """Compute set difference of elements in last dimension of `a` and `b`."""
  if not aminusb:
    raise NotImplementedError(
        'Argument `aminusb != True` is currently unimplemented.')
  if not validate_indices:
    raise NotImplementedError(
        'Argument `validate_indices != True` is currently unimplemented.')
  with tf.name_scope('setdiff1d'):
    dtype = dtype_util.as_numpy_dtype(
        dtype_util.common_dtype([a, b], dtype_hint=tf.int32))
    a_ = tf.get_static_value(a)
    b_ = tf.get_static_value(b)
    if a_ is None or b_ is None:
      a = tf.convert_to_tensor(a, dtype=dtype, name='a')
      b = tf.convert_to_tensor(b, dtype=dtype, name='b')
      return tf.sparse.to_dense(tf.sets.difference(
          a[tf.newaxis], b[tf.newaxis]))[0]
    a_ = np.array(a_, dtype=dtype)
    b_ = np.array(b_, dtype=dtype)
    # TODO(https://github.com/google/jax/issues/70): Jax lacks setdiff1d
    return np.setdiff1d(a_, b_)
setdiff1d = _copy_docstring(
    tf.sets.difference,
    _setdiff1d)


def _size(input, out_type=tf.int32, name=None):  # pylint: disable=redefined-builtin
  if not hasattr(input, 'shape'):
    x = np.array(input)
    input = tf.convert_to_tensor(input) if x.dtype is np.object else x
  n = tensorshape_util.num_elements(tf.TensorShape(input.shape))
  if n is None:
    return tf.size(input, out_type=out_type, name=name)
  return np.array(n).astype(_numpy_dtype(out_type))
size = _copy_docstring(tf.size, _size)


def _shape(input, out_type=tf.int32, name=None):  # pylint: disable=redefined-builtin,missing-docstring
  if not hasattr(input, 'shape'):
    x = np.array(input)
    input = tf.convert_to_tensor(input) if x.dtype is np.object else x
  input_shape = tf.TensorShape(input.shape)
  if tensorshape_util.is_fully_defined(input.shape):
    return np.array(tensorshape_util.as_list(input_shape)).astype(
        _numpy_dtype(out_type))
  # NOTE: tf.shape(x) can call `tf.convert_to_tensor(x)` **twice**, so we
  # pre-emptively convert-to-tensor.
  return tf.shape(tf.convert_to_tensor(input), out_type=out_type, name=name)
shape = _copy_docstring(tf.shape, _shape)


def _zeros_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  s = _shape(input)
  s_ = tf.get_static_value(s)
  if s_ is not None:
    return np.zeros(s, _numpy_dtype(dtype or input.dtype))
  return tf.zeros(s, dtype or s.dtype, name)
zeros_like = _copy_docstring(tf.zeros_like, _zeros_like)


def non_negative_axis(axis, rank, name=None):  # pylint:disable=redefined-outer-name
  """Make (possibly negatively indexed) `axis` argument non-negative."""
  with tf.name_scope(name or 'non_negative_axis'):
    if axis is None:
      return None
    if rank is None:
      raise ValueError('Argument `rank` cannot be `None`.')
    dtype = dtype_util.as_numpy_dtype(
        dtype_util.common_dtype([axis, rank], dtype_hint=tf.int32))
    rank_ = tf.get_static_value(rank)
    axis_ = tf.get_static_value(axis)
    if rank_ is None or axis_ is None:
      axis = tf.convert_to_tensor(axis, dtype=dtype, name='axis')
      rank = tf.convert_to_tensor(rank, dtype=dtype, name='rank')
      return tf.where(axis < 0, rank + axis, axis)
    axis_ = np.array(axis_, dtype=dtype)
    rank_ = np.array(rank_, dtype=dtype)
    return np.where(axis_ < 0, axis_ + rank_, axis_)


def is_numpy(x):
  """Returns true if `x` is a numpy object."""
  return isinstance(x, (np.ndarray, np.generic))


# The following functions only work in numpy if the inputs' *values are known
# statically*. Often (e.g., above) we dont need static values, just static
# properties.
abs = _prefer_static(tf.abs, nptf.abs)  # pylint: disable=redefined-builtin
add = _prefer_static(tf.add, nptf.add)
argmax = _prefer_static(tf.math.argmax, nptf.math.argmax)
argmin = _prefer_static(tf.math.argmin, nptf.math.argmin)
argsort = _prefer_static(tf.argsort, nptf.argsort)
broadcast_to = _prefer_static(tf.broadcast_to, nptf.broadcast_to)
cast = _prefer_static(tf.cast, nptf.cast)
ceil = _prefer_static(tf.math.ceil, nptf.math.ceil)
concat = _prefer_static(tf.concat, nptf.concat)
convert_to_shape_tensor = _prefer_static(
    tf.convert_to_tensor,
    _convert_to_shape_tensor_jax if JAX_MODE else tf.convert_to_tensor)
cumprod = _prefer_static(tf.math.cumprod, nptf.math.cumprod)
cumsum = _prefer_static(tf.math.cumsum, nptf.math.cumsum)
equal = _prefer_static(tf.equal, nptf.equal)
expm1 = _prefer_static(tf.math.expm1, nptf.math.expm1)
floor = _prefer_static(tf.math.floor, nptf.math.floor)
gather = _prefer_static(tf.gather, nptf.gather)
greater = _prefer_static(tf.greater, nptf.greater)
identity = _prefer_static(tf.identity, nptf.identity)
invert_permutation = _prefer_static(
    tf.math.invert_permutation, nptf.invert_permutation)
is_finite = _prefer_static(tf.math.is_finite, nptf.math.is_finite)
is_inf = _prefer_static(tf.math.is_inf, nptf.math.is_inf)
is_nan = _prefer_static(tf.math.is_nan, nptf.math.is_nan)
less = _prefer_static(tf.less, nptf.less)
linspace = _prefer_static(tf.linspace, nptf.linspace)
log = _prefer_static(tf.math.log, nptf.math.log)
log1p = _prefer_static(tf.math.log1p, nptf.math.log1p)
logical_and = _prefer_static(tf.logical_and, nptf.logical_and)
logical_not = _prefer_static(tf.logical_not, nptf.logical_not)
logical_or = _prefer_static(tf.logical_or, nptf.logical_or)
maximum = _prefer_static(tf.maximum, nptf.maximum)
minimum = _prefer_static(tf.minimum, nptf.minimum)
nextafter = _prefer_static(tf.math.nextafter, nptf.math.nextafter)
one_hot = _prefer_static(tf.one_hot, nptf.one_hot)
ones = _prefer_static(tf.ones, nptf.ones)
pad = _prefer_static(tf.pad, nptf.pad)
pow = _prefer_static(tf.math.pow, nptf.pow)  # pylint: disable=redefined-builtin
range = _prefer_static(tf.range, nptf.range)  # pylint: disable=redefined-builtin
reduce_all = _prefer_static(tf.reduce_all, nptf.reduce_all)
reduce_any = _prefer_static(tf.reduce_any, nptf.reduce_any)
reduce_max = _prefer_static(tf.reduce_max, nptf.reduce_max)
reduce_min = _prefer_static(tf.reduce_min, nptf.reduce_min)
reduce_prod = _prefer_static(tf.reduce_prod, nptf.reduce_prod)
reduce_sum = _prefer_static(tf.reduce_sum, nptf.reduce_sum)
reshape = _prefer_static(tf.reshape, nptf.reshape)
round = _prefer_static(tf.math.round, nptf.math.round)  # pylint: disable=redefined-builtin
rsqrt = _prefer_static(tf.math.rsqrt, nptf.math.rsqrt)
slice = _prefer_static(tf.slice, nptf.slice)  # pylint: disable=redefined-builtin
sort = _prefer_static(tf.sort, nptf.sort)
split = _prefer_static(tf.split, nptf.split)
sqrt = _prefer_static(tf.sqrt, nptf.sqrt)
stack = _prefer_static(tf.stack, nptf.stack)
tensor_scatter_nd_add = _prefer_static(
    tf.tensor_scatter_nd_add, nptf.tensor_scatter_nd_add)
tensor_scatter_nd_sub = _prefer_static(
    tf.tensor_scatter_nd_sub, nptf.tensor_scatter_nd_sub)
tensor_scatter_nd_update = _prefer_static(
    tf.tensor_scatter_nd_update, nptf.tensor_scatter_nd_update)
top_k = _prefer_static(tf.math.top_k, nptf.math.top_k)
unique = _prefer_static(tf.unique, nptf.unique)
unstack = _prefer_static(tf.unstack, nptf.unstack)
where = _prefer_static(tf.where, nptf.where)
zeros = _prefer_static(tf.zeros, nptf.zeros)
