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
import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy.numpy_array import _reverse
from tensorflow_probability.python.internal.backend.numpy.numpy_array import one_hot
from tensorflow_probability.python.internal.backend.numpy.ops import _convert_to_tensor

scipy_special = utils.try_import('scipy.special')


JAX_MODE = False

if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top


__all__ = [
    'abs',
    'accumulate_n',
    'acos',
    'acosh',
    'add',
    'add_n',
    'angle',
    'argmax',
    'argmin',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'bessel_i0',
    'bessel_i0e',
    'bessel_i1',
    'bessel_i1e',
    'betainc',
    'bincount',
    'ceil',
    'confusion_matrix',
    'conj',
    'cos',
    'cosh',
    'count_nonzero',
    'cumprod',
    'cumsum',
    'digamma',
    'divide',
    'divide_no_nan',
    'equal',
    'erf',
    'erfc',
    'erfinv',
    'exp',
    'expm1',
    'floor',
    'floordiv',
    'floormod',
    'greater',
    'greater_equal',
    'igamma',
    'igammac',
    'imag',
    # 'in_top_k',
    'invert_permutation',
    'is_finite',
    'is_inf',
    'is_nan',
    'is_non_decreasing',
    'is_strictly_increasing',
    'l2_normalize',
    'lbeta',
    'less',
    'less_equal',
    'lgamma',
    'log',
    'log1p',
    'log_sigmoid',
    'log_softmax',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'maximum',
    'minimum',
    'mod',
    'multiply',
    'multiply_no_nan',
    'ndtri',
    'negative',
    'nextafter',
    'not_equal',
    'polygamma',
    'polyval',
    'pow',
    'real',
    'reciprocal',
    'reciprocal_no_nan',
    'reduce_all',
    'reduce_any',
    # 'reduce_euclidean_norm',
    'reduce_logsumexp',
    'reduce_max',
    'reduce_mean',
    'reduce_min',
    'reduce_prod',
    'reduce_std',
    'reduce_sum',
    'reduce_variance',
    'rint',
    'round',
    'rsqrt',
    # 'scalar_mul',
    'segment_max',
    'segment_mean',
    'segment_min',
    'segment_prod',
    'segment_sum',
    'sigmoid',
    'sign',
    'sin',
    'sinh',
    'softmax',
    'softplus',
    'softsign',
    'sqrt',
    'square',
    'squared_difference',
    'subtract',
    'tan',
    'tanh',
    'top_k',
    'truediv',
    # 'unsorted_segment_max',
    # 'unsorted_segment_mean',
    # 'unsorted_segment_min',
    # 'unsorted_segment_prod',
    # 'unsorted_segment_sqrt_n',
    'unsorted_segment_sum',
    'xdivy',
    'xlogy',
    'xlog1py',
    # 'zero_fraction',
    'zeta',
]


def _astuple(x):
  """Attempt to convert the given argument to be a Python tuple."""
  try:
    return (int(x),)
  except TypeError:
    pass

  try:
    return tuple(x)
  except TypeError:
    pass

  # If `x` is a scalar array, then the above call `tuple(x)` will have failed,
  # because scalar arrays do not support iteration in NumPy or JAX.
  if getattr(x, 'shape', None) == ():  # pylint: disable=g-explicit-bool-comparison
    try:
      return tuple(x[np.newaxis])
    except TypeError:
      pass

  return x


def _bincount(arr, weights=None, minlength=None, maxlength=None,  # pylint: disable=unused-argument
              dtype=np.int32, name=None):  # pylint: disable=unused-argument
  """Counts number of occurences of each value in `arr`."""
  # TODO(https://github.com/google/jax/issues/5719): Use np.bincount directly?
  if not JAX_MODE:
    return np.bincount(arr, weights, minlength).astype(utils.numpy_dtype(dtype))

  dtype = utils.numpy_dtype(dtype)
  num_buckets = (np.max(arr) + 1) if np.size(arr) else 0
  if minlength is not None and maxlength is not None and minlength == maxlength:
    # In the case where we can use minlength directly, this helps avoids the
    # use of an abstract value, which prevents JAX JIT.
    num_buckets = minlength
  else:
    if minlength is not None:
      num_buckets = np.maximum(num_buckets, minlength)
    if maxlength is not None:
      num_buckets = np.minimum(num_buckets, maxlength)
  one_hots = one_hot(arr, num_buckets)
  # Reduce over every dimension except the last one.
  axes = tuple(range(0, one_hots.ndim - 1))
  if weights is not None:
    return np.sum(
        one_hots * weights[..., np.newaxis], axis=axes).astype(dtype)
  return np.sum(one_hots, axis=axes).astype(dtype)


def _confusion_matrix(
    labels, predictions, num_classes=None, weights=None,
    dtype=np.int32, name=None):
  """Return confusion matrix between predictions and labels."""
  del name
  if num_classes is None:
    num_classes = np.maximum(np.max(predictions), np.max(labels)) + 1
  cmatrix = np.zeros([num_classes, num_classes], dtype=utils.numpy_dtype(dtype))
  if weights is None:
    weights = 1
  if not JAX_MODE:
    np.add.at(cmatrix, [labels, predictions], weights)
    return cmatrix
  return jax.ops.index_add(cmatrix, (labels, predictions), weights)


def _cumop(op, x, axis=0, exclusive=False, reverse=False, name=None,
           initial_value=None):
  """Shared impl of cumsum/cumprod."""
  del name
  axis = int(axis)
  result = op(_reverse(x, axis) if reverse else x, axis)
  if reverse:
    result = _reverse(result, axis)
  if exclusive:
    paddings = [[0, 0]] * result.ndim
    if isinstance(axis, int):
      axis = (axis,)
    for ax in axis:
      paddings[ax] = [0, 1] if reverse else [1, 0]
    result = np.pad(result, paddings, mode='constant',
                    constant_values=initial_value)
    slices = [slice(None)] * result.ndim
    for ax in axis:
      slices[ax] = slice(1, None) if reverse else slice(None, -1)
    result = result[tuple(slices)]
  return result

_cumprod = utils.partial(_cumop, np.cumprod, initial_value=1.)
_cumsum = utils.partial(_cumop, np.cumsum, initial_value=0.)


def _equal(x, y, name=None):
  del name
  x = _convert_to_tensor(x)
  y = _convert_to_tensor(y)
  return np.equal(x, y)


def _invert_permutation(x, name=None):
  del name
  x = _convert_to_tensor(x, dtype_hint=np.int32)
  return np.argsort(x).astype(x.dtype)


def _l2_normalize(x, axis=None, epsilon=1e-12, name=None):  # pylint: disable=unused-argument
  x = _convert_to_tensor(x)
  norm = np.linalg.norm(x, ord=2, axis=_astuple(axis), keepdims=True)
  norm = np.maximum(norm, np.sqrt(epsilon))
  return x / norm


def _lbeta(x, name=None):  # pylint: disable=unused-argument
  x = _convert_to_tensor(x)
  log_prod_gamma_x = np.sum(scipy_special.gammaln(x), axis=-1)
  sum_x = np.sum(x, axis=-1)
  log_gamma_sum_x = scipy_special.gammaln(sum_x)
  return log_prod_gamma_x - log_gamma_sum_x


def _max_mask_non_finite(x, axis=-1, keepdims=False, mask=0):
  """Returns `max` or `mask` if `max` is not finite."""
  x = _convert_to_tensor(x)
  m = np.max(x, axis=_astuple(axis), keepdims=keepdims)
  needs_masking = ~np.isfinite(m)
  if needs_masking.ndim > 0:
    m = np.where(needs_masking, mask, m)
  elif needs_masking:
    m = mask
  return m


def _segment_ids_to_np_indices(segment_ids):
  if len(segment_ids) < 1:
    return segment_ids
  shifted = np.roll(segment_ids, 1, axis=0)
  if JAX_MODE:
    shifted = shifted.at[0].set(-1)
  else:
    shifted[0] = -1
  return np.nonzero(segment_ids - shifted)[0]


def _segment_max(data, segment_ids, name=None):  # pylint: disable=unused-argument
  indices = _segment_ids_to_np_indices(segment_ids)
  if JAX_MODE:
    base = np.array(data)[indices]
    return base.at[segment_ids].max(data)
  else:
    return np.maximum.reduceat(data, indices)


def _segment_min(data, segment_ids, name=None):  # pylint: disable=unused-argument
  indices = _segment_ids_to_np_indices(segment_ids)
  if JAX_MODE:
    base = np.array(data)[indices]
    return base.at[segment_ids].min(data)
  else:
    return np.minimum.reduceat(data, indices)


def _segment_prod(data, segment_ids, name=None):  # pylint: disable=unused-argument
  indices = _segment_ids_to_np_indices(segment_ids)
  if JAX_MODE:
    base = np.ones_like(np.array(data)[indices])
    return base.at[segment_ids].mul(data)
  else:
    return np.multiply.reduceat(data, indices)


def _segment_sum(data, segment_ids, name=None):  # pylint: disable=unused-argument
  indices = _segment_ids_to_np_indices(segment_ids)
  if JAX_MODE:
    base = np.zeros_like(np.array(data)[indices])
    return base.at[segment_ids].add(data)
  else:
    return np.add.reduceat(data, indices)


def _segment_mean(data, segment_ids, name=None):
  sm = segment_sum(data, segment_ids, name=name)
  denom = bincount(segment_ids - segment_ids[:1])
  return sm / denom.reshape(denom.shape + (1,) * (data.ndim - 1))


def _softmax(logits, axis=None, name=None):  # pylint: disable=unused-argument
  logits = _convert_to_tensor(logits)
  axis = -1 if axis is None else axis
  y = logits - _max_mask_non_finite(logits, axis=axis, keepdims=True)
  y = np.exp(y)
  y = y / np.sum(y, axis=_astuple(axis), keepdims=True)
  return y


def _reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):  # pylint: disable=unused-argument
  """Computes `log(sum(exp(input_tensor))) along the specified axis."""
  input_tensor = _convert_to_tensor(input_tensor)
  dtype = input_tensor.dtype
  if not (np.issubdtype(dtype, np.floating)
          or np.issubdtype(dtype, np.complexfloating)):
    # Match TF error
    raise TypeError('Input must be either real or complex')
  try:
    return scipy_special.logsumexp(
        input_tensor, axis=_astuple(axis), keepdims=keepdims)
  except NotImplementedError:
    # We offer a non SP version just in case SP isn't installed and this
    # because logsumexp is often used.
    m = _max_mask_non_finite(input_tensor, axis=axis, keepdims=True)
    y = input_tensor - m
    y = np.exp(y, out=y)
    if not keepdims:
      m = np.squeeze(m, axis=_astuple(axis))
    return m + np.log(np.sum(y, axis=_astuple(axis), keepdims=keepdims))


# Match the TF return type for top_k.
TopK = collections.namedtuple('TopKV2', ['values', 'indices'])


def _top_k(input, k=1, sorted=True, name=None):  # pylint: disable=unused-argument,redefined-builtin,missing-docstring
  # This currently ignores sorted=False. However, this should be safe since
  # call sites that don't invoke sorted=True will assume results are unsorted
  # (vs. never sorted), and so sorted results shouldn't impact results.
  input = _convert_to_tensor(input)
  if JAX_MODE:
    # JAX automatically returns values in sorted order.
    return TopK(*jax.lax.top_k(input, k))
  n = int(input.shape[-1] - 1)
  # For the values, we sort the negative entries and choose the smallest ones
  # and negate. This is equivalent to choosing the largest entries
  values = -np.sort(-input, axis=-1)[..., :k]
  # For the indices, we could argsort and reverse the entries and choose the
  # first k entries. However, this does not work in the case of ties, since the
  # first index a value occurs at is preferred. Thus we also reverse the input
  # to ensure the last tied value becomes first, and subtract this off from the
  # last index since the list is reversed.
  indices = (
      n - (np.argsort(input[..., ::-1], kind='stable', axis=-1)[..., ::-1])
      )[..., :k].astype(np.int32)
  return TopK(values, indices)


def _unsorted_segment_sum(data, segment_ids, num_segments, name=None):
  data = _convert_to_tensor(data)
  del name
  if not JAX_MODE:
    raise NotImplementedError
  sums = np.zeros(num_segments)
  return jax.ops.index_add(sums, jax.ops.index[segment_ids], data)


# --- Begin Public Functions --------------------------------------------------


abs = utils.copy_docstring(  # pylint: disable=redefined-builtin
    'tf.math.abs',
    lambda x, name=None: np.abs(_convert_to_tensor(x)))

accumulate_n = utils.copy_docstring(
    'tf.math.accumulate_n',
    lambda inputs, shape=None, tensor_dtype=None, name=None: (  # pylint: disable=g-long-lambda
        sum(map(np.array, inputs)).astype(utils.numpy_dtype(tensor_dtype))))

acos = utils.copy_docstring(
    'tf.math.acos',
    lambda x, name=None: np.arccos(x))

acosh = utils.copy_docstring(
    'tf.math.acosh',
    lambda x, name=None: np.arccosh(x))

add = utils.copy_docstring(
    'tf.math.add',
    lambda x, y, name=None: np.add(x, y))

add_n = utils.copy_docstring(
    'tf.math.add_n',
    lambda inputs, name=None: sum(map(np.array, inputs)))

angle = utils.copy_docstring(
    'tf.math.angle',
    lambda input, name=None: np.angle(input))

argmax = utils.copy_docstring(
    'tf.math.argmax',
    lambda input, axis=None, output_type=np.int64, name=None: (  # pylint: disable=g-long-lambda
        np.argmax(input, axis=0 if axis is None else int(axis))
        .astype(utils.numpy_dtype(output_type))))

argmin = utils.copy_docstring(
    'tf.math.argmin',
    lambda input, axis=None, output_type=np.int64, name=None: (  # pylint: disable=g-long-lambda
        np.argmin(_convert_to_tensor(
            input), axis=0 if axis is None else int(axis))
        .astype(utils.numpy_dtype(output_type))))

asin = utils.copy_docstring(
    'tf.math.asin',
    lambda x, name=None: np.arcsin(x))

asinh = utils.copy_docstring(
    'tf.math.asinh',
    lambda x, name=None: np.arcsinh(x))

atan = utils.copy_docstring(
    'tf.math.atan',
    lambda x, name=None: np.arctan(x))

atan2 = utils.copy_docstring(
    'tf.math.atan2',
    lambda y, x, name=None: np.arctan2(y, x))

atanh = utils.copy_docstring(
    'tf.math.atanh',
    lambda x, name=None: np.arctanh(x))

bessel_i0 = utils.copy_docstring(
    'tf.math.bessel_i0',
    lambda x, name=None: scipy_special.i0(x))

bessel_i0e = utils.copy_docstring(
    'tf.math.bessel_i0e',
    lambda x, name=None: scipy_special.i0e(x))

bessel_i1 = utils.copy_docstring(
    'tf.math.bessel_i1',
    lambda x, name=None: scipy_special.i1(x))

bessel_i1e = utils.copy_docstring(
    'tf.math.bessel_i1e',
    lambda x, name=None: scipy_special.i1e(x))

betainc = utils.copy_docstring(
    'tf.math.betainc',
    lambda a, b, x, name=None: scipy_special.betainc(a, b, x))

bincount = utils.copy_docstring(
    'tf.math.bincount', _bincount)

ceil = utils.copy_docstring(
    'tf.math.ceil',
    lambda x, name=None: np.ceil(x))

confusion_matrix = utils.copy_docstring(
    'tf.math.confusion_matrix', _confusion_matrix)

conj = utils.copy_docstring(
    'tf.math.conj',
    lambda x, name=None: np.conj(x))

cos = utils.copy_docstring(
    'tf.math.cos',
    lambda x, name=None: np.cos(x))

cosh = utils.copy_docstring(
    'tf.math.cosh',
    lambda x, name=None: np.cosh(x))

count_nonzero = utils.copy_docstring(
    'tf.math.count_nonzero',
    lambda input, axis=None, keepdims=None, dtype=np.int64, name=None: (  # pylint: disable=g-long-lambda
        utils.numpy_dtype(dtype)(np.count_nonzero(input, axis))))

cumprod = utils.copy_docstring(
    'tf.math.cumprod',
    _cumprod)

cumsum = utils.copy_docstring(
    'tf.math.cumsum',
    _cumsum)

digamma = utils.copy_docstring(
    'tf.math.digamma',
    lambda x, name=None: scipy_special.digamma(x))

divide = utils.copy_docstring(
    'tf.math.divide',
    lambda x, y, name=None: np.divide(x, y))


def _divide_no_nan(x, y, name=None):  # pylint: disable=unused-argument
  dtype = np.result_type(x, y)
  y_is_zero = np.equal(y, 0.)
  div = np.divide(x, np.where(y_is_zero, np.ones((), dtype=dtype), y))
  return np.where(y_is_zero, np.zeros((), dtype=dtype), div)

divide_no_nan = utils.copy_docstring(
    'tf.math.divide_no_nan', _divide_no_nan)

equal = utils.copy_docstring(
    'tf.math.equal',
    _equal)

erf = utils.copy_docstring(
    'tf.math.erf',
    lambda x, name=None: scipy_special.erf(x))

erfc = utils.copy_docstring(
    'tf.math.erfc',
    lambda x, name=None: scipy_special.erfc(x))

erfinv = utils.copy_docstring(
    'tf.math.erfinv',
    lambda x, name=None: scipy_special.erfinv(x))

exp = utils.copy_docstring(
    'tf.math.exp',
    lambda x, name=None: np.exp(_convert_to_tensor(x)))

expm1 = utils.copy_docstring(
    'tf.math.expm1',
    lambda x, name=None: np.expm1(x))

floor = utils.copy_docstring(
    'tf.math.floor',
    lambda x, name=None: np.floor(x))

floordiv = utils.copy_docstring(
    'tf.math.floordiv',
    lambda x, y, name=None: np.floor_divide(x, y))

floormod = utils.copy_docstring(
    'tf.math.floormod',
    lambda x, y, name=None: np.mod(x, y))

greater = utils.copy_docstring(
    'tf.math.greater',
    lambda x, y, name=None: np.greater(x, y))

greater_equal = utils.copy_docstring(
    'tf.math.greater_equal',
    lambda x, y, name=None: np.greater_equal(x, y))

igamma = utils.copy_docstring(
    'tf.math.igamma',
    lambda a, x, name=None: scipy_special.gammainc(a, x))

igammac = utils.copy_docstring(
    'tf.math.igammac',
    lambda a, x, name=None: scipy_special.gammaincc(a, x))

imag = utils.copy_docstring(
    'tf.math.imag',
    lambda input, name=None: np.imag(input))

# in_top_k = utils.copy_docstring(
#     'tf.math.in_top_k',
#     lambda targets, predictions, k, name=None: np.in_top_k)

# TODO(b/256095991): Add unit-test.
invert_permutation = utils.copy_docstring(
    'tf.math.invert_permutation',
    _invert_permutation)

is_finite = utils.copy_docstring(
    'tf.math.is_finite',
    lambda x, name=None: np.isfinite(x))

is_inf = utils.copy_docstring(
    'tf.math.is_inf',
    lambda x, name=None: np.isinf(x))

is_nan = utils.copy_docstring(
    'tf.math.is_nan',
    lambda x, name=None: np.isnan(x))

is_non_decreasing = utils.copy_docstring(
    'tf.math.is_non_decreasing',
    lambda x, name=None: np.all(x[1:] >= x[:-1]))

is_strictly_increasing = utils.copy_docstring(
    'tf.math.is_strictly_increasing',
    lambda x, name=None: np.all(x[1:] > x[:-1]))

l2_normalize = utils.copy_docstring('tf.math.l2_normalize', _l2_normalize)

lbeta = utils.copy_docstring(
    'tf.math.lbeta',
    _lbeta)

less = utils.copy_docstring(
    'tf.math.less',
    lambda x, y, name=None: np.less(x, y))

less_equal = utils.copy_docstring(
    'tf.math.less_equal',
    lambda x, y, name=None: np.less_equal(x, y))

lgamma = utils.copy_docstring(
    'tf.math.lgamma',
    lambda x, name=None: scipy_special.gammaln(x))

log = utils.copy_docstring(
    'tf.math.log',
    lambda x, name=None: np.log(_convert_to_tensor(x)))

log1p = utils.copy_docstring(
    'tf.math.log1p',
    lambda x, name=None: np.log1p(_convert_to_tensor(x)))

log_sigmoid = utils.copy_docstring(
    'tf.math.log_sigmoid',
    lambda x, name=None: -_softplus(-_convert_to_tensor(x)))

log_softmax = utils.copy_docstring(
    'tf.math.log_softmax',
    lambda logits, axis=None, name=None: (np.subtract(  # pylint: disable=g-long-lambda
        logits,
        reduce_logsumexp(logits, -1 if axis is None else axis, keepdims=True))))

logical_and = utils.copy_docstring(
    'tf.math.logical_and',
    lambda x, y, name=None: np.logical_and(x, y))

logical_not = utils.copy_docstring(
    'tf.math.logical_not',
    lambda x, name=None: np.logical_not(x))

logical_or = utils.copy_docstring(
    'tf.math.logical_or',
    lambda x, y, name=None: np.logical_or(x, y))

logical_xor = utils.copy_docstring(
    'tf.math.logical_xor',
    lambda x, y, name=None: np.logical_xor(x, y))


if JAX_MODE:

  # TF and Jax have differing behavior when the inputs to maximum/minimum are
  # equal. We modify to match TF's behavior.

  @jax.custom_jvp
  def _maximum_(x, y):
    return np.maximum(x, y)

  @_maximum_.defjvp
  def _maximum_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    selected_x = np.where(x >= y, np.ones_like(x), np.zeros_like(x))
    return _maximum_(x, y), selected_x * dx + (1 - selected_x) * dy

  @jax.custom_jvp
  def _minimum_(x, y):
    return np.minimum(x, y)

  @_minimum_.defjvp
  def _minimum_fwd(primals, tangents):
    x, y = primals
    dx, dy = tangents
    selected_x = np.where(x <= y, np.ones_like(x), np.zeros_like(x))
    return _minimum_(x, y), selected_x * dx + (1 - selected_x) * dy

  # Need to wrap in a function because jax custom transforms returns an object,
  # not a function which breaks docstring wrapping.

  def _promote_dtypes(x, y):
    # Need to explicitly promote types because of custom_transforms.
    # We also broadcast x and y to have the same shape, so we don't have to
    # deal with broadcasting when writing the custom gradients for min/max.
    out_dtype = np.result_type(x, y)
    x = np.array(x, out_dtype)
    y = np.array(y, out_dtype)
    return np.broadcast_arrays(x, y)

  _minimum = lambda x, y, name=None: _minimum_(*_promote_dtypes(x, y))
  _maximum = lambda x, y, name=None: _maximum_(*_promote_dtypes(x, y))

else:

  _minimum = lambda x, y, name=None: np.minimum(_convert_to_tensor(x),  # pylint: disable=g-long-lambda
                                                _convert_to_tensor(y))
  _maximum = lambda x, y, name=None: np.maximum(_convert_to_tensor(x),  # pylint: disable=g-long-lambda
                                                _convert_to_tensor(y))

maximum = utils.copy_docstring(
    'tf.math.maximum', _maximum)

minimum = utils.copy_docstring(
    'tf.math.minimum', _minimum)

mod = utils.copy_docstring(
    'tf.math.mod',
    lambda x, y, name=None: np.mod(x, y))

multiply = utils.copy_docstring(
    'tf.math.multiply',
    lambda x, y, name=None: np.multiply(x, y))


def _multiply_no_nan(x, y, name=None):  # pylint: disable=unused-argument
  dtype = np.result_type(x, y)
  # TODO(b/146385087): The gradient should be
  # `lambda dz: [multiply_no_nan(dz, y), multiply_no_nan(x, dz)]`.
  return np.where(np.equal(y, 0.), np.zeros((), dtype=dtype), np.multiply(x, y))

multiply_no_nan = utils.copy_docstring(
    'tf.math.multiply_no_nan', _multiply_no_nan)

ndtri = utils.copy_docstring(
    'tf.math.ndtri',
    lambda x, name=None: scipy_special.ndtri(x))

negative = utils.copy_docstring(
    'tf.math.negative',
    lambda x, name=None: np.negative(x))

nextafter = utils.copy_docstring(
    'tf.math.nextafter',
    lambda x1, x2, name=None: np.nextafter(x1, x2))

not_equal = utils.copy_docstring(
    'tf.math.not_equal',
    lambda x, y, name=None: np.not_equal(x, y))

polygamma = utils.copy_docstring(
    'tf.math.polygamma',
    lambda a, x, name=None: scipy_special.polygamma(np.int32(a), x).astype(  # pylint: disable=unused-argument,g-long-lambda
        utils.common_dtype([a, x], dtype_hint=np.float32)))

polyval = utils.copy_docstring(
    'tf.math.polyval',
    lambda coeffs, x, name=None: np.polyval(coeffs, x))

pow = utils.copy_docstring(  # pylint: disable=redefined-builtin
    'tf.math.pow',
    lambda x, y, name=None: np.power(x, y))

real = utils.copy_docstring(
    'tf.math.real',
    lambda input, name=None: np.real(input))

reciprocal = utils.copy_docstring(
    'tf.math.reciprocal',
    lambda x, name=None: np.reciprocal(x))


def _reciprocal_no_nan(x, name=None):  # pylint: disable=unused-argument
  x_is_zero = np.equal(x, 0.)
  safe_x = np.where(x_is_zero, 1., x)
  return np.where(x_is_zero, 0., np.reciprocal(safe_x))


reciprocal_no_nan = utils.copy_docstring(
    'tf.math.reciprocal_no_nan', _reciprocal_no_nan)


def _apply_reduction(op, input_tensor, axis=None, keepdims=False, name=None,  # pylint: disable=unused-argument
                     include_dtype_kwarg=False):
  """Implements reduce_* for nptf."""
  input_tensor = _convert_to_tensor(input_tensor)
  axis = _astuple(axis)
  kwargs = dict(dtype=input_tensor.dtype) if include_dtype_kwarg else {}
  return op(input_tensor, axis=axis, keepdims=keepdims, **kwargs)

reduce_all = utils.copy_docstring(
    'tf.math.reduce_all',
    utils.partial(_apply_reduction, np.all))

reduce_any = utils.copy_docstring(
    'tf.math.reduce_any',
    utils.partial(_apply_reduction, np.any))

# reduce_euclidean_norm = utils.copy_docstring(
#     'tf.math.reduce_euclidean_norm',
#     lambda input_tensor, axis=None, keepdims=False, name=None: (
#         np.reduce_euclidean_norm))

reduce_logsumexp = utils.copy_docstring(
    'tf.math.reduce_logsumexp',
    _reduce_logsumexp)

reduce_max = utils.copy_docstring(
    'tf.math.reduce_max',
    utils.partial(_apply_reduction, np.max))

reduce_mean = utils.copy_docstring(
    'tf.math.reduce_mean',
    utils.partial(_apply_reduction, np.mean, include_dtype_kwarg=True))

reduce_min = utils.copy_docstring(
    'tf.math.reduce_min',
    utils.partial(_apply_reduction, np.min))

reduce_prod = utils.copy_docstring(
    'tf.math.reduce_prod',
    utils.partial(_apply_reduction, np.prod, include_dtype_kwarg=True))

reduce_std = utils.copy_docstring(
    'tf.math.reduce_std',
    utils.partial(_apply_reduction, np.std, include_dtype_kwarg=True))

reduce_sum = utils.copy_docstring(
    'tf.math.reduce_sum',
    utils.partial(_apply_reduction, np.sum, include_dtype_kwarg=True))

reduce_variance = utils.copy_docstring(
    'tf.math.reduce_variance',
    utils.partial(_apply_reduction, np.var, include_dtype_kwarg=True))

rint = utils.copy_docstring(
    'tf.math.rint',
    # JAX doesn't have rint, but round/around are ~the same with decimals=0.
    lambda x, name=None: np.around(x))

round = utils.copy_docstring(  # pylint: disable=redefined-builtin
    'tf.math.round',
    lambda x, name=None: np.round(x))

rsqrt = utils.copy_docstring(
    'tf.math.rsqrt',
    lambda x, name=None: 1. / np.sqrt(x))

# scalar_mul = utils.copy_docstring(
#     'tf.math.scalar_mul',
#     lambda data, segment_ids, name=None: np.scalar_mul)

segment_max = utils.copy_docstring(
    'tf.math.segment_max',
    _segment_max)

segment_mean = utils.copy_docstring(
    'tf.math.segment_mean',
    _segment_mean)

segment_min = utils.copy_docstring(
    'tf.math.segment_min',
    _segment_min)

segment_prod = utils.copy_docstring(
    'tf.math.segment_prod',
    _segment_prod)

segment_sum = utils.copy_docstring(
    'tf.math.segment_sum',
    _segment_sum)

sigmoid = utils.copy_docstring(
    'tf.math.sigmoid',
    lambda x, name=None: scipy_special.expit(x))

sign = utils.copy_docstring(
    'tf.math.sign',
    lambda x, name=None: np.sign(x))

sin = utils.copy_docstring(
    'tf.math.sin',
    lambda x, name=None: np.sin(x))

sinh = utils.copy_docstring(
    'tf.math.sinh',
    lambda x, name=None: np.sinh(x))

softmax = utils.copy_docstring(
    'tf.math.softmax',
    _softmax)


def _softplus(x, name=None):  # pylint: disable=unused-argument
  if not JAX_MODE:
    # This is effectively inlining jax.nn.softplus, which is (currently)
    # defined as np.logaddexp(x, 0.).
    # Both are numerically fine (see discussion in b/146563881).
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.)
  return jax.nn.softplus(x)


softplus = utils.copy_docstring(
    'tf.math.softplus',
    _softplus)

softsign = utils.copy_docstring(
    'tf.math.softsign',
    lambda features, name=None: np.divide(features, (np.abs(features) + 1)))

sqrt = utils.copy_docstring(
    'tf.math.sqrt',
    lambda x, name=None: np.sqrt(x))

square = utils.copy_docstring(
    'tf.math.square',
    lambda x, name=None: np.square(_convert_to_tensor(x)))

squared_difference = utils.copy_docstring(
    'tf.math.squared_difference',
    lambda x, y, name=None: np.square(x - y))

subtract = utils.copy_docstring(
    'tf.math.subtract',
    lambda x, y, name=None: np.subtract(x, y))

tan = utils.copy_docstring(
    'tf.math.tan',
    lambda x, name=None: np.tan(x))

tanh = utils.copy_docstring(
    'tf.math.tanh',
    lambda x, name=None: np.tanh(x))

top_k = utils.copy_docstring(
    'tf.math.top_k',
    _top_k)

truediv = utils.copy_docstring(
    'tf.math.truediv',
    lambda x, y, name=None: np.true_divide(x, y))

# unsorted_segment_max = utils.copy_docstring(
#     'tf.math.unsorted_segment_max',
#     lambda data, segment_ids, num_segments, name=None: (
#         np.unsorted_segment_max))

# unsorted_segment_mean = utils.copy_docstring(
#     'tf.math.unsorted_segment_mean',
#     lambda data, segment_ids, num_segments, name=None: (
#         np.unsorted_segment_mean))

# unsorted_segment_min = utils.copy_docstring(
#     'tf.math.unsorted_segment_min',
#     lambda data, segment_ids, num_segments, name=None: (
#         np.unsorted_segment_min))

# unsorted_segment_prod = utils.copy_docstring(
#     'tf.math.unsorted_segment_prod',
#     lambda data, segment_ids, num_segments, name=None: (
#         np.unsorted_segment_prod))

# unsorted_segment_sqrt_n = utils.copy_docstring(
#     'tf.math.unsorted_segment_sqrt_n',
#     lambda data, segment_ids, num_segments, name=None: (
#         np.unsorted_segment_sqrt_n))

unsorted_segment_sum = utils.copy_docstring(
    'tf.math.unsorted_segment_sum',
    _unsorted_segment_sum)

xdivy = utils.copy_docstring(
    'tf.math.xdivy',
    lambda x, y, name=None: (  # pylint: disable=unused-argument,g-long-lambda
        np.where(np.equal(x, 0.),
                 np.zeros_like(np.multiply(x, y)),
                 np.divide(x, y))))

xlogy = utils.copy_docstring(
    'tf.math.xlogy',
    lambda x, y, name=None: scipy_special.xlogy(x, y))

xlog1py = utils.copy_docstring(
    'tf.math.xlog1py',
    lambda x, y, name=None: scipy_special.xlog1py(x, y))

# zero_fraction = utils.copy_docstring(
#     'tf.math.zero_fraction',
#     lambda value, name=None: np.zero_fraction)

zeta = utils.copy_docstring(
    'tf.math.zeta',
    lambda x, q, name=None: scipy_special.zeta(x, q))
