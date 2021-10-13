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
"""Experimental Numpy backend."""
import numpy as np
import six

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import ops


__all__ = [
    'Assert',
    'assert_all_finite',
    'assert_equal',
    'assert_greater',
    'assert_greater_equal',
    'assert_integer',
    'assert_less',
    'assert_less_equal',
    'assert_near',
    'assert_negative',
    'assert_non_negative',
    'assert_non_positive',
    'assert_none_equal',
    'assert_positive',
    'assert_proper_iterable',
    'assert_rank',
    'assert_rank_at_least',
    'assert_rank_in',
    'assert_scalar',
    'check_numerics',
]

JAX_MODE = False


def skip_assert_for_tracers(f):
  """Function decorator that returns None if JAX tracers are detected."""
  if not JAX_MODE:
    return f
  from jax import core as jax_core  # pylint: disable=g-import-not-at-top
  def wrapped(*args, **kwargs):
    if (isinstance(np.zeros([0]), jax_core.Tracer) or  # omnistaging
        any(isinstance(arg, jax_core.Tracer)
            for arg in args + tuple(kwargs.values()))):
      print('skip assert ' + f.__name__)
      return None
    return f(*args, **kwargs)
  return wrapped


@skip_assert_for_tracers
def _assert_all_finite(x, message, name=None):
  if not np.all(np.isfinite(x)):
    raise ValueError('Non-finite value detected. {}'.format(message))


@skip_assert_for_tracers
def _assert_binary(
    x, y, comparator, sym, summarize=None, message=None, name=None):
  del summarize
  del name
  x = ops.convert_to_tensor(x)
  y = ops.convert_to_tensor(y)
  if not np.all(comparator(x, y)):
    raise ValueError('Condition x {} y did not hold element-wise. {}'.format(
        sym, message or ''))


@skip_assert_for_tracers
def _assert_equal(x, y, summarize=None, message=None, name=None):
  del summarize
  del name
  x = ops.convert_to_tensor(x)
  y = ops.convert_to_tensor(y)
  if not np.all(np.equal(x, y)):
    raise ValueError('Expected x == y but got {} vs {} {}'.format(
        x, y, message or ''))


def _assert_greater(x, y, summarize=None, message=None, name=None):
  return _assert_binary(
      x, y, np.greater, '>', summarize=summarize,
      message=message, name=name)


def _assert_less(x, y, summarize=None, message=None, name=None):
  return _assert_binary(
      x, y, np.less, '<', summarize=summarize,
      message=message, name=name)


def _assert_greater_equal(
    x, y, summarize=None, message=None, name=None):
  return _assert_binary(
      x, y, np.greater_equal, '>=', summarize=summarize,
      message=message, name=name)


def _assert_less_equal(
    x, y, summarize=None, message=None, name=None):
  return _assert_binary(
      x, y, np.less_equal, '<=', summarize=summarize,
      message=message, name=name)


@skip_assert_for_tracers
def _assert_compare_to_zero(
    x, comparator, sym, summarize=None, message=None, name=None):
  del summarize
  del name
  x = ops.convert_to_tensor(x)
  if not np.all(comparator(x, 0)):
    raise ValueError(
        'Condition x {} 0 did not hold element-wise; got {} {}'.format(
            sym, x, message or ''))


def _assert_positive(x, summarize=None, message=None, name=None):
  return _assert_compare_to_zero(
      x, np.greater, '>', summarize=summarize, message=message, name=name)


def _assert_negative(x, summarize=None, message=None, name=None):
  return _assert_compare_to_zero(
      x, np.less, '<', summarize=summarize, message=message, name=name)


def _assert_non_negative(x, summarize=None, message=None, name=None):
  return _assert_compare_to_zero(
      x, np.greater_equal, '>=',
      summarize=summarize, message=message, name=name)


def _assert_non_positive(x, summarize=None, message=None, name=None):
  return _assert_compare_to_zero(
      x, np.less_equal, '<=', summarize=summarize, message=message, name=name)


def _assert_rank(x, rank, message=None, name=None):  # pylint: disable=unused-argument
  return _assert_equal(x=len(np.shape(x)), y=rank, message=message)


def _assert_scalar(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_integer(*_, **__):  # pylint: disable=unused-argument
  pass


@skip_assert_for_tracers
def _assert_near(x, y, rtol=None, atol=None,
                 message=None, summarize=None, name=None):  # pylint: disable=unused-argument
  """Raises an error if abs(x - y) > atol + rtol * abs(y)."""
  del summarize
  del name
  x = ops.convert_to_tensor(x)
  y = ops.convert_to_tensor(y)
  rtol = rtol if rtol else 10 * np.finfo(x.dtype).eps
  atol = atol if atol else 10 * np.finfo(x.dtype).eps
  if np.any(np.abs(x - y) > atol + rtol * np.abs(y)):
    raise ValueError('x = {} and y = {} are not equal to tolerance rtol = {}, '
                     'atol = {} {}'.format(x, y, rtol, atol, message or ''))


@skip_assert_for_tracers
def _assert_none_equal(x, y, summarize=None, message=None, name=None):
  del summarize
  del name
  x = ops.convert_to_tensor(x)
  y = ops.convert_to_tensor(y)
  if np.any(np.equal(x, y)):
    raise ValueError('Expected x != y but got {} vs {} {}'.format(
        x, y, message or ''))


def _assert_proper_iterable(values):
  unintentional_iterables = (ops.Tensor, np.ndarray, bytes, six.text_type)
  if isinstance(values, unintentional_iterables):
    raise TypeError(
        'Expected argument "values" to be a "proper" iterable.  Found: %s' %
        type(values))

  if not hasattr(values, '__iter__'):
    raise TypeError(
        'Expected argument "values" to be iterable.  Found: %s' % type(values))


def _assert_rank_at_least(x, rank, message=None, name=None):
  del name
  if len(x.shape) < rank:
    raise ValueError('Expected rank at least {} but got shape {} {}'.format(
        rank, x.shape, message or ''))


def _assert_rank_in(*_, **__):  # pylint: disable=unused-argument
  pass


# --- Begin Public Functions --------------------------------------------------


Assert = utils.copy_docstring(  # pylint: disable=invalid-name
    'tf.debugging.Assert',
    lambda condition, data, summarize=None, name=None: assert_equal(  # pylint: disable=g-long-lambda
        True, condition, message=data))

assert_all_finite = utils.copy_docstring(
    'tf.debugging.assert_all_finite',
    _assert_all_finite)

assert_equal = utils.copy_docstring(
    'tf.debugging.assert_equal',
    _assert_equal)

assert_greater = utils.copy_docstring(
    'tf.debugging.assert_greater',
    _assert_greater)

assert_less = utils.copy_docstring(
    'tf.debugging.assert_less',
    _assert_less)

assert_rank = utils.copy_docstring(
    'tf.debugging.assert_rank',
    _assert_rank)

assert_scalar = utils.copy_docstring(
    'tf.debugging.assert_scalar',
    _assert_scalar)

assert_greater_equal = utils.copy_docstring(
    'tf.debugging.assert_greater_equal',
    _assert_greater_equal)

assert_integer = utils.copy_docstring(
    'tf.debugging.assert_integer',
    _assert_integer)

assert_less_equal = utils.copy_docstring(
    'tf.debugging.assert_less_equal',
    _assert_less_equal)

assert_near = utils.copy_docstring(
    'tf.debugging.assert_near',
    _assert_near)

assert_negative = utils.copy_docstring(
    'tf.debugging.assert_negative',
    _assert_negative)

assert_non_negative = utils.copy_docstring(
    'tf.debugging.assert_non_negative',
    _assert_non_negative)

assert_non_positive = utils.copy_docstring(
    'tf.debugging.assert_non_positive',
    _assert_non_positive)

assert_none_equal = utils.copy_docstring(
    'tf.debugging.assert_none_equal',
    _assert_none_equal)

assert_positive = utils.copy_docstring(
    'tf.debugging.assert_positive',
    _assert_positive)

assert_proper_iterable = utils.copy_docstring(
    'tf.debugging.assert_proper_iterable',
    _assert_proper_iterable)

assert_rank_at_least = utils.copy_docstring(
    'tf.debugging.assert_rank_at_least',
    _assert_rank_at_least)

assert_rank_in = utils.copy_docstring(
    'tf.debugging.assert_rank_in',
    _assert_rank_in)

check_numerics = utils.copy_docstring(
    'tf.debugging.check_numerics',
    lambda x, *_, **__: x)

is_numeric_tensor = utils.copy_docstring(
    'tf.debugging.is_numeric_tensor',
    lambda x: ops.is_tensor(x) and np.issubdtype(x.dtype, np.number))
