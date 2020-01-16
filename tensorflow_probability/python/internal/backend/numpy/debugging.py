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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy.ops import convert_to_tensor
from tensorflow_probability.python.internal.backend.numpy.ops import is_tensor
from tensorflow_probability.python.internal.backend.numpy.ops import Tensor


__all__ = [
    'Assert',
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


def _assert_binary(
    x, y, comparator, sym, summarize=None, message=None, name=None):
  del summarize
  del name
  x = convert_to_tensor(x)
  y = convert_to_tensor(y)
  if not np.all(comparator(x, y)):
    raise ValueError('Condition x {} y did not hold element-wise. {}'.format(
        sym, message or ''))


def _assert_equal(x, y, summarize=None, message=None, name=None):
  del summarize
  del name
  x = convert_to_tensor(x)
  y = convert_to_tensor(y)
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


def _assert_compare_to_zero(
    x, comparator, sym, summarize=None, message=None, name=None):
  del summarize
  del name
  x = convert_to_tensor(x)
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


def _assert_rank(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_scalar(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_integer(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_near(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_none_equal(x, y, summarize=None, message=None, name=None):
  del summarize
  del name
  x = convert_to_tensor(x)
  y = convert_to_tensor(y)
  if np.any(np.equal(x, y)):
    raise ValueError('Expected x != y but got {} vs {} {}'.format(
        x, y, message or ''))


def _assert_proper_iterable(values):
  unintentional_iterables = (Tensor, np.ndarray, bytes, six.text_type)
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
    tf.debugging.Assert,
    lambda condition, data, summarize=None, name=None: None)

assert_equal = utils.copy_docstring(
    tf.debugging.assert_equal,
    _assert_equal)

assert_greater = utils.copy_docstring(
    tf.debugging.assert_greater,
    _assert_greater)

assert_less = utils.copy_docstring(
    tf.debugging.assert_less,
    _assert_less)

assert_rank = utils.copy_docstring(
    tf.debugging.assert_rank,
    _assert_rank)

assert_scalar = utils.copy_docstring(
    tf.debugging.assert_scalar,
    _assert_scalar)

assert_greater_equal = utils.copy_docstring(
    tf.debugging.assert_greater_equal,
    _assert_greater_equal)

assert_integer = utils.copy_docstring(
    tf.debugging.assert_integer,
    _assert_integer)

assert_less_equal = utils.copy_docstring(
    tf.debugging.assert_less_equal,
    _assert_less_equal)

assert_near = utils.copy_docstring(
    tf.debugging.assert_near,
    _assert_near)

assert_negative = utils.copy_docstring(
    tf.debugging.assert_negative,
    _assert_negative)

assert_non_negative = utils.copy_docstring(
    tf.debugging.assert_non_negative,
    _assert_non_negative)

assert_non_positive = utils.copy_docstring(
    tf.debugging.assert_non_positive,
    _assert_non_positive)

assert_none_equal = utils.copy_docstring(
    tf.debugging.assert_none_equal,
    _assert_none_equal)

assert_positive = utils.copy_docstring(
    tf.debugging.assert_positive,
    _assert_positive)

assert_proper_iterable = utils.copy_docstring(
    tf.debugging.assert_proper_iterable,
    _assert_proper_iterable)

assert_rank_at_least = utils.copy_docstring(
    tf.debugging.assert_rank_at_least,
    _assert_rank_at_least)

assert_rank_in = utils.copy_docstring(
    tf.debugging.assert_rank_in,
    _assert_rank_in)

check_numerics = utils.copy_docstring(
    tf.debugging.check_numerics,
    lambda x, *_, **__: x)

is_numeric_tensor = utils.copy_docstring(
    tf.debugging.is_numeric_tensor,
    lambda x: is_tensor(x) and np.issubdtype(x.dtype, np.number))
