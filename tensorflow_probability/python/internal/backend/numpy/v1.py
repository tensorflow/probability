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

import contextlib
# Dependency imports
import numpy as np
import six

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import initializers
from tensorflow_probability.python.internal.backend.numpy import numpy_logging as logging
from tensorflow_probability.python.internal.backend.numpy.numpy_array import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.ops import convert_to_tensor
from tensorflow_probability.python.internal.backend.numpy.ops import Module
from tensorflow_probability.python.internal.backend.numpy.ops import name_scope
from tensorflow_probability.python.internal.backend.numpy.ops import Tensor
from tensorflow_probability.python.internal.backend.numpy.ops import Variable


__all__ = [
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
    'colocate_with',
    'get_variable',
    'global_variables_initializer',
    'initializers',
    'logging',
    'name_scope',
    'placeholder_with_default',
    'Module',
    'Session',
    'set_random_seed',
]


def _assert_equal(x, y, data=None, summarize=None, message=None, name=None):
  del summarize
  del name
  x = convert_to_tensor(x)
  y = convert_to_tensor(y)
  if not np.all(np.equal(x, y)):
    raise ValueError('Expected x == y but got {} vs {} {} {}'.format(
        x, y, message or '', data or ''))


def _assert_greater(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_less(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_rank(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_scalar(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_greater_equal(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_integer(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_less_equal(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_near(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_negative(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_non_negative(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_non_positive(*_, **__):  # pylint: disable=unused-argument
  pass


def _assert_none_equal(x, y, summarize=None, message=None, name=None):
  del summarize
  del name
  x = convert_to_tensor(x)
  y = convert_to_tensor(y)
  if np.any(np.equal(x, y)):
    raise ValueError('Expected x != y but got {} vs {} {}'.format(
        x, y, message or ''))


def _assert_positive(x, data=None, summarize=None, message=None, name=None):
  del data
  del summarize
  del name
  x = convert_to_tensor(x)
  if np.any(x <= 0):
    raise ValueError('Condition x > 0 did not hold; got {} {}'.format(
        x, message or ''))


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


@contextlib.contextmanager
def _colocate_with(*_, **__):  # pylint: disable=unused-argument
  pass


def _get_variable(  # pylint: disable=unused-argument
    name, shape=None, dtype=None, initializer=None, regularizer=None,
    trainable=None, collections=None, caching_device=None, partitioner=None,
    validate_shape=True, use_resource=None, custom_getter=None, constraint=None,
    synchronization=None, aggregation=None):
  if callable(initializer):
    initial_value = initializer(shape)
  else:
    initial_value = initializer
  return Variable(
      initial_value=initial_value, trainable=trainable,
      validate_shape=validate_shape, caching_device=caching_device, name=name,
      variable_def=None, dtype=dtype, import_scope=None, constraint=None)


def _placeholder_with_default(input, shape, name=None):  # pylint: disable=redefined-builtin,unused-argument
  x = np.array(input)
  if shape is None or any(s is None for s in shape):
    return x
  return np.reshape(x, shape)


# --- Begin Public Functions --------------------------------------------------


assert_equal = utils.copy_docstring(
    tf1.assert_equal,
    _assert_equal)

assert_greater = utils.copy_docstring(
    tf1.assert_greater,
    _assert_greater)

assert_less = utils.copy_docstring(
    tf1.assert_less,
    _assert_less)

assert_rank = utils.copy_docstring(
    tf1.assert_rank,
    _assert_rank)

assert_scalar = utils.copy_docstring(
    tf1.assert_scalar,
    _assert_scalar)

assert_greater_equal = utils.copy_docstring(
    tf1.assert_greater_equal,
    _assert_greater_equal)

assert_integer = utils.copy_docstring(
    tf1.assert_integer,
    _assert_integer)

assert_less_equal = utils.copy_docstring(
    tf1.assert_less_equal,
    _assert_less_equal)

assert_near = utils.copy_docstring(
    tf1.assert_near,
    _assert_near)

assert_negative = utils.copy_docstring(
    tf1.assert_negative,
    _assert_negative)

assert_non_negative = utils.copy_docstring(
    tf1.assert_non_negative,
    _assert_non_negative)

assert_non_positive = utils.copy_docstring(
    tf1.assert_non_positive,
    _assert_non_positive)

assert_none_equal = utils.copy_docstring(
    tf1.assert_none_equal,
    _assert_none_equal)

assert_positive = utils.copy_docstring(
    tf1.assert_positive,
    _assert_positive)

assert_proper_iterable = utils.copy_docstring(
    tf1.assert_proper_iterable,
    _assert_proper_iterable)

assert_rank_at_least = utils.copy_docstring(
    tf1.assert_rank_at_least,
    _assert_rank_at_least)

assert_rank_in = utils.copy_docstring(
    tf1.assert_rank_in,
    _assert_rank_in)

colocate_with = utils.copy_docstring(
    tf1.colocate_with,
    _colocate_with)

get_variable = utils.copy_docstring(
    tf1.get_variable,
    _get_variable)

placeholder_with_default = utils.copy_docstring(
    tf1.placeholder_with_default,
    _placeholder_with_default)

global_variables_initializer = utils.copy_docstring(
    tf1.global_variables_initializer,
    lambda: None)

set_random_seed = utils.copy_docstring(
    tf1.set_random_seed,
    lambda seed: np.random.seed(seed % (2**32 - 1)))


class Session(object):

  def __enter__(self, *_, **__):
    return self

  def __exit__(self, *_, **__):
    pass

  def run(self, *args, **_):
    return args

del tf
