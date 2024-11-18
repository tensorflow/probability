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
"""TF assertions."""

import numpy as np

import tensorflow.compat.v2 as tf

# Note: These assertions raise tf.errors.InvalidArgumentError when they fail.
assert_equal = tf.debugging.assert_equal
assert_greater = tf.debugging.assert_greater
assert_less = tf.debugging.assert_less
assert_rank = tf.debugging.assert_rank

assert_greater_equal = tf.debugging.assert_greater_equal
assert_integer = tf.debugging.assert_integer
assert_less_equal = tf.debugging.assert_less_equal
assert_near = tf.debugging.assert_near
assert_negative = tf.debugging.assert_negative
assert_non_negative = tf.debugging.assert_non_negative
assert_non_positive = tf.debugging.assert_non_positive
assert_none_equal = tf.debugging.assert_none_equal
assert_positive = tf.debugging.assert_positive
assert_rank_at_least = tf.debugging.assert_rank_at_least
assert_rank_in = tf.debugging.assert_rank_in


def assert_finite(x, summarize=None, message=None, name=None):
  """Assert all elements of `x` are finite.

  Args:
    x:  Numeric `Tensor`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_finite".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or lower.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError: If static checks determine `x` is not finite.
  """
  with tf.name_scope(name or 'assert_finite'):
    if message is None:
      message = '`Tensor` contained non-finite values.'
    x = tf.convert_to_tensor(x)
    x_ = tf.get_static_value(x)
    if x_ is not None:
      if ~np.all(np.isfinite(x_)):
        raise ValueError(message)
      return x
    assertion = tf.debugging.assert_equal(
        tf.math.is_finite(x), tf.ones_like(x, tf.bool),
        summarize=summarize, message=message)
    with tf.control_dependencies([assertion]):
      return tf.identity(x)


def assert_not_nan(x, summarize=None, message=None, name=None):
  """Assert all elements of `x` are not NaN.

  Args:
    x:  Numeric `Tensor`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_finite".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or lower.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError: If static checks determine `x` is not finite.
  """
  with tf.name_scope(name or 'assert_not_nan'):
    if message is None:
      message = '`Tensor` contained NaN values.'
    x = tf.convert_to_tensor(x)
    x_ = tf.get_static_value(x)
    if x_ is not None:
      if np.any(np.isnan(x_)):
        raise ValueError(message)
      return x
    assertion = tf.debugging.assert_equal(
        tf.math.is_nan(x), tf.zeros_like(x, tf.bool),
        summarize=summarize, message=message)
    with tf.control_dependencies([assertion]):
      return tf.identity(x)


def assert_rank_at_most(x, rank, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank` or smaller.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank_at_most(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_at_most".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or lower.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with tf.name_scope(name or 'assert_rank_at_most'):
    return tf.debugging.assert_less_equal(
        tf.rank(x), rank, summarize=summarize, message=message)
