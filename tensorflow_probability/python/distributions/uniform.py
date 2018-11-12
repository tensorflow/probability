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
"""The Uniform distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.framework import tensor_shape


class Uniform(distribution.Distribution):
  """Uniform distribution with `low` and `high` parameters.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; a, b) = I[a <= x < b] / Z
  Z = b - a
  ```

  where

  - `low = a`,
  - `high = b`,
  - `Z` is the normalizing constant, and
  - `I[predicate]` is the [indicator function](
    https://en.wikipedia.org/wiki/Indicator_function) for `predicate`.

  The parameters `low` and `high` must be shaped in a way that supports
  broadcasting (e.g., `high - low` is a valid operation).

  #### Examples

  ```python
  # Without broadcasting:
  u1 = Uniform(low=3.0, high=4.0)  # a single uniform distribution [3, 4]
  u2 = Uniform(low=[1.0, 2.0],
               high=[3.0, 4.0])  # 2 distributions [1, 3], [2, 4]
  u3 = Uniform(low=[[1.0, 2.0],
                    [3.0, 4.0]],
               high=[[1.5, 2.5],
                     [3.5, 4.5]])  # 4 distributions
  ```

  ```python
  # With broadcasting:
  u1 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])  # 3 distributions
  ```

  """

  def __init__(self,
               low=0.,
               high=1.,
               validate_args=False,
               allow_nan_stats=True,
               name="Uniform"):
    """Initialize a batch of Uniform distributions.

    Args:
      low: Floating point tensor, lower boundary of the output interval. Must
        have `low < high`.
      high: Floating point tensor, upper boundary of the output interval. Must
        have `low < high`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      InvalidArgumentError: if `low >= high` and `validate_args=False`.
    """
    parameters = dict(locals())
    with tf.name_scope(name, values=[low, high]) as name:
      dtype = dtype_util.common_dtype([low, high], tf.float32)
      low = tf.convert_to_tensor(low, name="low", dtype=dtype)
      high = tf.convert_to_tensor(high, name="high", dtype=dtype)
      with tf.control_dependencies([
          tf.assert_less(
              low, high, message="uniform not defined when low >= high.")
      ] if validate_args else []):
        self._low = tf.identity(low)
        self._high = tf.identity(high)
        tf.assert_same_float_dtype([self._low, self._high])
    super(Uniform, self).__init__(
        dtype=self._low.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._low,
                       self._high],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("low", "high"),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @property
  def low(self):
    """Lower boundary of the output interval."""
    return self._low

  @property
  def high(self):
    """Upper boundary of the output interval."""
    return self._high

  def range(self, name="range"):
    """`high - low`."""
    with self._name_scope(name):
      return self.high - self.low

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.low),
        tf.shape(self.high))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.low.shape,
        self.high.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    samples = tf.random_uniform(shape=shape,
                                dtype=self.dtype,
                                seed=seed)
    return self.low + self.range() * samples

  def _prob(self, x):
    broadcasted_x = x * tf.ones(
        self.batch_shape_tensor(), dtype=x.dtype)
    return tf.where(
        tf.is_nan(broadcasted_x),
        broadcasted_x,
        tf.where(
            tf.logical_or(broadcasted_x < self.low,
                          broadcasted_x >= self.high),
            tf.zeros_like(broadcasted_x),
            tf.ones_like(broadcasted_x) / self.range()))

  def _cdf(self, x):
    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(x), self.batch_shape_tensor())
    zeros = tf.zeros(broadcast_shape, dtype=self.dtype)
    ones = tf.ones(broadcast_shape, dtype=self.dtype)
    broadcasted_x = x * ones
    result_if_not_big = tf.where(
        x < self.low, zeros, (broadcasted_x - self.low) / self.range())
    return tf.where(x >= self.high, ones, result_if_not_big)

  def _entropy(self):
    return tf.log(self.range())

  def _mean(self):
    return (self.low + self.high) / 2.

  def _variance(self):
    return tf.square(self.range()) / 12.

  def _stddev(self):
    return self.range() / math.sqrt(12.)
