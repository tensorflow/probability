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
"""The Triangular distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization


def _broadcast_to(tensor_to_broadcast, target_tensors):
  """Helper to broadcast a tensor using a list of target tensors."""
  output = tensor_to_broadcast
  for tensor in target_tensors:
    output += tf.zeros_like(tensor)
  return output


class Triangular(distribution.Distribution):
  r"""Triangular distribution with `low`, `high` and `peak` parameters.

  #### Mathematical Details

  The Triangular distribution is specified by two line segments in the plane,
  such that:

    * The first line segment starts at `(a, 0)` and ends at `(c, z)`.
    * The second line segment starts at `(c, z)` and ends at `(b, 0)`.

    ```none
    y

    ^
  z |           o  (c,z)
    |          / \
    |         /   \
    |        /     \
    | (a,0) /       \ (b,0)
  0 +------o---------o-------> x
    0      a    c    b
  ```

  where:
  * a <= c <= b, a < b
  * `low = a`,
  * `high = b`,
  * `peak = c`,
  * `z = 2 / (b - a)`

  The parameters `low`, `high` and `peak` must be shaped in a way that supports
  broadcasting (e.g., `high - low` is a valid operation).

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Specify a single Triangular distribution.
  u1 = tfd.Triangular(low=3., high=4., peak=3.5)
  u1.mean()
  # ==> 3.5

  # Specify two different Triangular distributions.
  u2 = tfd.Triangular(low=[1., 2.], high=[3., 4.], peak=[2., 3.])
  u2.mean()
  # ==> [2., 3.]

  # Specify three different Triangular distributions by leveraging broadcasting.
  u3 = tfd.Triangular(low=3., high=[5., 6., 7.], peak=3.)
  u3.mean()
  # ==> [3.6666, 4., 4.3333]
  ```

  """

  def __init__(self,
               low=0.,
               high=1.,
               peak=0.5,
               validate_args=False,
               allow_nan_stats=True,
               name="Triangular"):
    """Initialize a batch of Triangular distributions.

    Args:
      low: Floating point tensor, lower boundary of the output interval. Must
        have `low < high`.
        Default value: `0`.
      high: Floating point tensor, upper boundary of the output interval. Must
        have `low < high`.
        Default value: `1`.
      peak: Floating point tensor, mode of the output interval. Must have
        `low <= peak` and `peak <= high`.
        Default value: `0.5`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `'Triangular'`.

    Raises:
      InvalidArgumentError: if `validate_args=True` and one of the following is
        True:
        * `low >= high`.
        * `peak > high`.
        * `low > peak`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([low, high, peak], tf.float32)
      low = tf.convert_to_tensor(value=low, name="low", dtype=dtype)
      high = tf.convert_to_tensor(value=high, name="high", dtype=dtype)
      peak = tf.convert_to_tensor(value=peak, name="peak", dtype=dtype)

      with tf.control_dependencies([
          assert_util.assert_less(
              low, high, message="triangular not defined when low >= high."),
          assert_util.assert_less_equal(
              low, peak, message="triangular not defined when low > peak."),
          assert_util.assert_less_equal(
              peak, high, message="triangular not defined when peak > high."),
      ] if validate_args else []):
        self._low = tf.identity(low, name="low")
        self._high = tf.identity(high, name="high")
        self._peak = tf.identity(peak, name="peak")
        dtype_util.assert_same_float_dtype(
            [self._low, self._high, self._peak])
    super(Triangular, self).__init__(
        dtype=self._low.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._low, self._high, self._peak],
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(low=0, high=0, peak=0)

  @property
  def low(self):
    """Lower boundary of the interval."""
    return self._low

  @property
  def high(self):
    """Upper boundary of the interval."""
    return self._high

  @property
  def peak(self):
    """Peak of the distribution. Lies in the interval."""
    return self._peak

  def _pdf_at_peak(self):
    """Pdf evaluated at the peak."""
    return (self.peak - self.low) / (self.high - self.low)

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(input=self.peak),
        tf.broadcast_dynamic_shape(
            tf.shape(input=self.low), tf.shape(input=self.high)))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.peak.shape,
        tf.broadcast_static_shape(
            self.low.shape, self.high.shape))

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    stream = seed_stream.SeedStream(seed, salt="triangular")
    shape = tf.concat([[n], self.batch_shape_tensor()], axis=0)
    samples = tf.random.uniform(shape=shape, dtype=self.dtype, seed=stream())
    # We use Inverse CDF sampling here. Because the CDF is a quadratic function,
    # we must use sqrts here.
    interval_length = self.high - self.low
    return tf.where(
        # Note the CDF on the left side of the peak is
        # (x - low) ** 2 / ((high - low) * (peak - low)).
        # If we plug in peak for x, we get that the CDF at the peak
        # is (peak - low) / (high - low). Because of this we decide
        # which part of the piecewise CDF we should use based on the cdf samples
        # we drew.
        samples < (self.peak - self.low) / interval_length,
        # Inverse of (x - low) ** 2 / ((high - low) * (peak - low)).
        self.low + tf.sqrt(
            samples * interval_length * (self.peak - self.low)),
        # Inverse of 1 - (high - x) ** 2 / ((high - low) * (high - peak))
        self.high - tf.sqrt(
            (1. - samples) * interval_length * (self.high - self.peak)))

  def _prob(self, x):
    if self.validate_args:
      with tf.control_dependencies([
          assert_util.assert_greater_equal(x, self.low),
          assert_util.assert_less_equal(x, self.high)
      ]):
        x = tf.identity(x)

    broadcast_x_to_high = _broadcast_to(x, [self.high])
    left_of_peak = tf.logical_and(
        broadcast_x_to_high >= self.low, broadcast_x_to_high <= self.peak)

    interval_length = self.high - self.low
    # This is the pdf function when a low <= high <= x. This looks like
    # a triangle, so we have to treat each line segment separately.
    result_inside_interval = tf.where(
        left_of_peak,
        # Line segment from (self.low, 0) to (self.peak, 2 / (self.high -
        # self.low).
        2. * (x - self.low) / (interval_length * (self.peak - self.low)),
        # Line segment from (self.peak, 2 / (self.high - self.low)) to
        # (self.high, 0).
        2. * (self.high - x) / (interval_length * (self.high - self.peak)))

    broadcast_x_to_peak = _broadcast_to(x, [self.peak])
    outside_interval = tf.logical_or(
        broadcast_x_to_peak < self.low, broadcast_x_to_peak > self.high)

    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(input=x), self.batch_shape_tensor())

    return tf.where(
        outside_interval,
        tf.zeros(broadcast_shape, dtype=self.dtype),
        result_inside_interval)

  def _cdf(self, x):
    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(input=x), self.batch_shape_tensor())

    broadcast_x_to_high = _broadcast_to(x, [self.high])
    left_of_peak = tf.logical_and(
        broadcast_x_to_high > self.low, broadcast_x_to_high <= self.peak)

    interval_length = self.high - self.low
    # Due to the PDF being not smooth at the peak, we have to treat each side
    # somewhat differently. The PDF is two line segments, and thus we get
    # quadratics here for the CDF.
    result_inside_interval = tf.where(
        left_of_peak,
        # (x - low) ** 2 / ((high - low) * (peak - low))
        tf.math.squared_difference(x, self.low) / (interval_length *
                                                   (self.peak - self.low)),
        # 1 - (high - x) ** 2 / ((high - low) * (high - peak))
        1. - tf.math.squared_difference(self.high, x) /
        (interval_length * (self.high - self.peak)))

    broadcast_x_to_high_peak = _broadcast_to(broadcast_x_to_high, [self.peak])
    zeros = tf.zeros(broadcast_shape, dtype=self.dtype)
    # We now add that the left tail is 0 and the right tail is 1.
    result_if_not_big = tf.where(
        broadcast_x_to_high_peak < self.low, zeros, result_inside_interval)

    broadcast_x_to_peak_low = _broadcast_to(x, [self.low, self.peak])
    ones = tf.ones(broadcast_shape, dtype=self.dtype)
    return tf.where(
        broadcast_x_to_peak_low >= self.high, ones, result_if_not_big)

  def _entropy(self):
    return 0.5 - np.log(2.) + tf.math.log(self.high - self.low)

  def _mean(self):
    return (self.low + self.high + self.peak) / 3.

  def _variance(self):
    # ((high - low) ** 2 + (peak - low) ** 2 + (peak - high) ** 2) / 36
    return (tf.math.squared_difference(self.high, self.low) +
            tf.math.squared_difference(self.high, self.peak) +
            tf.math.squared_difference(self.peak, self.low)) / 36.
