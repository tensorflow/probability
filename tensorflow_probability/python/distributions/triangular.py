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

from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


class Triangular(distribution.AutoCompositeTensorDistribution):
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
               name='Triangular'):
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
      self._low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
      self._high = tensor_util.convert_nonref_to_tensor(
          high, name='high', dtype=dtype)
      self._peak = tensor_util.convert_nonref_to_tensor(
          peak, name='peak', dtype=dtype)
      super(Triangular, self).__init__(
          dtype=self._low.dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        low=parameter_properties.ParameterProperties(),
        # TODO(b/169874884): Support decoupled parameterization.
        high=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED,),
        # TODO(b/169874884): Support decoupled parameterization.
        peak=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED,))

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

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    peak = tf.convert_to_tensor(self.peak)

    seed = samplers.sanitize_seed(seed, salt='triangular')
    shape = ps.concat([[n], self._batch_shape_tensor(
        low=low, high=high, peak=peak)], axis=0)
    samples = samplers.uniform(shape=shape, dtype=self.dtype, seed=seed)
    # We use Inverse CDF sampling here. Because the CDF is a quadratic function,
    # we must use sqrts here.
    interval_length = high - low
    return tf.where(
        # Note the CDF on the left side of the peak is
        # (x - low) ** 2 / ((high - low) * (peak - low)).
        # If we plug in peak for x, we get that the CDF at the peak
        # is (peak - low) / (high - low). Because of this we decide
        # which part of the piecewise CDF we should use based on the cdf samples
        # we drew.
        samples < (peak - low) / interval_length,
        # Inverse of (x - low) ** 2 / ((high - low) * (peak - low)).
        low + tf.sqrt(samples * interval_length * (peak - low)),
        # Inverse of 1 - (high - x) ** 2 / ((high - low) * (high - peak))
        high - tf.sqrt((1. - samples) * interval_length * (high - peak)))

  def _prob(self, x):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    peak = tf.convert_to_tensor(self.peak)

    interval_length = high - low
    # This is the pdf function when a low <= high <= x. This looks like
    # a triangle, so we have to treat each line segment separately.
    result_inside_interval = tf.where(
        (x >= low) & (x <= peak),
        # Line segment from (low, 0) to (peak, 2 / (high - low)).
        2. * (x - low) / (interval_length * (peak - low)),
        # Line segment from (peak, 2 / (high - low)) to (high, 0).
        2. * (high - x) / (interval_length * (high - peak)))

    return tf.where((x < low) | (x > high),
                    tf.zeros_like(x),
                    result_inside_interval)

  def _cdf(self, x):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    peak = tf.convert_to_tensor(self.peak)

    interval_length = high - low
    # Due to the PDF being not smooth at the peak, we have to treat each side
    # somewhat differently. The PDF is two line segments, and thus we get
    # quadratics here for the CDF.
    result_inside_interval = tf.where(
        (x >= low) & (x <= peak),
        # (x - low) ** 2 / ((high - low) * (peak - low))
        tf.math.squared_difference(x, low) / (interval_length * (peak - low)),
        # 1 - (high - x) ** 2 / ((high - low) * (high - peak))
        1. - tf.math.squared_difference(high, x) / (
            interval_length * (high - peak)))

    # We now add that the left tail is 0 and the right tail is 1.
    result_if_not_big = tf.where(
        x < low, tf.zeros_like(x), result_inside_interval)

    return tf.where(x >= high, tf.ones_like(x), result_if_not_big)

  def _entropy(self):
    ans = 0.5 - np.log(2.) + tf.math.log(self.high - self.low)
    return tf.broadcast_to(ans, self._batch_shape_tensor())

  def _mean(self):
    return (self.low + self.high + self.peak) / 3.

  def _variance(self):
    # ((high - low) ** 2 + (peak - low) ** 2 + (peak - high) ** 2) / 36
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    peak = tf.convert_to_tensor(self.peak)
    return (tf.math.squared_difference(high, low) +
            tf.math.squared_difference(high, peak) +
            tf.math.squared_difference(peak, low)) / 36.

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(
        low=self.low, high=self.high, validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    peak = tf.convert_to_tensor(self.peak)
    assertions = []
    if (is_init != tensor_util.is_ref(self.low) and
        is_init != tensor_util.is_ref(self.high)):
      assertions.append(assert_util.assert_less(
          low, high, message='triangular not defined when low >= high.'))
    if (is_init != tensor_util.is_ref(self.low) and
        is_init != tensor_util.is_ref(self.peak)):
      assertions.append(
          assert_util.assert_less_equal(
              low, peak, message='triangular not defined when low > peak.'))
    if (is_init != tensor_util.is_ref(self.high) and
        is_init != tensor_util.is_ref(self.peak)):
      assertions.append(
          assert_util.assert_less_equal(
              peak, high, message='triangular not defined when peak > high.'))

    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_greater_equal(
        x, self.low, message='Sample must be greater than or equal to `low`.'))
    assertions.append(assert_util.assert_less_equal(
        x, self.high, message='Sample must be less than or equal to `high`.'))
    return assertions
