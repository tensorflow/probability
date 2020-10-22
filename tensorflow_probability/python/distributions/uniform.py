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

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


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
               name='Uniform'):
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
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([low, high], tf.float32)
      self._low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
      self._high = tensor_util.convert_nonref_to_tensor(
          high, name='high', dtype=dtype)
      super(Uniform, self).__init__(
          dtype=dtype,
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
            .BIJECTOR_NOT_IMPLEMENTED,))

  @property
  def low(self):
    """Lower boundary of the output interval."""
    return self._low

  @property
  def high(self):
    """Upper boundary of the output interval."""
    return self._high

  def range(self, name='range'):
    """`high - low`."""
    with self._name_and_control_scope(name):
      return self._range()

  def _range(self, low=None, high=None):
    low = self.low if low is None else low
    high = self.high if high is None else high
    return high - low

  def _batch_shape_tensor(self, low=None, high=None):
    return ps.broadcast_shape(
        ps.shape(self.low if low is None else low),
        ps.shape(self.high if high is None else high))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.low.shape,
        self.high.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    shape = ps.concat([[n], self._batch_shape_tensor(
        low=low, high=high)], 0)
    samples = samplers.uniform(shape=shape, dtype=self.dtype, seed=seed)
    return low + self._range(low=low, high=high) * samples

  def _prob(self, x):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    return tf.where(
        tf.math.is_nan(x),
        x,
        tf.where(
            # This > is only sound for continuous uniform
            (x < low) | (x > high),
            tf.zeros_like(x),
            tf.ones_like(x) / self._range(low=low, high=high)))

  def _cdf(self, x):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    batch_shape = self.batch_shape
    if not tensorshape_util.is_fully_defined(batch_shape):
      batch_shape = self._batch_shape_tensor(low=low, high=high)
    broadcast_shape = ps.broadcast_shape(
        ps.shape(x), batch_shape)
    zeros = tf.zeros(broadcast_shape, dtype=self.dtype)
    ones = tf.ones(broadcast_shape, dtype=self.dtype)
    result_if_not_big = tf.where(x < low, zeros,
                                 (x - low) / self._range(low=low, high=high))
    return tf.where(x >= high, ones, result_if_not_big)

  def _quantile(self, value):
    return (1. - value) * self.low + value * self.high

  def _entropy(self):
    return tf.math.log(self._range())

  def _mean(self):
    return (self.low + self.high) / 2.

  def _variance(self):
    return tf.square(self._range()) / 12.

  def _stddev(self):
    return self._range() / np.sqrt(12.)

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(
        low=self.low, high=self.high, validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    low = None
    high = None
    if is_init != tensor_util.is_ref(self.low):
      low = tf.convert_to_tensor(self.low)
      high = tf.convert_to_tensor(self.high)
      assertions.append(assert_util.assert_less(
          low, high, message='uniform not defined when `low` >= `high`.'))
    if is_init != tensor_util.is_ref(self.high):
      low = tf.convert_to_tensor(self.low) if low is None else low
      high = tf.convert_to_tensor(self.high) if high is None else high
      assertions.append(assert_util.assert_less(
          low, high, message='uniform not defined when `low` >= `high`.'))
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


@kullback_leibler.RegisterKL(Uniform, Uniform)
def _kl_uniform_uniform(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Uniform.

  Note that the KL divergence is infinite if the support of `a` is not a subset
  of the support of `b`.

  Args:
    a: instance of a Uniform distribution object.
    b: instance of a Uniform distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_uniform_uniform".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_uniform_uniform'):
    # Consistent with
    # http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 60
    # Watch out for the change in conventions--they use 'a' and 'b' to refer to
    # lower and upper bounds respectively there.
    dtype = dtype_util.common_dtype(
        [a.low, a.high, b.low, b.high], tf.float32)
    a_low = tf.convert_to_tensor(a.low)
    b_low = tf.convert_to_tensor(b.low)
    a_high = tf.convert_to_tensor(a.high)
    b_high = tf.convert_to_tensor(b.high)
    return tf.where((b_low <= a_low) & (a_high <= b_high),
                    tf.math.log(b_high - b_low) - tf.math.log(a_high - a_low),
                    dtype_util.as_numpy_dtype(dtype)(np.inf))
