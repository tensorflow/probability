# Copyright 2020 The TensorFlow Probability Authors.
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
"""LogLogistic distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties

__all__ = [
    'LogLogistic',
]


class LogLogistic(transformed_distribution.TransformedDistribution):
  """The log-logistic distribution."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='LogLogistic'):
    """Construct a log-logistic distribution.

    The LogLogistic distribution models positive-valued random variables
    whose logarithm is a logistic distribution with location `loc` and
    scale `scale`. It is constructed as the exponential
    transformation of a Logistic distribution.

    Args:
      loc: Floating-point `Tensor`; the location of the underlying logistic
        distribution(s).
      scale: Floating-point `Tensor`; the scale of the underlying logistic
        distribution(s).
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(LogLogistic, self).__init__(
          distribution=logistic.Logistic(
              loc=loc, scale=scale, allow_nan_stats=allow_nan_stats),
          bijector=exp_bijector.Exp(),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter of the underlying distribution."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter of the underlying distribution."""
    return self.distribution.scale

  def _mean(self):
    scale = tf.convert_to_tensor(self.scale)
    with tf.control_dependencies([] if self.allow_nan_stats else [  # pylint: disable=g-long-ternary
        assert_util.assert_less(
            scale,
            tf.ones([], dtype=self.dtype),
            message='Mean undefined for scale >= 1.'),
    ]):
      mean = tf.math.exp(self.loc) / sinc(scale)
      return tf.where(scale > 1., tf.cast(np.nan, dtype=self.dtype), mean)

  def _variance(self):
    scale = tf.convert_to_tensor(self.scale)
    with tf.control_dependencies([] if self.allow_nan_stats else [  # pylint: disable=g-long-ternary
        assert_util.assert_less(
            scale, tf.constant(0.5, self.dtype),
            message='Variance undefined for scale >= 1/2.'),
    ]):
      variance = tf.math.exp(
          2 * self.loc) * (1. / sinc(2 * scale) - 1. / sinc(scale)**2)
      return tf.where(scale > 0.5, tf.cast(np.nan, dtype=self.dtype), variance)

  def _mode(self):
    scale = tf.convert_to_tensor(self.scale)
    log_mode = self.loc + scale * (
        tf.math.log1p(-scale) - tf.math.log1p(scale))
    return tf.where(scale > 1., tf.cast(
        0., dtype=self.dtype), tf.math.exp(log_mode))

  def _entropy(self):
    return 2. + self.loc + tf.math.log(self.scale)

  def _log_z(self, x, loc=None, scale=None):
    """Returns log of the standardized input."""
    loc = self.loc if loc is None else loc
    scale = self.scale if scale is None else scale
    with tf.name_scope('log_standardize'):
      return (tf.math.log(x) - self.loc) / self.scale

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    loc = tf.convert_to_tensor(self.loc)
    log_z = self._log_z(x, loc=loc, scale=scale)
    return (-tf.math.log(scale) - loc +
            (1. - scale) * log_z - 2 * tf.math.softplus(log_z))

  def _log_cdf(self, x):
    return -tf.math.softplus(-self._log_z(x))

  def _log_survival_function(self, x):
    return self._log_cdf(x) - self._log_z(x)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(
        assert_util.assert_non_negative(
            x, message='Sample must be non-negative.'))
    return assertions

  def _default_event_space_bijector(self):
    return exp_bijector.Exp(validate_args=self.validate_args)


def sinc(x, name=None):
  """Calculate the (normalized) sinus cardinalis of x."""
  name = name or 'sinc'
  with tf.name_scope(name):
    x *= np.pi
    return tf.where(x != 0., tf.math.sin(x) / x, 1.)
