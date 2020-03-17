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
"""LogNormal distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'LogLogistic',
]


class LogLogistic(transformed_distribution.TransformedDistribution):
  """The log-logistic distribution."""

  def __init__(self,
               alpha,
               beta,
               validate_args=False,
               allow_nan_stats=True,
               name='LogLogistic'):
    """Construct a log-logistic distribution.

    The LogLogistic distribution models positive-valued random variables
    whose logarithm is logisticly distributed with mean `log(alpha)` and
    scale `1/beta`. It is constructed as the exponential
    transformation of a Logistic distribution.

    Args:
      alpha: Floating-point `Tensor`;
        the scale of the log-Logistic distribution(s).
      scale: Floating-point `Tensor`;
        the shape of the log- Logistic distribution(s).
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([alpha, beta], dtype_hint=tf.float32)
      self._alpha = tensor_util.convert_nonref_to_tensor(
          alpha, name='alpha', dtype=dtype)
      self._beta = tensor_util.convert_nonref_to_tensor(
          beta, name='beta', dtype=dtype)
      super(LogLogistic, self).__init__(
          distribution=logistic.Logistic(loc=tf.math.log(self.alpha),
                                         scale=1./self.beta),
          bijector=exp_bijector.Exp(),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(alpha=0, beta=0)

  @property
  def loc(self):
    """Distribution parameter for the pre-transformed mean."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for the pre-transformed scale."""
    return self.distribution.scale

  @property
  def alpha(self):
    """Distribution parameter."""
    return self._alpha

  @property
  def beta(self):
    """Distribution parameter."""
    return self._beta

  def _mean(self):
    b = 1. / self.beta
    mean = self.alpha / sinc(b)
    return tf.where(tf.greater(self.beta, 1.), mean, np.nan)

  def _variance(self):
    b = 1. / self.beta
    variance = self.alpha ** 2 * (1. / sinc(2*b) - 1. / sinc(b)**2)
    return tf.where(tf.greater(self.beta, 2.), variance, np.nan)

  def _mode(self):
    mode = self.alpha * ((self.beta - 1.)/(self.beta + 1.))**(1./self.beta)
    return tf.where(tf.greater(self.beta, 1.), mode, np.nan)

  def _entropy(self):
    return (tf.math.log(self.alpha / self.beta) + 2.) / tf.math.log(2.)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if is_init:
      dtype_util.assert_same_float_dtype([self.alpha, self.beta])
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._alpha):
      assertions.append(assert_util.assert_positive(
          self._alpha, message='Argument `alpha` must be positive.'))
    if is_init != tensor_util.is_ref(self._beta):
      assertions.append(assert_util.assert_positive(
          self._beta, message='Argument `beta` must be positive.'))
    return assertions

  def _default_event_space_bijector(self):
    return exp_bijector.Exp(validate_args=self.validate_args)


def sinc(x, name=None):
  """Calculate the (normalized) sinus cardinal of x"""
  name = name or "sinc"
  with tf.name_scope(name):
    x *= np.pi
    return tf.math.sin(x)/x
