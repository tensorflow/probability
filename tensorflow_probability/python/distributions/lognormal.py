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
import tensorflow as tf
from tensorflow_probability.python import bijectors
from tensorflow.python.ops.distributions import transformed_distribution


__all__ = [
    "LogNormal",
]


class LogNormal(transformed_distribution.TransformedDistribution):
  """The log-normal distribution."""

  def __init__(self,
               loc=None,
               scale=None,
               validate_args=False,
               allow_nan_stats=True,
               name="LogNormal"):
    """Construct a log-normal distribution.

    The LogNormal distribution models positive-valued random variables
    whose logarithm is normally distributed with mean `loc` and
    standard deviation `scale`. It is constructed as the exponential
    transformation of a Normal distribution.

    Args:
      loc: Floating-point `Tensor`; the means of the underlying
        Normal distribution(s).
      scale: Floating-point `Tensor`; the stddevs of the underlying
        Normal distribution(s).
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    with tf.name_scope(name, values=[loc, scale]) as name:
      super(LogNormal, self).__init__(
          distribution=tf.distributions.Normal(
              loc=loc,
              scale=scale),
          bijector=bijectors.Exp(),
          validate_args=validate_args,
          name=name)

  @property
  def loc(self):
    """Distribution parameter for the pre-transformed mean."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for the pre-transformed standard deviation."""
    return self.distribution.scale

  def _mean(self):
    return tf.exp(self.distribution.mean() + 0.5 * self.distribution.variance())

  def _variance(self):
    variance = self.distribution.variance()
    return (tf.expm1(variance) *
            tf.exp(2. * self.distribution.mean() + variance))

  def _mode(self):
    return tf.exp(self.distribution.mean() - self.distribution.variance())

  def _entropy(self):
    return (self.distribution.mean() + 0.5 +
            tf.log(self.distribution.stddev()) +
            0.5 * np.log(2 * np.pi))
