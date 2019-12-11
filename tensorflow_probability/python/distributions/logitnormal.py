# Copyright 2019 The TensorFlow Probability Authors.
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
"""LogitNormal distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util


__all__ = [
    'LogitNormal',
]


class LogitNormal(transformed_distribution.TransformedDistribution):
  """The logit-normal distribution."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='LogitNormal'):
    """Construct a logit-normal distribution.

    The LogititNormal distribution models positive-valued random variables whose
    logit (i.e., sigmoid_inverse, i.e., `log(p) - log1p(-p)`) is normally
    distributed with mean `loc` and standard deviation `scale`. It is
    constructed as the sigmoid transformation, (i.e., `1 / (1 + exp(-x))`) of a
    Normal distribution.

    Args:
      loc: Floating-point `Tensor`; the mean of the underlying
        Normal distribution(s). Must broadcast with `scale`.
      scale: Floating-point `Tensor`; the stddev of the underlying
        Normal distribution(s). Must broadcast with `loc`.
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
      super(LogitNormal, self).__init__(
          distribution=normal.Normal(loc=loc, scale=scale),
          bijector=sigmoid_bijector.Sigmoid(),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0)

  @property
  def loc(self):
    """Distribution parameter for the pre-transformed mean."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for the pre-transformed standard deviation."""
    return self.distribution.scale

  def mean_approx(self, name='mean_approx'):
    """Approximate the mean of a LogitNormal.

    Warning: accuracy is not guaranteed and can be large for small `loc` values.

    This calculation is `sigmoid(loc / sqrt(1 + np.pi / 8 * scale**2.))`
    and based on the idea that [sigmoid(x) approximately equals
    normal(0,1).cdf(sqrt(pi / 8) * x)](
    https://en.wikipedia.org/wiki/Logit#Comparison_with_probit).

    Derivation:

    ```None
    int[ sigmoid(x) phi(x; m, s), x]
    ~= int[ Phi(sqrt(pi / 8) x) phi(x; m, s), x]
     = Phi(sqrt(pi / 8) m / sqrt(1 + pi / 8 s**2))
    ~= sigmoid(m / sqrt(1 + pi/8 s**2))
    ```
    where the third line comes from [this table](
    https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions)
    or using the [difference of two independent standard Normals](
    https://math.stackexchange.com/a/1800648).

    See [this note](
    https://threeplusone.com/pubs/on_logistic_normal.pdf) for additional
    references.

    Args:
      name: Python `str` prepended to names of ops created by this function.
        Default value: `'mean_approx'`.

    Returns:
      mean_approx: An approximation of the mean of a LogitNormal.
    """
    with self._name_and_control_scope(name):
      b2 = (np.pi / 8.) * tf.square(self.scale)
      return tf.math.sigmoid(self.loc * tf.math.rsqrt(1. + b2))

  # TODO(b/143252788): Add `variance_approx` once owens_t is in TFP.
  # def variance_approx(self, name='variance_approx'):  # Needs verification.
  #   with self._name_and_control_scope(name):
  #     m = tf.convert_to_tensor(self.loc)
  #     b2 = (np.pi / 8.) * tf.square(self.scale)
  #     a = m * tf.math.rsqrt(1. + b2)
  #     return tf.math.sigmoid(a) * tf.math.sigmoid(-a) - 2. * owens_t(
  #         np.sqrt(np.pi / 8.) * m, tf.math.rsqrt(1. + 2. * b2))
  # def stddev_approx(self, name='stddev_approx'):
  #   with tf.name_scope(name):
  #     return tf.math.sqrt(self.variance_approx())

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    assertions.append(assert_util.assert_less_equal(
        x, tf.ones([], x.dtype),
        message='Sample must be less than or equal to `1`.'))
    return assertions


@kullback_leibler.RegisterKL(LogitNormal, LogitNormal)
def _kl_logitnormal_logitnormal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b LogitNormal.

  This is the same as the KL divergence between the underlying Normal
  distributions.

  Args:
    a: instance of a LogitNormal distribution object.
    b: instance of a LogitNormal distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_logitnormal_logitnormal'`).

  Returns:
    kl_div: Batchwise KL(a || b)
  """
  return kullback_leibler.kl_divergence(
      a.distribution,
      b.distribution,
      name=(name or 'kl_logitnormal_logitnormal'))
