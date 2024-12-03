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
"""The BetaQuotient distribution class."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import beta as beta_lib
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import hypergeometric as hypgeo
from tensorflow_probability.python.math import special

__all__ = [
    'BetaQuotient',
]


class BetaQuotient(distribution.AutoCompositeTensorDistribution):
  """BetaQuotient distribution.

  The Beta Quotient distribution is defined over the positive reals, as
  the ratio of two Independent Beta distributed random variables.

  In other words:

  ```none
  X ~ Beta(a0, b0)
  Y ~ Beta(a1, b1)
  X / Y ~ BetaQuotient(a0, b0, a1, b1)
  ```

  The distribution is defined over the positive reals, by four parameters
  `concentration0_numerator`, `concentration1_numerator`,
  `concentration0_denominator` and `concentration1_denominator`
  (aka `beta` and `alpha` of the numerator and denominator Beta distribution
  respectively).

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: The samples can be zero or inf due to finite precision.
  This happens more often when some of the concentrations are very small.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in [3].

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Create a batch of three BetaQuotient distributions.
  alpha0 = [1, 2, 3]
  alpha1 = [5]
  beta0 = [1, 2, 3]
  beta1 = [0.4]
  dist = tfd.BetaQuotient(alpha0, beta0, alpha1, beta1)

  dist.sample([4, 5])  # Shape [4, 5, 3]

  # `x` has three batch entries, each with two samples.
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  # Calculate the probability of each pair of samples under the corresponding
  # distribution in `dist`.
  dist.prob(x)         # Shape [2, 3]
  ```

  #### References
  [1] T. Pham-Gia, Distributions of the ratios of independent beta variables
      and applications. Communications in Statistics, Theory and Methods.
      Volume 29.

  [2] S. Nadarajah, Sums, products and ratios of generalized beta variables.
      Statistical Papers, 47.

  [3] Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit
      Reparameterization Gradients. https://arxiv.org/abs/1805.08498
  """

  def __init__(self,
               concentration1_numerator,
               concentration0_numerator,
               concentration1_denominator,
               concentration0_denominator,
               validate_args=False,
               allow_nan_stats=True,
               name='BetaQuotient'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([
          concentration1_numerator,
          concentration0_numerator,
          concentration1_denominator,
          concentration0_denominator], dtype_hint=tf.float32)
      self._concentration1_numerator = tensor_util.convert_nonref_to_tensor(
          concentration1_numerator,
          dtype=dtype, name='concentration1_numerator')
      self._concentration0_numerator = tensor_util.convert_nonref_to_tensor(
          concentration0_numerator,
          dtype=dtype, name='concentration0_numerator')
      self._concentration1_denominator = tensor_util.convert_nonref_to_tensor(
          concentration1_denominator,
          dtype=dtype, name='concentration1_denominator')
      self._concentration0_denominator = tensor_util.convert_nonref_to_tensor(
          concentration0_denominator,
          dtype=dtype, name='concentration0_denominator')
      super(BetaQuotient, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        concentration1_numerator=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration0_numerator=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration1_denominator=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration0_denominator=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def concentration1_numerator(self):
    """Concentration parameter associated with a `1` outcome."""
    return self._concentration1_numerator

  @property
  def concentration0_numerator(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._concentration0_numerator

  @property
  def concentration1_denominator(self):
    """Concentration parameter associated with a `1` outcome."""
    return self._concentration1_denominator

  @property
  def concentration0_denominator(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._concentration0_denominator

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    alpha0 = tf.convert_to_tensor(self.concentration1_numerator)
    beta0 = tf.convert_to_tensor(self.concentration0_numerator)
    alpha1 = tf.convert_to_tensor(self.concentration1_denominator)
    beta1 = tf.convert_to_tensor(self.concentration0_denominator)

    batch_shape = self._batch_shape_tensor(
        concentration1_numerator=alpha0,
        concentration0_numerator=beta0,
        concentration1_denominator=alpha1,
        concentration0_denominator=beta1)

    broadcasted_alpha0 = tf.broadcast_to(alpha0, batch_shape)
    broadcasted_alpha1 = tf.broadcast_to(alpha1, batch_shape)

    seed1, seed2 = samplers.split_seed(seed, salt='beta_quotient')
    numerator = beta_lib.Beta(broadcasted_alpha0, beta0)
    denominator = beta_lib.Beta(broadcasted_alpha1, beta1)
    return numerator.sample(n, seed=seed1) / denominator.sample(n, seed=seed2)

  def _log_prob(self, x):
    alpha0 = tf.convert_to_tensor(self.concentration1_numerator)
    beta0 = tf.convert_to_tensor(self.concentration0_numerator)
    alpha1 = tf.convert_to_tensor(self.concentration1_denominator)
    beta1 = tf.convert_to_tensor(self.concentration0_denominator)

    alpha_sum = alpha0 + alpha1

    log_normalization = (
        special.lbeta(alpha0, beta0) + special.lbeta(alpha1, beta1))
    log_normalization = log_normalization - special.lbeta(
        alpha_sum, tf.where(x > 1., beta0, beta1))

    b = tf.where(x > 1., 1. - beta1, 1 - beta0)
    c = alpha_sum + tf.where(x > 1., beta0, beta1)
    z = tf.where(x > 1., tf.math.reciprocal(x), x)

    # Here, c - a - b = beta0 + beta1 - 1, so the series always converges
    # conditionally.

    log_unnormalized_prob = tf.math.log(
        hypgeo.hyp2f1_small_argument(alpha_sum, b, c, z))
    log_unnormalized_prob = log_unnormalized_prob + tf.math.xlogy(
        tf.where(x > 1., -(alpha1 + 1.), alpha0 - 1.), x)

    return log_unnormalized_prob - log_normalization

  @distribution_util.AppendDocstring(
      """Expectation for beta quotient is defined only for
      `denominator_concentration > 1`. If `self.allow_nan_stats` is `False`,
      an exception will be raised rather than returning `NaN`.""")
  def _mean(self):
    alpha0 = tf.convert_to_tensor(self.concentration1_numerator)
    beta0 = tf.convert_to_tensor(self.concentration0_numerator)
    alpha1 = tf.convert_to_tensor(self.concentration1_denominator)
    beta1 = tf.convert_to_tensor(self.concentration0_denominator)
    mean = alpha0 * (alpha1 + beta1 - 1.) / ((alpha0 + beta0) * (alpha1 - 1.))

    if self.allow_nan_stats:
      assertions = []
    else:
      assertions = [assert_util.assert_less(
          tf.ones([], self.dtype), alpha1,
          message='mean undefined when any denominator_concentration <= 1')]
    with tf.control_dependencies(assertions):
      return tf.where(
          alpha1 > 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))

  def _sample_control_dependencies(self, x):
    """Checks the validity of a sample."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    for c in [
        self.concentration0_numerator,
        self.concentration1_numerator,
        self.concentration0_denominator,
        self.concentration1_denominator]:
      if is_init != tensor_util.is_ref(c):
        assertions.append(assert_util.assert_positive(
            c,
            message='`concentration` must be positive.'))
    return assertions

  def _default_event_space_bijector(self):
    return exp_bijector.Exp(validate_args=self.validate_args)
