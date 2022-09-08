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
"""The InverseGamma distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import reciprocal as reciprocal_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import special


__all__ = [
    'InverseGamma',
]


class InverseGamma(distribution.AutoCompositeTensorDistribution):
  """InverseGamma distribution.

  The `InverseGamma` distribution is defined over positive real numbers using
  parameters `concentration` (aka "alpha") and `scale` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta, x > 0) = x**(-alpha - 1) exp(-beta / x) / Z
  Z = Gamma(alpha) beta**-alpha
  ```

  where:

  * `concentration = alpha`,
  * `scale = beta`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta / x) / Gamma(alpha)
  ```

  where `GammaInc` is the [upper incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  The parameters can be intuited via their relationship to mean and variance
  when these moments exist,

  ```none
  mean = beta / (alpha - 1)                           when alpha > 1
  variance = beta**2 / (alpha - 1)**2 / (alpha - 2)   when alpha > 2
  ```

  i.e., under the same conditions:

  ```none
  alpha = mean**2 / variance + 2
  beta = mean * (mean**2 / variance + 1)
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

  #### Examples

  ```python
  tfd = tfp.distributions
  dist = tfd.InverseGamma(concentration=3.0, scale=2.0)
  dist2 = tfd.InverseGamma(concentration=[3.0, 4.0], scale=[2.0, 3.0])
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  tfd = tfp.distributions
  concentration = tf.constant(3.0)
  scale = tf.constant(2.0)
  dist = tfd.InverseGamma(concentration, scale)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [concentration, scale])
  ```

  """

  def __init__(self,
               concentration,
               scale=None,
               validate_args=False,
               allow_nan_stats=True,
               name='InverseGamma'):
    """Construct InverseGamma with `concentration` and `scale` parameters.

    The parameters `concentration` and `scale` must be shaped in a way that
    supports broadcasting (e.g. `concentration + scale` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      scale: Floating point tensor, the scale params of the distribution(s).
        Must contain only positive values.
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
      TypeError: if `concentration` and `scale` are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [concentration, scale], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      super(InverseGamma, self).__init__(
          dtype=self._concentration.dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def scale(self):
    """Scale parameter."""
    return self._scale

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    return tf.math.exp(-gamma_lib.random_gamma(
        shape=[n],
        concentration=self.concentration,
        rate=self.scale,
        seed=seed,
        log_space=True))

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    scale = tf.convert_to_tensor(self.scale)
    unnormalized_prob = -(1. + concentration) * tf.math.log(x) - scale / x
    normalization = (
        tf.math.lgamma(concentration) - concentration * tf.math.log(scale))
    return unnormalized_prob - normalization

  def _cdf(self, x):
    # Note that igammac returns the upper regularized incomplete gamma
    # function Q(a, x), which is what we want for the CDF.
    return tf.math.igammac(self.concentration, self.scale / x)

  def _quantile(self, p):
    return tf.math.reciprocal(
        special.igammacinv(self.concentration, p)) * self.scale

  def _entropy(self):
    concentration = tf.convert_to_tensor(self.concentration)
    scale = tf.convert_to_tensor(self.scale)
    return (concentration + tf.math.log(scale) +
            tf.math.lgamma(concentration) -
            ((1. + concentration) * tf.math.digamma(concentration)))

  @distribution_util.AppendDocstring(
      """The mean of an inverse gamma distribution is
      `scale / (concentration - 1)`, when `concentration > 1`, and `NaN`
      otherwise. If `self.allow_nan_stats` is `False`, an exception will be
      raised rather than returning `NaN`""")
  def _mean(self):
    concentration = tf.convert_to_tensor(self.concentration)
    scale = tf.convert_to_tensor(self.scale)
    mean = scale / (concentration - 1.)
    if self.allow_nan_stats:
      assertions = []
    else:
      assertions = [assert_util.assert_less(
          tf.ones([], self.dtype), concentration,
          message='mean undefined when any concentration <= 1')]
    with tf.control_dependencies(assertions):
      return tf.where(
          concentration > 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))

  @distribution_util.AppendDocstring(
      """Variance for inverse gamma is defined only for `concentration > 2`. If
      `self.allow_nan_stats` is `False`, an exception will be raised rather
      than returning `NaN`.""")
  def _variance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    scale = tf.convert_to_tensor(self.scale)
    var = (
        tf.square(scale) / tf.square(concentration - 1.) /
        (concentration - 2.))
    if self.allow_nan_stats:
      assertions = []
    else:
      assertions = [assert_util.assert_less(
          tf.constant(2., dtype=self.dtype),
          concentration,
          message='variance undefined when any concentration <= 2')]

    with tf.control_dependencies(assertions):
      return tf.where(
          concentration > 2.,
          var,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))

  @distribution_util.AppendDocstring(
      """The mode of an inverse gamma distribution is `scale / (concentration +
      1)`.""")
  def _mode(self):
    return self.scale / (1. + self.concentration)

  def _default_event_space_bijector(self):
    return chain_bijector.Chain([
        reciprocal_bijector.Reciprocal(validate_args=self.validate_args),
        softplus_bijector.Softplus(validate_args=self.validate_args)
    ], validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
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
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale,
          message='Argument `scale` must be positive.'))
    return assertions
