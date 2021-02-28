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
"""The ExpGamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import scale as scale_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'ExpGamma',
    'ExpInverseGamma',
]


class ExpGamma(distribution.Distribution):
  """ExpGamma distribution.

  The ExpGamma distribution is defined over the real line using
  parameters `concentration` (aka "alpha") and `rate` (aka "beta").

  This distribution is a transformation of the Gamma distribution such that
  X ~ ExpGamma(..) => exp(X) ~ Gamma(..).

  #### Mathematical Details

  The probability density function (pdf) can be derived from the change of
  variables rule (since the distribution is logically equivalent to
  `tfb.Log()(tfd.Gamma(..))`):

  ```none
  pdf(x; alpha, beta > 0) = exp(x)**(alpha) exp(-exp(x) beta) / Z
  Z = Gamma(alpha) beta**(-alpha)
  ```

  where:

  * `concentration = alpha`, `alpha > 0`,
  * `rate = beta`, `beta > 0`,
  * `Z` is the normalizing constant of the corresponding Gamma distribution, and
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x) = GammaInc(alpha, beta exp(x)) / Gamma(alpha)
  ```

  where `GammaInc` is the [lower incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in
  [(Figurnov et al., 2018)][1].

  #### Examples

  ```python
  tfd = tfp.distributions

  dist = tfd.ExpGamma(concentration=3.0, rate=2.0)
  dist2 = tfd.ExpGamma(concentration=[3.0, 4.0], rate=[2.0, 3.0])
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  concentration = tf.constant(3.0)
  rate = tf.constant(2.0)
  dist = tfd.ExpGamma(concentration, rate)
  with tf.GradientTape() as t:
    t.watch([concentration, rate])
    samples = dist.sample(5)  # Shape [5]
    loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = t.gradient(loss, [concentration, rate])
  ```

  #### References

  [1]: Michael Figurnov, Shakir Mohamed, Andriy Mnih.
       Implicit Reparameterization Gradients. _arXiv preprint arXiv:1805.08498_,
       2018. https://arxiv.org/abs/1805.08498

  """

  def __init__(self,
               concentration,
               rate=None,
               log_rate=None,
               validate_args=False,
               allow_nan_stats=True,
               name='ExpGamma'):
    """Construct ExpGamma with `concentration` and `rate` parameters.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g. `concentration + rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values. Mutually exclusive
        with `log_rate`.
      log_rate: Floating point tensor, natural logarithm of the inverse scale
        params of the distribution(s). Mutually exclusive with `rate`.
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
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    if (rate is None) == (log_rate is None):
      raise ValueError(
          'Exactly one of `rate` and `log_rate` must be specified.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [concentration, rate, log_rate], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, dtype=dtype, name='rate')
      self._log_rate = tensor_util.convert_nonref_to_tensor(
          log_rate, dtype=dtype, name='log_rate')

      super(ExpGamma, self).__init__(
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
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False),
        log_rate=parameter_properties.ParameterProperties())
    # pylint: enable=g-long-lambda

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  @property
  def log_rate(self):
    """Log-rate parameter."""
    return self._log_rate

  def _batch_shape_tensor(self):
    rate_or_log_rate = self.log_rate if self.rate is None else self.rate
    return ps.broadcast_shape(ps.shape(self.concentration),
                              ps.shape(rate_or_log_rate))

  def _batch_shape(self):
    rate_or_log_rate = self.log_rate if self.rate is None else self.rate
    return tf.broadcast_static_shape(self.concentration.shape,
                                     rate_or_log_rate.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_rate_parameter(self):  # Used internally, and by kl_gamma_gamma.
    if self.log_rate is None:
      return tf.math.log(self.rate)
    return tf.convert_to_tensor(self.log_rate)

  def _sample_n(self, n, seed=None):
    seed = samplers.sanitize_seed(seed, salt='exp_gamma')

    return gamma_lib.random_gamma(
        shape=ps.convert_to_shape_tensor([n]),
        concentration=tf.convert_to_tensor(self.concentration),
        rate=None if self.rate is None else tf.convert_to_tensor(self.rate),
        log_rate=(None if self.log_rate is None else
                  tf.convert_to_tensor(self.log_rate)),
        seed=seed,
        log_space=True)

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    if self.rate is None:
      log_rate = tf.convert_to_tensor(self.log_rate)
      y = tf.math.exp(x + log_rate)
    else:
      rate = tf.convert_to_tensor(self.rate)
      y = tf.math.exp(x) * rate
      log_rate = tf.math.log(rate)

    log_unnormalized_prob = concentration * x - y
    log_normalization = tf.math.lgamma(concentration) - concentration * log_rate
    return log_unnormalized_prob - log_normalization

  def _cdf(self, x):
    # Note that igamma returns the regularized incomplete gamma function,
    # which is what we want for the CDF.
    if self.rate is None:
      y = tf.math.exp(x + self.log_rate)
    else:
      y = tf.math.exp(x) * self.rate
    return tf.math.igamma(self.concentration, y)

  def _quantile(self, p):
    y = tfp_math.igammainv(self.concentration, p)
    if self.rate is None:
      return tf.math.log(y) - self.log_rate
    return tf.math.log(y / self.rate)

  def _mean(self):
    return tf.math.digamma(self.concentration) - self._log_rate_parameter()

  def _variance(self):
    var = tf.math.polygamma(tf.ones([], self.dtype), self.concentration)
    rate_or_log_rate = self.log_rate if self.rate is None else self.rate
    return tf.broadcast_to(var, ps.broadcast_shape(ps.shape(var),
                                                   ps.shape(rate_or_log_rate)))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    if self.rate is not None and is_init != tensor_util.is_ref(self.rate):
      assertions.append(assert_util.assert_positive(
          self.rate,
          message='Argument `rate` must be positive.'))
    return assertions


kullback_leibler.RegisterKL(ExpGamma, ExpGamma)(gamma_lib.kl_gamma_gamma)


class ExpInverseGamma(transformed_distribution.TransformedDistribution):
  """ExpInverseGamma distribution.

  The `ExpInverseGamma` distribution is defined over the real numbers such that
  X ~ ExpInverseGamma(..) => exp(X) ~ InverseGamma(..).

  The distribution is logically equivalent to `tfb.Log()(tfd.InverseGamma(..))`,
  but can be sampled with much better precision.

  #### Mathematical Details

  The probability density function (pdf) is very similar to ExpGamma,

  ```none
  pdf(x; alpha, beta > 0) = exp(-x)**(alpha) exp(-exp(-x) beta) / Z
  Z = Gamma(alpha) beta**(-alpha)
  ```

  where:

  * `concentration = alpha`,
  * `scale = beta`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x) = 1 - GammaInc(alpha, beta exp(-x)) / Gamma(alpha)
  ```

  where `GammaInc` is the [upper incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in [1].

  #### Examples

  ```python
  tfd = tfp.distributions
  dist = tfd.ExpInverseGamma(concentration=3.0, scale=2.0)
  dist2 = tfd.ExpInverseGamma(concentration=[3.0, 4.0], log_scale=[0.5, -1.])
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  concentration = tf.constant(3.0)
  log_scale = tf.constant(.5)
  dist = tfd.ExpInverseGamma(concentration, scale)
  with tf.GradientTape() as tape:
    tape.watch([concentration, log_scale])
    samples = dist.sample(5)  # Shape [5]
    loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tape.gradient(loss, [concentration, log_scale])
  ```

  #### References

  [1]: Michael Figurnov, Shakir Mohamed, Andriy Mnih.
       Implicit Reparameterization Gradients. _arXiv preprint arXiv:1805.08498_,
       2018. https://arxiv.org/abs/1805.08498

  """

  def __init__(self, concentration, scale=None, log_scale=None,
               validate_args=False, allow_nan_stats=True,
               name='ExpInverseGamma'):
    """Construct ExpInverseGamma with `concentration` and `scale` parameters.

    The parameters `concentration` and `scale` (or `log_scale`) must be shaped
    in a way that supports broadcasting (e.g. `concentration + scale` is a valid
    operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      scale: Floating point tensor, the scale params of the distribution(s).
        Must contain only positive values. Mutually exclusive with `log_scale`.
      log_scale: Floating point tensor, the natural logarithm of the scale
        params of the distribution(s). Mutually exclusive with `scale`.
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
      TypeError: if `concentration`, `scale`, or `log_scale` are different
        dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [concentration, scale, log_scale], dtype_hint=tf.float32)
      concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      log_scale = tensor_util.convert_nonref_to_tensor(
          log_scale, dtype=dtype, name='log_scale')
      bijector = scale_bijector.Scale(scale=-tf.ones([], dtype=dtype))
      to_transform = ExpGamma(
          concentration=concentration, rate=scale, log_rate=log_scale,
          validate_args=validate_args, allow_nan_stats=allow_nan_stats)
      super(ExpInverseGamma, self).__init__(
          bijector=bijector,
          distribution=to_transform,
          validate_args=validate_args,
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
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False),
        log_scale=parameter_properties.ParameterProperties())
    # pylint: enable=g-long-lambda

  @property
  def concentration(self):
    """Concentration parameter."""
    return self.distribution.concentration

  @property
  def scale(self):
    """Scale parameter."""
    return self.distribution.rate

  @property
  def log_scale(self):
    """Log of scale parameter."""
    return self.distribution.log_rate

  def _log_rate_parameter(self):  # Required by gamma_lib.kl_gamma_gamma.
    return self.distribution._log_rate_parameter()  # pylint: disable=protected-access

  def _default_event_space_bijector(self):
    return identity_bijector.Identity()

  def _variance(self):
    return self.distribution.variance()  # invariant under -1x scaling.


kullback_leibler.RegisterKL(ExpInverseGamma, ExpInverseGamma)(
    gamma_lib.kl_gamma_gamma)
