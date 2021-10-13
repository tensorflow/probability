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
"""The GammaGamma distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'GammaGamma',
]


class GammaGamma(distribution.AutoCompositeTensorDistribution):
  """Gamma-Gamma distribution.

  Gamma-Gamma is a [compound
  distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution)
  defined over positive real numbers using parameters `concentration`,
  `mixing_concentration` and `mixing_rate`.

  This distribution is also referred to as the beta of the second kind (B2), and
  can be useful for transaction value modeling, as [(Fader and Hardi, 2013)][1].

  #### Mathematical Details

  It is derived from the following Gamma-Gamma hierarchical model by integrating
  out the random variable `beta`.

  ```none
      beta ~ Gamma(alpha0, beta0)
  X | beta ~ Gamma(alpha, beta)
  ```
  where
  * `concentration = alpha`
  * `mixing_concentration = alpha0`
  * `mixing_rate = beta0`

  The probability density function (pdf) is

  ```none
                                         x**(alpha - 1)
  pdf(x; alpha, alpha0, beta0) = ---------------------------------
                                 Z * (x + beta0)**(alpha + alpha0)
  ```
  where the normalizing constant `Z = Beta(alpha, alpha0) * beta0**(-alpha0)`.

  Samples of this distribution are reparameterized as samples of the Gamma
  distribution are reparameterized using the technique described in
  [(Figurnov et al., 2018)][2].

  #### References

  [1]: Peter S. Fader, Bruce G. S. Hardi. The Gamma-Gamma Model of Monetary
       Value. _Technical Report_, 2013.
       http://www.brucehardie.com/notes/025/gamma_gamma.pdf

  [2]: Michael Figurnov, Shakir Mohamed, Andriy Mnih.
       Implicit Reparameterization Gradients. _arXiv preprint arXiv:1805.08498_,
       2018. https://arxiv.org/abs/1805.08498
  """

  def __init__(self,
               concentration,
               mixing_concentration,
               mixing_rate,
               validate_args=False,
               allow_nan_stats=True,
               name='GammaGamma'):
    """Initializes a batch of Gamma-Gamma distributions.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g.
    `concentration + mixing_concentration + mixing_rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      mixing_concentration: Floating point tensor, the concentration params of
        the mixing Gamma distribution(s). Must contain only positive values.
      mixing_rate: Floating point tensor, the rate params of the mixing Gamma
        distribution(s). Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype(
          [concentration, mixing_concentration, mixing_rate],
          dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration', dtype=dtype)
      self._mixing_concentration = tensor_util.convert_nonref_to_tensor(
          mixing_concentration, name='mixing_concentration', dtype=dtype)
      self._mixing_rate = tensor_util.convert_nonref_to_tensor(
          mixing_rate, name='mixing_rate', dtype=dtype)

      super(GammaGamma, self).__init__(
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
        mixing_concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        mixing_rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def mixing_concentration(self):
    """Concentration parameter for the mixing Gamma distribution."""
    return self._mixing_concentration

  @property
  def mixing_rate(self):
    """Rate parameter for the mixing Gamma distribution."""
    return self._mixing_rate

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    concentration = tf.convert_to_tensor(self.concentration)
    mixing_concentration = tf.convert_to_tensor(self.mixing_concentration)
    mixing_rate = tf.convert_to_tensor(self.mixing_rate)
    seed_rate, seed_samples = samplers.split_seed(seed, salt='gamma_gamma')
    log_rate = gamma_lib.random_gamma(
        shape=[n],
        # Be sure to draw enough rates for the fully-broadcasted gamma-gamma.
        concentration=mixing_concentration + tf.zeros_like(concentration),
        rate=mixing_rate,
        seed=seed_rate,
        log_space=True)
    return gamma_lib.random_gamma(
        shape=[],
        concentration=concentration,
        log_rate=log_rate,
        seed=seed_samples)

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    mixing_concentration = tf.convert_to_tensor(self.mixing_concentration)
    mixing_rate = tf.convert_to_tensor(self.mixing_rate)

    log_normalization = (
        tfp_math.lbeta(concentration, mixing_concentration) -
        mixing_concentration * tf.math.log(mixing_rate))

    log_unnormalized_prob = (tf.math.xlogy(concentration - 1., x) -
                             (concentration + mixing_concentration) *
                             tf.math.log(x + mixing_rate))
    # The formula computes `nan` for `x == +inf`.  However, it shouldn't be too
    # inaccurate for large finite `x`, because `x` only appears as `log(x)`, and
    # `log` is effectively discountinuous at `+inf`.
    log_unnormalized_prob = tf.where(
        x >= np.inf,
        tf.constant(-np.inf, dtype=log_unnormalized_prob.dtype),
        log_unnormalized_prob)

    return log_unnormalized_prob - log_normalization

  @distribution_util.AppendDocstring(
      """The mean of a Gamma-Gamma distribution is
      `concentration * mixing_rate / (mixing_concentration - 1)`, when
      `mixing_concentration > 1`, and `NaN` otherwise. If `self.allow_nan_stats`
      is `False`, an exception will be raised rather than returning `NaN`""")
  def _mean(self):
    concentration = tf.convert_to_tensor(self.concentration)
    mixing_concentration = tf.convert_to_tensor(self.mixing_concentration)
    mixing_rate = tf.convert_to_tensor(self.mixing_rate)

    mean = concentration * mixing_rate / (mixing_concentration - 1.)
    if self.allow_nan_stats:
      return tf.where(
          mixing_concentration > 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      with tf.control_dependencies([
          assert_util.assert_less(
              tf.ones([], self.dtype),
              mixing_concentration,
              message='mean undefined when `mixing_concentration` <= 1'),
      ]):
        return tf.identity(mean)

  @distribution_util.AppendDocstring(
      """The variance of a Gamma-Gamma distribution is
      `concentration**2 * mixing_rate**2 / ((mixing_concentration - 1)**2 *
      (mixing_concentration - 2))`, when `mixing_concentration > 2`, and `NaN`
      otherwise. If `self.allow_nan_stats` is `False`, an exception will be
      raised rather than returning `NaN`""")
  def _variance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    mixing_concentration = tf.convert_to_tensor(self.mixing_concentration)
    mixing_rate = tf.convert_to_tensor(self.mixing_rate)

    variance = (tf.square(concentration * mixing_rate /
                          (mixing_concentration - 1.)) /
                (mixing_concentration - 2.))
    if self.allow_nan_stats:
      return tf.where(
          mixing_concentration > 2.,
          variance,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      with tf.control_dependencies([
          assert_util.assert_less(
              tf.ones([], self.dtype) * 2.,
              mixing_concentration,
              message='variance undefined when `mixing_concentration` <= 2')]):
        return tf.identity(variance)

  def _default_event_space_bijector(self):
    return exp_bijector.Exp(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    dtype_util.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
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
    for param_name, param in dict(
        concentration=self.concentration,
        mixing_concentration=self.mixing_concentration,
        mixing_rate=self.mixing_rate).items():

      if is_init != tensor_util.is_ref(param):
        assertions.append(assert_util.assert_positive(
            param,
            message='Argument `{}` must be positive.'.format(param_name)))
    return assertions
