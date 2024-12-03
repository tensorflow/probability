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
"""The Generalized Normal (Generalized Gaussian) distribution class."""

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import random as tfp_random
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import special


__all__ = [
    'GeneralizedNormal',
]


class GeneralizedNormal(distribution.AutoCompositeTensorDistribution):
  """The Generalized Normal distribution.

  The Generalized Normal (or Generalized Gaussian) generalizes the Normal
  distribution with an additional shape parameter. It is parameterized by
  location `loc`, scale `scale` and shape `power`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale, power) = 1 / (2 * scale * Gamma(1 + 1 / power)) *
                             exp(-(|x - loc| / scale) ^ power)
  ```
  where `loc` is the mean, `scale` is the scale, and,  `power` is the shape
  parameter. If the power is above two, the distribution becomes platykurtic.
  A power equal to two results in a Normal distribution. A power smaller than
  two produces a leptokurtic (heavy-tailed) distribution. Mean and scale behave
  the same way as in the equivalent Normal distribution.

  See
  https://en.wikipedia.org/w/index.php?title=Generalized_normal_distribution&oldid=954254464
  for the definitions used here, including CDF, variance and entropy. See
  https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function
  for the sampling method used here.

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.GeneralizedNormal(loc=3.0, scale=2.0, power=1.0)
  dist2 = tfd.GeneralizedNormal(loc=0, scale=[3.0, 4.0], power=[[2.0], [3.0]])
  ```
  """

  def __init__(self,
               loc,
               scale,
               power,
               validate_args=False,
               allow_nan_stats=True,
               name='GeneralizedNormal'):
    """Construct Generalized Normal distributions.

    The Generalized Normal is parametrized with mean `loc`, scale
    `scale` and shape parameter `power`. The parameters must be shaped
    in a way that supports broadcasting (e.g. `loc + scale` is a valid
    operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the scale of the
        distribution(s). Must contain only positive values.
      power: Floating point tensor; the shape parameter of the distribution(s).
        Must contain only positive values. `loc`, `scale` and `power` must have
        compatible shapes for broadcasting.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc`, `scale`, and `power` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, power],
                                      dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._power = tensor_util.convert_nonref_to_tensor(
          power, dtype=dtype, name='power')
      super(GeneralizedNormal, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        power=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  @property
  def power(self):
    """Distribution parameter for shape."""
    return self._power

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None, name=None):
    n = ps.convert_to_shape_tensor(n, name='num', dtype=tf.int32)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    power = tf.convert_to_tensor(self.power)

    batch_shape = self._batch_shape_tensor(loc=loc, scale=scale, power=power)
    result_shape = ps.concat([[n], batch_shape], axis=0)

    ipower = tf.broadcast_to(tf.math.reciprocal(power), batch_shape)
    gamma_dist = gamma.Gamma(ipower, 1.)
    rademacher_seed, gamma_seed = samplers.split_seed(seed, salt='GenNormal')
    gamma_sample = gamma_dist.sample(n, seed=gamma_seed)
    binary_sample = tfp_random.rademacher(result_shape, dtype=self.dtype,
                                          seed=rademacher_seed)
    sampled = (binary_sample * tf.math.pow(tf.abs(gamma_sample), ipower))
    return loc + scale * sampled

  def _log_prob(self, x):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    power = tf.convert_to_tensor(self.power)
    one = tf.constant(1., dtype=self.dtype)
    two = tf.constant(2., dtype=self.dtype)
    log_normalization = (tf.math.log(two) + tf.math.log(scale) +
                         tf.math.lgamma(one + tf.math.reciprocal(power)))
    log_unnormalized = -tf.pow(tf.abs(x - loc) / scale, power)
    return log_unnormalized - log_normalization

  def _cdf_zero_mean(self, x):
    scale = tf.convert_to_tensor(self.scale)
    power = tf.convert_to_tensor(self.power)
    zero = tf.constant(0., dtype=self.dtype)
    half = tf.constant(0.5, dtype=self.dtype)
    one = tf.constant(1., dtype=self.dtype)
    # Double tf.where to avoid incorrect gradient at x == 0.
    x_is_zero = tf.equal(x, zero)
    safe_x = tf.where(x_is_zero, one, x)
    half_gamma = half * tf.math.igammac(
        tf.math.reciprocal(power),
        tf.pow(tf.abs(safe_x) / scale, power))
    return tf.where(
        x_is_zero,
        half,
        tf.where(x > zero, one - half_gamma, half_gamma),
    )

  def _cdf(self, x):
    loc = tf.convert_to_tensor(self.loc)
    return self._cdf_zero_mean(x - loc)

  def _survival_function(self, x):
    loc = tf.convert_to_tensor(self.loc)
    # sf(x) = cdf(-x) for loc == 0, because distribution is symmetric.
    return self._cdf_zero_mean(loc - x)

  def _quantile(self, p):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    power = tf.convert_to_tensor(self.power)
    ipower = tf.math.reciprocal(power)
    quantile = tf.where(
        p < 0.5,
        loc - tf.math.pow(
            special.igammacinv(ipower, 2. * p), ipower) * scale,
        loc + tf.math.pow(
            special.igammainv(ipower, 2. * p - 1.), ipower) * scale)
    return quantile

  def _entropy(self):
    scale = tf.convert_to_tensor(self.scale)
    power = tf.convert_to_tensor(self.power)
    ipower = tf.math.reciprocal(power)
    one = tf.constant(1., dtype=self.dtype)
    logtwo = tf.constant(np.log(2.), dtype=self.dtype)
    entropy = ipower + (logtwo + tf.math.log(scale) +
                        tf.math.lgamma(one + ipower))
    return tf.broadcast_to(entropy,
                           self._batch_shape_tensor(scale=scale, power=power))

  def _mean(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self._batch_shape_tensor(loc=loc))

  def _variance(self):
    ipower = tf.math.reciprocal(tf.convert_to_tensor(self.power))
    two = tf.constant(2., dtype=self.dtype)
    three = tf.constant(3., dtype=self.dtype)
    log_var = (two * tf.math.log(self.scale) +
               tf.math.lgamma(three * ipower) - tf.math.lgamma(ipower))
    var = tf.math.exp(log_var)
    return tf.broadcast_to(var, self._batch_shape_tensor())

  _mode = _mean

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if is_init:
      # _batch_shape() will raise error if it can statically prove that `loc`,
      # `scale`, and `power` have incompatible shapes.
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
            'Arguments `loc`, `scale` and `power` must have compatible shapes; '
            'loc.shape={}, scale.shape={}, power.shape={}.'.format(
                self.loc.shape, self.scale.shape, self.power.shape))
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access the three arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.power):
      assertions.append(assert_util.assert_positive(
          self.power, message='Argument `power` must be positive.'))

    return assertions
