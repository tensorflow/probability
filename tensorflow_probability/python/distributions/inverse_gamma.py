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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    "InverseGamma",
    "InverseGammaWithSoftplusConcentrationRate",
]


class InverseGamma(distribution.Distribution):
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

  @deprecation.deprecated_args(
      "2019-05-08", "The `rate` parameter is deprecated. Use `scale` instead."
      "The `rate` parameter was always interpreted as a `scale` parameter, "
      "but erroneously misnamed.", "rate")
  def __init__(self,
               concentration,
               scale=None,
               validate_args=False,
               allow_nan_stats=True,
               rate=None,
               name="InverseGamma"):
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
      rate: Deprecated (mis-named) alias for `scale`.
      name: Python `str` name prefixed to Ops created by this class.


    Raises:
      TypeError: if `concentration` and `scale` are different dtypes.
    """
    if rate is not None:
      scale = rate
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration, scale],
                                      dtype_hint=tf.float32)
      concentration = tf.convert_to_tensor(
          concentration, name="concentration", dtype=dtype)
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      with tf.control_dependencies([
          assert_util.assert_positive(
              concentration, message="Concentration must be positive."),
          assert_util.assert_positive(
              scale, message="Scale must be positive."),
      ] if validate_args else []):
        self._concentration = tf.identity(concentration, name="concentration")
        self._scale = tf.identity(scale, name="scale")
      dtype_util.assert_same_float_dtype([self._concentration, self._scale])

    super(InverseGamma, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration, self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("concentration", "scale"),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=0, rate=0, scale=0)

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  @deprecation.deprecated(
      "2019-05-08", "The `rate` parameter is deprecated. Use `scale` instead."
      "The `rate` parameter was always interpreted as a `scale`parameter, but "
      "erroneously misnamed.")
  def rate(self):
    """Scale parameter."""
    return self._scale

  @property
  def scale(self):
    """Scale parameter."""
    return self._scale

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.concentration), tf.shape(self.scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.concentration.shape,
                                     self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random_gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    return 1. / tf.random.gamma(
        shape=[n],
        alpha=self.concentration,
        beta=self.scale,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _cdf(self, x):
    x = self._maybe_assert_valid_sample(x)
    # Note that igammac returns the upper regularized incomplete gamma
    # function Q(a, x), which is what we want for the CDF.
    return tf.math.igammac(self.concentration, self.scale / x)

  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return -(1. + self.concentration) * tf.math.log(x) - self.scale / x

  def _log_normalization(self):
    return (tf.math.lgamma(self.concentration) -
            self.concentration * tf.math.log(self.scale))

  def _entropy(self):
    return (self.concentration + tf.math.log(self.scale) +
            tf.math.lgamma(self.concentration) -
            ((1. + self.concentration) * tf.math.digamma(self.concentration)))

  @distribution_util.AppendDocstring(
      """The mean of an inverse gamma distribution is
      `scale / (concentration - 1)`, when `concentration > 1`, and `NaN`
      otherwise. If `self.allow_nan_stats` is `False`, an exception will be
      raised rather than returning `NaN`""")
  def _mean(self):
    mean = self.scale / (self.concentration - 1.)
    if self.allow_nan_stats:
      return tf.where(
          self.concentration > 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], self.dtype),
              self.concentration,
              message="mean undefined when any concentration <= 1"),
      ], mean)

  @distribution_util.AppendDocstring(
      """Variance for inverse gamma is defined only for `concentration > 2`. If
      `self.allow_nan_stats` is `False`, an exception will be raised rather
      than returning `NaN`.""")
  def _variance(self):
    var = (
        tf.square(self.scale) / tf.square(self.concentration - 1.) /
        (self.concentration - 2.))
    if self.allow_nan_stats:
      return tf.where(
          self.concentration > 2.,
          var,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.constant(2., dtype=self.dtype),
              self.concentration,
              message="variance undefined when any concentration <= 2"),
      ], var)

  @distribution_util.AppendDocstring(
      """The mode of an inverse gamma distribution is `scale / (concentration +
      1)`.""")
  def _mode(self):
    return self.scale / (1. + self.concentration)

  def _maybe_assert_valid_sample(self, x):
    dtype_util.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
    if not self.validate_args:
      return x
    return distribution_util.with_dependencies([
        assert_util.assert_positive(x),
    ], x)


class _InverseGammaWithSoftplusConcentrationScale(InverseGamma):
  """`InverseGamma` with softplus of `concentration` and `scale`."""

  @deprecation.deprecated_args(
      "2019-05-08", "The `rate` parameter is deprecated. Use `scale` instead."
      "The `rate` parameter was always interpreted as a `scale`parameter, but "
      "erroneously misnamed.", "rate")
  def __init__(self,
               concentration,
               scale=None,
               validate_args=False,
               allow_nan_stats=True,
               rate=None,
               name="InverseGammaWithSoftplusConcentrationScale"):
    if rate is not None:
      scale = rate
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration, scale])
      concentration = tf.convert_to_tensor(
          concentration, name="softplus_concentration", dtype=dtype)
      scale = tf.convert_to_tensor(scale, name="softplus_scale", dtype=dtype)
      super(_InverseGammaWithSoftplusConcentrationScale, self).__init__(
          concentration=tf.math.softplus(
              concentration, name="softplus_concentration"),
          scale=tf.math.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters


_rate_deprecator = deprecation.deprecated(
    "2019-06-05",
    "InverseGammaWithSoftplusConcentrationRate is deprecated, use "
    "InverseGamma(concentration=tf.math.softplus(concentration), "
    "scale=tf.math.softplus(scale)) instead.",
    warn_once=True)
# pylint: disable=invalid-name
InverseGammaWithSoftplusConcentrationRate = _rate_deprecator(
    _InverseGammaWithSoftplusConcentrationScale)

_scale_deprecator = deprecation.deprecated(
    "2019-06-05",
    "InverseGammaWithSoftplusConcentrationScale is deprecated, use "
    "InverseGamma(concentration=tf.math.softplus(concentration), "
    "scale=tf.math.softplus(scale)) instead.",
    warn_once=True)
InverseGammaWithSoftplusConcentrationScale = _scale_deprecator(
    _InverseGammaWithSoftplusConcentrationScale)
