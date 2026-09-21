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
"""The Skew Generalized Normal (Generalized Normal v2) distribution class."""

import functools

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import scale as scale_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'SkewGeneralizedNormal',
]


class SkewGeneralizedNormal(distribution.AutoCompositeTensorDistribution):
  """The Skew Generalized Normal distribution.

  This is the "generalized normal version 2" (the skew variant) described in
  https://en.wikipedia.org/wiki/Generalized_normal_distribution. It is
  parameterized by location `loc` (xi), scale `scale` (alpha > 0) and shape
  `peak` (kappa, any nonzero real). It is constructed as a monotone transform of
  a standard normal: if `Z ~ Normal(0, 1)` then

  ```none
  X = loc + (scale / peak) * (1 - exp(-peak * Z))
  ```

  is `SkewGeneralizedNormal(loc, scale, peak)` distributed.

  #### Mathematical details

  Define the standardizing transform

  ```none
  y(x) = (-1 / peak) * log(1 - peak * (x - loc) / scale).
  ```

  Then the probability density function (pdf), cumulative distribution function
  (cdf) and quantile are

  ```none
  pdf(x; loc, scale, peak) = phi(y(x)) / (scale - peak * (x - loc))
  cdf(x; loc, scale, peak) = Phi(y(x))
  quantile(p; loc, scale, peak) = (
      loc + scale * (1 - exp(-peak * Phi^{-1}(p))) / peak)
  ```

  where `phi` and `Phi` are the standard normal pdf and cdf.

  #### Support and the mode

  Unlike most location-scale families, the support is a parameter-dependent
  half-line: `peak > 0` gives `x < loc + scale / peak` (bounded above,
  left-skewed) and `peak < 0` gives `x > loc + scale / peak` (bounded below,
  right-skewed). The distinguishing feature of this distribution is that the
  mode sits at or near the support boundary `x_edge = loc + scale / peak`:

  ```none
  x_edge - mode = (scale / peak) * exp(-peak**2),
  ```

  so the gap between the mode and the boundary vanishes like `exp(-peak**2)` as
  `|peak|` grows. Equivalently, `peak` is exactly the value of the standardized
  coordinate `y` at the mode (`y(mode) = peak`), which is why it is named
  `peak`. As `peak -> 0` the distribution converges to `Normal(loc, scale)`.

  Outside the support, `log_prob`/`cdf` return `NaN`/`-inf` while `prob` returns
  `0`.

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.SkewGeneralizedNormal(loc=3.0, scale=2.0, peak=1.0)
  dist2 = tfd.SkewGeneralizedNormal(
      loc=0, scale=[3.0, 4.0], peak=[[2.0], [-3.0]])
  ```
  """

  def __init__(self,
               loc,
               scale,
               peak,
               validate_args=False,
               allow_nan_stats=True,
               name='SkewGeneralizedNormal'):
    """Construct Skew Generalized Normal distributions.

    The parameters `loc`, `scale` and `peak` must be shaped in a way that
    supports broadcasting (e.g. `loc + scale + peak` is a valid operation).

    Args:
      loc: Floating point tensor; the location(s) of the distribution(s).
      scale: Floating point tensor; the scale(s) of the distribution(s). Must
        contain only positive values.
      peak: Floating point tensor; the shape parameter(s) of the
        distribution(s). Must contain only nonzero values. `loc`, `scale` and
        `peak` must have compatible shapes for broadcasting.
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
      TypeError: if `loc`, `scale`, and `peak` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, peak],
                                      dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._peak = tensor_util.convert_nonref_to_tensor(
          peak, dtype=dtype, name='peak')
      super(SkewGeneralizedNormal, self).__init__(
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
        peak=parameter_properties.ParameterProperties())
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  @property
  def peak(self):
    """Distribution parameter related to mode and skew."""
    return self._peak

  def _batch_shape_tensor(self, loc=None, scale=None, peak=None):
    return functools.reduce(ps.broadcast_shape, (
        ps.shape(self.loc if loc is None else loc),
        ps.shape(self.scale if scale is None else scale),
        ps.shape(self.peak if peak is None else peak)))

  def _batch_shape(self):
    return functools.reduce(tf.broadcast_static_shape, (
        self.loc.shape, self.scale.shape, self.peak.shape))

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _y(self, x, loc=None, scale=None, peak=None):
    """Standardizes `x` to a unit normal variate `y`."""
    loc = tf.convert_to_tensor(self.loc) if loc is None else loc
    scale = tf.convert_to_tensor(self.scale) if scale is None else scale
    peak = tf.convert_to_tensor(self.peak) if peak is None else peak
    # y = (-1 / peak) * log(1 - peak * (x - loc) / scale).
    return -tf.math.log1p(-peak * (x - loc) / scale) / peak

  def _log_prob(self, x):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    y = self._y(x, loc=loc, scale=scale, peak=peak)
    log_normalization = tf.constant(
        0.5 * np.log(2. * np.pi), dtype=self.dtype)
    # log f(x) = log phi(y) - log(scale - peak * (x - loc)), where the second
    # term is -log|dy/dx| (the change-of-variables Jacobian).
    return (-0.5 * tf.square(y) - log_normalization
            - tf.math.log(scale - peak * (x - loc)))

  def _prob(self, x):
    prob = tf.exp(self._log_prob(x))
    # Outside the support `log_prob` is `NaN`; the density there is zero.
    return tf.where(tf.math.is_nan(prob), tf.zeros_like(prob), prob)

  def _log_cdf(self, x):
    return special_math.log_ndtr(self._y(x))

  def _cdf(self, x):
    return special_math.ndtr(self._y(x))

  def _log_survival_function(self, x):
    return special_math.log_ndtr(-self._y(x))

  def _survival_function(self, x):
    return special_math.ndtr(-self._y(x))

  def _quantile(self, p):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    return loc + scale * (1. - tf.exp(-peak * tf.math.ndtri(p))) / peak

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    shape = ps.concat(
        [[n], self._batch_shape_tensor(loc=loc, scale=scale, peak=peak)],
        axis=0)
    probs = samplers.uniform(
        shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
    return loc + scale * (1. - tf.exp(-peak * tf.math.ndtri(probs))) / peak

  def _mean(self):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    mean = loc - scale * tf.math.expm1(0.5 * tf.square(peak)) / peak
    return tf.broadcast_to(
        mean, self._batch_shape_tensor(loc=loc, scale=scale, peak=peak))

  def _stddev(self):
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    # stddev = (scale / |peak|) * exp(peak**2 / 2) * sqrt(exp(peak**2) - 1).
    stddev = (scale / tf.abs(peak)) * tf.exp(0.5 * tf.square(peak)) * tf.sqrt(
        tf.math.expm1(tf.square(peak)))
    return tf.broadcast_to(
        stddev, self._batch_shape_tensor(scale=scale, peak=peak))

  def _variance(self):
    return tf.square(self._stddev())

  def _mode(self):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    mode = loc - scale * tf.math.expm1(-tf.square(peak)) / peak
    return tf.broadcast_to(
        mode, self._batch_shape_tensor(loc=loc, scale=scale, peak=peak))

  def _entropy(self):
    scale = tf.convert_to_tensor(self.scale)
    # H = H(Normal) + E[log|dx/dy|] = 0.5 * (1 + log(2*pi)) + log(scale).
    entropy = tf.math.log(scale) + tf.constant(
        0.5 * (1. + np.log(2. * np.pi)), dtype=self.dtype)
    return tf.broadcast_to(entropy, self._batch_shape_tensor(scale=scale))

  def _default_event_space_bijector(self):
    # The inverse of the `y` transform,
    # `x = loc + (scale / peak) * (1 - exp(-peak * y))`, is a monotone bijection
    # from R onto the support for either sign of `peak` (the sign is carried by
    # the `Scale` factors). Realized as the composition
    # `Shift(edge) o Scale(-scale/peak) o Exp o Scale(-peak)`.
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    edge = loc + scale / peak
    return chain_bijector.Chain([
        shift_bijector.Shift(shift=edge, validate_args=self.validate_args),
        scale_bijector.Scale(
            scale=-scale / peak, validate_args=self.validate_args),
        exp_bijector.Exp(validate_args=self.validate_args),
        scale_bijector.Scale(scale=-peak, validate_args=self.validate_args),
    ], validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if is_init:
      # _batch_shape() will raise error if it can statically prove that `loc`,
      # `scale` and `peak` have incompatible shapes.
      try:
        self._batch_shape()
      except ValueError as e:
        raise ValueError(
            'Arguments `loc`, `scale` and `peak` must have compatible shapes; '
            'loc.shape={}, scale.shape={}, peak.shape={}.'.format(
                self.loc.shape, self.scale.shape, self.peak.shape)) from e

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.peak):
      assertions.append(assert_util.assert_none_equal(
          self.peak,
          tf.zeros([], dtype=self.dtype),
          message='Argument `peak` must be nonzero.'))

    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    peak = tf.convert_to_tensor(self.peak)
    assertions.append(assert_util.assert_less(
        peak * (x - loc), scale,
        message='Sample must be in the support: `peak * (x - loc) < scale`.'))
    return assertions
