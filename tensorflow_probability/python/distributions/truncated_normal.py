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
"""The Truncated Normal distribution class."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
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
from tensorflow_probability.python.math.generic import log_sub_exp as _log_sub_exp


__all__ = [
    'TruncatedNormal',
]


def _normal_pdf(x):
  two_pi = tf.convert_to_tensor(2 * np.pi, dtype=x.dtype)
  return tf.math.rsqrt(two_pi) * tf.exp(-0.5 * tf.square(x))


def _normal_log_pdf(x):
  two_pi = tf.convert_to_tensor(2 * np.pi, dtype=x.dtype)
  return -0.5 * (tf.math.log(two_pi) + tf.square(x))


def _normal_cdf_difference(x, y):
  """Computes ndtr(x) - ndtr(y) assuming that x >= y."""
  # When x >= y >= 0, we will return ndtr(-y) - ndtr(-x)
  # because ndtr does not have a good precision for large positive x, y.
  is_y_positive = y >= 0
  x_hat = tf.where(is_y_positive, -y, x)
  y_hat = tf.where(is_y_positive, -x, y)
  return special_math.ndtr(x_hat) - special_math.ndtr(y_hat)


def _normal_cdf_log_difference(x, y):
  """Computes log(ndtr(x) - ndtr(y)) assuming that x >= y."""
  # When x >= y >= 0, we will return log(ndtr(-y) - ndtr(-x))
  # because ndtr does not have a good precision for large positive x, y.
  is_y_positive = y >= 0
  x_hat = tf.where(is_y_positive, -y, x)
  y_hat = tf.where(is_y_positive, -x, y)
  return _log_sub_exp(
      special_math.log_ndtr(x_hat), special_math.log_ndtr(y_hat))


class TruncatedNormal(distribution.AutoCompositeTensorDistribution):
  """The Truncated Normal distribution.

  The truncated normal is a normal distribution bounded between `low`
  and `high` (the pdf is 0 outside these bounds and renormalized).

  Samples from this distribution are differentiable with respect to `loc`,
  `scale` as well as the bounds, `low` and `high`, i.e., this
  implementation is fully reparameterized.

  For more details, see [here](
  https://en.wikipedia.org/wiki/Truncated_normal_distribution).

  ### Mathematical Details

  The probability density function (pdf) of this distribution is:
  ```none
    pdf(x; loc, scale, low, high) =
        { (2 pi)**(-0.5) exp(-0.5 y**2) / (scale * z) for low <= x <= high
        { 0                                    otherwise
    y = (x - loc)/scale
    z = NormalCDF((high - loc) / scale) - NormalCDF((low - loc) / scale)
  ```

  where:

  * `NormalCDF` is the cumulative density function of the Normal distribution
    with 0 mean and unit variance.

  This is a scalar distribution so the event shape is always scalar and the
  dimensions of the parameters define the batch_shape.

  #### Examples
  ```python

  tfd = tfp.distributions
  # Define a batch of two scalar TruncatedNormals with modes at 0. and 1.0
  dist = tfd.TruncatedNormal(loc=[0., 1.], scale=1.,
                             low=[-1., 0.],
                             high=[1., 1.])

  # Evaluate the pdf of the distributions at 0.5 and 0.8 respectively returning
  # a 2-vector tensor.
  dist.prob([0.5, 0.8])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```
  """

  def __init__(self,
               loc,
               scale,
               low,
               high,
               validate_args=False,
               allow_nan_stats=True,
               name='TruncatedNormal'):
    """Construct TruncatedNormal.

    All parameters of the distribution will be broadcast to the same shape,
    so the resulting distribution will have a batch_shape of the broadcast
    shape of all parameters.

    Args:
      loc: Floating point tensor; the mean of the normal distribution(s) (
        note that the mean of the resulting distribution will be different
        since it is modified by the bounds).
      scale: Floating point tensor; the std deviation of the normal
        distribution(s).
      low: `float` `Tensor` representing lower bound of the distribution's
        support. Must be such that `low < high`.
      high: `float` `Tensor` representing upper bound of the distribution's
        support. Must be such that `low < high`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked at run-time.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, low, high], tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      self._low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
      self._high = tensor_util.convert_nonref_to_tensor(
          high, name='high', dtype=dtype)
      dtype_util.assert_same_float_dtype(
          [self._loc, self._scale, self._low, self._high])

      super(TruncatedNormal, self).__init__(
          dtype=dtype,
          # This distribution is fully reparameterized. loc, scale have straight
          # through gradients. The gradients for the bounds are implemented
          # using custom derived expressions based on implicit gradients.
          # For the special case of lower bound zero and a positive upper bound
          # an equivalent expression can also be found in Sec 9.1.1.
          # of https://arxiv.org/pdf/1806.01851.pdf. The implementation here
          # handles arbitrary bounds.
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  def _loc_scale_low_high(self, loc=None, scale=None, low=None, high=None):
    loc = tf.convert_to_tensor(self.loc if loc is None else loc)
    scale = tf.convert_to_tensor(self.scale if scale is None else scale)
    low = tf.convert_to_tensor(self.low if low is None else low)
    high = tf.convert_to_tensor(self.high if high is None else high)
    return loc, scale, low, high

  def _standardized_low_and_high(self,
                                 loc=None,
                                 scale=None,
                                 low=None,
                                 high=None):
    loc, scale, low, high = self._loc_scale_low_high(
        loc=loc, scale=scale, low=low, high=high)
    return (low - loc) / scale, (high - loc) / scale

  def _log_normalizer(self,
                      loc=None,
                      scale=None,
                      low=None,
                      high=None,
                      std_low=None,
                      std_high=None):
    if std_low is None or std_high is None:
      std_low, std_high = self._standardized_low_and_high(
          loc=loc, scale=scale, low=low, high=high)
    return _normal_cdf_log_difference(std_high, std_low)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        low=parameter_properties.ParameterProperties(),
        # TODO(b/169874884): Support decoupled parameterization.
        high=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED,))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  @property
  def low(self):
    return self._low

  @property
  def high(self):
    return self._high

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc, scale, low, high = self._loc_scale_low_high()
    batch_shape = self._batch_shape_tensor(
        loc=loc, scale=scale, low=low, high=high)
    sample_and_batch_shape = ps.concat([[n], batch_shape], 0)
    return tf.random.stateless_parameterized_truncated_normal(
        shape=sample_and_batch_shape,
        means=loc,
        stddevs=scale,
        minvals=low,
        maxvals=high,
        seed=samplers.sanitize_seed(seed))

  def _log_prob(self, x):
    np_dtype = dtype_util.as_numpy_dtype(x.dtype)
    loc, scale, low, high = self._loc_scale_low_high()
    log_prob = -(np_dtype(0.5) * tf.square(
        (x - loc) / scale) + (0.5 * np.log(2. * np.pi)).astype(np_dtype) +
                 tf.math.log(scale) +
                 self._log_normalizer(loc=loc, scale=scale, low=low, high=high))
    # p(x) is 0 outside the bounds.
    bounded_log_prob = tf.where((x > high) | (x < low),
                                np_dtype(-np.inf),
                                log_prob)
    return bounded_log_prob

  def _cdf(self, x):
    return tf.exp(self._log_cdf(x))

  def _log_cdf(self, x):
    np_dtype = dtype_util.as_numpy_dtype(x.dtype)
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    std_x = (x - loc) / scale
    log_cdf = (
        _normal_cdf_log_difference(std_x, std_low) -
        self._log_normalizer(std_low=std_low, std_high=std_high))
    # cdf(x) is 0 when x < low and is 1 when x > high.
    bounded_log_cdf = tf.where(x > high, np_dtype(0.), log_cdf)
    bounded_log_cdf = tf.where(x < low, np_dtype(-np.inf), bounded_log_cdf)
    return bounded_log_cdf

  def _entropy(self):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        loc=loc, scale=scale, low=low, high=high)
    log_normalizer = self._log_normalizer(std_low=std_low, std_high=std_high)
    return (
        0.5 * (1 + np.log(2.) + np.log(np.pi)) + tf.math.log(scale) +
        log_normalizer + 0.5 *
        (std_low * _normal_pdf(std_low) - std_high * _normal_pdf(std_high)) /
        tf.exp(log_normalizer))

  def _mean(self):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        loc=loc, scale=scale, low=low, high=high)
    lse, sign = _log_sub_exp(_normal_log_pdf(std_low),
                             _normal_log_pdf(std_high),
                             return_sign=True)
    return loc + scale * sign * tf.math.exp(
        lse - self._log_normalizer(std_low=std_low, std_high=std_high))

  def _mode(self):
    # mode = { loc: for low <= loc <= high
    #          low: for loc < low
    #          high: for loc > high
    #        }
    loc = tf.convert_to_tensor(self.loc)
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    shp = self._batch_shape_tensor(loc=loc, low=low, high=high)
    # We *must* broadcast with scale to get a correctly shaped output, but
    # TODO(b/141460015): we should not have to explicitly broadcast the first
    # parameter to clip_by_value to align with the second and third parameters.
    bc_loc = tf.broadcast_to(loc, shp)
    return tf.clip_by_value(bc_loc, low, high)

  def _quantile(self, p):
    # TODO(b/188413116): This implementation is analytically correct, but might
    # not perform well in all cases. See
    # https://en.wikipedia.org/wiki/Truncated_normal_distribution#Generating_values_from_the_truncated_normal_distribution)
    # for a discussion on alternatives.
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    quantile = tf.math.ndtri(
        special_math.ndtr(std_low) +
        p * _normal_cdf_difference(std_high, std_low)) * scale + loc
    return quantile

  def _variance(self):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        loc=loc, scale=scale, low=low, high=high)
    log_normalizer = self._log_normalizer(std_low=std_low, std_high=std_high)
    var = (
        tf.square(scale) *
        (1. +
         (std_low * _normal_pdf(std_low) - std_high * _normal_pdf(std_high)) /
         tf.exp(log_normalizer) -
         tf.exp(2. * (
             _log_sub_exp(  # ignore sign because result gets squared
                 _normal_log_pdf(std_low), _normal_log_pdf(std_high))
             - log_normalizer))))
    return var

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(
        low=self.low, high=self.high, validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    low = None
    high = None
    if is_init != tensor_util.is_ref(self.low):
      low = tf.convert_to_tensor(self.low)
      assertions.append(
          assert_util.assert_finite(low, message='`low` is not finite'))
    if is_init != tensor_util.is_ref(self.high):
      high = tf.convert_to_tensor(self.high)
      assertions.append(
          assert_util.assert_finite(high, message='`high` is not finite'))
    if is_init != tensor_util.is_ref(self.loc):
      assertions.append(
          assert_util.assert_finite(self.loc, message='`loc` is not finite'))
    if is_init != tensor_util.is_ref(self.scale):
      scale = tf.convert_to_tensor(self.scale)
      assertions.extend([
          assert_util.assert_positive(
              scale, message='`scale` must be positive'),
          assert_util.assert_finite(scale, message='`scale` is not finite'),
      ])
    if (is_init != tensor_util.is_ref(self.low) or
        is_init != tensor_util.is_ref(self.high)):
      low = tf.convert_to_tensor(self.low) if low is None else low
      high = tf.convert_to_tensor(self.high) if high is None else high
      assertions.append(
          assert_util.assert_greater(
              high,
              low,
              message='TruncatedNormal not defined when `low >= high`.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_greater_equal(
        x, self.low, message='Sample must be greater than or equal to `low`.'))
    assertions.append(assert_util.assert_less_equal(
        x, self.high, message='Sample must be less than or equal to `high`.'))
    return assertions
