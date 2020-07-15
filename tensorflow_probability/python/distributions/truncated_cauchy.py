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
"""The Truncated Cauchy distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.generic import log_sub_exp as _log_sub_exp

__all__ = [
    'TruncatedCauchy',
]


def _cauchy_cdf(z):
  return tf.atan(z) / np.pi + 0.5


def _log_cauchy_cdf(z):
  return tf.math.log1p(2 / np.pi * tf.atan(z)) - np.log(2)


def _cauchy_quantile(p):
  return tf.tan(np.pi * (p - 0.5))


class TruncatedCauchy(distribution.Distribution):
  """The Truncated Cauchy distribution.

  The truncated Cauchy is a Cauchy distribution bounded between `low`
  and `high` (the pdf is 0 outside these bounds and renormalized).

  Samples from this distribution are differentiable with respect to `loc`
  and `scale`, but not with respect to the bounds `low` and `high`.

  ### Mathematical Details

  The probability density function (pdf) of this distribution is:
  ```none
    pdf(x; loc, scale, low, high) =
        { 1 / (pi * scale * (1 + z**2) * A) for low <= x <= high
        { 0                                 otherwise
    z = (x - loc) / scale
    A = CauchyCDF((high - loc) / scale) - CauchyCDF((low - loc) / scale)
  ```

  where:

  * `CauchyCDF` is the cumulative density function of the Cauchy distribution
    with 0 mean and unit variance.

  This is a scalar distribution so the event shape is always scalar and the
  dimensions of the parameters define the batch_shape.

  #### Examples
  ```python

  tfd = tfp.distributions
  # Define a batch of two scalar TruncatedCauchy distributions with modes
  # at 0. and 1.0 .
  dist = tfd.TruncatedCauchy(loc=[0., 1.], scale=1.,
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
               name='TruncatedCauchy'):
    """Construct a TruncatedCauchy.

    All parameters of the distribution will be broadcast to the same shape,
    so the resulting distribution will have a batch_shape of the broadcast
    shape of all parameters.

    Args:
      loc: Floating point tensor; the modes of the corresponding non-truncated
        Cauchy distribution(s).
      scale: Floating point tensor; the scales of the distribution(s).
        Must contain only positive values.
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

      super(TruncatedCauchy, self).__init__(
          dtype=dtype,
          # Samples do not have gradients with respect to `_low` and `_high`.
          # TODO(b/161297284): Implement these gradients.
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
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

  def _normalizer(self,
                  loc=None,
                  scale=None,
                  low=None,
                  high=None,
                  std_low=None,
                  std_high=None):
    if std_low is None or std_high is None:
      std_low, std_high = self._standardized_low_and_high(
          loc=loc, scale=scale, low=low, high=high)
    # NOTE: If `std_high - std_low` is small compared to `abs(std_high)` and
    # `abs(std_low)`, then, instead of `atan(std_high) - atan(std_low)` here,
    # it may be more numerically stable to use the arctan summation formula:
    #   ```
    #   atan(std_high) - atan(std_low)
    #     = atan((std_high - std_low) / (1 + std_high * std_low))
    #   ```
    # In the typical case, we expect `std_low` and `std_high` to be far apart,
    # relative to their magnitudes.  (This note applies to `_log_normalizer`,
    # `_mean`, and `_variance`, as well.)
    return _cauchy_cdf(std_high) - _cauchy_cdf(std_low)

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
    return _log_sub_exp(
        _log_cauchy_cdf(std_high), _log_cauchy_cdf(std_low))

  @staticmethod
  def _param_shapes(sample_shape):
    # All parameters are of the same shape
    shape = tf.convert_to_tensor(sample_shape, dtype=tf.int32)
    return {'loc': shape, 'scale': shape, 'high': shape, 'low': shape}

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0, low=0, high=0)

  @property
  def loc(self):
    return self._loc

  @property
  def scale(self):
    return self._scale

  @property
  def low(self):
    return self._low

  @property
  def high(self):
    return self._high

  def _batch_shape(self):
    return functools.reduce(
        tf.broadcast_static_shape,
        (self.loc.shape, self.scale.shape, self.low.shape, self.high.shape))

  def _batch_shape_tensor(self, loc=None, scale=None, low=None, high=None):
    return functools.reduce(
        prefer_static.broadcast_shape,
        (prefer_static.shape(self.loc if loc is None else loc),
         prefer_static.shape(self.scale if scale is None else scale),
         prefer_static.shape(self.low if low is None else low),
         prefer_static.shape(self.high if high is None else high)))

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    batch_shape = self._batch_shape_tensor(
        loc=loc, scale=scale, low=low, high=high)
    sample_and_batch_shape = tf.concat([[n], batch_shape], axis=0)

    std_samples = _cauchy_quantile(
        samplers.uniform(
            sample_and_batch_shape,
            minval=_cauchy_cdf(std_low),
            maxval=_cauchy_cdf(std_high),
            seed=seed))
    return std_samples * scale + loc

  def _log_prob(self, x):
    loc, scale, low, high = self._loc_scale_low_high()
    log_prob = (
        -tf.math.log1p(tf.square((x - loc) / scale))
        - (np.log(np.pi) + tf.math.log(scale))
        - self._log_normalizer(loc=loc, scale=scale, low=low, high=high))
    # p(x) is 0 outside the bounds.
    return tf.where((x > high) | (x < low),
                    dtype_util.as_numpy_dtype(x.dtype)(-np.inf),
                    log_prob)

  def _cdf(self, x):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    # NOTE: If `x - low` is small, it may be more numerically stable to use the
    # arctan summation formula here (and in `_log_cdf`) instead of computing
    # `atan((x - loc) / scale) - atan((low - loc) / scale)`.
    # See, e.g., https://proofwiki.org/wiki/Difference_of_Arctangents .
    return tf.clip_by_value(
        ((_cauchy_cdf((x - loc) / scale) - _cauchy_cdf(std_low))
         / self._normalizer(std_low=std_low, std_high=std_high)),
        clip_value_min=0., clip_value_max=1.)

  def _log_cdf(self, x):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    return (
        _log_sub_exp(_log_cauchy_cdf((x - loc) / scale),
                     _log_cauchy_cdf(std_low))
        - self._log_normalizer(std_low=std_low, std_high=std_high))

  def _mean(self):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)

    # Formula from David Olive, "Applied Robust Statistics" --
    # see http://parker.ad.siu.edu/Olive/ch4.pdf .
    t = (tf.math.log1p(tf.math.square(std_high))
         - tf.math.log1p(tf.math.square(std_low)))
    t = t / (2 * (tf.math.atan(std_high) - tf.math.atan(std_low)))
    return loc + scale * t

  def _mode(self):
    # mode = { loc: for low <= loc <= high
    #          low: for loc < low
    #          high: for loc > high
    #        }
    loc = tf.convert_to_tensor(self.loc)
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    shape = self._batch_shape_tensor(loc=loc, low=low, high=high)
    # We *must* broadcast with scale to get a correctly shaped output, but
    # TODO(b/141460015): we should not have to explicitly broadcast the first
    # parameter to clip_by_value to align with the second and third parameters.
    return tf.clip_by_value(tf.broadcast_to(loc, shape), low, high)

  def _variance(self):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)

    # Formula from David Olive, "Applied Robust Statistics" --
    # see http://parker.ad.siu.edu/Olive/ch4.pdf .
    atan_std_low = tf.math.atan(std_low)
    atan_std_high = tf.math.atan(std_high)
    t = ((std_high - std_low - (atan_std_high - atan_std_low))
         / (atan_std_high - atan_std_low))
    std_mean = ((tf.math.log1p(tf.math.square(std_high))
                 - tf.math.log1p(tf.math.square(std_low)))
                / (2 * (atan_std_high - atan_std_low)))
    return tf.math.square(scale) * (t - tf.math.square(std_mean))

  def _quantile(self, p):
    loc, scale, low, high = self._loc_scale_low_high()
    std_low, std_high = self._standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    x = _cauchy_quantile(
        p * self._normalizer(std_low=std_low, std_high=std_high)
        + _cauchy_cdf(std_low))
    return scale * x + loc

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
              message='TruncatedCauchy not defined when `low >= high`.'))
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
