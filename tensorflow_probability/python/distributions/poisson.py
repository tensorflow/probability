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
"""The Poisson distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization

__all__ = [
    "Poisson",
]


class Poisson(distribution.Distribution):
  """Poisson distribution.

  The Poisson distribution is parameterized by an event `rate` parameter.

  #### Mathematical Details

  The probability mass function (pmf) is,

  ```none
  pmf(k; lambda, k >= 0) = (lambda^k / k!) / Z
  Z = exp(lambda).
  ```

  where `rate = lambda` and `Z` is the normalizing constant.

  """

  def __init__(self,
               rate=None,
               log_rate=None,
               interpolate_nondiscrete=True,
               validate_args=False,
               allow_nan_stats=True,
               name="Poisson"):
    """Initialize a batch of Poisson distributions.

    Args:
      rate: Floating point tensor, the rate parameter. `rate` must be positive.
        Must specify exactly one of `rate` and `log_rate`.
      log_rate: Floating point tensor, the log of the rate parameter.
        Must specify exactly one of `rate` and `log_rate`.
      interpolate_nondiscrete: Python `bool`. When `False`,
        `log_prob` returns `-inf` (and `prob` returns `0`) for non-integer
        inputs. When `True`, `log_prob` evaluates the continuous function
        `k * log_rate - lgamma(k+1) - rate`, which matches the Poisson pmf
        at integer arguments `k` (note that this function is not itself
        a normalized probability log-density).
        Default value: `True`.
      validate_args: Python `bool`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: if none or both of `rate`, `log_rate` are specified.
      TypeError: if `rate` is not a float-type.
      TypeError: if `log_rate` is not a float-type.
    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(name, values=[rate]) as name:
      if (rate is None) == (log_rate is None):
        raise ValueError("Must specify exactly one of `rate` and `log_rate`.")
      elif log_rate is None:
        rate = tf.convert_to_tensor(
            value=rate,
            name="rate",
            dtype=dtype_util.common_dtype([rate], preferred_dtype=tf.float32))
        if not rate.dtype.is_floating:
          raise TypeError("rate.dtype ({}) is a not a float-type.".format(
              rate.dtype.name))
        with tf.control_dependencies(
            [tf.compat.v1.assert_positive(rate)] if validate_args else []):
          self._rate = tf.identity(rate, name="rate")
          self._log_rate = tf.math.log(rate, name="log_rate")
      else:
        log_rate = tf.convert_to_tensor(
            value=log_rate,
            name="log_rate",
            dtype=dtype_util.common_dtype([log_rate], tf.float32))
        if not log_rate.dtype.is_floating:
          raise TypeError("log_rate.dtype ({}) is a not a float-type.".format(
              log_rate.dtype.name))
        self._rate = tf.exp(log_rate, name="rate")
        self._log_rate = tf.convert_to_tensor(value=log_rate, name="log_rate")

    self._interpolate_nondiscrete = interpolate_nondiscrete
    super(Poisson, self).__init__(
        dtype=self._rate.dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._rate],
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(rate=0, log_rate=0)

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  @property
  def log_rate(self):
    """Log rate parameter."""
    return self._log_rate

  @property
  def interpolate_nondiscrete(self):
    """Interpolate (log) probs on non-integer inputs."""
    return self._interpolate_nondiscrete

  def _batch_shape_tensor(self):
    return tf.shape(input=self.rate)

  def _batch_shape(self):
    return self.rate.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    log_probs = self._log_unnormalized_prob(x) - self._log_normalization()
    if not self.interpolate_nondiscrete:
      # Ensure the gradient wrt `rate` is zero at non-integer points.
      neg_inf = tf.fill(
          tf.shape(input=log_probs),
          value=np.array(-np.inf, dtype=log_probs.dtype.as_numpy_dtype))
      log_probs = tf.where(tf.math.is_inf(log_probs), neg_inf, log_probs)
    return log_probs

  def _log_cdf(self, x):
    return tf.math.log(self.cdf(x))

  def _cdf(self, x):
    # CDF is the probability that the Poisson variable is less or equal to x.
    # For fractional x, the CDF is equal to the CDF at n = floor(x).
    # For negative x, the CDF is zero, but tf.igammac gives NaNs, so we impute
    # the values and handle this case explicitly.
    safe_x = tf.maximum(x if self.interpolate_nondiscrete else tf.floor(x), 0.)
    cdf = tf.math.igammac(1. + safe_x, self.rate)
    return tf.where(
        tf.broadcast_to(x < 0., tf.shape(input=cdf)), tf.zeros_like(cdf), cdf)

  def _log_normalization(self):
    return self.rate

  def _log_unnormalized_prob(self, x):
    # The log-probability at negative points is always -inf.
    # Catch such x's and set the output value accordingly.
    safe_x = tf.maximum(x if self.interpolate_nondiscrete else tf.floor(x), 0.)
    y = safe_x * self.log_rate - tf.math.lgamma(1. + safe_x)
    is_supported = tf.broadcast_to(tf.equal(x, safe_x), tf.shape(input=y))
    neg_inf = tf.fill(
        tf.shape(input=y),
        value=np.array(-np.inf, dtype=y.dtype.as_numpy_dtype))
    return tf.where(is_supported, y, neg_inf)

  def _mean(self):
    return tf.identity(self.rate)

  def _variance(self):
    return tf.identity(self.rate)

  @distribution_util.AppendDocstring(
      """Note: when `rate` is an integer, there are actually two modes: `rate`
      and `rate - 1`. In this case we return the larger, i.e., `rate`.""")
  def _mode(self):
    return tf.floor(self.rate)

  def _sample_n(self, n, seed=None):
    return tf.random.poisson(
        lam=self.rate, shape=[n], dtype=self.dtype, seed=seed)
