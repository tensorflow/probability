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
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'Poisson',
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
               name='Poisson'):
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
    if (rate is None) == (log_rate is None):
      raise ValueError('Must specify exactly one of `rate` and `log_rate`.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([rate, log_rate], dtype_hint=tf.float32)
      if not dtype_util.is_floating(dtype):
        raise TypeError('[log_]rate.dtype ({}) is a not a float-type.'.format(
            dtype_util.name(dtype)))
      self._rate = tensor_util.convert_immutable_to_tensor(
          rate, name='rate', dtype=dtype)
      self._log_rate = tensor_util.convert_immutable_to_tensor(
          log_rate, name='log_rate', dtype=dtype)

      self._interpolate_nondiscrete = interpolate_nondiscrete
      super(Poisson, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(rate=0, log_rate=0)

  @property
  def rate(self):
    """Rate parameter."""
    if self._rate is None:
      return self._rate_deprecated_behavior()
    return self._rate

  @property
  def log_rate(self):
    """Log rate parameter."""
    if self._log_rate is None:
      return self._log_rate_deprecated_behavior()
    return self._log_rate

  @property
  def interpolate_nondiscrete(self):
    """Interpolate (log) probs on non-integer inputs."""
    return self._interpolate_nondiscrete

  def _batch_shape_tensor(self):
    x = self._rate if self._log_rate is None else self._log_rate
    return tf.shape(x)

  def _batch_shape(self):
    x = self._rate if self._log_rate is None else self._log_rate
    return x.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    log_rate = self._log_rate_parameter_no_checks()
    log_probs = (self._log_unnormalized_prob(x, log_rate) -
                 self._log_normalization(log_rate))
    if not self.interpolate_nondiscrete:
      # Ensure the gradient wrt `rate` is zero at non-integer points.
      log_probs = tf.where(
          tf.math.is_inf(log_probs),
          dtype_util.as_numpy_dtype(log_probs.dtype)(-np.inf),
          log_probs)
    return log_probs

  def _log_cdf(self, x):
    return tf.math.log(self.cdf(x))

  def _cdf(self, x):
    # CDF is the probability that the Poisson variable is less or equal to x.
    # For fractional x, the CDF is equal to the CDF at n = floor(x).
    # For negative x, the CDF is zero, but tf.igammac gives NaNs, so we impute
    # the values and handle this case explicitly.
    safe_x = tf.maximum(x if self.interpolate_nondiscrete else tf.floor(x), 0.)
    cdf = tf.math.igammac(1. + safe_x, self._rate_parameter_no_checks())
    return tf.where(x < 0., tf.zeros_like(cdf), cdf)

  def _log_normalization(self, log_rate):
    return tf.exp(log_rate)

  def _log_unnormalized_prob(self, x, log_rate):
    # The log-probability at negative points is always -inf.
    # Catch such x's and set the output value accordingly.
    safe_x = tf.maximum(x if self.interpolate_nondiscrete else tf.floor(x), 0.)
    y = safe_x * log_rate - tf.math.lgamma(1. + safe_x)
    return tf.where(
        tf.equal(x, safe_x), y, dtype_util.as_numpy_dtype(y.dtype)(-np.inf))

  def _mean(self):
    return self._rate_parameter_no_checks()

  def _variance(self):
    return self._rate_parameter_no_checks()

  @distribution_util.AppendDocstring(
      """Note: when `rate` is an integer, there are actually two modes: `rate`
      and `rate - 1`. In this case we return the larger, i.e., `rate`.""")
  def _mode(self):
    return tf.floor(self._rate_parameter_no_checks())

  def _sample_n(self, n, seed=None):
    lam = self._rate_parameter_no_checks()
    return tf.random.poisson(lam=lam, shape=[n], dtype=self.dtype, seed=seed)

  def rate_parameter(self, name=None):
    """Rate vec computed from non-`None` input arg (`rate` or `log_rate`)."""
    with self._name_and_control_scope(name or 'rate_parameter'):
      return self._rate_parameter_no_checks()

  def _rate_parameter_no_checks(self):
    if self._rate is None:
      return tf.exp(self._log_rate)
    return tf.identity(self._rate)

  def log_rate_parameter(self, name=None):
    """Log-rate vec computed from non-`None` input arg (`rate`, `log_rate`)."""
    with self._name_and_control_scope(name or 'log_rate_parameter'):
      return self._log_rate_parameter_no_checks()

  def _log_rate_parameter_no_checks(self):
    if self._log_rate is None:
      return tf.math.log(self._rate)
    return tf.identity(self._log_rate)

  @deprecation.deprecated(
      '2019-10-01',
      'The `rate` property will return `None` when the distribution is '
      'parameterized with `rate=None`. Use `rate_parameter()` instead.',
      warn_once=True)
  def _rate_deprecated_behavior(self):
    return self.rate_parameter()

  @deprecation.deprecated(
      '2019-10-01',
      'The `log_rate` property will return `None` when the distribution is '
      'parameterized with `log_rate=None`. Use `log_rate_parameter()` instead.',
      warn_once=True)
  def _log_rate_deprecated_behavior(self):
    return self.log_rate_parameter()

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if self._rate is not None:
      if is_init != tensor_util.is_mutable(self._rate):
        assertions.append(assert_util.assert_positive(
            self._rate,
            message='Argument `rate` must be positive.'))
    return assertions
