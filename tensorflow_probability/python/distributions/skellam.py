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
"""The Skellam distribution class."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import poisson as poisson_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import bessel


__all__ = [
    'Skellam',
]


class Skellam(
    distribution.DiscreteDistributionMixin,
    distribution.AutoCompositeTensorDistribution):
  """Skellam distribution.

  The Skellam distribution is parameterized by two rate parameters,
  `rate1` and `rate2`. Its samples are defined as:

  ```
  x ~ Poisson(rate1)
  y ~ Poisson(rate2)
  z = x - y
  z ~ Skellam(rate1, rate2)
  ```
  where the samples `x` and `y` are assumed to be independent.

  #### Mathematical Details

  The probability mass function (pmf) is,

  ```none
  pmf(k; l1, l2) = (l1 / l2) ** (k / 2) * I_k(2 * sqrt(l1 * l2)) / Z
  Z = exp(l1 + l2).
  ```

  where `rate1 = l1`, `rate2 = l2`,  `Z` is the normalizing constant
  and `I_k` is the modified bessel function of the first kind.
  """

  def __init__(self,
               rate1=None,
               rate2=None,
               log_rate1=None,
               log_rate2=None,
               force_probs_to_zero_outside_support=False,
               validate_args=False,
               allow_nan_stats=True,
               name='Skellam'):
    """Initialize a batch of Skellam distributions.

    Args:
      rate1: Floating point tensor, the first rate parameter. `rate1` must be
        positive. Must specify exactly one of `rate1` and `log_rate1`
      rate2: Floating point tensor, the second rate parameter. `rate` must be
        positive.  Must specify exactly one of `rate2` and `log_rate2`.
      log_rate1: Floating point tensor, the log of the first rate parameter.
        Must specify exactly one of `rate1` and `log_rate1`.
      log_rate2: Floating point tensor, the log of the second rate parameter.
        Must specify exactly one of `rate2` and `log_rate2`.
      force_probs_to_zero_outside_support: Python `bool`. When `True`,
        `log_prob` returns `-inf` (and `prob` returns `0`) for non-integer
        inputs. When `False`, `log_prob` evaluates the Skellam pmf as a
        continuous function (note that this function is not itself
        a normalized probability log-density).
        Default value: `False`.
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
      ValueError: if none or both of `rate1`, `log_rate1` are specified.
      ValueError: if none or both of `rate2`, `log_rate2` are specified.
    """
    parameters = dict(locals())
    if (rate1 is None) == (log_rate1 is None):
      raise ValueError('Must specify exactly one of `rate1` and `log_rate1`.')
    if (rate2 is None) == (log_rate2 is None):
      raise ValueError('Must specify exactly one of `rate2` and `log_rate2`.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [rate1, rate2, log_rate1, log_rate2], dtype_hint=tf.float32)
      self._rate1 = tensor_util.convert_nonref_to_tensor(
          rate1, name='rate1', dtype=dtype)
      self._log_rate1 = tensor_util.convert_nonref_to_tensor(
          log_rate1, name='log_rate1', dtype=dtype)

      self._rate2 = tensor_util.convert_nonref_to_tensor(
          rate2, name='rate2', dtype=dtype)
      self._log_rate2 = tensor_util.convert_nonref_to_tensor(
          log_rate2, name='log_rate2', dtype=dtype)

      self._force_probs_to_zero_outside_support = force_probs_to_zero_outside_support
      super(Skellam, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        rate1=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False),
        rate2=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False),
        log_rate1=parameter_properties.ParameterProperties(),
        log_rate2=parameter_properties.ParameterProperties(),
    )

  @property
  def rate1(self):
    """First Rate parameter."""
    return self._rate1

  @property
  def rate2(self):
    """Second rate parameter."""
    return self._rate2

  @property
  def log_rate1(self):
    """First log rate parameter."""
    return self._log_rate1

  @property
  def log_rate2(self):
    """Second log rate parameter."""
    return self._log_rate2

  @property
  def force_probs_to_zero_outside_support(self):
    """Interpolate (log) probs on non-integer inputs."""
    return self._force_probs_to_zero_outside_support

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    # The log-probability at negative points is always -inf.
    # Catch such x's and set the output value accordingly.
    lr1, r1, lr2, r2 = self._all_rate_parameters()

    safe_x = tf.floor(x) if self.force_probs_to_zero_outside_support else x
    y = tf.math.multiply_no_nan(0.5 * (lr1 - lr2), safe_x)
    numpy_dtype = dtype_util.as_numpy_dtype(y.dtype)

    # When both rates are zero, the above computation gives a NaN, whereas
    # it should give zero.
    y = tf.where(
        tf.math.equal(r1, 0.) & tf.math.equal(r2, 0.),
        numpy_dtype(0.), y)
    y = y + bessel.log_bessel_ive(
        safe_x, 2. * tf.math.sqrt(r1 * r2)) - tf.math.square(
            tf.math.sqrt(r1) - tf.math.sqrt(r2))
    y = tf.where(tf.math.equal(x, safe_x), y, numpy_dtype(-np.inf))
    if self.force_probs_to_zero_outside_support:
      # Ensure the gradient wrt `rate` is zero at non-integer points.
      y = tf.where(
          (y < 0.) & tf.math.is_inf(y), numpy_dtype(-np.inf), y)
    return y

  def _mean(self):
    return (self._rate1_parameter_no_checks() -
            self._rate2_parameter_no_checks())

  def _variance(self):
    return (self._rate1_parameter_no_checks() +
            self._rate2_parameter_no_checks())

  def _sample_n(self, n, seed=None):
    seed = samplers.sanitize_seed(seed)
    seed1, seed2 = samplers.split_seed(seed, salt='Skellam')
    log_rate1 = self._log_rate1_parameter_no_checks()
    log_rate2 = self._log_rate2_parameter_no_checks()
    batch_shape = self._batch_shape_tensor(
        log_rate1=log_rate1, log_rate2=log_rate2)
    log_rate1 = ps.broadcast_to(log_rate1, batch_shape)
    log_rate2 = ps.broadcast_to(log_rate2, batch_shape)
    sample1 = poisson_lib.random_poisson(
        [n], log_rates=log_rate1, seed=seed1)[0]
    sample2 = poisson_lib.random_poisson(
        [n], log_rates=log_rate2, seed=seed2)[0]
    return sample1 - sample2

  def rate1_parameter(self, name=None):
    """Rate computed from non-`None` input arg (`rate1` or `log_rate1`)."""
    with self._name_and_control_scope(name or 'rate1_parameter'):
      return self._rate1_parameter_no_checks()

  def _rate1_parameter_no_checks(self):
    if self._rate1 is None:
      return tf.exp(self._log_rate1)
    return tensor_util.identity_as_tensor(self._rate1)

  def log_rate1_parameter(self, name=None):
    """Log-rate computed from non-`None` input arg (`rate1`, `log_rate1`)."""
    with self._name_and_control_scope(name or 'log_rate1_parameter'):
      return self._log_rate1_parameter_no_checks()

  def _log_rate1_parameter_no_checks(self):
    if self._log_rate1 is None:
      return tf.math.log(self._rate1)
    return tensor_util.identity_as_tensor(self._log_rate1)

  def rate2_parameter(self, name=None):
    """Rate computed from non-`None` input arg (`rate2` or `log_rate2`)."""
    with self._name_and_control_scope(name or 'rate2_parameter'):
      return self._rate2_parameter_no_checks()

  def _rate2_parameter_no_checks(self):
    if self._rate2 is None:
      return tf.exp(self._log_rate2)
    return tensor_util.identity_as_tensor(self._rate2)

  def log_rate2_parameter(self, name=None):
    """Log-rate computed from non-`None` input arg (`rate2`, `log_rate2`)."""
    with self._name_and_control_scope(name or 'log_rate2_parameter'):
      return self._log_rate2_parameter_no_checks()

  def _log_rate2_parameter_no_checks(self):
    if self._log_rate2 is None:
      return tf.math.log(self._rate2)
    return tensor_util.identity_as_tensor(self._log_rate2)

  def _all_rate_parameters(self):
    rate1 = None
    log_rate1 = None
    if self._rate1 is None:
      log_rate1 = tf.convert_to_tensor(self._log_rate1)
      rate1 = tf.math.exp(log_rate1)
    else:
      rate1 = tf.convert_to_tensor(self._rate1)
      log_rate1 = tf.math.log(rate1)

    if self._rate2 is None:
      log_rate2 = tf.convert_to_tensor(self._log_rate2)
      rate2 = tf.math.exp(log_rate2)
    else:
      rate2 = tf.convert_to_tensor(self._rate2)
      log_rate2 = tf.math.log(rate2)

    return log_rate1, rate1, log_rate2, rate2

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if self._rate1 is not None:
      if is_init != tensor_util.is_ref(self._rate1):
        assertions.append(assert_util.assert_non_negative(
            self._rate1,
            message='Argument `rate1` must be non-negative.'))
    if self._rate2 is not None:
      if is_init != tensor_util.is_ref(self._rate2):
        assertions.append(assert_util.assert_non_negative(
            self._rate2,
            message='Argument `rate2` must be non-negative.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(distribution_util.assert_integer_form(x))
    return assertions
