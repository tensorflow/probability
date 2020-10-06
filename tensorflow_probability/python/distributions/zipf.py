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
"""The Zipf distribution class."""

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
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'Zipf',
]


class Zipf(distribution.Distribution):
  """Zipf distribution.

  The Zipf distribution is parameterized by a `power` parameter.

  #### Mathematical Details

  The probability mass function (pmf) is,

  ```none
  pmf(k; alpha, k >= 0) = (k^(-alpha)) / Z
  Z = zeta(alpha).
  ```

  where `power = alpha` and Z is the normalization constant.
  `zeta` is the [Riemann zeta function](
  https://en.wikipedia.org/wiki/Riemann_zeta_function).

  Note that gradients with respect to the `power` parameter are not
  supported in the current implementation.
  """

  def __init__(self,
               power,
               dtype=tf.int32,
               interpolate_nondiscrete=True,
               sample_maximum_iterations=100,
               validate_args=False,
               allow_nan_stats=False,
               name='Zipf'):
    """Initialize a batch of Zipf distributions.

    Args:
      power: `Float` like `Tensor` representing the power parameter. Must be
        strictly greater than `1`.
      dtype: The `dtype` of `Tensor` returned by `sample`.
        Default value: `tf.int32`.
      interpolate_nondiscrete: Python `bool`. When `False`, `log_prob` returns
        `-inf` (and `prob` returns `0`) for non-integer inputs. When `True`,
        `log_prob` evaluates the continuous function `-power log(k) -
        log(zeta(power))` , which matches the Zipf pmf at integer arguments `k`
        (note that this function is not itself a normalized probability
        log-density).
        Default value: `True`.
      sample_maximum_iterations: Maximum number of iterations of allowable
        iterations in `sample`. When `validate_args=True`, samples which fail to
        reach convergence (subject to this cap) are masked out with
        `self.dtype.min` or `nan` depending on `self.dtype.is_integer`.
        Default value: `100`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `'Zipf'`.

    Raises:
      TypeError: if `power` is not `float` like.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._power = tensor_util.convert_nonref_to_tensor(
          power,
          name='power',
          dtype=dtype_util.common_dtype([power], dtype_hint=tf.float32))
      if (not dtype_util.is_floating(self._power.dtype) or
          dtype_util.base_equal(self._power.dtype, tf.float16)):
        raise TypeError(
            'power.dtype ({}) is not a supported `float` type.'.format(
                dtype_util.name(self._power.dtype)))
      self._interpolate_nondiscrete = interpolate_nondiscrete
      self._sample_maximum_iterations = sample_maximum_iterations
      super(Zipf, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(power=0)

  @property
  def power(self):
    """Exponent parameter."""
    return self._power

  @property
  def interpolate_nondiscrete(self):
    """Interpolate (log) probs on non-integer inputs."""
    return self._interpolate_nondiscrete

  @property
  def sample_maximum_iterations(self):
    """Maximum number of allowable iterations in `sample`."""
    return self._sample_maximum_iterations

  def _batch_shape_tensor(self):
    return tf.shape(self.power)

  def _batch_shape(self):
    return self.power.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x, power=None):
    # The log probability at positive integer points x is log(x^(-power) / Z)
    # where Z is the normalization constant. For x < 1 and non-integer points,
    # the log-probability is -inf.
    #
    # However, if interpolate_nondiscrete is True, we return the natural
    # continuous relaxation for x >= 1 which agrees with the log probability at
    # positive integer points.
    power = power if power is not None else tf.convert_to_tensor(self.power)
    x = tf.cast(x, power.dtype)
    log_normalization = tf.math.log(tf.math.zeta(power, 1.))

    safe_x = tf.maximum(x if self.interpolate_nondiscrete else tf.floor(x), 1.)
    y = -power * tf.math.log(safe_x)
    log_unnormalized_prob = tf.where(
        tf.equal(x, safe_x), y, dtype_util.as_numpy_dtype(y.dtype)(-np.inf))

    return log_unnormalized_prob - log_normalization

  def _cdf(self, x):
    # CDF(x) at positive integer x is the probability that the Zipf variable is
    # less than or equal to x; given by the formula:
    #     CDF(x) = 1 - (zeta(power, x + 1) / Z)
    # For fractional x, the CDF is equal to the CDF at n = floor(x).
    # For x < 1, the CDF is zero.

    # If interpolate_nondiscrete is True, we return a continuous relaxation
    # which agrees with the CDF at integer points.
    power = tf.convert_to_tensor(self.power)
    x = tf.cast(x, power.dtype)
    safe_x = tf.maximum(x if self.interpolate_nondiscrete else tf.floor(x), 0.)

    cdf = 1. - (
        tf.math.zeta(power, safe_x + 1.) / tf.math.zeta(power, 1.))
    return tf.where(x < 1., tf.zeros_like(cdf), cdf)

  @distribution_util.AppendDocstring(
      """Note: Zipf has an infinite mean when `power` <= 2.""")
  def _mean(self):
    zeta_p = tf.math.zeta(
        self.power[..., tf.newaxis] -
        np.array([0., 1.], dtype_util.as_numpy_dtype(self.dtype)), 1.)
    return zeta_p[..., 1] / zeta_p[..., 0]

  @distribution_util.AppendDocstring(
      """Note: Zipf has infinite variance when `power` <= 3.""")
  def _variance(self):
    zeta_p = tf.math.zeta(
        self.power[..., tf.newaxis] -
        np.array([0., 1., 2.], dtype_util.as_numpy_dtype(self.dtype)), 1.)
    return ((zeta_p[..., 0] * zeta_p[..., 2]) - (zeta_p[..., 1]**2)) / (
        zeta_p[..., 0]**2)

  def _mode(self):
    return tf.ones_like(self.power, dtype=self.dtype)

  @distribution_util.AppendDocstring(
      """The sampling algorithm is rejection-inversion; Algorithm ZRI of
      [Horman and Derflinger (1996)][1]. For simplicity, we don't use the
      squeeze function in our implementation.

      #### References
      [1]: W. Hormann , G. Derflinger, Rejection-inversion to generate variates
           from monotone discrete distributions, ACM Transactions on Modeling and
           Computer Simulation (TOMACS), v.6 n.3, p.169-184, July 1996.
      """)
  def _sample_n(self, n, seed=None):
    power = tf.convert_to_tensor(self.power)
    shape = tf.concat([[n], tf.shape(power)], axis=0)

    seed = samplers.sanitize_seed(seed, salt='zipf')

    minval_u = self._hat_integral(0.5, power=power) + 1.
    maxval_u = self._hat_integral(
        dtype_util.max(tf.int64) - 0.5, power=power)

    def loop_body(should_continue, k, seed):
      """Resample the non-accepted points."""
      u_seed, next_seed = samplers.split_seed(seed)
      # The range of U is chosen so that the resulting sample K lies in
      # [0, tf.int64.max). The final sample, if accepted, is K + 1.
      u = samplers.uniform(
          shape,
          minval=minval_u,
          maxval=maxval_u,
          dtype=power.dtype,
          seed=u_seed)
      # set_shape needed here because of b/139013403
      tensorshape_util.set_shape(u, should_continue.shape)

      # Sample the point X from the continuous density h(x) \propto x^(-power).
      x = self._hat_integral_inverse(u, power=power)

      # Rejection-inversion requires a `hat` function, h(x) such that
      # \int_{k - .5}^{k + .5} h(x) dx >= pmf(k + 1) for points k in the
      # support. A natural hat function for us is h(x) = x^(-power).
      #
      # After sampling X from h(x), suppose it lies in the interval
      # (K - .5, K + .5) for integer K. Then the corresponding K is accepted if
      # if lies to the left of x_K, where x_K is defined by:
      #   \int_{x_k}^{K + .5} h(x) dx = H(x_K) - H(K + .5) = pmf(K + 1),
      # where H(x) = \int_x^inf h(x) dx.

      # Solving for x_K, we find that x_K = H_inverse(H(K + .5) + pmf(K + 1)).
      # Or, the acceptance condition is X <= H_inverse(H(K + .5) + pmf(K + 1)).
      # Since X = H_inverse(U), this simplifies to U <= H(K + .5) + pmf(K + 1).

      # Update the non-accepted points.
      # Since X \in (K - .5, K + .5), the sample K is chosen as floor(X + 0.5).
      k = tf.where(should_continue, tf.floor(x + 0.5), k)
      accept = (u <= self._hat_integral(k + .5, power=power) + tf.exp(
          self._log_prob(k + 1, power=power)))

      return [should_continue & (~accept), k, next_seed]

    should_continue, samples, _ = tf.while_loop(
        cond=lambda should_continue, *ignore: tf.reduce_any(should_continue),
        body=loop_body,
        loop_vars=[
            tf.ones(shape, dtype=tf.bool),  # should_continue
            tf.zeros(shape, dtype=power.dtype),  # k
            seed,  # seed
        ],
        maximum_iterations=self.sample_maximum_iterations,
    )
    samples = samples + 1.

    if self.validate_args and dtype_util.is_integer(self.dtype):
      samples = distribution_util.embed_check_integer_casting_closed(
          samples, target_dtype=self.dtype, assert_positive=True)

    samples = tf.cast(samples, self.dtype)

    if self.validate_args:
      npdt = dtype_util.as_numpy_dtype(self.dtype)
      v = npdt(dtype_util.min(npdt) if dtype_util.is_integer(npdt) else np.nan)
      samples = tf.where(should_continue, v, samples)

    return samples

  def _hat_integral(self, x, power):
    """Integral of the `hat` function, used for sampling.

    We choose a `hat` function, h(x) = x^(-power), which is a continuous
    (unnormalized) density touching each positive integer at the (unnormalized)
    pmf. This function implements `hat` integral: H(x) = int_x^inf h(t) dt;
    which is needed for sampling purposes.

    Arguments:
      x: A Tensor of points x at which to evaluate H(x).
      power: Power that parameterized hat function.

    Returns:
      A Tensor containing evaluation H(x) at x.
    """
    x = tf.cast(x, power.dtype)
    t = power - 1.
    return tf.exp((-t) * tf.math.log1p(x) - tf.math.log(t))

  def _hat_integral_inverse(self, x, power):
    """Inverse function of _hat_integral."""
    x = tf.cast(x, power.dtype)
    t = power - 1.
    return tf.math.expm1(-(tf.math.log(t) + tf.math.log(x)) / t)

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.power):
      assertions.append(assert_util.assert_greater(
          self.power, np.ones([], dtype_util.as_numpy_dtype(self.power.dtype)),
          message='`power` must be greater than 1.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='samples must be non-negative'))
    if not self.interpolate_nondiscrete:
      assertions.append(distribution_util.assert_integer_form(
          x, message='samples cannot contain fractional components.'))
    return assertions
