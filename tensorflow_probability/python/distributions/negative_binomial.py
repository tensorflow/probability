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
"""The Negative Binomial distribution class."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor


class NegativeBinomial(distribution.AutoCompositeTensorDistribution):
  """NegativeBinomial distribution.

  The NegativeBinomial distribution is related to the experiment of performing
  Bernoulli trials in sequence. Given a Bernoulli trial with probability `p` of
  success, the NegativeBinomial distribution represents the distribution over
  the number of successes `s` that occur until we observe `f` failures.

  The probability mass function (pmf) is,

  ```none
  pmf(s; f, p) = p**s (1 - p)**f / Z
  Z = s! (f - 1)! / (s + f - 1)!
  ```

  where:
  * `total_count = f`,
  * `probs = p`,
  * `Z` is the normalizaing constant, and,
  * `n!` is the factorial of `n`.
  """

  def __init__(self,
               total_count,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name='NegativeBinomial'):
    """Construct NegativeBinomial distributions.

    Args:
      total_count: Positive floating-point `Tensor` with shape broadcastable to
        `[B1,..., Bb]` with `b >= 0` and the same dtype as `probs` or
        `logits`. Defines this as a batch of `N1 x ... x Nm` different Negative
        Binomial distributions. In practice, this represents the number of
        negative Bernoulli trials to stop at (the `total_count` of
        failures). Its components should be equal to integer values.
      logits: Floating-point `Tensor` with shape broadcastable to
        `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
        Each entry represents logits for the probability of success for
        independent Negative Binomial distributions and must be in the half-open
        interval `[-inf, inf)`. Only one of `logits` or `probs` should be
        specified.
      probs: Positive floating-point `Tensor` with shape broadcastable to
        `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
        Each entry represents the probability of success for independent
        Negative Binomial distributions and must be in the half-open interval
        `[0, 1)`. Only one of `logits` or `probs` should be specified.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """

    parameters = dict(locals())
    if (probs is None) == (logits is None):
      raise ValueError(
          'Construct `NegativeBinomial` with `probs` or `logits` but not both.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([total_count, logits, probs],
                                      dtype_hint=tf.float32)
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype=dtype, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype=dtype, name='logits')
      self._total_count = tensor_util.convert_nonref_to_tensor(
          total_count, dtype=dtype, name='total_count')

      super(NegativeBinomial, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def experimental_from_mean_dispersion(cls, mean, dispersion, **kwargs):
    """Constructs a NegativeBinomial from its mean and dispersion.

    **Experimental: Naming, location of this API may change.**

    In this parameterization, the dispersion is defined as the reciprocal of the
    total count of failures, i.e. `dispersion = 1 / total_count`.

    Args:
      mean: The mean of the constructed distribution.
      dispersion: The reciprocal of the total_count of the constructed
        distribution.
      **kwargs: Other keyword arguments passed directly to `__init__`, e.g.
        `validate_args`.

    Returns:
      neg_bin: A distribution with the given parameterization.
    """
    dtype = dtype_util.common_dtype([mean, dispersion], dtype_hint=tf.float32)
    dispersion = tensor_util.convert_nonref_to_tensor(dispersion, dtype=dtype)
    mean = tensor_util.convert_nonref_to_tensor(mean, dtype=dtype)

    total_count = DeferredTensor(dispersion, tf.math.reciprocal, dtype=dtype)

    return cls(
        total_count=total_count,
        probs=DeferredTensor(mean, lambda mean: mean / (mean + total_count)),
        **kwargs)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        total_count=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        logits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))

  @property
  def total_count(self):
    """Number of negative trials."""
    return self._total_count

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    # Here we use the fact that if:
    # lam ~ Gamma(concentration=total_count, rate=(1-probs)/probs)
    # then X ~ Poisson(lam) is Negative Binomially distributed.
    logits = self._logits_parameter_no_checks()
    gamma_seed, poisson_seed = samplers.split_seed(
        seed, salt='NegativeBinomial')
    # TODO(b/152785714): For some reason switching to gamma_lib.random_gamma
    # makes tests time out. Note: observed similar in jax_transformation_test.
    rate = samplers.gamma(
        shape=[n],
        alpha=self.total_count,
        beta=tf.math.exp(-logits),
        dtype=self.dtype,
        seed=gamma_seed)
    return samplers.poisson(
        shape=[], lam=rate, dtype=self.dtype, seed=poisson_seed)

  def _cdf(self, x):
    logits = self._logits_parameter_no_checks()
    total_count = tf.convert_to_tensor(self.total_count)
    shape = self._batch_shape_tensor(
        logits=logits, total_count=total_count)
    safe_x = tf.where(x >= 0, x, 0.)
    answer = tf.math.betainc(
        tf.broadcast_to(total_count, shape),
        tf.broadcast_to(1. + safe_x, shape),
        tf.broadcast_to(tf.sigmoid(-logits), shape))
    return distribution_util.extend_cdf_outside_support(x, answer, low=0)

  def _log_prob(self, x):
    total_count = tf.convert_to_tensor(self.total_count)
    logits = self._logits_parameter_no_checks()
    log_unnormalized_prob = (
        total_count * tf.math.log_sigmoid(-logits) +
        tf.math.multiply_no_nan(tf.math.log_sigmoid(logits), x))
    log_normalization = (tfp_math.lbeta(1. + x, total_count) +
                         tf.math.log(total_count + x))
    return log_unnormalized_prob - log_normalization

  def _mean(self, logits=None):
    logits = self._logits_parameter_no_checks() if logits is None else logits
    return self.total_count * tf.exp(logits)

  def _mode(self):
    total_count = tf.convert_to_tensor(self.total_count)
    adjusted_count = tf.where(1. < total_count, total_count - 1.,
                              tf.zeros_like(total_count))
    return tf.floor(adjusted_count * tf.exp(self._logits_parameter_no_checks()))

  def _variance(self):
    logits = self._logits_parameter_no_checks()
    return self._mean(logits=logits) / tf.sigmoid(-logits)

  def _logits_parameter_no_checks(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log(probs) - tf.math.log1p(-probs)
    return tensor_util.identity_as_tensor(self._logits)

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _probs_parameter_no_checks(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    if self._logits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return tf.math.sigmoid(self._logits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    return maybe_assert_negative_binomial_param_correctness(
        is_init, self.validate_args, self._total_count, self._probs,
        self._logits)

  def _sample_control_dependencies(self, x):
    """Check counts for proper shape and values, then return tensor version."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    return assertions


def maybe_assert_negative_binomial_param_correctness(
    is_init, validate_args, total_count, probs, logits):
  """Return assertions for `NegativeBinomial`-type distributions."""
  if is_init:
    x, name = (probs, 'probs') if logits is None else (logits, 'logits')
    if not dtype_util.is_floating(x.dtype):
      raise TypeError(
          'Argument `{}` must having floating type.'.format(name))

  if not validate_args:
    return []

  assertions = []
  if is_init != tensor_util.is_ref(total_count):
    total_count = tf.convert_to_tensor(total_count)
    assertions.extend([
        assert_util.assert_positive(
            total_count,
            message='`total_count` has components less than or equal to 0.'),
        distribution_util.assert_integer_form(
            total_count,
            message='`total_count` has fractional components.')
    ])
  if probs is not None:
    if is_init != tensor_util.is_ref(probs):
      probs = tf.convert_to_tensor(probs)
      one = tf.constant(1., probs.dtype)
      assertions.extend([
          assert_util.assert_non_negative(
              probs, message='`probs` has components less than 0.'),
          assert_util.assert_less_equal(
              probs, one, message='`probs` has components greater than 1.')
      ])

  return assertions
