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

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import batched_rejection_sampler as brs
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import implementation_selection
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'random_poisson',
    'Poisson',
]


def _random_poisson_cpu(
    shape,
    rates=None,
    log_rates=None,
    output_dtype=tf.float32,
    seed=None,
    name=None):
  """Sample using *fast* `tf.random.stateless_poisson`."""
  with tf.name_scope(name or 'poisson_cpu'):
    if rates is None:
      rates = tf.math.exp(log_rates)
    shape = tf.concat([shape, tf.shape(rates)], axis=0)
    return tf.random.stateless_poisson(
        shape=shape, seed=seed, lam=rates, dtype=output_dtype)


def _random_poisson_noncpu(
    shape,
    rates=None,
    log_rates=None,
    output_dtype=tf.float32,
    seed=None,
    name=None):
  """Sample using XLA-friendly python-based rejection sampler."""
  with tf.name_scope(name or 'poisson_noncpu'):
    if log_rates is None:
      log_rates = tf.math.log(rates)
    shape = ps.concat([shape, ps.shape(log_rates)], axis=0)
    good_params_mask = ~tf.math.is_nan(log_rates)
    internal_dtype = tf.float64

    seed_lo, seed_hi = samplers.split_seed(seed)

    # First, we sample the values for which rate >= 10.
    # When replacing NaN or < 10 values, use 100 for log rate, since that leads
    # to a high-likelihood of the rejection sampler accepting on the first pass.
    high_params_mask = good_params_mask & (log_rates >= np.log(10.))
    cast_log_rates = tf.cast(log_rates, internal_dtype)
    safe_log_rates = tf.where(high_params_mask, cast_log_rates, 100.)
    high_rate_samples = _random_poisson_high_rate(
        shape,
        log_rate=safe_log_rates,
        internal_dtype=internal_dtype,
        seed=seed_hi)
    high_rate_samples = tf.cast(high_rate_samples, output_dtype)

    # Next, we sample the values for which rate < 10. When replacing NaN or high
    # values, use a small number so that the sum-of-exponentials sampler
    # terminates on the first pass with high likelihood.
    low_params_mask = good_params_mask & ~high_params_mask
    safe_rate = tf.where(low_params_mask, tf.math.exp(cast_log_rates), 1e-5)
    low_rate_samples = _random_poisson_low_rate(
        shape, rate=safe_rate, internal_dtype=internal_dtype, seed=seed_lo)
    low_rate_samples = tf.cast(low_rate_samples, output_dtype)

    samples = tf.where(
        good_params_mask,
        tf.where(high_params_mask, high_rate_samples, low_rate_samples), np.nan)

  return samples


# tf.function required to access Grappler's implementation_selector.
@implementation_selection.never_runs_functions_eagerly
# TODO(b/163029794): Shape relaxation breaks XLA.
@tf.function(autograph=False)
def random_poisson(
    shape,
    rates=None,
    log_rates=None,
    output_dtype=tf.float32,
    seed=None,
    name=None):
  """Sample a poisson, CPU specialized to stateless_poisson.

  Args:
    shape: Shape of the full sample output. Trailing dims should match the
      broadcast shape of `counts` with `probs|logits`.
    rates: Batch of rates for Poisson distribution.
    log_rates: Batch of log rates for Poisson distribution.
    output_dtype: DType of samples.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Optional name for related ops.

  Returns:
    samples: Samples from poisson distributions.
    runtime_used_for_sampling: One of `implementation_selection._RUNTIME_*`.
  """
  with tf.name_scope(name or 'random_poisson'):
    seed = samplers.sanitize_seed(seed)
    shape = ps.convert_to_shape_tensor(shape, dtype_hint=tf.int32, name='shape')
    params = dict(shape=shape, rates=rates, log_rates=log_rates,
                  output_dtype=output_dtype, seed=seed, name=name)
    sampler_impl = implementation_selection.implementation_selecting(
        fn_name='poisson',
        default_fn=_random_poisson_noncpu,
        cpu_fn=_random_poisson_cpu)
    return sampler_impl(**params)


class Poisson(distribution.AutoCompositeTensorDistribution):
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

  @deprecation.deprecated_args(
      '2021-02-01',
      ('The `interpolate_nondiscrete` flag is deprecated; instead use '
       '`force_probs_to_zero_outside_support` (with the opposite sense).'),
      'interpolate_nondiscrete',
      warn_once=True)
  def __init__(self,
               rate=None,
               log_rate=None,
               force_probs_to_zero_outside_support=None,
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
      force_probs_to_zero_outside_support: Python `bool`. When `True`, negative
        and non-integer values are evaluated "strictly": `log_prob` returns
        `-inf`, `prob` returns `0`, and `cdf` and `sf` correspond.  When
        `False`, the implementation is free to save computation (and TF graph
        size) by evaluating something that matches the Poisson pmf at integer
        values `k` but produces an unrestricted result on other inputs.  In the
        case of Poisson, the `log_prob` formula in this case happens to be the
        continuous function `k * log_rate - lgamma(k+1) - rate`.  Note that this
        function is not itself a normalized probability log-density.
        Default value: `False`.
      interpolate_nondiscrete: Deprecated.  Use
        `force_probs_to_zero_outside_support` (with the opposite sense) instead.
        Python `bool`. When `False`, `log_prob` returns `-inf` (and `prob`
        returns `0`) for non-integer inputs. When `True`, `log_prob` evaluates
        the continuous function `k * log_rate - lgamma(k+1) - rate`, which
        matches the Poisson pmf at integer arguments `k` (note that this
        function is not itself a normalized probability log-density).
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
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, name='rate', dtype=dtype)
      self._log_rate = tensor_util.convert_nonref_to_tensor(
          log_rate, name='log_rate', dtype=dtype)

      self._interpolate_nondiscrete = interpolate_nondiscrete
      if force_probs_to_zero_outside_support is not None:
        # `force_probs_to_zero_outside_support` was explicitly set, so it
        # controls.
        self._force_probs_to_zero_outside_support = (
            force_probs_to_zero_outside_support)
      elif not self._interpolate_nondiscrete:
        # `interpolate_nondiscrete` was explicitly set by the caller, so it
        # should control until it is removed.
        self._force_probs_to_zero_outside_support = True
      else:
        # Default.
        self._force_probs_to_zero_outside_support = False
      super(Poisson, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False),
        log_rate=parameter_properties.ParameterProperties())
    # pylint: enable=g-long-lambda

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  @property
  def log_rate(self):
    """Log rate parameter."""
    return self._log_rate

  @property
  @deprecation.deprecated(
      '2021-02-01',
      ('The `interpolate_nondiscrete` property is deprecated; instead use '
       '`force_probs_to_zero_outside_support` (with the opposite sense).'),
      warn_once=True)
  def interpolate_nondiscrete(self):
    """Interpolate (log) probs on non-integer inputs."""
    return self._interpolate_nondiscrete

  @property
  def force_probs_to_zero_outside_support(self):
    """Return 0 probabilities on non-integer inputs."""
    return self._force_probs_to_zero_outside_support

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    log_rate = self._log_rate_parameter_no_checks()
    log_probs = (self._log_unnormalized_prob(x, log_rate) -
                 self._log_normalization(log_rate))
    if self.force_probs_to_zero_outside_support:
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
    safe_x = tf.maximum(
        tf.floor(x) if self.force_probs_to_zero_outside_support else x, 0.)
    cdf = tf.math.igammac(1. + safe_x, self._rate_parameter_no_checks())
    return tf.where(x < 0., tf.zeros_like(cdf), cdf)

  def _log_survival_function(self, x):
    return tf.math.log(self.survival_function(x))

  def _survival_function(self, x):
    safe_x = tf.maximum(
        tf.floor(x) if self.force_probs_to_zero_outside_support else x, 0.)
    survival = tf.math.igamma(1. + safe_x, self._rate_parameter_no_checks())
    return tf.where(x < 0., tf.ones_like(survival), survival)

  def _log_normalization(self, log_rate):
    return tf.exp(log_rate)

  def _log_unnormalized_prob(self, x, log_rate):
    # The log-probability at negative points is always -inf.
    # Catch such x's and set the output value accordingly.
    safe_x = tf.maximum(
        tf.floor(x) if self.force_probs_to_zero_outside_support else x, 0.)
    y = tf.math.multiply_no_nan(log_rate, safe_x) - tf.math.lgamma(1. + safe_x)
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
    seed = samplers.sanitize_seed(seed)
    return random_poisson(
        shape=ps.convert_to_shape_tensor([n]),
        rates=(None if self._rate is None else
               tf.convert_to_tensor(self._rate)),
        log_rates=(None if self._log_rate is None else
                   tf.convert_to_tensor(self._log_rate)),
        output_dtype=self.dtype,
        seed=seed)[0]

  def rate_parameter(self, name=None):
    """Rate vec computed from non-`None` input arg (`rate` or `log_rate`)."""
    with self._name_and_control_scope(name or 'rate_parameter'):
      return self._rate_parameter_no_checks()

  def _rate_parameter_no_checks(self):
    if self._rate is None:
      return tf.exp(self._log_rate)
    return tensor_util.identity_as_tensor(self._rate)

  def log_rate_parameter(self, name=None):
    """Log-rate vec computed from non-`None` input arg (`rate`, `log_rate`)."""
    with self._name_and_control_scope(name or 'log_rate_parameter'):
      return self._log_rate_parameter_no_checks()

  def _log_rate_parameter_no_checks(self):
    if self._log_rate is None:
      return tf.math.log(self._rate)
    return tensor_util.identity_as_tensor(self._log_rate)

  def _default_event_space_bijector(self):
    return

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'log_rate': tf.math.log(tf.reduce_mean(value, axis=0))}

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if self._rate is not None:
      if is_init != tensor_util.is_ref(self._rate):
        assertions.append(assert_util.assert_non_negative(
            self._rate,
            message='Argument `rate` must be non-negative.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    return assertions


def _random_poisson_high_rate(sample_shape,
                              log_rate,
                              internal_dtype=tf.float64,
                              seed=None):
  """Samples from the Poisson distribution using transformed rejection sampling.

  Given a CDF F(x), and G(x), a dominating distribution chosen such that it is
  close to the inverse CDF F^-1(x), compute the following steps:

  1) Generate U and V, two independent random variates. Set U = U - 0.5 (this
  step isn't strictly necessary, but is done to make some calculations symmetric
  and convenient. Henceforth, G is defined on [-0.5, 0.5]).

  2) If V <= alpha * F'(G(U)) * G'(U), return floor(G(U)), else return to
  step 1. alpha is the acceptance probability of the rejection algorithm.
  The dominating distribution in this case:
    G(u) = (2 * a / (2 - |u|) + b) * u + c

  For more details on transformed rejection, see [1].

  Args:
    sample_shape: The output sample shape. Must broadcast with `log_rate`.
    log_rate: Floating point tensor, log rate.
    internal_dtype: dtype to use for internal computations.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    Samples from the poisson distribution using transformed rejection.

  #### References

  [1]: W. Hormann, G. Derflinger, The Transformed Rejection Method For
  Generating Random Variables, An Alternative To The Ratio Of Uniforms Method
  (1994), Manuskript, Institut f. Statistik, Wirtschaftsuniversitat
  """
  rate = tf.math.exp(log_rate)

  b = 0.931 + 2.53 * tf.math.exp(0.5 * log_rate)
  a = -0.059 + 0.02483 * b
  inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)

  def generate_and_test_samples(seed):
    """Generate and test samples."""
    u_seed, v_seed = samplers.split_seed(seed)

    u = samplers.uniform(sample_shape, dtype=internal_dtype, seed=u_seed)
    u = u - 0.5
    u_shifted = 0.5 - tf.math.abs(u)

    v = samplers.uniform(sample_shape, dtype=internal_dtype, seed=v_seed)

    k = tf.math.floor(((2. * a) / u_shifted + b) * u + rate + 0.43)

    good_sample_mask = (u_shifted >= 0.07) & (v <= 0.9277 - 3.6224 / (b - 2.))

    s = tf.math.log(v * inverse_alpha / (a / tf.math.square(u_shifted) + b))
    t = -rate + k * log_rate - tf.math.lgamma(k + 1)

    good_sample_mask = good_sample_mask | (s <= t)
    # Make sure the sample is within bounds.
    good_sample_mask = good_sample_mask & (k >= 0) & ((u_shifted >= 0.013) |
                                                      (v <= u_shifted))
    return k, good_sample_mask

  samples = brs.batched_las_vegas_algorithm(
      generate_and_test_samples, seed=seed)[0]

  return samples


def _random_poisson_low_rate(sample_shape,
                             rate,
                             internal_dtype=tf.float64,
                             seed=None):
  """Samples from the Poisson distribution using Knuth's algorithm.

  We use an algorithm attributed to Knuth: Seminumerical Algorithms. Art of
  Computer Programming, Volume 2. This algorithm runs in O(rate) time, and
  requires O(rate) uniform variates. This algorithm is performant for rate ~<10.

  Given a Poisson process, the time between events is exponentially distributed.
  If we have a Poisson process with rate lambda, then, the time between events
  is distributed as Exp(lambda). If X ~ Uniform(0, 1), then Y ~ Exp(lambda)
  where Y = -log(X) / lambda. Thus, to simulate a Poisson draw, we can sample
  X_i ~ Exp(lambda), and we will haver N ~ Poisson(lambda), where N is the
  smallest number such that sum_i^N X_i > 1.

  Args:
    sample_shape: The output sample shape. Must broadcast with `rate`.
    rate: Floating point tensor, rate.
    internal_dtype: (optional) dtype to use for internal computations.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    Samples from the poisson distribution.
  """
  exp_neg_rate = tf.math.exp(-rate)

  def loop_body(should_continue, samples, prod, num_iters, seed):
    u_seed, next_seed = samplers.split_seed(seed)
    prod = prod * samplers.uniform(
        sample_shape, dtype=internal_dtype, seed=u_seed)
    accept = should_continue & (prod <= exp_neg_rate)
    samples = tf.where(accept, num_iters, samples)
    return [
        should_continue & (~accept), samples, prod, num_iters + 1, next_seed
    ]

  _, samples, _, _, _ = tf.while_loop(
      cond=lambda should_continue, *ignore: tf.reduce_any(should_continue),
      body=loop_body,
      loop_vars=[
          tf.ones(sample_shape, dtype=tf.bool),  # should_continue
          tf.zeros(sample_shape, dtype=tf.int32),  # samples
          tf.ones(sample_shape, dtype=internal_dtype),  # prod
          tf.zeros([], dtype=tf.int32),  # num_iters
          seed,  # seed
      ],
      # Using a Chernoff-like bound, we can show that for lambda < 10,
      # Pr[X >= lambda + n] <= exp(-n^2 / 2(lambda + n)) < exp(-90). Hence,
      # there is miniscule probability that, even after a union bound over
      # batch size, a poisson sample with rate < 10 would attain a value > 200.
      maximum_iterations=200,
  )
  return samples
