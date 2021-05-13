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
"""The Binomial distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import batched_rejection_sampler
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import implementation_selection
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


_binomial_sample_note = """
For each batch member of counts `value`, `P[value]` is the probability that
after sampling `self.total_count` draws from this Binomial distribution, the
number of successes is `value`. Since different sequences of draws can result in
the same counts, the probability includes a combinatorial coefficient.

Note: `value` must be a non-negative tensor with dtype `dtype` and whose shape
can be broadcast with `self.probs` and `self.total_count`. `value` is only legal
if it is less than or equal to `self.total_count` and its components are equal
to integer values.
"""


def _bdtr(k, n, p):
  """The binomial cumulative distribution function.

  Args:
    k: floating point `Tensor`.
    n: floating point `Tensor`.
    p: floating point `Tensor`.

  Returns:
    `sum_{j=0}^k p^j (1 - p)^(n - j)`.
  """
  # Trick for getting safe backprop/gradients into n, k when
  #   betainc(a = 0, ..) = nan
  # Write:
  #   where(unsafe, safe_output, betainc(where(unsafe, safe_input, input)))
  ones = tf.ones_like(n - k)
  safe_dn = tf.where(tf.logical_or(k < 0, k >= n), ones, n - k)
  dk = tf.math.betainc(a=safe_dn, b=k + 1, x=1 - p)
  return distribution_util.extend_cdf_outside_support(k, dk, low=0, high=n)


def _random_binomial_cpu(
    shape,
    counts,
    probs,
    output_dtype=tf.float32,
    seed=None,
    name=None):
  """Sample using *fast* `tf.random.stateless_binomial`."""
  with tf.name_scope(name or 'binomial_cpu'):
    probs = tf.where(counts > 0, probs, 0)
    batch_shape = ps.broadcast_shape(ps.shape(counts), ps.shape(probs))
    samples = tf.random.stateless_binomial(
        shape=ps.concat([shape, batch_shape], axis=0),
        seed=seed, counts=counts, probs=probs, output_dtype=output_dtype)
  return samples


# These functions are ported from random_binomial_op.cc in TF and manually
# vectorized.


def _binomial_inversion(counts, probs, full_shape, seed):
  """Use multiple geometric samples to sample binomials with count*prob < 10."""
  # Binomial inversion. Given probs, sum geometric random variables until they
  # exceed counts. The number of random variables used is binomially
  # distributed. This is also known as binomial inversion, as this is equivalent
  # to inverting the Binomial CDF.
  seed = samplers.sanitize_seed(seed)
  zero_init = tf.zeros(full_shape, counts.dtype)

  def cond(keep_going, *_):
    return keep_going

  # If probs were 1 we would loop forever. :(
  # But probs is always <= 0.5 here, as guaranteed by the caller.
  log1minusprob = tf.math.log1p(-probs)

  def body(unused_keep_going, geom_sum, num_geom, seed):
    u_seed, next_seed = samplers.split_seed(seed)
    u = samplers.uniform(full_shape, seed=u_seed, dtype=counts.dtype)
    geom = tf.math.ceil(tf.math.log(u) / log1minusprob)
    geom_sum += geom
    keep_going = (geom_sum <= counts)
    num_geom = tf.where(keep_going, num_geom + 1, num_geom)
    return tf.reduce_any(keep_going), geom_sum, num_geom, next_seed

  _, _, num_geom, _ = tf.while_loop(
      cond, body, (True, zero_init, zero_init, seed))
  return num_geom


def _stirling_approx_tail(k):
  """Utility for `_btrs`."""
  tail_values = tf.constant(
      [0.0810614667953272, 0.0413406959554092, 0.0276779256849983,
       0.02079067210376509, 0.0166446911898211, 0.0138761288230707,
       0.0118967099458917, 0.0104112652619720, 0.00925546218271273,
       0.00833056343336287],
      dtype=k.dtype)

  safe_tail_k = tf.clip_by_value(tf.cast(k, tf.int32), 0, 9)
  kp1sq = (k + 1) * (k + 1)
  nontail = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1)
  return tf.where(k <= 9, tf.gather(tail_values, safe_tail_k), nontail)


def _btrs(counts, probs, full_shape, seed):
  """Binomial transformed rejection sampler, for count*prob >= 10."""
  # We use a transformation-rejection algorithm from
  # pairs of uniform random variables due to Hormann.
  # https://www.tandfonline.com/doi/abs/10.1080/00949659308811496

  seed = samplers.sanitize_seed(seed)
  # This is spq in the paper.
  stddev = tf.math.sqrt(counts * probs * (1 - probs))

  # Other coefficients for Transformed Rejection sampling.
  b = 1.15 + 2.53 * stddev
  a = -0.0873 + 0.0248 * b + 0.01 * probs
  c = counts * probs + 0.5
  r = probs / (1 - probs)

  alpha = (2.83 + 5.1 / b) * stddev
  m = tf.math.floor((counts + 1) * probs)

  def batched_las_vegas_trial_fn(seed):
    u_seed, v_seed = samplers.split_seed(seed)
    u = samplers.uniform(full_shape, seed=u_seed, dtype=counts.dtype) - 0.5
    v = samplers.uniform(full_shape, seed=v_seed, dtype=counts.dtype)
    us = 0.5 - tf.math.abs(u)
    k = tf.math.floor((2 * a / us + b) * u + c)

    # When the bounding box is tight, this criteria is more numerically stable
    # and equally valid. Particularly on GPU/TPU, it may make the difference
    # between terminating and non-terminating loops.
    v_r = 0.92 - 4.2 / b
    accept_boxed = (us >= 0.07) & (v <= v_r)

    # Reject non-sensical answers.
    reject = (k < 0) | (k > counts)

    # This deviates from Hormann's BTRS algorithm, as there is a log missing.
    # For all (u, v) pairs outside of the bounding box, this calculates the
    # transformed-reject ratio.
    v = tf.math.log(v * alpha / (a / (us * us) + b))
    upperbound = (
        (m + 0.5) * tf.math.log((m + 1) / (r * (counts - m + 1))) +
        (counts + 1) * tf.math.log((counts - m + 1) / (counts - k + 1)) +
        (k + 0.5) * tf.math.log(r * (counts - k + 1) / (k + 1)) +
        _stirling_approx_tail(m) + _stirling_approx_tail(counts - m) -
        _stirling_approx_tail(k) - _stirling_approx_tail(counts - k))
    accept_bounded = v <= upperbound
    return k, (~reject) & (accept_boxed | accept_bounded)

  return batched_rejection_sampler.batched_las_vegas_algorithm(
      batched_las_vegas_trial_fn, seed=seed)[0]  # Pick out samples.


def _random_binomial_noncpu(
    shape,
    counts,
    probs,
    output_dtype=tf.float32,
    seed=None,
    name=None):
  """Sample using XLA-friendly python-based rejection sampler."""
  with tf.name_scope(name or 'binomial_noncpu'):
    probs = tf.where(counts > 0, probs, 0)

    batch_shape = ps.broadcast_shape(ps.shape(counts), ps.shape(probs))
    full_shape = ps.concat([shape, batch_shape], axis=0)

    inversion_seed, btrs_seed = samplers.split_seed(seed)

    p_lt_half = probs < .5
    q = tf.where(p_lt_half, probs, 1 - probs)
    q_is_nan = tf.math.is_nan(q)
    q_le_0 = (q <= 0.)
    q = tf.where(q_is_nan | q_le_0, tf.constant(.01, q.dtype), q)

    use_inversion = counts * q < 10.
    counts_for_inversion = tf.where(use_inversion, counts, 0)
    inversion_samples = _binomial_inversion(
        counts_for_inversion, q, full_shape, inversion_seed)

    counts_for_btrs = tf.where(
        use_inversion, tf.constant(10000., counts.dtype), counts)
    q_for_btrs = tf.where(use_inversion, tf.constant(.5, q.dtype), q)
    btrs_samples = _btrs(counts_for_btrs, q_for_btrs, full_shape, btrs_seed)

    samples = tf.where(use_inversion, inversion_samples, btrs_samples)
    samples = tf.where(q_le_0, tf.zeros([], samples.dtype), samples)
    samples = tf.where(
        q_is_nan, tf.constant(float('nan'), samples.dtype), samples)
    samples = tf.where(p_lt_half, samples, counts - samples)
    return tf.stop_gradient(tf.cast(samples, output_dtype))


# tf.function required to access Grappler's implementation_selector.
@implementation_selection.never_runs_functions_eagerly
# TODO(b/163029794): Shape relaxation breaks XLA.
@tf.function(autograph=False)
def _random_binomial(
    shape,
    counts,
    probs,
    output_dtype=tf.float32,
    seed=None,
    name=None):
  """Sample a binomial, CPU specialized to stateless_binomial.

  Args:
    shape: Shape of the full sample output. Trailing dims should match the
      broadcast shape of `counts` with `probs|logits`.
    counts: Batch of total_count.
    probs: Batch of p(success).
    output_dtype: DType of samples.
    seed: int or Tensor seed.
    name: Optional name for related ops.

  Returns:
    samples: Samples from binomial distributions.
    runtime_used_for_sampling: One of `implementation_selection._RUNTIME_*`.
  """
  with tf.name_scope(name or 'random_binomial'):
    seed = samplers.sanitize_seed(seed)
    shape = ps.convert_to_shape_tensor(shape, dtype_hint=tf.int32, name='shape')
    params = dict(shape=shape, counts=counts, probs=probs,
                  output_dtype=output_dtype, seed=seed, name=name)
    sampler_impl = implementation_selection.implementation_selecting(
        fn_name='binomial',
        default_fn=_random_binomial_noncpu,
        cpu_fn=_random_binomial_cpu)
    return sampler_impl(**params)


class Binomial(distribution.Distribution):
  """Binomial distribution.

  This distribution is parameterized by `probs`, a (batch of) probabilities for
  drawing a `1`, and `total_count`, the number of trials per draw from the
  Binomial.

  #### Mathematical Details

  The Binomial is a distribution over the number of `1`'s in `total_count`
  independent trials, with each trial having the same probability of `1`, i.e.,
  `probs`.

  The probability mass function (pmf) is,

  ```none
  pmf(k; n, p) = p**k (1 - p)**(n - k) / Z
  Z = k! (n - k)! / n!
  ```

  where:
  * `total_count = n`,
  * `probs = p`,
  * `Z` is the normalizing constant, and,
  * `n!` is the factorial of `n`.

  #### Examples

  Create a single distribution, corresponding to 5 coin flips.

  ```python
  dist = Binomial(total_count=5., probs=.5)
  ```

  Create a single distribution (using logits), corresponding to 5 coin flips.

  ```python
  dist = Binomial(total_count=5., logits=0.)
  ```

  Creates 3 distributions with the third distribution most likely to have
  successes.

  ```python
  p = [.2, .3, .8]
  # n will be broadcast to [4., 4., 4.], to match p.
  dist = Binomial(total_count=4., probs=p)
  ```

  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 2, 3]
  dist.prob(counts)  # Shape [3]

  # p will be broadcast to [[.2, .3, .8], [.2, .3, .8]] to match counts.
  counts = [[1., 2, 1], [2, 2, 4]]
  dist.prob(counts)  # Shape [2, 3]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7, 3]
  ```
  """

  def __init__(self,
               total_count,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Initialize a batch of Binomial distributions.

    Args:
      total_count: Non-negative floating point tensor with shape broadcastable
        to `[N1,..., Nm]` with `m >= 0` and the same dtype as `probs` or
        `logits`. Defines this as a batch of `N1 x ...  x Nm` different Binomial
        distributions. Its components should be equal to integer values.
      logits: Floating point tensor representing the log-odds of a
        positive event with shape broadcastable to `[N1,..., Nm]` `m >= 0`, and
        the same dtype as `total_count`. Each entry represents logits for the
        probability of success for independent Binomial distributions. Only one
        of `logits` or `probs` should be passed in.
      probs: Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm]` `m >= 0`, `probs in [0, 1]`. Each entry represents the
        probability of success for independent Binomial distributions. Only one
        of `logits` or `probs` should be passed in.
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
          'Construct `Binomial` with `probs` or `logits`, but not both.')
    with tf.name_scope(name or 'Binomial') as name:
      dtype = dtype_util.common_dtype([total_count, logits, probs], tf.float32)
      self._total_count = tensor_util.convert_nonref_to_tensor(
          total_count, dtype=dtype, name='total_count')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype=dtype, name='logits')
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype=dtype, name='probs')
      super(Binomial, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        total_count=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED),
        logits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))

  @property
  def total_count(self):
    """Number of trials."""
    return self._total_count

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _batch_shape_tensor(self):
    x = self._probs if self._logits is None else self._logits
    return ps.broadcast_shape(
        ps.shape(self._total_count), ps.shape(x))

  def _batch_shape(self):
    x = self._probs if self._logits is None else self._logits
    return tf.broadcast_static_shape(self.total_count.shape, x.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _log_prob(self, counts):
    total_count = tf.convert_to_tensor(self.total_count)
    if self._logits is not None:
      unnorm = _log_unnormalized_prob_logits(self._logits, counts, total_count)
    else:
      unnorm = _log_unnormalized_prob_probs(self._probs, counts, total_count)
    norm = _log_normalization(counts, total_count)
    return unnorm - norm

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _prob(self, counts):
    return tf.exp(self._log_prob(counts))

  def _cdf(self, counts):
    total_count = tf.convert_to_tensor(self.total_count)
    probs = self._probs_parameter_no_checks(total_count=total_count)
    probs, counts = _maybe_broadcast(probs, counts)

    return _bdtr(k=counts, n=total_count, p=probs)

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _sample_n(self, n, seed=None):
    seed = samplers.sanitize_seed(seed, salt='binomial')
    total_count = tf.convert_to_tensor(self._total_count)
    if self._probs is None:
      probs = self._probs_parameter_no_checks(total_count=total_count)
    else:
      probs = tf.convert_to_tensor(self._probs)

    return _random_binomial(
        shape=ps.convert_to_shape_tensor([n]),
        counts=total_count,
        probs=probs,
        output_dtype=self.dtype,
        seed=seed)[0]

  def _mean(self, probs=None, total_count=None):
    if total_count is None:
      total_count = tf.convert_to_tensor(self._total_count)
    if probs is None:
      probs = self._probs_parameter_no_checks(total_count=total_count)
    return total_count * probs

  def _variance(self):
    total_count = tf.convert_to_tensor(self._total_count)
    probs = self._probs_parameter_no_checks(total_count=total_count)
    return self._mean(probs=probs, total_count=total_count) * (1. - probs)

  @distribution_util.AppendDocstring(
      """Note that when `(1 + total_count) * probs` is an integer, there are
      actually two modes. Namely, `(1 + total_count) * probs` and
      `(1 + total_count) * probs - 1` are both modes. Here we return only the
      larger of the two modes.""")
  def _mode(self):
    total_count = tf.convert_to_tensor(self._total_count)
    probs = self._probs_parameter_no_checks(total_count=total_count)
    return tf.math.minimum(
        total_count, tf.floor((1. + total_count) * probs))

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log(probs) - tf.math.log1p(-probs)
    return tf.identity(self._logits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self, total_count=None):
    if self._logits is None:
      probs = tf.identity(self._probs)
    else:
      probs = tf.math.sigmoid(self._logits)
    # Suppress potentially nasty probs like `nan` b/c they don't matter where
    # total_count == 0.
    if total_count is None:
      total_count = self.total_count
    return tf.where(total_count > 0, probs, 0)

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    assertions = []

    if is_init != tensor_util.is_ref(self.total_count):
      total_count = tf.convert_to_tensor(self.total_count)
      msg1 = 'Argument `total_count` must be non-negative.'
      msg2 = 'Argument `total_count` cannot contain fractional components.'
      assertions += [
          assert_util.assert_non_negative(total_count, message=msg1),
          distribution_util.assert_integer_form(total_count, message=msg2),
      ]

    if self._probs is not None:
      if is_init != tensor_util.is_ref(self._probs):
        probs = tf.convert_to_tensor(self._probs)
        one = tf.constant(1., probs.dtype)
        assertions += [
            assert_util.assert_non_negative(
                probs, message='probs has components less than 0.'),
            assert_util.assert_less_equal(
                probs, one, message='probs has components greater than 1.')
        ]

    return assertions

  def _sample_control_dependencies(self, counts):
    """Check counts for proper values."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(distribution_util.assert_casting_closed(
        counts, target_dtype=tf.int32,
        message='counts cannot contain fractional components.'))
    assertions.append(assert_util.assert_non_negative(
        counts, message='counts must be non-negative.'))
    assertions.append(
        assert_util.assert_less_equal(
            counts, self.total_count,
            message=('Sampled counts must be itemwise less than '
                     'or equal to `total_count` parameter.')))
    return assertions


def _log_unnormalized_prob_logits(logits, counts, total_count):
  """Log unnormalized probability from logits."""
  logits = tf.convert_to_tensor(logits)
  # softplus(x) = log(1 + exp(x))
  # sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (exp(x) + 1)
  # log(probs) = log(sigmoid(logits))
  #            = -log(1 + exp(-logits))
  #            = -softplus(-logits)
  # log1p(-probs) = log(1 - sigmoid(logits))
  #               = log(1 - 1 / (1 + exp(-logits)))
  #               = log((1 + exp(-logits) - 1) / (1 + exp(-logits)))
  #               = log(exp(-logits) / (1 + exp(-logits)))
  #               = log(sigmoid(-logits))
  #               = -softplus(logits)  # by log(sigmoid(x)) = -softplus(-x))
  return (-tf.math.multiply_no_nan(tf.math.softplus(-logits), counts) -
          tf.math.multiply_no_nan(
              tf.math.softplus(logits), total_count - counts))


def _log_unnormalized_prob_probs(probs, counts, total_count):
  """Log unnormalized probability from probs."""
  probs = tf.convert_to_tensor(probs)
  return (tf.math.multiply_no_nan(tf.math.log(probs), counts) +
          tf.math.multiply_no_nan(tf.math.log1p(-probs), total_count - counts))


def _log_normalization(counts, total_count):
  return (tfp_math.lbeta(1. + counts, 1. + total_count - counts) +
          tf.math.log(1. + total_count))


def _maybe_broadcast(a, b):
  if not (tensorshape_util.is_fully_defined(a.shape) and
          tensorshape_util.is_fully_defined(b.shape) and
          tensorshape_util.is_compatible_with(a.shape, b.shape)):
    # If both shapes are well defined and equal, we skip broadcasting.
    b = b + tf.zeros_like(a)
    a = a + tf.zeros_like(b)
  return a, b
