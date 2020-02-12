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
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import batched_rejection_sampler
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.random_ops import random_rademacher
from tensorflow_probability.python.util.seed_stream import SeedStream


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
  k_eq_n = tf.equal(k, n)
  safe_dn = tf.where(k_eq_n, ones, n - k)
  dk = tf.math.betainc(a=safe_dn, b=k + 1, x=1 - p)
  return tf.where(k_eq_n, ones, dk)


def _log_concave_rejection_sampler(
    log_concave_distribution,
    sample_shape=(),
    distribution_minimum=None,
    distribution_maximum=None,
    seed=None):
  """Utility for rejection sampling from log-concave discrete distributions.

  This utility constructs an easy-to-sample-from upper bound for a discrete
  univariate log-concave distribution (for discrete univariate distributions, a
  necessary and sufficient condition is p_k^2 >= p_{k-1} p_{k+1} for all k).
  The method requires that the mode of the distribution is known. While a better
  method can likely be derived for any given distribution, this method is
  general and easy to implement. The expected number of iterations is bounded by
  4+m, where m is the probability of the mode. For details, see [(Devroye,
  1979)][1].

  Args:
   log_concave_distribution: An object satisfying TensorFlow Probability shape
      semantics [(Dillon et al., 2017)][2] and having `mode` and `prob` methods.
      The caller is responsible for ensuring that discreteness and log-concavity
      hold.
    sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
    distribution_minimum: Tensor of type `distribution.dtype`. The minimum value
      taken by the distribution. The `prob` method will only be called on values
      greater than equal to the specified minimum. The shape must broadcast with
      the batch shape of the distribution. If unspecified, the domain is treated
      as unbounded below.
    distribution_maximum: Tensor of type `distribution.dtype`. The maximum value
      taken by the distribution. See `distribution_minimum` for details.
    seed: Python integer or `tfp.util.SeedStream` instance, for seeding PRNG.

  Returns:
    samples: a `Tensor` with prepended dimensions `sample_shape`.

  #### References

  [1] Luc Devroye. A Simple Generator for Discrete Log-Concave
      Distributions. Computing, 1987.

  [2] Dillon et al. TensorFlow Distributions. 2017.
      https://arxiv.org/abs/1711.10604
  """
  dtype = log_concave_distribution.dtype

  mode = log_concave_distribution.mode()
  mode = tf.broadcast_to(
      mode,
      tf.concat((sample_shape, prefer_static.shape(mode)), axis=0))

  mode_height = log_concave_distribution.prob(mode)
  mode_shape = prefer_static.shape(mode)

  top_width = 1. + mode_height / 2.  # w in ref [1].
  top_fraction = top_width / (1 + top_width)
  exponential_distribution = exponential.Exponential(
      rate=tf.constant(1., dtype=dtype))  # E in ref [1].

  if distribution_minimum is None:
    distribution_minimum = tf.constant(-np.inf, dtype)
  if distribution_maximum is None:
    distribution_maximum = tf.constant(np.inf, dtype)

  def proposal(seed):
    """Proposal for log-concave rejection sampler."""
    seed_stream = SeedStream(seed, 'log_concave_rejection_sampler_proposal')

    top_lobe_fractions = tf.random.uniform(
        mode_shape, seed=seed_stream(), dtype=dtype)  # V in ref [1].
    top_offsets = top_lobe_fractions * top_width / mode_height

    exponential_samples = exponential_distribution.sample(
        mode_shape, seed=seed_stream())  # E in ref [1].
    exponential_height = (exponential_distribution.prob(exponential_samples) *
                          mode_height)
    exponential_offsets = (top_width + exponential_samples) / mode_height

    top_selector = tf.random.uniform(
        mode_shape, seed=seed_stream(), dtype=dtype)  # U in ref [1].
    on_top_mask = tf.less_equal(top_selector, top_fraction)

    unsigned_offsets = tf.where(on_top_mask, top_offsets, exponential_offsets)
    offsets = tf.round(
        random_rademacher(mode_shape, seed=seed_stream(), dtype=dtype) *
        unsigned_offsets)

    potential_samples = mode + offsets
    envelope_height = tf.where(on_top_mask, mode_height, exponential_height)

    return potential_samples, envelope_height

  def target(values):
    in_range_mask = (
        (values >= distribution_minimum) & (values <= distribution_maximum))
    in_range_values = tf.where(in_range_mask, values, 0.)
    return tf.where(in_range_mask,
                    log_concave_distribution.prob(in_range_values), 0.)

  return batched_rejection_sampler.batched_rejection_sampler(
      proposal, target, seed, dtype=dtype)[0]  # Discard `num_iters`.


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
               name='Binomial'):
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
    with tf.name_scope(name) as name:
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
  def _params_event_ndims(cls):
    return dict(total_count=0, logits=0, probs=0)

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
    return tf.broadcast_dynamic_shape(
        tf.shape(self._total_count), tf.shape(x))

  def _batch_shape(self):
    x = self._probs if self._logits is None else self._logits
    return tf.broadcast_static_shape(self.total_count.shape, x.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _log_prob(self, counts):
    logits = self._logits_parameter_no_checks()
    total_count = tf.convert_to_tensor(self.total_count)
    unnorm = _log_unnormalized_prob(logits, counts, total_count)
    norm = _log_normalization(counts, total_count)
    return unnorm - norm

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _prob(self, counts):
    return tf.exp(self._log_prob(counts))

  def _cdf(self, counts):
    probs = self._probs_parameter_no_checks()
    probs, counts = _maybe_broadcast(probs, counts)

    return _bdtr(k=counts, n=tf.convert_to_tensor(self.total_count), p=probs)

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _sample_n(self, n, seed=None):
    return _log_concave_rejection_sampler(
        self, [n], tf.zeros((), dtype=self.dtype), self.total_count, seed)

  def _mean(self, probs=None):
    if probs is None:
      probs = self._probs_parameter_no_checks()
    return self._total_count * probs

  def _variance(self):
    probs = self._probs_parameter_no_checks()
    return self._mean(probs) * (1. - probs)

  @distribution_util.AppendDocstring(
      """Note that when `(1 + total_count) * probs` is an integer, there are
      actually two modes. Namely, `(1 + total_count) * probs` and
      `(1 + total_count) * probs - 1` are both modes. Here we return only the
      larger of the two modes.""")
  def _mode(self):
    total_count = tf.convert_to_tensor(self._total_count)
    return tf.math.minimum(
        total_count,
        tf.floor(
            (1. + total_count) * self._probs_parameter_no_checks()))

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

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tf.identity(self._probs)
    return tf.math.sigmoid(self._logits)

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
    assertions.extend(distribution_util.assert_nonnegative_integer_form(counts))
    assertions.append(
        assert_util.assert_less_equal(
            counts, self.total_count,
            message=('Sampled counts must be itemwise less than '
                     'or equal to `total_count` parameter.')))
    return assertions


def _log_unnormalized_prob(logits, counts, total_count):
  """Log unnormalized probability."""
  return (-tf.math.multiply_no_nan(tf.math.softplus(-logits), counts) -
          tf.math.multiply_no_nan(
              tf.math.softplus(logits), total_count - counts))


def _log_normalization(counts, total_count):
  return (tf.math.lgamma(1. + total_count - counts) +
          tf.math.lgamma(1. + counts) - tf.math.lgamma(1. + total_count))


def _maybe_broadcast(a, b):
  if not (tensorshape_util.is_fully_defined(a.shape) and
          tensorshape_util.is_fully_defined(b.shape) and
          tensorshape_util.is_compatible_with(a.shape, b.shape)):
    # If both shapes are well defined and equal, we skip broadcasting.
    b = b + tf.zeros_like(a)
    a = a + tf.zeros_like(b)
  return a, b
