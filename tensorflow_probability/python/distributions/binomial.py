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

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
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
  k_eq_n = tf.equal(k, n)
  safe_dn = tf.where(k_eq_n, ones, n - k)
  dk = tf.math.betainc(a=safe_dn, b=k + 1, x=1 - p)
  return tf.where(k_eq_n, ones, dk)


class Binomial(distribution.Distribution):
  """Binomial distribution.

  This distribution is parameterized by `probs`, a (batch of) probabilities for
  drawing a `1` and `total_count`, the number of trials per draw from the
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
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([total_count, logits, probs], tf.float32)
      self._total_count = self._maybe_assert_valid_total_count(
          tf.convert_to_tensor(total_count, name='total_count', dtype=dtype),
          validate_args)
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          name=name,
          dtype=dtype)
    super(Binomial, self).__init__(
        dtype=self._probs.dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._total_count, self._logits, self._probs],
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
    return tf.broadcast_dynamic_shape(
        tf.shape(self.total_count), tf.shape(self.probs))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.total_count.shape,
                                     self.probs.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _log_prob(self, counts):
    return self._log_unnormalized_prob(counts) - self._log_normalization(counts)

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _prob(self, counts):
    return tf.exp(self._log_prob(counts))

  def _cdf(self, counts):
    counts = self._maybe_assert_valid_sample(counts)
    probs = self.probs
    if not (tensorshape_util.is_fully_defined(counts.shape) and
            tensorshape_util.is_fully_defined(self.probs.shape) and
            tensorshape_util.is_compatible_with(counts.shape,
                                                self.probs.shape)):
      # If both shapes are well defined and equal, we skip broadcasting.
      probs += tf.zeros_like(counts)
      counts += tf.zeros_like(self.probs)

    return _bdtr(k=counts, n=self.total_count, p=probs)

  def _log_unnormalized_prob(self, counts):
    counts = self._maybe_assert_valid_sample(counts)
    return counts * self.logits - self.total_count * tf.math.softplus(
        self.logits)

  def _log_normalization(self, counts):
    counts = self._maybe_assert_valid_sample(counts)
    return (tf.math.lgamma(1. + self.total_count - counts) +
            tf.math.lgamma(1. + counts) - tf.math.lgamma(1. + self.total_count))

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _sample_n(self, n, seed=None):
    # Need to create logits corresponding to [p, 1 - p].
    # Note that for this distributions, logits corresponds to
    # inverse sigmoid(p) while in multivariate distributions,
    # such as multinomial this corresponds to log(p).
    # Because of this, when we construct the logits for the multinomial
    # sampler, we'll have to be careful.
    # log(p) = log(sigmoid(logits)) = logits - softplus(logits)
    # log(1 - p) = log(1 - sigmoid(logits)) = -softplus(logits)
    # Because softmax is invariant to a constnat shift in all inputs,
    # we can offset the logits by softplus(logits) so that we can use
    # [logits, 0.] as our input.
    logits = tf.stack(
        [self.logits,
         tf.zeros_like(self.logits)],
        axis=-1)
    return multinomial.draw_sample(
        num_samples=n,
        num_classes=2,
        logits=logits,
        num_trials=tf.cast(self.total_count, dtype=tf.int32),
        dtype=self.dtype,
        seed=seed)[..., 0]

  def _mean(self):
    return self.total_count * self.probs

  def _variance(self):
    return self._mean() * (1. - self.probs)

  @distribution_util.AppendDocstring(
      """Note that when `(1 + total_count) * probs` is an integer, there are
      actually two modes. Namely, `(1 + total_count) * probs` and
      `(1 + total_count) * probs - 1` are both modes. Here we return only the
      larger of the two modes.""")
  def _mode(self):
    return tf.floor((1. + self.total_count) * self.probs)

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      if self.logits is None:
        return tf.math.log(self.probs) - tf.math.log1p(-self.probs)
      return tf.identity(self.logits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      if self.logits is None:
        return tf.identity(self.probs)
      return tf.math.sigmoid(self.logits)

  def _maybe_assert_valid_total_count(self, total_count, validate_args):
    if not validate_args:
      return total_count
    return distribution_util.with_dependencies([
        assert_util.assert_non_negative(
            total_count, message='total_count must be non-negative.'),
        distribution_util.assert_integer_form(
            total_count,
            message='total_count cannot contain fractional components.'),
    ], total_count)

  def _maybe_assert_valid_sample(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    if not self.validate_args:
      return counts
    counts = distribution_util.embed_check_nonnegative_integer_form(counts)
    return distribution_util.with_dependencies([
        assert_util.assert_less_equal(
            counts,
            self.total_count,
            message='counts are not less than or equal to n.'),
    ], counts)
