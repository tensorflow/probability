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
"""The Multinomial distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    "Multinomial",
]


_multinomial_sample_note = """For each batch of counts, `value = [n_0, ...
,n_{k-1}]`, `P[value]` is the probability that after sampling `self.total_count`
draws from this Multinomial distribution, the number of draws falling in class
`j` is `n_j`. Since this definition is [exchangeable](
https://en.wikipedia.org/wiki/Exchangeable_random_variables); different
sequences have the same counts so the probability includes a combinatorial
coefficient.

Note: `value` must be a non-negative tensor with dtype `self.dtype`, have no
fractional components, and such that
`tf.reduce_sum(value, -1) = self.total_count`. Its shape must be broadcastable
with `self.probs` and `self.total_count`."""


class Multinomial(distribution.Distribution):
  """Multinomial distribution.

  This Multinomial distribution is parameterized by `probs`, a (batch of)
  length-`K` `prob` (probability) vectors (`K > 1`) such that
  `tf.reduce_sum(probs, -1) = 1`, and a `total_count` number of trials, i.e.,
  the number of trials per draw from the Multinomial. It is defined over a
  (batch of) length-`K` vector `counts` such that
  `tf.reduce_sum(counts, -1) = total_count`. The Multinomial is identically the
  Binomial distribution when `K = 2`.

  #### Mathematical Details

  The Multinomial is a distribution over `K`-class counts, i.e., a length-`K`
  vector of non-negative integer `counts = n = [n_0, ..., n_{K-1}]`.

  The probability mass function (pmf) is,

  ```none
  pmf(n; pi, N) = prod_j (pi_j)**n_j / Z
  Z = (prod_j n_j!) / N!
  ```

  where:
  * `probs = pi = [pi_0, ..., pi_{K-1}]`, `pi_j > 0`, `sum_j pi_j = 1`,
  * `total_count = N`, `N` a positive integer,
  * `Z` is the normalization constant, and,
  * `N!` denotes `N` factorial.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Pitfalls

  The number of classes, `K`, must not exceed:
  - the largest integer representable by `self.dtype`, i.e.,
    `2**(mantissa_bits+1)` (IEE754),
  - the maximum `Tensor` index, i.e., `2**31-1`.

  In other words,

  ```python
  K <= min(2**31-1, {
    tf.float16: 2**11,
    tf.float32: 2**24,
    tf.float64: 2**53 }[param.dtype])
  ```

  Note: This condition is validated only when `self.validate_args = True`.

  #### Examples

  Create a 3-class distribution, with the 3rd class is most likely to be drawn,
  using logits.

  ```python
  logits = [-50., -43, 0]
  dist = Multinomial(total_count=4., logits=logits)
  ```

  Create a 3-class distribution, with the 3rd class is most likely to be drawn.

  ```python
  p = [.2, .3, .5]
  dist = Multinomial(total_count=4., probs=p)
  ```

  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 0, 3]
  dist.prob(counts)  # Shape []

  # p will be broadcast to [[.2, .3, .5], [.2, .3, .5]] to match counts.
  counts = [[1., 2, 1], [2, 2, 0]]
  dist.prob(counts)  # Shape [2]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7]
  ```

  Create a 2-batch of 3-class distributions.

  ```python
  p = [[.1, .2, .7], [.3, .3, .4]]  # Shape [2, 3]
  dist = Multinomial(total_count=[4., 5], probs=p)

  counts = [[2., 1, 1], [3, 1, 1]]
  dist.prob(counts)  # Shape [2]

  dist.sample(5) # Shape [5, 2, 3]
  ```
  """

  def __init__(self,
               total_count,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Multinomial"):
    """Initialize a batch of Multinomial distributions.

    Args:
      total_count: Non-negative floating point tensor with shape broadcastable
        to `[N1,..., Nm]` with `m >= 0`. Defines this as a batch of
        `N1 x ... x Nm` different Multinomial distributions. Its components
        should be equal to integer values.
      logits: Floating point tensor representing unnormalized log-probabilities
        of a positive event with shape broadcastable to
        `[N1,..., Nm, K]` `m >= 0`, and the same dtype as `total_count`. Defines
        this as a batch of `N1 x ... x Nm` different `K` class Multinomial
        distributions. Only one of `logits` or `probs` should be passed in.
      probs: Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm, K]` `m >= 0` and same dtype as `total_count`. Defines
        this as a batch of `N1 x ... x Nm` different `K` class Multinomial
        distributions. `probs`'s components in the last portion of its shape
        should sum to `1`. Only one of `logits` or `probs` should be passed in.
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
      self._total_count = tf.convert_to_tensor(
          value=total_count, name="total_count", dtype=dtype)
      if validate_args:
        self._total_count = (
            distribution_util.embed_check_nonnegative_integer_form(
                self._total_count))
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          multidimensional=True,
          validate_args=validate_args,
          name=name,
          dtype=dtype)
      self._mean_val = self._total_count[..., tf.newaxis] * self._probs
    super(Multinomial, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._total_count, self._logits, self._probs],
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(total_count=0, logits=1, probs=1)

  @property
  def total_count(self):
    """Number of trials used to construct a sample."""
    return self._total_count

  @property
  def logits(self):
    """Vector of coordinatewise logits."""
    return self._logits

  @property
  def probs(self):
    """Probability of drawing a `1` in that coordinate."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.shape(input=self._mean_val)[:-1]

  def _batch_shape(self):
    return tensorshape_util.with_rank_at_least(self._mean_val.shape, 1)[:-1]

  def _event_shape_tensor(self):
    return tf.shape(input=self._mean_val)[-1:]

  def _event_shape(self):
    return tensorshape_util.with_rank_at_least(self._mean_val.shape, 1)[-1:]

  def _sample_n(self, n, seed=None):
    n_draws = tf.cast(self.total_count, dtype=tf.int32)
    k = self.event_shape_tensor()[0]

    return draw_sample(n, k, self.logits, n_draws, self.dtype, seed)

  @distribution_util.AppendDocstring(_multinomial_sample_note)
  def _log_prob(self, counts):
    return self._log_unnormalized_prob(counts) - self._log_normalization(counts)

  def _log_unnormalized_prob(self, counts):
    counts = self._maybe_assert_valid_sample(counts)
    return tf.reduce_sum(
        input_tensor=counts * tf.nn.log_softmax(self.logits), axis=-1)

  def _log_normalization(self, counts):
    counts = self._maybe_assert_valid_sample(counts)
    return -distribution_util.log_combinations(self.total_count, counts)

  def _mean(self):
    return tf.identity(self._mean_val)

  def _covariance(self):
    p = self.probs * tf.ones_like(
        self.total_count)[..., tf.newaxis]
    return tf.linalg.set_diag(
        -tf.matmul(self._mean_val[..., tf.newaxis],
                   p[..., tf.newaxis, :]),  # outer product
        self._variance())

  def _variance(self):
    p = self.probs * tf.ones_like(
        self.total_count)[..., tf.newaxis]
    return self._mean_val - self._mean_val * p

  def _maybe_assert_valid_sample(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    if not self.validate_args:
      return counts
    counts = distribution_util.embed_check_nonnegative_integer_form(counts)
    return distribution_util.with_dependencies([
        assert_util.assert_equal(
            self.total_count,
            tf.reduce_sum(input_tensor=counts, axis=-1),
            message="counts must sum to `self.total_count`"),
    ], counts)


def draw_sample(num_samples, num_classes, logits, num_trials, dtype, seed):
  """Sample a multinomial.

  The batch shape is given by broadcasting num_trials with
  remove_last_dimension(logits).

  Args:
    num_samples: Python int or singleton integer Tensor: number of multinomial
      samples to draw.
    num_classes: Python int or singleton integer Tensor: number of classes.
    logits: Floating Tensor with last dimension k, of (unnormalized) logit
      probabilities per class.
    num_trials: Tensor of number of categorical trials each multinomial consists
      of.  num_trials[..., tf.newaxis] must broadcast with logits.
    dtype: dtype at which to emit samples.
    seed: Random seed.

  Returns:
    samples: Tensor of given dtype and shape [n] + batch_shape + [k].
  """
  with tf.name_scope("multinomial.draw_sample"):
    # broadcast the num_trials and logits to same shape
    num_trials = tf.ones_like(
        logits[..., 0], dtype=num_trials.dtype) * num_trials
    logits = tf.ones_like(
        num_trials[..., tf.newaxis], dtype=logits.dtype) * logits

    # flatten the total_count and logits
    # flat_logits has shape [B1B2...Bm, num_classes]
    flat_logits = tf.reshape(logits, [-1, num_classes])
    flat_num_trials = num_samples * tf.reshape(num_trials, [-1])  # [B1B2...Bm]

    # Computes each logits and num_trials situation by map_fn.

    # Using just one batch tf.random.categorical call doesn't work because that
    # requires num_trials to be the same across all members of the batch of
    # logits.  This restriction makes sense for tf.random.categorical because
    # for it, num_trials is part of the returned shape.  However, the
    # multinomial sampler does not need that restriction, because it sums out
    # exactly that dimension.

    # One possibility would be to draw a batch categorical whose sample count is
    # max(num_trials) and mask out the excess ones.  However, if the elements of
    # num_trials vary widely, this can be wasteful of memory.

    # TODO(b/123763054, b/112152209): Revisit the possibility of writing this
    # with a batch categorical followed by batch unsorted_segment_sum, once both
    # of those work and are memory-efficient enough.
    def _sample_one_batch_member(args):
      logits, num_cat_samples = args[0], args[1]  # [K], []
      # x has shape [1, num_cat_samples = num_samples * num_trials]
      x = tf.random.categorical(
          logits[tf.newaxis, ...], num_cat_samples, seed=seed)
      x = tf.reshape(x, shape=[num_samples, -1])  # [num_samples, num_trials]
      x = tf.one_hot(
          x, depth=num_classes)  # [num_samples, num_trials, num_classes]
      x = tf.reduce_sum(input_tensor=x, axis=-2)  # [num_samples, num_classes]
      return tf.cast(x, dtype=dtype)

    x = tf.map_fn(
        _sample_one_batch_member, [flat_logits, flat_num_trials],
        dtype=dtype)  # [B1B2...Bm, num_samples, num_classes]

    # reshape the results to proper shape
    x = tf.transpose(a=x, perm=[1, 0, 2])
    final_shape = tf.concat([[num_samples],
                             tf.shape(input=num_trials), [num_classes]],
                            axis=0)
    x = tf.reshape(x, final_shape)

    return x
