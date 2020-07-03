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

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.distributions import binomial
from tensorflow_probability.python.distributions import categorical as categorical_lib
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'Multinomial',
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
               name='Multinomial'):
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

    if (probs is None) == (logits is None):
      raise ValueError('Must pass probs or logits, but not both.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([total_count, logits, probs],
                                      dtype_hint=tf.float32)
      self._total_count = tensor_util.convert_nonref_to_tensor(
          total_count, name='total_count', dtype=dtype)
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype=dtype, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype=dtype, name='logits')
      super(Multinomial, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
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
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(self._probs if self._logits is None else self._logits)[:-1],
        tf.shape(self.total_count))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        (self._probs if self._logits is None else self._logits).shape[:-1],
        self.total_count.shape)

  def _event_shape_tensor(self):
    # We will never broadcast the num_categories with total_count.
    return tf.shape(self._probs if self._logits is None else self._logits)[-1:]

  def _event_shape(self):
    # We will never broadcast the num_categories with total_count.
    return tensorshape_util.with_rank(
        (self._probs if self._logits is None else self._logits).shape[-1:],
        rank=1)

  def _sample_n(self, n, seed=None):
    n_draws = tf.cast(self.total_count, dtype=tf.int32)
    probs = self._probs_parameter_no_checks()
    k = tf.compat.dimension_value(probs.shape[-1])
    if k is None:
      k = tf.shape(probs)[-1]
    return _sample_multinomial_as_iterated_binomial(
        n, k, probs, n_draws, self.dtype, seed)

  @distribution_util.AppendDocstring(_multinomial_sample_note)
  def _log_prob(self, counts):
    log_p = (
        tf.math.log(self._probs)
        if self._logits is None else tf.math.log_softmax(self._logits))
    log_unnorm_prob = tf.reduce_sum(
        tf.math.multiply_no_nan(log_p, counts), axis=-1)
    neg_log_normalizer = tfp_math.log_combinations(self.total_count, counts)
    return log_unnorm_prob + neg_log_normalizer

  def _mean(self):
    p = self._probs_parameter_no_checks()
    k = tf.convert_to_tensor(self.total_count)
    return k[..., tf.newaxis] * p

  def _covariance(self):
    p = self._probs_parameter_no_checks()
    k = tf.convert_to_tensor(self.total_count)
    return tf.linalg.set_diag(
        (-k[..., tf.newaxis, tf.newaxis] *
         (p[..., :, tf.newaxis] * p[..., tf.newaxis, :])),  # Outer product.
        k[..., tf.newaxis] * p * (1. - p))

  def _variance(self):
    p = self._probs_parameter_no_checks()
    k = tf.convert_to_tensor(self.total_count)
    return k[..., tf.newaxis] * p * (1. - p)

  def logits_parameter(self, name=None):
    """Logits vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      return tf.math.log(self._probs)
    return tf.identity(self._logits)

  def probs_parameter(self, name=None):
    """Probs vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tf.identity(self._probs)
    return tf.math.softmax(self._logits)

  def _default_event_space_bijector(self):
    return

  def _sample_control_dependencies(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(counts))
    assertions.append(assert_util.assert_equal(
        self.total_count,
        tf.reduce_sum(counts, axis=-1),
        message='counts must sum to `self.total_count`'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    assertions = categorical_lib.maybe_assert_categorical_param_correctness(
        is_init, self.validate_args, self._probs, self._logits)
    if not self.validate_args:
      return assertions
    if is_init != tensor_util.is_ref(self.total_count):
      assertions.extend(distribution_util.assert_nonnegative_integer_form(
          self.total_count))
    return assertions


def draw_sample(num_samples, num_classes, logits, num_trials, dtype, seed):
  """Sample a multinomial.

  The batch shape is given by broadcasting num_trials with
  remove_last_dimension(logits).

  Args:
    num_samples: Python int or singleton integer Tensor: number of multinomial
      samples to draw.
    num_classes: Python int or singleton integer Tensor: number of classes.
    logits: Floating Tensor with last dimension `num_classes`, of (unnormalized)
      logit probabilities per class.
    num_trials: Tensor of number of categorical trials each multinomial consists
      of.  num_trials[..., tf.newaxis] must broadcast with logits.
    dtype: dtype at which to emit samples.
    seed: Random seed.

  Returns:
    samples: Tensor of given dtype and shape [num_samples] + batch_shape +
      [num_classes].
  """
  probs = tf.math.softmax(logits)
  return _sample_multinomial_as_iterated_binomial(
      num_samples, num_classes, probs, num_trials, dtype, seed)


def _sample_multinomial_as_iterated_binomial(
    num_samples, num_classes, probs, num_trials, dtype, seed):
  """Sample a multinomial by drawing one binomial sample per class.

  The batch shape is given by broadcasting num_trials with
  remove_last_dimension(probs).

  The loop over binomial samples is a `tf.while_loop`, thus supporting a dynamic
  number of classes.

  Args:
    num_samples: Singleton integer Tensor: number of multinomial samples to
      draw.
    num_classes: Singleton integer Tensor: number of classes.
    probs: Floating Tensor with last dimension `num_classes`, of normalized
      probabilities per class.
    num_trials: Tensor of number of categorical trials each multinomial consists
      of.  num_trials[..., tf.newaxis] must broadcast with probs.
    dtype: dtype at which to emit samples.
    seed: Random seed.

  Returns:
    samples: Tensor of given dtype and shape [num_samples] + batch_shape +
      [num_classes].
  """
  with tf.name_scope('draw_sample'):
    # `convert_to_tensor(num_classes) here to avoid unstacking inside
    # `split_seed`.  We can't take advantage of the Python-list code path anyway
    # because the index at which we will take the seed is a Tensor.
    seeds = samplers.split_seed(
        seed, n=tf.convert_to_tensor(num_classes),
        salt='multinomial_draw_sample')

    def fn(i, num_trials, consumed_prob, accum):
      """Sample the counts for one class using binomial."""
      probs_here = tf.gather(probs, i, axis=-1)
      binomial_probs = tf.clip_by_value(probs_here / (1. - consumed_prob), 0, 1)
      seed_here = tf.gather(seeds, i, axis=0)
      binom = binomial.Binomial(total_count=num_trials, probs=binomial_probs)
      # Not passing `num_samples` to `binom.sample`, as it's is already in
      # `num_trials.shape`.
      sample = binom.sample(seed=seed_here)
      accum = accum.write(i, tf.cast(sample, dtype=dtype))
      return i + 1, num_trials - sample, consumed_prob + probs_here, accum

    num_trials = tf.cast(num_trials, probs.dtype)
    # Pre-broadcast with probs
    num_trials += tf.zeros_like(probs[..., 0])
    # Pre-enlarge for different output samples
    num_trials = _replicate_along_left(num_trials, num_samples)
    i = tf.constant(0)
    consumed_prob = tf.zeros_like(probs[..., 0])
    accum = tf.TensorArray(
        dtype, size=num_classes, element_shape=num_trials.shape)
    _, num_trials_left, _, accum = tf.while_loop(
        cond=lambda index, _0, _1, _2: tf.less(index, num_classes - 1),
        body=fn,
        loop_vars=(i, num_trials, consumed_prob, accum))
    # Force the last iteration to put all the trials into the last bucket,
    # because probs[..., -1] / (1. - consumed_prob) might numerically not be 1.
    # Also saves one iteration around the while_loop and one run of the binomial
    # sampler.
    accum = accum.write(num_classes - 1, tf.cast(num_trials_left, dtype=dtype))
    # This stop_gradient is necessary to prevent spurious zero gradients coming
    # from b/138796859, and a spurious gradient through num_trials_left.
    results = tf.stop_gradient(accum.stack())
    return distribution_util.move_dimension(results, 0, -1)


def _replicate_along_left(tensor, count):
  """Replicates `tensor` `count` times along a new leading axis.

  In other words, returns a Tensor of shape '[count] + tensor.shape`, with
  `count` copies of `tensor`.

  Args:
    tensor: Tensor to replicate.
    count: A scalar integer Tensor giving the number of times to replicate.

  Returns:
    result: Replicated Tensor.
  """
  # I am surpised that I can't seem to find this utility in the TF API.  tf.tile
  # is more general, but making it do this seems to require complicated shape
  # gymnastics.  tf.repeat repeats in place, hence this idiom:
  desired_shape = prefer_static.concat(
      [[count], prefer_static.shape(tensor)], axis=0)
  return tf.broadcast_to(tensor[tf.newaxis, ...], desired_shape)
