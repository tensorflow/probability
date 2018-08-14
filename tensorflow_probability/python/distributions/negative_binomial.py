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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python.distributions import seed_stream
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.distributions import util as distribution_util


class NegativeBinomial(tf.distributions.Distribution):
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
               name="NegativeBinomial"):
    """Construct NegativeBinomial distributions.

    Args:
      total_count: Non-negative floating-point `Tensor` with shape
        broadcastable to `[B1,..., Bb]` with `b >= 0` and the same dtype as
        `probs` or `logits`. Defines this as a batch of `N1 x ... x Nm`
        different Negative Binomial distributions. In practice, this represents
        the number of negative Bernoulli trials to stop at (the `total_count`
        of failures), but this is still a valid distribution when
        `total_count` is a non-integer.
      logits: Floating-point `Tensor` with shape broadcastable to
        `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
        Each entry represents logits for the probability of success for
        independent Negative Binomial distributions and must be in the open
        interval `(-inf, inf)`. Only one of `logits` or `probs` should be
        specified.
      probs: Positive floating-point `Tensor` with shape broadcastable to
        `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
        Each entry represents the probability of success for independent
        Negative Binomial distributions and must be in the open interval
        `(0, 1)`. Only one of `logits` or `probs` should be specified.
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
    with tf.name_scope(name, values=[total_count, logits, probs]) as name:
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits, probs, validate_args=validate_args, name=name)
      with tf.control_dependencies([tf.assert_positive(total_count)]
                                   if validate_args else []):
        self._total_count = tf.identity(total_count)

    super(NegativeBinomial, self).__init__(
        dtype=self._probs.dtype,
        reparameterization_type=tf.distributions.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._total_count, self._probs, self._logits],
        name=name)

  @property
  def total_count(self):
    """Number of negative trials."""
    return self._total_count

  @property
  def logits(self):
    """Log-odds of a `1` outcome (vs `0`)."""
    return self._logits

  @property
  def probs(self):
    """Probability of a `1` outcome (vs `0`)."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.total_count), tf.shape(self.probs))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.total_count.get_shape(),
                                     self.probs.get_shape())

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # Here we use the fact that if:
    # lam ~ Gamma(concentration=total_count, rate=(1-probs)/probs)
    # then X ~ Poisson(lam) is Negative Binomially distributed.
    stream = seed_stream.SeedStream(seed, salt="NegativeBinomial")
    rate = tf.random_gamma(
        shape=[n],
        alpha=self.total_count,
        beta=tf.exp(-self.logits),
        dtype=self.dtype,
        seed=stream())
    return tf.random_poisson(
        rate,
        shape=[],
        dtype=self.dtype,
        seed=stream())

  def _cdf(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    return tf.betainc(self.total_count, 1. + x, tf.sigmoid(-self.logits))

  def _log_prob(self, x):
    return (self._log_unnormalized_prob(x)
            - self._log_normalization(x))

  def _log_unnormalized_prob(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    return (self.total_count * tf.log_sigmoid(-self.logits) +
            x * tf.log_sigmoid(self.logits))

  def _log_normalization(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    return (-tf.lgamma(self.total_count + x) + tf.lgamma(1. + x) + tf.lgamma(
        self.total_count))

  def _mean(self):
    return self.total_count * tf.exp(self.logits)

  def _mode(self):
    adjusted_count = tf.where(1. < self.total_count, self.total_count - 1.,
                              tf.zeros_like(self.total_count))
    return tf.floor(adjusted_count * tf.exp(self.logits))

  def _variance(self):
    return self._mean() / tf.sigmoid(-self.logits)
