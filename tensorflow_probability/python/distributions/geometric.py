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
"""The Geometric distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops


class Geometric(distribution.Distribution):
  """Geometric distribution.

  The Geometric distribution is parameterized by p, the probability of a
  positive event. It represents the probability that in k + 1 Bernoulli trials,
  the first k trials failed, before seeing a success.

  The pmf of this distribution is:

  #### Mathematical Details

  ```none
  pmf(k; p) = (1 - p)**k * p
  ```

  where:

  * `p` is the success probability, `0 < p <= 1`, and,
  * `k` is a non-negative integer.

  """

  def __init__(self,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Geometric"):
    """Construct Geometric distributions.

    Args:
      logits: Floating-point `Tensor` with shape `[B1, ..., Bb]` where `b >= 0`
        indicates the number of batch dimensions. Each entry represents logits
        for the probability of success for independent Geometric distributions
        and must be in the range `(-inf, inf]`. Only one of `logits` or `probs`
        should be specified.
      probs: Positive floating-point `Tensor` with shape `[B1, ..., Bb]`
        where `b >= 0` indicates the number of batch dimensions. Each entry
        represents the probability of success for independent Geometric
        distributions and must be in the range `(0, 1]`. Only one of `logits`
        or `probs` should be specified.
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
    with tf.name_scope(name, values=[logits, probs]) as name:
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits, probs, validate_args=validate_args, name=name)

      with tf.control_dependencies([tf.assert_positive(self._probs)]
                                   if validate_args else []):
        self._probs = tf.identity(self._probs, name="probs")

    super(Geometric, self).__init__(
        dtype=self._probs.dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._probs, self._logits],
        name=name)

  @property
  def logits(self):
    """Log-odds of a `1` outcome (vs `0`)."""
    return self._logits

  @property
  def probs(self):
    """Probability of a `1` outcome (vs `0`)."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.shape(self._probs)

  def _batch_shape(self):
    return self.probs.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use `np.finfo(self.dtype.as_numpy_dtype).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    sampled = tf.random_uniform(
        tf.concat([[n], tf.shape(self._probs)], 0),
        minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
        maxval=1.,
        seed=seed,
        dtype=self.dtype)

    return tf.floor(tf.log(sampled) / tf.log1p(-self.probs))

  def _cdf(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    else:
      # Whether or not x is integer-form, the following is well-defined.
      # However, scipy takes the floor, so we do too.
      x = tf.floor(x)
    x *= tf.ones_like(self.probs)
    return tf.where(x < 0., tf.zeros_like(x), -tf.math.expm1(
        (1. + x) * tf.log1p(-self.probs)))

  def _log_prob(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    else:
      # For consistency with cdf, we take the floor.
      x = tf.floor(x)
    x *= tf.ones_like(self.probs)
    probs = self.probs * tf.ones_like(x)
    safe_domain = tf.where(tf.equal(x, 0.), tf.zeros_like(probs), probs)
    return x * tf.log1p(-safe_domain) + tf.log(probs)

  def _entropy(self):
    probs = self._probs
    if self.validate_args:
      probs = control_flow_ops.with_dependencies([
          tf.assert_less(
              probs,
              tf.constant(1., probs.dtype),
              message="Entropy is undefined when logits = inf or probs = 1.")
      ], probs)
    # Claim: entropy(p) = softplus(s)/p - s
    # where s=logits and p=probs.
    #
    # Proof:
    #
    # entropy(p)
    # := -[(1-p)log(1-p) + plog(p)]/p
    # = -[log(1-p) + plog(p/(1-p))]/p
    # = -[-softplus(s) + ps]/p
    # = softplus(s)/p - s
    #
    # since,
    # log[1-sigmoid(s)]
    # = log[1/(1+exp(s)]
    # = -log[1+exp(s)]
    # = -softplus(s)
    #
    # using the fact that,
    # 1-sigmoid(s) = sigmoid(-s) = 1/(1+exp(s))
    return tf.nn.softplus(self.logits) / probs - self.logits

  def _mean(self):
    return tf.exp(-self.logits)

  def _variance(self):
    return self._mean() / self.probs

  def _mode(self):
    return tf.zeros(self.batch_shape_tensor(), dtype=self.dtype)
