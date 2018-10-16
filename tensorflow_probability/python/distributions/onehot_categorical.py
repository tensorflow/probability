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
"""The OneHotCategorical distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.ops import control_flow_ops


class OneHotCategorical(distribution.Distribution):
  """OneHotCategorical distribution.

  The categorical distribution is parameterized by the log-probabilities
  of a set of classes. The difference between OneHotCategorical and Categorical
  distributions is that OneHotCategorical is a discrete distribution over
  one-hot bit vectors whereas Categorical is a discrete distribution over
  positive integers. OneHotCategorical is equivalent to Categorical except
  Categorical has event_dim=() while OneHotCategorical has event_dim=K, where
  K is the number of classes.

  This class provides methods to create indexed batches of OneHotCategorical
  distributions. If the provided `logits` or `probs` is rank 2 or higher, for
  every fixed set of leading dimensions, the last dimension represents one
  single OneHotCategorical distribution. When calling distribution
  functions (e.g. `dist.prob(x)`), `logits` and `x` are broadcast to the
  same shape (if possible). In all cases, the last dimension of `logits,x`
  represents single OneHotCategorical distributions.

  #### Examples

  Creates a 3-class distribution, with the 2nd class, the most likely to be
  drawn from.

  ```python
  p = [0.1, 0.5, 0.4]
  dist = OneHotCategorical(probs=p)
  ```

  Creates a 3-class distribution, with the 2nd class the most likely to be
  drawn from, using logits.

  ```python
  logits = [-2, 2, 0]
  dist = OneHotCategorical(logits=logits)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be drawn.

  ```python
  # counts is a scalar.
  p = [0.1, 0.4, 0.5]
  dist = OneHotCategorical(probs=p)
  dist.prob([0,1,0])  # Shape []

  # p will be broadcast to [[0.1, 0.4, 0.5], [0.1, 0.4, 0.5]] to match.
  samples = [[0,1,0], [1,0,0]]
  dist.prob(samples)  # Shape [2]
  ```

  """

  def __init__(self,
               logits=None,
               probs=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               name="OneHotCategorical"):
    """Initialize OneHotCategorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
        set of Categorical distributions. The first `N - 1` dimensions index
        into a batch of independent distributions and the last dimension
        represents a vector of logits for each class. Only one of `logits` or
        `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities of a set
        of Categorical distributions. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of probabilities for each class. Only one of `logits` or `probs`
        should be passed in.
      dtype: The type of the event samples (default: int32).
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
          name=name, logits=logits, probs=probs, validate_args=validate_args,
          multidimensional=True)

      logits_shape_static = self._logits.shape.with_rank_at_least(1)
      if logits_shape_static.ndims is not None:
        self._batch_rank = tf.convert_to_tensor(
            logits_shape_static.ndims - 1, dtype=tf.int32, name="batch_rank")
      else:
        with tf.name_scope(name="batch_rank"):
          self._batch_rank = tf.rank(self._logits) - 1

      with tf.name_scope(name="event_size"):
        self._event_size = tf.shape(self._logits)[-1]

    super(OneHotCategorical, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._logits, self._probs],
        name=name)

  @property
  def event_size(self):
    """Scalar `int32` tensor: the number of classes."""
    return self._event_size

  @property
  def logits(self):
    """Vector of coordinatewise logits."""
    return self._logits

  @property
  def probs(self):
    """Vector of coordinatewise probabilities."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.shape(self.logits)[:-1]

  def _batch_shape(self):
    return self.logits.shape[:-1]

  def _event_shape_tensor(self):
    return tf.shape(self.logits)[-1:]

  def _event_shape(self):
    return self.logits.shape.with_rank_at_least(1)[-1:]

  def _sample_n(self, n, seed=None):
    sample_shape = tf.concat([[n], tf.shape(self.logits)], 0)
    logits = self.logits
    if logits.shape.ndims == 2:
      logits_2d = logits
    else:
      logits_2d = tf.reshape(logits, [-1, self.event_size])
    samples = tf.multinomial(logits_2d, n, seed=seed)
    samples = tf.transpose(samples)
    samples = tf.one_hot(samples, self.event_size, dtype=self.dtype)
    ret = tf.reshape(samples, sample_shape)
    return ret

  def _log_prob(self, x):
    x = self._assert_valid_sample(x)
    # broadcast logits or x if need be.
    logits = self.logits
    if (not x.shape.is_fully_defined() or
        not logits.shape.is_fully_defined() or
        x.shape != logits.shape):
      logits = tf.ones_like(x, dtype=logits.dtype) * logits
      x = tf.ones_like(logits, dtype=x.dtype) * x

    logits_shape = tf.shape(tf.reduce_sum(logits, -1))
    logits_2d = tf.reshape(logits, [-1, self.event_size])
    x_2d = tf.reshape(x, [-1, self.event_size])
    ret = -tf.nn.softmax_cross_entropy_with_logits(
        labels=x_2d, logits=logits_2d)
    # Reshape back to user-supplied batch and sample dims prior to 2D reshape.
    ret = tf.reshape(ret, logits_shape)
    return ret

  def _entropy(self):
    return -tf.reduce_sum(tf.nn.log_softmax(self.logits) * self.probs, axis=-1)

  def _mode(self):
    ret = tf.argmax(self.logits, axis=self._batch_rank)
    ret = tf.one_hot(ret, self.event_size, dtype=self.dtype)
    ret.set_shape(self.logits.shape)
    return ret

  def _covariance(self):
    p = self.probs
    ret = -tf.matmul(p[..., None], p[..., None, :])
    return tf.matrix_set_diag(ret, self._variance())

  def _variance(self):
    return self.probs * (1. - self.probs)

  def _assert_valid_sample(self, x):
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        tf.assert_non_positive(x),
        tf.assert_near(
            tf.zeros([], dtype=self.dtype), tf.reduce_logsumexp(x, axis=[-1])),
    ], x)


@kullback_leibler.RegisterKL(OneHotCategorical, OneHotCategorical)
def _kl_categorical_categorical(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a, b OneHotCategorical.

  Args:
    a: instance of a OneHotCategorical distribution object.
    b: instance of a OneHotCategorical distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_categorical_categorical".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(
      name, "kl_categorical_categorical", values=[a.logits, b.logits]):
    # sum(p ln(p / q))
    return tf.reduce_sum(
        tf.nn.softmax(a.logits) *
        (tf.nn.log_softmax(a.logits) - tf.nn.log_softmax(b.logits)),
        axis=-1)
