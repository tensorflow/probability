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
"""The Bernoulli distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import distribution_util as util
from tensorflow_probability.python.internal import reparameterization


class Bernoulli(distribution.Distribution):
  """Bernoulli distribution.

  The Bernoulli distribution with `probs` parameter, i.e., the probability of a
  `1` outcome (vs a `0` outcome).
  """

  def __init__(self,
               logits=None,
               probs=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               name="Bernoulli"):
    """Construct Bernoulli distributions.

    Args:
      logits: An N-D `Tensor` representing the log-odds of a `1` event. Each
        entry in the `Tensor` parametrizes an independent Bernoulli distribution
        where the probability of an event is sigmoid(logits). Only one of
        `logits` or `probs` should be passed in.
      probs: An N-D `Tensor` representing the probability of a `1`
        event. Each entry in the `Tensor` parameterizes an independent
        Bernoulli distribution. Only one of `logits` or `probs` should be passed
        in.
      dtype: The type of the event samples. Default: `int32`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: If p and logits are passed, or if neither are passed.
    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(name) as name:
      self._logits, self._probs = util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          name=name)
    super(Bernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._logits, self._probs],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"logits": tf.convert_to_tensor(value=sample_shape, dtype=tf.int32)}

  @classmethod
  def _params_event_ndims(cls):
    return dict(logits=0, probs=0)

  @property
  def logits(self):
    """Log-odds of a `1` outcome (vs `0`)."""
    return self._logits

  @property
  def probs(self):
    """Probability of a `1` outcome (vs `0`)."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.shape(input=self._logits)

  def _batch_shape(self):
    return self._logits.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    new_shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    uniform = tf.random.uniform(new_shape, seed=seed, dtype=self.probs.dtype)
    sample = tf.less(uniform, self.probs)
    return tf.cast(sample, self.dtype)

  def _log_prob(self, event):
    if self.validate_args:
      event = util.embed_check_integer_casting_closed(
          event, target_dtype=tf.bool)

    # TODO(jaana): The current sigmoid_cross_entropy_with_logits has
    # inconsistent behavior for logits = inf/-inf.
    event = tf.cast(event, self.logits.dtype)
    logits = self.logits
    # sigmoid_cross_entropy_with_logits doesn't broadcast shape,
    # so we do this here.

    def _broadcast(logits, event):
      return (tf.ones_like(event) * logits,
              tf.ones_like(logits) * event)

    if not (event.shape.is_fully_defined() and
            logits.shape.is_fully_defined() and
            event.shape == logits.shape):
      logits, event = _broadcast(logits, event)
    return -tf.nn.sigmoid_cross_entropy_with_logits(labels=event, logits=logits)

  def _entropy(self):
    return (-self.logits * (tf.sigmoid(self.logits) - 1) +
            tf.nn.softplus(-self.logits))

  def _mean(self):
    return tf.identity(self.probs)

  def _variance(self):
    return self._mean() * (1. - self.probs)

  def _mode(self):
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return tf.cast(self.probs > 0.5, self.dtype)


# TODO(b/117098119): Remove tf.distribution references once they're gone.
@kullback_leibler.RegisterKL(Bernoulli, tf.compat.v1.distributions.Bernoulli)
@kullback_leibler.RegisterKL(tf.compat.v1.distributions.Bernoulli, Bernoulli)
@kullback_leibler.RegisterKL(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Bernoulli.

  Args:
    a: instance of a Bernoulli distribution object.
    b: instance of a Bernoulli distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_bernoulli_bernoulli".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.compat.v1.name_scope(
      name, "kl_bernoulli_bernoulli", values=[a.logits, b.logits]):
    delta_probs0 = tf.nn.softplus(-b.logits) - tf.nn.softplus(-a.logits)
    delta_probs1 = tf.nn.softplus(b.logits) - tf.nn.softplus(a.logits)
    return (tf.sigmoid(a.logits) * delta_probs0
            + tf.sigmoid(-a.logits) * delta_probs1)
