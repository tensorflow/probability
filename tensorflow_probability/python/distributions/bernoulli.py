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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


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
               name='Bernoulli'):
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
    if (probs is None) == (logits is None):
      raise ValueError('Must pass probs or logits, but not both.')
    with tf.name_scope(name) as name:
      self._probs = tensor_util.convert_immutable_to_tensor(
          probs, dtype_hint=tf.float32, name='probs')
      self._logits = tensor_util.convert_immutable_to_tensor(
          logits, dtype_hint=tf.float32, name='logits')
    super(Bernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {'logits': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

  @classmethod
  def _params_event_ndims(cls):
    return dict(logits=0, probs=0)

  @property
  def logits(self):
    """Input argument `logits`."""
    if self._logits is None:
      return self._logits_deprecated_behavior()
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    if self._probs is None:
      return self._probs_deprecated_behavior()
    return self._probs

  def _batch_shape_tensor(self):
    x = self._probs if self._logits is None else self._logits
    return tf.shape(x)

  def _batch_shape(self):
    x = self._probs if self._logits is None else self._logits
    return x.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    probs = self._probs_parameter_no_checks()
    new_shape = tf.concat([[n], tf.shape(probs)], 0)
    uniform = tf.random.uniform(new_shape, seed=seed, dtype=probs.dtype)
    sample = tf.less(uniform, probs)
    return tf.cast(sample, self.dtype)

  def _log_prob(self, event):
    if self.validate_args:
      event = distribution_util.embed_check_integer_casting_closed(
          event, target_dtype=tf.bool)

    log_probs0, log_probs1 = self._outcome_log_probs()
    event = tf.cast(event, log_probs0.dtype)
    return event * (log_probs1 - log_probs0) + log_probs0

  def _outcome_log_probs(self):
    if self._logits is None:
      p = tf.convert_to_tensor(self._probs)
      return tf.math.log1p(-p), tf.math.log(p)
    s = tf.convert_to_tensor(self._logits)
    return -tf.math.softplus(s), -tf.math.softplus(-s)

  def _entropy(self):
    logits = self._logits_parameter_no_checks()
    return -logits * (tf.sigmoid(logits) - 1) + tf.math.softplus(-logits)

  def _mean(self):
    return self._probs_parameter_no_checks()

  def _variance(self):
    mean = self._mean()
    return mean * (1. - mean)

  def _mode(self):
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return tf.cast(self._probs_parameter_no_checks() > 0.5, self.dtype)

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

  @deprecation.deprecated(
      '2019-10-01',
      'The `logits` property will return `None` when the distribution is '
      'parameterized with `logits=None`. Use `logits_parameter()` instead.',
      warn_once=True)
  def _logits_deprecated_behavior(self):
    return self.logits_parameter()

  @deprecation.deprecated(
      '2019-10-01',
      'The `probs` property will return `None` when the distribution is '
      'parameterized with `probs=None`. Use `probs_parameter()` instead.',
      warn_once=True)
  def _probs_deprecated_behavior(self):
    return self.probs_parameter()

  def _parameter_control_dependencies(self, is_init):
    return maybe_assert_bernoulli_param_correctness(
        is_init, self.validate_args, self._probs, self._logits)


def maybe_assert_bernoulli_param_correctness(
    is_init, validate_args, probs, logits):
  """Return assertions for `Bernoulli`-type distributions."""
  if is_init:
    x, name = (probs, 'probs') if logits is None else (logits, 'logits')
    if not dtype_util.is_floating(x.dtype):
      raise TypeError(
          'Argument `{}` must having floating type.'.format(name))

  if not validate_args:
    return []

  assertions = []

  if probs is not None:
    if is_init != tensor_util.is_mutable(probs):
      probs = tf.convert_to_tensor(probs)
      one = tf.constant(1., probs.dtype)
      assertions += [
          assert_util.assert_non_negative(
              probs, message='probs has components less than 0.'),
          assert_util.assert_less_equal(
              probs, one, message='probs has components greater than 1.')
      ]

  return assertions


@kullback_leibler.RegisterKL(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Bernoulli.

  Args:
    a: instance of a Bernoulli distribution object.
    b: instance of a Bernoulli distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_bernoulli_bernoulli'`).

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_bernoulli_bernoulli'):
    a_logits = a._logits_parameter_no_checks()  # pylint:disable=protected-access
    b_logits = b._logits_parameter_no_checks()  # pylint:disable=protected-access
    return (
        tf.sigmoid(a_logits) * (
            tf.math.softplus(-b_logits) - tf.math.softplus(-a_logits)) +
        tf.sigmoid(-a_logits) * (
            tf.math.softplus(b_logits) - tf.math.softplus(a_logits)))
