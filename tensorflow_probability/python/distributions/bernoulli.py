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
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


class Bernoulli(distribution.AutoCompositeTensorDistribution):
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
        entry in the `Tensor` parameterizes an independent Bernoulli
        distribution where the probability of an event is sigmoid(logits). Only
        one of `logits` or `probs` should be passed in.
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
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=tf.float32, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype_hint=tf.float32, name='logits')
    super(Bernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        logits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    probs = self._probs_parameter_no_checks()
    new_shape = ps.concat([[n], ps.shape(probs)], 0)
    uniform = samplers.uniform(new_shape, seed=seed, dtype=probs.dtype)
    sample = tf.less(uniform, probs)
    return tf.cast(sample, self.dtype)

  def _log_prob(self, event):
    log_probs0, log_probs1 = self._outcome_log_probs()
    event = tf.cast(event, log_probs0.dtype)
    return (tf.math.multiply_no_nan(log_probs0, 1 - event) +
            tf.math.multiply_no_nan(log_probs1, event))

  def _outcome_log_probs(self):
    if self._logits is None:
      p = tf.convert_to_tensor(self._probs)
      return tf.math.log1p(-p), tf.math.log(p)
    s = tf.convert_to_tensor(self._logits)
    # softplus(s) = -Log[1 - p]
    # -softplus(-s) = Log[p]
    # softplus(+inf) = +inf, softplus(-inf) = 0, so...
    #  logits = -inf ==> log_probs0 = 0, log_probs1 = -inf (as desired)
    #  logits = +inf ==> log_probs0 = -inf, log_probs1 = 0 (as desired)
    return -tf.math.softplus(s), -tf.math.softplus(-s)

  def _cdf(self, event):
    prob = self._probs_parameter_no_checks()
    return tf.where(event < 0, 0.0, tf.where(event < 1, 1.0 - prob, 1.0))

  def _entropy(self):
    probs0, probs1, log_probs0, log_probs1 = _probs_and_log_probs(
        probs=self._probs, logits=self._logits, return_log_probs=True)
    return -1. * (
        tf.math.multiply_no_nan(log_probs0, probs0) +
        tf.math.multiply_no_nan(log_probs1, probs1))

  def _mean(self):
    return self._probs_parameter_no_checks()

  def _variance(self):
    probs0, probs1 = _probs_and_log_probs(
        probs=self._probs, logits=self._logits, return_log_probs=False)
    return probs0 * probs1

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
    return tensor_util.identity_as_tensor(self._logits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return tf.math.sigmoid(self._logits)

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    return maybe_assert_bernoulli_param_correctness(
        is_init, self.validate_args, self._probs, self._logits)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    assertions.append(
        assert_util.assert_less_equal(
            x, tf.ones([], dtype=x.dtype),
            message='Sample must be less than or equal to `1`.'))
    return assertions


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
    if is_init != tensor_util.is_ref(probs):
      probs = tf.convert_to_tensor(probs)
      one = tf.constant(1., probs.dtype)
      assertions += [
          assert_util.assert_non_negative(
              probs, message='probs has components less than 0.'),
          assert_util.assert_less_equal(
              probs, one, message='probs has components greater than 1.')
      ]

  return assertions


def _probs_and_log_probs(probs=None,
                         logits=None,
                         return_probs=True,
                         return_log_probs=True):
  """Get parts/all of (1 - p, p, Log[1 - p], Log[p]);  only one conversion."""
  to_return = ()

  # All use cases provide exactly one of probs or logits.  If we were to choose,
  # we prefer logits.  Why?  Because, for p very close to 1,
  # 1 - p will equal 0 (in finite precision), whereas sigmoid(-logits) will give
  # the correct (tiny) value.
  assert (probs is None) != (logits is None), 'Provide exactly one.'

  if logits is None:
    p = tf.convert_to_tensor(probs)
    if return_probs:
      to_return += (1 - p, p)
    if return_log_probs:
      to_return += (tf.math.log1p(-p), tf.math.log(p))
    return to_return

  s = tf.convert_to_tensor(logits)
  if return_probs:
    to_return += (tf.math.sigmoid(-s), tf.math.sigmoid(s))
  if return_log_probs:
    to_return += (-tf.math.softplus(s), -tf.math.softplus(-s))
  return to_return


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
    # KL[a || b] = Pa * Log[Pa / Pb] + (1 - Pa) * Log[(1 - Pa) / (1 - Pb)]
    # This is defined iff (Pb = 0 ==> Pa = 0) AND (Pb = 1 ==> Pa = 1).
    a_logits = a._logits_parameter_no_checks()  # pylint:disable=protected-access
    b_logits = b._logits_parameter_no_checks()  # pylint:disable=protected-access

    one_minus_pa, pa, log_one_minus_pa, log_pa = _probs_and_log_probs(
        logits=a_logits)
    log_one_minus_pb, log_pb = _probs_and_log_probs(logits=b_logits,
                                                    return_probs=False)

    # Multiply each factor individually to avoid Inf - Inf.
    return (
        tf.math.multiply_no_nan(log_pa, pa) -
        tf.math.multiply_no_nan(log_pb, pa) +
        tf.math.multiply_no_nan(log_one_minus_pa, one_minus_pa) -
        tf.math.multiply_no_nan(log_one_minus_pb, one_minus_pa)
    )
