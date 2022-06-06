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
"""The ProbitBernoulli distribution class."""

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
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util


class ProbitBernoulli(
    distribution.DiscreteDistributionMixin,
    distribution.AutoCompositeTensorDistribution):
  """ProbitBernoulli distribution.

  The ProbitBernoulli distribution with `probs` parameter, i.e., the probability
  of a `1` outcome (vs a `0` outcome). Unlike a regular Bernoulli distribution,
  which uses the logistic (aka 'sigmoid') function to go from the un-constrained
  parameters to probabilities, this distribution uses the CDF of the [standard
  normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):

  ```none
  p(x=1; probits) = 0.5 * (1 + erf(probits / sqrt(2)))
  p(x=0; probits) = 1 - p(x=1; probits)
  ```

  Where `erf` is the [error
  function](https://en.wikipedia.org/wiki/Error_function). A typical application
  of this distribution is in [probit
  regression](https://en.wikipedia.org/wiki/Probit_model).
  """

  def __init__(self,
               probits=None,
               probs=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               name='ProbitBernoulli'):
    """Construct ProbitBernoulli distributions.

    Args:
      probits: An N-D `Tensor` representing the probit-odds of a `1` event. Each
        entry in the `Tensor` parameterizes an independent ProbitBernoulli
        distribution where the probability of an event is normal_cdf(probits).
        Only one of `probits` or `probs` should be passed in.
      probs: An N-D `Tensor` representing the probability of a `1`
        event. Each entry in the `Tensor` parameterizes an independent
        ProbitBernoulli distribution. Only one of `probits` or `probs` should be
        passed in.
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
      ValueError: If probs and probits are passed, or if neither are passed.
    """
    parameters = dict(locals())
    if (probs is None) == (probits is None):
      raise ValueError('Must pass probs or probits, but not both.')
    with tf.name_scope(name) as name:
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=tf.float32, name='probs')
      self._probits = tensor_util.convert_nonref_to_tensor(
          probits, dtype_hint=tf.float32, name='probits')
    super(ProbitBernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        probits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))

  @property
  def probits(self):
    """Input argument `probits`."""
    return self._probits

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
    if self._probits is None:
      p = tf.convert_to_tensor(self._probs)
      return tf.math.log1p(-p), tf.math.log(p)
    s = tf.convert_to_tensor(self._probits)
    return special_math.log_ndtr(-s), special_math.log_ndtr(s)

  def _entropy(self):
    log_probs0, log_probs1 = self._outcome_log_probs()
    probs1 = tf.exp(log_probs1)
    return -(1. - probs1) * log_probs0 - probs1 * log_probs1

  def _mean(self):
    return self._probs_parameter_no_checks()

  def _variance(self):
    mean = self._mean()
    return mean * (1. - mean)

  def _mode(self):
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return tf.cast(self._probs_parameter_no_checks() > 0.5, self.dtype)

  def probits_parameter(self, name=None):
    """Probits computed from non-`None` input arg (`probs` or `probits`)."""
    with self._name_and_control_scope(name or 'probits_parameter'):
      return self._probits_parameter_no_checks()

  def _probits_parameter_no_checks(self):
    if self._probits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.ndtri(probs)
    return tensor_util.identity_as_tensor(self._probits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `probits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._probits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return special_math.ndtr(self._probits)

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    return maybe_assert_bernoulli_param_correctness(
        is_init, self.validate_args, self._probs, self._probits)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    assertions.append(
        assert_util.assert_less_equal(x, tf.ones([], dtype=x.dtype),
                                      message='Elements cannot exceed 1.'))
    return assertions


def maybe_assert_bernoulli_param_correctness(
    is_init, validate_args, probs, probits):
  """Return assertions for `ProbitBernoulli`-type distributions."""
  if is_init:
    x, name = (probs, 'probs') if probits is None else (probits, 'probits')
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


@kullback_leibler.RegisterKL(ProbitBernoulli, ProbitBernoulli)
def _kl_bernoulli_bernoulli(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b ProbitBernoulli.

  Args:
    a: instance of a ProbitBernoulli distribution object.
    b: instance of a ProbitBernoulli distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_bernoulli_bernoulli'`).

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_probit_bernoulli_probit_bernoulli'):
    a_log_probs0, a_log_probs1 = a._outcome_log_probs()  # pylint: disable=protected-access
    b_log_probs0, b_log_probs1 = b._outcome_log_probs()  # pylint: disable=protected-access
    a_prob1 = tf.exp(a_log_probs1)

    return (1. - a_prob1) * (a_log_probs0 - b_log_probs0) + a_prob1 * (
        a_log_probs1 - b_log_probs1)
