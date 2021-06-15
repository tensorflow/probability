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
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


class Geometric(distribution.AutoCompositeTensorDistribution):
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
               force_probs_to_zero_outside_support=False,
               validate_args=False,
               allow_nan_stats=True,
               name='Geometric'):
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
      force_probs_to_zero_outside_support: Python `bool`. When `True`, negative
        and non-integer values are evaluated "strictly": `log_prob` returns
        `-inf`, `prob` returns `0`, and `cdf` and `sf` correspond.  When
        `False`, the implementation is free to save computation (and TF graph
        size) by evaluating something that matches the Geometric pmf at integer
        values `k` but produces an unrestricted result on other inputs. In the
        case of Geometric distribution, the `log_prob` formula in this case
        happens to be the continuous function `k * log(1 - probs) + log(probs)`.
        Note that this function is not a normalized probability log-density.
        Default value: `False`.
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
      dtype = dtype_util.common_dtype([logits, probs], dtype_hint=tf.float32)
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype=dtype, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype=dtype, name='logits')
      self._force_probs_to_zero_outside_support = (
          force_probs_to_zero_outside_support)
      super(Geometric, self).__init__(
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
            default_constraining_bijector_fn=softmax_centered_bijector
            .SoftmaxCentered,
            is_preferred=False))

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  @property
  def force_probs_to_zero_outside_support(self):
    """Return 0 probabilities on non-integer inputs."""
    return self._force_probs_to_zero_outside_support

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use
    # `np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny`
    # because it is the smallest, positive, 'normal' number. A 'normal' number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    probs = self._probs_parameter_no_checks()
    sampled = samplers.uniform(
        ps.concat([[n], ps.shape(probs)], 0),
        minval=np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny,
        maxval=1.,
        seed=seed,
        dtype=self.dtype)

    return tf.floor(tf.math.log(sampled) / tf.math.log1p(-probs))

  def _log_survival_function(self, x):
    probs = self._probs_parameter_no_checks()
    if not self.validate_args:
      # Whether or not x is integer-form, the following is well-defined.
      # However, scipy takes the floor, so we do too.
      x = tf.floor(x)
    return tf.where(
        x < 0.,
        dtype_util.as_numpy_dtype(x.dtype)(0.),
        (1. + x) * tf.math.log1p(-probs))

  def _log_cdf(self, x):
    probs = self._probs_parameter_no_checks()
    if not self.validate_args:
      # Whether or not x is integer-form, the following is well-defined.
      # However, scipy takes the floor, so we do too.
      x = tf.floor(x)
    return tf.where(
        x < 0.,
        dtype_util.as_numpy_dtype(x.dtype)(-np.inf),
        tfp_math.log1mexp((1. + x) * tf.math.log1p(-probs)))

  def _log_prob(self, x):
    probs = self._probs_parameter_no_checks()
    if not self.validate_args:
      # For consistency with cdf, we take the floor.
      x = tf.floor(x)

    log_probs = tf.math.xlog1py(x, -probs) + tf.math.log(probs)

    if self.force_probs_to_zero_outside_support:
      # Set log_prob = -inf when value is less than 0, ie prob = 0.
      log_probs = tf.where(
          x < 0.,
          dtype_util.as_numpy_dtype(x.dtype)(-np.inf),
          log_probs)
    return log_probs

  def _entropy(self):
    logits, probs = self._logits_and_probs_no_checks()
    if not self.validate_args:
      assertions = []
    else:
      assertions = [assert_util.assert_less(
          probs, dtype_util.as_numpy_dtype(self.dtype)(1.),
          message='Entropy is undefined when logits = inf or probs = 1.')]
    with tf.control_dependencies(assertions):
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
      return tf.math.softplus(logits) / probs - logits

  def _mean(self):
    return tf.exp(-self._logits_parameter_no_checks())

  def _variance(self):
    logits, probs = self._logits_and_probs_no_checks()
    return tf.exp(-logits) / probs

  def _mode(self):
    return tf.zeros(self.batch_shape_tensor(), dtype=self.dtype)

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      if self._logits is None:
        return tf.math.log(self._probs) - tf.math.log1p(-self._probs)
      return tf.identity(self._logits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      if self._logits is None:
        return tf.identity(self._probs)
      return tf.math.sigmoid(self._logits)

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log(probs) - tf.math.log1p(-probs)
    return tensor_util.identity_as_tensor(self._logits)

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return tf.math.sigmoid(self._logits)

  def _logits_and_probs_no_checks(self):
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      logits = tf.math.log(probs) - tf.math.log1p(-probs)
    else:
      logits = tf.convert_to_tensor(self._logits)
      probs = tf.math.sigmoid(logits)
    return logits, probs

  def _default_event_space_bijector(self):
    return

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if self._probs is not None:
      if is_init != tensor_util.is_ref(self._probs):
        probs = tf.convert_to_tensor(self._probs)
        assertions.append(assert_util.assert_positive(
            probs, message='Argument `probs` must be positive.'))
        assertions.append(assert_util.assert_less_equal(
            probs, dtype_util.as_numpy_dtype(self.dtype)(1.),
            message='Argument `probs` must be less than or equal to 1.'))
    return assertions
