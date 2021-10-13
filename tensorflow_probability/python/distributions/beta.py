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
"""The Beta distribution class."""

import functools

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'Beta',
]


_beta_sample_note = """Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`."""


class Beta(distribution.AutoCompositeTensorDistribution):
  """Beta distribution.

  The Beta distribution is defined over the `(0, 1)` interval using parameters
  `concentration1` (aka 'alpha') and `concentration0` (aka 'beta').

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
  Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
  ```

  where:

  * `concentration1 = alpha`,
  * `concentration0 = beta`,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The concentration parameters represent mean total counts of a `1` or a `0`,
  i.e.,

  ```none
  concentration1 = alpha = mean * total_concentration
  concentration0 = beta  = (1. - mean) * total_concentration
  ```

  where `mean` in `(0, 1)` and `total_concentration` is a positive real number
  representing a mean `total_count = concentration1 + concentration0`.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: The samples can be zero due to finite precision.
  This happens more often when some of the concentrations are very small.
  Make sure to round the samples to `np.finfo(dtype).tiny` before computing the
  density.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Create a batch of three Beta distributions.
  alpha = [1, 2, 3]
  beta = [1, 2, 3]
  dist = tfd.Beta(alpha, beta)

  dist.sample([4, 5])  # Shape [4, 5, 3]

  # `x` has three batch entries, each with two samples.
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  # Calculate the probability of each pair of samples under the corresponding
  # distribution in `dist`.
  dist.prob(x)         # Shape [2, 3]
  ```

  ```python
  # Create batch_shape=[2, 3] via parameter broadcast:
  alpha = [[1.], [2]]      # Shape [2, 1]
  beta = [3., 4, 5]        # Shape [3]
  dist = tfd.Beta(alpha, beta)

  # alpha broadcast as: [[1., 1, 1,],
  #                      [2, 2, 2]]
  # beta broadcast as:  [[3., 4, 5],
  #                      [3, 4, 5]]
  # batch_Shape [2, 3]
  dist.sample([4, 5])  # Shape [4, 5, 2, 3]

  x = [.2, .3, .5]
  # x will be broadcast as [[.2, .3, .5],
  #                         [.2, .3, .5]],
  # thus matching batch_shape [2, 3].
  dist.prob(x)         # Shape [2, 3]
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  alpha = tf.constant(1.0)
  beta = tf.constant(2.0)
  dist = tfd.Beta(alpha, beta)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [alpha, beta])
  ```

  """

  def __init__(self,
               concentration1,
               concentration0,
               validate_args=False,
               allow_nan_stats=True,
               force_probs_to_zero_outside_support=False,
               name='Beta'):
    """Initialize a batch of Beta distributions.

    Args:
      concentration1: Positive floating-point `Tensor` indicating mean
        number of successes; aka 'alpha'.
      concentration0: Positive floating-point `Tensor` indicating mean
        number of failures; aka 'beta'.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      force_probs_to_zero_outside_support: If `True`, force `prob(x) == 0` and
        `log_prob(x) == -inf` for values of x outside the distribution support.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    self._force_probs_to_zero_outside_support = (
        force_probs_to_zero_outside_support)
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration1, concentration0],
                                      dtype_hint=tf.float32)
      self._concentration1 = tensor_util.convert_nonref_to_tensor(
          concentration1, dtype=dtype, name='concentration1')
      self._concentration0 = tensor_util.convert_nonref_to_tensor(
          concentration0, dtype=dtype, name='concentration0')
      super(Beta, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        concentration1=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration0=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def concentration1(self):
    """Concentration parameter associated with a `1` outcome."""
    return self._concentration1

  @property
  def concentration0(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._concentration0

  @property
  def force_probs_to_zero_outside_support(self):
    return self._force_probs_to_zero_outside_support

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    seed1, seed2 = samplers.split_seed(seed, salt='beta')
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    shape = self._batch_shape_tensor(concentration1=concentration1,
                                     concentration0=concentration0)
    expanded_concentration1 = tf.broadcast_to(concentration1, shape)
    expanded_concentration0 = tf.broadcast_to(concentration0, shape)
    log_gamma1 = gamma_lib.random_gamma(
        shape=[n], concentration=expanded_concentration1, seed=seed1,
        log_space=True)
    log_gamma2 = gamma_lib.random_gamma(
        shape=[n], concentration=expanded_concentration0, seed=seed2,
        log_space=True)
    return tf.math.sigmoid(log_gamma1 - log_gamma2)

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _log_prob(self, x):
    concentration0 = tf.convert_to_tensor(self.concentration0)
    concentration1 = tf.convert_to_tensor(self.concentration1)
    lp = (self._log_unnormalized_prob(x, concentration1, concentration0) -
          self._log_normalization(concentration1, concentration0))
    if self.force_probs_to_zero_outside_support:
      return tf.where((x >= 0) & (x <= 1), lp, -float('inf'))
    return lp

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _prob(self, x):
    return tf.exp(self._log_prob(x))

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _log_cdf(self, x):
    return tf.math.log(self._cdf(x))

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _cdf(self, x):
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    shape = functools.reduce(
        ps.broadcast_shape,
        [ps.shape(concentration1),
         ps.shape(concentration0),
         ps.shape(x)])
    concentration1 = tf.broadcast_to(concentration1, shape)
    concentration0 = tf.broadcast_to(concentration0, shape)
    x = tf.broadcast_to(x, shape)

    safe_x = tf.where(tf.logical_and(x >= 0, x < 1), x, 0.5)
    answer = tf.math.betainc(concentration1, concentration0, safe_x)
    return distribution_util.extend_cdf_outside_support(
        x, answer, low=0., high=1.)

  def _log_unnormalized_prob(self, x, concentration1, concentration0):
    return (tf.math.xlogy(concentration1 - 1., x) +
            tf.math.xlog1py(concentration0 - 1., -x))

  def _log_normalization(self, concentration1, concentration0):
    return tfp_math.lbeta(concentration1, concentration0)

  def _entropy(self):
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    total_concentration = concentration1 + concentration0
    return (self._log_normalization(concentration1, concentration0) -
            (concentration1 - 1.) * tf.math.digamma(concentration1) -
            (concentration0 - 1.) * tf.math.digamma(concentration0) +
            (total_concentration - 2.) * tf.math.digamma(total_concentration))

  def _mean(self):
    concentration1 = tf.convert_to_tensor(self.concentration1)
    return concentration1 / (concentration1 + self.concentration0)

  def _variance(self):
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    total_concentration = concentration1 + concentration0
    return (concentration1 * concentration0 /
            ((total_concentration)**2 * (total_concentration + 1.)))

  @distribution_util.AppendDocstring(
      """Note: The mode is undefined when `concentration1 <= 1` or
      `concentration0 <= 1`. If `self.allow_nan_stats` is `True`, `NaN`
      is used for undefined modes. If `self.allow_nan_stats` is `False` an
      exception is raised when one or more modes are undefined.""")
  def _mode(self):
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    mode = (concentration1 - 1.) / (concentration1 + concentration0 - 2.)
    with tf.control_dependencies([] if self.allow_nan_stats else [  # pylint: disable=g-long-ternary
        assert_util.assert_less(
            tf.ones([], dtype=self.dtype),
            concentration1,
            message='Mode undefined for concentration1 <= 1.'),
        assert_util.assert_less(
            tf.ones([], dtype=self.dtype),
            concentration0,
            message='Mode undefined for concentration0 <= 1.')
    ]):
      return tf.where(
          (concentration1 > 1.) & (concentration0 > 1.),
          mode,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    """Checks the validity of a sample."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    assertions.append(assert_util.assert_less_equal(
        x, tf.ones([], x.dtype),
        message='Sample must be less than or equal to `1`.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    for concentration in [self.concentration0, self.concentration1]:
      if is_init != tensor_util.is_ref(concentration):
        assertions.append(assert_util.assert_positive(
            concentration,
            message='Concentration parameter must be positive.'))
    return assertions


@kullback_leibler.RegisterKL(Beta, Beta)
def _kl_beta_beta(d1, d2, name=None):
  """Calculate the batchwise KL divergence KL(d1 || d2) with d1 and d2 Beta.

  Args:
    d1: instance of a Beta distribution object.
    d2: instance of a Beta distribution object.
    name: (optional) Name to use for created operations.
      default is 'kl_beta_beta'.

  Returns:
    Batchwise KL(d1 || d2)
  """
  with tf.name_scope(name or 'kl_beta_beta'):
    d1_concentration1 = tf.convert_to_tensor(d1.concentration1)
    d1_concentration0 = tf.convert_to_tensor(d1.concentration0)
    d2_concentration1 = tf.convert_to_tensor(d2.concentration1)
    d2_concentration0 = tf.convert_to_tensor(d2.concentration0)
    d1_total_concentration = d1_concentration1 + d1_concentration0
    d2_total_concentration = d2_concentration1 + d2_concentration0

    d1_log_normalization = d1._log_normalization(  # pylint: disable=protected-access
        d1_concentration1, d1_concentration0)
    d2_log_normalization = d2._log_normalization(  # pylint: disable=protected-access
        d2_concentration1, d2_concentration0)
    return ((d2_log_normalization - d1_log_normalization) -
            (tf.math.digamma(d1_concentration1) *
             (d2_concentration1 - d1_concentration1)) -
            (tf.math.digamma(d1_concentration0) *
             (d2_concentration0 - d1_concentration0)) +
            (tf.math.digamma(d1_total_concentration) *
             (d2_total_concentration - d1_total_concentration)))
