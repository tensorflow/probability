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
"""The Kumaraswamy distribution class."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import kumaraswamy_cdf
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.math import special

__all__ = [
    'Kumaraswamy',
]

_kumaraswamy_sample_note = """Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`."""


def _harmonic_number(x):
  """Compute the harmonic number from its analytic continuation.

  Derivation from [here](
  https://en.wikipedia.org/wiki/Digamma_function#Relation_to_harmonic_numbers)
  and [Euler's constant](
  https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant).

  Args:
    x: input float.

  Returns:
    z: The analytic continuation of the harmonic number for the input.
  """
  one = tf.ones([], dtype=x.dtype)
  return tf.math.digamma(x + one) - tf.math.digamma(one)


class Kumaraswamy(transformed_distribution.TransformedDistribution):
  """Kumaraswamy distribution.

  The Kumaraswamy distribution is defined over the `(0, 1)` interval using
  parameters
  `concentration1` (aka 'alpha') and `concentration0` (aka 'beta').  It has a
  shape similar to the Beta distribution, but is easier to reparameterize.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta) = alpha * beta * x**(alpha - 1) * (1 - x**alpha)**(beta -
  1)
  ```

  where:

  * `concentration1 = alpha`,
  * `concentration0 = beta`,

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Examples

  ```python
  # Create a batch of three Kumaraswamy distributions.
  alpha = [1, 2, 3]
  beta = [1, 2, 3]
  dist = Kumaraswamy(alpha, beta)

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
  dist = Kumaraswamy(alpha, beta)

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

  """

  def __init__(self,
               concentration1=1.,
               concentration0=1.,
               validate_args=False,
               allow_nan_stats=True,
               name='Kumaraswamy'):
    """Initialize a batch of Kumaraswamy distributions.

    Args:
      concentration1: Positive floating-point `Tensor` indicating mean
        number of successes; aka 'alpha'. Implies `self.dtype` and
        `self.batch_shape`, i.e.,
        `concentration1.shape = [N1, N2, ..., Nm] = self.batch_shape`.
      concentration0: Positive floating-point `Tensor` indicating mean
        number of failures; aka 'beta'. Otherwise has same semantics as
        `concentration1`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration1, concentration0],
                                      dtype_hint=tf.float32)
      concentration1 = tensor_util.convert_nonref_to_tensor(
          concentration1, name='concentration1', dtype=dtype)
      concentration0 = tensor_util.convert_nonref_to_tensor(
          concentration0, name='concentration0', dtype=dtype)
      self._kumaraswamy_cdf = kumaraswamy_cdf.KumaraswamyCDF(
          concentration1=concentration1,
          concentration0=concentration0,
          validate_args=validate_args)
      super(Kumaraswamy, self).__init__(
          distribution=uniform.Uniform(
              low=tf.zeros([], dtype=dtype),
              high=tf.ones([], dtype=dtype),
              allow_nan_stats=allow_nan_stats),
          bijector=invert.Invert(
              self._kumaraswamy_cdf, validate_args=validate_args),
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
    return self._kumaraswamy_cdf.concentration1

  @property
  def concentration0(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._kumaraswamy_cdf.concentration0

  experimental_is_sharded = False

  def _entropy(self):
    a = tf.convert_to_tensor(self.concentration1)
    b = tf.convert_to_tensor(self.concentration0)
    return ((1 - 1. / b) + (1 - 1. / a) * _harmonic_number(b) -
            tf.math.log(a) - tf.math.log(b))

  def _log_cdf(self, x):
    return generic.log1mexp(self.concentration0 * tf.math.log1p(
        -x ** self.concentration1))

  def _log_moment(self, n, concentration1=None, concentration0=None):
    """Compute the n'th (uncentered) moment."""
    concentration0 = tf.convert_to_tensor(
        self.concentration0) if concentration0 is None else concentration0
    concentration1 = tf.convert_to_tensor(
        self.concentration1) if concentration1 is None else concentration1
    total_concentration = concentration1 + concentration0
    expanded_concentration1 = tf.broadcast_to(concentration1,
                                              tf.shape(total_concentration))
    expanded_concentration0 = tf.broadcast_to(concentration0,
                                              tf.shape(total_concentration))
    beta_arg = 1 + n / expanded_concentration1
    return (tf.math.log(expanded_concentration0) +
            special.lbeta(beta_arg, expanded_concentration0))

  def _mean(self):
    return tf.exp(self._log_moment(1))

  def _variance(self):
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    log_moment2 = self._log_moment(
        2, concentration1=concentration1, concentration0=concentration0)
    log_moment1 = self._log_moment(
        1, concentration1=concentration1, concentration0=concentration0)
    lswe, sign = generic.reduce_weighted_logsumexp(
        tf.stack([log_moment2, 2 * log_moment1], axis=-1),
        [1., -1],
        axis=-1,
        return_sign=True)
    return sign * tf.exp(lswe)

  @distribution_util.AppendDocstring(
      """Note: The mode is undefined when `concentration1 <= 1` or
      `concentration0 <= 1`. If `self.allow_nan_stats` is `True`, `NaN`
      is used for undefined modes. If `self.allow_nan_stats` is `False` an
      exception is raised when one or more modes are undefined.""")
  def _mode(self):
    a = tf.convert_to_tensor(self.concentration1)
    b = tf.convert_to_tensor(self.concentration0)
    mode = ((a - 1) / (a * b - 1))**(1. / a)
    if self.allow_nan_stats:
      return tf.where((a > 1.) & (b > 1.), mode,
                      dtype_util.as_numpy_dtype(self.dtype)(np.nan))

    return distribution_util.with_dependencies([
        assert_util.assert_less(
            tf.ones([], dtype=a.dtype),
            a,
            message='Mode undefined for concentration1 <= 1.'),
        assert_util.assert_less(
            tf.ones([], dtype=b.dtype),
            b,
            message='Mode undefined for concentration0 <= 1.')
    ], mode)

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    return self.bijector.bijector._parameter_control_dependencies(is_init)  # pylint: disable=protected-access

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
