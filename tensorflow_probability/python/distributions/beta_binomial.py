# Copyright 2020 The TensorFlow Probability Authors.
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
"""The BetaBinomial distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import binomial
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'BetaBinomial',
]

_beta_binomial_sample_note = """
For each batch member of counts `value`, `P[value]` is the probability that
after sampling `self.total_count` draws from this Binomial distribution, the
number of successes is `value`. Since different sequences of draws can result in
the same counts, the probability includes a combinatorial coefficient.

Note: `value` must be a non-negative tensor with dtype `dtype` and whose shape
can be broadcast with `self.total_count`, `self.concentration1` and
`self.concentration0`. `value` is only legal if it is less than or equal to
`self.total_count` and its components are equal to integer values.
"""


class BetaBinomial(distribution.Distribution):
  """Beta-Binomial compound distribution.

  The Beta-Binomial distribution is parameterized by (a batch of) `total_count`
  parameters, the number of trials per draw from Binomial distributions where
  the probabilities of success per trial are drawn from underlying Beta
  distributions; the Beta distributions are parameterized by `concentration1`
  (aka 'alpha') and `concentration0` (aka 'beta').

  Mathematically, it is (equivalent to) a special case of the
  Dirichlet-Multinomial over two classes, although the computational
  representation is slightly different: while the Beta-Binomial is a
  distribution over the number of successes in `total_count` trials, the
  two-class Dirichlet-Multinomial is a distribution over the number of successes
  and failures.

  #### Mathematical Details

  The Beta-Binomial is a distribution over the number of successes in
  `total_count` independent Binomial trials, with each trial having the same
  probability of success, the underlying probability being unknown but drawn
  from a Beta distribution with known parameters.

  The probability mass function (pmf) is,

  ```none
  pmf(k; n, a, b) = Beta(k + a, n - k + b) / Z
  Z = (k! (n - k)! / n!) * Beta(a, b)
  ```

  where:
  * `concentration1 = a > 0`,
  * `concentration0 = b > 0`,
  * `total_count = n`, `n` a positive integer,
  * `n!` is `n` factorial,
  * `Beta(x, y) = Gamma(x) Gamma(y) / Gamma(x + y)` is the
    [beta function](https://en.wikipedia.org/wiki/Beta_function), and
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  Dirichlet-Multinomial is a [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e., its
  samples are generated as follows.

    1. Choose success probabilities:
       `probs ~ Beta(concentration1, concentration0)`
    2. Draw integers representing the number of successes:
       `counts ~ Binomial(total_count, probs)`

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Examples

  Create a single distribution, corresponding to 5 coin flips.

  ```python
  dist = BetaBinomial(total_count=5., concentration1=.5, concentration0=.5)
  ```

  Creates 3 distributions with differing numbers of coin flips. The
  concentration parameters are broadcast.

  ```python
  dist = BetaBinomial(
     total_count=[5., 10., 20.], concentration1=.5, concentration0=.5)
  ```

  Creates 3 distribution, with differing numbers of coin flips and differing
  concentration parameters.

  ```python
  dist = BetaBinomial(
     total_count=[5., 10., 20.],
     concentration1=[.5, 2., 3.],
     concentration0=[4., 3., 2.])
  ```

  The distribution `log_prob` functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 2, 3]
  dist.log_prob(counts)  # Shape [3]

  # p will be broadcast to [[.2, .3, .8], [.2, .3, .8]] to match counts.
  counts = [[1., 2, 1], [2, 2, 4]]
  dist.log_prob(counts)  # Shape [2, 3]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.log_prob(counts)  # Shape [5, 7, 3]
  ```
  """

  def __init__(self,
               total_count,
               concentration1,
               concentration0,
               validate_args=False,
               allow_nan_stats=True,
               name='BetaBinomial'):
    """Initialize a batch of BetaBinomial distributions.

    Args:
      total_count: Non-negative integer-valued tensor, whose dtype is the same
        as `concentration1` and `concentration0`. The shape is broadcastable to
        `[N1,..., Nm]` with `m >= 0`. When `total_count` is broadcast with
        `concentration1` and `concentration0`, it defines the distribution as a
        batch of `N1 x ... x Nm` different Beta-Binomial distributions. Its
        components should be equal to integer values.
      concentration1: Positive floating-point `Tensor` indicating mean number of
        successes. Specifically, the expected number of successes is
        `total_count * concentration1 / (concentration1 + concentration0)`.
      concentration0: Positive floating-point `Tensor` indicating mean number of
        failures; see description of `concentration1` for details.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [total_count, concentration1, concentration0], dtype_hint=tf.float32)
      self._total_count = tensor_util.convert_nonref_to_tensor(
          total_count, dtype=dtype, name='total_count')
      self._concentration1 = tensor_util.convert_nonref_to_tensor(
          concentration1, dtype=dtype, name='concentration1')
      self._concentration0 = tensor_util.convert_nonref_to_tensor(
          concentration0, dtype=dtype, name='concentration0')

      super(BetaBinomial, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        total_count=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED),
        concentration1=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration0=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def total_count(self):
    """Number of trials."""
    return self._total_count

  @property
  def concentration1(self):
    """Concentration parameter associated with a `success` outcome."""
    return self._concentration1

  @property
  def concentration0(self):
    """Concentration parameter associated with a `failure` outcome."""
    return self._concentration0

  def _params_list(self):
    return [self._total_count, self._concentration1, self._concentration0]

  def _params_list_as_tensors(self):
    return [tf.convert_to_tensor(p) for p in self._params_list()]

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    gamma1_seed, gamma2_seed, binomial_seed = samplers.split_seed(
        seed, n=3, salt='beta_binomial')

    total_count, concentration1, concentration0 = self._params_list_as_tensors()
    batch_shape = self._batch_shape_tensor(total_count=total_count,
                                           concentration1=concentration1,
                                           concentration0=concentration0)

    expanded_concentration1 = tf.broadcast_to(concentration1, batch_shape)
    expanded_concentration0 = tf.broadcast_to(concentration0, batch_shape)
    # probs = g1 / (g1 + g2)
    # logits = log(probs) - log(1 - probs)
    #        = log(g1 / (g1 + g2)) - log(1 - g1 / (g1 + g2))
    #        = log(g1) - log(g1 + g2) - log(((g1 + g2) - g1) / (g1 + g2))
    #        = log(g1) - log(g1 + g2) - (log(g1 + g2 - g1) - log(g1 + g2))
    #        = log(g1) - log(g1 + g2) - log(g2) + log(g1 + g2))
    #        = log(g1) - log(g2)
    log_gamma1 = gamma_lib.random_gamma(
        shape=[n], concentration=expanded_concentration1, seed=gamma1_seed,
        log_space=True)
    log_gamma2 = gamma_lib.random_gamma(
        shape=[n], concentration=expanded_concentration0, seed=gamma2_seed,
        log_space=True)
    return binomial.Binomial(
        total_count, logits=log_gamma1 - log_gamma2,
        validate_args=self.validate_args).sample(seed=binomial_seed)

  @distribution_util.AppendDocstring(_beta_binomial_sample_note)
  def _log_prob(self, counts):
    n, c1, c0 = self._params_list_as_tensors()
    return (_log_combinations(n, counts)
            + tfp_math.lbeta(c1 + counts, n + c0 - counts)
            - tfp_math.lbeta(c1, c0))

  @distribution_util.AppendDocstring(_beta_binomial_sample_note)
  def _prob(self, counts):
    return tf.exp(self._log_prob(counts))

  def _mean(self):
    n, c1, c0 = self._params_list_as_tensors()
    return n * c1 / (c1 + c0)

  def _variance(self):
    n, c1, c0 = self._params_list_as_tensors()
    c_sum = c1 + c0
    return (n * c1 * c0 * (c_sum + n)) / (c_sum * c_sum * (c_sum + 1))

  def _sample_control_dependencies(self, counts):
    """Check counts for proper values."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(counts))
    assertions.append(
        assert_util.assert_less_equal(
            counts,
            self.total_count,
            message=('Sampled counts must be itemwise less than '
                     'or equal to `total_count` parameter.')))
    return assertions

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    assertions = []

    if is_init != tensor_util.is_ref(self.total_count):
      total_count = tf.convert_to_tensor(self.total_count)
      msg1 = 'Argument `total_count` must be non-negative.'
      msg2 = 'Argument `total_count` cannot contain fractional components.'
      assertions += [
          assert_util.assert_non_negative(total_count, message=msg1),
          distribution_util.assert_integer_form(total_count, message=msg2),
      ]

    for concentration in [self.concentration1, self.concentration0]:
      if is_init != tensor_util.is_ref(concentration):
        assertions.append(
            assert_util.assert_positive(
                concentration,
                message='Concentration parameter must be positive.'))
    return assertions


def _log_combinations(n, k):
  """Computes log(Gamma(n+1) / (Gamma(k+1) * Gamma(n-k+1))."""
  return -tfp_math.lbeta(k + 1, n - k + 1) - tf.math.log(n + 1)
