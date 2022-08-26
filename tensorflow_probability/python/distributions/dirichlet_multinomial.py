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
"""The DirichletMultinomial distribution class."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import generic


__all__ = [
    'DirichletMultinomial',
]


_dirichlet_multinomial_sample_note = """For each batch of counts,
`value = [n_0, ..., n_{K-1}]`, `P[value]` is the probability that after
sampling `self.total_count` draws from this Dirichlet-Multinomial distribution,
the number of draws falling in class `j` is `n_j`. Since this definition is
[exchangeable](https://en.wikipedia.org/wiki/Exchangeable_random_variables);
different sequences have the same counts so the probability includes a
combinatorial coefficient.

Note: `value` must be a non-negative tensor with dtype `self.dtype`, have no
fractional components, and such that
`tf.reduce_sum(value, -1) = self.total_count`. Its shape must be broadcastable
with `self.concentration` and `self.total_count`."""


class DirichletMultinomial(
    distribution.DiscreteDistributionMixin,
    distribution.AutoCompositeTensorDistribution):
  """Dirichlet-Multinomial compound distribution.

  The Dirichlet-Multinomial distribution is parameterized by a (batch of)
  length-`K` `concentration` vectors (`K > 1`) and a `total_count` number of
  trials, i.e., the number of trials per draw from the DirichletMultinomial. It
  is defined over a (batch of) length-`K` vector `counts` such that
  `tf.reduce_sum(counts, -1) = total_count`. The Dirichlet-Multinomial is
  identically the Beta-Binomial distribution when `K = 2`.

  #### Mathematical Details

  The Dirichlet-Multinomial is a distribution over `K`-class counts, i.e., a
  length-`K` vector of non-negative integer `counts = n = [n_0, ..., n_{K-1}]`.

  The probability mass function (pmf) is,

  ```none
  pmf(n; alpha, N) = Beta(alpha + n) / (prod_j n_j!) / Z
  Z = Beta(alpha) / N!
  ```

  where:

  * `concentration = alpha = [alpha_0, ..., alpha_{K-1}]`, `alpha_j > 0`,
  * `total_count = N`, `N` a positive integer,
  * `N!` is `N` factorial, and,
  * `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the
    [multivariate beta function](
    https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function),
    and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  Dirichlet-Multinomial is a [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e., its
  samples are generated as follows.

    1. Choose class probabilities:
       `probs = [p_0,...,p_{K-1}] ~ Dir(concentration)`
    2. Draw integers:
       `counts = [n_0,...,n_{K-1}] ~ Multinomial(total_count, probs)`

  The last `concentration` dimension parameterizes a single
  Dirichlet-Multinomial distribution. When calling distribution functions
  (e.g., `dist.prob(counts)`), `concentration`, `total_count` and `counts` are
  broadcast to the same shape. The last dimension of `counts` corresponds to
  single Dirichlet-Multinomial distributions.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Pitfalls

  The number of classes, `K`, must not exceed:

  - the largest integer representable by `self.dtype`, i.e.,
    `2**(mantissa_bits+1)` (IEE754),
  - the maximum `Tensor` index, i.e., `2**31-1`.

  In other words,

  ```python
  K <= min(2**31-1, {
    tf.float16: 2**11,
    tf.float32: 2**24,
    tf.float64: 2**53 }[param.dtype])
  ```

  Note: This condition is validated only when `self.validate_args = True`.

  #### Examples

  ```python
  alpha = [1., 2., 3.]
  n = 2.
  dist = DirichletMultinomial(n, alpha)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be
  drawn.
  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as alpha.
  counts = [0., 0., 2.]
  dist.prob(counts)  # Shape []

  # alpha will be broadcast to [[1., 2., 3.], [1., 2., 3.]] to match counts.
  counts = [[1., 1., 0.], [1., 0., 1.]]
  dist.prob(counts)  # Shape [2]

  # alpha will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  alpha = [[1., 2., 3.], [4., 5., 6.]]  # Shape [2, 3]
  n = [3., 3.]
  dist = DirichletMultinomial(n, alpha)

  # counts will be broadcast to [[2., 1., 0.], [2., 1., 0.]] to match alpha.
  counts = [2., 1., 0.]
  dist.prob(counts)  # Shape [2]
  ```

  """

  def __init__(self,
               total_count,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='DirichletMultinomial'):
    """Initialize a batch of DirichletMultinomial distributions.

    Args:
      total_count: Non-negative integer-valued tensor, whose dtype is the same
        as `concentration`. The shape is broadcastable to `[N1,..., Nm]` with
        `m >= 0`. Defines this as a batch of `N1 x ... x Nm` different
        Dirichlet multinomial distributions. Its components should be equal to
        integer values.
      concentration: Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm, K]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
        different `K` class Dirichlet multinomial distributions.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
     allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, variance) use the value "`NaN`" to indicate the result is
        undefined. When `False`, an exception is raised if one or more of the
        statistic's batch members are undefined.
     name: Python `str` name prefixed to Ops created by this class.
    """
    # Broadcasting works because:
    # * The broadcasting convention is to prepend dimensions of size [1], and
    #   we use the last dimension for the distribution, whereas
    #   the batch dimensions are the leading dimensions, which forces the
    #   distribution dimension to be defined explicitly (i.e. it cannot be
    #   created automatically by prepending). This forces enough explicitness.
    # * All calls involving `counts` eventually require a broadcast between
    #  `counts` and concentration.
    # * We broadcast explicitly to include the effect of `counts` on
    #   `concentration` for calls that do not involve `counts`.
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([total_count, concentration], tf.float32)
      self._total_count = tensor_util.convert_nonref_to_tensor(
          total_count, dtype=dtype, name='total_count')
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration')

      super(DirichletMultinomial, self).__init__(
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
        concentration=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def total_count(self):
    """Number of trials used to construct a sample."""
    return self._total_count

  @property
  def concentration(self):
    """Concentration parameter; expected prior counts for that coordinate."""
    return self._concentration

  def compute_total_concentration(self):
    """Compute and return the sum of last dim of concentration parameter."""
    with self._name_and_control_scope('compute_total_concentration'):
      return self._compute_total_concentration()

  def _compute_total_concentration(self, concentration=None):
    if concentration is None:
      concentration = tf.convert_to_tensor(self._concentration)
    return tf.reduce_sum(concentration, axis=-1)

  def _event_shape_tensor(self, concentration=None):
    if concentration is None:
      concentration = tf.convert_to_tensor(self.concentration)
    # Event shape depends only on concentration, not total_count.
    return ps.shape(concentration)[-1:]

  def _event_shape(self):
    # Event shape depends only on concentration, not total_count.
    return tensorshape_util.with_rank(self.concentration.shape[-1:], rank=1)

  def _sample_n(self, n, seed=None):
    gamma_seed, multinomial_seed = samplers.split_seed(
        seed, salt='dirichlet_multinomial')

    concentration = tf.convert_to_tensor(self._concentration)
    total_count = tf.convert_to_tensor(self._total_count)

    n_draws = tf.cast(total_count, dtype=tf.int32)
    k = self._event_shape_tensor(concentration)[0]
    alpha = tf.math.multiply(
        tf.ones_like(total_count[..., tf.newaxis]),
        concentration,
        name='alpha')

    unnormalized_logits = gamma_lib.random_gamma(
        shape=[n], concentration=alpha, seed=gamma_seed, log_space=True)
    x = multinomial.draw_sample(
        1, k, unnormalized_logits, n_draws, self.dtype, multinomial_seed)
    final_shape = ps.concat(
        [[n], self._batch_shape_tensor(concentration=concentration,
                                       total_count=total_count), [k]], 0)
    return tf.reshape(x, final_shape)

  @distribution_util.AppendDocstring(_dirichlet_multinomial_sample_note)
  def _log_prob(self, counts):
    concentration = tf.convert_to_tensor(self.concentration)
    ordered_prob = (
        tf.math.lbeta(concentration + counts) -
        tf.math.lbeta(concentration))
    return ordered_prob + generic.log_combinations(
        self.total_count, counts)

  @distribution_util.AppendDocstring(_dirichlet_multinomial_sample_note)
  def _prob(self, counts):
    return tf.exp(self._log_prob(counts))

  def _mean(self, total_count=None, concentration=None):
    if total_count is None:
      total_count = tf.convert_to_tensor(self._total_count)
    if concentration is None:
      concentration = tf.convert_to_tensor(self._concentration)

    total_concentration = self._compute_total_concentration(concentration)
    scaled_concentration = (
        concentration / total_concentration[..., tf.newaxis])
    return total_count[..., tf.newaxis] * scaled_concentration

  @distribution_util.AppendDocstring(
      """The covariance for each batch member is defined as the following:

      ```none
      Var(X_j) = n * alpha_j / alpha_0 * (1 - alpha_j / alpha_0) *
      (n + alpha_0) / (1 + alpha_0)
      ```

      where `concentration = alpha` and
      `total_concentration = alpha_0 = sum_j alpha_j`.

      The covariance between elements in a batch is defined as:

      ```none
      Cov(X_i, X_j) = -n * alpha_i * alpha_j / alpha_0 ** 2 *
      (n + alpha_0) / (1 + alpha_0)
      ```
      """)
  def _covariance(self):
    total_count = tf.convert_to_tensor(self._total_count)
    concentration = tf.convert_to_tensor(self._concentration)

    scale = self._variance_scale_term(total_count, concentration)
    x = scale * self._mean(total_count, concentration)

    return tf.linalg.set_diag(
        -tf.matmul(x[..., tf.newaxis], x[..., tf.newaxis, :]),  # outer prod
        self._variance(total_count, concentration))

  def _variance(self, total_count=None, concentration=None):
    if total_count is None:
      total_count = tf.convert_to_tensor(self._total_count)
    if concentration is None:
      concentration = tf.convert_to_tensor(self._concentration)

    scale = self._variance_scale_term(total_count, concentration)
    x = scale * self._mean(total_count, concentration)
    return x * (total_count[..., tf.newaxis] * scale - x)

  def _variance_scale_term(self, total_count=None, concentration=None):
    """Helper to `_covariance` and `_variance` which computes a shared scale."""
    if total_count is None:
      total_count = tf.convert_to_tensor(self._total_count)
    if concentration is None:
      concentration = tf.convert_to_tensor(self._concentration)

    # Expand back the last dim so the shape of _variance_scale_term matches the
    # shape of self.concentration.
    c0 = self._compute_total_concentration(concentration)[..., tf.newaxis]
    return tf.sqrt((1. + c0 / total_count[..., tf.newaxis]) / (1. + c0))

  def _default_event_space_bijector(self):
    return

  def _sample_control_dependencies(self, x):
    """Checks the validity of a sample."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    assertions.append(assert_util.assert_equal(
        self.total_count,
        tf.reduce_sum(x, axis=-1),
        message='counts last-dimension must sum to `self.total_count`'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init and self.validate_args:
      # assert_categorical_event_shape handles both the static and dynamic case.
      assertions.extend(
          distribution_util.assert_categorical_event_shape(self._concentration))

    if is_init != tensor_util.is_ref(self._total_count):
      if self.validate_args:
        total_count = tf.convert_to_tensor(self._total_count)
        assertions.append(
            distribution_util.assert_casting_closed(
                total_count, target_dtype=tf.int32,
                message='total_count cannot contain fractional components.'))
        assertions.append(assert_util.assert_non_negative(
            total_count, message='total_count must be non-negative'))

    if is_init != tensor_util.is_ref(self._concentration):
      if self.validate_args:
        assertions.append(
            assert_util.assert_positive(
                self._concentration,
                message='Concentration parameter must be positive.'))
    return assertions
