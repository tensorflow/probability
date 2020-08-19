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
"""The Dirichlet distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'Dirichlet',
]


_dirichlet_sample_note = """Note: `value` must be a non-negative tensor with
dtype `self.dtype` and be in the `(self.event_shape() - 1)`-simplex, i.e.,
`tf.reduce_sum(value, -1) = 1`. It must have a shape compatible with
`self.batch_shape() + self.event_shape()`."""


class Dirichlet(distribution.Distribution):
  """Dirichlet distribution.

  The Dirichlet distribution is defined over the
  [`(k-1)`-simplex](https://en.wikipedia.org/wiki/Simplex) using a positive,
  length-`k` vector `concentration` (`k > 1`). The Dirichlet is identically the
  Beta distribution when `k = 2`.

  #### Mathematical Details

  The Dirichlet is a distribution over the open `(k-1)`-simplex, i.e.,

  ```none
  S^{k-1} = { (x_0, ..., x_{k-1}) in R^k : sum_j x_j = 1 and all_j x_j > 0 }.
  ```

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha) = prod_j x_j**(alpha_j - 1) / Z
  Z = prod_j Gamma(alpha_j) / Gamma(sum_j alpha_j)
  ```

  where:

  * `x in S^{k-1}`, i.e., the `(k-1)`-simplex,
  * `concentration = alpha = [alpha_0, ..., alpha_{k-1}]`, `alpha_j > 0`,
  * `Z` is the normalization constant aka the [multivariate beta function](
    https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function),
    and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The `concentration` represents mean total counts of class occurrence, i.e.,

  ```none
  concentration = alpha = mean * total_concentration
  ```

  where `mean` in `S^{k-1}` and `total_concentration` is a positive real number
  representing a mean total count.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: Some components of the samples can be zero due to finite precision.
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

  # Create a single trivariate Dirichlet, with the 3rd class being three times
  # more frequent than the first. I.e., batch_shape=[], event_shape=[3].
  alpha = [1., 2, 3]
  dist = tfd.Dirichlet(alpha)

  dist.sample([4, 5])  # shape: [4, 5, 3]

  # x has one sample, one batch, three classes:
  x = [.2, .3, .5]   # shape: [3]
  dist.prob(x)       # shape: []

  # x has two samples from one batch:
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  dist.prob(x)         # shape: [2]

  # alpha will be broadcast to shape [5, 7, 3] to match x.
  x = [[...]]   # shape: [5, 7, 3]
  dist.prob(x)  # shape: [5, 7]
  ```

  ```python
  # Create batch_shape=[2], event_shape=[3]:
  alpha = [[1., 2, 3],
           [4, 5, 6]]   # shape: [2, 3]
  dist = tfd.Dirichlet(alpha)

  dist.sample([4, 5])  # shape: [4, 5, 2, 3]

  x = [.2, .3, .5]
  # x will be broadcast as [[.2, .3, .5],
  #                         [.2, .3, .5]],
  # thus matching batch_shape [2, 3].
  dist.prob(x)         # shape: [2]
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  alpha = tf.constant([1.0, 2.0, 3.0])
  dist = tfd.Dirichlet(alpha)
  samples = dist.sample(5)  # Shape [5, 3]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, alpha)
  ```

  """

  def __init__(self,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='Dirichlet'):
    """Initialize a batch of Dirichlet distributions.

    Args:
      concentration: Positive floating-point `Tensor` indicating mean number
        of class occurrences; aka "alpha". Implies `self.dtype`, and
        `self.batch_shape`, `self.event_shape`, i.e., if
        `concentration.shape = [N1, N2, ..., Nm, k]` then
        `batch_shape = [N1, N2, ..., Nm]` and
        `event_shape = [k]`.
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
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      super(Dirichlet, self).__init__(
          dtype=self._concentration.dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=1)

  @property
  def concentration(self):
    """Concentration parameter; expected counts for that coordinate."""
    return self._concentration

  def _batch_shape_tensor(self):
    # NOTE: In TF1, tf.shape(x) can call `tf.convert_to_tensor(x)` **twice**,
    # so we pre-emptively convert-to-tensor.
    concentration = tf.convert_to_tensor(self.concentration)
    return ps.shape(concentration)[:-1]

  def _batch_shape(self):
    return self.concentration.shape[:-1]

  def _event_shape_tensor(self):
    # NOTE: In TF1, tf.shape(x) can call `tf.convert_to_tensor(x)` **twice**,
    # so we pre-emptively convert-to-tensor.
    concentration = tf.convert_to_tensor(self.concentration)
    return ps.shape(concentration)[-1:]

  def _event_shape(self):
    return tensorshape_util.with_rank(self.concentration.shape[-1:], rank=1)

  def _sample_n(self, n, seed=None):
    # We use the log-space gamma sampler to avoid the bump-up-from-0 correction,
    # and to apply the concentration < 1 recurrence in log-space. This improves
    # accuracy for small concentrations.
    log_gamma_sample = gamma_lib.random_gamma(
        shape=[n], concentration=self.concentration, seed=seed, log_space=True)
    return tf.math.exp(
        log_gamma_sample -
        tf.math.reduce_logsumexp(log_gamma_sample, axis=-1, keepdims=True))

  @distribution_util.AppendDocstring(_dirichlet_sample_note)
  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    return (tf.reduce_sum(tf.math.xlogy(concentration - 1., x), axis=-1) -
            tf.math.lbeta(concentration))

  @distribution_util.AppendDocstring(_dirichlet_sample_note)
  def _prob(self, x):
    return tf.exp(self._log_prob(x))

  def _entropy(self):
    concentration = tf.convert_to_tensor(self.concentration)
    k = tf.cast(tf.shape(concentration)[-1], self.dtype)
    total_concentration = tf.reduce_sum(concentration, axis=-1)
    return (tf.math.lbeta(concentration) +
            ((total_concentration - k) * tf.math.digamma(total_concentration)) -
            tf.reduce_sum((concentration - 1.) * tf.math.digamma(concentration),
                          axis=-1))

  def _mean(self):
    concentration = tf.convert_to_tensor(self.concentration)
    total_concentration = tf.reduce_sum(concentration, axis=-1, keepdims=True)
    return concentration / total_concentration

  def _covariance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    total_concentration = tf.reduce_sum(concentration, axis=-1, keepdims=True)
    mean = concentration / total_concentration
    scale = tf.math.rsqrt(1. + total_concentration)
    x = scale * mean
    variance = x * (scale - x)
    return tf.linalg.set_diag(
        tf.matmul(-x[..., tf.newaxis], x[..., tf.newaxis, :]),
        variance)

  def _variance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    total_concentration = tf.reduce_sum(concentration, axis=-1, keepdims=True)
    mean = concentration / total_concentration
    scale = tf.math.rsqrt(1. + total_concentration)
    x = scale * mean
    return x * (scale - x)

  @distribution_util.AppendDocstring(
      """Note: The mode is undefined when any `concentration <= 1`. If
      `self.allow_nan_stats` is `True`, `NaN` is used for undefined modes. If
      `self.allow_nan_stats` is `False` an exception is raised when one or more
      modes are undefined.""")
  def _mode(self):
    concentration = tf.convert_to_tensor(self.concentration)
    k = tf.cast(tf.shape(concentration)[-1], self.dtype)
    total_concentration = tf.reduce_sum(concentration, axis=-1)
    mode = (concentration - 1.) / (total_concentration[..., tf.newaxis] - k)
    if self.allow_nan_stats:
      return tf.where(
          tf.reduce_all(concentration > 1., axis=-1, keepdims=True),
          mode,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    assertions = [
        assert_util.assert_less(
            tf.ones([], self.dtype),
            concentration,
            message='Mode undefined when any concentration <= 1')
    ]
    with tf.control_dependencies(assertions):
      return tf.identity(mode)

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return softmax_centered_bijector.SoftmaxCentered(
        validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    """Checks the validity of a sample."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Samples must be non-negative.'))
    assertions.append(assert_util.assert_near(
        tf.ones([], dtype=self.dtype),
        tf.reduce_sum(x, axis=-1),
        message='Sample last-dimension must sum to `1`.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    """Checks the validity of the concentration parameter."""
    assertions = []

    # In init, we can always build shape and dtype checks because
    # we assume shape doesn't change for Variable backed args.
    if is_init:
      if not dtype_util.is_floating(self.concentration.dtype):
        raise TypeError('Argument `concentration` must be float type.')

      msg = 'Argument `concentration` must have rank at least 1.'
      ndims = tensorshape_util.rank(self.concentration.shape)
      if ndims is not None:
        if ndims < 1:
          raise ValueError(msg)
      elif self.validate_args:
        assertions.append(assert_util.assert_rank_at_least(
            self.concentration, 1, message=msg))

      msg = 'Argument `concentration` must have `event_size` at least 2.'
      event_size = tf.compat.dimension_value(self.concentration.shape[-1])
      if event_size is not None:
        if event_size < 2:
          raise ValueError(msg)
      elif self.validate_args:
        assertions.append(assert_util.assert_less(
            1,
            tf.shape(self.concentration)[-1],
            message=msg))

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))

    return assertions


@kullback_leibler.RegisterKL(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(d1, d2, name=None):
  """Batchwise KL divergence KL(d1 || d2) with d1 and d2 Dirichlet.

  Args:
    d1: instance of a Dirichlet distribution object.
    d2: instance of a Dirichlet distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_dirichlet_dirichlet'`).

  Returns:
    kl_div: Batchwise KL(d1 || d2)
  """
  with tf.name_scope(name or 'kl_dirichlet_dirichlet'):
    # The KL between Dirichlet distributions can be derived as follows. We have
    #
    #   Dir(x; a) = 1 / B(a) * prod_i[x[i]^(a[i] - 1)]
    #
    # where B(a) is the multivariate Beta function:
    #
    #   B(a) = Gamma(a[1]) * ... * Gamma(a[n]) / Gamma(a[1] + ... + a[n])
    #
    # The KL is
    #
    #   KL(Dir(x; a), Dir(x; b)) = E_Dir(x; a){log(Dir(x; a) / Dir(x; b))}
    #
    # so we'll need to know the log density of the Dirichlet. This is
    #
    #   log(Dir(x; a)) = sum_i[(a[i] - 1) log(x[i])] - log B(a)
    #
    # The only term that matters for the expectations is the log(x[i]). To
    # compute the expectation of this term over the Dirichlet density, we can
    # use the following facts about the Dirichlet in exponential family form:
    #   1. log(x[i]) is a sufficient statistic
    #   2. expected sufficient statistics (of any exp family distribution) are
    #      equal to derivatives of the log normalizer with respect to
    #      corresponding natural parameters: E{T[i](x)} = dA/d(eta[i])
    #
    # To proceed, we can rewrite the Dirichlet density in exponential family
    # form as follows:
    #
    #   Dir(x; a) = exp{eta(a) . T(x) - A(a)}
    #
    # where '.' is the dot product of vectors eta and T, and A is a scalar:
    #
    #   eta[i](a) = a[i] - 1
    #     T[i](x) = log(x[i])
    #        A(a) = log B(a)
    #
    # Now, we can use fact (2) above to write
    #
    #   E_Dir(x; a)[log(x[i])]
    #       = dA(a) / da[i]
    #       = d/da[i] log B(a)
    #       = d/da[i] (sum_j lgamma(a[j])) - lgamma(sum_j a[j])
    #       = digamma(a[i])) - digamma(sum_j a[j])
    #
    # Putting it all together, we have
    #
    # KL[Dir(x; a) || Dir(x; b)]
    #     = E_Dir(x; a){log(Dir(x; a) / Dir(x; b)}
    #     = E_Dir(x; a){sum_i[(a[i] - b[i]) log(x[i])} - (lbeta(a) - lbeta(b))
    #     = sum_i[(a[i] - b[i]) * E_Dir(x; a){log(x[i])}] - lbeta(a) + lbeta(b)
    #     = sum_i[(a[i] - b[i]) * (digamma(a[i]) - digamma(sum_j a[j]))]
    #          - lbeta(a) + lbeta(b))

    concentration1 = tf.convert_to_tensor(d1.concentration)
    concentration2 = tf.convert_to_tensor(d2.concentration)
    digamma_sum_d1 = tf.math.digamma(
        tf.reduce_sum(concentration1, axis=-1, keepdims=True))
    digamma_diff = tf.math.digamma(concentration1) - digamma_sum_d1
    concentration_diff = concentration1 - concentration2

    return (
        tf.reduce_sum(concentration_diff * digamma_diff, axis=-1) -
        tf.math.lbeta(concentration1) + tf.math.lbeta(concentration2))
