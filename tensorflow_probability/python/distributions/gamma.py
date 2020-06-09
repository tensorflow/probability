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
"""The Gamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import batched_rejection_sampler as brs
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'Gamma',
]


class Gamma(distribution.Distribution):
  """Gamma distribution.

  The Gamma distribution is defined over positive real numbers using
  parameters `concentration` (aka "alpha") and `rate` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta, x > 0) = x**(alpha - 1) exp(-x beta) / Z
  Z = Gamma(alpha) beta**(-alpha)
  ```

  where:

  * `concentration = alpha`, `alpha > 0`,
  * `rate = beta`, `beta > 0`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta x) / Gamma(alpha)
  ```

  where `GammaInc` is the [lower incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  The parameters can be intuited via their relationship to mean and stddev,

  ```none
  concentration = alpha = (mean / stddev)**2
  rate = beta = mean / stddev**2 = concentration / mean
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: The samples of this distribution are always non-negative. However,
  the samples that are smaller than `np.finfo(dtype).tiny` are rounded
  to this value, so it appears more often than it should.
  This should only be noticeable when the `concentration` is very small, or the
  `rate` is very large. See note in `tf.random.gamma` docstring.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.Gamma(concentration=3.0, rate=2.0)
  dist2 = tfd.Gamma(concentration=[3.0, 4.0], rate=[2.0, 3.0])
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  concentration = tf.constant(3.0)
  rate = tf.constant(2.0)
  dist = tfd.Gamma(concentration, rate)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [concentration, rate])
  ```

  """

  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name='Gamma'):
    """Construct Gamma with `concentration` and `rate` parameters.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g. `concentration + rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [concentration, rate], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, dtype=dtype, name='rate')

      super(Gamma, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('concentration', 'rate'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=0, rate=0)

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  def _batch_shape_tensor(self, concentration=None, rate=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(
            self.concentration if concentration is None else concentration),
        prefer_static.shape(self.rate if rate is None else rate))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.concentration.shape,
        self.rate.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    """Gamma sampler.

    Rather than use `tf.random.gamma` (which is as of February 2020 implemented
    in C++ for CPU only), we implement our own gamma sampler in Python, using
    `batched_las_vegas_algorithm` as a substrate. This has the advantage that
    our sampler is XLA compilable.

    If sampling becomes a bottleneck on CPU, one way to gain speed would be to
    consider switching back to the C++ sampler.

    Args:
      n: Number of samples to draw.
      seed: (optional) The random seed.

    Returns:
      n samples from the gamma distribution.
    """
    n = tf.convert_to_tensor(n, name='shape', dtype=tf.int32)
    alpha = tf.convert_to_tensor(self.concentration, name='alpha')
    beta = tf.convert_to_tensor(self.rate, name='beta')
    broadcast_shape = prefer_static.broadcast_shape(
        prefer_static.shape(alpha), prefer_static.shape(beta))
    result_shape = tf.concat([[n], broadcast_shape], axis=0)

    return random_gamma(result_shape, alpha, beta, seed=seed)

  def _log_prob(self, x, concentration=None, rate=None):
    concentration = tf.convert_to_tensor(
        self.concentration if concentration is None else concentration)
    rate = tf.convert_to_tensor(self.rate if rate is None else rate)
    log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
    log_normalization = (tf.math.lgamma(concentration) -
                         concentration * tf.math.log(rate))
    return log_unnormalized_prob - log_normalization

  def _cdf(self, x):
    # Note that igamma returns the regularized incomplete gamma function,
    # which is what we want for the CDF.
    return tf.math.igamma(self.concentration, self.rate * x)

  def _entropy(self):
    concentration = tf.convert_to_tensor(self.concentration)
    return (concentration - tf.math.log(self.rate) +
            tf.math.lgamma(concentration) +
            ((1. - concentration) * tf.math.digamma(concentration)))

  def _mean(self):
    return self.concentration / self.rate

  def _variance(self):
    return self.concentration / tf.square(self.rate)

  def _stddev(self):
    return tf.sqrt(self.concentration) / self.rate

  @distribution_util.AppendDocstring(
      """The mode of a gamma distribution is `(shape - 1) / rate` when
      `shape > 1`, and `NaN` otherwise. If `self.allow_nan_stats` is `False`,
      an exception will be raised rather than returning `NaN`.""")
  def _mode(self):
    concentration = tf.convert_to_tensor(self.concentration)
    rate = tf.convert_to_tensor(self.rate)
    mode = (concentration - 1.) / rate
    if self.allow_nan_stats:
      assertions = []
    else:
      assertions = [assert_util.assert_less(
          tf.ones([], self.dtype), concentration,
          message='Mode not defined when any concentration <= 1.')]
    with tf.control_dependencies(assertions):
      return tf.where(
          concentration > 1.,
          mode,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    if is_init != tensor_util.is_ref(self.rate):
      assertions.append(assert_util.assert_positive(
          self.rate,
          message='Argument `rate` must be positive.'))
    return assertions


@kullback_leibler.RegisterKL(Gamma, Gamma)
def _kl_gamma_gamma(g0, g1, name=None):
  """Calculate the batched KL divergence KL(g0 || g1) with g0 and g1 Gamma.

  Args:
    g0: Instance of a `Gamma` distribution object.
    g1: Instance of a `Gamma` distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_gamma_gamma'`).

  Returns:
    kl_gamma_gamma: `Tensor`. The batchwise KL(g0 || g1).
  """
  with tf.name_scope(name or 'kl_gamma_gamma'):
    # Result from:
    #   http://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps
    # For derivation see:
    #   http://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions   pylint: disable=line-too-long
    g0_concentration = tf.convert_to_tensor(g0.concentration)
    g0_rate = tf.convert_to_tensor(g0.rate)
    g1_concentration = tf.convert_to_tensor(g1.concentration)
    g1_rate = tf.convert_to_tensor(g1.rate)
    return (((g0_concentration - g1_concentration) *
             tf.math.digamma(g0_concentration)) +
            tf.math.lgamma(g1_concentration) -
            tf.math.lgamma(g0_concentration) +
            g1_concentration * tf.math.log(g0_rate) -
            g1_concentration * tf.math.log(g1_rate) + g0_concentration *
            (g1_rate / g0_rate - 1.))


def random_gamma(
    sample_shape, alpha, beta, internal_dtype=tf.float64, seed=None):
  """Samples from the gamma distribution.

  The sampling algorithm is rejection sampling [1], and pathwise gradients with
  respect to alpha are computed via implicit differentiation [2].

  Args:
    sample_shape: The output sample shape. Must broadcast with both
      `alpha` and `beta`.
    alpha: Floating point tensor, the alpha params of the
      distribution(s). Must contain only positive values. Must broadcast with
      `beta`.
    beta: Floating point tensor, the inverse scale params of the
      distribution(s). Must contain only positive values. Must broadcast with
      `alpha`.
    internal_dtype: dtype to use for internal computations.
    seed: (optional) The random seed.

  Returns:
    Differentiable samples from the gamma distribution.

  #### References

  [1] George Marsaglia and Wai Wan Tsang. A simple method for generating Gamma
      variables. ACM Transactions on Mathematical Software, 2000.

  [2] Michael Figurnov, Shakir Mohamed, and Andriy Mnih. Implicit
      Reparameterization Gradients. Neural Information Processing Systems, 2018.
  """
  generate_and_test_samples_seed, alpha_fix_seed = samplers.split_seed(
      seed, salt='random_gamma')
  output_dtype = dtype_util.common_dtype([alpha, beta], dtype_hint=tf.float32)

  @tf.custom_gradient
  def rejection_sample(alpha):
    """Gamma rejection sampler."""
    # Note that alpha here already has a shape that is broadcast with beta.
    cast_alpha = tf.cast(alpha, internal_dtype)

    good_params_mask = (alpha > 0.)
    # When replacing NaN values, use 100. for alpha, since that leads to a
    # high-likelihood of the rejection sampler accepting on the first pass.
    safe_alpha = tf.where(good_params_mask, cast_alpha, 100.)

    modified_safe_alpha = tf.where(
        safe_alpha < 1., safe_alpha + 1., safe_alpha)

    one_third = tf.constant(1. / 3, dtype=internal_dtype)
    d = modified_safe_alpha - one_third
    c = one_third / tf.sqrt(d)

    def generate_and_test_samples(seed):
      """Generate and test samples."""
      v_seed, u_seed = samplers.split_seed(seed)

      def generate_positive_v():
        """Generate positive v."""
        def _inner(seed):
          x = samplers.normal(
              sample_shape, dtype=internal_dtype, seed=seed)
          # This implicitly broadcasts alpha up to sample shape.
          v = 1 + c * x
          return (x, v), v > 0.

        # Note: It should be possible to remove this 'inner' call to
        # `batched_las_vegas_algorithm` and merge the v > 0 check into the
        # overall check for a good sample. This would lead to a slightly simpler
        # implementation; it is unclear whether it would be faster. We include
        # the inner loop so this implementation is more easily comparable to
        # Ref. [1] and other implementations.
        return brs.batched_las_vegas_algorithm(_inner, v_seed)[0]

      (x, v) = generate_positive_v()
      x2 = x * x
      v3 = v * v * v
      u = samplers.uniform(
          sample_shape, dtype=internal_dtype, seed=u_seed)

      # In [1], the suggestion is to first check u < 1 - 0.331 * x2 * x2, and to
      # run the check below only if it fails, in order to avoid the relatively
      # expensive logarithm calls. Our algorithm operates in batch mode: we will
      # have to compute or not compute the logarithms for the entire batch, and
      # as the batch gets larger, the odds we compute it grow. Therefore we
      # don't bother with the "cheap" check.
      good_sample_mask = (
          tf.math.log(u) < x2 / 2. + d * (1 - v3 + tf.math.log(v3)))

      return v3, good_sample_mask

    samples = brs.batched_las_vegas_algorithm(
        generate_and_test_samples, seed=generate_and_test_samples_seed)[0]

    samples = samples * d

    one = tf.constant(1., dtype=internal_dtype)

    alpha_lt_one_fix = tf.where(
        safe_alpha < 1.,
        tf.math.pow(
            samplers.uniform(
                sample_shape, dtype=internal_dtype, seed=alpha_fix_seed),
            one / safe_alpha),
        one)
    samples = samples * alpha_lt_one_fix
    samples = tf.where(good_params_mask, samples, np.nan)

    output_type_samples = tf.cast(samples, output_dtype)

    # We use `tf.where` instead of `tf.maximum` because we need to allow for
    # `samples` to be `nan`, but `tf.maximum(nan, x) == x`.
    output_type_samples = tf.where(
        output_type_samples == 0,
        np.finfo(dtype_util.as_numpy_dtype(output_type_samples.dtype)).tiny,
        output_type_samples)

    def grad(dy):
      """The gradient of the normalized (beta=1) gamma samples w.r.t alpha."""
      # Recall that cast_alpha has shape broadcast with beta, and samples and dy
      # have shape sample_shape (which further expands the alpha-beta broadcast
      # shape on the left).
      cast_dy = tf.cast(dy, internal_dtype)
      partial_alpha = tf.raw_ops.RandomGammaGrad(
          alpha=cast_alpha, sample=samples)
      grad = tf.cast(
          tf.math.reduce_sum(
              cast_dy * partial_alpha,
              axis=tf.range(tf.rank(partial_alpha) - tf.rank(alpha))),
          output_dtype)
      return grad

    return output_type_samples, grad  # rejection_sample

  broadcast_alpha_shape = prefer_static.broadcast_shape(
      prefer_static.shape(alpha), prefer_static.shape(beta))
  broadcast_alpha = tf.broadcast_to(alpha, broadcast_alpha_shape)
  alpha_samples = rejection_sample(broadcast_alpha)

  corrected_beta = tf.where(beta > 0., beta, np.nan)
  return alpha_samples / corrected_beta
