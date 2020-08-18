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
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import implementation_selection
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

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
               rate=None,
               log_rate=None,
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
        distribution(s). Must contain only positive values. Mutually exclusive
        with `log_rate`.
      log_rate: Floating point tensor, natural logarithm of the inverse scale
        params of the distribution(s). Mutually exclusive with `rate`.
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
    if (rate is None) == (log_rate is None):
      raise ValueError('Exactly one of `rate` or `log_rate` must be specified.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [concentration, rate, log_rate], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, dtype=dtype, name='rate')
      self._log_rate = tensor_util.convert_nonref_to_tensor(
          log_rate, dtype=dtype, name='log_rate')

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
        zip(('concentration', 'log_rate'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=0, rate=0, log_rate=0)

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  @property
  def log_rate(self):
    """Log-rate parameter."""
    return self._log_rate

  def _batch_shape_tensor(self, concentration=None, rate=None):
    return ps.broadcast_shape(
        ps.shape(
            self.concentration if concentration is None else concentration),
        _shape_or_scalar(self.rate, self.log_rate))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.concentration.shape,
        _tensorshape_or_scalar(self.rate, self.log_rate))

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _rate_parameter(self):
    if self.rate is None:
      return tf.math.exp(self.log_rate)
    return tf.convert_to_tensor(self.rate)

  def _log_rate_parameter(self):
    if self.log_rate is None:
      return tf.math.log(self.rate)
    return tf.convert_to_tensor(self.log_rate)

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    seed = samplers.sanitize_seed(seed, salt='gamma')

    return random_gamma(
        shape=ps.convert_to_shape_tensor([n]),
        concentration=tf.convert_to_tensor(self.concentration),
        rate=None if self.rate is None else tf.convert_to_tensor(self.rate),
        log_rate=(None if self.log_rate is None else
                  tf.convert_to_tensor(self.log_rate)),
        seed=seed)

  def _log_prob(self, x, rate=None):
    concentration = tf.convert_to_tensor(self.concentration)
    if rate is not None:
      rate = tf.convert_to_tensor(rate)
      log_rate = tf.math.log(rate)
    elif self.rate is None:
      log_rate = tf.convert_to_tensor(self.log_rate)
      rate = tf.math.exp(log_rate)
    else:
      rate = tf.convert_to_tensor(self.rate)
      log_rate = tf.math.log(rate)
    log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
    log_normalization = tf.math.lgamma(concentration) - concentration * log_rate
    return log_unnormalized_prob - log_normalization

  def _cdf(self, x):
    # Note that igamma returns the regularized incomplete gamma function,
    # which is what we want for the CDF.
    return tf.math.igamma(self.concentration, self._rate_parameter() * x)

  def _entropy(self):
    concentration = tf.convert_to_tensor(self.concentration)
    log_rate = self._log_rate_parameter()
    return (concentration - log_rate + tf.math.lgamma(concentration) +
            ((1. - concentration) * tf.math.digamma(concentration)))

  def _mean(self):
    return self.concentration / self._rate_parameter()

  def _variance(self):
    rate_sq = (tf.math.exp(self.log_rate * 2) if self.rate is None else
               tf.square(self.rate))
    return self.concentration / rate_sq

  def _stddev(self):
    return tf.sqrt(self.concentration) / self._rate_parameter()

  @distribution_util.AppendDocstring(
      """The mode of a gamma distribution is `(shape - 1) / rate` when
      `shape > 1`, and `NaN` otherwise. If `self.allow_nan_stats` is `False`,
      an exception will be raised rather than returning `NaN`.""")
  def _mode(self):
    concentration = tf.convert_to_tensor(self.concentration)
    mode = (concentration - 1.) / self._rate_parameter()
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
    if self.rate is not None and is_init != tensor_util.is_ref(self.rate):
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
    g0_log_rate = g0._log_rate_parameter()  # pylint: disable=protected-access
    g1_concentration = tf.convert_to_tensor(g1.concentration)
    g1_log_rate = g1._log_rate_parameter()  # pylint: disable=protected-access
    return (((g0_concentration - g1_concentration) *
             tf.math.digamma(g0_concentration)) +
            tf.math.lgamma(g1_concentration) -
            tf.math.lgamma(g0_concentration) +
            g1_concentration * g0_log_rate -
            g1_concentration * g1_log_rate +
            g0_concentration * tf.math.expm1(g1_log_rate - g0_log_rate))


def _shape_or_scalar(v0, v1):
  if v0 is not None:
    return ps.shape(v0)
  if v1 is not None:
    return ps.shape(v1)
  return ps.convert_to_shape_tensor([], dtype=tf.int32)


def _tensorshape_or_scalar(v0, v1):
  if v0 is not None:
    if not hasattr(v0, 'shape'):
      v0 = tf.convert_to_tensor(v0)
    return v0.shape
  if v1 is not None:
    if not hasattr(v1, 'shape'):
      v1 = tf.convert_to_tensor(v1)
    return v1.shape
  return tf.TensorShape([])


def _random_gamma_cpu(
    shape, concentration, rate=None, log_rate=None, seed=None, log_space=False):
  """Sample using *fast* `tf.random.stateless_gamma`."""
  bad_concentration = (concentration <= 0.) | tf.math.is_nan(concentration)
  safe_concentration = tf.where(
      bad_concentration,
      dtype_util.as_numpy_dtype(concentration.dtype)(100.), concentration)

  if rate is None and log_rate is None:
    rate = tf.ones([], concentration.dtype)
    log_rate = tf.zeros([], concentration.dtype)

  if log_space:
    # The underlying gamma sampler uses a recurrence for conc < 1.  When
    # a ~ gamma(conc + 1) and x ~ uniform(0, 1), we have
    #   b = a * x ** (1/conc) ~ gamma(conc)
    # Given that we want log(b) anyway, it's more accurate to just ask the
    # sampler for a (by passing conc + 1 to it in the first place) and
    # do the correction in log-space below.
    orig_safe_concentration = safe_concentration
    safe_concentration = tf.where(
        orig_safe_concentration < 1,
        orig_safe_concentration + 1.,
        orig_safe_concentration)
    seed, conc_fix_seed = samplers.split_seed(seed)
    log_rate = tf.math.log(rate) if log_rate is None else log_rate
    rate = tf.ones_like(log_rate)  # Do the division later in log-space.

  if rate is None:
    rate = tf.math.exp(log_rate)

  bad_rate = (rate <= 0.) | tf.math.is_nan(rate)
  safe_rate = tf.where(
      bad_rate,
      dtype_util.as_numpy_dtype(concentration.dtype)(100.), rate)
  samples = tf.random.stateless_gamma(
      shape=shape, seed=seed, alpha=safe_concentration,
      beta=safe_rate, dtype=concentration.dtype)

  if log_space:
    # Apply the concentration < 1 recurrence here, in log-space.
    samples = tf.math.log(samples)
    conc_fix_unif = samplers.uniform(  # in [0, 1)
        shape, dtype=samples.dtype, seed=conc_fix_seed)

    conc_lt_one_fix = tf.where(
        orig_safe_concentration < 1,
        # Why do we use log1p(-x)? x is in [0, 1) and log(0) = -inf, is bad.
        # x ~ U(0,1) => 1-x ~ U(0,1)
        # But at the boundary, 1-x in (0, 1]. Good.
        # So we can take log(unif(0,1)) safely as log(1-unif(0,1)).
        # log1p(-0) = 0, and log1p(-almost_one) = -not_quite_inf. Good.
        tf.math.log1p(-conc_fix_unif) / orig_safe_concentration,
        tf.zeros((), dtype=samples.dtype))
    samples += (conc_lt_one_fix - log_rate)

  return tf.where(
      bad_rate | bad_concentration,
      dtype_util.as_numpy_dtype(concentration.dtype)(np.nan), samples)


def _random_gamma_noncpu(
    shape, concentration, rate=None, log_rate=None, seed=None, log_space=False):
  """Sample using XLA-friendly python-based rejection sampler."""
  return random_gamma_rejection(
      shape, concentration, rate, log_rate, seed, log_space)


# tf.function required to access Grappler's implementation_selector.
@implementation_selection.never_runs_functions_eagerly
# TODO(b/163029794): Shape relaxation breaks XLA.
@tf.function(autograph=False, experimental_relax_shapes=False)
def _random_gamma_no_gradient(
    shape, concentration, rate, log_rate, seed, log_space):
  """Sample a gamma, CPU specialized to stateless_gamma.

  Args:
    shape: Sample shape.
    concentration: Concentration of gamma distribution.
    rate: Rate parameter of gamma distribution.
    log_rate: Log-rate parameter of gamma distribution.
    seed: int or Tensor seed.
    log_space: If `True`, draw log-of-gamma samples.

  Returns:
    samples: Samples from gamma distributions.
  """
  seed = samplers.sanitize_seed(seed)

  sampler_impl = implementation_selection.implementation_selecting(
      fn_name='gamma',
      default_fn=_random_gamma_noncpu,
      cpu_fn=_random_gamma_cpu)
  return sampler_impl(
      shape=shape, concentration=concentration, rate=rate, log_rate=log_rate,
      seed=seed, log_space=log_space)


def _compute_partials(samples, concentration, rate, log_rate, log_space):
  """Shared partials between forward and reverse mode."""
  if log_space:
    # The function is log(gamma_sample(conc, rate, log_rate)).
    # So:
    # d log(gamma_sample(..)) / d conc
    #   = (d gamma_sample(..) / d conc) / gamma_sample(..)
    # d log(gamma_sample(..)) / d rate
    #   = d log(gamma_sample(.., rate=1) / rate) / d rate
    #   = d (log(gamma_sample(.., rate=1)) - log(rate)) / d rate
    #   = d (-log(rate)) / d rate
    #   = - 1. / rate
    # d log(gamma_sample(..)) / d log_rate
    #   = d (log(gamma_sample(.., rate=1)) - log_rate) / d log_rate
    #   = d (- log_rate) / d log_rate
    #   = -1.
    partial_rate = 0.
    partial_log_rate = 0.
    if log_rate is not None:
      exp_samples_rate = tf.math.exp(samples + log_rate)
      partial_log_rate = -1.
    elif rate is not None:
      exp_samples_rate = tf.math.exp(samples) * rate
      partial_rate = -1. / rate
    else:
      exp_samples_rate = tf.math.exp(samples)
    partial_concentration = tf.raw_ops.RandomGammaGrad(
        alpha=concentration, sample=exp_samples_rate) / exp_samples_rate

  else:
    partial_rate = 0.
    partial_log_rate = 0.
    if log_rate is not None:
      # d gamma_sample(exp(log_rate)) / d log_rate
      #   = d gamma_sample(rate) / d rate * d exp(log_rate) / d log_rate
      #   = d gamma_sample(rate) / d rate * exp(log_rate)
      #   = d gamma_sample(rate) / d rate * rate
      # Note that d gamma_sample(rate) / d rate = -gamma_sample(rate) / rate
      # So d gamma_sample(exp(log_rate)) / d log_rate = -gamma_sample(rate)
      rate = tf.math.exp(log_rate)
      partial_log_rate = -samples
    elif rate is not None:
      partial_rate = -samples / rate
    else:
      rate = 1.
    partial_concentration = tf.raw_ops.RandomGammaGrad(
        alpha=concentration, sample=samples * rate) / rate

  return partial_concentration, partial_rate, partial_log_rate


def _random_gamma_fwd(shape, concentration, rate, log_rate, seed, log_space):
  """Compute output, aux (collaborates with _random_gamma_bwd)."""
  samples, impl = _random_gamma_no_gradient(
      shape, concentration, rate, log_rate, seed, log_space)
  return ((samples, impl),
          (samples, shape, concentration, rate, log_rate, log_space))


def _random_gamma_bwd(aux, g):
  """The gradient of the gamma samples."""
  samples, shape, concentration, rate, log_rate, log_space = aux
  dsamples, dimpl = g
  # Ignore any gradient contributions that come from the implementation enum.
  del dimpl
  partial_concentration, partial_rate, partial_log_rate = _compute_partials(
      samples, concentration, rate, log_rate, log_space)

  # These will need to be shifted by the extra dimensions added from
  # `sample_shape`.
  rate_shape = _shape_or_scalar(rate, log_rate)
  reduce_dims = tf.range(tf.size(shape) - tf.maximum(tf.rank(concentration),
                                                     tf.size(rate_shape)))
  grad_concentration = tf.math.reduce_sum(
      dsamples * partial_concentration, axis=reduce_dims)
  grad_log_rate = None
  grad_rate = None
  if rate is not None:
    grad_rate = tf.math.reduce_sum(dsamples * partial_rate, axis=reduce_dims)
  elif log_rate is not None:
    grad_log_rate = tf.math.reduce_sum(
        dsamples * partial_log_rate, axis=reduce_dims)

  rate_tensorshape = _tensorshape_or_scalar(rate, log_rate)
  if (tensorshape_util.is_fully_defined(concentration.shape) and
      tensorshape_util.is_fully_defined(rate_tensorshape) and
      concentration.shape == rate_tensorshape):
    return grad_concentration, grad_rate, grad_log_rate, None  # seed=None

  ax_conc, ax_rate = tf.raw_ops.BroadcastGradientArgs(
      s0=tf.shape(concentration), s1=rate_shape)
  grad_concentration = tf.reshape(
      tf.math.reduce_sum(grad_concentration, axis=ax_conc),
      tf.shape(concentration))
  if grad_rate is not None:
    grad_rate = tf.reshape(
        tf.math.reduce_sum(grad_rate, axis=ax_rate), rate_shape)
  if grad_log_rate is not None:
    grad_log_rate = tf.reshape(
        tf.math.reduce_sum(grad_log_rate, axis=ax_rate), rate_shape)

  return grad_concentration, grad_rate, grad_log_rate, None  # seed=None


def _random_gamma_jvp(shape, log_space, primals, tangents):
  """Computes JVP for gamma sample (supports JAX custom derivative)."""
  concentration, rate, log_rate, seed = primals
  dconcentration, drate, dlog_rate, dseed = tangents
  del dseed
  # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
  dconcentration = tf.broadcast_to(dconcentration, shape)
  drate = 0 if drate is None else tf.broadcast_to(drate, shape)
  dlog_rate = 0 if dlog_rate is None else tf.broadcast_to(dlog_rate, shape)

  samples, impl = _random_gamma_no_gradient(
      shape, concentration, rate, log_rate, seed, log_space)

  partial_concentration, partial_rate, partial_log_rate = _compute_partials(
      samples, concentration, rate, log_rate, log_space)

  dsamples = (partial_concentration * dconcentration +
              partial_rate * drate +
              partial_log_rate * dlog_rate)
  return (
      (samples, impl),
      (dsamples, tf.zeros_like(impl)))


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_random_gamma_fwd,
    vjp_bwd=_random_gamma_bwd,
    jvp_fn=_random_gamma_jvp,
    nondiff_argnums=(0, 5))
def _random_gamma_gradient(
    shape, concentration, rate, log_rate, seed, log_space):
  return _random_gamma_no_gradient(
      shape, concentration, rate, log_rate, seed, log_space)


def _fix_zero_samples(s):
  # We use `tf.where` instead of `tf.maximum` because we need to allow for
  # `samples` to be `nan`, but `tf.maximum(nan, x) == x`.
  return tf.where(
      s == 0, np.finfo(dtype_util.as_numpy_dtype(s.dtype)).tiny, s)


# TF custom_gradient doesn't support kwargs, so we wrap _random_gamma_gradient.
def random_gamma_with_runtime(
    shape, concentration, rate=None, log_rate=None, seed=None, log_space=False):
  """Returns both a sample and the id of the implementation-selected runtime."""
  # This method exists chiefly for testing purposes.
  dtype = dtype_util.common_dtype([concentration, rate, log_rate], tf.float32)
  concentration = tf.convert_to_tensor(concentration, dtype=dtype)
  shape = ps.convert_to_shape_tensor(shape, dtype_hint=tf.int32, name='shape')

  if rate is not None and log_rate is not None:
    raise ValueError('At most one of `rate` and `log_rate` may be specified.')
  if rate is not None:
    rate = tf.convert_to_tensor(rate, dtype=dtype)
  if log_rate is not None:
    log_rate = tf.convert_to_tensor(log_rate, dtype=dtype)
  total_shape = ps.concat(
      [shape, ps.broadcast_shape(ps.shape(concentration),
                                 _shape_or_scalar(rate, log_rate))],
      axis=0)
  seed = samplers.sanitize_seed(seed, salt='random_gamma')
  return _random_gamma_gradient(
      total_shape, concentration, rate, log_rate, seed, log_space)


def random_gamma(
    shape, concentration, rate=None, log_rate=None, seed=None, log_space=False):
  return random_gamma_with_runtime(
      shape, concentration, rate, log_rate, seed, log_space)[0]


def random_gamma_rejection(
    shape, concentration, rate=None, log_rate=None, seed=None, log_space=False,
    internal_dtype=tf.float64):
  """Samples from the gamma distribution.

  The sampling algorithm is rejection sampling [1], and pathwise gradients with
  respect to concentration are computed via implicit differentiation [2].

  Args:
    shape: The output sample shape. Trailing dims must match broadcast of
      `concentration` with `rate` or `log_rate`.
    concentration: Floating point tensor, the concentration params of the
      distribution(s). Must contain only positive values. Must broadcast with
      `rate` or `log_rate`, if given.
    rate: Floating point tensor, the inverse scale params of the
      distribution(s). Must contain only positive values. Must broadcast with
      `concentration`. If `None`, handled as if 1 (but possibly more
      efficiently). Mutually exclusive with `log_rate`.
    log_rate: Floating point tensor, log of the inverse scale params of the
      distribution(s). Must broadcast with `concentration`. If `None`, handled
      as if 0 (but possibly more efficiently). Mutually exclusive with `rate`.
    seed: (optional) The random seed.
    log_space: Optionally sample log(gamma) variates.
    internal_dtype: dtype to use for internal computations.

  Returns:
    Differentiable samples from the gamma distribution.

  #### References

  [1] George Marsaglia and Wai Wan Tsang. A simple method for generating Gamma
      variables. ACM Transactions on Mathematical Software, 2000.

  [2] Michael Figurnov, Shakir Mohamed, and Andriy Mnih. Implicit
      Reparameterization Gradients. Neural Information Processing Systems, 2018.
  """
  generate_and_test_samples_seed, concentration_fix_seed = samplers.split_seed(
      seed, salt='random_gamma')
  output_dtype = dtype_util.common_dtype([concentration, rate, log_rate],
                                         dtype_hint=tf.float32)

  def rejection_sample(concentration):
    """Gamma rejection sampler."""
    # Note, concentration here already has a shape that is broadcast with rate.
    cast_concentration = tf.cast(concentration, internal_dtype)

    good_params_mask = (concentration > 0.)
    # When replacing NaN values, use 100. for concentration, since that leads to
    # a high-likelihood of the rejection sampler accepting on the first pass.
    safe_concentration = tf.where(good_params_mask, cast_concentration, 100.)

    modified_safe_concentration = tf.where(
        safe_concentration < 1., safe_concentration + 1., safe_concentration)

    one_third = tf.constant(1. / 3, dtype=internal_dtype)
    d = modified_safe_concentration - one_third
    c = one_third * tf.math.rsqrt(d)

    def generate_and_test_samples(seed):
      """Generate and test samples."""
      v_seed, u_seed = samplers.split_seed(seed)

      def generate_positive_v():
        """Generate positive v."""
        def _inner(seed):
          x = samplers.normal(shape, dtype=internal_dtype, seed=seed)
          # This implicitly broadcasts concentration up to sample shape.
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
      logv = tf.math.log1p(c * x)
      x2 = x * x
      v3 = v * v * v
      logv3 = logv * 3

      u = samplers.uniform(
          shape, dtype=internal_dtype, seed=u_seed)

      # In [1], the suggestion is to first check u < 1 - 0.331 * x2 * x2, and to
      # run the check below only if it fails, in order to avoid the relatively
      # expensive logarithm calls. Our algorithm operates in batch mode: we will
      # have to compute or not compute the logarithms for the entire batch, and
      # as the batch gets larger, the odds we compute it grow. Therefore we
      # don't bother with the "cheap" check.
      good_sample_mask = tf.math.log(u) < (x2 / 2. + d * (1 - v3 + logv3))

      return logv3 if log_space else v3, good_sample_mask

    samples = brs.batched_las_vegas_algorithm(
        generate_and_test_samples, seed=generate_and_test_samples_seed)[0]

    concentration_fix_unif = samplers.uniform(  # in [0, 1)
        shape, dtype=internal_dtype, seed=concentration_fix_seed)

    if log_space:
      concentration_lt_one_fix = tf.where(
          safe_concentration < 1.,
          # Why do we use log1p(-x)? x is in [0, 1) and log(0) = -inf, is bad.
          # x ~ U(0,1) => 1-x ~ U(0,1)
          # But at the boundary, 1-x in (0, 1]. Good.
          # So we can take log(unif(0,1)) safely as log(1-unif(0,1)).
          # log1p(-0) = 0, and log1p(-almost_one) = -not_quite_inf. Good.
          tf.math.log1p(-concentration_fix_unif) / safe_concentration,
          tf.zeros((), dtype=internal_dtype))
      samples = samples + tf.math.log(d) + concentration_lt_one_fix
    else:
      concentration_lt_one_fix = tf.where(
          safe_concentration < 1.,
          tf.math.pow(concentration_fix_unif,
                      tf.math.reciprocal(safe_concentration)),
          tf.ones((), dtype=internal_dtype))
      samples = samples * d * concentration_lt_one_fix

    samples = tf.where(good_params_mask, samples, np.nan)
    output_type_samples = tf.cast(samples, output_dtype)

    return output_type_samples

  broadcast_conc_shape = ps.broadcast_shape(ps.shape(concentration),
                                            _shape_or_scalar(rate, log_rate))
  broadcast_concentration = tf.broadcast_to(concentration, broadcast_conc_shape)
  concentration_samples = rejection_sample(broadcast_concentration)

  if rate is not None and log_rate is not None:
    raise ValueError('`rate` and `log_rate` are mutually exclusive.')

  if rate is None and log_rate is None:
    if not log_space:
      concentration_samples = _fix_zero_samples(concentration_samples)
    return concentration_samples

  if log_space:
    if log_rate is None:
      log_rate = tf.math.log(tf.where(rate > 0., rate, np.nan))
    return concentration_samples - log_rate
  else:
    if rate is None:
      rate = tf.math.exp(log_rate)
    corrected_rate = tf.where(rate > 0., rate, np.nan)  # log_rate=-inf case
    return _fix_zero_samples(concentration_samples / corrected_rate)
