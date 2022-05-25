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
"""Gibbs sampling inference for (a special case of) STS models.

These methods implement Gibbs sampling steps for STS models that combine a
single LocalLevel or LocalLinearTrend component with a linear regression
component, with conjugate
InverseGamma priors on the scale and a Gaussian prior on the weights. This model
class is somewhat general, in that we assume that any seasonal/holiday variation
can be encoded in the design matrix of the linear regression. The intent is to
support deployment of STS inference in latency-sensitive applications.

This Gibbs sampler tends to reach acceptable answers much more quickly than
fitting the same models by gradient-based methods (VI or HMC). Because it does
not marginalize out the linear Gaussian latents analytically, it may be more
prone to getting stuck at a single (perhaps suboptimal) posterior explanation;
however, in practice it often finds good solutions.

The speed advantage of Gibbs sampling in this model likely arises from a
combination of:

- Analytically sampling the regression weights once per sampling cycle, instead
  of requiring a quadratically-expensive update at each timestep of Kalman
  filtering (as in DynamicLinearRegression), or relying on gradient-based
  approximate inference (as in LinearRegression).
- Exploiting conjugacy to sample the scale parameters directly.
- Specializing the Gibbs step for the latent level to the case of a
  scalar process with identity transitions.

It would be possible to expand this sampler to support additional STS models,
potentially at a cost with respect to some of these performance advantages (and
additional code):

- To support general latent state-space models, one would augment the sampler
  state to track all parameters in the model. Each component would need to
  register Gibbs sampling steps for its parameters (assuming conjugate priors),
  as a function of the sampled latent trajectory. The resampling steps for the
  observation_noise_scale and level_scale parameters would then be replaced with
  a generic loop over all parameters in the model.
- For specific models it may be possible to implement an efficient prior
  sampling algorithm, analagous to `LocalLevelStateSpaceModel._joint_sample_n`.
  This may be significantly faster than the generic sampler and can speed up
  the posterior sampling step for the latent trajectory.
"""

import collections

import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import sts
from tensorflow_probability.python.distributions import normal_conjugate_posteriors
from tensorflow_probability.python.experimental import distributions as tfde
from tensorflow_probability.python.experimental.sts_gibbs import dynamic_spike_and_slab
from tensorflow_probability.python.experimental.sts_gibbs import spike_and_slab
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.sts import components as sts_components
from tensorflow_probability.python.sts.internal import util as sts_util

JAX_MODE = False

# The sampler state stores current values for each model parameter,
# and auxiliary quantities such as the latent level. It should have the property
# that `model.make_state_space_model(num_timesteps, GibbsSamplerState(...))`
# behaves properly -- i.e., that the state contains all model
# parameters *in the same order* as they are listed in `model.parameters`. This
# is currently enforced by construction in `build_gibbs_fittable_model`.
GibbsSamplerState = collections.namedtuple(  # pylint: disable=unexpected-keyword-arg
    'GibbsSamplerState', [
        'observation_noise_scale',
        'level_scale',
        'weights',
        'level',
        'seed',
        'slope_scale',
        'slope',
    ])
# Make the two slope-related quantities optional, for backwards compatibility.
GibbsSamplerState.__new__.__defaults__ = (
    0.,  # slope_scale
    0.)  # slope


# TODO(b/151571025): revert to `tfd.InverseGamma` once its sampler is XLA-able.
class XLACompilableInverseGamma(tfd.InverseGamma):

  def _sample_n(self, n, seed=None):
    return 1. / tfd.Gamma(
        concentration=self.concentration, rate=self.scale).sample(
            n, seed=seed)


class DummySpikeAndSlabPrior(tfd.Distribution):
  """Dummy prior on sparse regression weights."""

  def __init__(self, dtype=tf.float32):
    super().__init__(
        dtype=dtype,
        reparameterization_type=tfd.FULLY_REPARAMETERIZED,
        validate_args=False,
        allow_nan_stats=True,
        name='dummy_spike_and_slab_prior')

  @property
  def event_shape(self):
    # Present as a vector-valued distribution.
    return tf.TensorShape([1])

  def _parameter_control_dependencies(self, is_init):
    if not is_init:
      raise ValueError('Cannot explicitly operate on a spike-and-slab prior; '
                       'only Gibbs sampling is supported.')
    return []

  def _default_event_space_bijector(self):
    return tfb.Identity()


class SpikeAndSlabSparseLinearRegression(sts_components.LinearRegression):
  """Dummy component for sparse regression with a spike-and-slab prior."""

  def __init__(self,
               design_matrix,
               weights_prior=None,
               sparse_weights_nonzero_prob=0.5,
               name=None):
    # Extract precision matrix from a multivariate normal prior.
    weights_prior_precision = None
    if hasattr(weights_prior, 'precision'):
      weights_prior_precision = weights_prior.precision()
    elif weights_prior is not None:
      inverse_scale = weights_prior.scale.inverse()
      weights_prior_precision = inverse_scale.matmul(
          inverse_scale, adjoint=True).to_dense()
    self._weights_prior_precision = weights_prior_precision
    self._sparse_weights_nonzero_prob = sparse_weights_nonzero_prob
    super().__init__(
        design_matrix=design_matrix,
        weights_prior=DummySpikeAndSlabPrior(
            dtype=dtype_util.common_dtype([design_matrix])),
        name=name)


def _tile_normal_to_mvn_diag(normal_dist, dim):
  return tfd.MultivariateNormalDiag(
      loc=normal_dist.loc[..., tf.newaxis],
      scale_diag=(normal_dist.scale[..., tf.newaxis] *
                  tf.ones([dim], dtype=normal_dist.scale.dtype)))


def _is_multivariate_normal(dist):
  return (isinstance(dist, tfd.MultivariateNormalLinearOperator) or isinstance(
      dist, tfde.MultivariateNormalPrecisionFactorLinearOperator))


def build_model_for_gibbs_fitting(observed_time_series,
                                  design_matrix,
                                  weights_prior,
                                  level_variance_prior,
                                  observation_noise_variance_prior,
                                  slope_variance_prior=None,
                                  initial_level_prior=None,
                                  sparse_weights_nonzero_prob=None):
  """Builds a StructuralTimeSeries model instance that supports Gibbs sampling.

  To support Gibbs sampling, a model must have have conjugate priors on all
  scale and weight parameters, and must be constructed so that
  `model.parameters` matches the parameters and ordering specified by the
  `GibbsSamplerState` namedtuple. Currently, this includes (only) models
  consisting of the sum of a LocalLevel or LocalLinearTrend component with
  (optionally) a LinearRegression or SpikeAndSlabSparseLinearRegression
  component.

  Args:
    observed_time_series: optional `float` `Tensor` of shape [..., T, 1]`
      (omitting the trailing unit dimension is also supported when `T > 1`),
      specifying an observed time series. May optionally be an instance of
      `tfp.sts.MaskedTimeSeries`, which includes a mask `Tensor` to specify
      timesteps with missing observations.
    design_matrix: Optional float `Tensor` of shape `concat([batch_shape,
      [num_timesteps, num_features]])`. This may also optionally be an instance
      of `tf.linalg.LinearOperator`. If None, no regression is done.
    weights_prior: Optional distribution instance specifying a normal prior on
      weights. This may be a multivariate normal instance with event shape
      `[num_features]`, or a scalar normal distribution with event shape `[]`.
      In either case, the batch shape must broadcast to the batch shape of
      `observed_time_series`. If a `sparse_weights_nonzero_prob` is specified,
      requesting sparse regression, then the `weights_prior` mean is ignored
      (because nonzero means are not currently implemented by the spike-and-slab
      sampler). In this case, `weights_prior=None` is also valid, and will use
      the default prior of the spike-and-slab sampler.
    level_variance_prior: An instance of `tfd.InverseGamma` representing a prior
      on the level variance (`level_scale**2`) of a local level model. May have
      batch shape broadcastable to the batch shape of `observed_time_series`.
    observation_noise_variance_prior: An instance of `tfd.InverseGamma`
      representing a prior on the observation noise variance (
      `observation_noise_scale**2`). May have batch shape broadcastable to the
      batch shape of `observed_time_series`.
    slope_variance_prior: Optional instance of `tfd.InverseGamma` representing a
      prior on slope variance (`slope_scale**2`) of a local linear trend model.
      May have batch shape broadcastable to the batch shape of
      `observed_time_series`. If specified, a local linear trend model is used
      rather than a local level model.
      Default value: `None`.
    initial_level_prior: optional `tfd.Distribution` instance specifying a
      prior on the initial level. If `None`, a heuristic default prior is
      constructed based on the provided `observed_time_series`.
      Default value: `None`.
    sparse_weights_nonzero_prob: Optional scalar float `Tensor` prior
      probability that any given feature has nonzero weight. If specified, this
      triggers a sparse regression with a spike-and-slab prior, where
      `sparse_weights_nonzero_prob` is the prior probability of the 'slab'
      component.
      Default value: `None`.

  Returns:
    model: A `tfp.sts.StructuralTimeSeries` model instance.
  """
  if design_matrix is None:
    if sparse_weights_nonzero_prob is not None:
      raise ValueError(
          'Design matrix is None thus sparse_weights_nonzero_prob should '
          'not be defined, as it will not be used.')
    if weights_prior is not None:
      raise ValueError(
          'Design matrix is None thus weights_prior should not be defined, '
          'as it will not be used.')

  if isinstance(weights_prior, tfd.Normal):
    # Canonicalize scalar normal priors as diagonal MVNs.
    # design_matrix must be defined, otherwise we threw an exception earlier.
    if isinstance(design_matrix, tf.linalg.LinearOperator):
      num_features = design_matrix.shape_tensor()[-1]
    else:
      num_features = prefer_static.dimension_size(design_matrix, -1)
    weights_prior = _tile_normal_to_mvn_diag(weights_prior, num_features)
  elif weights_prior is not None and not _is_multivariate_normal(weights_prior):
    raise ValueError('Weights prior must be a normal distribution or `None`.')
  if not isinstance(level_variance_prior, tfd.InverseGamma):
    raise ValueError(
        'Level variance prior must be an inverse gamma distribution.')
  if (slope_variance_prior is not None and
      not isinstance(slope_variance_prior, tfd.InverseGamma)):
    raise ValueError(
        'Slope variance prior must be an inverse gamma distribution; got: {}.'
        .format(slope_variance_prior))
  if not isinstance(observation_noise_variance_prior, tfd.InverseGamma):
    raise ValueError('Observation noise variance prior must be an inverse '
                     'gamma distribution.')

  sqrt = tfb.Invert(tfb.Square())  # Converts variance priors to scale priors.
  components = []

  # Level or trend component.
  if slope_variance_prior:
    components.append(
        sts.LocalLinearTrend(
            observed_time_series=observed_time_series,
            level_scale_prior=sqrt(level_variance_prior),
            slope_scale_prior=sqrt(slope_variance_prior),
            initial_level_prior=initial_level_prior,
            name='local_linear_trend'))
  else:
    components.append(
        sts.LocalLevel(
            observed_time_series=observed_time_series,
            level_scale_prior=sqrt(level_variance_prior),
            initial_level_prior=initial_level_prior,
            name='local_level'))

  # Regression component.
  if design_matrix is None:
    pass
  elif sparse_weights_nonzero_prob is not None:
    components.append(
        SpikeAndSlabSparseLinearRegression(
            design_matrix=design_matrix,
            weights_prior=weights_prior,
            sparse_weights_nonzero_prob=sparse_weights_nonzero_prob,
            name='sparse_regression'))
  else:
    components.append(
        sts.LinearRegression(
            design_matrix=design_matrix,
            weights_prior=weights_prior,
            name='regression'))
  model = sts.Sum(
      components,
      observed_time_series=observed_time_series,
      observation_noise_scale_prior=sqrt(observation_noise_variance_prior),
      # The Gibbs sampling steps in this file do not account for an
      # offset to the observed series. Instead, we assume the
      # observed series has already been centered and
      # scale-normalized.
      constant_offset=0.)
  model.supports_gibbs_sampling = True
  return model


def _get_design_matrix(model):
  """Returns the design matrix for an STS model with a regression component.

  If there is not a design matrix, None is returned.

  Args:
    model: A `tfp.sts.StructuralTimeSeries` model instance return by
      `build_model_for_gibbs_fitting`.
  """
  design_matrices = [
      component.design_matrix
      for component in model.components
      if hasattr(component, 'design_matrix')
  ]
  if not design_matrices:
    return None
  if len(design_matrices) > 1:
    raise ValueError('Model contains multiple regression components.')
  return design_matrices[0]


def fit_with_gibbs_sampling(model,
                            observed_time_series,
                            num_chains=(),
                            num_results=2000,
                            num_warmup_steps=200,
                            initial_state=None,
                            seed=None,
                            default_pseudo_observations=None,
                            experimental_use_dynamic_cholesky=False):
  """Fits parameters for an STS model using Gibbs sampling.

  Args:
    model: A `tfp.sts.StructuralTimeSeries` model instance return by
      `build_model_for_gibbs_fitting`.
    observed_time_series: `float` `Tensor` of shape [..., T, 1]` (omitting the
      trailing unit dimension is also supported when `T > 1`), specifying an
      observed time series. May optionally be an instance of
      `tfp.sts.MaskedTimeSeries`, which includes a mask `Tensor` to specify
      timesteps with missing observations.
    num_chains: Optional int to indicate the number of parallel MCMC chains.
      Default to an empty tuple to sample a single chain.
    num_results: Optional int to indicate number of MCMC samples.
    num_warmup_steps: Optional int to indicate number of MCMC samples.
    initial_state: A `GibbsSamplerState` structure of the initial states of the
      MCMC chains.
    seed: Optional `Python` `int` seed controlling the sampled values.
    default_pseudo_observations: Optional scalar float `Tensor` Controls the
      number of pseudo-observations for the prior precision matrix over the
      weights.
    experimental_use_dynamic_cholesky: Optional bool - in case of spike and slab
      sampling, will dynamically select the subset of the design matrix with
      active features to perform the Cholesky decomposition. This may provide
      a speedup when the number of true features is small compared to the size
      of the design matrix. *Note*: If this is true, neither batch shape nor
      `jit_compile` is supported.


  Returns:
    model: A `GibbsSamplerState` structure of posterior samples.
  """
  if not hasattr(model, 'supports_gibbs_sampling'):
    raise ValueError('This STS model does not support Gibbs sampling. Models '
                     'for Gibbs sampling must be created using the '
                     'method `build_model_for_gibbs_fitting`.')
  if not tf.nest.is_nested(num_chains):
    num_chains = [num_chains]

  [observed_time_series, is_missing
  ] = sts_util.canonicalize_observed_time_series_with_mask(observed_time_series)
  dtype = observed_time_series.dtype

  # The canonicalized time series always has trailing dimension `1`,
  # because although LinearGaussianSSMs support vector observations, STS models
  # describe scalar time series only. For our purposes it'll be cleaner to
  # remove this dimension.
  observed_time_series = observed_time_series[..., 0]
  batch_shape = prefer_static.concat(
      [num_chains, prefer_static.shape(observed_time_series)[:-1]], axis=-1)
  level_slope_shape = prefer_static.concat(
      [num_chains, prefer_static.shape(observed_time_series)], axis=-1)

  # Treat a LocalLevel model as the special case of LocalLinearTrend where
  # the slope_scale is always zero.
  initial_slope_scale = 0.
  initial_slope = 0.
  if isinstance(model.components[0], sts.LocalLinearTrend):
    initial_slope_scale = 1. * tf.ones(batch_shape, dtype=dtype)
    initial_slope = tf.zeros(level_slope_shape, dtype=dtype)

  if initial_state is None:
    design_matrix = _get_design_matrix(model)
    weights = tf.zeros(0, dtype=dtype) if design_matrix is None else tf.zeros(  # pylint:disable=g-long-ternary
        prefer_static.concat([batch_shape, design_matrix.shape[-1:]],
                             axis=0),
        dtype=dtype)
    initial_state = GibbsSamplerState(
        observation_noise_scale=tf.ones(batch_shape, dtype=dtype),
        level_scale=tf.ones(batch_shape, dtype=dtype),
        slope_scale=initial_slope_scale,
        weights=weights,
        level=tf.zeros(level_slope_shape, dtype=dtype),
        slope=initial_slope,
        seed=None)  # Set below.

  if isinstance(seed, six.integer_types):
    tf.random.set_seed(seed)

  # Always use the passed-in `seed` arg, ignoring any seed in the initial state.
  initial_state = initial_state._replace(
      seed=samplers.sanitize_seed(seed, salt='initial_GibbsSamplerState'))

  sampler_loop_body = _build_sampler_loop_body(
      model, observed_time_series, is_missing, default_pseudo_observations,
      experimental_use_dynamic_cholesky)

  samples = tf.scan(sampler_loop_body,
                    np.arange(num_warmup_steps + num_results), initial_state)
  return tf.nest.map_structure(lambda x: x[num_warmup_steps:], samples)


def one_step_predictive(model,
                        posterior_samples,
                        num_forecast_steps=0,
                        original_mean=0.,
                        original_scale=1.,
                        thin_every=10,
                        use_zero_step_prediction=False):
  """Constructs a one-step-ahead predictive distribution at every timestep.

  Unlike the generic `tfp.sts.one_step_predictive`, this method uses the
  latent levels from Gibbs sampling to efficiently construct a predictive
  distribution that mixes over posterior samples. The predictive distribution
  may also include additional forecast steps.

  This method returns the predictive distributions for each timestep given
  previous timesteps and sampled model parameters, `p(observed_time_series[t] |
  observed_time_series[:t], weights, observation_noise_scale)`. Note that the
  posterior values of the weights and noise scale will in general be informed
  by observations from all timesteps *including the step being predicted*, so
  this is not a strictly kosher probabilistic quantity, but in general we assume
  that it's close, i.e., that the step being predicted had very small individual
  impact on the overall parameter posterior.

  Args:
    model: A `tfd.sts.StructuralTimeSeries` model instance. This must be of the
      form constructed by `build_model_for_gibbs_sampling`.
    posterior_samples: A `GibbsSamplerState` instance in which each element is a
      `Tensor` with initial dimension of size `num_samples`.
    num_forecast_steps: Python `int` number of additional forecast steps to
      append.
      Default value: `0`.
    original_mean: Optional scalar float `Tensor`, added to the predictive
      distribution to undo the effect of input normalization.
      Default value: `0.`
    original_scale: Optional scalar float `Tensor`, used to rescale the
      predictive distribution to undo the effect of input normalization.
      Default value: `1.`
    thin_every: Optional Python `int` factor by which to thin the posterior
      samples, to reduce complexity of the predictive distribution. For example,
      if `thin_every=10`, every `10`th sample will be used.
      Default value: `10`.
    use_zero_step_prediction: If true, instead of using the local level
      and trend from the timestep before, just use the local level from the
      same timestep.

  Returns:
    predictive_dist: A `tfd.MixtureSameFamily` instance of event shape
      `[num_timesteps + num_forecast_steps]` representing the predictive
      distribution of each timestep given previous timesteps.
  """
  dtype = dtype_util.common_dtype([
      posterior_samples.level_scale, posterior_samples.observation_noise_scale,
      posterior_samples.level, original_mean, original_scale
  ],
                                  dtype_hint=tf.float32)
  num_observed_steps = prefer_static.shape(posterior_samples.level)[-1]

  original_mean = tf.convert_to_tensor(original_mean, dtype=dtype)
  original_scale = tf.convert_to_tensor(original_scale, dtype=dtype)
  thinned_samples = tf.nest.map_structure(lambda x: x[::thin_every],
                                          posterior_samples)

  if prefer_static.rank_from_shape(  # If no slope was inferred, treat as zero.
      prefer_static.shape(thinned_samples.slope)) <= 1:
    thinned_samples = thinned_samples._replace(
        slope=tf.zeros_like(thinned_samples.level),
        slope_scale=tf.zeros_like(thinned_samples.level_scale))

  num_steps_from_last_observation = tf.concat(
      [(tf.zeros([num_observed_steps], dtype=dtype) if use_zero_step_prediction
        else tf.ones([num_observed_steps], dtype=dtype)),
       tf.range(1, num_forecast_steps + 1, dtype=dtype)],
      axis=0)

  # The local linear trend model expects that the level at step t + 1 is equal
  # to the level at step t, plus the slope at time t - 1,
  # plus transition noise of scale 'level_scale' (which we account for below).
  if num_forecast_steps > 0:
    num_batch_dims = prefer_static.rank_from_shape(
        prefer_static.shape(thinned_samples.level)) - 2
    # All else equal, the current level will remain stationary.
    forecast_level = tf.tile(
        thinned_samples.level[..., -1:],
        tf.concat([
            tf.ones([num_batch_dims + 1], dtype=tf.int32), [num_forecast_steps]
        ],
                  axis=0))
    # If the model includes slope, the level will steadily increase.
    forecast_level += (
        thinned_samples.slope[..., -1:] *
        tf.range(1., num_forecast_steps + 1., dtype=forecast_level.dtype))

  level_pred = tf.concat(
      ([thinned_samples.level] if use_zero_step_prediction else [
          thinned_samples.level[..., :1],  # t == 0
          (thinned_samples.level[..., :-1] + thinned_samples.slope[..., :-1]
          )  # 1 <= t < T. Constructs the next level from previous level
          # and previous slope.
      ]) + ([forecast_level] if num_forecast_steps > 0 else []),
      axis=-1)

  design_matrix = _get_design_matrix(model)
  if design_matrix is not None:
    design_matrix = design_matrix.to_dense()[:num_observed_steps +
                                             num_forecast_steps]
    regression_effect = tf.linalg.matvec(design_matrix, thinned_samples.weights)
  else:
    regression_effect = 0

  y_mean = ((level_pred + regression_effect) * original_scale[..., tf.newaxis] +
            original_mean[..., tf.newaxis])

  # To derive a forecast variance, including slope uncertainty, let
  #  `r[:k]` be iid Gaussian RVs with variance `level_scale**2` and `s[:k]` be
  # iid Gaussian RVs with variance `slope_scale**2`. Then the forecast level at
  # step `T + k` can be written as
  #   (level[T] +           # Last known level.
  #    r[0] + ... + r[k] +  # Sum of random walk terms on level.
  #    slope[T] * k         # Contribution from last known slope.
  #    (k - 1) * s[0] +     # Contributions from random walk terms on slope.
  #    (k - 2) * s[1] +
  #    ... +
  #    1 * s[k - 1])
  # which has variance of
  #  (level_scale**2 * k +
  #   slope_scale**2 * ( (k - 1)**2 +
  #                      (k - 2)**2 +
  #                      ... + 1 ))
  # Here the `slope_scale` coefficient is the `k - 1`th square pyramidal
  # number [1], which is given by
  #  (k - 1) * k * (2 * k - 1) / 6.
  #
  # [1] https://en.wikipedia.org/wiki/Square_pyramidal_number
  variance_from_level = (
      thinned_samples.level_scale[..., tf.newaxis]**2 *
      num_steps_from_last_observation)
  variance_from_slope = thinned_samples.slope_scale[..., tf.newaxis]**2 * (
      (num_steps_from_last_observation - 1) * num_steps_from_last_observation *
      (2 * num_steps_from_last_observation - 1)) / 6.
  y_scale = (
      original_scale *
      tf.sqrt(thinned_samples.observation_noise_scale[..., tf.newaxis]**2 +
              variance_from_level + variance_from_slope))

  num_posterior_draws = prefer_static.shape(y_mean)[0]
  return tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          logits=tf.zeros([num_posterior_draws], dtype=y_mean.dtype)),
      components_distribution=tfd.Normal(
          loc=dist_util.move_dimension(y_mean, 0, -1),
          scale=dist_util.move_dimension(y_scale, 0, -1)))


def _resample_weights(design_matrix,
                      target_residuals,
                      observation_noise_scale,
                      weights_prior_scale,
                      seed=None):
  """Samples regression weights from their conditional posterior.

  This assumes a conjugate normal regression model,

  ```
  weights ~ Normal(loc=0., covariance_matrix=weights_prior_scale**2 * I)
  target_residuals ~ Normal(loc=matvec(design_matrix, weights),
                            covariance_matrix=observation_noise_scale**2 * I)
  ```

  and returns a sample from `p(weights | target_residuals,
    observation_noise_scale, design_matrix)`.

  Args:
    design_matrix: Float `Tensor` design matrix of shape `[..., num_timesteps,
      num_features]`.
    target_residuals:  Float `Tensor` of shape `[..., num_observations]`
    observation_noise_scale: Scalar float `Tensor` (with optional batch shape)
      standard deviation of the iid observation noise.
    weights_prior_scale: Instance of `tf.linalg.LinearOperator` of shape
      `[num_features, num_features]` (with optional batch shape), specifying the
      scale of a multivariate Normal prior on regression weights.
    seed: Optional `Python` `int` seed controlling the sampled values.

  Returns:
    weights: Float `Tensor` of shape `[..., num_features]`, sampled from
      the conditional posterior `p(weights | target_residuals,
      observation_noise_scale, weights_prior_scale)`.
  """
  weights_mean, weights_prec = (
      normal_conjugate_posteriors.mvn_conjugate_linear_update(
          linear_transformation=design_matrix,
          observation=target_residuals,
          prior_scale=weights_prior_scale,
          likelihood_scale=tf.linalg.LinearOperatorScaledIdentity(
              num_rows=prefer_static.shape(design_matrix)[-2],
              multiplier=observation_noise_scale)))
  sampled_weights = weights_prec.cholesky().solvevec(
      samplers.normal(
          shape=prefer_static.shape(weights_mean),
          dtype=design_matrix.dtype,
          seed=seed),
      adjoint=True)
  return weights_mean + sampled_weights


def _resample_latents(observed_residuals,
                      level_scale,
                      observation_noise_scale,
                      initial_state_prior,
                      slope_scale=None,
                      is_missing=None,
                      sample_shape=(),
                      seed=None):
  """Uses Durbin-Koopman sampling to resample the latent level and slope.

  Durbin-Koopman sampling [1] is an efficient algorithm to sample from the
  posterior latents of a linear Gaussian state space model. This method
  implements the algorithm.

  [1] Durbin, J. and Koopman, S.J. (2002) A simple and efficient simulation
      smoother for state space time series analysis.

  Args:
    observed_residuals: Float `Tensor` of shape `[..., num_observations]`,
      specifying the centered observations `(x - loc)`.
    level_scale: Float scalar `Tensor` (may contain batch dimensions) specifying
      the standard deviation of the level random walk steps.
    observation_noise_scale: Float scalar `Tensor` (may contain batch
      dimensions) specifying the standard deviation of the observation noise.
    initial_state_prior: instance of `tfd.MultivariateNormalLinearOperator`.
    slope_scale: Optional float scalar `Tensor` (may contain batch dimensions)
      specifying the standard deviation of slope random walk steps. If provided,
      a `LocalLinearTrend` model is used, otherwise, a `LocalLevel` model is
      used.
    is_missing: Optional `bool` `Tensor` missingness mask.
    sample_shape: Optional `int` `Tensor` shape of samples to draw.
    seed: `int` `Tensor` of shape `[2]` controlling stateless sampling.

  Returns:
    latents: Float `Tensor` resampled latent level, of shape
      `[..., num_timesteps, latent_size]`, where `...` concatenates the
      sample shape with any batch shape from `observed_time_series`.
  """

  num_timesteps = prefer_static.shape(observed_residuals)[-1]
  if slope_scale is None:
    ssm = sts.LocalLevelStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale)
  else:
    ssm = sts.LocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale,
        slope_scale=slope_scale)

  return ssm.posterior_sample(
      observed_residuals[..., tf.newaxis],
      sample_shape=sample_shape,
      mask=is_missing,
      seed=seed)


def _resample_scale(prior, observed_residuals, is_missing=None, seed=None):
  """Samples a scale parameter from its conditional posterior.

  We assume the conjugate InverseGamma->Normal model:

  ```
  scale ~ Sqrt(InverseGamma(prior.concentration, prior.scale))
  for i in [1, ..., num_observations]:
    x[i] ~ Normal(loc, scale)
  ```

  in which `loc` is known, and return a sample from `p(scale | x)`.

  Args:
    prior: Prior distribution as a `tfd.InverseGamma` instance.
    observed_residuals: Float `Tensor` of shape `[..., num_observations]`,
      specifying the centered observations `(x - loc)`.
    is_missing: Optional `bool` `Tensor` of shape `[..., num_observations]`. A
      `True` value indicates that the corresponding observation is missing.
    seed: Optional `Python` `int` seed controlling the sampled value.

  Returns:
    sampled_scale: A `Tensor` sample from the posterior `p(scale | x)`.
  """
  dtype = observed_residuals.dtype

  if is_missing is not None:
    num_missing = tf.reduce_sum(tf.cast(is_missing, dtype), axis=-1)
  num_observations = prefer_static.shape(observed_residuals)[-1]
  if is_missing is not None:
    observed_residuals = tf.where(is_missing, tf.zeros_like(observed_residuals),
                                  observed_residuals)
    num_observations -= num_missing

  variance_posterior = type(prior)(
      concentration=prior.concentration + tf.cast(num_observations / 2., dtype),
      scale=prior.scale +
      tf.reduce_sum(tf.square(observed_residuals), axis=-1) / 2.)
  new_scale = tf.sqrt(variance_posterior.sample(seed=seed))

  # Support truncated priors.
  if hasattr(prior, 'upper_bound') and prior.upper_bound is not None:
    new_scale = tf.minimum(new_scale, prior.upper_bound)

  return new_scale


def _build_sampler_loop_body(model,
                             observed_time_series,
                             is_missing=None,
                             default_pseudo_observations=None,
                             experimental_use_dynamic_cholesky=False):
  """Builds a Gibbs sampler for the given model and observed data.

  Args:
    model: A `tf.sts.StructuralTimeSeries` model instance. This must be of the
      form constructed by `build_model_for_gibbs_sampling`.
    observed_time_series: Float `Tensor` time series of shape `[...,
      num_timesteps]`.
    is_missing: Optional `bool` `Tensor` of shape `[..., num_timesteps]`. A
      `True` value indicates that the observation for that timestep is missing.
    default_pseudo_observations: Optional scalar float `Tensor` Controls the
      number of pseudo-observations for the prior precision matrix over the
      weights.
    experimental_use_dynamic_cholesky: Optional bool - in case of spike and slab
      sampling, will dynamically select the subset of the design matrix with
      active features to perform the Cholesky decomposition. This may provide
      a speedup when the number of true features is small compared to the size
      of the design matrix.

  Returns:
    sampler_loop_body: Python callable that performs a single cycle of Gibbs
      sampling. Its first argument is a `GibbsSamplerState`, and it returns a
      new `GibbsSamplerState`. The second argument (passed by `tf.scan`) is
      ignored.
  """
  if JAX_MODE and experimental_use_dynamic_cholesky:
    raise ValueError('Dynamic Cholesky decomposition not supported in JAX')
  level_component = model.components[0]
  if not (isinstance(level_component, sts.LocalLevel) or
          isinstance(level_component, sts.LocalLinearTrend)):
    raise ValueError('Expected the first model component to be an instance of '
                     '`tfp.sts.LocalLevel` or `tfp.sts.LocalLinearTrend`; '
                     'instead saw {}'.format(level_component))
  model_has_slope = isinstance(level_component, sts.LocalLinearTrend)

  # TODO(kloveless): When we add support for more flexible models, remove
  # this assumption.
  regression_component = (None if len(model.components) != 2 else
                          model.components[1])
  if regression_component:
    if not (isinstance(regression_component, sts.LinearRegression) or
            isinstance(regression_component,
                       SpikeAndSlabSparseLinearRegression)):
      raise ValueError(
          'Expected the second model component to be an instance of '
          '`tfp.sts.LinearRegression` or '
          '`SpikeAndSlabSparseLinearRegression`; '
          'instead saw {}'.format(regression_component))
    model_has_spike_slab_regression = isinstance(
        regression_component, SpikeAndSlabSparseLinearRegression)

  if is_missing is not None:  # Ensure series does not contain NaNs.
    observed_time_series = tf.where(is_missing,
                                    tf.zeros_like(observed_time_series),
                                    observed_time_series)

  num_observed_steps = prefer_static.shape(observed_time_series)[-1]

  design_matrix = _get_design_matrix(model)
  if design_matrix is not None:
    design_matrix = design_matrix.to_dense()[:num_observed_steps]
    if is_missing is not None:
      # Replace design matrix with zeros at unobserved timesteps. This ensures
      # they will not affect the posterior on weights.
      design_matrix = tf.where(is_missing[..., tf.newaxis],
                               tf.zeros_like(design_matrix), design_matrix)

  # Untransform scale priors -> variance priors by reaching thru Sqrt bijector.
  observation_noise_param = model.parameters[0]
  if 'observation_noise' not in observation_noise_param.name:
    raise ValueError('Model parameters {} do not match the expected sampler '
                     'state.'.format(model.parameters))
  observation_noise_variance_prior = observation_noise_param.prior.distribution
  if model_has_slope:
    level_scale_variance_prior, slope_scale_variance_prior = [
        p.prior.distribution for p in level_component.parameters
    ]
  else:
    level_scale_variance_prior = (
        level_component.parameters[0].prior.distribution)

  if regression_component:
    if model_has_spike_slab_regression:
      if experimental_use_dynamic_cholesky:
        sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler
      else:
        sampler = spike_and_slab.SpikeSlabSampler

      spike_and_slab_sampler = sampler(
          design_matrix,
          weights_prior_precision=regression_component._weights_prior_precision,  # pylint: disable=protected-access
          nonzero_prior_prob=regression_component._sparse_weights_nonzero_prob,  # pylint: disable=protected-access
          observation_noise_variance_prior_concentration=(
              observation_noise_variance_prior.concentration),
          observation_noise_variance_prior_scale=(
              observation_noise_variance_prior.scale),
          observation_noise_variance_upper_bound=(
              # The given bound is for the scale, so it must be squared to get
              # the upper bound for the variance.
              tf.math.square(observation_noise_variance_prior.upper_bound)
              if hasattr(observation_noise_variance_prior, 'upper_bound') else
              None),
          **({
              'default_pseudo_observations': default_pseudo_observations
          } if default_pseudo_observations is not None else {}))
      # In case the nonzero probability is exactly one, any proposal with any
      # zero weights will have log prob of -infinity, so we will pin the
      # proposals to one.
      # TODO(colcarroll): Can we short-circuit the feature selection loop in
      # case this is `True`?
      pin_to_nonzero = tf.greater_equal(
          regression_component._sparse_weights_nonzero_prob, 1.)  # pylint: disable=protected-access

    else:
      weights_prior_scale = (regression_component.parameters[0].prior.scale)

  def sampler_loop_body(previous_sample, _):
    """Runs one sampler iteration, resampling all model variables."""

    (weights_seed, level_seed, observation_noise_scale_seed, level_scale_seed,
     loop_seed) = samplers.split_seed(
         previous_sample.seed, n=5, salt='sampler_loop_body')
    # Preserve backward-compatible seed behavior by splitting slope separately.
    slope_scale_seed, = samplers.split_seed(
        previous_sample.seed, n=1, salt='sampler_loop_body_slope')

    if regression_component:
      # We encourage a reasonable initialization by sampling the weights first,
      # so at the first step they are regressed directly against the observed
      # time series. If we instead sampled the level first it might 'explain
      # away' some observed variation that we would ultimately prefer to explain
      # through the regression weights, because the level can represent
      # arbitrary variation, while the weights are limited to representing
      # variation in the subspace given by the design matrix.
      if model_has_spike_slab_regression:
        (observation_noise_variance,
         weights) = spike_and_slab_sampler.sample_noise_variance_and_weights(
             initial_nonzeros=tf.math.logical_or(
                 tf.not_equal(previous_sample.weights, 0.), pin_to_nonzero),
             targets=observed_time_series - previous_sample.level,
             seed=weights_seed)
        observation_noise_scale = tf.sqrt(observation_noise_variance)

      else:
        weights = _resample_weights(
            design_matrix=design_matrix,
            target_residuals=observed_time_series - previous_sample.level,
            observation_noise_scale=previous_sample.observation_noise_scale,
            weights_prior_scale=weights_prior_scale,
            seed=weights_seed)
        # Noise scale will be resampled below.
        observation_noise_scale = previous_sample.observation_noise_scale

      regression_residuals = observed_time_series - tf.linalg.matvec(
          design_matrix, weights)
    else:
      # If there is no regression, then the entire timeseries is a residual.
      regression_residuals = observed_time_series
      # Noise scale will be resampled below.
      observation_noise_scale = previous_sample.observation_noise_scale
      weights = previous_sample.weights

    latents = _resample_latents(
        observed_residuals=regression_residuals,
        level_scale=previous_sample.level_scale,
        slope_scale=previous_sample.slope_scale if model_has_slope else None,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=level_component.initial_state_prior,
        is_missing=is_missing,
        seed=level_seed)
    level = latents[..., 0]
    level_residuals = level[..., 1:] - level[..., :-1]
    if model_has_slope:
      slope = latents[..., 1]
      level_residuals -= slope[..., :-1]
      slope_residuals = slope[..., 1:] - slope[..., :-1]

    # Estimate level scale from the empirical changes in level.
    level_scale = _resample_scale(
        prior=level_scale_variance_prior,
        observed_residuals=level_residuals,
        is_missing=None,
        seed=level_scale_seed)
    if model_has_slope:
      slope_scale = _resample_scale(
          prior=slope_scale_variance_prior,
          observed_residuals=slope_residuals,
          is_missing=None,
          seed=slope_scale_seed)
    if not (regression_component and model_has_spike_slab_regression):
      # Estimate noise scale from the residuals.
      observation_noise_scale = _resample_scale(
          prior=observation_noise_variance_prior,
          observed_residuals=regression_residuals - level,
          is_missing=is_missing,
          seed=observation_noise_scale_seed)

    return GibbsSamplerState(
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale,
        slope_scale=(slope_scale
                     if model_has_slope else previous_sample.slope_scale),
        weights=weights,
        level=level,
        slope=(slope if model_has_slope else previous_sample.slope),
        seed=loop_seed)

  return sampler_loop_body
