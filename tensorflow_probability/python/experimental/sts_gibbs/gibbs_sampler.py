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
import itertools

import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import square
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import normal_conjugate_posteriors
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.experimental.distributions import mvn_precision_factor_linop as mvnpflo
from tensorflow_probability.python.experimental.sts_gibbs import dynamic_spike_and_slab
from tensorflow_probability.python.experimental.sts_gibbs import sample_parameters
from tensorflow_probability.python.experimental.sts_gibbs import spike_and_slab
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.sts.components import local_level
from tensorflow_probability.python.sts.components import local_linear_trend
from tensorflow_probability.python.sts.components import regression
from tensorflow_probability.python.sts.components import seasonal
from tensorflow_probability.python.sts.components import sum as sum_lib
from tensorflow_probability.python.sts.internal import util as sts_util

JAX_MODE = False


__all__ = [
    'GibbsSamplerState',
    'get_seasonal_latents_shape',
    'build_model_for_gibbs_fitting',
    'fit_with_gibbs_sampling',
    'one_step_predictive',
]


# The sampler state stores current values for each model parameter,
# and auxiliary quantities such as the latent level.
GibbsSamplerState = collections.namedtuple(  # pylint: disable=unexpected-keyword-arg
    'GibbsSamplerState',
    [
        'observation_noise_scale',
        'level_scale',
        'weights',
        'level',
        'seed',
        'slope_scale',
        'slope',
        # [batch shape, num_seasonal_components]
        'seasonal_drift_scales',
        # [batch shape,
        #  timeseries_length
        #  num seasonal latents across all components]
        'seasonal_levels',
    ])
# Make the slope-and-season related quantities optional, for backwards
# compatibility.
GibbsSamplerState.__new__.__defaults__ = (
    0.,  # slope_scale
    0.,  # slope
    0.,  # seasonal_drift_scales
    0.)  # seasonal_levels


# TODO(b/151571025): revert to `tfd.InverseGamma` once its sampler is XLA-able.
class XLACompilableInverseGamma(inverse_gamma.InverseGamma):

  def _sample_n(self, n, seed=None):
    return 1. / gamma.Gamma(
        concentration=self.concentration, rate=self.scale).sample(
            n, seed=seed)


class DummySpikeAndSlabPrior(distribution.Distribution):
  """Dummy prior on sparse regression weights."""

  def __init__(self, dtype=tf.float32):
    super().__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
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
    return identity.Identity()


class SpikeAndSlabSparseLinearRegression(regression.LinearRegression):
  """Dummy component for sparse regression with a spike-and-slab prior."""

  def __init__(self,
               design_matrix,
               weights_prior,
               sparse_weights_nonzero_prob=0.5,
               name=None):
    # Extract precision matrix from a multivariate normal prior.
    weights_prior_precision = None
    if hasattr(weights_prior, 'precision'):
      if isinstance(weights_prior.precision, tf.linalg.LinearOperator):
        weights_prior_precision = weights_prior.precision.to_dense()
      else:
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
  return mvn_diag.MultivariateNormalDiag(
      loc=normal_dist.loc[..., tf.newaxis],
      scale_diag=(normal_dist.scale[..., tf.newaxis] *
                  tf.ones([dim], dtype=normal_dist.scale.dtype)))


def _is_multivariate_normal(dist):
  return (
      isinstance(dist, mvn_linear_operator.MultivariateNormalLinearOperator) or
      isinstance(dist, mvnpflo.MultivariateNormalPrecisionFactorLinearOperator))


def build_model_for_gibbs_fitting(observed_time_series,
                                  design_matrix,
                                  weights_prior,
                                  level_variance_prior,
                                  observation_noise_variance_prior,
                                  slope_variance_prior=None,
                                  initial_level_prior=None,
                                  sparse_weights_nonzero_prob=None,
                                  seasonal_components=None):
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
    seasonal_components: An (optional) list of Seasonal components to include
      in the model. There are restrictions about what priors may be specified
      (InverseGamma drift scale prior).

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

  if isinstance(weights_prior, normal.Normal):
    # Canonicalize scalar normal priors as diagonal MVNs.
    # design_matrix must be defined, otherwise we threw an exception earlier.
    if isinstance(design_matrix, tf.linalg.LinearOperator):
      num_features = design_matrix.shape_tensor()[-1]
    else:
      num_features = prefer_static.dimension_size(design_matrix, -1)
    weights_prior = _tile_normal_to_mvn_diag(weights_prior, num_features)
  elif weights_prior is not None and not _is_multivariate_normal(weights_prior):
    raise ValueError('Weights prior must be a normal distribution or `None`.')
  if not isinstance(level_variance_prior, inverse_gamma.InverseGamma):
    raise ValueError(
        'Level variance prior must be an inverse gamma distribution.')
  if (slope_variance_prior is not None and
      not isinstance(slope_variance_prior, inverse_gamma.InverseGamma)):
    raise ValueError(
        'Slope variance prior must be an inverse gamma distribution; got: {}.'
        .format(slope_variance_prior))
  if not isinstance(observation_noise_variance_prior,
                    inverse_gamma.InverseGamma):
    raise ValueError('Observation noise variance prior must be an inverse '
                     'gamma distribution.')
  if seasonal_components is None:
    seasonal_components = []
  for seasonal_component in seasonal_components:
    if not seasonal_component.allow_drift:
      raise NotImplementedError(
          'Only seasonality with drift is supported by Gibbs sampling.')
    prior = seasonal_component.get_parameter('drift_scale').prior
    # TODO(kloveless): Create a helper function to help with this verification,
    # to be shared with seasonal.py.
    if not (isinstance(prior, transformed_distribution.TransformedDistribution)
            and isinstance(prior.bijector, invert.Invert) and
            isinstance(prior.bijector.bijector, square.Square) and
            isinstance(prior.distribution, inverse_gamma.InverseGamma)):
      raise NotImplementedError(
          'Seasonal components drift scale must be the square root of an '
          'inverse gamma distribution; got {}'.format(prior))

  sqrt = invert.Invert(
      square.Square())  # Converts variance priors to scale priors.
  components = []

  # Level or trend component.
  if slope_variance_prior:
    components.append(
        local_linear_trend.LocalLinearTrend(
            observed_time_series=observed_time_series,
            level_scale_prior=sqrt(level_variance_prior),
            slope_scale_prior=sqrt(slope_variance_prior),
            initial_level_prior=initial_level_prior,
            name='local_linear_trend'))
  else:
    components.append(
        local_level.LocalLevel(
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
        regression.LinearRegression(
            design_matrix=design_matrix,
            weights_prior=weights_prior,
            name='regression'))
  model = sum_lib.Sum(
      components + seasonal_components,
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


def get_seasonal_latents_shape(timeseries, model, num_chains=()):
  """Computes the shape of seasonal latents.

  Args:
    timeseries: Timeseries that is being modeled. Used to extract the timeseries
      length and batch shape.
    model: The `sts.Sum` model that the seasonal components will be found in.
      Must be a Gibbs-samplable model built with
      `build_model_for_gibbs_fitting`.
    num_chains: Optional int to indicate the number of parallel MCMC chains.
      Default to an empty tuple to sample a single chain.

  Returns:
    A shape list.
  """
  _, _, seasonal_indices_and_components = _get_components_from_model(model)

  seasonal_total_size = 0
  for _, seasonal_component in seasonal_indices_and_components:
    seasonal_total_size += seasonal_component.latent_size

  batch_shape = prefer_static.concat(
      [num_chains, prefer_static.shape(timeseries)[:-1]], axis=-1)
  timeseries_shape = prefer_static.shape(timeseries)[-1:]
  # Shape of all the seasonality component levels added together.
  seasonal_levels_shape = [seasonal_total_size]
  return prefer_static.concat(
      [batch_shape, timeseries_shape, seasonal_levels_shape], axis=0)


def fit_with_gibbs_sampling(model,
                            observed_time_series,
                            num_chains=(),
                            num_results=2000,
                            num_warmup_steps=200,
                            initial_state=None,
                            seed=None,
                            default_pseudo_observations=None,
                            experimental_use_dynamic_cholesky=False,
                            experimental_use_weight_adjustment=False):
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
    experimental_use_weight_adjustment: Optional bool - use a nonstandard
      update for the posterior precision of the weight in case of a spike and
      slab sampler.

  Returns:
    model: A `GibbsSamplerState` structure of posterior samples.
  """
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
  (_, level_component), _, seasonal_indices_and_components = (
      _get_components_from_model(model))
  if isinstance(level_component, local_linear_trend.LocalLinearTrend):
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
        seed=None,  # Set below.
        seasonal_drift_scales=tf.ones(
            prefer_static.concat(
                [batch_shape, [len(seasonal_indices_and_components)]], axis=0),
            dtype=dtype),
        seasonal_levels=tf.zeros(
            get_seasonal_latents_shape(
                observed_time_series, model, num_chains=num_chains),
            dtype=dtype))

  if isinstance(seed, six.integer_types):
    tf.random.set_seed(seed)

  # Always use the passed-in `seed` arg, ignoring any seed in the initial state.
  initial_state = initial_state._replace(
      seed=samplers.sanitize_seed(seed, salt='initial_GibbsSamplerState'))

  sampler_loop_body = _build_sampler_loop_body(
      model=model,
      observed_time_series=observed_time_series,
      is_missing=is_missing,
      default_pseudo_observations=default_pseudo_observations,
      experimental_use_dynamic_cholesky=experimental_use_dynamic_cholesky,
      experimental_use_weight_adjustment=experimental_use_weight_adjustment
  )

  samples = tf.scan(sampler_loop_body,
                    np.arange(num_warmup_steps + num_results), initial_state)
  return tf.nest.map_structure(lambda x: x[num_warmup_steps:], samples)


def model_parameter_samples_from_gibbs_samples(model, gibbs_samples):
  """Constructs parameter samples to match the model (e.g. for sts.forecast).

  This unpacks the Gibbs samples, dropping the latents, to match the order
  needed for `make_state_space_model`.

  Args:
    model: A `tfd.sts.StructuralTimeSeries` model instance. This must be of the
      form constructed by `build_model_for_gibbs_sampling`.
    gibbs_samples: A `GibbsSamplerState` instance, presumably from
      `fit_with_gibbs_sampling`.

  Returns:
    A set of posterior samples, that can be used with `make_state_space_model`
    or `sts.forecast`.
  """
  # Make use of the indices in the model to avoid requiring a specific
  # order of components.
  ((level_component_index, level_component), (regression_component_index, _),
   seasonal_indices_and_components) = (
       _get_components_from_model(model))
  seasonal_index_set = set([x[0] for x in seasonal_indices_and_components])
  model_parameter_samples = (gibbs_samples.observation_noise_scale,)

  # The last axis is the seasonal axis.
  seasonal_parameter_samples = tf.unstack(
      gibbs_samples.seasonal_drift_scales, axis=-1)
  # Allow seasonal components to not be consecutive by tracking how many
  # have been unpacked.
  seasonal_component_index = 0
  for index in range(len(model.components)):
    if index == level_component_index:
      model_parameter_samples += (gibbs_samples.level_scale,)
      if isinstance(level_component, local_linear_trend.LocalLinearTrend):
        model_parameter_samples += (gibbs_samples.slope_scale,)
    elif index == regression_component_index:
      model_parameter_samples += (gibbs_samples.weights,)
    elif index in seasonal_index_set:
      model_parameter_samples += (
          seasonal_parameter_samples[seasonal_component_index],)
      seasonal_component_index += 1

  return model_parameter_samples


def one_step_predictive(model,
                        posterior_samples,
                        num_forecast_steps=0,
                        original_mean=0.,
                        original_scale=1.,
                        thin_every=1,
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
      Default value: `1`.
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
      ([thinned_samples.level] if use_zero_step_prediction else [  # pylint:disable=g-long-ternary
          thinned_samples.level[..., :1],  # t == 0
          (thinned_samples.level[..., :-1] + thinned_samples.slope[..., :-1]
          )  # 1 <= t < T. Constructs the next level from previous level
          # and previous slope.
      ]) + ([forecast_level] if num_forecast_steps > 0 else []),
      axis=-1)

  # Calculate the seasonal effect at each time point.
  _, _, seasonal_indices_and_components = _get_components_from_model(model)
  if seasonal_indices_and_components and num_forecast_steps > 0:
    raise NotImplementedError(
        'Forecasting with one_step_predictive is not supported with '
        'seasonality. Instead, use fit_with_gibbs_sampling with missing '
        'values.')
  seasonal_pred = _compute_seasonal_effect_from_levels(
      thinned_samples.seasonal_levels,
      [x[1] for x in seasonal_indices_and_components],
      default_dtype=dtype)

  design_matrix = _get_design_matrix(model)
  if design_matrix is not None:
    design_matrix = design_matrix.to_dense()[:num_observed_steps +
                                             num_forecast_steps]
    regression_effect = tf.linalg.matvec(design_matrix, thinned_samples.weights)
  else:
    regression_effect = 0

  y_mean = ((level_pred + seasonal_pred + regression_effect) *
            original_scale[..., tf.newaxis] + original_mean[..., tf.newaxis])

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
  return mixture_same_family.MixtureSameFamily(
      mixture_distribution=categorical.Categorical(
          logits=tf.zeros([num_posterior_draws], dtype=y_mean.dtype)),
      components_distribution=normal.Normal(
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
                      seasonal_components=None,
                      seasonal_drift_scales=None,
                      is_missing=None,
                      sample_shape=(),
                      seed=None):
  """Uses Durbin-Koopman sampling to resample the model's latents.

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
    seasonal_components: An optional list of sts.Season components. This and
      `seasonal_drift_scales` must have the same number of seasonal components.
    seasonal_drift_scales: An optional float scalar `Tensor` (may contain batch
      dimensions) of shape [num_seasonal_components].
    is_missing: Optional `bool` `Tensor` missingness mask.
    sample_shape: Optional `int` `Tensor` shape of samples to draw.
    seed: `int` `Tensor` of shape `[2]` controlling stateless sampling.

  Returns:
    latents: Float `Tensor` resampled latent level and (optional) seasonal,
       of shape `[..., num_timesteps, latent_size]`, where `...` concatenates
       the sample shape with any batch shape from `observed_time_series`.
       The level latents are returned first, then each of the seasonal
       components.
  """
  if seasonal_components is None:
    seasonal_components = []

  num_timesteps = prefer_static.shape(observed_residuals)[-1]
  if slope_scale is None:
    local_ssm = local_level.LocalLevelStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale)
  else:
    local_ssm = local_linear_trend.LocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale,
        slope_scale=slope_scale)

  if seasonal_components:
    # It is assumed the the local ssm is added in first.
    ssms = [local_ssm]
    # Add an SSM for each season.
    for season_index, seasonal_component in enumerate(seasonal_components):
      seasonal_drift_scale = seasonal_drift_scales[..., season_index]
      ssms.append(
          seasonal_component.make_state_space_model(
              num_timesteps=num_timesteps,
              initial_state_prior=seasonal_component.initial_state_prior,
              param_vals={},
              drift_scale=seasonal_drift_scale,
              # Set this to exactly 0, since we want the observation noise
              # of the entire additive space to match observation_noise_scale,
              # but that is already specified on the local ssm.
              observation_noise_scale=0.,
          ))
    # No observation_noise_scale is needed here since it is taken from the local
    # ssm (which is equivalent).
    ssm = sum_lib.AdditiveStateSpaceModel(ssms)
  else:
    # Only the local value needs to be sampled, no addition needed.
    ssm = local_ssm

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
  posterior = sample_parameters.normal_scale_posterior_inverse_gamma_conjugate(
      prior, observed_residuals, is_missing)
  return sample_parameters.sample_with_optional_upper_bound(
      posterior, seed=seed)


def _get_components_from_model(model):
  """Returns the split-apart components from an STS model.

  Args:
    model: A `tf.sts.StructuralTimeSeries` to split apart.

  Returns:
    A tuple of (index and level component, index and regression component,
    indices and seasonal components) or an exception.
    The regression component may be None (along with its index - (None, None)).

    Each 'index and component' is a tuple of (index, component), where index
    is the position in the model.
  """
  if not hasattr(model, 'supports_gibbs_sampling'):
    raise ValueError('This STS model does not support Gibbs sampling. Models '
                     'for Gibbs sampling must be created using the '
                     'method `build_model_for_gibbs_fitting`.')

  level_components = []
  regression_components = []
  seasonal_components = []

  for index, component in enumerate(model.components):
    if (isinstance(component, local_level.LocalLevel) or
        isinstance(component, local_linear_trend.LocalLinearTrend)):
      level_components.append((index, component))
    elif (isinstance(component, regression.LinearRegression) or
          isinstance(component, SpikeAndSlabSparseLinearRegression)):
      regression_components.append((index, component))
    elif isinstance(component, seasonal.Seasonal):
      seasonal_components.append((index, component))
    else:
      raise NotImplementedError(
          'Found unsupported model component for Gibbs Sampling: {}'.format(
              component))

  if len(level_components) != 1:
    raise ValueError(
        'Expected exactly one level component, found {} components.'.format(
            len(level_components)))
  level_component = level_components[0]

  regression_component = (None, None)
  if len(regression_components) > 1:
    raise ValueError(
        'Expected at most one regression component, found {} components.'
        .format(len(regression_components)))
  elif len(regression_components) == 1:
    regression_component = regression_components[0]
  return level_component, regression_component, seasonal_components


def _resample_seasonal_scales(
    seasonal_components,
    seasonal_latents,
    seasonal_seed,
):
  """Samples the scales of the seasonal components given their latents.

  Args:
    seasonal_components: A non-empty list of seasonal components.
    seasonal_latents: A tensor of shape [batch shape, timeseries length, latents
      across all seasonal components].
    seasonal_seed: The per-seasonal component list of seeds.

  Returns:
    A tensor of seasonal scale samples, of shape [batch shape,
    number of seasonal components].
  """
  seasonal_seeds = samplers.split_seed(
      seasonal_seed, n=len(seasonal_components), salt='sample_seasonal_scales')
  seasonal_drift_scales_list = []
  # Seasonal latents are in sequence, thus track the index.
  next_latent_index = 0
  for seasonal_component, seasonal_seed in itertools.zip_longest(
      seasonal_components, seasonal_seeds):
    # Select the latents for this seasonal component.
    num_component_latents = seasonal_component.latent_size
    seasonal_component_latents = seasonal_latents[
        ..., next_latent_index:next_latent_index + num_component_latents]
    next_latent_index += num_component_latents
    seasonal_drift_scales_list.append(
        seasonal_component.experimental_resample_drift_scale(
            latents=seasonal_component_latents,
            seed=seasonal_seed,
        ))
  number_of_latents = seasonal_latents.shape[-1]
  if next_latent_index != number_of_latents:
    raise TypeError(
        f'Some seasonal latent values were not used. {next_latent_index} were '
        'used, but there were {number_of_latents} latents. Are you using '
        'unsupported structual components?')

  return tf.stack(
      seasonal_drift_scales_list,
      # Seasonal component is the last dimension.
      axis=-1)


def _compute_seasonal_effect_from_levels(seasonal_levels, seasonal_components,
                                         default_dtype):
  """Given the seasonal levels, computes the effect on the timeseries."""
  if not seasonal_components:
    # Use a passed in dtype for the results for cases where the seasonal_levels
    # are just empty, and their dtype may not match the computation dtype since
    # they are default initialized.
    return tf.zeros(shape=(1), dtype=default_dtype)
  ret = tf.zeros(
      dtype=seasonal_levels.dtype,
      # Retain the batch shape.
      shape=seasonal_levels.shape[:-1])
  next_latent_index = 0
  for seasonal_component in seasonal_components:
    ret += seasonal_levels[
        ...,
        # The observation matrix is always just [1., 0., ..., 0.] for
        # the seasonal state space models. Thus as an efficiency gain,
        # just access it directly rather than constructing all the
        # observation matrices at each timestep.
        next_latent_index]
    next_latent_index += seasonal_component.latent_size
  return ret


def _build_sampler_loop_body(model,
                             observed_time_series,
                             is_missing=None,
                             default_pseudo_observations=None,
                             experimental_use_dynamic_cholesky=False,
                             experimental_use_weight_adjustment=False):
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
    experimental_use_weight_adjustment: Optional bool - use a nonstandard
      update for the posterior precision of the weight in case of a spike and
      slab sampler.

  Returns:
    sampler_loop_body: Python callable that performs a single cycle of Gibbs
      sampling. Its first argument is a `GibbsSamplerState`, and it returns a
      new `GibbsSamplerState`. The second argument (passed by `tf.scan`) is
      ignored.
  """
  if JAX_MODE and experimental_use_dynamic_cholesky:
    raise ValueError('Dynamic Cholesky decomposition not supported in JAX')
  ((_, level_component), (_, regression_component),
   seasonal_indices_and_components) = _get_components_from_model(model)
  seasonal_components = [x[1] for x in seasonal_indices_and_components]
  model_has_slope = isinstance(level_component,
                               local_linear_trend.LocalLinearTrend)

  if regression_component is not None:
    model_has_spike_slab_regression = isinstance(
        regression_component, SpikeAndSlabSparseLinearRegression)

  if is_missing is not None:  # Ensure series does not contain NaNs.
    observed_time_series = tf.where(is_missing,
                                    tf.zeros_like(observed_time_series),
                                    observed_time_series)
  num_observed_steps = prefer_static.shape(observed_time_series)[-1]

  design_matrix = _get_design_matrix(model)
  num_missing = 0.
  if design_matrix is not None:
    design_matrix = design_matrix.to_dense()[:num_observed_steps]
    if is_missing is None:
      num_missing = 0.
      is_missing = tf.zeros(num_observed_steps, dtype=bool)
    else:
      # Replace design matrix with zeros at unobserved timesteps. This ensures
      # they will not affect the posterior on weights.
      design_matrix = tf.where(is_missing[..., tf.newaxis],
                               tf.zeros_like(design_matrix), design_matrix)
      num_missing = tf.reduce_sum(
          tf.cast(is_missing, design_matrix.dtype), axis=-1)

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
          num_missing=num_missing,
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

  # Sub-selects in `forward_filter_sequential` take up a lot of the runtime
  # with a dynamic Cholesky, but compiling here seems to help.
  # TODO(b/234726324): Should this always be compiled?
  if experimental_use_dynamic_cholesky:
    resample_latents = tf.function(
        jit_compile=True, autograph=False)(
            _resample_latents)
    resample_scale = tf.function(
        jit_compile=True, autograph=False)(
            _resample_scale)
    resample_seasonal_scales = tf.function(
        jit_compile=True, autograph=False)(
            _resample_seasonal_scales)
  else:
    resample_latents = _resample_latents
    resample_scale = _resample_scale
    resample_seasonal_scales = _resample_seasonal_scales

  def sampler_loop_body(previous_sample, _):
    """Runs one sampler iteration, resampling all model variables."""

    (weights_seed, latent_seed, observation_noise_scale_seed, level_scale_seed,
     loop_seed, seasonal_seed) = samplers.split_seed(
         previous_sample.seed, n=6, salt='sampler_loop_body')
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
      all_seasons_effect = _compute_seasonal_effect_from_levels(
          previous_sample.seasonal_levels,
          [x[1] for x in seasonal_indices_and_components],
          default_dtype=previous_sample.level.dtype)
      target_residuals = (
          observed_time_series - previous_sample.level - all_seasons_effect)
      if model_has_spike_slab_regression:
        if experimental_use_weight_adjustment:
          previous_observation_noise_variance = tf.square(
              previous_sample.observation_noise_scale)
        else:
          previous_observation_noise_variance = 1.
        targets = tf.where(is_missing, tf.zeros_like(observed_time_series),
                           target_residuals)
        (observation_noise_variance, weights
        ) = spike_and_slab_sampler.sample_noise_variance_and_weights(
            initial_nonzeros=tf.math.logical_or(
                tf.not_equal(previous_sample.weights, 0.), pin_to_nonzero),
            previous_observation_noise_variance=previous_observation_noise_variance,
            targets=targets,
            seed=weights_seed)
        observation_noise_scale = tf.sqrt(observation_noise_variance)

      else:
        weights = _resample_weights(
            design_matrix=design_matrix,
            target_residuals=target_residuals,
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

    latents = resample_latents(
        observed_residuals=regression_residuals,
        level_scale=previous_sample.level_scale,
        slope_scale=previous_sample.slope_scale if model_has_slope else None,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=level_component.initial_state_prior,
        is_missing=is_missing,
        seed=latent_seed,
        seasonal_components=seasonal_components,
        seasonal_drift_scales=previous_sample.seasonal_drift_scales)
    level = latents[..., 0]
    level_residuals = level[..., 1:] - level[..., :-1]
    if model_has_slope:
      slope = latents[..., 1]
      level_residuals -= slope[..., :-1]
      slope_residuals = slope[..., 1:] - slope[..., :-1]
    next_latent_index = 2 if model_has_slope else 1
    if seasonal_components:
      num_seasonal_latents = sum(
          [component.latent_size for component in seasonal_components])
      # Seasonal levels are after the local latents.
      seasonal_levels = latents[..., next_latent_index:next_latent_index +
                                num_seasonal_latents]
      next_latent_index += num_seasonal_latents
      seasonal_drift_scales = resample_seasonal_scales(seasonal_components,
                                                       seasonal_levels,
                                                       seasonal_seed)
    else:
      # When there are no seasons, the lists will be empty and tf.stack
      # will fail. Thus, just re-use whatever (unchanging) shape was
      # initially set so it is unchanging on each iteration.
      seasonal_drift_scales = previous_sample.seasonal_drift_scales
      seasonal_levels = previous_sample.seasonal_levels
    all_seasons_effect = _compute_seasonal_effect_from_levels(
        seasonal_levels, [x[1] for x in seasonal_indices_and_components],
        default_dtype=level.dtype)

    number_of_latents = latents.shape[-1]
    if next_latent_index != number_of_latents:
      raise TypeError(
          f'Some latent values were not used. {next_latent_index} were used, '
          'but there were {number_of_latents} latents. Are you using '
          'unsupported structual components?')

    # Estimate level scale from the empirical changes in level.
    level_scale = resample_scale(
        prior=level_scale_variance_prior,
        observed_residuals=level_residuals,
        is_missing=None,
        seed=level_scale_seed)
    if model_has_slope:
      slope_scale = resample_scale(
          prior=slope_scale_variance_prior,
          observed_residuals=slope_residuals,
          is_missing=None,
          seed=slope_scale_seed)
    if not (regression_component and model_has_spike_slab_regression):
      # Estimate noise scale from the residuals.
      observation_noise_scale = resample_scale(
          prior=observation_noise_variance_prior,
          observed_residuals=regression_residuals - level - all_seasons_effect,
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
        seasonal_drift_scales=seasonal_drift_scales,
        seasonal_levels=seasonal_levels,
        seed=loop_seed)

  return sampler_loop_body
