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
single LocalLevel component with a linear regression component, with conjugate
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
- To support regression with non-Gaussian (e.g., spike-and-slab) priors, one
  would need to write a custom prior-specific sampler, analogous to the current
  `resample_weights` function.
- For specific models it may be possible to implement an efficient prior
  sampling algorithm, analagous to `LocalLevelStateSpaceModel._joint_sample_n`.
  This may be significantly faster than the generic sampler and can speed up
  the posterior sampling step for the latent trajectory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import sts
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.distributions import normal_conjugate_posteriors
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.sts.internal import util as sts_util

# The sampler state stores current values for each model parameter,
# and auxiliary quantities such as the latent level. It should have the property
# that `model.make_state_space_model(num_timesteps, GibbsSamplerState(...))`
# behaves properly -- i.e., that the state contains all model
# parameters *in the same order* as they are listed in `model.parameters`. This
# is currently enforced by construction in `build_gibbs_fittable_model`.
GibbsSamplerState = collections.namedtuple('GibbsSamplerState', [
    'observation_noise_scale', 'level_scale', 'weights', 'level', 'seed'])


def build_model_for_gibbs_fitting(observed_time_series,
                                  design_matrix,
                                  weights_prior,
                                  level_variance_prior,
                                  observation_noise_variance_prior):
  """Builds a StructuralTimeSeries model instance that supports Gibbs sampling.

  To support Gibbs sampling, a model must have have conjugate priors on all
  scale and weight parameters, and must be constructed so that
  `model.parameters` matches the parameters and ordering specified by the
  the `GibbsSamplerState` namedtuple. Currently, this includes (only) models
  consisting of the sum of a LocalLevel and a LinearRegression component.

  Args:
    observed_time_series: optional `float` `Tensor` of shape [..., T, 1]`
      (omitting the trailing unit dimension is also supported when `T > 1`),
      specifying an observed time series. May optionally be an instance of
      `tfp.sts.MaskedTimeSeries`, which includes a mask `Tensor` to specify
      timesteps with missing observations.
    design_matrix: float `Tensor` of shape `concat([batch_shape,
      [num_timesteps, num_features]])`. This may also optionally be
      an instance of `tf.linalg.LinearOperator`.
    weights_prior: An instance of `tfd.Normal` representing a scalar prior on
      each regression weight. May have batch shape broadcastable to the batch
      shape of `observed_time_series`.
    level_variance_prior: An instance of `tfd.InverseGamma` representing a prior
      on the level variance (`level_scale**2`) of a local level model. May have
      batch shape broadcastable to the batch shape of `observed_time_series`.
    observation_noise_variance_prior: An instance of `tfd.InverseGamma`
      representing a prior on the observation noise variance (
      `observation_noise_scale**2`). May have batch shape broadcastable to the
      batch shape of `observed_time_series`.
  Returns:
    model: A `tfp.sts.StructuralTimeSeries` model instance.
  """
  if not isinstance(weights_prior, tfd.Normal):
    raise ValueError('Weights prior must be a univariate normal distribution.')
  if not isinstance(level_variance_prior, tfd.InverseGamma):
    raise ValueError(
        'Level variance prior must be an inverse gamma distribution.')
  if not isinstance(observation_noise_variance_prior, tfd.InverseGamma):
    raise ValueError('Observation noise variance prior must be an inverse '
                     'gamma distribution.')

  sqrt = tfb.Invert(tfb.Square())  # Converts variance priors to scale priors.
  local_level = sts.LocalLevel(observed_time_series=observed_time_series,
                               level_scale_prior=sqrt(level_variance_prior),
                               name='local_level')
  regression = sts.LinearRegression(design_matrix=design_matrix,
                                    weights_prior=weights_prior,
                                    name='regression')
  model = sts.Sum([local_level, regression],
                  observed_time_series=observed_time_series,
                  observation_noise_scale_prior=sqrt(
                      observation_noise_variance_prior),
                  # The Gibbs sampling steps in this file do not account for an
                  # offset to the observed series. Instead, we assume the
                  # observed series has already been centered and
                  # scale-normalized.
                  constant_offset=0.)
  model.supports_gibbs_sampling = True
  return model


def _get_design_matrix(model):
  """Returns the design matrix for an STS model with a regression component."""
  design_matrices = [component.design_matrix for component in model.components
                     if hasattr(component, 'design_matrix')]
  if not design_matrices:
    raise ValueError('Model does not contain a regression component.')
  if len(design_matrices) > 1:
    raise ValueError('Model contains multiple regression components.')
  return design_matrices[0]


def fit_with_gibbs_sampling(model,
                            observed_time_series,
                            num_results=2000,
                            num_warmup_steps=200,
                            compile_steps_with_xla=False,
                            initial_state=None,
                            seed=None):
  """Fits parameters for an STS model using Gibbs sampling."""
  if not hasattr(model, 'supports_gibbs_sampling'):
    raise ValueError('This STS model does not support Gibbs sampling. Models '
                     'for Gibbs sampling must be created using the '
                     'method `build_model_for_gibbs_fitting`.')

  [
      observed_time_series,
      is_missing
  ] = sts_util.canonicalize_observed_time_series_with_mask(
      observed_time_series)
  dtype = observed_time_series.dtype

  # The canonicalized time series always has trailing dimension `1`,
  # because although LinearGaussianSSMs support vector observations, STS models
  # describe scalar time series only. For our purposes it'll be cleaner to
  # remove this dimension.
  observed_time_series = observed_time_series[..., 0]

  batch_shape = prefer_static.shape(observed_time_series)[:-1]
  if initial_state is None:
    initial_state = GibbsSamplerState(
        observation_noise_scale=tf.ones(batch_shape, dtype=dtype),
        level_scale=tf.ones(batch_shape, dtype=dtype),
        weights=tf.zeros(prefer_static.concat([
            batch_shape,
            _get_design_matrix(model).shape[-1:]], axis=0), dtype=dtype),
        level=tf.zeros_like(observed_time_series),
        seed=None)  # Set below.

  if seed and isinstance(seed, six.integer_types):
    tf.random.set_seed(seed)

  # Always use the passed-in `seed` arg, ignoring any seed in the initial state.
  seeded_state = initial_state._asdict()
  seeded_state['seed'] = samplers.sanitize_seed(
      seed, salt='initial_GibbsSamplerState')
  initial_state = GibbsSamplerState(**seeded_state)

  sampler_loop_body = _build_sampler_loop_body(
      model, observed_time_series, is_missing,
      compile_steps_with_xla=compile_steps_with_xla,
      seed=seed)  # This is still an `int` seed, because the InverseGamma
                  # sampler currently requires stateful semantics.

  samples = tf.scan(sampler_loop_body,
                    np.arange(num_warmup_steps + num_results),
                    initial_state)
  return tf.nest.map_structure(lambda x: x[num_warmup_steps:], samples)


def one_step_predictive(model,
                        posterior_samples,
                        num_forecast_steps=0,
                        original_mean=0.,
                        original_scale=1.,
                        thin_every=10):
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
  Returns:
    predictive_dist: A `tfd.MixtureSameFamily` instance of event shape
      `[num_timesteps + num_forecast_steps]` representing the predictive
      distribution of each timestep given previous timesteps.
  """
  dtype = dtype_util.common_dtype([
      posterior_samples.level_scale.dtype,
      posterior_samples.observation_noise_scale.dtype,
      posterior_samples.level.dtype,
      original_mean,
      original_scale], dtype_hint=tf.float32)
  num_observed_steps = prefer_static.shape(posterior_samples.level)[-1]

  original_mean = tf.convert_to_tensor(original_mean, dtype=dtype)
  original_scale = tf.convert_to_tensor(original_scale, dtype=dtype)
  thinned_samples = tf.nest.map_structure(lambda x: x[::thin_every],
                                          posterior_samples)

  # The local level model expects that the level at step t+1 is equal
  # to the level at step t (plus transition noise of scale 'level_scale', which
  # we account for below).
  if num_forecast_steps > 0:
    num_batch_dims = prefer_static.rank_from_shape(
        prefer_static.shape(thinned_samples.level)) - 2
    forecast_level = tf.tile(thinned_samples.level[..., -1:],
                             tf.concat([tf.ones([num_batch_dims + 1],
                                                dtype=tf.int32),
                                        [num_forecast_steps]], axis=0))
  level_pred = tf.concat([thinned_samples.level[..., :1],  # t == 0
                          thinned_samples.level[..., :-1]  # 1 <= t < T
                         ] + ([forecast_level] if num_forecast_steps > 0
                              else []),
                         axis=-1)

  design_matrix = _get_design_matrix(
      model).to_dense()[:num_observed_steps + num_forecast_steps]
  regression_effect = tf.linalg.matvec(design_matrix, thinned_samples.weights)

  y_mean = ((level_pred + regression_effect) *
            original_scale[..., tf.newaxis] + original_mean[..., tf.newaxis])

  num_steps_from_last_observation = tf.concat([
      tf.ones([num_observed_steps], dtype=dtype),
      tf.range(1, num_forecast_steps + 1, dtype=dtype)], axis=0)
  y_scale = (original_scale * tf.sqrt(
      thinned_samples.observation_noise_scale[..., tf.newaxis]**2 +
      thinned_samples.level_scale[..., tf.newaxis]**2 *
      num_steps_from_last_observation))

  num_posterior_draws = prefer_static.shape(y_mean)[0]
  return tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          logits=tf.zeros([num_posterior_draws], dtype=y_mean.dtype)),
      components_distribution=tfd.Normal(
          loc=dist_util.move_dimension(y_mean, 0, -1),
          scale=dist_util.move_dimension(y_scale, 0, -1)))


def _resample_weights(design_matrix, target_residuals, observation_noise_scale,
                      weights_prior_scale, is_missing=None, seed=None):
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
    design_matrix: Float `Tensor` design matrix of shape
      `[..., num_timesteps, num_features]`.
    target_residuals:  Float `Tensor` of shape `[..., num_observations]`
    observation_noise_scale: Scalar float `Tensor` (with optional batch shape)
      standard deviation of the iid observation noise.
    weights_prior_scale: Scalar float `Tensor` (with optional batch shape)
      specifying the standard deviation of the Normal prior on regression
      weights.
    is_missing: Optional `bool` `Tensor` of shape `[..., num_timesteps]`. A
      `True` value indicates that the observation for that timestep is missing.
    seed: Optional `Python` `int` seed controlling the sampled values.
  Returns:
    weights: Float `Tensor` of shape `[..., num_features]`, sampled from
      the conditional posterior `p(weights | target_residuals,
      observation_noise_scale, weights_prior_scale)`.
  """
  if is_missing is not None:
    # Replace design matrix with zeros at unobserved timesteps. This ensures
    # they will not affect the posterior on weights.
    design_matrix = tf.where(is_missing[..., tf.newaxis],
                             tf.zeros_like(design_matrix),
                             design_matrix)
  design_shape = prefer_static.shape(design_matrix)
  num_outputs = design_shape[-2]
  num_features = design_shape[-1]

  iid_prior_scale = tf.linalg.LinearOperatorScaledIdentity(
      num_rows=num_features, multiplier=weights_prior_scale)
  iid_likelihood_scale = tf.linalg.LinearOperatorScaledIdentity(
      num_rows=num_outputs, multiplier=observation_noise_scale)
  weights_mean, weights_prec = (
      normal_conjugate_posteriors.mvn_conjugate_linear_update(
          linear_transformation=design_matrix,
          observation=target_residuals,
          prior_scale=iid_prior_scale,
          likelihood_scale=iid_likelihood_scale))
  sampled_weights = weights_prec.cholesky().solvevec(
      samplers.normal(
          shape=prefer_static.shape(weights_mean),
          dtype=design_matrix.dtype, seed=seed), adjoint=True)
  return weights_mean + sampled_weights


# `resample_level` requires an explicit builder function because the compiled
# code can only accept `Tensor` arguments, but we need to pass in the
# initial state prior which is a tfd.Distribution.
def _build_resample_level_fn(initial_state_prior,
                             is_missing=None,
                             compile_with_xla=False):
  """Builds a method to sample the latent level from its Gibbs posterior."""

  @tf.function(autograph=False, experimental_compile=compile_with_xla)
  def resample_level(observed_residuals,
                     level_scale,
                     observation_noise_scale,
                     sample_shape=(),
                     seed=None):
    """Uses Durbin-Koopman sampling to resample the latent level.

    Durbin-Koopman sampling [1] is an efficient algorithm to sample from the
    posterior latents of a linear Gaussian state space model. This method
    implements the algorithm, specialized to the case of a one-dimensional
    latent local level model.

    [1] Durbin, J. and Koopman, S.J. (2002) A simple and efficient simulation
        smoother for state space time series analysis.

    Args:
      observed_residuals: Float `Tensor` of shape `[..., num_observations]`,
        specifying the centered observations `(x - loc)`.
      level_scale: Float scalar `Tensor` (may contain batch dimensions)
        specifying the standard deviation of the level random walk steps.
      observation_noise_scale: Float scalar `Tensor` (may contain batch
        dimensions) specifying the standard deviation of the observation noise.
      sample_shape: Optional `int` `Tensor` shape of samples to draw.
      seed: `int` `Tensor` of shape `[2]` controlling stateless sampling.
    Returns:
      level: Float `Tensor` resampled latent level, of shape
        `[..., num_timesteps]`, where `...` concatenates the sample shape
        with any batch shape from `observed_time_series`.
    """

    num_timesteps = prefer_static.shape(observed_residuals)[-1]
    ssm = sts.LocalLevelStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale)
    return ssm.posterior_sample(observed_residuals[..., tf.newaxis],
                                sample_shape=sample_shape,
                                mask=is_missing,
                                seed=seed)[..., 0]

  return resample_level


def _resample_scale(prior_concentration, prior_scale,
                    observed_residuals, is_missing=None, seed=None):
  """Samples a scale parameter from its conditional posterior.

  We assume the conjugate InverseGamma->Normal model:

  ```
  scale ~ Sqrt(InverseGamma(prior_concentration, prior_scale))
  for i in [1, ..., num_observations]:
    x[i] ~ Normal(loc, scale)
  ```

  in which `loc` is known, and return a sample from `p(scale | x)`.

  Args:
    prior_concentration: Float `Tensor` concentration parameter of the
      InverseGamma prior distribution.
    prior_scale: Float `Tensor` scale parameter of the InverseGamma prior
      distribution.
    observed_residuals: Float `Tensor` of shape `[..., num_observations]`,
      specifying the centered observations `(x - loc)`.
    is_missing: Optional `bool` `Tensor` of shape `[..., num_observations]`. A
      `True` value indicates that the corresponding observation is missing.
    seed: Optional `Python` `int` seed controlling the sampled value.
  Returns:
    sampled_scale: A `Tensor` sample from the posterior `p(scale | x)`.
  """
  if is_missing is not None:
    num_missing = tf.reduce_sum(tf.cast(is_missing, observed_residuals.dtype),
                                axis=-1)
  num_observations = prefer_static.shape(observed_residuals)[-1]
  if is_missing is not None:
    observed_residuals = tf.where(is_missing,
                                  tf.zeros_like(observed_residuals),
                                  observed_residuals)
    num_observations -= num_missing

  variance_posterior = tfd.InverseGamma(
      concentration=prior_concentration + num_observations / 2.,
      scale=prior_scale + tf.reduce_sum(
          tf.square(observed_residuals), axis=-1) / 2.)
  return tf.sqrt(variance_posterior.sample(seed=seed))


def _build_sampler_loop_body(model,
                             observed_time_series,
                             is_missing=None,
                             compile_steps_with_xla=False,
                             seed=None):
  """Builds a Gibbs sampler for the given model and observed data.

  Args:
    model: A `tf.sts.StructuralTimeSeries` model instance. This must be of the
      form constructed by `build_model_for_gibbs_sampling`.
    observed_time_series: Float `Tensor` time series of shape
      `[..., num_timesteps]`.
    is_missing: Optional `bool` `Tensor` of shape `[..., num_timesteps]`. A
      `True` value indicates that the observation for that timestep is missing.
    compile_steps_with_xla: Optional Python `bool`. If `True`, XLA compilation
      is used to accelerate sampling steps when supported.
    seed: Optional `Python` `int` seed controlling the sampled values.
  Returns:
    sampler_loop_body: Python callable that performs a single cycle of Gibbs
      sampling. Its first argument is a `GibbsSamplerState`, and it returns a
      new `GibbsSamplerState`. The second argument (passed by `tf.scan`) is
      ignored.
  """

  # Require that the model has exactly the parameters expected by
  # `GibbsSamplerState`.
  observation_noise_param, level_scale_param, weights_param = model.parameters
  if (('observation_noise' not in observation_noise_param.name) or
      ('level_scale' not in level_scale_param.name) or
      ('weights' not in weights_param.name)):
    raise ValueError('Model parameters {} do not match the expected sampler '
                     'state.'.format(model.parameters))

  level_component = model.components[0]
  if not isinstance(level_component, sts.LocalLevel):
    raise ValueError('Expected the first model component to be an instance of '
                     '`tfp.sts.LocalLevel`; instead saw {}'.format(
                         level_component))

  if is_missing is not None:  # Ensure series does not contain NaNs.
    observed_time_series = tf.where(is_missing,
                                    tf.zeros_like(observed_time_series),
                                    observed_time_series)

  num_observed_steps = prefer_static.shape(observed_time_series)[-1]
  design_matrix = _get_design_matrix(model).to_dense()[:num_observed_steps]

  # Compile the functions that sample from Gibbs conditional posteriors.
  # In principle, we should XLA-compile the entire loop body or even the entire
  # `fit_with_gibbs_sampling` loop. However, XLA can't currently compile the
  # gamma sampling op inside `_resample_scale` (b/141253568), so for now we
  # leave that method uncompiled but compile the other two sampling steps.
  # Empirically, the vast majority of sampling time is spent in
  # `resample_level`, so compiling it gives us most of the wins.
  # TODO(davmre): Wrap the entire sampling loop in `tf.function` while still
  #  XLA-compiling these pieces as appropriate.
  # TODO(b/141253568): XLA-compile the entire sampling loop.
  compiled_resample_level = _build_resample_level_fn(
      initial_state_prior=level_component.initial_state_prior,
      is_missing=is_missing,
      compile_with_xla=compile_steps_with_xla)
  compiled_resample_weights = tf.function(
      _resample_weights, autograph=False,
      experimental_compile=compile_steps_with_xla)
  compiled_resample_scale = tf.function(
      _resample_scale, autograph=False,
      experimental_compile=False)

  # Untransform scale priors -> variance priors by reaching thru Sqrt bijector.
  level_scale_variance_prior = level_scale_param.prior.distribution
  observation_noise_variance_prior = observation_noise_param.prior.distribution

  # InverseGamma samplers are currently stateful, so we only need (and want)
  # a single seed for each, shared across loop iterations.
  strm = tfp_util.SeedStream(seed, salt='_sampler_loop_body')
  observation_noise_scale_seed = strm()
  level_scale_seed = strm()

  def sampler_loop_body(previous_sample, _):
    """Runs one sampler iteration, resampling all model variables."""

    (weights_seed,
     level_seed,
     loop_seed) = samplers.split_seed(
         previous_sample.seed, n=3, salt='sampler_loop_body')

    # We encourage a reasonable initialization by sampling the weights first,
    # so at the first step they are regressed directly against the observed
    # time series. If we instead sampled the level first it might 'explain away'
    # some observed variation that we would ultimately prefer to explain through
    # the regression weights, because the level can represent arbitrary
    # variation, while the weights are limited to representing variation in the
    # subspace given by the design matrix.
    weights = compiled_resample_weights(
        design_matrix=design_matrix,
        target_residuals=(observed_time_series - previous_sample.level),
        observation_noise_scale=previous_sample.observation_noise_scale,
        weights_prior_scale=weights_param.prior.distribution.scale,
        is_missing=is_missing,
        seed=weights_seed)

    regression_residuals = observed_time_series - tf.linalg.matvec(
        design_matrix, weights)
    level = compiled_resample_level(
        observed_residuals=regression_residuals,
        level_scale=previous_sample.level_scale,
        observation_noise_scale=previous_sample.observation_noise_scale,
        seed=level_seed)

    # Estimate level scale from the empirical changes in level.
    level_scale = compiled_resample_scale(
        prior_scale=level_scale_variance_prior.scale,
        prior_concentration=level_scale_variance_prior.concentration,
        observed_residuals=level[..., 1:] - level[..., :-1],
        is_missing=None, seed=level_scale_seed)
    # Estimate noise scale from the residuals.
    observation_noise_scale = compiled_resample_scale(
        prior_scale=observation_noise_variance_prior.scale,
        prior_concentration=observation_noise_variance_prior.concentration,
        observed_residuals=regression_residuals - level,
        is_missing=is_missing, seed=observation_noise_scale_seed)

    return GibbsSamplerState(
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale,
        weights=weights,
        level=level,
        seed=loop_seed)
  return sampler_loop_body
