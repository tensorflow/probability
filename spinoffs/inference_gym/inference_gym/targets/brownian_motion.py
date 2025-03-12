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
"""Brownian Motion model."""

import functools

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from inference_gym.internal import data
from inference_gym.targets import bayesian_model
from inference_gym.targets import model
from inference_gym.targets.ground_truth import brownian_motion_missing_middle_observations
from inference_gym.targets.ground_truth import brownian_motion_unknown_scales_missing_middle_observations

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'BrownianMotion',
    'BrownianMotionMissingMiddleObservations',
    'BrownianMotionUnknownScales',
    'BrownianMotionUnknownScalesMissingMiddleObservations',
]

Root = tfd.JointDistributionCoroutine.Root


def brownian_motion_as_markov_chain(num_timesteps, innovation_noise_scale):
  return tfd.MarkovChain(
      initial_state_prior=tfd.Normal(loc=0., scale=innovation_noise_scale),
      transition_fn=lambda _, x_t: tfd.Normal(  # pylint: disable=g-long-lambda
          loc=x_t, scale=innovation_noise_scale),
      num_steps=num_timesteps,
      name='locs')


def brownian_motion_prior_fn(num_timesteps,
                             innovation_noise_scale):
  """Generative process for the Brownian Motion model."""
  prior_loc = 0.
  new = yield Root(tfd.Normal(loc=prior_loc,
                              scale=innovation_noise_scale,
                              name='x_0'))
  for t in range(1, num_timesteps):
    new = yield tfd.Normal(loc=new,
                           scale=innovation_noise_scale,
                           name='x_{}'.format(t))


def brownian_motion_unknown_scales_prior_fn(
    num_timesteps, use_markov_chain, dtype
):
  """Generative process for the Brownian Motion model with unknown scales."""
  zero = tf.zeros([], dtype)
  innovation_noise_scale = yield Root(
      tfd.LogNormal(zero, 2.0, name='innovation_noise_scale')
  )
  _ = yield Root(tfd.LogNormal(zero, 2.0, name='observation_noise_scale'))
  if use_markov_chain:
    yield brownian_motion_as_markov_chain(
        num_timesteps=num_timesteps,
        innovation_noise_scale=innovation_noise_scale,
    )
  else:
    yield from brownian_motion_prior_fn(
        num_timesteps, innovation_noise_scale=innovation_noise_scale
    )


def brownian_motion_log_likelihood_fn(values,
                                      observed_locs,
                                      use_markov_chain,
                                      dtype,
                                      observation_noise_scale=None):
  """Likelihood of observed data under the Brownian Motion model."""
  if observation_noise_scale is None:
    (_, observation_noise_scale) = values[:2]
    latents = values[2] if use_markov_chain else tf.stack(values[2:], axis=-1)
  else:
    latents = values if use_markov_chain else tf.stack(values, axis=-1)

  observation_noise_scale = tf.convert_to_tensor(
      observation_noise_scale, dtype=dtype, name='observation_noise_scale')
  observed_locs = tf.cast(
      observed_locs,
      dtype=dtype,
      name='observed_locs',
  )
  is_observed = ~tf.math.is_nan(observed_locs)
  lps = tfd.Normal(
      loc=latents, scale=observation_noise_scale[..., tf.newaxis]).log_prob(
          tf.where(is_observed, observed_locs, 0.))
  return tf.reduce_sum(tf.where(is_observed, lps, 0.), axis=-1)


class BrownianMotion(bayesian_model.BayesianModel):
  """Construct the Brownian Motion model.

  This models a Brownian Motion process with a Gaussian observation model.

  ```none
  locs[0] ~ Normal(loc=0, scale=innovation_noise_scale)
  for t in range(1, num_timesteps):
    locs[t] ~ Normal(loc=locs[t - 1], scale=innovation_noise_scale)

  for t in range(num_timesteps):
    observed_locs[t] ~ Normal(loc=locs[t], scale=observation_noise_scale)
  ```

  This model supports missing observations, indicated by NaNs in the
  `observed_locs` parameter.
  """

  def __init__(self,
               observed_locs,
               innovation_noise_scale,
               observation_noise_scale,
               use_markov_chain=False,
               dtype=tf.float32,
               name='brownian_motion',
               pretty_name='Brownian Motion'):
    """Construct the Brownian Motion model.

    Args:
      observed_locs: Array of loc parameters with NaN value if loc is
        unobserved.
      innovation_noise_scale: Python `float`.
      observation_noise_scale: Python `float`.
      use_markov_chain: Python `bool` indicating whether to use the
        `MarkovChain` distribution in place of separate random variables for
        each time step. The default of `False` is for backwards compatibility;
        setting this to `True` should significantly improve performance.
      dtype: Dtype to use for floating point quantities.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """
    with tf.name_scope(name):
      num_timesteps = observed_locs.shape[0]
      innovation_noise_scale = tf.convert_to_tensor(
          innovation_noise_scale,
          dtype=dtype,
          name='innovation_noise_scale',
      )

      if use_markov_chain:
        self._prior_dist = brownian_motion_as_markov_chain(
            num_timesteps=num_timesteps,
            innovation_noise_scale=innovation_noise_scale)
      else:
        self._prior_dist = tfd.JointDistributionCoroutine(
            functools.partial(
                brownian_motion_prior_fn,
                num_timesteps=num_timesteps,
                innovation_noise_scale=innovation_noise_scale))

      self._log_likelihood_fn = functools.partial(
          brownian_motion_log_likelihood_fn,
          observation_noise_scale=observation_noise_scale,
          observed_locs=observed_locs,
          use_markov_chain=use_markov_chain,
          dtype=dtype)

      def _ext_identity(params):
        return tf.stack(params, axis=-1)

      def _ext_identity_markov_chain(params):
        return params

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=(_ext_identity_markov_chain
                      if use_markov_chain else _ext_identity),
                  pretty_name='Identity',
                  dtype=dtype,
              )
      }

    if use_markov_chain:
      event_space_bijector = tfb.Identity()
    else:
      event_space_bijector = type(
          self._prior_dist.dtype)(*([tfb.Identity()] * num_timesteps))
    super(BrownianMotion, self).__init__(
        default_event_space_bijector=event_space_bijector,
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _prior_distribution(self):
    return self._prior_dist

  def log_likelihood(self, value):
    return self._log_likelihood_fn(value)


class BrownianMotionMissingMiddleObservations(BrownianMotion):
  """A simple Brownian Motion with 30 timesteps where 10 are unobservable."""

  GROUND_TRUTH_MODULE = brownian_motion_missing_middle_observations

  def __init__(self, use_markov_chain=False, dtype=tf.float32):
    dataset = data.brownian_motion_missing_middle_observations()
    super(BrownianMotionMissingMiddleObservations, self).__init__(
        name='brownian_motion_missing_middle_observations',
        pretty_name='Brownian Motion Missing Middle Observations',
        use_markov_chain=use_markov_chain,
        dtype=dtype,
        **dataset)


class BrownianMotionUnknownScales(bayesian_model.BayesianModel):
  """Construct the Brownian Motion model with unknown scale parameters.

  This models a Brownian Motion process with a Gaussian observation model.

  ```none
  innovation_noise_scale ~ LogNormal(loc=0, scale=2)
  observation_noise_scale ~ LogNormal(loc=0, scale=2)

  locs[0] ~ Normal(loc=0, scale=innovation_noise_scale)
  for t in range(1, num_timesteps):
    locs[t] ~ Normal(loc=locs[t - 1], scale=innovation_noise_scale)

  for t in range(num_timesteps):
    observed_locs[t] ~ Normal(loc=locs[t], scale=observation_noise_scale)
  ```

  This model supports missing observations, indicated by NaNs in the
  `observed_locs` parameter.
  """

  def __init__(self,
               observed_locs,
               use_markov_chain=False,
               dtype=tf.float32,
               name='brownian_motion_unknown_scales',
               pretty_name='Brownian Motion with Unknown Scales'):
    """Construct the Brownian Motion model with unknown scales.

    Args:
      observed_locs: Array of loc parameters with nan value if loc is
        unobserved.
      use_markov_chain: Python `bool` indicating whether to use the
        `MarkovChain` distribution in place of separate random variables for
        each time step. The default of `False` is for backwards compatibility;
        setting this to `True` should significantly improve performance.
        Default value: `False`.
      dtype: Dtype to use for floating point quantities.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """
    with tf.name_scope(name):
      num_timesteps = observed_locs.shape[0]
      self._prior_dist = tfd.JointDistributionCoroutine(
          functools.partial(
              brownian_motion_unknown_scales_prior_fn,
              use_markov_chain=use_markov_chain,
              num_timesteps=num_timesteps,
              dtype=dtype))

      self._log_likelihood_fn = functools.partial(
          brownian_motion_log_likelihood_fn,
          use_markov_chain=use_markov_chain,
          observed_locs=observed_locs,
          dtype=dtype)

      def _ext_identity(params):
        return {'innovation_noise_scale': params[0],
                'observation_noise_scale': params[1],
                'locs': (params[2]
                         if use_markov_chain
                         else tf.stack(params[2:], axis=-1))}

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=_ext_identity,
                  pretty_name='Identity',
                  dtype={'innovation_noise_scale': dtype,
                         'observation_noise_scale': dtype,
                         'locs': dtype})
      }

    event_space_bijector = type(
        self._prior_dist.dtype)(*(
            [tfb.Softplus(),
             tfb.Softplus()
             ] + [tfb.Identity()] * (
                 1 if use_markov_chain else num_timesteps)))
    super(BrownianMotionUnknownScales, self).__init__(
        default_event_space_bijector=event_space_bijector,
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _prior_distribution(self):
    return self._prior_dist

  def log_likelihood(self, value):
    return self._log_likelihood_fn(value)


class BrownianMotionUnknownScalesMissingMiddleObservations(
    BrownianMotionUnknownScales):
  """A simple Brownian Motion with 30 timesteps where 10 are unobservable."""

  GROUND_TRUTH_MODULE = (
      brownian_motion_unknown_scales_missing_middle_observations)

  def __init__(self, use_markov_chain=False, dtype=tf.float32):
    dataset = data.brownian_motion_missing_middle_observations()
    del dataset['innovation_noise_scale']
    del dataset['observation_noise_scale']
    super(BrownianMotionUnknownScalesMissingMiddleObservations, self).__init__(
        name='brownian_motion_unknown_scales_missing_middle_observations',
        pretty_name='Brownian Motion with Unknown Scales',
        use_markov_chain=use_markov_chain,
        dtype=dtype,
        **dataset)
