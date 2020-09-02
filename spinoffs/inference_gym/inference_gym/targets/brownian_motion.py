# Lint as: python3
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
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from inference_gym.internal import data
from inference_gym.targets import bayesian_model
from inference_gym.targets import model
from inference_gym.targets.ground_truth import brownian_motion_missing_middle_observations

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'BrownianMotion',
    'BrownianMotionMissingMiddleObservations',
]

Root = tfd.JointDistributionCoroutine.Root


def brownian_motion_prior_fn(num_timesteps, innovation_noise):
  """Generative process for the Brownian Motion model."""
  prior_loc = 0.

  new = yield Root(tfd.Normal(loc=prior_loc, scale=innovation_noise))
  for t in range(num_timesteps - 1):
    new = yield tfd.Normal(
        loc=new, scale=innovation_noise, name='x_{}'.format(t))


def brownian_motion_log_likelihood_fn(values, locs, num_timesteps,
                                      observation_noise):
  """Likelihood of observed data under the Brownian Motion model."""

  observed_locs = locs[~np.isnan(locs)]

  # reformat locs into JointDistributionCoroutine expected format
  fixed_locs = np.split(
      observed_locs, indices_or_sections=observed_locs.shape[0])

  def likelihood_model():
    for i in range(num_timesteps):
      if not np.isnan(locs[i]):
        yield Root(tfd.Normal(loc=values[i], scale=observation_noise))

  observation_dist = tfd.JointDistributionCoroutine(likelihood_model)

  return observation_dist.log_prob(fixed_locs)


class BrownianMotion(bayesian_model.BayesianModel):
  """Construct the Brownian Motion model.

  This models a Brownian Motion process. Each timestep consists of a Normal
  distribution with a `loc` parameter. If there are no observations from a given
  timestep, the loc value is np.nan. The constants `innovation_noise` and
  `observation noise` are shared across all timesteps.

  ```none
  # The actual value of the loc parameter at timestep t is:
  loc_{t+1} | loc_{t} ~ Normal(loc_t, innovation_noise)

  # The observed loc at each timestep t (which make up the locs array) is:
  observed_loc_{t} ~ Normal(loc_{t}, observation_noise)
  ```
  """

  def __init__(self,
               locs,
               innovation_noise,
               observation_noise,
               name='brownian_motion',
               pretty_name='Brownian Motion'):
    """Construct the Brownian Motion model.

    Args:
      locs: Array of loc parameters with nan value if loc is unobserved.
      innovation_noise: Python `float`.
      observation_noise: Python `float`.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """
    with tf.name_scope(name):

      num_timesteps = locs.shape[0]

      self._prior_dist = tfd.JointDistributionCoroutine(
          functools.partial(
              brownian_motion_prior_fn,
              num_timesteps=num_timesteps,
              innovation_noise=innovation_noise))

      self._log_likelihood_fn = functools.partial(
          brownian_motion_log_likelihood_fn,
          num_timesteps=num_timesteps,
          observation_noise=observation_noise,
          locs=locs)

      def _ext_identity(params):

        params = tf.stack(params, axis=-1)
        return params

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=_ext_identity,
                  pretty_name='Identity',
              )
      }

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

  def __init__(self):
    dataset = data.brownian_motion_missing_middle_observations()
    super(BrownianMotionMissingMiddleObservations, self).__init__(
        name='brownian_motion_missing_middle_observations',
        pretty_name='Brownian Motion Missing Middle Observations',
        **dataset)
