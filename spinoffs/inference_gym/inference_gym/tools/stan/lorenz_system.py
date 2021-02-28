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
# ============================================================================
"""Lorenz System model, implemented in Stan."""

import numpy as np

from inference_gym.tools.stan import stan_model
from inference_gym.tools.stan import util

__all__ = [
    'partially_observed_lorenz_system',
    'partially_observed_lorenz_system_unknown_scales'
]


# Model code block used by both the fixed- and unknown-scale models.
LORENZ_SYSTEM_OBSERVATIONS_MODEL = """
    real x;
    real y;
    real z;
    row_vector[3] delta;
    latents[1] ~ normal(0, 1);
    for (t in 2:num_timesteps){
      x = latents[t - 1, 1];
      y = latents[t - 1, 2];
      z = latents[t - 1, 3];
      delta[1] = 10. * (y - x);
      delta[2] = x * (28. - z) - y;
      delta[3] = x * y - 8. / 3. * z;
      latents[t] ~ normal(latents[t - 1] + step_size * delta,
                          sqrt(step_size) * innovation_scale);
    }
    observations ~ normal(
      latents[observation_time_indices, observation_state_index],
      observation_scale);
"""


def partially_observed_lorenz_system(observed_values, innovation_scale,
                                     observation_scale, observation_mask,
                                     step_size, observation_index):
  """Lorenz System model.

  Args:
    observed_values: Array of observed values.
    innovation_scale: Python `float`.
    observation_scale: Python `float`.
    observation_mask: `bool` array used to occlude observations.
    step_size: Python `float`.
    observation_index: `int` index used to pick which latent time series
      is observed.

  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_timesteps;
    int<lower=0> num_observations;
    int<lower=1> observation_state_index;
    int<lower = 1, upper = num_timesteps> observation_time_indices[
      num_observations];
    vector[num_observations] observations;
    real<lower=0> innovation_scale;
    real<lower=0> observation_scale;
    real<lower=0> step_size;
  }
  parameters {
    matrix[num_timesteps, 3] latents;
  }
  model {""" + LORENZ_SYSTEM_OBSERVATIONS_MODEL + '}'

  stan_data = {
      'num_timesteps':
          len(observed_values),
      'num_observations':
          len(observed_values[observation_mask]),
      'observation_time_indices':
          np.arange(1, len(observed_values) + 1)[observation_mask],
      'observations':
          observed_values[observation_mask],
      'observation_state_index':
          observation_index + 1,
      'innovation_scale':
          innovation_scale,
      'observation_scale':
          observation_scale,
      'step_size':
          step_size
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts the values of all latent variables."""
    latents = util.get_columns(samples, r'^latents\.\d+\.\d+$')
    # Last two dimensions are swapped in Stan output.
    return latents.reshape((-1, 3, 30)).swapaxes(1, 2)

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )


def partially_observed_lorenz_system_unknown_scales(
    observed_values, observation_mask, step_size, observation_index):
  """Lorenz System model with unknown scale parameters.

  Args:
    observed_values: Array of observed values.
    observation_mask: `bool` array used to occlude observations.
    step_size: Python `float`.
    observation_index: `int` index used to pick which latent time series
      is observed.

  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_timesteps;
    int<lower=0> num_observations;
    int<lower=1> observation_state_index;
    int<lower = 1, upper = num_timesteps> observation_time_indices[
      num_observations];
    vector[num_observations] observations;
    real<lower=0> step_size;
  }
  parameters {
    real<lower=0> innovation_scale;
    real<lower=0> observation_scale;
    matrix[num_timesteps, 3] latents;
  }
  model {
    innovation_scale ~ lognormal(-1, 1);
    observation_scale ~ lognormal(-1, 1);
    """ + LORENZ_SYSTEM_OBSERVATIONS_MODEL + '}'

  stan_data = {
      'num_timesteps':
          len(observed_values),
      'num_observations':
          len(observed_values[observation_mask]),
      'observation_time_indices':
          np.arange(1, len(observed_values) + 1)[observation_mask],
      'observations':
          observed_values[observation_mask],
      'observation_state_index':
          observation_index + 1,
      'step_size':
          step_size
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts the values of all latent variables."""
    latents = util.get_columns(samples, r'^latents\.\d+\.\d+$')
    return {
        'innovation_scale': util.get_columns(samples,
                                             r'^innovation_scale$')[:, 0],
        'observation_scale': util.get_columns(samples,
                                              r'^observation_scale$')[:, 0],
        # Last two dimensions are swapped in Stan output.
        'latents': latents.reshape((-1, 3, 30)).swapaxes(1, 2)}

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
