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
"""Brownian Motion model, implemented in Stan."""

import numpy as np

from inference_gym.tools.stan import stan_model
from inference_gym.tools.stan import util

__all__ = [
    'brownian_motion',
    'brownian_motion_unknown_scales'
]


def brownian_motion(observed_locs, innovation_noise_scale,
                    observation_noise_scale):
  """Brownian Motion model.

  Args:
    observed_locs: Array of loc parameters with np.nan value if loc is
      unobserved.
    innovation_noise_scale: Python `float`.
    observation_noise_scale: Python `float`.

  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_timesteps;
    int<lower=0> num_observations;
    int<lower = 1, upper = num_timesteps> observation_indices[num_observations];
    vector[num_observations] observations;
    real<lower=0> innovation_noise_scale;
    real<lower=0> observation_noise_scale;
  }
  parameters {
    vector[num_timesteps] loc;
  }
  model {
    loc[1] ~ normal(0, innovation_noise_scale);
    for (t in 2:num_timesteps){
      loc[t] ~ normal(loc[t-1], innovation_noise_scale);
      }
    observations ~ normal(loc[observation_indices], observation_noise_scale);
  }
  """

  stan_data = {
      'num_timesteps': len(observed_locs),
      'num_observations': len(observed_locs[~np.isnan(observed_locs)]),
      'observation_indices':
          (np.arange(1,
                     len(observed_locs) + 1)[~np.isnan(observed_locs)]),
      'observations': observed_locs[~np.isnan(observed_locs)],
      'innovation_noise_scale': innovation_noise_scale,
      'observation_noise_scale': observation_noise_scale
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts the values of all latent variables."""
    locs = util.get_columns(samples, r'^loc\[\d+\]$')
    return locs

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )


def brownian_motion_unknown_scales(observed_locs):
  """Brownian Motion model with scale parameters treated as random variables.

  Args:
    observed_locs: Array of loc parameters with np.nan value if loc is
      unobserved.
  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_timesteps;
    int<lower=0> num_observations;
    int<lower = 1, upper = num_timesteps> observation_indices[num_observations];
    vector[num_observations] observations;
  }
  parameters {
    real<lower=0> innovation_noise_scale;
    real<lower=0> observation_noise_scale;
    vector[num_timesteps] loc;
  }
  model {
    innovation_noise_scale ~ lognormal(0, 2);
    observation_noise_scale ~ lognormal(0, 2);
    loc[1] ~ normal(0, innovation_noise_scale);
    for (t in 2:num_timesteps){
      loc[t] ~ normal(loc[t-1], innovation_noise_scale);
      }
    observations ~ normal(loc[observation_indices], observation_noise_scale);
  }
  """

  stan_data = {
      'num_timesteps': len(observed_locs),
      'num_observations': len(observed_locs[~np.isnan(observed_locs)]),
      'observation_indices':
          (np.arange(1,
                     len(observed_locs) + 1)[~np.isnan(observed_locs)]),
      'observations': observed_locs[~np.isnan(observed_locs)],
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts the values of all latent variables."""
    return {'innovation_noise_scale': util.get_columns(
                samples, r'^innovation_noise_scale$')[:, 0],
            'observation_noise_scale': util.get_columns(
                samples, r'^observation_noise_scale$')[:, 0],
            'locs': util.get_columns(samples, r'^loc\[\d+\]$')}

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
