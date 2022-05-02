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
"""Radon model, implemented in Stan."""

import collections

import numpy as np

from inference_gym.tools.stan import stan_model
from inference_gym.tools.stan import util

__all__ = [
    'radon_contextual_effects',
]


def radon_contextual_effects(
    num_counties,
    train_log_uranium,
    train_floor,
    train_county,
    train_floor_by_county,
    train_log_radon):
  """Heirarchical model of measured radon concentration in homes.

  This is a modified version of the Stan model from:
  https://mc-stan.org/users/documentation/case-studies/radon.html#Correlations-among-levels
  Instead of a `Uniform` prior on the scale parameters, it has a `HalfNormal`
  prior on the scale parameters.

  Args:
    num_counties: `int`, number of counties represented in the data.
    train_log_uranium: Floating-point `Tensor` with shape
      `[num_train_points]`. Soil uranium measurements.
    train_floor: Integer `Tensor` with shape `[num_train_points]`. Floor of
      the house on which the measurement was taken.
    train_county: Integer `Tensor` with values in `range(0, num_counties)` of
      shape `[num_train_points]`. County in which the measurement was taken.
    train_floor_by_county: Floating-point `Tensor` with shape
      `[num_train_points]`. Average floor on which the measurement was taken
      for the county in which each house is located (the `Tensor` will have
      `num_counties` unique values). This represents the contextual effect.
    train_log_radon: Floating-point `Tensor` with shape `[num_train_points]`.
      Radon measurement for each house (the dependent variable in the model).
  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_counties;
    int<lower=0> num_train;
    int<lower=0,upper=num_counties-1> county[num_train];
    vector[num_train] log_uranium;
    vector[num_train] which_floor;
    vector[num_train] floor_by_county;
    vector[num_train] log_radon;
  }
  parameters {
    vector[num_counties] county_effect;
    vector[3] weight;
    real county_effect_mean;
    real<lower=0> county_effect_scale;
    real<lower=0> log_radon_scale;
  }
  transformed parameters {
    vector[num_train] log_radon_mean;

    for (i in 1:num_train)
      log_radon_mean[i] <- county_effect[county[i] + 1]
                           + log_uranium[i] * weight[1]
                           + which_floor[i] * weight[2]
                           + floor_by_county[i] * weight[3];
  }
  model {
    county_effect_mean ~ normal(0, 1);
    county_effect_scale ~ normal(0, 1);
    county_effect ~ normal(county_effect_mean, county_effect_scale);
    weight ~ normal(0, 1);
    log_radon_scale ~ normal(0, 1);
    log_radon ~ normal(log_radon_mean, log_radon_scale);
  }
  """

  stan_data = {
      'num_train': train_log_radon.shape[0],
      'num_counties': num_counties,
      'county': np.array(train_county),
      'log_uranium': np.array(train_log_uranium),
      'floor_by_county': np.array(train_floor_by_county),
      'which_floor': np.array(train_floor),  # `floor` conflicts with a Stan fn
      'log_radon': np.array(train_log_radon)}

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts the values of all latent variables."""
    res = collections.OrderedDict()
    res['county_effect_mean'] = util.get_columns(
        samples, r'^county_effect_mean$')[:, 0]
    res['county_effect_scale'] = util.get_columns(
        samples, r'^county_effect_scale$')[:, 0]
    res['county_effect'] = util.get_columns(samples, r'^county_effect\[\d+\]$')
    res['weight'] = util.get_columns(samples, r'^weight\[\d+\]$')
    res['log_radon_scale'] = (
        util.get_columns(samples, r'^log_radon_scale$')[:, 0])
    return res

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
