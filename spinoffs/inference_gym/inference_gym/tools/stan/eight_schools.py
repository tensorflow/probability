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
"""Eight Schools model, implemented in Stan."""

import collections

import numpy as np

from inference_gym.tools.stan import stan_model
from inference_gym.tools.stan import util

__all__ = [
    'eight_schools',
]


def eight_schools():
  """Eight Schools model.

  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_schools;
    real treatment_effects[num_schools];
    real<lower=0> treatment_stddevs[num_schools];
  }
  parameters {
    real avg_effect;
    real log_stddev;
    vector[num_schools] std_school_effects;
  }
  transformed parameters {
    vector[num_schools] school_effects;
    school_effects <- std_school_effects * exp(log_stddev) + avg_effect;
  }
  model {
    avg_effect ~ normal(0, 10);
    log_stddev ~ normal(5, 1);
    std_school_effects ~ normal(0, 1);
    treatment_effects ~ normal(school_effects, treatment_stddevs);
  }
  """

  stan_data = {
      'num_schools': 8,
      'treatment_effects': np.array(
          [28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32),
      'treatment_stddevs': np.array(
          [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)}

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts the values of all latent variables."""
    res = collections.OrderedDict()
    res['avg_effect'] = util.get_columns(
        samples, r'^avg_effect$')[:, 0]
    res['log_stddev'] = util.get_columns(
        samples, r'^log_stddev$')[:, 0]
    res['school_effects'] = util.get_columns(
        samples, r'^school_effects\[\d+\]$')
    return res

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
