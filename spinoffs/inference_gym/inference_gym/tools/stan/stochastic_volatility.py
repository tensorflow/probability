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
"""Stochastic volatility model, implemented in Stan."""

import collections

from inference_gym.tools.stan import stan_model
from inference_gym.tools.stan import util

__all__ = [
    'stochastic_volatility',
]


def stochastic_volatility(centered_returns):
  # pylint: disable=long-lines
  """Stochastic volatility model.

  This formulation is inspired by (a version in the Stan users' manual)[
  https://mc-stan.org/docs/2_21/stan-users-guide/stochastic-volatility-models.html].

  Args:
    centered_returns: float `Tensor` of shape `[num_timesteps]` giving the
      mean-adjusted return (change in asset price, minus the average change)
      observed at each step.

  Returns:
    model: `StanModel`.
  """
  # pylint: enable=long-lines

  # This model is specified in 'noncentered' parameterization, in terms of
  # standardized residuals `log_volatilities_std`. We expect this form of the
  # model to mix more easily than a direct specification would. This makes
  # it valuable for obtaining ground truth, but caution should be used when
  # comparing performance of inference algorithms across parameterizations.
  code = """
  data {
    int<lower=0> num_timesteps;
    vector[num_timesteps] centered_returns;
  }
  parameters {
    real<lower=-1,upper=1> persistence;
    real mean_log_volatility;
    real<lower=0> white_noise_shock_scale;
    vector[num_timesteps] log_volatilities_std;
  }
  transformed parameters {
    vector[num_timesteps] log_volatilities = (
      log_volatilities_std * white_noise_shock_scale);
    log_volatilities[1] /= sqrt(1 - square(persistence));
    log_volatilities += mean_log_volatility;
    for (t in 2:num_timesteps)
      log_volatilities[t] += persistence * (
          log_volatilities[t - 1] - mean_log_volatility);
  }
  model {
    (persistence + 1) * 0.5 ~ beta(20, 1.5);
    white_noise_shock_scale ~ cauchy(0, 2);
    mean_log_volatility ~ cauchy(0, 5);
    log_volatilities_std ~ std_normal();

    centered_returns ~ normal(0, exp(log_volatilities / 2));
  }
  """

  stan_data = {
      'num_timesteps': len(centered_returns),
      'centered_returns': centered_returns
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts the values of all latent variables."""
    res = collections.OrderedDict()
    res['mean_log_volatility'] = util.get_columns(
        samples, r'^mean_log_volatility$')[:, 0]
    res['white_noise_shock_scale'] = util.get_columns(
        samples, r'^white_noise_shock_scale$')[:, 0]
    res['persistence_of_volatility'] = util.get_columns(
        samples, r'^persistence$')[:, 0]
    res['log_volatility'] = util.get_columns(
        samples, r'^log_volatilities\[\d+\]$',)
    return res

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
