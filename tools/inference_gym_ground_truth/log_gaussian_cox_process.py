# Lint as: python2, python3
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
"""Log-gaussian Cox Process, implemented in Stan."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tools.inference_gym_ground_truth import stan_model
from tools.inference_gym_ground_truth import util

__all__ = [
    'log_gaussian_cox_process',
]


def log_gaussian_cox_process(
    train_locations,
    train_extents,
    train_counts,
):
  """Log-Gaussian Cox Process model.

  Args:
    train_locations: Float `Tensor` with shape `[num_train_points, D]`. Training
      set locations where counts were measured.
    train_extents: Float `Tensor` with shape `[num_train_points]`. Training set
      location extents, must be positive.
    train_counts: Integer `Tensor` with shape `[num_train_points]`. Training set
      counts, must be positive.

  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_points;
    int<lower=0> num_features;
    vector[num_features] locations[num_points];
    real<lower=0> extents[num_points];
    int<lower=0> counts[num_points];
  }
  transformed data {
    vector[num_points] loc;
    real mean_log_intensity;
    {
      mean_log_intensity = 0;
      for (i in 1:num_points) {
        mean_log_intensity += (
          log(counts[i]) - log(extents[i])) / num_points;
      }
      for (i in 1:num_points) loc[i] = mean_log_intensity;  // otherwise nan!
    }
  }
  parameters {
    real<lower=0> amplitude;
    real<lower=0> length_scale;
    vector[num_points] log_intensity;
  }
  model {
    {
      matrix[num_points, num_points] L_K;
      matrix[num_points, num_points] K = gp_matern32_cov(
          locations, amplitude + .001, length_scale + .001);
      for (i in 1:num_points) K[i,i] += 1e-6;  // GP jitter
      L_K = cholesky_decompose(K);

      amplitude ~ lognormal(-1., .5);
      length_scale ~ lognormal(-1., 1.);
      log_intensity ~ multi_normal_cholesky(loc, L_K);
      for (i in 1:num_points) {
        counts[i] ~ poisson_log(
          log(extents[i]) + log_intensity[i]);
      }
    }
  }
  """

  num_points = train_locations.shape[0]
  num_features = train_locations.shape[1]
  stan_data = {
      'num_points': num_points,
      'num_features': num_features,
      'locations': train_locations,
      'extents': train_extents,
      'counts': train_counts,
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extract all the parameters."""
    res = collections.OrderedDict()
    res['amplitude'] = util.get_columns(
        samples,
        r'^amplitude$',
    )[:, 0]
    res['length_scale'] = util.get_columns(
        samples,
        r'^length_scale$',
    )[:, 0]
    res['log_intensity'] = util.get_columns(
        samples,
        r'^log_intensity\.\d+$',
    )
    return res

  extract_fns = {'identity': _ext_identity}

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
