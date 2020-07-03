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
"""Logistic regression, implemented in Stan."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tools.inference_gym_ground_truth import stan_model
from tools.inference_gym_ground_truth import util

__all__ = [
    'logistic_regression',
]


def _add_bias(features):
  return np.concatenate([features, np.ones([features.shape[0], 1])], axis=-1)


def logistic_regression(
    train_features,
    train_labels,
    test_features=None,
    test_labels=None,
):
  """Bayesian logistic regression with a Gaussian prior.

  Args:
    train_features: Floating-point `Tensor` with shape `[num_train_points,
      num_features]`. Training features.
    train_labels: Integer `Tensor` with shape `[num_train_points]`. Training
      labels.
    test_features: Floating-point `Tensor` with shape `[num_test_points,
      num_features]`. Testing features. Can be `None`, in which case
      test-related sample transformations are not computed.
    test_labels: Integer `Tensor` with shape `[num_test_points]`. Testing
      labels. Can be `None`, in which case test-related sample transformations
      are not computed.

  Returns:
    model: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_train_points;
    int<lower=0> num_test_points;
    int<lower=0> num_features;
    matrix[num_train_points,num_features] train_features;
    int<lower=0,upper=1> train_labels[num_train_points];
    matrix[num_test_points,num_features] test_features;
    int<lower=0,upper=1> test_labels[num_test_points];
  }
  parameters {
    vector[num_features] weights;
  }
  model {
    {
      vector[num_train_points] logits;
      logits = train_features * weights;

      weights ~ normal(0, 1);
      train_labels ~ bernoulli_logit(logits);
    }
  }
  generated quantities {
    real test_nll;
    real per_example_test_nll[num_test_points];
    {
      vector[num_test_points] logits;
      logits = test_features * weights;

      test_nll = -bernoulli_logit_lpmf(test_labels | logits);
      for (i in 1:num_test_points) {
        per_example_test_nll[i] = -bernoulli_logit_lpmf(
            test_labels[i] | logits[i]);
      }
    }
  }
  """
  have_test = test_features is not None
  train_features = _add_bias(train_features)
  if have_test:
    test_features = _add_bias(test_features)
  else:
    # cmdstanpy can't handle zero-sized arrays at the moment:
    # https://github.com/stan-dev/cmdstanpy/issues/203
    test_features = train_features[:1]
    test_labels = train_labels[:1]
  stan_data = {
      'num_train_points': train_features.shape[0],
      'num_test_points': test_features.shape[0],
      'num_features': train_features.shape[1],
      'train_features': train_features,
      'train_labels': train_labels,
      'test_features': test_features,
      'test_labels': test_labels,
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    return util.get_columns(samples, r'^weights\.\d+$')

  def _ext_test_nll(samples):
    return util.get_columns(samples, r'^test_nll$')[:, 0]

  def _ext_per_example_test_nll(samples):
    return util.get_columns(samples, r'^per_example_test_nll\.\d+$')

  extract_fns = {'identity': _ext_identity}
  if have_test:
    extract_fns['test_nll'] = _ext_test_nll
    extract_fns['per_example_test_nll'] = _ext_per_example_test_nll

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
