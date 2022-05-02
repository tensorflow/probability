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
"""1PL item-response theory model, implemented in Stan."""

import collections

from inference_gym.tools.stan import stan_model
from inference_gym.tools.stan import util

__all__ = [
    'item_response_theory',
]


def item_response_theory(
    train_student_ids,
    train_question_ids,
    train_correct,
    test_student_ids=None,
    test_question_ids=None,
    test_correct=None,
):
  """One-parameter logistic item-response theory (IRT) model.

  Args:
    train_student_ids: integer `tensor` with shape `[num_train_points]`.
      training student ids, ranging from 0 to `num_students`.
    train_question_ids: integer `tensor` with shape `[num_train_points]`.
      training question ids, ranging from 0 to `num_questions`.
    train_correct: integer `tensor` with shape `[num_train_points]`. whether the
      student in the training set answered the question correctly, either 0 or
      1.
    test_student_ids: Integer `Tensor` with shape `[num_test_points]`. Testing
      student ids, ranging from 0 to `num_students`. Can be `None`, in which
      case test-related sample transformations are not computed.
    test_question_ids: Integer `Tensor` with shape `[num_test_points]`. Testing
      question ids, ranging from 0 to `num_questions`. Can be `None`, in which
      case test-related sample transformations are not computed.
    test_correct: Integer `Tensor` with shape `[num_test_points]`. Whether the
      student in the testing set answered the question correctly, either 0 or 1.
      Can be `None`, in which case test-related sample transformations are not
      computed.

  Returns:
    target: `StanModel`.
  """

  code = """
  data {
    int<lower=0> num_students;
    int<lower=0> num_questions;
    int<lower=0> num_train_pairs;
    int<lower=0> num_test_pairs;
    int<lower=1,upper=num_students> train_student_ids[num_train_pairs];
    int<lower=1,upper=num_questions> train_question_ids[num_train_pairs];
    int<lower=0,upper=1> train_responses[num_train_pairs];
    int<lower=1,upper=num_students> test_student_ids[num_test_pairs];
    int<lower=1,upper=num_questions> test_question_ids[num_test_pairs];
    int<lower=0,upper=1> test_responses[num_test_pairs];
  }
  parameters {
    real mean_student_ability;
    vector[num_students] student_ability;
    vector[num_questions] question_difficulty;
  }
  model {
    {
      mean_student_ability ~ normal(0.75, 1);
      student_ability ~ normal(0, 1);
      question_difficulty ~ normal(0, 1);

      for (i in 1:num_train_pairs) {
        real pair_logit;
        pair_logit = (
            mean_student_ability + student_ability[train_student_ids[i]] -
            question_difficulty[train_question_ids[i]]
        );
        train_responses[i] ~ bernoulli_logit(pair_logit);
      }
    }
  }
  generated quantities {
    real test_nll = 0.;
    real per_example_test_nll[num_test_pairs];
    {
      for (i in 1:num_test_pairs) {
        real pair_logit;
        pair_logit = (
            mean_student_ability + student_ability[test_student_ids[i]] -
            question_difficulty[test_question_ids[i]]
        );
        per_example_test_nll[i] = -bernoulli_logit_lpmf(test_responses[i] | pair_logit);
      }
      test_nll = sum(per_example_test_nll);
    }
  }
  """

  have_test = test_student_ids is not None
  # cmdstanpy can't handle zero-sized arrays at the moment:
  # https://github.com/stan-dev/cmdstanpy/issues/203
  if not have_test:
    test_student_ids = train_student_ids[:1]
    test_question_ids = train_question_ids[:1]
    test_correct = train_correct[:1]
  stan_data = {
      'num_train_pairs':
          train_student_ids.shape[0],
      'num_test_pairs':
          test_student_ids.shape[0],
      'num_students':
          max(int(train_student_ids.max()), int(test_student_ids.max())) + 1,
      'num_questions':
          max(int(train_question_ids.max()), int(test_question_ids.max())) + 1,
      'train_student_ids':
          train_student_ids + 1,  # N.B. Stan arrays are 1-indexed.
      'train_question_ids':
          train_question_ids + 1,
      'train_responses':
          train_correct,
      'test_student_ids':
          test_student_ids + 1,
      'test_question_ids':
          test_question_ids + 1,
      'test_responses':
          test_correct,
  }

  model = util.cached_stan_model(code)

  def _ext_identity(samples):
    """Extracts all the parameters."""
    res = collections.OrderedDict()
    res['mean_student_ability'] = util.get_columns(
        samples,
        r'^mean_student_ability$',
    )[:, 0]
    res['student_ability'] = util.get_columns(
        samples,
        r'^student_ability\[\d+\]$',
    )
    res['question_difficulty'] = util.get_columns(
        samples,
        r'^question_difficulty\[\d+\]$',
    )
    return res

  def _ext_test_nll(samples):
    return util.get_columns(samples, r'^test_nll$')[:, 0]

  def _ext_per_example_test_nll(samples):
    return util.get_columns(samples, r'^per_example_test_nll\[\d+\]$')

  extract_fns = {'identity': _ext_identity}
  if have_test:
    extract_fns['test_nll'] = _ext_test_nll
    extract_fns['per_example_test_nll'] = _ext_per_example_test_nll

  return stan_model.StanModel(
      extract_fns=extract_fns,
      sample_fn=util.make_sample_fn(model, data=stan_data),
  )
