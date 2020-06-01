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
"""Stan models, used as a source of ground truth."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tools.inference_gym_ground_truth import logistic_regression
from tensorflow_probability.python.experimental.inference_gym.internal import data

__all__ = [
    'german_credit_numeric_logistic_regression',
]


def german_credit_numeric_logistic_regression():
  """German credit (numeric) logistic regression.

  Returns:
    target: StanModel.
  """
  dataset = data.german_credit_numeric()
  del dataset['test_features']
  del dataset['test_labels']
  return logistic_regression.logistic_regression(**dataset)
