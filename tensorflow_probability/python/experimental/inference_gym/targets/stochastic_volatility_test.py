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
# See the License for the modelific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for inference_gym.targets.stochastic_volatility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.inference_gym.internal import test_util
from tensorflow_probability.python.experimental.inference_gym.targets import stochastic_volatility


class StochasticVolatilitySP500Test(test_util.InferenceGymTestCase):

  def testBasic(self):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = stochastic_volatility.StochasticVolatility(
        centered_returns=tf.convert_to_tensor([5., -2.1, 8., 4., 1.1]))
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'persistence_of_volatility': [],
                'mean_log_volatility': [],
                'white_noise_shock_scale': [],
                'log_volatility': [5]
            }))

  # Verify that data loading works using the small model only, since the full
  # dataset leads to an unwieldy prior containing 2518 RVs.
  def testSP500Small(self):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = (
        stochastic_volatility.StochasticVolatilitySP500Small())
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'persistence_of_volatility': [],
                'mean_log_volatility': [],
                'white_noise_shock_scale': [],
                'log_volatility': [100]
            }))


if __name__ == '__main__':
  tf.test.main()
