# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for PoI."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gaussian_process_regression_model
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.bayesopt.acquisition import probability_of_improvement
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


class _ProbabilityOfImprovementTest(object):

  def test_probability_of_improvement(self):
    shape = [6, 2, 15]
    loc = 2. * np.ones(shape, dtype=self.dtype)
    stddev = 3.
    observations = np.array([2., 3., 6.], self.dtype)
    best_observed = tf.reduce_max(observations)
    model = normal.Normal(loc, stddev, validate_args=True)
    actual_poi = probability_of_improvement.normal_probability_of_improvement(
        best_observed, mean=loc, stddev=stddev)
    expected_poi = model.survival_function(best_observed)
    self.assertAllEqual(actual_poi.shape, shape)
    self.assertAllClose(actual_poi, expected_poi)
    self.assertDTypeEqual(actual_poi, self.dtype)

  @test_util.numpy_disable_gradient_test
  def test_gp_expected_improvement(self):
    observation_index_points = np.linspace(
        1., 10., 20, dtype=self.dtype).reshape(5, 4)
    observations = np.sum(observation_index_points, axis=-1)
    index_points = np.linspace(4., 5., 12, dtype=self.dtype).reshape(3, 4)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(self.dtype(1.), 1.)

    model = gaussian_process_regression_model.GaussianProcessRegressionModel(
        kernel=kernel,
        observations=observations,
        observation_index_points=observation_index_points)

    loc = model.mean(index_points=index_points)
    stddev = model.stddev(index_points=index_points)
    best_observed = tf.reduce_max(observations)

    expected_poi = probability_of_improvement.normal_probability_of_improvement(
        best_observed=best_observed,
        mean=loc,
        stddev=stddev)
    gp_poi = probability_of_improvement.GaussianProcessProbabilityOfImprovement(
        predictive_distribution=model,
        observations=observations)
    actual_poi, grads = self.evaluate(
        gradient.value_and_gradient(
            lambda x: gp_poi(index_points=x),
            tf.convert_to_tensor(index_points)))

    self.assertAllClose(actual_poi, expected_poi)
    self.assertAllNotNan(grads)
    self.assertDTypeEqual(actual_poi, self.dtype)


@test_util.test_all_tf_execution_regimes
class ProbabilityOfImprovementFloat32Test(_ProbabilityOfImprovementTest,
                                          test_util.TestCase):

  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class ProbabilityOfImprovementFloat64Test(_ProbabilityOfImprovementTest,
                                          test_util.TestCase):

  dtype = np.float64


del _ProbabilityOfImprovementTest


if __name__ == '__main__':
  test_util.main()
