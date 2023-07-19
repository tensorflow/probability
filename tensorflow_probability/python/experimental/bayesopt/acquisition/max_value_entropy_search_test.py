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
"""Tests for Max Value Entropy Search."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gaussian_process_regression_model as gprm
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.bayesopt.acquisition import max_value_entropy_search
from tensorflow_probability.python.experimental.bayesopt.acquisition.max_value_entropy_search import fit_max_value_distribution
from tensorflow_probability.python.experimental.bayesopt.acquisition.max_value_entropy_search import inverse_mills_ratio
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


class _MaxValueEntropySearchTest(object):

  def test_inverse_mills_ratio(self):

    x = tf.random.stateless_normal(
        [int(1e2)], dtype=self.dtype,
        seed=test_util.test_seed(sampler_type='stateless'))
    actual = inverse_mills_ratio(x)
    normal_dist = normal.Normal(self.dtype(0.), 1.)
    expected = tf.math.exp(normal_dist.log_prob(x) - normal_dist.log_cdf(x))
    actual_, expected_ = self.evaluate([actual, expected])
    self.assertAllClose(actual_, expected_)
    self.assertDTypeEqual(actual_, self.dtype)

  def test_max_value_distribution(self):
    # Test that we can create the max-value distribution
    # and it has the right shape.

    index_points = np.linspace(-4., 4., 5, dtype=self.dtype)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    index_points = np.stack([index_points]*6)

    observation_index_points = np.linspace(4., 10., 4, dtype=self.dtype)
    observation_index_points = np.stack(
        np.meshgrid(observation_index_points, observation_index_points),
        axis=-1)
    observation_index_points = np.reshape(observation_index_points, [-1, 2])

    observations = np.linspace(2., 10., 16, dtype=self.dtype)

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], self.dtype).reshape([2, 1, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], self.dtype).reshape([1, 4, 1, 1])

    predictive_distribution = gprm.GaussianProcessRegressionModel(
        kernel=exponentiated_quadratic.ExponentiatedQuadratic(
            amplitude=amplitude, length_scale=length_scale),
        observations=observations,
        observation_index_points=observation_index_points,
        index_points=index_points)

    max_value_distribution = fit_max_value_distribution(
        predictive_distribution=predictive_distribution,
        observations=observations,
        num_grid_points=100,
        seed=test_util.test_seed(sampler_type='stateless'))

    actual_mes = max_value_entropy_search.GaussianProcessMaxValueEntropySearch(
        predictive_distribution=predictive_distribution,
        observations=observations,
        num_max_value_samples=1,
        seed=test_util.test_seed())()

    # Ensure that we properly take in to account the batch shape from various
    # sources.
    self.assertEqual(max_value_distribution.batch_shape, [2, 4, 1, 6])
    # Ensure that the max-value distribution is scalar.
    self.assertEqual(max_value_distribution.event_shape, [])

    # Test MES shape.
    shape = [2, 4, 1, 6, 25]
    self.assertAllEqual(actual_mes.shape, shape)
    self.assertDTypeEqual(actual_mes, self.dtype)

  @test_util.numpy_disable_gradient_test
  def test_gradient(self):
    index_points = np.linspace(-4., 4., 10, dtype=self.dtype).reshape([-1, 2])
    observations = np.linspace(2., 10., 8, dtype=self.dtype)
    observation_index_points = np.linspace(4., 10., 16, dtype=self.dtype
                                           ).reshape([-1, 2])
    predictive_distribution = gprm.GaussianProcessRegressionModel(
        kernel=exponentiated_quadratic.ExponentiatedQuadratic(
            amplitude=self.dtype(1.)),
        observations=observations,
        observation_index_points=observation_index_points,
        observation_noise_variance=self.dtype(0.),
        index_points=index_points)
    mes = max_value_entropy_search.GaussianProcessMaxValueEntropySearch(
        predictive_distribution=predictive_distribution,
        observations=observations,
        num_max_value_samples=1,
        seed=test_util.test_seed())
    _, grad = gradient.value_and_gradient(
        lambda x: mes(index_points=x), tf.convert_to_tensor(index_points))
    self.assertAllNotNan(grad)


@test_util.test_all_tf_execution_regimes
class MaxValueEntropySearchFloat32Test(
    _MaxValueEntropySearchTest, test_util.TestCase):

  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class MaxValueEntropySearchFloat64Test(
    _MaxValueEntropySearchTest, test_util.TestCase):

  dtype = np.float64


del _MaxValueEntropySearchTest


if __name__ == '__main__':
  test_util.main()
