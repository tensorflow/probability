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
"""Tests for UCB."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gaussian_process_regression_model
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.bayesopt.acquisition import upper_confidence_bound
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


class _UpperConfidenceBoundTest(object):

  def test_upper_confidence_bound(self):
    shape = [7, 2, 3, 20]
    loc = 3. * np.ones(shape, dtype=self.dtype)
    stddev = 2.
    exploration = 2.
    actual_ucb = upper_confidence_bound.normal_upper_confidence_bound(
        mean=loc, stddev=stddev, exploration=exploration)

    expected_ucb = loc + exploration * stddev
    self.assertAllEqual(actual_ucb.shape, shape)
    self.assertAllClose(actual_ucb, expected_ucb)
    self.assertDTypeEqual(actual_ucb, self.dtype)

  @test_util.numpy_disable_gradient_test
  def test_gp_upper_confidence_bound(self):
    observation_index_points = np.linspace(1., 10., 20, dtype=self.dtype
                                           ).reshape(5, 4)
    observations = np.sum(observation_index_points, axis=-1)
    index_points = np.linspace(4., 5., 12, dtype=self.dtype).reshape(3, 4)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(self.dtype(1.), 1.)

    model = gaussian_process_regression_model.GaussianProcessRegressionModel(
        kernel=kernel,
        observations=observations,
        observation_index_points=observation_index_points)

    loc = model.mean(index_points=index_points)
    stddev = model.stddev(index_points=index_points)

    exploration = 0.1
    expected_ucb = upper_confidence_bound.normal_upper_confidence_bound(
        mean=loc,
        stddev=stddev,
        exploration=exploration)
    gp_ucb = upper_confidence_bound.GaussianProcessUpperConfidenceBound(
        predictive_distribution=model,
        observations=observations,
        exploration=exploration)
    actual_ucb, grads = self.evaluate(
        gradient.value_and_gradient(
            lambda x: gp_ucb(index_points=x),
            tf.convert_to_tensor(index_points)))
    self.assertAllClose(actual_ucb, expected_ucb)
    self.assertAllNotNan(grads)
    self.assertDTypeEqual(actual_ucb, self.dtype)

  def test_normal_ucb_matches_parallel(self):
    shape = [5, 20, 1]
    loc = 2. * np.random.uniform(size=shape).astype(self.dtype)
    scale = 3. + np.random.uniform(size=[20]).astype(self.dtype)
    exploration = 0.
    observations = [2., 3., 4.]
    actual_ucb = upper_confidence_bound.normal_upper_confidence_bound(
        mean=loc[..., 0],
        stddev=scale,
        exploration=exploration)

    model = normal.Normal(loc, scale, validate_args=True)
    expected_ucb = upper_confidence_bound.ParallelUpperConfidenceBound(
        predictive_distribution=model,
        observations=observations,
        exploration=exploration,
        num_samples=int(2e4),
        seed=test_util.test_seed())()
    self.assertAllClose(
        self.evaluate(actual_ucb), self.evaluate(expected_ucb), rtol=1e-3)
    self.assertDTypeEqual(actual_ucb, self.dtype)

  def test_transform_fn_identity(self):
    shape = [5, 20, 1]
    loc = 2. * np.random.uniform(size=shape).astype(self.dtype)
    scale = 3. + np.random.uniform(size=[20]).astype(self.dtype)
    model = normal.Normal(loc, scale, validate_args=True)
    observations = [2., 3., 4.]
    shared_seed = test_util.test_seed(sampler_type='stateless')

    expected_ucb = upper_confidence_bound.ParallelUpperConfidenceBound(
        predictive_distribution=model,
        observations=observations,
        exploration=1.,
        num_samples=int(1e4),
        seed=shared_seed)()

    actual_ucb = upper_confidence_bound.ParallelUpperConfidenceBound(
        predictive_distribution=model,
        observations=observations,
        exploration=1.,
        transform_fn=lambda x: x,
        num_samples=int(1e4),
        seed=shared_seed)()
    self.assertAllClose(
        self.evaluate(actual_ucb),
        self.evaluate(expected_ucb), rtol=8e-3)


@test_util.test_all_tf_execution_regimes
class UpperConfidenceBoundFloat32Test(_UpperConfidenceBoundTest,
                                      test_util.TestCase):

  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class UpperConfidenceBoundFloat64Test(_UpperConfidenceBoundTest,
                                      test_util.TestCase):

  dtype = np.float64


del _UpperConfidenceBoundTest


if __name__ == '__main__':
  test_util.main()
