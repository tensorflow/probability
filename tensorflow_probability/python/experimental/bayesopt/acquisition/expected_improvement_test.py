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
"""Tests for expected_improvement."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gaussian_process_regression_model
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import student_t_process_regression_model
from tensorflow_probability.python.experimental.bayesopt.acquisition import expected_improvement
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


class _ExpectedImprovementTest(object):

  def test_normal_expected_improvement(self):
    shape = [5, 1, 20]
    loc = 2. * np.ones(shape, dtype=self.dtype)
    stddev = 3.
    observations = np.array([2., 3., 4.]).astype(self.dtype)
    best_observed = tf.reduce_max(observations)
    exploration = 0.8
    actual_ei = expected_improvement.normal_expected_improvement(
        best_observed=best_observed,
        mean=loc,
        stddev=stddev,
        exploration=exploration)

    norm = normal.Normal(self.dtype(0.), 1.)
    imp = loc - best_observed - exploration
    z = imp / stddev
    expected_ei = imp * norm.cdf(z) + stddev * norm.prob(z)
    self.assertAllEqual(actual_ei.shape, shape)
    self.assertAllClose(actual_ei, expected_ei)
    self.assertDTypeEqual(actual_ei, self.dtype)

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

    exploration = self.dtype(0.1)
    expected_ei = expected_improvement.normal_expected_improvement(
        best_observed=best_observed,
        mean=loc,
        stddev=stddev,
        exploration=exploration)
    gp_ei = expected_improvement.GaussianProcessExpectedImprovement(
        predictive_distribution=model,
        observations=observations,
        exploration=exploration)
    actual_ei, grads = self.evaluate(
        gradient.value_and_gradient(
            lambda x: gp_ei(index_points=x),
            tf.convert_to_tensor(index_points)))

    self.assertAllClose(actual_ei, expected_ei)
    self.assertAllNotNan(grads)
    self.assertDTypeEqual(actual_ei, self.dtype)

  def test_student_t_expected_improvement(self):
    shape = [5, 1, 20]
    loc = 2. * np.ones(shape, dtype=self.dtype)
    stddev = 3.
    df = self.dtype(5.)
    observations = np.array([2., 3., 4.]).astype(self.dtype)
    best_observed = tf.reduce_max(observations)
    exploration = 0.8
    actual_ei = expected_improvement.student_t_expected_improvement(
        best_observed=best_observed,
        df=df,
        mean=loc,
        stddev=stddev,
        exploration=exploration)

    st = student_t.StudentT(df, 0., 1.)
    imp = loc - best_observed - exploration
    z = imp / stddev
    expected_ei = (
        imp * st.cdf(z) +
        df / (df - 1.) * (
            1. + tf.math.square(z) / df) * stddev * st.prob(z))
    self.assertAllClose(actual_ei, expected_ei)
    self.assertDTypeEqual(actual_ei, self.dtype)

  @test_util.numpy_disable_gradient_test
  def test_stp_expected_improvement(self):
    observation_index_points = np.linspace(
        1., 10., 20, dtype=self.dtype).reshape(5, 4)
    observations = np.sum(observation_index_points, axis=-1)
    index_points = np.linspace(4., 5., 12, dtype=self.dtype).reshape(3, 4)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(self.dtype(1.), 1.)
    df = self.dtype(25.)

    model = student_t_process_regression_model.StudentTProcessRegressionModel(
        df=df,
        kernel=kernel,
        observations=observations,
        observation_index_points=observation_index_points)

    loc = model.mean(index_points=index_points)
    stddev = model.stddev(index_points=index_points)
    best_observed = tf.reduce_max(observations)

    exploration = self.dtype(0.1)
    # For a STPRM, the effective degrees of freedom is df + num_observations.
    expected_ei = expected_improvement.student_t_expected_improvement(
        best_observed=best_observed,
        df=df + self.dtype(observations.shape[-1]),
        mean=loc,
        stddev=stddev,
        exploration=exploration)
    stp_ei = expected_improvement.StudentTProcessExpectedImprovement(
        predictive_distribution=model,
        observations=observations,
        exploration=exploration)
    actual_ei, grads = self.evaluate(
        gradient.value_and_gradient(
            lambda x: stp_ei(index_points=x),
            tf.convert_to_tensor(index_points)))

    self.assertAllNotNan(grads)
    self.assertAllClose(actual_ei, expected_ei)
    self.assertDTypeEqual(actual_ei, self.dtype)

  def test_normal_expected_improvement_matches_parallel(self):
    shape = [5, 20]
    loc = 2. * np.random.uniform(size=shape).astype(self.dtype)
    scale = 3. + np.random.uniform(size=[20]).astype(self.dtype)
    exploration = 0.
    observations = np.array([2., 3., 4.]).astype(self.dtype)
    best_observed = tf.reduce_max(observations)
    actual_ei = expected_improvement.normal_expected_improvement(
        best_observed=best_observed,
        mean=loc,
        stddev=scale,
        exploration=exploration)

    model = normal.Normal(
        loc[..., tf.newaxis], scale[..., tf.newaxis], validate_args=True)
    expected_ei = expected_improvement.ParallelExpectedImprovement(
        predictive_distribution=model,
        observations=observations,
        exploration=exploration,
        num_samples=int(2e5),
        seed=test_util.test_seed())()
    self.assertAllClose(
        self.evaluate(actual_ei), self.evaluate(expected_ei), rtol=1e-1)
    self.assertDTypeEqual(actual_ei, self.dtype)

  def test_student_t_expected_improvement_matches_parallel(self):
    shape = [5, 20]
    df = self.dtype(50.)
    loc = 2. * np.random.uniform(size=shape).astype(self.dtype)
    scale = 3. + np.random.uniform(size=[20]).astype(self.dtype)
    exploration = 0.
    observations = np.array([2., 3., 4.]).astype(self.dtype)
    best_observed = tf.reduce_max(observations)
    actual_ei = expected_improvement.student_t_expected_improvement(
        df=df,
        best_observed=best_observed,
        mean=loc,
        stddev=scale,
        exploration=exploration)

    model = student_t.StudentT(
        df, loc[..., tf.newaxis], scale[..., tf.newaxis], validate_args=True)
    expected_ei = expected_improvement.ParallelExpectedImprovement(
        predictive_distribution=model,
        observations=observations,
        exploration=exploration,
        num_samples=int(2e5),
        seed=test_util.test_seed())()

    self.assertAllClose(
        self.evaluate(actual_ei), self.evaluate(expected_ei), rtol=1e-1)
    self.assertDTypeEqual(actual_ei, self.dtype)

  def test_transform_fn_identity(self):
    shape = [5, 20]
    df = self.dtype(50.)
    loc = 2. * np.random.uniform(size=shape).astype(self.dtype)
    scale = 3. + np.random.uniform(size=[20]).astype(self.dtype)
    observations = np.array([2., 3., 4.]).astype(self.dtype)
    model = student_t.StudentT(
        df, loc[..., tf.newaxis], scale[..., tf.newaxis], validate_args=True)
    shared_seed = test_util.test_seed(sampler_type='stateless')
    expected_ei = expected_improvement.ParallelExpectedImprovement(
        predictive_distribution=model,
        observations=observations,
        exploration=1.,
        num_samples=30,
        seed=shared_seed)()
    actual_ei = expected_improvement.ParallelExpectedImprovement(
        predictive_distribution=model,
        observations=observations,
        exploration=1.,
        transform_fn=lambda x: x,
        num_samples=30,
        seed=shared_seed)()
    self.assertAllClose(self.evaluate(actual_ei), self.evaluate(expected_ei))


@test_util.test_all_tf_execution_regimes
class ExpectedImprovementFloat32Test(
    _ExpectedImprovementTest, test_util.TestCase):

  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class ExpectedImprovementFloat64Test(
    _ExpectedImprovementTest, test_util.TestCase):

  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
