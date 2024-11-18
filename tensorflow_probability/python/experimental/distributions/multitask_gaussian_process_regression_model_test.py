# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for MultiTaskGaussianProcessRegressionModel."""

from unittest import mock
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gaussian_process_regression_model as gprm
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process_regression_model as mtgprm_lib
from tensorflow_probability.python.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels.internal import test_util as psd_kernel_test_util


@test_util.test_all_tf_execution_regimes
class MultiTaskGaussianProcessRegressionModelTest(
    test_util.TestCase):
  # TODO(b/202181168): Add shape inference tests with None shapes.

  def testMeanShapeBroadcasts(self):
    observation_index_points = tf.Variable(
        np.random.random((10, 5)), dtype=np.float32)
    observations = tf.Variable(np.random.random((10, 3)), dtype=np.float32)
    index_points = tf.Variable(np.random.random((4, 5)), dtype=np.float32)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)
    mean = tf.Variable(np.random.random((3,)), dtype=np.float32)
    gp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        multi_task_kernel,
        observation_index_points=observation_index_points,
        observations=observations,
        index_points=index_points,
        mean_fn=lambda _: mean,
        observation_noise_variance=np.float32(1e-2))
    self.assertAllEqual(self.evaluate(gp.event_shape_tensor()), [4, 3])

  @parameterized.parameters(1, 3, 5)
  def testShapes(self, num_tasks):
    # 3x3 grid of index points in R^2 and flatten to 9x2
    index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])

    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 9, 2]

    # ==> shape = [9, 2]
    observations = np.linspace(-20., 20., num_tasks * 9).reshape(9, num_tasks)

    test_index_points = np.random.uniform(-6., 6., [5, 2])
    # ==> shape = [3, 1, 5, 2]

    # Kernel with batch_shape [2, 4, 3, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float64).reshape([1, 4, 1, 1])
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-5], np.float64).reshape([1, 1, 3, 1])
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)
    gp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        multi_task_kernel,
        observation_index_points=batched_index_points,
        observations=observations,
        index_points=test_index_points,
        observation_noise_variance=observation_noise_variance,
        predictive_noise_variance=0.,
        validate_args=True)

    batch_shape = [2, 4, 3, 6]
    event_shape = [5, num_tasks]
    sample_shape = [5, 3]

    samples = gp.sample(sample_shape, seed=test_util.test_seed())

    self.assertAllEqual(gp.batch_shape, batch_shape)
    self.assertAllEqual(gp.event_shape, event_shape)
    self.assertAllEqual(self.evaluate(gp.batch_shape_tensor()), batch_shape)
    self.assertAllEqual(self.evaluate(gp.event_shape_tensor()), event_shape)
    self.assertAllEqual(
        self.evaluate(samples).shape,
        sample_shape + batch_shape + event_shape)
    self.assertAllEqual(
        self.evaluate(gp.log_prob(samples)).shape,
        sample_shape + batch_shape)
    self.assertAllEqual(
        self.evaluate(tf.shape(gp.mean())), batch_shape + event_shape)

  def testValidateArgs(self):
    index_points = np.linspace(-4., 4., 10, dtype=np.float32)
    index_points = np.reshape(index_points, [5, 2])
    index_points = np.linspace(-4., 4., 16, dtype=np.float32)
    observation_index_points = np.reshape(index_points, [8, 2])

    observation_noise_variance = 1e-4
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)
    with self.assertRaisesRegex(ValueError, 'match the number of tasks'):
      observations = np.linspace(-1., 1., 24).astype(np.float32)
      mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
          multi_task_kernel,
          observation_index_points=observation_index_points,
          observations=observations,
          index_points=index_points,
          observation_noise_variance=observation_noise_variance,
          validate_args=True)

    with self.assertRaisesRegex(ValueError, 'match the number of tasks'):
      observations = np.linspace(-1., 1., 32).reshape(8, 4).astype(np.float32)
      mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
          multi_task_kernel,
          observation_index_points=observation_index_points,
          observations=observations,
          index_points=index_points,
          observation_noise_variance=observation_noise_variance,
          validate_args=True)

    with self.assertRaisesRegex(
        ValueError, 'match the second to last dimension'):
      observations = np.linspace(-1., 1., 18).reshape(6, 3).astype(np.float32)
      mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
          multi_task_kernel,
          observation_index_points=observation_index_points,
          observations=observations,
          index_points=index_points,
          observation_noise_variance=observation_noise_variance,
          validate_args=True)

  @parameterized.parameters(1, 3, 5)
  def testBindingIndexPoints(self, num_tasks):
    amplitude = np.float64(0.5)
    length_scale = np.float64(2.)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)

    # 5x5 grid of index points in R^2 and flatten to 9x2
    index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    observation_index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [9, 2]

    observations = np.linspace(-20., 20., 9 * num_tasks).reshape(9, num_tasks)

    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)
    observation_noise_variance = np.float64(1e-2)
    mtgp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        kernel=multi_task_kernel,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    gp = gprm.GaussianProcessRegressionModel(
        kernel=kernel,
        observation_index_points=observation_index_points,
        # Batch of num_task observations.
        observations=tf.linalg.matrix_transpose(observations),
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    test_points = np.random.uniform(-1., 1., [10, 2])
    test_observations = np.random.uniform(-1., 1., [10, num_tasks])

    multi_task_log_prob = mtgp.log_prob(
        test_observations, index_points=test_points)
    # Reduce over the first dimension which is tasks.
    single_task_log_prob = tf.reduce_sum(
        gp.log_prob(
            tf.linalg.matrix_transpose(test_observations),
            index_points=test_points), axis=0)
    self.assertAllClose(
        self.evaluate(single_task_log_prob),
        self.evaluate(multi_task_log_prob), rtol=1e-5)

    multi_task_mean_ = self.evaluate(mtgp.mean(index_points=test_points))
    # Reshape so that task dimension is last.
    single_task_mean_ = np.swapaxes(
        self.evaluate(gp.mean(index_points=test_points)),
        -1, -2)
    self.assertAllClose(
        single_task_mean_, multi_task_mean_, rtol=1e-5)

  @parameterized.parameters(1, 3, 5)
  def testLogProbMatchesGPNoiseless(self, num_tasks):
    # Check that the independent kernel parameterization matches using a
    # single-task GP.

    # 5x5 grid of index points in R^2 and flatten to 9x2
    index_points = np.linspace(-4., 4., 3, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [9, 2]

    amplitude = np.float32(0.5)
    length_scale = np.float32(2.)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    observation_noise_variance = None
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)

    observations = np.linspace(
        -20., 20., 9 * num_tasks).reshape(9, num_tasks).astype(np.float32)

    test_points = np.random.uniform(-1., 1., [10, 2]).astype(np.float32)

    mtgp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        multi_task_kernel,
        observation_index_points=index_points,
        index_points=test_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    test_observations = self.evaluate(mtgp.sample(seed=test_util.test_seed()))

    # For the single task GP, we move the task dimension to the front of the
    # batch shape.
    gp = gprm.GaussianProcessRegressionModel(
        kernel,
        observation_index_points=index_points,
        index_points=test_points,
        observations=tf.linalg.matrix_transpose(observations),
        observation_noise_variance=0.,
        validate_args=True)
    multitask_log_prob = mtgp.log_prob(test_observations)
    single_task_log_prob = tf.reduce_sum(
        gp.log_prob(
            tf.linalg.matrix_transpose(test_observations)), axis=0)
    self.assertAllClose(
        self.evaluate(single_task_log_prob),
        self.evaluate(multitask_log_prob), rtol=4e-3)

    multi_task_mean_ = self.evaluate(mtgp.mean())
    # Reshape so that task dimension is last.
    single_task_mean_ = np.swapaxes(
        self.evaluate(gp.mean()), -1, -2)
    self.assertAllClose(
        single_task_mean_, multi_task_mean_, rtol=1e-5)

  @parameterized.parameters(1, 3, 5)
  def testLogProbMatchesGP(self, num_tasks):
    # Check that the independent kernel parameterization matches using a
    # single-task GP.

    # 5x5 grid of index points in R^2 and flatten to 9x2
    index_points = np.linspace(-4., 4., 3, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [9, 2]

    amplitude = np.float32(0.5)
    length_scale = np.float32(2.)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    observation_noise_variance = np.float32(1e-2)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)

    observations = np.linspace(
        -20., 20., 9 * num_tasks).reshape(9, num_tasks).astype(np.float32)

    test_points = np.random.uniform(-1., 1., [10, 2]).astype(np.float32)
    test_observations = np.random.uniform(
        -20., 20., [10, num_tasks]).astype(np.float32)

    mtgp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        multi_task_kernel,
        observation_index_points=index_points,
        index_points=test_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    # For the single task GP, we move the task dimension to the front of the
    # batch shape.
    gp = gprm.GaussianProcessRegressionModel(
        kernel,
        observation_index_points=index_points,
        index_points=test_points,
        observations=tf.linalg.matrix_transpose(observations),
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    # Print batch of covariance matrices.
    multitask_log_prob = mtgp.log_prob(test_observations)
    single_task_log_prob = tf.reduce_sum(
        gp.log_prob(
            tf.linalg.matrix_transpose(test_observations)), axis=0)
    self.assertAllClose(
        self.evaluate(single_task_log_prob),
        self.evaluate(multitask_log_prob), rtol=4e-3)

    multi_task_mean_ = self.evaluate(mtgp.mean())
    # Reshape so that task dimension is last.
    single_task_mean_ = np.swapaxes(
        self.evaluate(gp.mean()),
        -1, -2)
    self.assertAllClose(
        single_task_mean_, multi_task_mean_, rtol=1e-5)

  @parameterized.parameters(1, 3, 5)
  def testNonTrivialMeanMatchesGP(self, num_tasks):
    # Check that the independent kernel parameterization matches using a
    # single-task GP.

    # 5x5 grid of index points in R^2 and flatten to 9x2
    index_points = np.linspace(-4., 4., 3, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [9, 2]

    amplitude = np.float32(0.5)
    length_scale = np.float32(2.)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    observation_noise_variance = np.float32(1e-2)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)

    observations = np.linspace(
        -20., 20., 9 * num_tasks).reshape(9, num_tasks).astype(np.float32)

    test_points = np.random.uniform(-1., 1., [10, 2]).astype(np.float32)
    test_observations = np.random.uniform(
        -20., 20., [10, num_tasks]).astype(np.float32)

    # Constant mean per task.
    mean_fn = lambda x: tf.linspace(1., 3., num_tasks)

    mtgp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        multi_task_kernel,
        observation_index_points=index_points,
        index_points=test_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        mean_fn=mean_fn,
        validate_args=True)

    # For the single task GP, we move the task dimension to the front of the
    # batch shape.
    gp = gprm.GaussianProcessRegressionModel(
        kernel,
        observation_index_points=index_points,
        index_points=test_points,
        observations=tf.linalg.matrix_transpose(observations),
        observation_noise_variance=observation_noise_variance,
        mean_fn=lambda x: tf.linspace(1., 3., num_tasks)[..., tf.newaxis],
        validate_args=True)
    # Print batch of covariance matrices.
    multitask_log_prob = mtgp.log_prob(test_observations)
    single_task_log_prob = tf.reduce_sum(
        gp.log_prob(
            tf.linalg.matrix_transpose(test_observations)), axis=0)
    self.assertAllClose(
        self.evaluate(single_task_log_prob),
        self.evaluate(multitask_log_prob), rtol=4e-3)

    multi_task_mean_ = self.evaluate(mtgp.mean())
    # Reshape so that task dimension is last.
    single_task_mean_ = np.swapaxes(
        self.evaluate(gp.mean()),
        -1, -2)
    self.assertAllClose(
        single_task_mean_, multi_task_mean_, rtol=1e-5)

  def testMasking(self):
    seed_idx, seed_obs, seed_test, seed_sample = (
        samplers.split_seed(test_util.test_seed(), 4))
    index_points = uniform.Uniform(-1., 1.).sample((4, 3, 2, 2), seed=seed_idx)
    observations = uniform.Uniform(-1., 1.).sample((4, 3, 2), seed=seed_obs)
    test_points = uniform.Uniform(-1., 1.).sample((4, 5, 2, 2), seed=seed_test)

    observations_is_missing = np.array([
        [[True, True], [False, True], [True, False]],
        [[False, True], [False, True], [False, True]],
        [[False, False], [True, True], [True, False]],
        [[True, False], [False, True], [False, False]]
    ])
    observations = tf.where(~observations_is_missing, observations, np.nan)

    amplitude = tf.convert_to_tensor([0.5, 1.0, 1.75, 3.5])
    length_scale = tf.convert_to_tensor([0.3, 0.6, 0.9, 1.2])
    kernel = multitask_kernel.Independent(
        2,
        exponentiated_quadratic.ExponentiatedQuadratic(
            amplitude, length_scale, feature_ndims=2),
        validate_args=True)

    def mean_fn(x):
      return (tf.math.reduce_sum(x, axis=[-1, -2])[..., tf.newaxis]
              * tf.convert_to_tensor([-0.5, 2.0]))

    mtgp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        kernel,
        observation_index_points=index_points,
        observations=observations,
        observations_is_missing=observations_is_missing,
        index_points=test_points,
        predictive_noise_variance=0.05,
        mean_fn=mean_fn,
        validate_args=True)

    # Compare to a GPRM where the task dimension has been moved to be the
    # rightmost batch dimension.
    gp = gprm.GaussianProcessRegressionModel.precompute_regression_model(
        kernel.base_kernel[..., tf.newaxis],
        observation_index_points=index_points[:, tf.newaxis],
        observations=tf.linalg.matrix_transpose(observations),
        observations_is_missing=tf.linalg.matrix_transpose(
            observations_is_missing),
        index_points=test_points[:, tf.newaxis],
        predictive_noise_variance=0.05,
        mean_fn=lambda x: tf.linalg.matrix_transpose(mean_fn(x[:, 0])),
        validate_args=True)

    x = mtgp.sample(2, seed=seed_sample)
    self.assertAllNotNan(mtgp.log_prob(x))
    self.assertAllClose(
        tf.math.reduce_sum(gp.log_prob(tf.linalg.matrix_transpose(x)), axis=-1),
        mtgp.log_prob(x))

    self.assertAllNotNan(mtgp.mean())
    self.assertAllClose(tf.linalg.matrix_transpose(gp.mean()), mtgp.mean())

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='Numpy has no jitting functionality')
  def testMeanVarianceJit(self):
    num_tasks = 3
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)

    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, 7, 2)).astype(np.float64))
    observations = np.linspace(
        -20., 20., 7 * num_tasks).reshape(7, num_tasks).astype(np.float64)

    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)
    mtgprm = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        kernel=multi_task_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    # Check that Jit compiling mean and variance doesn't raise an error.
    tf.function(jit_compile=True)(mtgprm.mean)()
    tf.function(jit_compile=True)(mtgprm.variance)()

  @parameterized.parameters(True, False)
  def testMeanVarianceAndCovariancePrecomputed(self, has_missing_observations):
    num_tasks = 3
    num_obs = 7
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)

    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, num_obs, 2)).astype(np.float64))
    observations = np.linspace(
        -20., 20., num_obs * num_tasks).reshape(
            num_obs, num_tasks).astype(np.float64)

    if has_missing_observations:
      observations_is_missing = np.stack(
          [np.random.randint(2, size=(num_obs,))] * num_tasks, axis=-1
          ).astype(np.bool_)
    else:
      observations_is_missing = None

    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)
    mtgprm = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        kernel=multi_task_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        observations_is_missing=observations_is_missing,
        validate_args=True)

    precomputed_mtgprm = mtgprm_lib.MultiTaskGaussianProcessRegressionModel.precompute_regression_model(
        kernel=multi_task_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        observations_is_missing=observations_is_missing,
        validate_args=True)

    mock_cholesky_fn = mock.Mock(return_value=None)
    rebuilt_precomputed_mtgprm = mtgprm_lib.MultiTaskGaussianProcessRegressionModel.precompute_regression_model(
        kernel=multi_task_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        observations_is_missing=observations_is_missing,
        _precomputed_divisor_matrix_cholesky=precomputed_mtgprm._precomputed_divisor_matrix_cholesky,
        _precomputed_solve_on_observation=precomputed_mtgprm._precomputed_solve_on_observation,
        cholesky_fn=mock_cholesky_fn,
        validate_args=True)
    mock_cholesky_fn.assert_not_called()

    rebuilt_precomputed_mtgprm = rebuilt_precomputed_mtgprm.copy(
        cholesky_fn=None)
    self.assertAllClose(self.evaluate(precomputed_mtgprm.variance()),
                        self.evaluate(mtgprm.variance()))
    self.assertAllClose(self.evaluate(precomputed_mtgprm.mean()),
                        self.evaluate(mtgprm.mean()))
    self.assertAllClose(self.evaluate(rebuilt_precomputed_mtgprm.variance()),
                        self.evaluate(mtgprm.variance()))
    self.assertAllClose(self.evaluate(rebuilt_precomputed_mtgprm.mean()),
                        self.evaluate(mtgprm.mean()))

  def testPrecomputedWithMasking(self):
    num_tasks = 2
    amplitude = np.array([1., 2., 3., 4.], np.float64)
    length_scale = np.array([[.1], [.2], [.3]], np.float64)
    observation_noise_variance = np.array([[1e-2], [1e-4], [1e-6]], np.float64)

    rng = test_util.test_np_rng()
    # [4, 3, num_tasks]
    observations_is_missing = np.array([
        [[True, False], [False, True], [True, False]],
        [[False, True], [False, True], [False, True]],
        [[False, False], [False, True], [True, False]],
        [[True, False], [False, True], [False, False]]
    ])
    observations = np.linspace(
        -20., 20., 3 * num_tasks).reshape(3, num_tasks).astype(np.float64)
    observations = tf.where(~observations_is_missing, observations, np.nan)

    index_points = np.linspace(1., 4., 25).reshape(5, 5).astype(np.float64)

    observation_index_points = rng.uniform(
        -1., 1., (3, 1, 3, 5)).astype(np.float64)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    task_kernel_matrix = np.array([[6., 2.],
                                   [2., 7.]],
                                  dtype=np.float64)
    task_kernel_matrix_linop = tf.linalg.LinearOperatorFullMatrix(
        task_kernel_matrix)
    multi_task_kernel = multitask_kernel.Separable(
        num_tasks=num_tasks,
        task_kernel_matrix_linop=task_kernel_matrix_linop,
        base_kernel=kernel)
    mtgprm = mtgprm_lib.MultiTaskGaussianProcessRegressionModel.precompute_regression_model(
        kernel=multi_task_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observations_is_missing=observations_is_missing,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    mock_cholesky_fn = mock.Mock(return_value=None)
    rebuilt_mtgprm = mtgprm_lib.MultiTaskGaussianProcessRegressionModel.precompute_regression_model(
        kernel=multi_task_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        _precomputed_divisor_matrix_cholesky=mtgprm._precomputed_divisor_matrix_cholesky,
        _precomputed_solve_on_observation=mtgprm._precomputed_solve_on_observation,
        cholesky_fn=mock_cholesky_fn,
        validate_args=True)
    mock_cholesky_fn.assert_not_called()

    rebuilt_mtgprm = rebuilt_mtgprm.copy(cholesky_fn=None)
    self.assertAllNotNan(mtgprm.mean())
    self.assertAllNotNan(mtgprm.variance())
    self.assertAllClose(self.evaluate(mtgprm.variance()),
                        self.evaluate(rebuilt_mtgprm.variance()))
    self.assertAllClose(self.evaluate(mtgprm.mean()),
                        self.evaluate(rebuilt_mtgprm.mean()))

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Numpy has no notion of CompositeTensor/Pytree/saved_model')
  def testPrecomputedCompositeTensor(self):
    num_tasks = 3
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)

    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, 7, 2)).astype(np.float64))
    observations = np.random.uniform(
        -1., 1., (1, 1, 7, num_tasks)).astype(np.float64)

    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)

    precomputed_mtgprm = mtgprm_lib.MultiTaskGaussianProcessRegressionModel.precompute_regression_model(
        kernel=multi_task_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    flat = tf.nest.flatten(precomputed_mtgprm, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        precomputed_mtgprm, flat, expand_composites=True)
    self.assertIsInstance(unflat,
                          mtgprm_lib.MultiTaskGaussianProcessRegressionModel)
    self.assertIsInstance(unflat, tf.__internal__.CompositeTensor)
    # Check that we don't recompute the scale matrix on flattening /
    # unflattening. In this case it's a kronecker product of a lower triangular
    # and an identity, so we only check the first factor.
    self.assertIs(precomputed_mtgprm._observation_scale.operators[0]._tril,  # pylint:disable=protected-access
                  unflat._observation_scale.operators[0]._tril)  # pylint:disable=protected-access

  def testStructuredIndexPoints(self):
    num_tasks = 3
    observation_index_points = np.random.uniform(
        -1, 1, (12, 8)).astype(np.float32)
    observations = np.random.uniform(-1, 1, (12, num_tasks)).astype(np.float32)
    index_points = np.random.uniform(-1, 1, (6, 8)).astype(np.float32)
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    base_mtk = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=base_kernel)
    base_mtgp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        base_mtk,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations)

    structured_kernel = psd_kernel_test_util.MultipartTestKernel(base_kernel)
    structured_mtk = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=structured_kernel)
    structured_obs_index_points = dict(
        zip(('foo', 'bar'),
            tf.split(observation_index_points, [5, 3], axis=-1)))
    structured_index_points = dict(
        zip(('foo', 'bar'), tf.split(index_points, [5, 3], axis=-1)))
    structured_mtgp = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        structured_mtk,
        index_points=structured_index_points,
        observation_index_points=structured_obs_index_points,
        observations=observations)

    s = structured_mtgp.sample(3, seed=test_util.test_seed())
    self.assertAllClose(base_mtgp.log_prob(s), structured_mtgp.log_prob(s))
    self.assertAllClose(base_mtgp.mean(), structured_mtgp.mean())
    self.assertAllClose(base_mtgp.variance(), structured_mtgp.variance())
    self.assertAllEqual(base_mtgp.event_shape, structured_mtgp.event_shape)
    self.assertAllEqual(base_mtgp.event_shape_tensor(),
                        structured_mtgp.event_shape_tensor())
    self.assertAllEqual(base_mtgp.batch_shape, structured_mtgp.batch_shape)
    self.assertAllEqual(base_mtgp.batch_shape_tensor(),
                        structured_mtgp.batch_shape_tensor())

    # Iterable index points should be interpreted as single Tensors if the
    # kernel is not structured.
    index_points_list = tf.unstack(index_points)
    obs_index_points_nested_list = tf.nest.map_structure(
        tf.unstack, tf.unstack(observation_index_points))
    mtgp_with_lists = mtgprm_lib.MultiTaskGaussianProcessRegressionModel(
        base_mtk,
        index_points=index_points_list,
        observation_index_points=obs_index_points_nested_list,
        observations=observations)
    self.assertAllEqual(base_mtgp.event_shape_tensor(),
                        mtgp_with_lists.event_shape_tensor())
    self.assertAllEqual(base_mtgp.batch_shape_tensor(),
                        mtgp_with_lists.batch_shape_tensor())
    self.assertAllClose(base_mtgp.log_prob(s), mtgp_with_lists.log_prob(s))

if __name__ == '__main__':
  test_util.main()
