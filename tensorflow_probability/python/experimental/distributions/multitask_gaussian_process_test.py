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
"""Tests for MultiTaskGaussianProcess."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gaussian_process
from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process
from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process_regression_model as mtgprm
from tensorflow_probability.python.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


class InefficientSeparable(multitask_kernel.MultiTaskKernel):
  """A version of the Separable kernel that's inefficient."""

  def __init__(self,
               num_tasks,
               base_kernel,
               task_kernel_matrix_linop,
               name='InefficientSeparable',
               validate_args=False):

    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype(
          [task_kernel_matrix_linop, base_kernel], tf.float32)
      self._base_kernel = base_kernel
      self._task_kernel_matrix_linop = tensor_util.convert_nonref_to_tensor(
          task_kernel_matrix_linop, dtype, name='task_kernel_matrix_linop')
      super(InefficientSeparable, self).__init__(
          num_tasks=num_tasks,
          dtype=dtype,
          feature_ndims=base_kernel.feature_ndims,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def base_kernel(self):
    return self._base_kernel

  @property
  def task_kernel_matrix_linop(self):
    return self._task_kernel_matrix_linop

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        base_kernel=parameter_properties.BatchedComponentProperties(),
        task_kernel_matrix_linop=(
            parameter_properties.BatchedComponentProperties()))

  def _matrix_over_all_tasks(self, x1, x2):
    # Because the kernel computations are independent of task,
    # we can use a Kronecker product of an identity matrix.
    base_kernel_matrix = tf.linalg.LinearOperatorFullMatrix(
        self.base_kernel.matrix(x1, x2))
    operator = tf.linalg.LinearOperatorKronecker(
        [base_kernel_matrix, self._task_kernel_matrix_linop])
    return tf.linalg.LinearOperatorFullMatrix(operator.to_dense())


@test_util.test_all_tf_execution_regimes
class MultiTaskGaussianProcessTest(test_util.TestCase):

  def testShapes(self):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 3, 1]
    amplitude = np.array([1., 2.], np.float32).reshape([2, 1, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float32).reshape([1, 4, 1, 1])
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-5], np.float32).reshape([1, 1, 3, 1])
    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 25, 2]
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)
    gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    batch_shape = [2, 4, 3, 6]
    event_shape = [25, 3]
    sample_shape = [5, 3]

    samples = gp.sample(sample_shape, seed=test_util.test_seed())

    self.assertAllEqual(gp.batch_shape, batch_shape)
    self.assertAllEqual(self.evaluate(gp.batch_shape_tensor()), batch_shape)
    self.assertAllEqual(gp.event_shape, event_shape)
    self.assertAllEqual(self.evaluate(gp.event_shape_tensor()), event_shape)
    self.assertAllEqual(
        self.evaluate(samples).shape,
        sample_shape + batch_shape + event_shape)
    self.assertAllEqual(
        self.evaluate(tf.shape(gp.mean())), batch_shape + event_shape)

    self.assertAllEqual(
        self.evaluate(tf.shape(gp.variance())), batch_shape + event_shape)

  def testBindingIndexPoints(self):
    amplitude = np.float64(0.5)
    length_scale = np.float64(2.)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    num_tasks = 3
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=num_tasks, base_kernel=kernel)
    mean_fn = lambda x: tf.stack([x[..., 0]] * num_tasks, axis=-1)
    observation_noise_variance = np.float64(1e-3)
    mtgp = multitask_gaussian_process.MultiTaskGaussianProcess(
        kernel=multi_task_kernel,
        mean_fn=mean_fn,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    gp = gaussian_process.GaussianProcess(
        kernel=kernel,
        mean_fn=lambda x: x[..., 0],
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    index_points = np.random.uniform(-1., 1., [10, 4])
    observations = np.random.uniform(-1., 1., [10, num_tasks])

    multi_task_log_prob = mtgp.log_prob(
        observations, index_points=index_points)
    single_task_log_prob = sum(
        gp.log_prob(
            observations[..., i], index_points=index_points)
        for i in range(num_tasks))
    self.assertAllClose(
        self.evaluate(single_task_log_prob),
        self.evaluate(multi_task_log_prob), rtol=4e-3)

    multi_task_mean_ = self.evaluate(mtgp.mean(index_points=index_points))
    single_task_mean_ = self.evaluate(gp.mean(index_points=index_points))
    for i in range(3):
      self.assertAllClose(
          single_task_mean_, multi_task_mean_[..., i], rtol=1e-3)

    multi_task_var_ = self.evaluate(mtgp.variance(index_points=index_points))
    single_task_var_ = self.evaluate(gp.variance(index_points=index_points))
    for i in range(3):
      self.assertAllClose(
          single_task_var_, multi_task_var_[..., i], rtol=1e-3)

  def testConstantMeanFunction(self):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 3, 1]
    amplitude = np.array([1., 2.], np.float32).reshape([2, 1, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float32).reshape([1, 4, 1, 1])
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-5], np.float32).reshape([1, 1, 3, 1])
    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 25, 2]
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)

    mean_fn = lambda x: np.float32(0.)

    gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        mean_fn=mean_fn,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    batch_shape = [2, 4, 3, 6]
    event_shape = [25, 3]

    self.assertAllEqual(
        self.evaluate(gp.mean()).shape,
        batch_shape + event_shape)

    mean_fn = lambda x: tf.zeros([3], dtype=tf.float32)

    gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        mean_fn=mean_fn,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    self.assertAllEqual(
        self.evaluate(gp.mean()).shape,
        batch_shape + event_shape)

  def testLogProbMatchesGPNoiseless(self):
    # Check that the independent kernel parameterization matches using a
    # single-task GP.

    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float32).reshape([2, 1, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float32).reshape([1, 4, 1, 1])
    observation_noise_variance = None
    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 25, 2]
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)
    multitask_gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    gp = gaussian_process.GaussianProcess(
        kernel,
        batched_index_points,
        observation_noise_variance=0.,
        validate_args=True)
    observations = np.linspace(-20., 20., 75).reshape(25, 3).astype(np.float32)
    multitask_log_prob = multitask_gp.log_prob(observations)
    single_task_log_prob = sum(
        gp.log_prob(observations[..., i]) for i in range(3))
    self.assertAllClose(
        self.evaluate(single_task_log_prob),
        self.evaluate(multitask_log_prob), rtol=4e-3)

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=False,
      reason='Jit not available in numpy.')
  def testJitMultitaskGaussianProcess(self):
    # 3x3 grid of index points in R^2 and flatten to 9x2
    index_points = np.linspace(-4., 4., 3, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [9, 2]

    # Kernel with batch_shape [2, 4, 3]
    amplitude = np.array([1., 2.], np.float32).reshape([2, 1,])
    length_scale = np.array([1., 2., 3., 4.], np.float32).reshape([1, 4,])
    observation_noise_variance = np.float32(1e-5)
    batched_index_points = np.stack([index_points]*4)
    # ==> shape = [4, 9, 2]
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)
    multitask_gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    @tf.function(jit_compile=True)
    def log_prob(o):
      return multitask_gp.log_prob(o)

    @tf.function(jit_compile=True)
    def sample():
      return multitask_gp.sample(seed=test_util.test_seed())

    observations = tf.convert_to_tensor(
        np.linspace(-20., 20., 27).reshape(9, 3).astype(np.float32))
    self.assertAllEqual(log_prob(observations).shape, [2, 4])
    self.assertAllEqual(sample().shape, [2, 4, 9, 3])

    multitask_gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        observation_noise_variance=None,
        validate_args=True)

    @tf.function(jit_compile=True)
    def log_prob_no_noise(o):
      return multitask_gp.log_prob(o)

    @tf.function(jit_compile=True)
    def sample_no_noise():
      return multitask_gp.sample(seed=test_util.test_seed())

    self.assertAllEqual(log_prob_no_noise(observations).shape, [2, 4])
    self.assertAllEqual(sample_no_noise().shape, [2, 4, 9, 3])

  def testMultiTaskBlockSeparable(self):
    # Check that the naive implementation matches any optimizations for a
    # Separable Kernel.

    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-10., 10., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 3, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float64).reshape([1, 4, 1, 1])
    observation_noise_variance = np.array(
        [1e-3, 1e-2, 1e-1], np.float64).reshape([1, 1, 3, 1])
    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 25, 2]
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    # Ensure Symmetric + Strictly Diagonally Dominant -> Positive Definite.
    task_kernel_matrix = np.array([[6., 2., 3.],
                                   [2., 7., 4.],
                                   [3., 4., 8.]],
                                  dtype=np.float64)
    task_kernel_matrix_linop = tf.linalg.LinearOperatorFullMatrix(
        task_kernel_matrix)
    multi_task_kernel = multitask_kernel.Separable(
        num_tasks=3,
        task_kernel_matrix_linop=task_kernel_matrix_linop,
        base_kernel=kernel)
    multitask_gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    naive_multi_task_kernel = InefficientSeparable(
        num_tasks=3, task_kernel_matrix_linop=task_kernel_matrix_linop,
        base_kernel=kernel)
    actual_multitask_gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        naive_multi_task_kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=False)

    observations = np.linspace(-20., 20., 75).reshape(25, 3).astype(np.float64)
    multitask_log_prob = multitask_gp.log_prob(observations)
    actual_multitask_log_prob = actual_multitask_gp.log_prob(observations)
    self.assertAllClose(
        self.evaluate(actual_multitask_log_prob),
        self.evaluate(multitask_log_prob), rtol=4e-3)

    multitask_mean = multitask_gp.mean()
    actual_multitask_mean = actual_multitask_gp.mean()
    self.assertAllClose(
        self.evaluate(actual_multitask_mean),
        self.evaluate(multitask_mean), rtol=4e-3)

    multitask_var = multitask_gp.variance()
    actual_multitask_var = actual_multitask_gp.variance()
    self.assertAllClose(
        self.evaluate(actual_multitask_var),
        self.evaluate(multitask_var), rtol=4e-3)

  def testLogProbValidateArgs(self):
    index_points = np.linspace(-4., 4., 10, dtype=np.float32)
    index_points = np.reshape(index_points, [-1, 2])

    observation_noise_variance = 1e-4
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)
    multitask_gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    with self.assertRaisesRegexp(ValueError, 'match the number of tasks'):
      observations = np.linspace(-1., 1., 15).astype(np.float32)
      multitask_gp.log_prob(observations)

    with self.assertRaisesRegexp(ValueError, 'match the number of tasks'):
      observations = np.linspace(-1., 1., 20).reshape(5, 4).astype(np.float32)
      multitask_gp.log_prob(observations)

    with self.assertRaisesRegexp(
        ValueError, 'match the second to last dimension'):
      observations = np.linspace(-1., 1., 18).reshape(6, 3).astype(np.float32)
      multitask_gp.log_prob(observations)

  def testLogProbMatchesGP(self):
    # Check that the independent kernel parameterization matches using a
    # single-task GP.
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 3, 1]
    amplitude = np.array([1., 2.], np.float32).reshape([2, 1, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float32).reshape([1, 4, 1, 1])
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-5], np.float32).reshape([1, 1, 3, 1])
    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 25, 2]
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    multi_task_kernel = multitask_kernel.Independent(
        num_tasks=3, base_kernel=kernel)
    multitask_gp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    gp = gaussian_process.GaussianProcess(
        kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)
    observations = np.linspace(-20., 20., 75).reshape(25, 3).astype(np.float32)
    multitask_log_prob = multitask_gp.log_prob(observations)
    single_task_log_prob = sum(
        gp.log_prob(observations[..., i]) for i in range(3))
    self.assertAllClose(
        self.evaluate(single_task_log_prob),
        self.evaluate(multitask_log_prob), rtol=4e-3)

  def testMTGPPosteriorPredictive(self):
    amplitude = np.float64(.5)
    length_scale = np.float64(2.)
    observation_noise_variance = np.float64(3e-3)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    # Ensure Symmetric + Strictly Diagonally Dominant -> Positive Definite.
    task_kernel_matrix = np.array([[6., 2., 3.],
                                   [2., 7., 4.],
                                   [3., 4., 8.]],
                                  dtype=np.float64)
    task_kernel_matrix_linop = tf.linalg.LinearOperatorFullMatrix(
        task_kernel_matrix)
    multi_task_kernel = multitask_kernel.Separable(
        num_tasks=3,
        task_kernel_matrix_linop=task_kernel_matrix_linop,
        base_kernel=kernel)

    index_points = np.random.uniform(-1., 1., 10)[..., np.newaxis]

    mtgp = multitask_gaussian_process.MultiTaskGaussianProcess(
        multi_task_kernel,
        index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    predictive_index_points = np.random.uniform(1., 2., 10)[..., np.newaxis]
    observations = np.linspace(1., 10., 30).reshape(10, 3)

    expected_mtgprm = mtgprm.MultiTaskGaussianProcessRegressionModel(
        kernel=multi_task_kernel,
        observation_index_points=index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        index_points=predictive_index_points,
        validate_args=True)

    actual_mtgprm = mtgp.posterior_predictive(
        predictive_index_points=predictive_index_points,
        observations=observations)

    samples = self.evaluate(
        actual_mtgprm.sample(10, seed=test_util.test_seed()))

    self.assertAllClose(
        self.evaluate(expected_mtgprm.mean()),
        self.evaluate(actual_mtgprm.mean()))

    self.assertAllClose(
        self.evaluate(expected_mtgprm.log_prob(samples)),
        self.evaluate(actual_mtgprm.log_prob(samples)))


if __name__ == '__main__':
  test_util.main()
