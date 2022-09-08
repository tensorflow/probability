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
# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import gaussian_process_regression_model as gprm
from tensorflow_probability.python.distributions import student_t_process
from tensorflow_probability.python.distributions import student_t_process_regression_model as stprm
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import psd_kernels


@test_util.test_all_tf_execution_regimes
class StudentTProcessRegressionModelTest(test_util.TestCase):

  def testInstantiate(self):
    df = np.float64(1.)
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 1, 3]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape(
        [1, 4, 1, 1])
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-9], np.float64).reshape([1, 1, 1, 3])

    observation_index_points = (
        np.random.uniform(-1., 1., (3, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (3, 7)).astype(np.float64)

    def cholesky_fn(x):
      return tf.linalg.cholesky(
          tf.linalg.set_diag(x, tf.linalg.diag_part(x) + 1.))

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
    dist = stprm.StudentTProcessRegressionModel(
        df=df,
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=cholesky_fn)
    batch_shape = [2, 4, 1, 3]
    event_shape = [25]
    sample_shape = [7, 2]

    self.assertIs(cholesky_fn, dist.cholesky_fn)

    samples = dist.sample(sample_shape, seed=test_util.test_seed())
    self.assertAllEqual(dist.batch_shape_tensor(), batch_shape)
    self.assertAllEqual(dist.event_shape_tensor(), event_shape)
    self.assertAllEqual(self.evaluate(samples).shape,
                        sample_shape + batch_shape + event_shape)

  def testMeanSameAsGPRM(self):
    df = np.float64(3.)
    index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])

    # Kernel with batch_shape [5, 3]
    amplitude = np.array([1., 2., 3., 4., 5.], np.float64).reshape([5, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape(
        [1, 3])
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-9], np.float64).reshape([1, 3])

    observation_index_points = (
        np.random.uniform(-1., 1., (3, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (3, 7)).astype(np.float64)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
    dist = stprm.StudentTProcessRegressionModel(
        df=df,
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance)
    dist_gp = gprm.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance)

    self.assertAllClose(
        self.evaluate(dist.mean()), self.evaluate(dist_gp.mean()))

  def testLogProbNearGPRM(self):
    # For large df, the log_prob calculations should be the same.
    df = np.float64(1e6)
    index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])

    # Kernel with batch_shape [5, 3]
    amplitude = np.array([1., 2., 3., 4., 5.], np.float64).reshape([5, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape(
        [1, 3])
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-9], np.float64).reshape([1, 3])

    observation_index_points = (
        np.random.uniform(-1., 1., (3, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (3, 7)).astype(np.float64)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
    dist = stprm.StudentTProcessRegressionModel(
        df=df,
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance)
    dist_gp = gprm.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance)

    x = np.linspace(-3., 3., 25)

    self.assertAllClose(
        self.evaluate(dist.log_prob(x)),
        self.evaluate(dist_gp.log_prob(x)),
        rtol=2e-5)

  def testMeanVarianceAndCovariancePrecomputed(self):
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)
    df = np.float64(3.)

    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (1, 1, 7)).astype(np.float64)

    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
    dist = stprm.StudentTProcessRegressionModel(
        df=df,
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    precomputed_dist = stprm.StudentTProcessRegressionModel.precompute_regression_model(
        df=df,
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    self.assertAllClose(
        self.evaluate(precomputed_dist.covariance()),
        self.evaluate(dist.covariance()))
    self.assertAllClose(
        self.evaluate(precomputed_dist.variance()),
        self.evaluate(dist.variance()))
    self.assertAllClose(
        self.evaluate(precomputed_dist.mean()), self.evaluate(dist.mean()))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='Numpy and JAX have no notion of CompositeTensor/saved_model')
  def testPrecomputedCompositeTensor(self):
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)

    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (1, 1, 7)).astype(np.float64)

    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    precomputed_dist = stprm.StudentTProcessRegressionModel.precompute_regression_model(
        df=3.,
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    flat = tf.nest.flatten(precomputed_dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        precomputed_dist, flat, expand_composites=True)
    self.assertIsInstance(unflat, stprm.StudentTProcessRegressionModel)
    # Check that we don't recompute the divisor matrix on flattening /
    # unflattening.
    self.assertIs(
        precomputed_dist.kernel.schur_complement
        ._precomputed_divisor_matrix_cholesky,  # pylint:disable=line-too-long
        unflat.kernel.schur_complement._precomputed_divisor_matrix_cholesky)

    # TODO(b/196219597): Enable this test once dist works across TF function
    # boundaries.
    # index_observations = np.random.uniform(-1., 1., (6,)).astype(np.float64)
    # @tf.function
    # def log_prob(d):
    #   return d.log_prob(index_observations)

    # lp = self.evaluate(precomputed_dist.log_prob(index_observations))

    # self.assertAllClose(lp, self.evaluate(log_prob(precomputed_dist)))
    # self.assertAllClose(lp, self.evaluate(log_prob(unflat)))

  def testEmptyDataMatchesStPPrior(self):
    df = np.float64(3.5)
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    index_points = np.random.uniform(-1., 1., (10, 1)).astype(np.float64)

    # k_xx - k_xn @ (k_nn + sigma^2) @ k_nx + sigma^2
    mean_fn = lambda x: x[:, 0]**2

    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)
    stp = student_t_process.StudentTProcess(
        df, kernel, index_points, mean_fn=mean_fn, validate_args=True)

    dist_nones = stprm.StudentTProcessRegressionModel(
        df,
        kernel=kernel,
        index_points=index_points,
        mean_fn=mean_fn,
        validate_args=True)

    dist_zero_shapes = stprm.StudentTProcessRegressionModel(
        df,
        kernel=kernel,
        index_points=index_points,
        observation_index_points=tf.ones([0, 1], tf.float64),
        observations=tf.ones([0], tf.float64),
        mean_fn=mean_fn,
        validate_args=True)

    for dist in [dist_nones, dist_zero_shapes]:
      self.assertAllClose(self.evaluate(stp.mean()), self.evaluate(dist.mean()))
      self.assertAllClose(
          self.evaluate(stp.covariance()), self.evaluate(dist.covariance()))
      self.assertAllClose(
          self.evaluate(stp.variance()), self.evaluate(dist.variance()))

      observations = np.random.uniform(-1., 1., 10).astype(np.float64)
      self.assertAllClose(
          self.evaluate(stp.log_prob(observations)),
          self.evaluate(dist.log_prob(observations)))

  def testCopy(self):
    # 5 random index points in R^2
    index_points_1 = np.random.uniform(-4., 4., (5, 2)).astype(np.float32)
    # 10 random index points in R^2
    index_points_2 = np.random.uniform(-4., 4., (10, 2)).astype(np.float32)

    observation_index_points_1 = (
        np.random.uniform(-4., 4., (7, 2)).astype(np.float32))
    observation_index_points_2 = (
        np.random.uniform(-4., 4., (9, 2)).astype(np.float32))

    observations_1 = np.random.uniform(-1., 1., 7).astype(np.float32)
    observations_2 = np.random.uniform(-1., 1., 9).astype(np.float32)

    # ==> shape = [6, 25, 2]
    mean_fn = lambda x: np.array([0.], np.float32)
    kernel_1 = psd_kernels.ExponentiatedQuadratic()
    kernel_2 = psd_kernels.ExpSinSquared()

    dist1 = stprm.StudentTProcessRegressionModel(
        df=5.,
        kernel=kernel_1,
        index_points=index_points_1,
        observation_index_points=observation_index_points_1,
        observations=observations_1,
        mean_fn=mean_fn,
        validate_args=True)
    dist2 = dist1.copy(
        kernel=kernel_2,
        index_points=index_points_2,
        observation_index_points=observation_index_points_2,
        observations=observations_2)

    precomputed_dist1 = (
        stprm.StudentTProcessRegressionModel.precompute_regression_model(
            df=5.,
            kernel=kernel_1,
            index_points=index_points_1,
            observation_index_points=observation_index_points_1,
            observations=observations_1,
            mean_fn=mean_fn,
            validate_args=True))
    precomputed_dist2 = precomputed_dist1.copy(index_points=index_points_2)
    self.assertIs(precomputed_dist1.mean_fn, precomputed_dist2.mean_fn)
    self.assertIs(precomputed_dist1.kernel, precomputed_dist2.kernel)

    event_shape_1 = [5]
    event_shape_2 = [10]

    self.assertIsInstance(dist1.kernel.schur_complement.base_kernel,
                          psd_kernels.ExponentiatedQuadratic)
    self.assertIsInstance(dist2.kernel.schur_complement.base_kernel,
                          psd_kernels.ExpSinSquared)
    self.assertAllEqual(
        self.evaluate(dist1.batch_shape_tensor()),
        self.evaluate(dist2.batch_shape_tensor()))
    self.assertAllEqual(
        self.evaluate(dist1.event_shape_tensor()), event_shape_1)
    self.assertAllEqual(
        self.evaluate(dist2.event_shape_tensor()), event_shape_2)
    self.assertAllEqual(self.evaluate(dist1.index_points), index_points_1)
    self.assertAllEqual(self.evaluate(dist2.index_points), index_points_2)

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Numpy has no notion of CompositeTensor/Pytree.')
  def testCompositeTensorOrPytree(self):
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)
    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (1, 1, 7)).astype(np.float64)
    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    def cholesky_fn(x):
      return tf.linalg.cholesky(
          tf.linalg.set_diag(x, tf.linalg.diag_part(x) + 1.))

    dist = stprm.StudentTProcessRegressionModel(
        df=np.float64(5.),
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=cholesky_fn)

    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.assertIsInstance(unflat, stprm.StudentTProcessRegressionModel)
    # Check that we don't recompute the divisor matrix on flattening /
    # unflattening.
    self.assertIs(
        dist.kernel.schur_complement._precomputed_divisor_matrix_cholesky,
        unflat.kernel.schur_complement._precomputed_divisor_matrix_cholesky)

    x = self.evaluate(dist.sample(3, seed=test_util.test_seed()))
    actual = self.evaluate(dist.log_prob(x))
    self.assertAllClose(self.evaluate(unflat.log_prob(x)), actual)

    # TODO(b/196219597): Enable this test once dist works across TF function
    # boundaries.
    # @tf.function
    # def call_log_prob(d):
    #   return d.log_prob(x)
    # self.assertAllClose(actual, call_log_prob(dist))
    # self.assertAllClose(actual, call_log_prob(unflat))


if __name__ == '__main__':
  test_util.main()
