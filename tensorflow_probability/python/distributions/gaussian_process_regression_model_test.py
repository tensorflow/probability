# Copyright 2018 The TensorFlow Probability Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import positive_semidefinite_kernels as psd_kernels

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def _np_kernel_matrix_fn(amp, len_scale, x, y):
  x = np.expand_dims(x, -2)[..., 0]
  y = np.expand_dims(y, -3)[..., 0]
  return amp ** 2 * np.exp(-.5 * ((x - y)**2) / (len_scale**2))


np.random.seed(42)


class _GaussianProcessRegressionModelTest(object):

  def testShapes(self):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]
    batched_index_points = np.expand_dims(np.stack([index_points]*6), -3)
    # ==> shape = [6, 1, 25, 2]

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape([1, 4, 1, 1])

    jitter = np.float64(1e-6)
    observation_noise_variance = np.float64(1e-2)
    observation_index_points = (
        np.random.uniform(-1., 1., (3, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (3, 7)).astype(np.float64)

    if not self.is_static:
      amplitude = tf.compat.v1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf.compat.v1.placeholder_with_default(
          length_scale, shape=None)
      batched_index_points = tf.compat.v1.placeholder_with_default(
          batched_index_points, shape=None)

      observation_index_points = tf.compat.v1.placeholder_with_default(
          observation_index_points, shape=None)
      observations = tf.compat.v1.placeholder_with_default(
          observations, shape=None)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    gprm = tfd.GaussianProcessRegressionModel(
        kernel,
        batched_index_points,
        observation_index_points,
        observations,
        observation_noise_variance,
        jitter=jitter)

    batch_shape = [2, 4, 6, 3]
    event_shape = [25]
    sample_shape = [9, 3]

    samples = gprm.sample(sample_shape)

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(gprm.batch_shape_tensor(), batch_shape)
      self.assertAllEqual(gprm.event_shape_tensor(), event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(gprm.batch_shape, batch_shape)
      self.assertAllEqual(gprm.event_shape, event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
    else:
      self.assertAllEqual(self.evaluate(gprm.batch_shape_tensor()), batch_shape)
      self.assertAllEqual(self.evaluate(gprm.event_shape_tensor()), event_shape)
      self.assertAllEqual(self.evaluate(samples).shape,
                          sample_shape + batch_shape + event_shape)
      self.assertIsNone(samples.shape.ndims)
      self.assertIsNone(gprm.batch_shape.ndims)
      self.assertEqual(gprm.event_shape.ndims, 1)
      self.assertIsNone(tf.compat.dimension_value(gprm.event_shape.dims[0]))

  def testMeanVarianceAndCovariance(self):
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    observation_noise_variance = np.float64(1e-3)
    jitter = np.float64(1e-4)
    num_test = 10
    num_obs = 3
    index_points = np.random.uniform(-1., 1., (num_test, 1)).astype(np.float64)
    observation_index_points = (
        np.random.uniform(-1., 1., (num_obs, 1)).astype(np.float64))
    observations = np.random.uniform(-1., 1., 3).astype(np.float64)

    # k_xx - k_xn @ (k_nn + sigma^2) @ k_nx + sigma^2
    k = lambda x, y: _np_kernel_matrix_fn(amp, len_scale, x, y)
    k_xx_ = k(index_points, index_points)
    k_xn_ = k(index_points, observation_index_points)
    k_nn_plus_noise_ = (
        k(observation_index_points, observation_index_points) +
        (jitter + observation_noise_variance) * np.eye(num_obs))

    expected_predictive_covariance_no_noise = (
        k_xx_ - np.dot(k_xn_, np.linalg.solve(k_nn_plus_noise_, k_xn_.T)) +
        np.eye(num_test) * jitter)

    expected_predictive_covariance_with_noise = (
        expected_predictive_covariance_no_noise +
        np.eye(num_test) * observation_noise_variance)

    mean_fn = lambda x: x[:, 0]**2
    prior_mean = mean_fn(observation_index_points)
    expected_mean = (
        mean_fn(index_points) +
        np.dot(k_xn_,
               np.linalg.solve(k_nn_plus_noise_, observations - prior_mean)))

    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)
    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        mean_fn=mean_fn,
        jitter=jitter)

    self.assertAllClose(expected_predictive_covariance_with_noise,
                        self.evaluate(gprm.covariance()))
    self.assertAllClose(np.diag(expected_predictive_covariance_with_noise),
                        self.evaluate(gprm.variance()))
    self.assertAllClose(expected_mean,
                        self.evaluate(gprm.mean()))

    gprm_no_predictive_noise = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        predictive_noise_variance=0.,
        mean_fn=mean_fn,
        jitter=jitter)

    self.assertAllClose(expected_predictive_covariance_no_noise,
                        self.evaluate(gprm_no_predictive_noise.covariance()))
    self.assertAllClose(np.diag(expected_predictive_covariance_no_noise),
                        self.evaluate(gprm_no_predictive_noise.variance()))
    self.assertAllClose(expected_mean,
                        self.evaluate(gprm_no_predictive_noise.mean()))

  def testEmptyDataMatchesGPPrior(self):
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    jitter = np.float64(1e-4)
    index_points = np.random.uniform(-1., 1., (10, 1)).astype(np.float64)

    # k_xx - k_xn @ (k_nn + sigma^2) @ k_nx + sigma^2
    mean_fn = lambda x: x[:, 0]**2

    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)
    gp = tfd.GaussianProcess(
        kernel,
        index_points,
        mean_fn=mean_fn,
        jitter=jitter)

    gprm_nones = tfd.GaussianProcessRegressionModel(
        kernel,
        index_points,
        mean_fn=mean_fn,
        jitter=jitter)

    gprm_zero_shapes = tfd.GaussianProcessRegressionModel(
        kernel,
        index_points,
        observation_index_points=tf.ones([5, 0], tf.float64),
        observations=tf.ones([5, 0], tf.float64),
        mean_fn=mean_fn,
        jitter=jitter)

    for gprm in [gprm_nones, gprm_zero_shapes]:
      self.assertAllClose(self.evaluate(gp.mean()), self.evaluate(gprm.mean()))
      self.assertAllClose(self.evaluate(gp.covariance()),
                          self.evaluate(gprm.covariance()))
      self.assertAllClose(self.evaluate(gp.variance()),
                          self.evaluate(gprm.variance()))

      observations = np.random.uniform(-1., 1., 10).astype(np.float64)
      self.assertAllClose(self.evaluate(gp.log_prob(observations)),
                          self.evaluate(gprm.log_prob(observations)))

  def testErrorCases(self):
    kernel = psd_kernels.ExponentiatedQuadratic()
    index_points = np.random.uniform(-1., 1., (10, 1)).astype(np.float64)
    observation_index_points = (
        np.random.uniform(-1., 1., (5, 1)).astype(np.float64))
    observations = np.random.uniform(-1., 1., 3).astype(np.float64)

    # Both or neither of `observation_index_points` and `observations` must be
    # specified.
    with self.assertRaises(ValueError):
      tfd.GaussianProcessRegressionModel(
          kernel,
          index_points,
          observation_index_points=None,
          observations=observations)
    with self.assertRaises(ValueError):
      tfd.GaussianProcessRegressionModel(
          kernel,
          index_points,
          observation_index_points,
          observations=None)

    # If specified, mean_fn must be a callable.
    with self.assertRaises(ValueError):
      tfd.GaussianProcessRegressionModel(
          kernel,
          index_points,
          mean_fn=0.)

    # Observation index point and observation counts must be broadcastable.
    # Errors based on conditions of dynamic shape in graph mode cannot be
    # caught, so we only check this error case in static shape or eager mode.
    if self.is_static or tf.executing_eagerly():
      with self.assertRaises(ValueError):
        tfd.GaussianProcessRegressionModel(
            kernel,
            index_points,
            observation_index_points=np.ones([2, 2, 2]),
            observations=np.ones([5, 5]))

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
    if not self.is_static:
      index_points_1 = tf.compat.v1.placeholder_with_default(
          index_points_1, shape=None)
      index_points_2 = tf.compat.v1.placeholder_with_default(
          index_points_2, shape=None)
      observation_index_points_1 = tf.compat.v1.placeholder_with_default(
          observation_index_points_1, shape=None)
      observation_index_points_2 = tf.compat.v1.placeholder_with_default(
          observation_index_points_2, shape=None)
      observations_1 = tf.compat.v1.placeholder_with_default(
          observations_1, shape=None)
      observations_2 = tf.compat.v1.placeholder_with_default(
          observations_2, shape=None)

    mean_fn = lambda x: np.array([0.], np.float32)
    kernel_1 = psd_kernels.ExponentiatedQuadratic()
    kernel_2 = psd_kernels.ExpSinSquared()

    gprm1 = tfd.GaussianProcessRegressionModel(
        kernel=kernel_1,
        index_points=index_points_1,
        observation_index_points=observation_index_points_1,
        observations=observations_1,
        mean_fn=mean_fn,
        jitter=1e-5)
    gprm2 = gprm1.copy(
        kernel=kernel_2,
        index_points=index_points_2,
        observation_index_points=observation_index_points_2,
        observations=observations_2)

    event_shape_1 = [5]
    event_shape_2 = [10]

    self.assertEqual(gprm1.mean_fn, gprm2.mean_fn)
    self.assertIsInstance(gprm1.kernel, psd_kernels.ExponentiatedQuadratic)
    self.assertIsInstance(gprm2.kernel, psd_kernels.ExpSinSquared)

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(gprm1.batch_shape, gprm2.batch_shape)
      self.assertAllEqual(gprm1.event_shape, event_shape_1)
      self.assertAllEqual(gprm2.event_shape, event_shape_2)
      self.assertAllEqual(gprm1.index_points, index_points_1)
      self.assertAllEqual(gprm2.index_points, index_points_2)
      self.assertAllEqual(
          tf.get_static_value(gprm1.jitter), tf.get_static_value(gprm2.jitter))
    else:
      self.assertAllEqual(self.evaluate(gprm1.batch_shape_tensor()),
                          self.evaluate(gprm2.batch_shape_tensor()))
      self.assertAllEqual(self.evaluate(gprm1.event_shape_tensor()),
                          event_shape_1)
      self.assertAllEqual(self.evaluate(gprm2.event_shape_tensor()),
                          event_shape_2)
      self.assertEqual(self.evaluate(gprm1.jitter), self.evaluate(gprm2.jitter))
      self.assertAllEqual(self.evaluate(gprm1.index_points), index_points_1)
      self.assertAllEqual(self.evaluate(gprm2.index_points), index_points_2)


@test_util.run_all_in_graph_and_eager_modes
class GaussianProcessRegressionModelStaticTest(
    _GaussianProcessRegressionModelTest, tf.test.TestCase):
  is_static = True


@test_util.run_all_in_graph_and_eager_modes
class GaussianProcessRegressionModelDynamicTest(
    _GaussianProcessRegressionModelTest, tf.test.TestCase):
  is_static = False


if __name__ == "__main__":
  tf.test.main()
