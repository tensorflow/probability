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
"""Tests for VariationalGaussianProcess."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as psd_kernels

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def _np_kernel_matrix_fn(amp, length_scale, x, y):
  x = np.expand_dims(x, -2)[..., 0]
  y = np.expand_dims(y, -3)[..., 0]
  return amp ** 2 * np.exp(-.5 * ((x - y)**2) / (length_scale**2))


# TODO(cgs, srvasude): Figure out good tests for correctness for VGP, and add
# them here.
# Potential start is constructing kernels for which the Nystrom approximation is
# almost exact. This imples the VGP replicates the GP.
class _VariationalGaussianProcessTest(object):

  def testShapes(self):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]
    batched_index_points = np.expand_dims(np.stack([index_points]*6), -3)
    # ==> shape = [6, 1, 25, 2]

    # 9 inducing index points in R^2
    inducing_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    inducing_index_points = np.stack(np.meshgrid(inducing_index_points,
                                                 inducing_index_points),
                                     axis=-1)
    inducing_index_points = np.reshape(inducing_index_points, [-1, 2])
    # ==> shape = [9, 2]

    variational_inducing_observations_loc = np.zeros([3, 9], dtype=np.float64)
    variational_inducing_observations_scale = np.eye(9, dtype=np.float64)

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape([1, 4, 1, 1])

    jitter = np.float64(1e-6)
    observation_noise_variance = np.float64(1e-2)

    if not self.is_static:
      amplitude = tf.compat.v1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf.compat.v1.placeholder_with_default(
          length_scale, shape=None)
      batched_index_points = tf.compat.v1.placeholder_with_default(
          batched_index_points, shape=None)

      inducing_index_points = tf.compat.v1.placeholder_with_default(
          inducing_index_points, shape=None)
      variational_inducing_observations_loc = tf.compat.v1.placeholder_with_default(
          variational_inducing_observations_loc, shape=None)
      variational_inducing_observations_scale = tf.compat.v1.placeholder_with_default(
          variational_inducing_observations_scale, shape=None)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    vgp = tfd.VariationalGaussianProcess(
        kernel=kernel,
        index_points=batched_index_points,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=(
            variational_inducing_observations_loc),
        variational_inducing_observations_scale=(
            variational_inducing_observations_scale),
        observation_noise_variance=observation_noise_variance,
        jitter=jitter)

    batch_shape = [2, 4, 6, 3]
    event_shape = [25]
    sample_shape = [9, 3]

    samples = vgp.sample(sample_shape)

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(vgp.batch_shape_tensor(), batch_shape)
      self.assertAllEqual(vgp.event_shape_tensor(), event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(vgp.batch_shape, batch_shape)
      self.assertAllEqual(vgp.event_shape, event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
    else:
      self.assertAllEqual(self.evaluate(vgp.batch_shape_tensor()), batch_shape)
      self.assertAllEqual(self.evaluate(vgp.event_shape_tensor()), event_shape)
      self.assertAllEqual(self.evaluate(samples).shape,
                          sample_shape + batch_shape + event_shape)
      self.assertIsNone(samples.shape.ndims)
      self.assertIsNone(vgp.batch_shape.ndims)
      self.assertEqual(vgp.event_shape.ndims, 1)
      self.assertIsNone(tf.compat.dimension_value(vgp.event_shape.dims[0]))

  def testOptimalVariationalShapes(self):
    # 5x5 grid of observation index points in R^2 and flatten to 25x2
    observation_index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    observation_index_points = np.stack(
        np.meshgrid(
            observation_index_points, observation_index_points), axis=-1)
    observation_index_points = np.reshape(
        observation_index_points, [-1, 2])
    # ==> shape = [25, 2]
    observation_index_points = np.expand_dims(
        np.stack([observation_index_points]*6), -3)
    # ==> shape = [6, 1, 25, 2]
    observations = np.sin(observation_index_points[..., 0])
    # ==> shape = [6, 1, 25]

    # 9 inducing index points in R^2
    inducing_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    inducing_index_points = np.stack(np.meshgrid(inducing_index_points,
                                                 inducing_index_points),
                                     axis=-1)
    inducing_index_points = np.reshape(inducing_index_points, [-1, 2])
    # ==> shape = [9, 2]

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape([1, 4, 1, 1])

    jitter = np.float64(1e-6)
    observation_noise_variance = np.float64(1e-2)

    if not self.is_static:
      amplitude = tf.compat.v1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf.compat.v1.placeholder_with_default(
          length_scale, shape=None)
      observation_index_points = tf.compat.v1.placeholder_with_default(
          observation_index_points, shape=None)

      inducing_index_points = tf.compat.v1.placeholder_with_default(
          inducing_index_points, shape=None)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    loc, scale = tfd.VariationalGaussianProcess.optimal_variational_posterior(
        kernel=kernel,
        inducing_index_points=inducing_index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter,
    )
    # We should expect that loc has shape [2, 4, 6, 1, 9]. This is because:
    # * [2, 4] comes from the batch shape of the kernel.
    # * [6, 1] comes from the batch shape of the observations / observation
    # index points.
    # * [9] comes from the number of inducing points.
    # Similar reasoning applies to scale.
    self.assertAllEqual([2, 4, 6, 1, 9], tf.shape(input=loc))
    self.assertAllEqual([2, 4, 6, 1, 9, 9], tf.shape(input=scale))

  def testVariationalLossShapes(self):
    # 2x2 grid of index points in R^2 and flatten to 4x2
    index_points = np.linspace(-4., 4., 2, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [4, 2]
    batched_index_points = np.expand_dims(np.stack([index_points]*6), -3)
    # ==> shape = [6, 1, 4, 2]

    # 3x3 grid of index points in R^2 and flatten to 9x2
    observation_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    observation_index_points = np.stack(
        np.meshgrid(
            observation_index_points, observation_index_points), axis=-1)
    observation_index_points = np.reshape(
        observation_index_points, [-1, 2])
    # ==> shape = [9, 2]
    observation_index_points = np.expand_dims(
        np.stack([observation_index_points]*6), -3)
    # ==> shape = [6, 1, 9, 2]
    observations = np.sin(observation_index_points[..., 0])
    # ==> shape = [6, 1, 9]

    # 9 inducing index points in R^2
    inducing_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    inducing_index_points = np.stack(np.meshgrid(inducing_index_points,
                                                 inducing_index_points),
                                     axis=-1)
    inducing_index_points = np.reshape(inducing_index_points, [-1, 2])
    # ==> shape = [9, 2]

    variational_inducing_observations_loc = np.zeros([3, 9], dtype=np.float64)
    variational_inducing_observations_scale = np.eye(9, dtype=np.float64)

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape([1, 4, 1, 1])

    jitter = np.float64(1e-6)
    observation_noise_variance = np.float64(1e-2)

    if not self.is_static:
      amplitude = tf.compat.v1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf.compat.v1.placeholder_with_default(
          length_scale, shape=None)
      batched_index_points = tf.compat.v1.placeholder_with_default(
          batched_index_points, shape=None)

      observations = tf.compat.v1.placeholder_with_default(
          observations, shape=None)
      observation_index_points = tf.compat.v1.placeholder_with_default(
          observation_index_points, shape=None)
      inducing_index_points = tf.compat.v1.placeholder_with_default(
          inducing_index_points, shape=None)
      variational_inducing_observations_loc = tf.compat.v1.placeholder_with_default(
          variational_inducing_observations_loc, shape=None)
      variational_inducing_observations_scale = tf.compat.v1.placeholder_with_default(
          variational_inducing_observations_scale, shape=None)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    vgp = tfd.VariationalGaussianProcess(
        kernel=kernel,
        index_points=batched_index_points,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=(
            variational_inducing_observations_loc),
        variational_inducing_observations_scale=(
            variational_inducing_observations_scale),
        observation_noise_variance=observation_noise_variance,
        jitter=jitter)

    loss = vgp.variational_loss(
        observations=observations,
        observation_index_points=observation_index_points)
    # Expect a scalar loss.
    self.assertAllClose([], tf.shape(input=loss))


@test_util.run_all_in_graph_and_eager_modes
class VariationalGaussianProcessStaticTest(
    _VariationalGaussianProcessTest, tf.test.TestCase):
  is_static = True


@test_util.run_all_in_graph_and_eager_modes
class VariationalGaussianProcessDynamicTest(
    _VariationalGaussianProcessTest, tf.test.TestCase):
  is_static = False


if __name__ == "__main__":
  tf.test.main()
