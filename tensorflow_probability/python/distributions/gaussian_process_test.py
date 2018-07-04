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

from tensorflow.python.framework import test_util


class _GaussianProcessTest(object):

  @test_util.run_in_graph_and_eager_modes()
  def testShapes(self):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 1]
    amplitude = np.array([1., 2.], np.float32).reshape([2, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float32).reshape([1, 4, 1])
    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 25, 2]
    if not self.is_static:
      amplitude = tf.placeholder_with_default(amplitude, shape=None)
      length_scale = tf.placeholder_with_default(length_scale, shape=None)
      batched_index_points = tf.placeholder_with_default(
          batched_index_points, shape=None)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
    gp = tfd.GaussianProcess(
        kernel,
        batched_index_points,
        jitter=1e-5)

    batch_shape = [2, 4, 6]
    event_shape = [25]
    sample_shape = [5, 3]

    samples = gp.sample(sample_shape)

    with self.test_session():
      if self.is_static or tf.executing_eagerly():
        self.assertAllEqual(gp.batch_shape_tensor(), batch_shape)
        self.assertAllEqual(gp.event_shape_tensor(), event_shape)
        self.assertAllEqual(samples.shape,
                            sample_shape + batch_shape + event_shape)
        self.assertAllEqual(gp.batch_shape, batch_shape)
        self.assertAllEqual(gp.event_shape, event_shape)
        self.assertAllEqual(samples.shape,
                            sample_shape + batch_shape + event_shape)
      else:
        self.assertAllEqual(gp.batch_shape_tensor().eval(), batch_shape)
        self.assertAllEqual(gp.event_shape_tensor().eval(), event_shape)
        self.assertAllEqual(samples.eval().shape,
                            sample_shape + batch_shape + event_shape)
        self.assertIsNone(samples.shape.ndims)
        self.assertIsNone(gp.batch_shape.ndims)
        self.assertEqual(gp.event_shape.ndims, 1)
        self.assertIsNone(gp.event_shape.dims[0].value)

  @test_util.run_in_graph_and_eager_modes()
  def testVarianceAndCovarianceMatrix(self):
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    jitter = np.float64(1e-4)

    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)

    index_points = np.expand_dims(np.random.uniform(-1., 1., 10), -1)

    gp = tfd.GaussianProcess(
        kernel,
        index_points,
        jitter=jitter)

    def _kernel_fn(x, y):
      return amp * np.exp(-.5 * (np.squeeze((x - y)**2)) / (len_scale**2))

    expected_covariance = (
        _kernel_fn(np.expand_dims(index_points, 0),
                   np.expand_dims(index_points, 1)) +
        jitter * np.eye(10))

    self.assertAllClose(expected_covariance,
                        self.evaluate(gp.covariance()))
    self.assertAllClose(np.diag(expected_covariance),
                        self.evaluate(gp.variance()))

  @test_util.run_in_graph_and_eager_modes()
  def testMean(self):
    mean_fn = lambda x: x[:, 0]**2
    kernel = psd_kernels.ExponentiatedQuadratic()
    index_points = np.expand_dims(np.random.uniform(-1., 1., 10), -1)
    gp = tfd.GaussianProcess(kernel, index_points, mean_fn=mean_fn)
    expected_mean = mean_fn(index_points)
    self.assertAllClose(expected_mean,
                        self.evaluate(gp.mean()))


class GaussianProcessStaticTest(_GaussianProcessTest, tf.test.TestCase):
  is_static = True


class GaussianProcessDynamicTest(_GaussianProcessTest, tf.test.TestCase):
  is_static = False


if __name__ == "__main__":
  tf.test.main()
