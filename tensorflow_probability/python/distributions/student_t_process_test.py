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

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import psd_kernels


class _StudentTProcessTest(object):

  def testShapes(self):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float32)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]

    # Kernel with batch_shape [2, 4, 1]
    df = np.array(
        [[3., 4., 5., 4.], [7.5, 8, 5., 5.]],
        dtype=np.float32).reshape([2, 4, 1])
    amplitude = np.array([1., 2.], np.float32).reshape([2, 1, 1])
    length_scale = np.array([1., 2., 3., 4.], np.float32).reshape([1, 4, 1])
    batched_index_points = np.stack([index_points]*6)
    # ==> shape = [6, 25, 2]
    if not self.is_static:
      df = tf1.placeholder_with_default(df, shape=None)
      amplitude = tf1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf1.placeholder_with_default(length_scale, shape=None)
      batched_index_points = tf1.placeholder_with_default(
          batched_index_points, shape=None)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
    tp = tfd.StudentTProcess(
        df, kernel, batched_index_points, jitter=1e-5, validate_args=True)

    batch_shape = [2, 4, 6]
    event_shape = [25]
    sample_shape = [5, 3]

    samples = tp.sample(sample_shape, seed=test_util.test_seed())

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(tp.batch_shape_tensor(), batch_shape)
      self.assertAllEqual(tp.event_shape_tensor(), event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(tp.batch_shape, batch_shape)
      self.assertAllEqual(tp.event_shape, event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(tp.mean().shape, batch_shape + event_shape)
      self.assertAllEqual(tp.variance().shape, batch_shape + event_shape)
    else:
      self.assertAllEqual(self.evaluate(tp.batch_shape_tensor()), batch_shape)
      self.assertAllEqual(self.evaluate(tp.event_shape_tensor()), event_shape)
      self.assertAllEqual(
          self.evaluate(samples).shape,
          sample_shape + batch_shape + event_shape)
      self.assertIsNone(tensorshape_util.rank(samples.shape))
      self.assertIsNone(tensorshape_util.rank(tp.batch_shape))
      self.assertEqual(tensorshape_util.rank(tp.event_shape), 1)
      self.assertIsNone(
          tf.compat.dimension_value(tensorshape_util.dims(tp.event_shape)[0]))
      self.assertAllEqual(
          self.evaluate(tf.shape(tp.mean())), batch_shape + event_shape)
      self.assertAllEqual(self.evaluate(
          tf.shape(tp.variance())), batch_shape + event_shape)

  def testVarianceAndCovarianceMatrix(self):
    df = np.float64(4.)
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    jitter = np.float64(1e-4)

    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)

    index_points = np.expand_dims(np.random.uniform(-1., 1., 10), -1)

    tp = tfd.StudentTProcess(
        df=df,
        kernel=kernel,
        index_points=index_points,
        jitter=jitter,
        validate_args=True)

    def _kernel_fn(x, y):
      return amp ** 2 * np.exp(-.5 * (np.squeeze((x - y)**2)) / (len_scale**2))

    expected_covariance = (
        _kernel_fn(np.expand_dims(index_points, 0),
                   np.expand_dims(index_points, 1)))

    self.assertAllClose(expected_covariance,
                        self.evaluate(tp.covariance()))
    self.assertAllClose(np.diag(expected_covariance),
                        self.evaluate(tp.variance()))

  def testMean(self):
    mean_fn = lambda x: x[:, 0]**2
    kernel = psd_kernels.ExponentiatedQuadratic()
    index_points = np.expand_dims(np.random.uniform(-1., 1., 10), -1)
    tp = tfd.StudentTProcess(
        df=3.,
        kernel=kernel,
        index_points=index_points,
        mean_fn=mean_fn,
        validate_args=True)
    expected_mean = mean_fn(index_points)
    self.assertAllClose(expected_mean,
                        self.evaluate(tp.mean()))

  def testCopy(self):
    # 5 random index points in R^2
    index_points_1 = np.random.uniform(-4., 4., (5, 2)).astype(np.float32)
    # 10 random index points in R^2
    index_points_2 = np.random.uniform(-4., 4., (10, 2)).astype(np.float32)

    # ==> shape = [6, 25, 2]
    if not self.is_static:
      index_points_1 = tf1.placeholder_with_default(index_points_1, shape=None)
      index_points_2 = tf1.placeholder_with_default(index_points_2, shape=None)

    mean_fn = lambda x: np.array([0.], np.float32)
    kernel_1 = psd_kernels.ExponentiatedQuadratic()
    kernel_2 = psd_kernels.ExpSinSquared()

    tp1 = tfd.StudentTProcess(
        df=3.,
        kernel=kernel_1,
        index_points=index_points_1,
        mean_fn=mean_fn,
        jitter=1e-5,
        validate_args=True)
    tp2 = tp1.copy(df=4., index_points=index_points_2, kernel=kernel_2)

    event_shape_1 = [5]
    event_shape_2 = [10]

    self.assertEqual(tp1.mean_fn, tp2.mean_fn)
    self.assertIsInstance(tp1.kernel, psd_kernels.ExponentiatedQuadratic)
    self.assertIsInstance(tp2.kernel, psd_kernels.ExpSinSquared)

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(tp1.batch_shape, tp2.batch_shape)
      self.assertAllEqual(tp1.event_shape, event_shape_1)
      self.assertAllEqual(tp2.event_shape, event_shape_2)
      self.assertEqual(self.evaluate(tp1.df), 3.)
      self.assertEqual(self.evaluate(tp2.df), 4.)
      self.assertAllEqual(tp2.index_points, index_points_2)
      self.assertAllEqual(tp1.index_points, index_points_1)
      self.assertAllEqual(tp2.index_points, index_points_2)
      self.assertAllEqual(
          tf.get_static_value(tp1.jitter), tf.get_static_value(tp2.jitter))
    else:
      self.assertAllEqual(
          self.evaluate(tp1.batch_shape_tensor()),
          self.evaluate(tp2.batch_shape_tensor()))
      self.assertAllEqual(
          self.evaluate(tp1.event_shape_tensor()), event_shape_1)
      self.assertAllEqual(
          self.evaluate(tp2.event_shape_tensor()), event_shape_2)
      self.assertEqual(self.evaluate(tp1.jitter), self.evaluate(tp2.jitter))
      self.assertEqual(self.evaluate(tp1.df), 3.)
      self.assertEqual(self.evaluate(tp2.df), 4.)
      self.assertAllEqual(self.evaluate(tp1.index_points), index_points_1)
      self.assertAllEqual(self.evaluate(tp2.index_points), index_points_2)

  def testLateBindingIndexPoints(self):
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)
    mean_fn = lambda x: x[:, 0]**2
    jitter = np.float64(1e-4)

    tp = tfd.StudentTProcess(
        df=np.float64(3.),
        kernel=kernel,
        mean_fn=mean_fn,
        jitter=jitter,
        validate_args=True)

    index_points = np.random.uniform(-1., 1., [10, 1]).astype(np.float64)

    expected_mean = mean_fn(index_points)
    self.assertAllClose(expected_mean,
                        self.evaluate(tp.mean(index_points=index_points)))

    def _kernel_fn(x, y):
      return amp ** 2 * np.exp(-.5 * (np.squeeze((x - y)**2)) / (len_scale**2))

    expected_covariance = _kernel_fn(np.expand_dims(index_points, -3),
                                     np.expand_dims(index_points, -2))

    self.assertAllClose(expected_covariance,
                        self.evaluate(tp.covariance(index_points=index_points)))
    self.assertAllClose(np.diag(expected_covariance),
                        self.evaluate(tp.variance(index_points=index_points)))
    self.assertAllClose(np.sqrt(np.diag(expected_covariance)),
                        self.evaluate(tp.stddev(index_points=index_points)))

    # Calling mean with no index_points should raise an Error
    with self.assertRaises(ValueError):
      tp.mean()

  def testMarginalHasCorrectTypes(self):
    tp = tfd.StudentTProcess(
        df=3., kernel=psd_kernels.ExponentiatedQuadratic(), validate_args=True)

    self.assertIsInstance(
        tp.get_marginal_distribution(
            index_points=np.ones([1, 1], dtype=np.float32)),
        tfd.StudentT)

    self.assertIsInstance(
        tp.get_marginal_distribution(
            index_points=np.ones([10, 1], dtype=np.float32)),
        tfd.MultivariateStudentTLinearOperator)


@test_util.test_all_tf_execution_regimes
class StudentTProcessStaticTest(_StudentTProcessTest, test_util.TestCase):
  is_static = True


@test_util.test_all_tf_execution_regimes
class StudentTProcessDynamicTest(_StudentTProcessTest, test_util.TestCase):
  is_static = False


if __name__ == "__main__":
  tf.test.main()
