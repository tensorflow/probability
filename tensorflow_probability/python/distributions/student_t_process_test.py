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
# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import multivariate_student_t as mvst
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import student_t_process
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import psd_kernels
from tensorflow_probability.python.math.psd_kernels.internal import test_util as psd_kernel_test_util


class _StudentTProcessTest(test_util.TestCase):

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
    observation_noise_variance = np.array(
        [1e-5, 1e-6, 1e-5], np.float32).reshape([3, 1, 1, 1])
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
    tp = student_t_process.StudentTProcess(
        df,
        kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    batch_shape = [3, 2, 4, 6]
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

    tp = student_t_process.StudentTProcess(
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
    tp = student_t_process.StudentTProcess(
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

    tp1 = student_t_process.StudentTProcess(
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
    mean_fn = lambda x: x[..., 0]**2
    jitter = np.float64(1e-4)

    tp = student_t_process.StudentTProcess(
        df=np.float64(3.),
        kernel=kernel,
        mean_fn=mean_fn,
        jitter=jitter,
        validate_args=True)

    index_points = np.random.uniform(-1., 1., [5, 10, 1]).astype(np.float64)

    # Check that the internal batch_shape and event_shape methods work.
    self.assertAllEqual([5], tp._batch_shape(index_points=index_points))  # pylint: disable=protected-access
    self.assertAllEqual([10], tp._event_shape(index_points=index_points))  # pylint: disable=protected-access

    expected_mean = mean_fn(index_points)
    self.assertAllClose(expected_mean,
                        self.evaluate(tp.mean(index_points=index_points)))

    # Check that late-binding samples work.
    self.evaluate(tp.sample(
        seed=test_util.test_seed(), index_points=index_points))

    def _kernel_fn(x, y):
      return amp ** 2 * np.exp(-.5 * (np.squeeze((x - y)**2)) / (len_scale**2))

    expected_covariance = _kernel_fn(np.expand_dims(index_points, -3),
                                     np.expand_dims(index_points, -2))

    self.assertAllClose(expected_covariance,
                        self.evaluate(tp.covariance(index_points=index_points)))
    self.assertAllClose(np.diagonal(expected_covariance, axis1=-2, axis2=-1),
                        self.evaluate(tp.variance(index_points=index_points)))
    self.assertAllClose(
        np.sqrt(np.diagonal(expected_covariance, axis1=-2, axis2=-1)),
        self.evaluate(tp.stddev(index_points=index_points)))

    # Calling mean with no index_points should raise an Error
    with self.assertRaises(ValueError):
      tp.mean()

  def testMarginalHasCorrectTypes(self):
    tp = student_t_process.StudentTProcess(
        df=3., kernel=psd_kernels.ExponentiatedQuadratic(), validate_args=True)

    self.assertIsInstance(
        tp.get_marginal_distribution(
            index_points=np.ones([1, 1], dtype=np.float32)), student_t.StudentT)

    self.assertIsInstance(
        tp.get_marginal_distribution(
            index_points=np.ones([10, 1], dtype=np.float32)),
        mvst.MultivariateStudentTLinearOperator)

  @parameterized.parameters(
      {"foo_feature_shape": [5], "bar_feature_shape": [3]},
      {"foo_feature_shape": [3, 2], "bar_feature_shape": [5]},
      {"foo_feature_shape": [3, 2], "bar_feature_shape": [4, 3]},
  )
  def testStructuredIndexPoints(self, foo_feature_shape, bar_feature_shape):
    base_kernel = psd_kernels.ExponentiatedQuadratic()
    structured_kernel = psd_kernel_test_util.MultipartTestKernel(
        base_kernel,
        feature_ndims={"foo": len(foo_feature_shape),
                       "bar": len(bar_feature_shape)})

    foo_num_features = np.prod(foo_feature_shape)
    bar_num_features = np.prod(bar_feature_shape)
    batch_and_example_shape = [3, 2, 10]
    index_points = np.random.uniform(
        -1, 1, batch_and_example_shape + [foo_num_features + bar_num_features]
        ).astype(np.float32)
    split_index_points = tf.split(
        index_points, [foo_num_features, bar_num_features], axis=-1)
    structured_index_points = {
        "foo": tf.reshape(split_index_points[0],
                          batch_and_example_shape + foo_feature_shape),
        "bar": tf.reshape(split_index_points[1],
                          batch_and_example_shape + bar_feature_shape)}
    df = np.float32(3.)
    base_stp = student_t_process.StudentTProcess(
        df, kernel=base_kernel, index_points=index_points)
    structured_stp = student_t_process.StudentTProcess(
        df, kernel=structured_kernel, index_points=structured_index_points)

    self.assertAllEqual(base_stp.event_shape, structured_stp.event_shape)
    self.assertAllEqual(base_stp.event_shape_tensor(),
                        structured_stp.event_shape_tensor())
    self.assertAllEqual(base_stp.batch_shape, structured_stp.batch_shape)
    self.assertAllEqual(base_stp.batch_shape_tensor(),
                        structured_stp.batch_shape_tensor())

    s = structured_stp.sample(3, seed=test_util.test_seed())
    self.assertAllClose(base_stp.log_prob(s), structured_stp.log_prob(s))
    self.assertAllClose(base_stp.mean(), structured_stp.mean())
    self.assertAllClose(base_stp.variance(), structured_stp.variance())

    # Check that batch shapes and number of index points broadcast across
    # different parts of index_points.
    bcast_structured_index_points = {
        "foo": np.random.uniform(
            -1, 1, [2, 1] + foo_feature_shape).astype(np.float32),
        "bar": np.random.uniform(
            -1, 1, [3, 1, 10] + bar_feature_shape).astype(np.float32),
    }
    bcast_structured_stp = student_t_process.StudentTProcess(
        df, kernel=structured_kernel,
        index_points=bcast_structured_index_points)
    self.assertAllEqual(base_stp.event_shape, bcast_structured_stp.event_shape)
    self.assertAllEqual(base_stp.event_shape_tensor(),
                        bcast_structured_stp.event_shape_tensor())
    self.assertAllEqual(base_stp.batch_shape, bcast_structured_stp.batch_shape)
    self.assertAllEqual(base_stp.batch_shape_tensor(),
                        bcast_structured_stp.batch_shape_tensor())

    index_points_bad_num_examples = {
        "foo": np.random.uniform(
            -1, 1, [5] + foo_feature_shape).astype(np.float32),
        "bar": np.random.uniform(
            -1, 1, [2] + bar_feature_shape).astype(np.float32),
    }

    # Non-broadcasting numbers of index points should raise.
    structured_stp_bad_num_examples = student_t_process.StudentTProcess(
        df, kernel=structured_kernel,
        index_points=index_points_bad_num_examples)
    with self.assertRaisesRegex(ValueError, "the same or broadcastable"):
      _ = structured_stp_bad_num_examples.event_shape

    # Iterable index points should be interpreted as single Tensors if the
    # kernel is not structured.
    index_points_list = tf.unstack(index_points)
    stp_with_list = student_t_process.StudentTProcess(
        df, kernel=base_kernel, index_points=index_points_list)
    self.assertAllEqual(base_stp.event_shape_tensor(),
                        stp_with_list.event_shape_tensor())
    self.assertAllEqual(base_stp.batch_shape_tensor(),
                        stp_with_list.batch_shape_tensor())
    self.assertAllClose(base_stp.log_prob(s), stp_with_list.log_prob(s))

  def testAlwaysYieldMultivariateStudentT(self):
    df = np.float32(3.)
    stp = student_t_process.StudentTProcess(
        df=df,
        kernel=psd_kernels.ExponentiatedQuadratic(),
        index_points=tf.ones([5, 1, 2]),
        always_yield_multivariate_student_t=False,
    )
    self.assertAllEqual([5], self.evaluate(stp.batch_shape_tensor()))
    self.assertAllEqual([], self.evaluate(stp.event_shape_tensor()))

    stp = student_t_process.StudentTProcess(
        df=df,
        kernel=psd_kernels.ExponentiatedQuadratic(),
        index_points=tf.ones([5, 1, 2]),
        always_yield_multivariate_student_t=True,
    )
    self.assertAllEqual([5], self.evaluate(stp.batch_shape_tensor()))
    self.assertAllEqual([1], self.evaluate(stp.event_shape_tensor()))

  def testLogProbMatchesMVT(self):
    df = tf.convert_to_tensor(3.)
    index_points = tf.convert_to_tensor(
        [[-1.0, 0.0], [-0.5, -0.5], [1.5, 0.0], [1.6, 1.5]])
    amplitude = tf.convert_to_tensor(1.1)
    length_scale = tf.convert_to_tensor(0.9)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    stp = student_t_process.StudentTProcess(
        df=df,
        kernel=kernel,
        index_points=index_points,
        mean_fn=lambda x: tf.reduce_mean(x, axis=-1),
        observation_noise_variance=.05,
        jitter=0.0)
    x = stp.sample(5, seed=test_util.test_seed())
    scale = kernel.matrix(index_points, index_points)
    scale = tf.linalg.set_diag(
        scale, tf.linalg.diag_part(scale) + .05)
    scale = (df - 2.) / df * scale
    scale = tf.linalg.cholesky(scale)
    mvt = mvst.MultivariateStudentTLinearOperator(
        loc=tf.reduce_mean(index_points, axis=-1),
        df=df,
        scale=tf.linalg.LinearOperatorLowerTriangular(scale))
    actual_log_prob, expected_log_prob = self.evaluate([
        stp.log_prob(x), mvt.log_prob(x)])
    self.assertAllClose(actual_log_prob, expected_log_prob)

  def testLogProbWithIsMissing(self):
    df = tf.convert_to_tensor(3.)
    index_points = tf.Variable(
        [[-1.0, 0.0], [-0.5, -0.5], [1.5, 0.0], [1.6, 1.5]],
        shape=None if self.is_static else tf.TensorShape(None))
    self.evaluate(index_points.initializer)
    amplitude = tf.convert_to_tensor(1.1)
    length_scale = tf.convert_to_tensor(0.9)

    stp = student_t_process.StudentTProcess(
        df=df,
        kernel=psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
        index_points=index_points,
        mean_fn=lambda x: tf.reduce_mean(x, axis=-1),
        observation_noise_variance=.05,
        jitter=0.0)

    x = stp.sample(5, seed=test_util.test_seed())

    is_missing = np.array([
        [False, True, False, False],
        [False, False, False, False],
        [True, False, True, True],
        [True, False, False, True],
        [False, False, True, True],
    ])

    lp = stp.log_prob(tf.where(is_missing, np.nan, x), is_missing=is_missing)

    # For each batch member, check that the log_prob is the same as for a
    # StudentTProcess without the missing index points.
    for i in range(5):
      stp_i = student_t_process.StudentTProcess(
          df=df,
          kernel=psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
          index_points=tf.gather(index_points, (~is_missing[i]).nonzero()[0]),
          mean_fn=lambda x: tf.reduce_mean(x, axis=-1),
          observation_noise_variance=.05,
          jitter=0.0)
      lp_i = stp_i.log_prob(tf.gather(x[i], (~is_missing[i]).nonzero()[0]))
      # NOTE: This reshape is necessary because lp_i has shape [1] when
      # stp_i.index_points contains a single index point.
      self.assertAllClose(tf.reshape(lp_i, []), lp[i])

    # The log_prob should be zero when all points are missing out.
    self.assertAllClose(tf.zeros((3, 2)),
                        stp.log_prob(
                            tf.ones((3, 1, 4)) * np.nan,
                            is_missing=tf.constant(True, shape=(2, 4))))

  def testUnivariateLogProbWithIsMissing(self):
    index_points = tf.convert_to_tensor([[[0.0, 0.0]], [[0.5, 1.0]]])
    df = tf.convert_to_tensor(3.)
    amplitude = tf.convert_to_tensor(1.1)
    length_scale = tf.convert_to_tensor(0.9)

    stp = student_t_process.StudentTProcess(
        df=df,
        kernel=psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
        index_points=index_points,
        mean_fn=lambda x: tf.reduce_mean(x, axis=-1),
        observation_noise_variance=.05,
        jitter=0.0)

    x = stp.sample(3, seed=test_util.test_seed())
    lp = stp.log_prob(x)

    self.assertAllClose(lp, stp.log_prob(x, is_missing=[[False], [False]]))
    self.assertAllClose(tf.convert_to_tensor([np.zeros((3, 2)), lp]),
                        stp.log_prob(x, is_missing=[[[[True]]], [[[False]]]]))
    self.assertAllClose(
        tf.convert_to_tensor([[lp[0, 0], 0.0], [0.0, 0.0], [0., lp[2, 1]]]),
        stp.log_prob(
            x,
            is_missing=[[[False], [True]], [[True], [True]], [[True], [False]]])
    )

  @test_util.numpy_disable_gradient_test
  def test_gradient_non_nan(self):
    x_obs = np.linspace(1., 10., num=30).reshape([5, 2, 3]).astype(np.float32)
    y_obs = tf.reduce_sum(x_obs, axis=(-1, -2))
    x_obs = tf.concat(
        [x_obs, np.full([3, 2, 3], np.nan, dtype=np.float32)], axis=0)
    y_obs = tf.concat([y_obs, np.full([3], np.nan, dtype=np.float32)], axis=0)
    is_missing = np.array([False] * 5 + [True] * 3)

    def loss(x, y, l, o):
      kernel = psd_kernels.ExponentiatedQuadratic(
          amplitude=1e-2, feature_ndims=2)
      kernel = psd_kernels.FeatureScaled(
          kernel, scale_diag=tf.math.sqrt(l))
      return student_t_process.StudentTProcess(
          df=4.,
          kernel=kernel,
          index_points=x,
          observation_noise_variance=o
      ).log_prob(y, is_missing=is_missing)

    observation_noise_variance = tf.constant(1e-3)
    lscales = tf.convert_to_tensor([[11.67385626, 1.21246016, 1.5215677],
                                    [1.08823962, 1.22416186, 1.16885594]])

    value, grads = gradient.value_and_gradient(
        loss, [x_obs, y_obs, lscales, observation_noise_variance])
    self.assertAllNotNan(value)
    for g in grads:
      self.assertAllNotNan(g)


@test_util.test_all_tf_execution_regimes
class StudentTProcessStaticTest(_StudentTProcessTest):
  is_static = True


@test_util.test_all_tf_execution_regimes
class StudentTProcessDynamicTest(_StudentTProcessTest):
  is_static = False


del _StudentTProcessTest


if __name__ == "__main__":
  test_util.main()
