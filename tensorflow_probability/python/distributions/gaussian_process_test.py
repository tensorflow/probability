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
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import psd_kernels


class _GaussianProcessTest(object):

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
    if not self.is_static:
      amplitude = tf1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf1.placeholder_with_default(length_scale, shape=None)
      batched_index_points = tf1.placeholder_with_default(
          batched_index_points, shape=None)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
    gp = tfd.GaussianProcess(
        kernel,
        batched_index_points,
        observation_noise_variance=observation_noise_variance,
        jitter=1e-5,
        validate_args=True)

    batch_shape = [2, 4, 3, 6]
    event_shape = [25]
    sample_shape = [5, 3]

    samples = gp.sample(sample_shape, seed=test_util.test_seed())

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(gp.batch_shape_tensor(), batch_shape)
      self.assertAllEqual(gp.event_shape_tensor(), event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(gp.batch_shape, batch_shape)
      self.assertAllEqual(gp.event_shape, event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(gp.mean().shape, batch_shape + event_shape)
      self.assertAllEqual(gp.variance().shape, batch_shape + event_shape)
    else:
      self.assertAllEqual(self.evaluate(gp.batch_shape_tensor()), batch_shape)
      self.assertAllEqual(self.evaluate(gp.event_shape_tensor()), event_shape)
      self.assertAllEqual(
          self.evaluate(samples).shape,
          sample_shape + batch_shape + event_shape)
      self.assertIsNone(tensorshape_util.rank(samples.shape))
      self.assertIsNone(tensorshape_util.rank(gp.batch_shape))
      self.assertEqual(tensorshape_util.rank(gp.event_shape), 1)
      self.assertIsNone(
          tf.compat.dimension_value(tensorshape_util.dims(gp.event_shape)[0]))
      self.assertAllEqual(
          self.evaluate(tf.shape(gp.mean())), batch_shape + event_shape)
      self.assertAllEqual(self.evaluate(
          tf.shape(gp.variance())), batch_shape + event_shape)

  def testVarianceAndCovarianceMatrix(self):
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    jitter = np.float64(1e-4)
    observation_noise_variance = np.float64(3e-3)

    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)

    index_points = np.expand_dims(np.random.uniform(-1., 1., 10), -1)

    gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter,
        validate_args=True)

    def _kernel_fn(x, y):
      return amp ** 2 * np.exp(-.5 * (np.squeeze((x - y)**2)) / (len_scale**2))

    expected_covariance = (
        _kernel_fn(np.expand_dims(index_points, 0),
                   np.expand_dims(index_points, 1)) +
        observation_noise_variance * np.eye(10))

    self.assertAllClose(expected_covariance,
                        self.evaluate(gp.covariance()))
    self.assertAllClose(np.diag(expected_covariance),
                        self.evaluate(gp.variance()))

  def testMean(self):
    mean_fn = lambda x: x[:, 0]**2
    kernel = psd_kernels.ExponentiatedQuadratic()
    index_points = np.expand_dims(np.random.uniform(-1., 1., 10), -1)
    gp = tfd.GaussianProcess(
        kernel, index_points, mean_fn=mean_fn, validate_args=True)
    expected_mean = mean_fn(index_points)
    self.assertAllClose(expected_mean,
                        self.evaluate(gp.mean()))

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

    gp1 = tfd.GaussianProcess(
        kernel_1, index_points_1, mean_fn, jitter=1e-5, validate_args=True)
    gp2 = gp1.copy(index_points=index_points_2,
                   kernel=kernel_2)

    event_shape_1 = [5]
    event_shape_2 = [10]

    self.assertEqual(gp1.mean_fn, gp2.mean_fn)
    self.assertIsInstance(gp1.kernel, psd_kernels.ExponentiatedQuadratic)
    self.assertIsInstance(gp2.kernel, psd_kernels.ExpSinSquared)

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(gp1.batch_shape, gp2.batch_shape)
      self.assertAllEqual(gp1.event_shape, event_shape_1)
      self.assertAllEqual(gp2.event_shape, event_shape_2)
      self.assertAllEqual(gp1.index_points, index_points_1)
      self.assertAllEqual(gp2.index_points, index_points_2)
      self.assertAllEqual(
          tf.get_static_value(gp1.jitter), tf.get_static_value(gp2.jitter))
    else:
      self.assertAllEqual(
          self.evaluate(gp1.batch_shape_tensor()),
          self.evaluate(gp2.batch_shape_tensor()))
      self.assertAllEqual(
          self.evaluate(gp1.event_shape_tensor()), event_shape_1)
      self.assertAllEqual(
          self.evaluate(gp2.event_shape_tensor()), event_shape_2)
      self.assertEqual(self.evaluate(gp1.jitter), self.evaluate(gp2.jitter))
      self.assertAllEqual(self.evaluate(gp1.index_points), index_points_1)
      self.assertAllEqual(self.evaluate(gp2.index_points), index_points_2)

  def testLateBindingIndexPoints(self):
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)
    mean_fn = lambda x: x[:, 0]**2
    jitter = np.float64(1e-4)
    observation_noise_variance = np.float64(3e-3)

    gp = tfd.GaussianProcess(
        kernel=kernel,
        mean_fn=mean_fn,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter,
        validate_args=True)

    index_points = np.random.uniform(-1., 1., [10, 1])

    expected_mean = mean_fn(index_points)
    self.assertAllClose(expected_mean,
                        self.evaluate(gp.mean(index_points=index_points)))

    def _kernel_fn(x, y):
      return amp ** 2 * np.exp(-.5 * (np.squeeze((x - y)**2)) / (len_scale**2))

    expected_covariance = (
        _kernel_fn(np.expand_dims(index_points, -3),
                   np.expand_dims(index_points, -2)) +
        observation_noise_variance * np.eye(10))

    self.assertAllClose(expected_covariance,
                        self.evaluate(gp.covariance(index_points=index_points)))
    self.assertAllClose(np.diag(expected_covariance),
                        self.evaluate(gp.variance(index_points=index_points)))
    self.assertAllClose(np.sqrt(np.diag(expected_covariance)),
                        self.evaluate(gp.stddev(index_points=index_points)))

    # Calling mean with no index_points should raise an Error
    with self.assertRaises(ValueError):
      gp.mean()

  def testMarginalHasCorrectTypes(self):
    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(), validate_args=True)

    self.assertIsInstance(
        gp.get_marginal_distribution(
            index_points=np.ones([1, 1], dtype=np.float32)),
        tfd.Normal)

    self.assertIsInstance(
        gp.get_marginal_distribution(
            index_points=np.ones([10, 1], dtype=np.float32)),
        tfd.MultivariateNormalLinearOperator)

  def testOneOfCholeskyAndMarginalFn(self):
    with self.assertRaises(ValueError):
      index_points = np.array([3., 4., 5.])[..., np.newaxis]
      tfd.GaussianProcess(
          kernel=psd_kernels.ExponentiatedQuadratic(),
          index_points=index_points,
          marginal_fn=lambda x: x,
          cholesky_fn=lambda x: x,
          validate_args=True)

  def testCustomCholeskyFn(self):
    def test_cholesky(x):
      return tf.linalg.cholesky(tf.linalg.set_diag(
          x, tf.linalg.diag_part(x) + 3.))

    # Make sure the points are far away so that this is roughly diagonal.
    index_points = np.array([-100., -50., 50., 100])[..., np.newaxis]

    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(),
        index_points=index_points,
        cholesky_fn=test_cholesky,
        validate_args=True)

    # Roughly, the kernel matrix will look like the identity matrix.
    # When we add 3 to the diagonal, this leads to 2's on the diagonal
    # for the cholesky factor.
    self.assertAllClose(
        2 * np.ones([4], dtype=np.float64),
        gp.get_marginal_distribution().stddev())

  def testCustomMarginalFn(self):
    def test_marginal_fn(
        loc,
        covariance,
        validate_args=False,
        allow_nan_stats=False,
        name="custom_marginal"):
      return tfd.MultivariateNormalDiag(
          loc=loc,
          scale_diag=tf.math.sqrt(tf.linalg.diag_part(covariance)),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)

    index_points = np.expand_dims(np.random.uniform(-1., 1., 10), -1)

    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(),
        index_points=index_points,
        marginal_fn=test_marginal_fn,
        validate_args=True)

    self.assertAllClose(
        np.eye(10),
        gp.get_marginal_distribution().covariance())

  def testGPPosteriorPredictive(self):
    amplitude = np.float64(.5)
    length_scale = np.float64(2.)
    jitter = np.float64(1e-4)
    observation_noise_variance = np.float64(3e-3)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    index_points = np.random.uniform(-1., 1., 10)[..., np.newaxis]

    gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter,
        validate_args=True)

    predictive_index_points = np.random.uniform(1., 2., 10)[..., np.newaxis]
    observations = np.linspace(1., 10., 10)

    expected_gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        observation_index_points=index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter,
        index_points=predictive_index_points,
        validate_args=True)

    actual_gprm = gp.posterior_predictive(
        predictive_index_points=predictive_index_points,
        observations=observations)

    samples = self.evaluate(actual_gprm.sample(10, seed=test_util.test_seed()))

    self.assertAllClose(
        self.evaluate(expected_gprm.mean()),
        self.evaluate(actual_gprm.mean()))

    self.assertAllClose(
        self.evaluate(expected_gprm.covariance()),
        self.evaluate(actual_gprm.covariance()))

    self.assertAllClose(
        self.evaluate(expected_gprm.log_prob(samples)),
        self.evaluate(actual_gprm.log_prob(samples)))

  def testLogProbWithIsMissing(self):
    index_points = tf.Variable(
        [[-1.0, 0.0], [-0.5, -0.5], [1.5, 0.0], [1.6, 1.5]],
        shape=None if self.is_static else tf.TensorShape(None))
    self.evaluate(index_points.initializer)
    amplitude = tf.convert_to_tensor(1.1)
    length_scale = tf.convert_to_tensor(0.9)

    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(
            amplitude, length_scale),
        index_points=index_points,
        mean_fn=lambda x: tf.reduce_mean(x, axis=-1),
        observation_noise_variance=.05,
        jitter=0.0)

    x = gp.sample(5, seed=test_util.test_seed())

    is_missing = np.array([
        [False, True, False, False],
        [False, False, False, False],
        [True, False, True, True],
        [True, False, False, True],
        [False, False, True, True],
    ])

    lp = gp.log_prob(tf.where(is_missing, np.nan, x), is_missing=is_missing)

    # For each batch member, check that the log_prob is the same as for a
    # GaussianProcess without the missing index points.
    for i in range(5):
      gp_i = tfd.GaussianProcess(
          kernel=psd_kernels.ExponentiatedQuadratic(
              amplitude, length_scale),
          index_points=tf.gather(index_points, (~is_missing[i]).nonzero()[0]),
          mean_fn=lambda x: tf.reduce_mean(x, axis=-1),
          observation_noise_variance=.05,
          jitter=0.0)
      lp_i = gp_i.log_prob(tf.gather(x[i], (~is_missing[i]).nonzero()[0]))
      # NOTE: This reshape is necessary because lp_i has shape [1] when
      # gp_i.index_points contains a single index point.
      self.assertAllClose(tf.reshape(lp_i, []), lp[i])

    # The log_prob should be zero when all points are missing out.
    self.assertAllClose(tf.zeros((3, 2)),
                        gp.log_prob(tf.ones((3, 1, 4)) * np.nan,
                                    is_missing=tf.constant(True, shape=(2, 4))))

  def testUnivariateLogProbWithIsMissing(self):
    index_points = tf.convert_to_tensor([[[0.0, 0.0]], [[0.5, 1.0]]])
    amplitude = tf.convert_to_tensor(1.1)
    length_scale = tf.convert_to_tensor(0.9)

    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(
            amplitude, length_scale),
        index_points=index_points,
        mean_fn=lambda x: tf.reduce_mean(x, axis=-1),
        observation_noise_variance=.05,
        jitter=0.0)

    x = gp.sample(3, seed=test_util.test_seed())
    lp = gp.log_prob(x)

    self.assertAllClose(lp, gp.log_prob(x, is_missing=[False, False]))
    self.assertAllClose(tf.convert_to_tensor([np.zeros((3, 2)), lp]),
                        gp.log_prob(x, is_missing=[[[True]], [[False]]]))
    self.assertAllClose(
        tf.convert_to_tensor([[lp[0, 0], 0.0], [0.0, 0.0], [0., lp[2, 1]]]),
        gp.log_prob(x, is_missing=[[False, True], [True, True], [True, False]]))

  def testAlwaysYieldMultivariateNormal(self):
    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(),
        index_points=tf.ones([5, 1, 2]),
        always_yield_multivariate_normal=False,
    )
    self.assertAllEqual([5], self.evaluate(gp.batch_shape_tensor()))
    self.assertAllEqual([], self.evaluate(gp.event_shape_tensor()))

    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(),
        index_points=tf.ones([5, 1, 2]),
        always_yield_multivariate_normal=True,
    )
    self.assertAllEqual([5], self.evaluate(gp.batch_shape_tensor()))
    self.assertAllEqual([1], self.evaluate(gp.event_shape_tensor()))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason="Numpy and JAX have no notion of CompositeTensor.")
  def testCompositeTensor(self):
    index_points = np.random.uniform(-1., 1., 10)[..., np.newaxis]
    gp = tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(),
        index_points=index_points)

    flat = tf.nest.flatten(gp, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        gp, flat, expand_composites=True)
    self.assertIsInstance(unflat, tfd.GaussianProcess)

    x = self.evaluate(gp.sample(3, seed=test_util.test_seed()))
    actual = self.evaluate(gp.log_prob(x))

    self.assertAllClose(self.evaluate(unflat.log_prob(x)), actual)

    @tf.function
    def call_log_prob(d):
      return d.log_prob(x)
    self.assertAllClose(actual, call_log_prob(gp))
    self.assertAllClose(actual, call_log_prob(unflat))


@test_util.test_all_tf_execution_regimes
class GaussianProcessStaticTest(_GaussianProcessTest, test_util.TestCase):
  is_static = True


@test_util.test_all_tf_execution_regimes
class GaussianProcessDynamicTest(_GaussianProcessTest, test_util.TestCase):
  is_static = False


if __name__ == "__main__":
  test_util.main()
