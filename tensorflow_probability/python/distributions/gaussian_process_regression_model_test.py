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
from unittest import mock
# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import gaussian_process
from tensorflow_probability.python.distributions import gaussian_process_regression_model as gprm
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exp_sin_squared
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels.internal import test_util as psd_kernel_test_util


def _np_kernel_matrix_fn(amp, len_scale, x, y):
  x = np.expand_dims(x, -2)[..., 0]
  y = np.expand_dims(y, -3)[..., 0]
  return amp ** 2 * np.exp(-.5 * ((x - y)**2) / (len_scale**2))


np.random.seed(42)


@test_util.test_all_tf_execution_regimes
class _GaussianProcessRegressionModelTest(test_util.TestCase):

  def testShapes(self):
    # We'll use a batch shape of [2, 3, 5, 7, 11]

    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]
    batched_index_points = np.reshape(index_points, [1, 1, 25, 2])
    batched_index_points = np.stack([batched_index_points] * 5)
    # ==> shape = [5, 1, 1, 25, 2]

    # Kernel with batch_shape [2, 3, 1, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape(
        [1, 3, 1, 1, 1])
    observation_noise_variance = np.array(
        [1e-9], np.float64).reshape([1, 1, 1, 1, 1])

    jitter = np.float64(1e-6)
    observation_index_points = (
        np.random.uniform(-1., 1., (7, 1, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (11, 7)).astype(np.float64)

    def cholesky_fn(x):
      return tf.linalg.cholesky(
          tf.linalg.set_diag(x, tf.linalg.diag_part(x) + 1.))

    if not self.is_static:
      amplitude = tf1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf1.placeholder_with_default(length_scale, shape=None)
      batched_index_points = tf1.placeholder_with_default(
          batched_index_points, shape=None)

      observation_index_points = tf1.placeholder_with_default(
          observation_index_points, shape=None)
      observations = tf1.placeholder_with_default(observations, shape=None)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)

    dist = gprm.GaussianProcessRegressionModel(
        kernel,
        batched_index_points,
        observation_index_points,
        observations,
        observation_noise_variance,
        cholesky_fn=cholesky_fn,
        jitter=jitter,
        validate_args=True)

    batch_shape = [2, 3, 5, 7, 11]
    event_shape = [25]
    sample_shape = [9, 3]

    samples = dist.sample(sample_shape, seed=test_util.test_seed())

    self.assertIs(cholesky_fn, dist.cholesky_fn)

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(dist.batch_shape_tensor(), batch_shape)
      self.assertAllEqual(dist.event_shape_tensor(), event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(dist.batch_shape, batch_shape)
      self.assertAllEqual(dist.event_shape, event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
    else:
      self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), batch_shape)
      self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), event_shape)
      self.assertAllEqual(self.evaluate(samples).shape,
                          sample_shape + batch_shape + event_shape)
      self.assertIsNone(tensorshape_util.rank(samples.shape))
      self.assertIsNone(tensorshape_util.rank(dist.batch_shape))
      self.assertEqual(tensorshape_util.rank(dist.event_shape), 1)
      self.assertIsNone(
          tf.compat.dimension_value(tensorshape_util.dims(dist.event_shape)[0]))

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
        k_xx_ - np.dot(k_xn_, np.linalg.solve(k_nn_plus_noise_, k_xn_.T)))

    expected_predictive_covariance_with_noise = (
        expected_predictive_covariance_no_noise +
        np.eye(num_test) * observation_noise_variance)

    mean_fn = lambda x: x[:, 0]**2
    prior_mean = mean_fn(observation_index_points)
    expected_mean = (
        mean_fn(index_points) +
        np.dot(k_xn_,
               np.linalg.solve(k_nn_plus_noise_, observations - prior_mean)))

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(amp, len_scale)
    dist = gprm.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        mean_fn=mean_fn,
        jitter=jitter,
        validate_args=True)

    self.assertAllClose(expected_predictive_covariance_with_noise,
                        self.evaluate(dist.covariance()))
    self.assertAllClose(
        np.diag(expected_predictive_covariance_with_noise),
        self.evaluate(dist.variance()))
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)

    x = self.evaluate(dist.sample(3, seed=test_util.test_seed()))
    actual = self.evaluate(dist.log_prob(x))
    self.assertAllClose(self.evaluate(unflat.log_prob(x)), actual)

    dist_no_predictive_noise = gprm.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        predictive_noise_variance=0.,
        mean_fn=mean_fn,
        jitter=jitter,
        validate_args=True)

    self.assertAllClose(expected_predictive_covariance_no_noise,
                        self.evaluate(dist_no_predictive_noise.covariance()))
    self.assertAllClose(
        np.diag(expected_predictive_covariance_no_noise),
        self.evaluate(dist_no_predictive_noise.variance()))
    self.assertAllClose(expected_mean,
                        self.evaluate(dist_no_predictive_noise.mean()))

  def testMeanVarianceAndCovariancePrecomputed(self):
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)

    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (1, 1, 7)).astype(np.float64)

    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    dist = gprm.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    precomputed_dist = gprm.GaussianProcessRegressionModel.precompute_regression_model(
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

  def testPrecomputedWithMasking(self):
    amplitude = np.array([1., 2.], np.float64)
    length_scale = np.array([[.1], [.2], [.3]], np.float64)
    observation_noise_variance = np.array([[1e-2], [1e-4], [1e-6]], np.float64)

    rng = test_util.test_np_rng()
    observations_is_missing = np.array([
        [False, True, False, True, False, True],
        [False, False, False, False, False, False],
        [True, True, False, False, True, True],
    ]).reshape((3, 1, 6))
    observation_index_points = np.where(
        observations_is_missing[..., np.newaxis],
        np.nan,
        rng.uniform(-1., 1., (3, 1, 6, 2)).astype(np.float64))
    observations = np.where(
        observations_is_missing,
        np.nan,
        rng.uniform(-1., 1., (3, 1, 6)).astype(np.float64))

    index_points = rng.uniform(-1., 1., (5, 2)).astype(np.float64)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    dist = gprm.GaussianProcessRegressionModel.precompute_regression_model(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observations_is_missing=observations_is_missing,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    self.assertAllNotNan(dist.mean())
    self.assertAllNotNan(dist.variance())
    self.assertAllNotNan(dist.covariance())

    # For each batch member of `gprm`, check that the distribution is the same
    # as a GaussianProcessRegressionModel with no masking but conditioned on
    # only the not-masked-out index points.
    x = dist.sample(seed=test_util.test_seed())
    for i in range(3):
      observation_index_points_i = tf.gather(
          observation_index_points[i, 0],
          (~observations_is_missing[i, 0]).nonzero()[0])
      observations_i = tf.gather(
          observations[i, 0], (~observations_is_missing[i, 0]).nonzero()[0])
      dist_i = gprm.GaussianProcessRegressionModel.precompute_regression_model(
          kernel=kernel[i],
          index_points=index_points,
          observation_index_points=observation_index_points_i,
          observations=observations_i,
          observation_noise_variance=observation_noise_variance[i, 0],
          validate_args=True)

      self.assertAllClose(dist.mean()[i], dist_i.mean())
      self.assertAllClose(dist.variance()[i], dist_i.variance())
      self.assertAllClose(dist.covariance()[i], dist_i.covariance())
      self.assertAllClose(dist.log_prob(x)[i], dist_i.log_prob(x[i]))

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Numpy has no notion of CompositeTensor/Pytree.')
  def testPrecomputedCompositeTensorOrPytree(self):
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1])
    length_scale = np.array([.1, .2, .3], np.float64).reshape([1, 3])
    observation_noise_variance = np.array([1e-9], np.float64)

    observation_index_points = (
        np.random.uniform(-1., 1., (1, 1, 7, 2)).astype(np.float64))
    observations = np.random.uniform(-1., 1., (1, 1, 7)).astype(np.float64)

    index_points = np.random.uniform(-1., 1., (6, 2)).astype(np.float64)

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)

    precomputed_dist = gprm.GaussianProcessRegressionModel.precompute_regression_model(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        validate_args=True)

    flat = tf.nest.flatten(precomputed_dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        precomputed_dist, flat, expand_composites=True)
    self.assertIsInstance(unflat, gprm.GaussianProcessRegressionModel)
    # Check that we don't recompute the divisor matrix on flattening /
    # unflattening.
    self.assertIs(precomputed_dist.kernel._precomputed_divisor_matrix_cholesky,
                  unflat.kernel._precomputed_divisor_matrix_cholesky)

    x = self.evaluate(precomputed_dist.sample(3, seed=test_util.test_seed()))
    actual = self.evaluate(precomputed_dist.log_prob(x))
    self.assertAllClose(self.evaluate(unflat.log_prob(x)), actual)

    # TODO(b/196219597): Enable this test once GPRM works across TF function
    # boundaries.
    # index_observations = np.random.uniform(-1., 1., (6,)).astype(np.float64)
    # @tf.function
    # def log_prob(d):
    #   return d.log_prob(index_observations)

    # lp = self.evaluate(precomputed_gprm.log_prob(index_observations))

    # self.assertAllClose(lp, self.evaluate(log_prob(precomputed_gprm)))
    # self.assertAllClose(lp, self.evaluate(log_prob(unflat)))

  def testEmptyDataMatchesGPPrior(self):
    amp = np.float64(.5)
    len_scale = np.float64(.2)
    index_points = np.random.uniform(-1., 1., (10, 1)).astype(np.float64)

    # k_xx - k_xn @ (k_nn + sigma^2) @ k_nx + sigma^2
    mean_fn = lambda x: x[:, 0]**2

    kernel = exponentiated_quadratic.ExponentiatedQuadratic(amp, len_scale)
    gp = gaussian_process.GaussianProcess(
        kernel,
        index_points,
        mean_fn=mean_fn,
        validate_args=True)

    dist_nones = gprm.GaussianProcessRegressionModel(
        kernel,
        index_points,
        mean_fn=mean_fn,
        validate_args=True)

    dist_zero_shapes = gprm.GaussianProcessRegressionModel(
        kernel,
        index_points,
        observation_index_points=tf.ones([0, 1], tf.float64),
        observations=tf.ones([0], tf.float64),
        mean_fn=mean_fn,
        validate_args=True)

    for dist in [dist_nones, dist_zero_shapes]:
      self.assertAllClose(self.evaluate(gp.mean()), self.evaluate(dist.mean()))
      self.assertAllClose(
          self.evaluate(gp.covariance()), self.evaluate(dist.covariance()))
      self.assertAllClose(
          self.evaluate(gp.variance()), self.evaluate(dist.variance()))

      observations = np.random.uniform(-1., 1., 10).astype(np.float64)
      self.assertAllClose(
          self.evaluate(gp.log_prob(observations)),
          self.evaluate(dist.log_prob(observations)))

  def testErrorCases(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    index_points = np.random.uniform(-1., 1., (10, 1)).astype(np.float64)
    observation_index_points = (
        np.random.uniform(-1., 1., (5, 1)).astype(np.float64))
    observations = np.random.uniform(-1., 1., 3).astype(np.float64)

    # Both or neither of `observation_index_points` and `observations` must be
    # specified.
    with self.assertRaises(ValueError):
      gprm.GaussianProcessRegressionModel(
          kernel,
          index_points,
          observation_index_points=None,
          observations=observations,
          validate_args=True)
    with self.assertRaises(ValueError):
      gprm.GaussianProcessRegressionModel(
          kernel,
          index_points,
          observation_index_points,
          observations=None,
          validate_args=True)

    # If specified, mean_fn must be a callable.
    with self.assertRaises(ValueError):
      gprm.GaussianProcessRegressionModel(
          kernel, index_points, mean_fn=0., validate_args=True)

    # Observation index point and observation counts must be broadcastable.
    # Errors based on conditions of dynamic shape in graph mode cannot be
    # caught, so we only check this error case in static shape or eager mode.
    if self.is_static or tf.executing_eagerly():
      with self.assertRaises(ValueError):
        gprm.GaussianProcessRegressionModel(
            kernel,
            index_points,
            observation_index_points=np.ones([2, 2, 2]),
            observations=np.ones([5, 5]),
            validate_args=True)

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
      index_points_1 = tf1.placeholder_with_default(index_points_1, shape=None)
      index_points_2 = tf1.placeholder_with_default(index_points_2, shape=None)
      observation_index_points_1 = tf1.placeholder_with_default(
          observation_index_points_1, shape=None)
      observation_index_points_2 = tf1.placeholder_with_default(
          observation_index_points_2, shape=None)
      observations_1 = tf1.placeholder_with_default(observations_1, shape=None)
      observations_2 = tf1.placeholder_with_default(observations_2, shape=None)

    mean_fn = lambda x: np.array([0.], np.float32)
    kernel_1 = exponentiated_quadratic.ExponentiatedQuadratic()
    kernel_2 = exp_sin_squared.ExpSinSquared()

    dist1 = gprm.GaussianProcessRegressionModel(
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
        gprm.GaussianProcessRegressionModel.precompute_regression_model(
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

    self.assertIsInstance(dist1.kernel.base_kernel,
                          exponentiated_quadratic.ExponentiatedQuadratic)
    self.assertIsInstance(dist2.kernel.base_kernel,
                          exp_sin_squared.ExpSinSquared)

    if self.is_static or tf.executing_eagerly():
      self.assertAllEqual(dist1.batch_shape, dist2.batch_shape)
      self.assertAllEqual(dist1.event_shape, event_shape_1)
      self.assertAllEqual(dist2.event_shape, event_shape_2)
      self.assertAllEqual(dist1.index_points, index_points_1)
      self.assertAllEqual(dist2.index_points, index_points_2)
    else:
      self.assertAllEqual(
          self.evaluate(dist1.batch_shape_tensor()),
          self.evaluate(dist2.batch_shape_tensor()))
      self.assertAllEqual(
          self.evaluate(dist1.event_shape_tensor()), event_shape_1)
      self.assertAllEqual(
          self.evaluate(dist2.event_shape_tensor()), event_shape_2)
      self.assertAllEqual(self.evaluate(dist1.index_points), index_points_1)
      self.assertAllEqual(self.evaluate(dist2.index_points), index_points_2)

  # What is the behavior?
  #   - observation_noise_variance defaults to 0
  #   - predictive_noise_variance defaults to observation_noise_variance
  #
  # Expectation:
  #   - covariance matrix is affected by observation_noise_variance. namely,
  #     covariance matrix is given by
  #       (K_tt - K_tx @ (K_xx + ONV ** 2 * I) @ K_xt) + PNV
  #   - predictive_noise_variance property is a Tensor-converted version of the
  #     predictive_noise_variance argument to __init__
  #   - observation_noise_variance property is a Tensor-converted version of the
  #     effective predictive_noise_variance value. This is because of the
  #     confusing naming choice we made with the GaussianProcess base class.
  #
  # Scenarios:
  #   - ONV not given, PNV not given
  #   - ONV given, PNV not given
  #   - ONV not given, PNV given
  #   - ONV given, PNV given
  @parameterized.named_parameters(
      {'testcase_name': '_no_args',
       'noise_kwargs': {},
       'implied_values': {'observation_noise_variance_parameter': 0.,
                          'predictive_noise_variance_parameter': 0.}},

      {'testcase_name': '_pnv_none',
       'noise_kwargs': {'predictive_noise_variance': None},
       'implied_values': {'observation_noise_variance_parameter': 0.,
                          'predictive_noise_variance_parameter': 0.}},

      {'testcase_name': '_pnv_2',
       'noise_kwargs': {'predictive_noise_variance': 2.},
       'implied_values': {'observation_noise_variance_parameter': 0.,
                          'predictive_noise_variance_parameter': 2.}},

      {'testcase_name': '_onv_1',
       'noise_kwargs': {'observation_noise_variance': 1.},
       'implied_values': {'observation_noise_variance_parameter': 1.,
                          'predictive_noise_variance_parameter': 1.}},

      {'testcase_name': '_onv_1_pnv_2',
       'noise_kwargs': {'observation_noise_variance': 1.,
                        'predictive_noise_variance': 2.},
       'implied_values': {'observation_noise_variance_parameter': 1.,
                          'predictive_noise_variance_parameter': 2.}}
  )
  def testInitParameterVariations(self, noise_kwargs, implied_values):
    num_test_points = 3
    num_obs_points = 4
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    index_points = np.random.uniform(-1., 1., (num_test_points, 1))
    observation_index_points = np.random.uniform(-1., 1., (num_obs_points, 1))
    observations = np.random.uniform(-1., 1., num_obs_points)
    jitter = 1e-6

    dist = gprm.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        jitter=jitter,
        validate_args=True,
        **noise_kwargs)

    # 'Property' means what was passed to CTOR. 'Parameter' is the effective
    # value by which the distribution is parameterized.
    implied_onv_param = implied_values['observation_noise_variance_parameter']
    implied_pnv_param = implied_values['predictive_noise_variance_parameter']

    # k_xx - k_xn @ (k_nn + ONV) @ k_nx + PNV
    k = lambda x, y: _np_kernel_matrix_fn(1., 1., x, y)
    k_tt_ = k(index_points, index_points)
    k_tx_ = k(index_points, observation_index_points)
    k_xx_plus_noise_ = (
        k(observation_index_points, observation_index_points) +
        (jitter + implied_onv_param) * np.eye(num_obs_points))

    expected_predictive_covariance = (
        k_tt_ - np.dot(k_tx_, np.linalg.solve(k_xx_plus_noise_, k_tx_.T)) +
        implied_pnv_param * np.eye(num_test_points))

    # Assertion 1: predictive covariance is correct.
    self.assertAllClose(
        self.evaluate(dist.covariance()), expected_predictive_covariance)

    # Assertion 2: predictive_noise_variance property is correct
    self.assertIsInstance(dist.predictive_noise_variance, tf.Tensor)
    self.assertAllClose(
        self.evaluate(dist.predictive_noise_variance), implied_pnv_param)

    # Assertion 3: observation_noise_variance property is correct.
    self.assertIsInstance(dist.observation_noise_variance, tf.Tensor)
    self.assertAllClose(
        self.evaluate(dist.observation_noise_variance),
        # Note that this is, somewhat unintuitively, expceted to equal the
        # predictive_noise_variance. This is because of 1) the inheritance
        # structure of GPRM as a subclass of GaussianProcess and 2) the poor
        # choice of name of the GaussianProcess noise parameter. The latter
        # issue is being cleaned up in cl/256413439.
        implied_pnv_param)

  def testStructuredIndexPoints(self):
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    observation_index_points = np.random.uniform(
        -1, 1, (12, 8)).astype(np.float32)
    observations = np.sum(observation_index_points, axis=-1)
    index_points = np.random.uniform(-1, 1, (6, 8)).astype(np.float32)
    base_gprm = gprm.GaussianProcessRegressionModel(
        base_kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations)

    structured_kernel = psd_kernel_test_util.MultipartTestKernel(base_kernel)
    structured_obs_index_points = dict(
        zip(('foo', 'bar'),
            tf.split(observation_index_points, [5, 3], axis=-1)))
    structured_index_points = dict(
        zip(('foo', 'bar'), tf.split(index_points, [5, 3], axis=-1)))
    structured_gprm = gprm.GaussianProcessRegressionModel(
        structured_kernel,
        index_points=structured_index_points,
        observation_index_points=structured_obs_index_points,
        observations=observations)

    s = structured_gprm.sample(3, seed=test_util.test_seed())
    self.assertAllClose(base_gprm.log_prob(s), structured_gprm.log_prob(s))
    self.assertAllClose(base_gprm.mean(), structured_gprm.mean())
    self.assertAllClose(base_gprm.variance(), structured_gprm.variance())
    self.assertAllEqual(base_gprm.event_shape, structured_gprm.event_shape)
    self.assertAllEqual(base_gprm.event_shape_tensor(),
                        structured_gprm.event_shape_tensor())
    self.assertAllEqual(base_gprm.batch_shape, structured_gprm.batch_shape)
    self.assertAllEqual(base_gprm.batch_shape_tensor(),
                        structured_gprm.batch_shape_tensor())

    # Iterable index points should be interpreted as single Tensors if the
    # kernel is not structured.
    index_points_list = tf.unstack(index_points)
    obs_index_points_nested_list = tf.nest.map_structure(
        tf.unstack, tf.unstack(observation_index_points))
    gprm_with_lists = gprm.GaussianProcessRegressionModel(
        base_kernel,
        index_points=index_points_list,
        observation_index_points=obs_index_points_nested_list,
        observations=observations)
    self.assertAllEqual(base_gprm.event_shape_tensor(),
                        gprm_with_lists.event_shape_tensor())
    self.assertAllEqual(base_gprm.batch_shape_tensor(),
                        gprm_with_lists.batch_shape_tensor())
    self.assertAllClose(base_gprm.log_prob(s), gprm_with_lists.log_prob(s))

  def testPrivateArgPreventsCholeskyRecomputation(self):
    x = np.random.uniform(-1, 1, (4, 7)).astype(np.float32)
    x_obs = np.random.uniform(-1, 1, (4, 7)).astype(np.float32)
    y_obs = np.random.uniform(-1, 1, (4,)).astype(np.float32)
    chol = np.eye(4).astype(np.float32)
    mock_cholesky_fn = mock.Mock(return_value=chol)
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    d = gprm.GaussianProcessRegressionModel.precompute_regression_model(
        base_kernel,
        index_points=x,
        observation_index_points=x_obs,
        observations=y_obs,
        cholesky_fn=mock_cholesky_fn)
    mock_cholesky_fn.assert_called_once()

    mock_cholesky_fn.reset_mock()
    d2 = gprm.GaussianProcessRegressionModel.precompute_regression_model(
        base_kernel,
        index_points=x,
        observation_index_points=x_obs,
        observations=y_obs,
        cholesky_fn=mock_cholesky_fn,
        _precomputed_divisor_matrix_cholesky=(
            d._precomputed_divisor_matrix_cholesky),
        _precomputed_solve_on_observation=d._precomputed_solve_on_observation)
    mock_cholesky_fn.assert_not_called()

    # The Cholesky is computed just once in each call to log_prob (on the
    # index points kernel matrix).
    self.assertAllClose(d.log_prob(y_obs), d2.log_prob(y_obs))
    self.assertEqual(mock_cholesky_fn.call_count, 2)

  def test_batch_slice_precomputed_gprm(self):
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        length_scale=tf.linspace(tf.ones([]), 2., 64), feature_ndims=0)
    x = tf.linspace(tf.zeros([]), 1., 126)
    y = tf.linspace(tf.zeros([]), 1.5, 162)
    d = gprm.GaussianProcessRegressionModel.precompute_regression_model(
        base_kernel,
        index_points=y,
        observation_index_points=x,
        observations=tf.math.sin(x),
        observation_noise_variance=1e-3)
    self.assertEqual((64,), d.batch_shape)
    self.assertEqual((162,), d.event_shape)
    self.assertEqual((64, 162,), d.sample(seed=test_util.test_seed()).shape)

    self.assertEqual((), d[2].batch_shape)
    self.assertEqual((162,), d[2].event_shape)
    self.assertEqual((162,), d[2].sample(seed=test_util.test_seed()).shape)


class GaussianProcessRegressionModelStaticTest(
    _GaussianProcessRegressionModelTest):
  is_static = True


class GaussianProcessRegressionModelDynamicTest(
    _GaussianProcessRegressionModelTest):
  is_static = False

del _GaussianProcessRegressionModelTest

if __name__ == '__main__':
  test_util.main()
