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
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import psd_kernels


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

    if not self.is_static:
      amplitude = tf1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf1.placeholder_with_default(length_scale, shape=None)
      batched_index_points = tf1.placeholder_with_default(
          batched_index_points, shape=None)

      observation_index_points = tf1.placeholder_with_default(
          observation_index_points, shape=None)
      observations = tf1.placeholder_with_default(observations, shape=None)

    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    gprm = tfd.GaussianProcessRegressionModel(
        kernel,
        batched_index_points,
        observation_index_points,
        observations,
        observation_noise_variance,
        jitter=jitter,
        validate_args=True)

    batch_shape = [2, 3, 5, 7, 11]
    event_shape = [25]
    sample_shape = [9, 3]

    samples = gprm.sample(sample_shape, seed=test_util.test_seed())

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
      self.assertIsNone(tensorshape_util.rank(samples.shape))
      self.assertIsNone(tensorshape_util.rank(gprm.batch_shape))
      self.assertEqual(tensorshape_util.rank(gprm.event_shape), 1)
      self.assertIsNone(
          tf.compat.dimension_value(tensorshape_util.dims(gprm.event_shape)[0]))

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

    kernel = psd_kernels.ExponentiatedQuadratic(amp, len_scale)
    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        mean_fn=mean_fn,
        jitter=jitter,
        validate_args=True)

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
        jitter=jitter,
        validate_args=True)

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
        jitter=jitter,
        validate_args=True)

    gprm_nones = tfd.GaussianProcessRegressionModel(
        kernel,
        index_points,
        mean_fn=mean_fn,
        jitter=jitter,
        validate_args=True)

    gprm_zero_shapes = tfd.GaussianProcessRegressionModel(
        kernel,
        index_points,
        observation_index_points=tf.ones([5, 0, 1], tf.float64),
        observations=tf.ones([5, 0], tf.float64),
        mean_fn=mean_fn,
        jitter=jitter,
        validate_args=True)

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
          observations=observations,
          validate_args=True)
    with self.assertRaises(ValueError):
      tfd.GaussianProcessRegressionModel(
          kernel,
          index_points,
          observation_index_points,
          observations=None,
          validate_args=True)

    # If specified, mean_fn must be a callable.
    with self.assertRaises(ValueError):
      tfd.GaussianProcessRegressionModel(
          kernel, index_points, mean_fn=0., validate_args=True)

    # Observation index point and observation counts must be broadcastable.
    # Errors based on conditions of dynamic shape in graph mode cannot be
    # caught, so we only check this error case in static shape or eager mode.
    if self.is_static or tf.executing_eagerly():
      with self.assertRaises(ValueError):
        tfd.GaussianProcessRegressionModel(
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
    kernel_1 = psd_kernels.ExponentiatedQuadratic()
    kernel_2 = psd_kernels.ExpSinSquared()

    gprm1 = tfd.GaussianProcessRegressionModel(
        kernel=kernel_1,
        index_points=index_points_1,
        observation_index_points=observation_index_points_1,
        observations=observations_1,
        mean_fn=mean_fn,
        jitter=1e-5,
        validate_args=True)
    gprm2 = gprm1.copy(
        kernel=kernel_2,
        index_points=index_points_2,
        observation_index_points=observation_index_points_2,
        observations=observations_2)

    event_shape_1 = [5]
    event_shape_2 = [10]

    self.assertIsInstance(gprm1.kernel.base_kernel,
                          psd_kernels.ExponentiatedQuadratic)
    self.assertIsInstance(gprm2.kernel.base_kernel, psd_kernels.ExpSinSquared)

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
    kernel = psd_kernels.ExponentiatedQuadratic()
    index_points = np.random.uniform(-1., 1., (num_test_points, 1))
    observation_index_points = np.random.uniform(-1., 1., (num_obs_points, 1))
    observations = np.random.uniform(-1., 1., num_obs_points)
    jitter = 1e-6

    gprm = tfd.GaussianProcessRegressionModel(
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
    self.assertAllClose(self.evaluate(gprm.covariance()),
                        expected_predictive_covariance)

    # Assertion 2: predictive_noise_variance property is correct
    self.assertIsInstance(gprm.predictive_noise_variance, tf.Tensor)
    self.assertAllClose(
        self.evaluate(gprm.predictive_noise_variance), implied_pnv_param)

    # Assertion 3: observation_noise_variance property is correct.
    self.assertIsInstance(gprm.observation_noise_variance, tf.Tensor)
    self.assertAllClose(
        self.evaluate(gprm.observation_noise_variance),
        # Note that this is, somewhat unintuitively, expceted to equal the
        # predictive_noise_variance. This is because of 1) the inheritance
        # structure of GPRM as a subclass of GaussianProcess and 2) the poor
        # choice of name of the GaussianProcess noise parameter. The latter
        # issue is being cleaned up in cl/256413439.
        implied_pnv_param)


class GaussianProcessRegressionModelStaticTest(
    _GaussianProcessRegressionModelTest):
  is_static = True


class GaussianProcessRegressionModelDynamicTest(
    _GaussianProcessRegressionModelTest):
  is_static = False

del _GaussianProcessRegressionModelTest

if __name__ == '__main__':
  tf.test.main()
