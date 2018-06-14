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
"""Linear Gaussian State Space Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.linear_gaussian_ssm import _augment_sample_shape
from tensorflow_probability.python.distributions.linear_gaussian_ssm import build_kalman_cov_step
from tensorflow_probability.python.distributions.linear_gaussian_ssm import build_kalman_filter_step
from tensorflow_probability.python.distributions.linear_gaussian_ssm import build_kalman_mean_step
from tensorflow_probability.python.distributions.linear_gaussian_ssm import kalman_transition
from tensorflow_probability.python.distributions.linear_gaussian_ssm import KalmanFilterState
from tensorflow_probability.python.distributions.linear_gaussian_ssm import linear_gaussian_update

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

tfl = tf.contrib.linalg
tfd = tfp.distributions


class IIDNormalTest(test.TestCase):

  def setUp(self):
    pass

  def _build_iid_normal_model(self,
                              num_timesteps,
                              latent_size,
                              observation_size,
                              transition_variance,
                              obs_variance):
    """Build a model whose outputs are IID normal by construction."""

    # Use orthogonal matrices to project a (potentially
    # high-dimensional) latent space of IID normal variables into a
    # low-dimensional observation that is still IID normal.
    random_orthogonal_matrix = lambda: np.linalg.qr(
        np.random.randn(latent_size, latent_size))[0][:observation_size, :]
    obs_matrix = tf.convert_to_tensor(random_orthogonal_matrix(),
                                      dtype=tf.float32)

    model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=tf.zeros((latent_size, latent_size)),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.sqrt(transition_variance)*tf.ones((latent_size))),
        observation_matrix=obs_matrix,
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.sqrt(obs_variance)*tf.ones((observation_size))),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=tf.sqrt(transition_variance)*tf.ones((latent_size))),
        validate_args=True)

    return model

  @test_util.run_in_graph_and_eager_modes()
  def test_iid_normal_sample(self):
    num_timesteps = 10
    latent_size = 3
    observation_size = 2
    num_samples = 10000

    for transition_variance_val in [.3, 100.]:
      for obs_variance_val in [.6, 40.]:

        iid_latents = self._build_iid_normal_model(
            num_timesteps=num_timesteps,
            latent_size=latent_size,
            observation_size=observation_size,
            transition_variance=transition_variance_val,
            obs_variance=obs_variance_val)

        x = iid_latents.sample(num_samples)

        x_val = self.evaluate(x)
        result_shape = [num_timesteps, observation_size]
        marginal_variance = transition_variance_val + obs_variance_val

        stderr_mean = np.sqrt(num_samples * marginal_variance)
        stderr_variance = marginal_variance * np.sqrt(2./(num_samples-1))

        self.assertAllClose(np.mean(x_val, axis=0),
                            np.zeros(result_shape),
                            atol=5*stderr_mean)
        self.assertAllClose(np.var(x_val, axis=0),
                            np.ones(result_shape) * marginal_variance,
                            rtol=5*stderr_variance)

  @test_util.run_in_graph_and_eager_modes()
  def test_iid_normal_logprob(self):

    # In the case where the latent states are iid normal (achieved by
    # setting the transition matrix to zero, so there's no dependence
    # between timesteps), and observations are also independent
    # (achieved by using an orthogonal matrix as the observation model),
    # we can verify log_prob as a simple iid Gaussian log density.
    delta = 1e-4
    for transition_variance_val in [1., 1e-8]:
      for obs_variance_val in [1., 1e-8]:

        iid_latents = self._build_iid_normal_model(
            num_timesteps=10,
            latent_size=4,
            observation_size=2,
            transition_variance=transition_variance_val,
            obs_variance=obs_variance_val)

        x = iid_latents.sample([5, 3])
        lp_kalman = iid_latents.log_prob(x)

        marginal_variance = transition_variance_val + obs_variance_val
        lp_iid = tf.reduce_sum(
            tfd.Normal(0., tf.sqrt(marginal_variance)).log_prob(x),
            axis=(-2, -1))

        lp_kalman_val, lp_iid_val = self.evaluate((lp_kalman, lp_iid))
        self.assertAllClose(lp_kalman_val,
                            lp_iid_val,
                            rtol=delta, atol=0.)


class BatchTest(test.TestCase):
  """Test that methods broadcast batch dimensions for each parameter."""

  def setUp(self):
    pass

  def _build_random_model(self,
                          num_timesteps,
                          latent_size,
                          observation_size,
                          prior_batch_shape=None,
                          transition_matrix_batch_shape=None,
                          transition_noise_batch_shape=None,
                          observation_matrix_batch_shape=None,
                          observation_noise_batch_shape=None):
    """Builds a LGSSM with random normal ops of specified shape."""

    prior_batch_shape = (
        [] if prior_batch_shape is None else prior_batch_shape)
    transition_matrix_batch_shape = ([] if transition_matrix_batch_shape is None
                                     else transition_matrix_batch_shape)
    transition_noise_batch_shape = ([] if transition_noise_batch_shape is None
                                    else transition_noise_batch_shape)
    observation_matrix_batch_shape = ([]
                                      if observation_matrix_batch_shape is None
                                      else observation_matrix_batch_shape)
    observation_noise_batch_shape = ([] if observation_noise_batch_shape is None
                                     else observation_noise_batch_shape)

    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=tf.random_normal(
            transition_matrix_batch_shape + [latent_size, latent_size]),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.nn.softplus(tf.random_normal(
                transition_noise_batch_shape + [latent_size]))),
        observation_matrix=tf.random_normal(
            observation_matrix_batch_shape + [observation_size, latent_size]),
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.nn.softplus(tf.random_normal(
                observation_noise_batch_shape + [observation_size]))),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=tf.nn.softplus(tf.random_normal(
                prior_batch_shape + [latent_size]))),
        validate_args=True)

  def _sanity_check_shapes(self, model,
                           batch_shape,
                           event_shape,
                           sample_shape=(2, 1)):

    # Lists can't be default arguments, but we'll want sample_shape to
    # be a list so we can concatenate with other shapes passed as
    # lists.
    sample_shape = list(sample_shape)

    self.assertEqual(model.event_shape.as_list(), event_shape)
    self.assertEqual(model.batch_shape.as_list(), batch_shape)

    y = model.sample(sample_shape)
    self.assertEqual(y.shape.as_list(),
                     sample_shape + batch_shape + event_shape)

    lp = model.log_prob(y)
    self.assertEqual(lp.shape.as_list(), sample_shape + batch_shape)

    # Try an argument with no batch shape to ensure we broadcast
    # correctly.
    unbatched_y = tf.random_normal(event_shape)
    lp = model.log_prob(unbatched_y)
    self.assertEqual(lp.shape.as_list(), batch_shape)

    self.assertEqual(model.mean().shape.as_list(),
                     batch_shape + event_shape)
    self.assertEqual(model.variance().shape.as_list(),
                     batch_shape + event_shape)

  @test_util.run_in_graph_and_eager_modes()
  def test_constant_batch_shape(self):
    """Simple case where all components have the same batch shape."""
    num_timesteps = 5
    latent_size = 3
    observation_size = 2
    batch_shape = [3, 4]
    event_shape = [num_timesteps, observation_size]

    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     prior_batch_shape=batch_shape,
                                     transition_matrix_batch_shape=batch_shape,
                                     transition_noise_batch_shape=batch_shape,
                                     observation_matrix_batch_shape=batch_shape,
                                     observation_noise_batch_shape=batch_shape)

    # check that we get the basic shapes right
    self.assertEqual(model.latent_size, latent_size)
    self.assertEqual(model.observation_size, observation_size)
    self._sanity_check_shapes(model, batch_shape, event_shape)

  @test_util.run_in_graph_and_eager_modes()
  def test_broadcast_batch_shape(self):
    """Broadcasting when only one component has batch shape."""

    num_timesteps = 5
    latent_size = 3
    observation_size = 2
    batch_shape = [3, 4]
    event_shape = [num_timesteps, observation_size]

    # Test batching only over the prior
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     prior_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape)

    # Test batching only over the transition op
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     transition_matrix_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape)

    # Test batching only over the transition noise
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     transition_noise_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape)

    # Test batching only over the observation op
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     observation_matrix_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape)

    # Test batching only over the observation noise
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     observation_noise_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape)

  def test_batch_shape_error(self):
    # build a dist where components have incompatible batch
    # shapes. this should cause a problem somehow.
    pass


class _KalmanStepsTest(object):

  def setUp(self):
    # Define a simple model with 2D latents and 1D observations.

    self.transition_matrix = np.asarray([[1., .5], [-.2, .3]], dtype=np.float32)
    self.get_transition_matrix_for_timestep = (
        lambda t: tfl.LinearOperatorFullMatrix(self.transition_matrix))

    self.bias = np.asarray([-4.3, .9], dtype=np.float32)
    self.get_transition_noise_for_timestep = (
        lambda t: tfd.MultivariateNormalDiag(self.bias, [1., 1.]))

    self.get_observation_matrix_for_timestep = (
        lambda t: tfl.LinearOperatorFullMatrix([[1., 1.]]))

    self.observation_bias = np.asarray([-.9], dtype=np.float32)
    self.get_observation_noise_for_timestep = (
        lambda t: tfd.MultivariateNormalDiag(self.observation_bias, [1.]))

  @test_util.run_in_graph_and_eager_modes()
  def testKalmanFilterStep(self):
    prev_mean = np.asarray([[-2], [.4]], dtype=np.float32)
    prev_cov = np.asarray([[.5, .1], [.2, .6]], dtype=np.float32)
    x_observed = np.asarray([[4.]], dtype=np.float32)

    prev_mean_tensor = self.build_tensor(prev_mean)
    prev_cov_tensor = self.build_tensor(prev_cov)
    x_observed_tensor = self.build_tensor(x_observed)

    filter_step = build_kalman_filter_step(
        self.get_transition_matrix_for_timestep,
        self.get_transition_noise_for_timestep,
        self.get_observation_matrix_for_timestep,
        self.get_observation_noise_for_timestep)

    initial_filter_state = KalmanFilterState(
        filtered_mean=None,
        filtered_cov=None,
        predicted_mean=prev_mean_tensor,
        predicted_cov=prev_cov_tensor,
        observation_mean=None,
        observation_cov=None,
        log_marginal_likelihood=self.build_tensor(0.),
        timestep=self.build_tensor(0))

    filter_state = self.evaluate(
        filter_step(initial_filter_state,
                    x_observed_tensor))

    # Computed by running a believed-correct version of
    # the code.
    expected_filtered_mean = [[-0.104167], [2.295833]]
    expected_filtered_cov = [[0.325, -0.075], [-0.033333, 0.366667]]
    expected_predicted_mean = [[-3.25625], [1.609583]]
    expected_predicted_cov = [[1.3625, -0.0125], [-0.029167, 1.0525]]
    expected_observation_mean = [[-2.5]]
    expected_observation_cov = [[2.4]]
    expected_log_marginal_likelihood = -10.1587553024292

    self.assertAllClose(filter_state.filtered_mean,
                        expected_filtered_mean)
    self.assertAllClose(filter_state.filtered_cov,
                        expected_filtered_cov)
    self.assertAllClose(filter_state.predicted_mean,
                        expected_predicted_mean)
    self.assertAllClose(filter_state.predicted_cov,
                        expected_predicted_cov)
    self.assertAllClose(filter_state.observation_mean,
                        expected_observation_mean)
    self.assertAllClose(filter_state.observation_cov,
                        expected_observation_cov)
    self.assertAllClose(filter_state.log_marginal_likelihood,
                        expected_log_marginal_likelihood)
    self.assertAllClose(filter_state.timestep, 1)

  @test_util.run_in_graph_and_eager_modes()
  def testKalmanTransition(self):

    prev_mean = np.asarray([[-2], [.4]], dtype=np.float32)
    prev_cov = np.asarray([[.5, -.2], [-.2, .9]], dtype=np.float32)
    prev_mean_tensor = self.build_tensor(prev_mean)
    prev_cov_tensor = self.build_tensor(prev_cov)

    predicted_mean, predicted_cov = kalman_transition(
        prev_mean_tensor, prev_cov_tensor,
        self.get_transition_matrix_for_timestep(0),
        self.get_transition_noise_for_timestep(0))

    self.assertAllClose(self.evaluate(predicted_mean),
                        np.dot(self.transition_matrix,
                               prev_mean) + self.bias[:, np.newaxis])
    self.assertAllClose(self.evaluate(predicted_cov),
                        np.dot(self.transition_matrix,
                               np.dot(prev_cov,
                                      self.transition_matrix.T)) + np.eye(2))

  @test_util.run_in_graph_and_eager_modes()
  def testLinearGaussianObservation(self):

    prev_mean = np.asarray([[-2], [.4]], dtype=np.float32)
    prev_cov = np.asarray([[.5, -.2], [-.2, .9]], dtype=np.float32)
    x_observed = np.asarray([[4.]], dtype=np.float32)

    prev_mean_tensor = self.build_tensor(prev_mean)
    prev_cov_tensor = self.build_tensor(prev_cov)
    x_observed_tensor = self.build_tensor(x_observed)

    observation_matrix = self.get_observation_matrix_for_timestep(0)
    observation_noise = self.get_observation_noise_for_timestep(0)

    (posterior_mean,
     posterior_cov,
     predictive_dist) = linear_gaussian_update(
         prev_mean_tensor, prev_cov_tensor,
         observation_matrix, observation_noise,
         x_observed_tensor)

    expected_posterior_mean = [[-1.025], [2.675]]
    expected_posterior_cov = [[0.455, -0.305], [-0.305, 0.655]]
    expected_predicted_mean = [-2.5]
    expected_predicted_cov = [[2.]]

    self.assertAllClose(self.evaluate(posterior_mean),
                        expected_posterior_mean)
    self.assertAllClose(self.evaluate(posterior_cov),
                        expected_posterior_cov)
    self.assertAllClose(self.evaluate(predictive_dist.mean()),
                        expected_predicted_mean)
    self.assertAllClose(self.evaluate(predictive_dist.covariance()),
                        expected_predicted_cov)

  @test_util.run_in_graph_and_eager_modes()
  def testMeanStep(self):
    prev_mean = np.asarray([[-2], [.4]], dtype=np.float32)
    prev_mean_tensor = self.build_tensor(prev_mean)

    mean_step = build_kalman_mean_step(
        self.get_transition_matrix_for_timestep,
        self.get_transition_noise_for_timestep,
        self.get_observation_matrix_for_timestep,
        self.get_observation_noise_for_timestep)
    new_mean, obs_mean = mean_step((prev_mean_tensor, None), t=0)

    self.assertAllClose(self.evaluate(new_mean),
                        np.dot(self.transition_matrix,
                               prev_mean) + self.bias[:, np.newaxis])
    self.assertAllClose(self.evaluate(obs_mean),
                        np.sum(self.evaluate(new_mean)) +
                        self.observation_bias[:, np.newaxis])

  @test_util.run_in_graph_and_eager_modes()
  def testCovStep(self):

    prev_cov = np.asarray([[.5, -.2], [-.2, .9]], dtype=np.float32)
    prev_cov_tensor = self.build_tensor(prev_cov)

    cov_step = build_kalman_cov_step(
        self.get_transition_matrix_for_timestep,
        self.get_transition_noise_for_timestep,
        self.get_observation_matrix_for_timestep,
        self.get_observation_noise_for_timestep)
    new_cov, obs_cov = cov_step((prev_cov_tensor, None), t=0)

    self.assertAllClose(self.evaluate(new_cov),
                        np.dot(self.transition_matrix,
                               np.dot(prev_cov,
                                      self.transition_matrix.T)) + np.eye(2))
    self.assertAllClose(self.evaluate(obs_cov),
                        [[np.sum(self.evaluate(new_cov)) + 1.]])


class KalmanStepsTestStatic(test.TestCase, _KalmanStepsTest):

  def setUp(self):
    return _KalmanStepsTest.setUp(self)

  def build_tensor(self, tensor):
    return tf.convert_to_tensor(tensor)


class KalmanStepsTestDynamic(test.TestCase, _KalmanStepsTest):

  def setUp(self):
    return _KalmanStepsTest.setUp(self)

  def build_tensor(self, tensor):
    return tf.placeholder_with_default(input=tf.convert_to_tensor(tensor),
                                       shape=None)


class _AugmentSampleShapeTest(object):

  @test_util.run_in_graph_and_eager_modes()
  def testAugmentsShape(self):

    full_batch_shape, dist = self.build_inputs([5, 4, 2, 3], [2, 3])

    sample_shape = _augment_sample_shape(dist, full_batch_shape,
                                         validate_args=True)

    self.assertAllEqual(self.maybe_evaluate(sample_shape), [5, 4])

  @test_util.run_in_graph_and_eager_modes()
  def testSameShape(self):

    full_batch_shape, dist = self.build_inputs([5, 4, 2, 3], [5, 4, 2, 3])
    sample_shape = _augment_sample_shape(dist, full_batch_shape,
                                         validate_args=True)

    self.assertAllEqual(self.maybe_evaluate(sample_shape), [])

  # We omit the eager-mode decorator for error handling checks,
  # because eager mode throws dynamic errors statically which confuses
  # the test harness.
  def testNotPrefixThrowsError(self):

    full_batch_shape, dist = self.build_inputs([5, 4, 2, 3], [1, 3])

    with self.assertRaisesError("Broadcasting is not supported"):
      self.maybe_evaluate(
          _augment_sample_shape(dist, full_batch_shape,
                                validate_args=True))

  def testTooManyDimsThrowsError(self):

    full_batch_shape, dist = self.build_inputs([5, 4, 2, 3], [6, 5, 4, 2, 3])

    with self.assertRaisesError("Cannot broadcast"):
      self.maybe_evaluate(
          _augment_sample_shape(dist, full_batch_shape,
                                validate_args=True))


class AugmentSampleShapeTestStatic(test.TestCase, _AugmentSampleShapeTest):

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def build_inputs(self, full_batch_shape, partial_batch_shape):

    full_batch_shape = np.asarray(full_batch_shape, dtype=np.int32)
    dist = tfd.Normal(tf.random_normal(partial_batch_shape), 1.)

    return full_batch_shape, dist

  def maybe_evaluate(self, x):
    return x


class AugmentSampleShapeTestDynamic(test.TestCase, _AugmentSampleShapeTest):

  def assertRaisesError(self, msg):
    return self.assertRaisesOpError(msg)

  def build_inputs(self, full_batch_shape, partial_batch_shape):
    full_batch_shape = tf.placeholder_with_default(
        input=np.asarray(full_batch_shape, dtype=np.int32),
        shape=None)

    partial_batch_shape = tf.placeholder_with_default(
        input=np.asarray(partial_batch_shape, dtype=np.int32),
        shape=None)
    dist = tfd.Normal(tf.random_normal(partial_batch_shape), 1.)

    return full_batch_shape, dist

  def maybe_evaluate(self, x):
    return self.evaluate(x)


if __name__ == "__main__":
  test.main()
