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
from tensorflow_probability.python.distributions.linear_gaussian_ssm import backward_smoothing_update
from tensorflow_probability.python.distributions.linear_gaussian_ssm import BackwardPassState
from tensorflow_probability.python.distributions.linear_gaussian_ssm import build_backward_pass_step
from tensorflow_probability.python.distributions.linear_gaussian_ssm import build_kalman_cov_step
from tensorflow_probability.python.distributions.linear_gaussian_ssm import build_kalman_filter_step
from tensorflow_probability.python.distributions.linear_gaussian_ssm import build_kalman_mean_step
from tensorflow_probability.python.distributions.linear_gaussian_ssm import kalman_transition
from tensorflow_probability.python.distributions.linear_gaussian_ssm import KalmanFilterState
from tensorflow_probability.python.distributions.linear_gaussian_ssm import linear_gaussian_update

tfd = tfp.distributions
tfe = tf.contrib.eager
tfl = tf.linalg


@tfe.run_all_tests_in_graph_and_eager_modes
class _IIDNormalTest(object):

  def setUp(self):
    pass

  def _build_iid_normal_model(self, num_timesteps, latent_size,
                              observation_size, transition_variance,
                              observation_variance):
    """Build a model whose outputs are IID normal by construction."""

    transition_variance = self._build_placeholder(transition_variance)
    observation_variance = self._build_placeholder(observation_variance)

    # Use orthogonal matrices to project a (potentially
    # high-dimensional) latent space of IID normal variables into a
    # low-dimensional observation that is still IID normal.
    random_orthogonal_matrix = lambda: np.linalg.qr(
        np.random.randn(latent_size, latent_size))[0][:observation_size, :]
    observation_matrix = self._build_placeholder(random_orthogonal_matrix())

    model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=self._build_placeholder(
            np.zeros([latent_size, latent_size])),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.sqrt(transition_variance) *
            tf.ones([latent_size], dtype=self.dtype)),
        observation_matrix=observation_matrix,
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.sqrt(observation_variance) *
            tf.ones([observation_size], dtype=self.dtype)),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=tf.sqrt(transition_variance) *
            tf.ones([latent_size], dtype=self.dtype)),
        validate_args=True)

    return model

  def test_iid_normal_sample(self):
    num_timesteps = 10
    latent_size = 3
    observation_size = 2
    num_samples = 10000

    for transition_variance_val in [.3, 100.]:
      for observation_variance_val in [.6, 40.]:

        iid_latents = self._build_iid_normal_model(
            num_timesteps=num_timesteps,
            latent_size=latent_size,
            observation_size=observation_size,
            transition_variance=transition_variance_val,
            observation_variance=observation_variance_val)

        x = iid_latents.sample(num_samples)

        x_val = self.evaluate(x)
        result_shape = [num_timesteps, observation_size]
        marginal_variance = transition_variance_val + observation_variance_val

        stderr_mean = np.sqrt(num_samples * marginal_variance)
        stderr_variance = marginal_variance * np.sqrt(2./(num_samples-1))

        self.assertAllClose(np.mean(x_val, axis=0),
                            np.zeros(result_shape),
                            atol=5*stderr_mean)
        self.assertAllClose(np.var(x_val, axis=0),
                            np.ones(result_shape) * marginal_variance,
                            rtol=5*stderr_variance)

  def test_iid_normal_logprob(self):

    # In the case where the latent states are iid normal (achieved by
    # setting the transition matrix to zero, so there's no dependence
    # between timesteps), and observations are also independent
    # (achieved by using an orthogonal matrix as the observation model),
    # we can verify log_prob as a simple iid Gaussian log density.
    delta = 1e-4
    for transition_variance_val in [1., 1e-8]:
      for observation_variance_val in [1., 1e-8]:

        iid_latents = self._build_iid_normal_model(
            num_timesteps=10,
            latent_size=4,
            observation_size=2,
            transition_variance=transition_variance_val,
            observation_variance=observation_variance_val)

        x = iid_latents.sample([5, 3])
        lp_kalman = iid_latents.log_prob(x)

        marginal_variance = tf.convert_to_tensor(
            transition_variance_val + observation_variance_val,
            dtype=self.dtype)
        lp_iid = tf.reduce_sum(
            tfd.Normal(
                loc=tf.zeros([], dtype=self.dtype),
                scale=tf.sqrt(marginal_variance)).log_prob(x),
            axis=(-2, -1))

        lp_kalman_val, lp_iid_val = self.evaluate((lp_kalman, lp_iid))
        self.assertAllClose(lp_kalman_val,
                            lp_iid_val,
                            rtol=delta, atol=0.)

  def _build_placeholder(self, ndarray):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)


@tfe.run_all_tests_in_graph_and_eager_modes
class IIDNormalTestStatic32(_IIDNormalTest, tf.test.TestCase):
  use_static_shape = True
  dtype = np.float32


@tfe.run_all_tests_in_graph_and_eager_modes
class IIDNormalTestStatic64(_IIDNormalTest, tf.test.TestCase):
  use_static_shape = True
  dtype = np.float64


@tfe.run_all_tests_in_graph_and_eager_modes
class IIDNormalTestDynamic32(_IIDNormalTest, tf.test.TestCase):
  use_static_shape = False
  dtype = np.float32


@tfe.run_all_tests_in_graph_and_eager_modes
class SanityChecks(tf.test.TestCase):

  def test_deterministic_system(self):

    # Define a deterministic linear system
    num_timesteps = 5
    prior_mean = 17.
    step_coef = 1.3
    step_shift = -1.5
    observation_coef = 0.3
    observation_shift = -1.
    model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=[[step_coef]],
        transition_noise=tfd.MultivariateNormalDiag(loc=[step_shift],
                                                    scale_diag=[0.]),
        observation_matrix=[[observation_coef]],
        observation_noise=tfd.MultivariateNormalDiag(loc=[observation_shift],
                                                     scale_diag=[0.]),
        initial_state_prior=tfd.MultivariateNormalDiag(loc=[prior_mean],
                                                       scale_diag=[0.]))

    # Manually compute expected output.
    expected_latents = [prior_mean]
    for _ in range(num_timesteps-1):
      expected_latents.append(step_coef * expected_latents[-1] + step_shift)
    expected_latents = np.asarray(expected_latents)
    expected_observations = (
        expected_latents * observation_coef + observation_shift)

    mean_, sample_ = self.evaluate([model.mean(), model.sample()])
    self.assertAllClose(mean_[..., 0], expected_observations)
    self.assertAllClose(sample_[..., 0], expected_observations)

  def test_variance(self):

    num_timesteps = 5
    prior_scale = 17.
    step_coef = 1.3
    step_scale = 0.5
    observation_coef = 20.0
    observation_scale = 1.9

    model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=[[step_coef]],
        transition_noise=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[step_scale]),
        observation_matrix=[[observation_coef]],
        observation_noise=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[observation_scale]),
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[prior_scale]))

    # Manually compute the marginal variance at each step
    latent_variance = [prior_scale**2]
    for _ in range(num_timesteps-1):
      latent_variance.append(step_coef**2 * latent_variance[-1] + step_scale**2)
    latent_variance = np.asarray(latent_variance)
    observation_variance = (
        latent_variance * observation_coef**2 + observation_scale**2)

    variance_ = self.evaluate(model.variance())
    self.assertAllClose(variance_[..., 0], observation_variance)

  def test_time_varying_operators(self):

    num_timesteps = 5
    prior_mean = 1.3
    prior_scale = 1e-4
    transition_scale = 0.
    observation_noise_scale = 1e-4

    # Define time-varying transition and observation matrices.
    def transition_matrix(t):
      t = tf.cast(t, tf.float32)
      return tf.linalg.LinearOperatorFullMatrix([[t+1]])

    def observation_matrix(t):
      t = tf.cast(t, tf.float32)
      return tf.linalg.LinearOperatorFullMatrix([[tf.sqrt(t+1.)]])

    model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=transition_matrix,
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=[transition_scale]),
        observation_matrix=observation_matrix,
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=[observation_noise_scale]),
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[prior_mean], scale_diag=[prior_scale]))

    mean_, sample_ = self.evaluate([model.mean(), model.sample()])

    # Manually compute the expected output of the model (i.e., the prior
    # mean). Since the model is near-deterministic, we expect any samples to
    # be close to this value, so we can also use this to test the `sample`
    # method.
    latent_means = [prior_mean]
    for t in range(0, num_timesteps-1):
      latent_means.append((t+1) * latent_means[-1])
    observation_means = [latent_means[t] * np.sqrt(t+1)
                         for t in range(num_timesteps)]

    self.assertAllClose(observation_means, mean_[..., 0], atol=1e-4)
    self.assertAllClose(observation_means, sample_[..., 0], atol=3.)

    # this model is near-deterministic, so the log density of a sample will
    # be high (about 35) if computed using the correct model, and very low
    # (approximately -1e10) if there's an off-by-one-timestep error.
    lp_ = self.evaluate(model.log_prob(sample_))
    self.assertGreater(lp_, 0.)

  def test_time_varying_noise(self):

    num_timesteps = 5
    prior_mean = 0.
    prior_scale = 1.

    # Define time-varying transition and observation noise models.
    def transition_noise(t):
      t = tf.cast(t, tf.float32)
      return tfd.MultivariateNormalDiag(scale_diag=[t])

    def observation_noise(t):
      t = tf.cast(t, tf.float32)
      return tfd.MultivariateNormalDiag(scale_diag=[tf.log(t+1.)])

    model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=[[1.]],
        transition_noise=transition_noise,
        observation_matrix=[[1.]],
        observation_noise=observation_noise,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[prior_mean], scale_diag=[prior_scale]))

    variance_ = self.evaluate(model.variance())

    # Manually compute the prior variance at each timestep.
    latent_variances = [prior_scale**2]
    for t in range(0, num_timesteps-1):
      latent_variances.append(latent_variances[t] + t**2)
    observation_variances = [latent_variances[t] + np.log(t+1)**2
                             for t in range(num_timesteps)]

    self.assertAllClose(observation_variances, variance_[..., 0])


@tfe.run_all_tests_in_graph_and_eager_modes
class BatchTest(tf.test.TestCase):
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
                           num_timesteps,
                           latent_size,
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

    (posterior_means, posterior_covs) = model.posterior_marginals(y)
    self.assertEqual(posterior_means.shape.as_list(),
                     sample_shape + batch_shape + [num_timesteps, latent_size])
    self.assertEqual(posterior_covs.shape.as_list(),
                     batch_shape + [num_timesteps, latent_size, latent_size])

    # Try an argument with no batch shape to ensure we broadcast
    # correctly.
    unbatched_y = tf.random_normal(event_shape)
    lp = model.log_prob(unbatched_y)
    self.assertEqual(lp.shape.as_list(), batch_shape)

    self.assertEqual(model.mean().shape.as_list(),
                     batch_shape + event_shape)
    self.assertEqual(model.variance().shape.as_list(),
                     batch_shape + event_shape)

    (posterior_means, posterior_covs) = model.posterior_marginals(unbatched_y)
    self.assertEqual(posterior_means.shape.as_list(),
                     batch_shape + [num_timesteps, latent_size])
    self.assertEqual(posterior_covs.shape.as_list(),
                     batch_shape + [num_timesteps, latent_size, latent_size])

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
    self._sanity_check_shapes(model, batch_shape, event_shape,
                              num_timesteps, latent_size)

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
    self._sanity_check_shapes(model, batch_shape, event_shape,
                              num_timesteps, latent_size)

    # Test batching only over the transition op
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     transition_matrix_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape,
                              num_timesteps, latent_size)

    # Test batching only over the transition noise
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     transition_noise_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape,
                              num_timesteps, latent_size)

    # Test batching only over the observation op
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     observation_matrix_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape,
                              num_timesteps, latent_size)

    # Test batching only over the observation noise
    model = self._build_random_model(num_timesteps,
                                     latent_size,
                                     observation_size,
                                     observation_noise_batch_shape=batch_shape)
    self._sanity_check_shapes(model, batch_shape, event_shape,
                              num_timesteps, latent_size)


@tfe.run_all_tests_in_graph_and_eager_modes
class KalmanSmootherTest(tf.test.TestCase):

  def build_kf(self):
    # Define a simple model with 3D latents and 2D observations.

    self.transition_matrix = np.array(
        [[1., 0.5, 0.], [-0.2, 0.3, 0.], [0.01, 0.02, 0.97]], dtype=np.float32)

    self.transition_noise = tfd.MultivariateNormalDiag(
        loc=np.array([-4.3, 0.9, 0.], dtype=np.float32),
        scale_diag=np.array([1., 1., 0.5], dtype=np.float32))

    self.observation_matrix = np.array(
        [[1., 1., 0.2], [0.3, -0.7, -0.2]], dtype=np.float32)
    self.observation_noise = tfd.MultivariateNormalDiag(
        loc=np.array([-0.9, 0.1], dtype=np.float32),
        scale_diag=np.array([0.3, 0.1], dtype=np.float32))

    self.initial_state_prior = tfd.MultivariateNormalDiag(
        loc=np.zeros(shape=[3,], dtype=np.float32),
        scale_diag=np.ones(shape=[3,], dtype=np.float32))

    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=5,
        transition_matrix=self.transition_matrix,
        transition_noise=self.transition_noise,
        observation_matrix=self.observation_matrix,
        observation_noise=self.observation_noise,
        initial_state_prior=self.initial_state_prior,
        initial_step=0)

  def testKalmanSmoother(self):
    obs = np.array(
        [[[1.36560337, 0.28252135],
          [-0.44638565, -0.76692033],
          [0.43440145, -1.65087236],
          [-0.96462844, -0.29173164],
          [-0.46593086, 0.23341251]]],
        dtype=np.float32)

    kf = self.build_kf()
    _, filtered_means, filtered_covs, _, _, _, _ = kf.forward_filter(obs)
    smoothed_means, smoothed_covs = kf.posterior_marginals(obs)

    # Numbers are checked against results from well-tested open source package.
    # In order to replicate the numbers below, one could run the following
    # script with PyKalman installed. https://pykalman.github.io with v.0.9.2.
    # """
    # import numpy as np
    # import pykalman
    # kf = pykalman.KalmanFilter(
    #     transition_matrices=np.array(
    #         [[1., 0.5, 0.], [-0.2, 0.3, 0.], [0.01, 0.02, 0.97]],
    # .       dtype=np.float32),
    #     observation_matrices=np.array(
    #         [[1., 1., 0.2], [0.3, -0.7, -0.2]], dtype=np.float32),
    #     transition_covariance=np.diag(np.square([1., 1., 0.5])),
    #     observation_covariance=np.diag(np.square([0.3, 0.1])),
    #     transition_offsets=np.array([-4.3, 0.9, 0.], dtype=np.float32),
    #     observation_offsets=np.array([-0.9, 0.1], dtype=np.float32),
    #     initial_state_mean=np.zeros(shape=[3,], dtype=np.float32),
    #     initial_state_covariance=np.diag(np.ones(shape=[3,],
    # .                                            dtype=np.float32)),
    #     n_dim_state=3, n_dim_obs=2)
    # x = np.array([[1.36560337, 0.28252135],
    #               [-0.44638565, -0.76692033],
    #               [0.43440145, -1.65087236],
    #               [-0.96462844, -0.29173164],
    #               [-0.46593086, 0.23341251]],
    #              dtype=np.float32)
    # filtered_means, filtered_covs = kf.filter(x)
    # smoothed_means, smoothed_covs = kf.smooth(x)
    # """

    self.assertAllClose(self.evaluate(filtered_means),
                        [[[1.67493705, 0.46825252, 0.02124943],
                          [-0.64631546, 1.00897487, -0.09965568],
                          [-1.01912747, 2.20042742, -0.35873311],
                          [-0.67203603, 0.65843169, -1.13269043],
                          [0.08385944, 0.50706669, -2.05841075]]])
    self.assertAllClose(self.evaluate(filtered_covs),
                        [[[0.05451537, -0.00583471, 0.05521206],
                          [-0.00583471, 0.07889925, -0.23913612],
                          [0.05521206, -0.23913612, 0.93451188]],
                         [[0.05475972, -0.00706799, 0.05972831],
                          [-0.00706799, 0.08838377, -0.27438752],
                          [0.05972831, -0.27438752, 1.06529626]],
                         [[0.05507039, -0.00857061, 0.06554467],
                          [-0.00857061, 0.09565483, -0.30253237],
                          [0.06554467, -0.30253237, 1.17423936]],
                         [[0.05534107, -0.00984834, 0.07049446],
                          [-0.00984834, 0.10168645, -0.3258982],
                          [0.07049446, -0.3258982, 1.26475611]],
                         [[0.05556491, -0.01090359, 0.07458252],
                          [-0.01090359, 0.10666106, -0.34516996],
                          [0.07458252, -0.34516996, 1.33941529]]])
    self.assertAllClose(self.evaluate(smoothed_means),
                        [[[1.6779677, 0.85140403, -1.35974017],
                          [-0.56246908, 1.46082297, -1.62395504],
                          [-0.90536, 2.63540628, -1.83427299],
                          [-0.47239553, 0.95851585, -2.01734974],
                          [0.08385944, 0.50706669, -2.05841075]]])
    self.assertAllClose(self.evaluate(smoothed_covs),
                        [[[0.05213916, -0.00658443, 0.05523982],
                          [-0.00658443, 0.07103678, -0.21066964],
                          [0.05523982, -0.21066964, 0.82790034]],
                         [[0.05249696, -0.00812691, 0.06099242],
                          [-0.00812691, 0.0799351, -0.24409068],
                          [0.06099242, -0.24409068, 0.95324973]],
                         [[0.05297552, -0.01009223, 0.06865306],
                          [-0.01009223, 0.08801685, -0.27559063],
                          [0.06865306, -0.27559063, 1.07602637]],
                         [[0.05343939, -0.0120551, 0.07628306],
                          [-0.0120551, 0.09641572, -0.30821036],
                          [0.07628306, -0.30821036, 1.20272402]],
                         [[0.05556491, -0.01090359, 0.07458252],
                          [-0.01090359, 0.10666106, -0.34516996],
                          [0.07458252, -0.34516996, 1.33941529]]])


@tfe.run_all_tests_in_graph_and_eager_modes
class _KalmanStepsTest(object):

  def setUp(self):
    # Define a simple model with 2D latents and 1D observations.

    self.transition_matrix = np.asarray([[1., .5], [-.2, .3]], dtype=np.float32)
    self.get_transition_matrix_for_timestep = (
        lambda t: tfl.LinearOperatorFullMatrix(self.transition_matrix))

    self.bias = np.asarray([-4.3, .9], dtype=np.float32)
    self.get_transition_noise_for_timestep = (
        lambda t: tfd.MultivariateNormalDiag(self.bias, [1., 1.]))

    self.observation_matrix = np.asarray([[1., 1.], [0.3, -0.7]],
                                         dtype=np.float32)
    self.get_observation_matrix_for_timestep = (
        lambda t: tfl.LinearOperatorFullMatrix(self.observation_matrix))

    self.observation_bias = np.asarray([-.9, .1], dtype=np.float32)
    self.observation_noise_scale_diag = np.asarray([1., 0.3], dtype=np.float32)
    def get_observation_noise_for_timestep(t):
      del t  # unused
      return tfd.MultivariateNormalDiag(
          loc=self.observation_bias,
          scale_diag=self.observation_noise_scale_diag)
    self.get_observation_noise_for_timestep = get_observation_noise_for_timestep

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
    expected_filtered_mean = [[1.41887522], [-2.82712531]]
    expected_filtered_cov = [[0.27484685, 0.05869688],
                             [0.06994689, 0.1369969]]
    expected_predicted_mean = [[-4.29468775], [-0.23191273]]
    expected_predicted_cov = [[1.37341797, -0.01930545],
                              [-0.02380546, 1.01560497]]
    expected_observation_mean = [[-2.5], [-0.77999997]]
    expected_observation_cov = [[2.4000001, -0.28000003],
                                [-0.28000003, 0.366]]
    expected_log_marginal_likelihood = -56.5381

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

  def testLinearGaussianObservation(self):
    prior_mean = np.asarray([[-2], [.4]], dtype=np.float32)
    prior_cov = np.asarray([[.5, -.2], [-.2, .9]], dtype=np.float32)
    x_observed = np.asarray([[4.], [-1.]], dtype=np.float32)

    prior_mean_tensor = self.build_tensor(prior_mean)
    prior_cov_tensor = self.build_tensor(prior_cov)
    x_observed_tensor = self.build_tensor(x_observed)

    observation_matrix = self.get_observation_matrix_for_timestep(0)
    observation_noise = self.get_observation_noise_for_timestep(0)
    (posterior_mean,
     posterior_cov,
     predictive_dist) = linear_gaussian_update(
         prior_mean_tensor, prior_cov_tensor,
         observation_matrix, observation_noise,
         x_observed_tensor)

    expected_posterior_mean = [[-0.373276], [1.65086186]]
    expected_posterior_cov = [[0.24379307, 0.02689655],
                              [0.02689655, 0.13344827]]
    expected_predicted_mean = [-2.5, -0.77999997]
    expected_predicted_cov = [[1.99999988, -0.39999998],
                              [-0.39999998, 0.65999997]]

    self.assertAllClose(self.evaluate(posterior_mean),
                        expected_posterior_mean)
    self.assertAllClose(self.evaluate(posterior_cov),
                        expected_posterior_cov)
    self.assertAllClose(self.evaluate(predictive_dist.mean()),
                        expected_predicted_mean)
    self.assertAllClose(self.evaluate(predictive_dist.covariance()),
                        expected_predicted_cov)

  def testBackwardSmoothingStep(self):
    filtered_mean = [[2.], [-3.]]
    filtered_cov = [[1.2, 0.4],
                    [0.4, 2.3]]
    predicted_mean = [[2.1], [-2.7]]
    predicted_cov = [[1.1, 0.5],
                     [0.5, 2.]]
    next_smoothed_mean = [[1.9], [-2.9]]
    next_smoothed_cov = [[1.4, 0.4],
                         [0.4, 2.1]]
    transition_matrix = [[0.6, 0.3],
                         [0.4, 0.7]]

    filtered_mean = self.build_tensor(filtered_mean)
    filtered_cov = self.build_tensor(filtered_cov)
    predicted_mean = self.build_tensor(predicted_mean)
    predicted_cov = self.build_tensor(predicted_cov)
    next_smoothed_mean = self.build_tensor(next_smoothed_mean)
    next_smoothed_cov = self.build_tensor(next_smoothed_cov)
    get_transition_matrix_for_timestep = (
        lambda t: tfl.LinearOperatorFullMatrix(transition_matrix))
    transition_matrix = get_transition_matrix_for_timestep(0)

    posterior_mean, posterior_cov = backward_smoothing_update(
        filtered_mean, filtered_cov,
        predicted_mean, predicted_cov,
        next_smoothed_mean, next_smoothed_cov,
        transition_matrix)

    # The expected results are calculated by analytical calculation.
    self.assertAllClose(self.evaluate(posterior_mean),
                        [[1.824], [-3.252]])
    self.assertAllClose(self.evaluate(posterior_cov),
                        [[1.30944, 0.45488],
                         [0.45488, 2.35676]])

  def testBackwardPassStep(self):
    filtered_mean = [[2.], [-3.]]
    filtered_cov = [[1.2, 0.4],
                    [0.4, 2.3]]
    predicted_mean = [[2.1], [-2.7]]
    predicted_cov = [[1.1, 0.5],
                     [0.5, 2.]]
    next_smoothed_mean = [[1.9], [-2.9]]
    next_smoothed_cov = [[1.4, 0.4],
                         [0.4, 2.1]]
    transition_matrix = [[0.6, 0.3],
                         [0.4, 0.7]]

    filtered_mean = self.build_tensor(filtered_mean)
    filtered_cov = self.build_tensor(filtered_cov)
    predicted_mean = self.build_tensor(predicted_mean)
    predicted_cov = self.build_tensor(predicted_cov)
    next_smoothed_mean = self.build_tensor(next_smoothed_mean)
    next_smoothed_cov = self.build_tensor(next_smoothed_cov)
    get_transition_matrix_for_timestep = (
        lambda t: tfl.LinearOperatorFullMatrix(transition_matrix))

    smooth_step = build_backward_pass_step(
        get_transition_matrix_for_timestep)

    initial_backward_state = BackwardPassState(
        backward_mean=next_smoothed_mean,
        backward_cov=next_smoothed_cov,
        timestep=self.build_tensor(0))

    smoothed_state = self.evaluate(
        smooth_step(initial_backward_state,
                    [filtered_mean,
                     filtered_cov,
                     predicted_mean,
                     predicted_cov]))

    expected_posterior_mean = [[1.824], [-3.252]]
    expected_posterior_cov = [[1.30944, 0.45488],
                              [0.45488, 2.35676]]

    self.assertAllClose(smoothed_state.backward_mean,
                        expected_posterior_mean)
    self.assertAllClose(smoothed_state.backward_cov,
                        expected_posterior_cov)
    self.assertAllClose(smoothed_state.timestep,
                        -1)

  def testLinearGaussianObservationScalarPath(self):

    # Construct observed data with a scalar observation.
    prior_mean_tensor = self.build_tensor(
        np.asarray([[-2], [.4]], dtype=np.float32))
    prior_cov_tensor = self.build_tensor(
        np.asarray([[.5, -.2], [-.2, .9]], dtype=np.float32))
    x_observed_tensor = self.build_tensor(
        np.asarray([[4.]], dtype=np.float32))

    observation_matrix = tfl.LinearOperatorFullMatrix(
        self.build_tensor([[1., 1.]]))
    observation_noise = tfd.MultivariateNormalDiag(
        loc=self.build_tensor([-.9]), scale_diag=self.build_tensor([1.]))

    (posterior_mean,
     posterior_cov,
     predictive_dist) = linear_gaussian_update(
         prior_mean_tensor, prior_cov_tensor,
         observation_matrix, observation_noise,
         x_observed_tensor)

    # Ensure we take the scalar-optimized path when shape is static.
    if self.use_static_shape or tf.executing_eagerly():
      self.assertIsInstance(predictive_dist, tfd.Independent)
    else:
      self.assertIsInstance(predictive_dist, tfd.MultivariateNormalTriL)
    self.assertAllEqual(
        self.evaluate(predictive_dist.event_shape_tensor()), [1])
    self.assertAllEqual(
        self.evaluate(predictive_dist.batch_shape_tensor()), [])

    # Extend `x_observed` with an extra dimension to force the vector path.
    # The added dimension is non-informative, so we can check that the scalar
    # and vector paths yield the same posterior.
    observation_matrix_extended = tfl.LinearOperatorFullMatrix(
        self.build_tensor([[1., 1.], [0., 0.]]))
    observation_noise_extended = tfd.MultivariateNormalDiag(
        loc=self.build_tensor([-.9, 0.]),
        scale_diag=self.build_tensor([1., 1e15]))
    x_observed_extended_tensor = self.build_tensor(
        np.asarray([[4.], [0.]], dtype=np.float32))

    (posterior_mean_extended,
     posterior_cov_extended,
     predictive_dist_extended) = linear_gaussian_update(
         prior_mean_tensor, prior_cov_tensor,
         observation_matrix_extended, observation_noise_extended,
         x_observed_extended_tensor)

    # Ensure we took the vector path.
    self.assertIsInstance(predictive_dist_extended, tfd.MultivariateNormalTriL)
    self.assertAllEqual(
        self.evaluate(predictive_dist_extended.event_shape_tensor()), [2])
    self.assertAllEqual(
        self.evaluate(predictive_dist_extended.batch_shape_tensor()), [])

    # Test that the results are the same.
    self.assertAllClose(*self.evaluate((posterior_mean,
                                        posterior_mean_extended)))
    self.assertAllClose(*self.evaluate((posterior_cov,
                                        posterior_cov_extended)))
    self.assertAllClose(*self.evaluate((predictive_dist.mean(),
                                        predictive_dist_extended.mean()[:1])))
    self.assertAllClose(
        *self.evaluate((predictive_dist.covariance(),
                        predictive_dist_extended.covariance()[:1, :1])))

  def testMeanStep(self):
    prev_mean = np.asarray([[-2], [.4]], dtype=np.float32)
    prev_mean_tensor = self.build_tensor(prev_mean)

    mean_step = build_kalman_mean_step(
        self.get_transition_matrix_for_timestep,
        self.get_transition_noise_for_timestep,
        self.get_observation_matrix_for_timestep,
        self.get_observation_noise_for_timestep)
    new_mean, obs_mean = self.evaluate(mean_step((prev_mean_tensor, None), t=0))

    self.assertAllClose(new_mean,
                        np.dot(self.transition_matrix, prev_mean) +
                        self.bias[:, np.newaxis])
    self.assertAllClose(obs_mean,
                        np.dot(self.observation_matrix, new_mean) +
                        self.observation_bias[:, np.newaxis])

  def testCovStep(self):

    prev_cov = np.asarray([[.5, -.2], [-.2, .9]], dtype=np.float32)
    prev_cov_tensor = self.build_tensor(prev_cov)

    cov_step = build_kalman_cov_step(
        self.get_transition_matrix_for_timestep,
        self.get_transition_noise_for_timestep,
        self.get_observation_matrix_for_timestep,
        self.get_observation_noise_for_timestep)
    new_cov, obs_cov = self.evaluate(cov_step((prev_cov_tensor, None), t=0))

    self.assertAllClose(new_cov,
                        np.dot(self.transition_matrix,
                               np.dot(prev_cov,
                                      self.transition_matrix.T)) + np.eye(2))
    self.assertAllClose(obs_cov,
                        np.dot(self.observation_matrix,
                               np.dot(new_cov, self.observation_matrix.T)) +
                        np.diag(self.observation_noise_scale_diag**2))


@tfe.run_all_tests_in_graph_and_eager_modes
class KalmanStepsTestStatic(tf.test.TestCase, _KalmanStepsTest):

  use_static_shape = True

  def setUp(self):
    return _KalmanStepsTest.setUp(self)

  def build_tensor(self, tensor):
    return tf.convert_to_tensor(tensor)


@tfe.run_all_tests_in_graph_and_eager_modes
class KalmanStepsTestDynamic(tf.test.TestCase, _KalmanStepsTest):

  use_static_shape = False

  def setUp(self):
    return _KalmanStepsTest.setUp(self)

  def build_tensor(self, tensor):
    return tf.placeholder_with_default(input=tf.convert_to_tensor(tensor),
                                       shape=None)


@tfe.run_all_tests_in_graph_and_eager_modes
class _AugmentSampleShapeTest(object):

  def testAugmentsShape(self):

    full_batch_shape, dist = self.build_inputs([5, 4, 2, 3], [2, 3])

    sample_shape = _augment_sample_shape(dist, full_batch_shape,
                                         validate_args=True)

    self.assertAllEqual(self.maybe_evaluate(sample_shape), [5, 4])

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

    with self.assertRaisesError(
        "(Broadcasting is not supported|Cannot broadcast)"):
      self.maybe_evaluate(
          _augment_sample_shape(dist, full_batch_shape,
                                validate_args=True))


@tfe.run_all_tests_in_graph_and_eager_modes
class AugmentSampleShapeTestStatic(tf.test.TestCase, _AugmentSampleShapeTest):

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def build_inputs(self, full_batch_shape, partial_batch_shape):

    full_batch_shape = np.asarray(full_batch_shape, dtype=np.int32)
    dist = tfd.Normal(tf.random_normal(partial_batch_shape), 1.)

    return full_batch_shape, dist

  def maybe_evaluate(self, x):
    return x


@tfe.run_all_tests_in_graph_and_eager_modes
class AugmentSampleShapeTestDynamic(tf.test.TestCase, _AugmentSampleShapeTest):

  def assertRaisesError(self, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(Exception, msg)
    else:
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
  tf.test.main()
