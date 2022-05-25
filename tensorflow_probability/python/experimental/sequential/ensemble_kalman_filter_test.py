# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for the Ensemble Kalman Filter."""

import collections

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfs = tfp.experimental.sequential

NUMPY_MODE = False


@test_util.test_all_tf_execution_regimes
class EnsembleKalmanFilterTest(test_util.TestCase):

  def test_ensemble_kalman_filter_expect_mvn(self):

    state = tfs.EnsembleKalmanFilterState(step=0, particles=[1.], extra=None)

    with self.assertRaises(ValueError):
      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=[0.],
          observation_fn=lambda t, p, e: (tfd.Poisson(rate=p), e))

  def test_ensemble_kalman_filter_constant_univariate_shapes(self):
    # Simple transition model where the state doesn't change,
    # so we are estimating a constant.

    def transition_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-11]), extra

    def observation_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(loc=particles, scale_diag=[1e-2]), extra

    # Initialize the ensemble.
    particles = self.evaluate(
        tf.random.normal(shape=[100, 1], seed=test_util.test_seed()))

    state = tfs.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={'unchanged': 1})

    predicted_state = tfs.ensemble_kalman_filter_predict(
        state,
        transition_fn=transition_fn,
        inflate_fn=None,
        seed=test_util.test_seed())

    observation = tf.convert_to_tensor([0.], dtype=particles.dtype)
    log_ml = tfs.ensemble_kalman_filter_log_marginal_likelihood(
        predicted_state,
        observation=observation,
        observation_fn=observation_fn,
        seed=test_util.test_seed())
    self.assertAllEqual(observation.shape[:-1], log_ml.shape)
    log_ml_krazy_obs = tfs.ensemble_kalman_filter_log_marginal_likelihood(
        predicted_state,
        observation=observation + 10.,
        observation_fn=observation_fn,
        seed=test_util.test_seed())
    self.assertAllLess(log_ml_krazy_obs, log_ml)

    # Check that extra is correctly propagated.
    self.assertIn('unchanged', predicted_state.extra)
    self.assertEqual(1, predicted_state.extra['unchanged'])

    # Check that the state respected the constant dynamics.
    self.assertAllClose(state.particles, predicted_state.particles)

    updated_state = tfs.ensemble_kalman_filter_update(
        predicted_state,
        # The observation is the constant 0.
        observation=observation,
        seed=test_util.test_seed(),
        observation_fn=observation_fn)

    # Check that extra is correctly propagated.
    self.assertIn('unchanged', updated_state.extra)
    self.assertEqual(1, updated_state.extra['unchanged'])
    self.assertAllEqual(state.particles.shape, updated_state.particles.shape)

  def test_ensemble_kalman_filter_linear_model(self):
    # Simple transition model where the state doesn't change,
    # so we are estimating a constant.

    def transition_fn(_, particles, extra):
      particles = {
          'x': particles['x'] + particles['xdot'],
          'xdot': particles['xdot']
      }
      extra['transition_count'] += 1
      return tfd.JointDistributionNamed(
          dict(
              x=tfd.MultivariateNormalDiag(
                  loc=particles['x'], scale_diag=[1e-11]),
              xdot=tfd.MultivariateNormalDiag(
                  particles['xdot'], scale_diag=[1e-11]))), extra

    def observation_fn(_, particles, extra):
      extra['observation_count'] += 1
      return tfd.MultivariateNormalDiag(
          loc=particles['x'], scale_diag=[1e-2]), extra

    seed_stream = test_util.test_seed_stream()

    # Initialize the ensemble.
    particles = {
        'x':
            self.evaluate(
                tf.random.normal(shape=[300, 5, 1], seed=seed_stream())),
        'xdot':
            self.evaluate(
                tf.random.normal(shape=[300, 5, 1], seed=seed_stream()))
    }

    state = tfs.EnsembleKalmanFilterState(
        step=0,
        particles=particles,
        extra={
            'observation_count': 0,
            'transition_count': 0
        })

    for i in range(5):
      state = tfs.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)
      self.assertIn('transition_count', state.extra)
      self.assertEqual(i + 1, state.extra['transition_count'])

      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=[1. * i],
          observation_fn=observation_fn,
          seed=seed_stream())

      self.assertIn('observation_count', state.extra)
      self.assertEqual(i + 1, state.extra['observation_count'])

    self.assertAllClose(
        [4.] * 5,
        self.evaluate(tf.reduce_mean(state.particles['x'], axis=[0, -1])),
        rtol=0.05)

  def test_ensemble_kalman_filter_constant_model_multivariate(self):

    def transition_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-11] * 2), extra

    def observation_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-1] * 2), extra

    seed_stream = test_util.test_seed_stream()

    # Initialize the ensemble.
    particles = self.evaluate(
        tf.random.normal(
            shape=[300, 3, 2], seed=seed_stream(), dtype=tf.float64))

    state = tfs.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={'unchanged': 1})

    for _ in range(8):
      state = tfs.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)

      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=[0., 0.],
          observation_fn=observation_fn,
          seed=seed_stream())

    self.assertAllClose(
        [[0., 0.]] * 3,
        self.evaluate(tf.reduce_mean(state.particles, axis=0)),
        atol=1e-2)

  def test_ensemble_kalman_filter_linear_model_multivariate(self):

    def transition_fn(_, particles, extra):
      particles = {
          'x': particles['x'] + particles['xdot'],
          'xdot': particles['xdot']
      }
      extra['transition_count'] += 1
      return tfd.JointDistributionNamed(
          dict(
              x=tfd.MultivariateNormalDiag(
                  particles['x'], scale_diag=[1e-11] * 2),
              xdot=tfd.MultivariateNormalDiag(
                  particles['xdot'], scale_diag=[1e-11] * 2))), extra

    def observation_fn(_, particles, extra):
      extra['observation_count'] += 1
      return tfd.MultivariateNormalDiag(
          loc=particles['x'], scale_diag=[1e-2] * 2), extra

    seed_stream = test_util.test_seed_stream()

    # Initialize the ensemble.
    particles_shape = (300, 3, 2)
    particles = {
        'x':
            self.evaluate(
                tf.random.normal(
                    shape=particles_shape, seed=seed_stream(),
                    dtype=tf.float64)),
        'xdot':
            self.evaluate(
                tf.random.normal(
                    shape=particles_shape, seed=seed_stream(),
                    dtype=tf.float64))
    }

    state = tfs.EnsembleKalmanFilterState(
        step=0,
        particles=particles,
        extra={
            'observation_count': 0,
            'transition_count': 0
        })

    for i in range(10):
      # Predict.
      state = tfs.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)
      self.assertIn('transition_count', state.extra)
      self.assertEqual(i + 1, state.extra['transition_count'])

      # Marginal likelihood.
      observation = tf.convert_to_tensor([1. * i, 2. * i], dtype=tf.float64)
      log_ml = tfs.ensemble_kalman_filter_log_marginal_likelihood(
          state,
          observation=observation,
          observation_fn=observation_fn,
          seed=test_util.test_seed())
      self.assertAllEqual(particles_shape[1:-1], log_ml.shape)
      self.assertIn('observation_count', state.extra)
      self.assertEqual(3 * i + 1, state.extra['observation_count'])

      log_ml_krazy_obs = tfs.ensemble_kalman_filter_log_marginal_likelihood(
          state,
          observation=observation + 10.,
          observation_fn=observation_fn,
          seed=test_util.test_seed())
      np.testing.assert_array_less(
          self.evaluate(log_ml_krazy_obs), self.evaluate(log_ml))
      self.assertEqual(3 * i + 2, state.extra['observation_count'])

      # Update.
      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=observation,
          observation_fn=observation_fn,
          seed=seed_stream())
      print(self.evaluate(tf.reduce_mean(state.particles['x'], axis=0)))
      self.assertEqual(3 * i + 3, state.extra['observation_count'])

    self.assertAllClose(
        [[9., 18.]] * 3,
        self.evaluate(tf.reduce_mean(state.particles['x'], axis=0)),
        rtol=0.05)


# Parameters defining a linear/Gaussian state space model.
LinearModelParams = collections.namedtuple('LinearModelParams', [
    'dtype',
    'n_states',
    'n_observations',
    'prior_mean',
    'prior_cov',
    'transition_mat',
    'observation_mat',
    'transition_cov',
    'observation_noise_cov',
])

# Parameters specific to an EnKF. Used together with LinearModelParams.
EnKFParams = collections.namedtuple(
    'EnKFParams', ['n_ensemble', 'state', 'observation_fn', 'transition_fn'])


@test_util.test_all_tf_execution_regimes
class KalmanFilterVersusEnKFTest(test_util.TestCase):
  """Compare KF to EnKF with large ensemble sizes.

  If the model is linear and Gaussian the EnKF sample mean/cov and marginal
  likelihood converges to that of a KF in the large ensemble limit.

  This class tests that they are the same. It does that by implementing a
  one-step KF. It also does some simple checks on the KF, to make sure we didn't
  just replicate misunderstanding in the EnKF.
  """

  def _random_spd_matrix(self, n, noise_level, seed, dtype):
    """Random SPD matrix with inflated diagonal."""
    wigner_mat = (
        tf.random.normal(shape=[n, n], seed=seed, dtype=dtype) /
        tf.sqrt(tf.cast(n, dtype)))
    eye = tf.linalg.eye(n, dtype=dtype)
    return noise_level**2 * (
        tf.linalg.matmul(wigner_mat, wigner_mat, adjoint_b=True) + 0.5 * eye)

  def _get_linear_model_params(
      self,
      noise_level,
      n_states,
      n_observations,
      seed_stream,
      dtype,
  ):
    """Get parameters defining a linear state space model (for KF & EnKF)."""

    def _normal(shape):
      return tf.random.normal(shape, seed=seed_stream(), dtype=dtype)

    def _uniform(shape):
      return tf.random.uniform(
          # Setting minval > 0 helps test with rtol.
          shape,
          minval=1.0,
          maxval=2.0,
          seed=seed_stream(),
          dtype=dtype)

    return LinearModelParams(
        dtype=dtype,
        n_states=n_states,
        n_observations=n_observations,
        prior_mean=_uniform([n_states]),
        prior_cov=self._random_spd_matrix(
            n_states, 1.0, seed_stream(), dtype=dtype),
        transition_mat=_normal([n_states, n_states]),
        observation_mat=_normal([n_observations, n_states]),
        transition_cov=self._random_spd_matrix(
            n_states, noise_level, seed_stream(), dtype=dtype),
        observation_noise_cov=self._random_spd_matrix(
            n_observations, noise_level, seed_stream(), dtype=dtype),
    )

  def _kalman_filter_solve(self, observation, linear_model_params):
    """Solve one assimilation step using a KF."""
    # See http://screen/tnjSAEuo5nPKmYt for equations.
    # pylint: disable=unnecessary-lambda
    p = linear_model_params  # Simple & Sweet

    # With A, B matrices and x a vector, we define the operations...
    a_x = lambda a, x: tf.linalg.matvec(a, x)  # Ax
    a_b = lambda a, b: tf.linalg.matmul(a, b)  # AB
    a_bt = lambda a, b: tf.linalg.matmul(a, b, adjoint_b=True)  # ABᵀ
    a_b_at = lambda c, d: a_b(c, a_bt(d, c))  # ABAᵀ

    predictive_mean = a_x(p.transition_mat, p.prior_mean)
    predictive_cov = a_b_at(p.transition_mat, p.prior_cov) + p.transition_cov

    kalman_gain = a_b(
        a_bt(predictive_cov, p.observation_mat),
        tf.linalg.inv(
            a_b_at(p.observation_mat, predictive_cov) +
            p.observation_noise_cov))
    updated_mean = (
        predictive_mean +
        a_x(kalman_gain, observation - a_x(p.observation_mat, predictive_mean)))
    updated_cov = a_b(
        tf.linalg.eye(p.n_states, dtype=p.dtype) -
        a_b(kalman_gain, p.observation_mat), predictive_cov)

    # p(Y | X_{predictive})
    marginal_dist = tfd.MultivariateNormalTriL(
        loc=a_x(p.observation_mat, predictive_mean),
        scale_tril=tf.linalg.cholesky(
            a_b_at(p.observation_mat, predictive_cov) +
            p.observation_noise_cov),
    )

    return dict(
        predictive_mean=predictive_mean,
        predictive_cov=predictive_cov,
        predictive_stddev=tf.sqrt(tf.linalg.diag_part(predictive_cov)),
        updated_mean=updated_mean,
        updated_cov=updated_cov,
        updated_stddev=tf.sqrt(tf.linalg.diag_part(updated_cov)),
        log_marginal_likelihood=marginal_dist.log_prob(observation),
    )
    # pylint: enable=unnecessary-lambda

  def _get_enkf_params(
      self,
      n_ensemble,
      linear_model_params,
      prior_dist,
      seed_stream,
      dtype,
  ):
    """Get parameters specific to EnKF reconstructions."""
    particles = prior_dist.sample(n_ensemble, seed=seed_stream())
    state = tfs.EnsembleKalmanFilterState(step=0, particles=particles, extra={})

    def observation_fn(_, particles, extra):
      observation_particles_dist = tfd.MultivariateNormalTriL(
          loc=tf.linalg.matvec(linear_model_params.observation_mat, particles),
          scale_tril=tf.linalg.cholesky(
              linear_model_params.observation_noise_cov))
      return observation_particles_dist, extra

    def transition_fn(_, particles, extra):
      new_particles_dist = tfd.MultivariateNormalTriL(
          loc=tf.linalg.matvec(linear_model_params.transition_mat, particles),
          scale_tril=tf.linalg.cholesky(linear_model_params.transition_cov))
      return new_particles_dist, extra

    return EnKFParams(
        state=state,
        n_ensemble=n_ensemble,
        observation_fn=observation_fn,
        transition_fn=transition_fn,
    )

  def _enkf_solve(self, observation, enkf_params, predict_kwargs, update_kwargs,
                  log_marginal_likelihood_kwargs, seed_stream):
    """Solve one data assimilation step using an EnKF."""
    predicted_state = tfs.ensemble_kalman_filter_predict(
        enkf_params.state,
        enkf_params.transition_fn,
        seed=seed_stream(),
        **predict_kwargs)
    updated_state = tfs.ensemble_kalman_filter_update(
        predicted_state,
        observation,
        enkf_params.observation_fn,
        seed=seed_stream(),
        **update_kwargs)
    log_marginal_likelihood = tfs.ensemble_kalman_filter_log_marginal_likelihood(
        predicted_state,
        observation,
        enkf_params.observation_fn,
        seed=seed_stream(),
        **log_marginal_likelihood_kwargs)

    return dict(
        predictive_mean=tf.reduce_mean(predicted_state.particles, axis=0),
        predictive_cov=tfp.stats.covariance(predicted_state.particles),
        predictive_stddev=tfp.stats.stddev(predicted_state.particles),
        updated_mean=tf.reduce_mean(updated_state.particles, axis=0),
        updated_cov=tfp.stats.covariance(updated_state.particles),
        updated_stddev=tfp.stats.stddev(updated_state.particles),
        log_marginal_likelihood=log_marginal_likelihood,
    )

  @test_combinations.generate(
      test_combinations.combine(
          noise_level=[0.001, 0.1, 1.0],
          n_states=[2, 5],
          n_observations=[2, 5],
      ))
  def test_same_solution(self, noise_level, n_states, n_observations):
    """Check that the KF and EnKF solutions are the same."""
    # Tests pass with n_ensemble = 1e7. The KF vs. EnKF tolerance is
    # proportional to 1 / sqrt(n_ensemble), so this shows good agreement.
    n_ensemble = int(1e4) if NUMPY_MODE else int(1e6)

    salt = str(noise_level) + str(n_states) + str(n_observations)
    seed_stream = test_util.test_seed_stream(salt)
    dtype = tf.float64
    predict_kwargs = {}
    update_kwargs = {}
    log_marginal_likelihood_kwargs = {}

    linear_model_params = self._get_linear_model_params(
        noise_level=noise_level,
        n_states=n_states,
        n_observations=n_observations,
        seed_stream=seed_stream,
        dtype=dtype)

    # Ensure that our observation comes from a state that ~ prior.
    prior_dist = tfd.MultivariateNormalTriL(
        loc=linear_model_params.prior_mean,
        scale_tril=tf.linalg.cholesky(linear_model_params.prior_cov))
    true_state = prior_dist.sample(seed=seed_stream())
    observation = tf.linalg.matvec(linear_model_params.observation_mat,
                                   true_state)

    kf_soln = self._kalman_filter_solve(observation, linear_model_params)

    enkf_params = self._get_enkf_params(n_ensemble, linear_model_params,
                                        prior_dist, seed_stream, dtype)
    enkf_soln = self._enkf_solve(observation, enkf_params, predict_kwargs,
                                 update_kwargs, log_marginal_likelihood_kwargs,
                                 seed_stream)

    # In the low noise limit, the spectral norm of the posterior covariance is
    # bounded by reconstruction_tol**2.
    # http://screen/96UV8kiXMvp8QSM
    reconstruction_tol = noise_level / tf.reduce_min(
        tf.linalg.svd(linear_model_params.observation_mat, compute_uv=False))

    # Evaluate at the same time, so both use the same randomness!
    # Do not use anything that was not evaluated here!
    true_state, reconstruction_tol, kf_soln, enkf_soln = self.evaluate(
        [true_state, reconstruction_tol, kf_soln, enkf_soln])

    max_updated_scale = self.evaluate(
        tf.sqrt(
            tf.reduce_max(
                tf.linalg.svd(kf_soln['updated_cov'], compute_uv=False))))

    if noise_level < 0.2 and n_states == n_observations:
      # Check that the theoretical error bound is obeyed.
      # We use max_updated_scale below to check reconstruction error, but
      # without this check here, it's possible that max_updated_scale is large
      # due to some error in the kalman filter...which would invalidate checks
      # below.
      slop = 2. + 5 * noise_level
      self.assertLess(max_updated_scale, slop * reconstruction_tol)

    # The KF should reconstruct the correct value up to 5 stddevs.
    # The relevant stddev is that of a χ² random variable.
    reconstruction_error = np.linalg.norm(
        kf_soln['updated_mean'] - true_state, axis=-1)
    self.assertLess(reconstruction_error,
                    5 * np.sqrt(2 * n_states) * max_updated_scale)

    # We know the EnKF converges at rate 1 / Sqrt(n_ensemble). The factor in
    # front is set empirically.
    tol_scale = 1 / np.sqrt(n_ensemble)  # 1 / Sqrt(1e6) = 0.001
    self.assertAllCloseNested(
        kf_soln, enkf_soln, atol=20 * tol_scale, rtol=50 * tol_scale)


if __name__ == '__main__':
  test_util.main()
