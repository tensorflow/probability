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
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.experimental.sequential import ensemble_kalman_filter as ekf
from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.stats import sample_stats

NUMPY_MODE = False


@test_util.test_all_tf_execution_regimes
class EnsembleKalmanFilterTest(test_util.TestCase):

  def test_ensemble_kalman_filter_expect_mvn(self):

    state = ekf.EnsembleKalmanFilterState(step=0, particles=[1.], extra=None)

    with self.assertRaises(ValueError):
      state = ekf.ensemble_kalman_filter_update(
          state,
          observation=[0.],
          observation_fn=lambda t, p, e: (poisson.Poisson(rate=p), e))

  def test_ensemble_kalman_filter_constant_univariate_shapes(self):
    # Simple transition model where the state doesn't change,
    # so we are estimating a constant.

    def transition_fn(_, particles, extra):
      return mvn_diag.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-11]), extra

    def observation_fn(_, particles, extra):
      return mvn_diag.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-2]), extra

    # Initialize the ensemble.
    particles = self.evaluate(
        tf.random.normal(shape=[100, 1], seed=test_util.test_seed()))

    state = ekf.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={'unchanged': 1})

    predicted_state = ekf.ensemble_kalman_filter_predict(
        state,
        transition_fn=transition_fn,
        inflate_fn=None,
        seed=test_util.test_seed())

    observation = tf.convert_to_tensor([0.], dtype=particles.dtype)
    log_ml = ekf.ensemble_kalman_filter_log_marginal_likelihood(
        predicted_state,
        observation=observation,
        observation_fn=observation_fn,
        seed=test_util.test_seed())
    self.assertAllEqual(observation.shape[:-1], log_ml.shape)
    log_ml_krazy_obs = ekf.ensemble_kalman_filter_log_marginal_likelihood(
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

    updated_state = ekf.ensemble_kalman_filter_update(
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
      return jdn.JointDistributionNamed(
          dict(
              x=mvn_diag.MultivariateNormalDiag(
                  loc=particles['x'], scale_diag=[1e-11]),
              xdot=mvn_diag.MultivariateNormalDiag(
                  particles['xdot'], scale_diag=[1e-11]))), extra

    def observation_fn(_, particles, extra):
      extra['observation_count'] += 1
      return mvn_diag.MultivariateNormalDiag(
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

    state = ekf.EnsembleKalmanFilterState(
        step=0,
        particles=particles,
        extra={
            'observation_count': 0,
            'transition_count': 0
        })

    for i in range(5):
      state = ekf.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)
      self.assertIn('transition_count', state.extra)
      self.assertEqual(i + 1, state.extra['transition_count'])

      state = ekf.ensemble_kalman_filter_update(
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
      return mvn_diag.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-11] * 2), extra

    def observation_fn(_, particles, extra):
      return mvn_diag.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-1] * 2), extra

    seed_stream = test_util.test_seed_stream()

    # Initialize the ensemble.
    particles = self.evaluate(
        tf.random.normal(
            shape=[300, 3, 2], seed=seed_stream(), dtype=tf.float64))

    state = ekf.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={'unchanged': 1})

    for _ in range(8):
      state = ekf.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)

      state = ekf.ensemble_kalman_filter_update(
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
      return jdn.JointDistributionNamed(
          dict(
              x=mvn_diag.MultivariateNormalDiag(
                  particles['x'], scale_diag=[1e-11] * 2),
              xdot=mvn_diag.MultivariateNormalDiag(
                  particles['xdot'], scale_diag=[1e-11] * 2))), extra

    def observation_fn(_, particles, extra):
      extra['observation_count'] += 1
      return mvn_diag.MultivariateNormalDiag(
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

    state = ekf.EnsembleKalmanFilterState(
        step=0,
        particles=particles,
        extra={
            'observation_count': 0,
            'transition_count': 0
        })

    for i in range(10):
      # Predict.
      state = ekf.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)
      self.assertIn('transition_count', state.extra)
      self.assertEqual(i + 1, state.extra['transition_count'])

      # Marginal likelihood.
      observation = tf.convert_to_tensor([1. * i, 2. * i], dtype=tf.float64)
      log_ml = ekf.ensemble_kalman_filter_log_marginal_likelihood(
          state,
          observation=observation,
          observation_fn=observation_fn,
          seed=test_util.test_seed())
      self.assertAllEqual(particles_shape[1:-1], log_ml.shape)
      self.assertIn('observation_count', state.extra)
      self.assertEqual(3 * i + 1, state.extra['observation_count'])
      self.assertFalse(np.any(np.isnan(self.evaluate(log_ml))))

      log_ml_krazy_obs = ekf.ensemble_kalman_filter_log_marginal_likelihood(
          state,
          observation=observation + 10.,
          observation_fn=observation_fn,
          seed=test_util.test_seed())
      np.testing.assert_array_less(
          self.evaluate(log_ml_krazy_obs), self.evaluate(log_ml))
      self.assertEqual(3 * i + 2, state.extra['observation_count'])

      # Update.
      state = ekf.ensemble_kalman_filter_update(
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

  def test_log_marginal_likelihood_with_small_ensemble_no_perturb_obs(self):
    # With perturbed_observations=False, we should be able to handle the small
    # ensemble without NaN.

    # Initialize an ensemble with that is smaller than the event size.
    seed_stream = test_util.test_seed_stream()
    n_ensemble = 3
    event_size = 5
    self.assertLess(n_ensemble, event_size)
    particles_shape = (n_ensemble, event_size)

    particles = {
        'x':
            self.evaluate(
                tf.random.normal(shape=particles_shape, seed=seed_stream())),
    }

    def observation_fn(_, particles, extra):
      return mvn_diag.MultivariateNormalDiag(
          loc=particles['x'], scale_diag=[1e-2] * event_size), extra

    # Marginal likelihood.
    log_ml = ekf.ensemble_kalman_filter_log_marginal_likelihood(
        state=ekf.EnsembleKalmanFilterState(
            step=0, particles=particles, extra={}),
        observation=tf.random.normal(shape=(event_size,), seed=seed_stream()),
        observation_fn=observation_fn,
        perturbed_observations=False,
        seed=test_util.test_seed())
    self.assertAllEqual(particles_shape[1:-1], log_ml.shape)
    self.assertFalse(np.any(np.isnan(self.evaluate(log_ml))))


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
class ComparingMethodsTest(test_util.TestCase):
  """Compare various KF EnKF versions.

  If the model is linear and Gaussian the EnKF sample mean/cov and marginal
  likelihood converges to that of a KF in the large ensemble limit.
  This class tests that they are the same. It does that by implementing a
  one-step KF. It also does some simple checks on the KF, to make sure we didn't
  just replicate misunderstanding in the EnKF.

  This class also checks that various flavors of the EnKF are the same.
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
    marginal_dist = mvn_tril.MultivariateNormalTriL(
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
    state = ekf.EnsembleKalmanFilterState(step=0, particles=particles, extra={})

    def observation_fn(_, particles, extra):
      observation_particles_dist = mvn_tril.MultivariateNormalTriL(
          loc=tf.linalg.matvec(linear_model_params.observation_mat, particles),
          scale_tril=tf.linalg.cholesky(
              linear_model_params.observation_noise_cov))
      return observation_particles_dist, extra

    def transition_fn(_, particles, extra):
      new_particles_dist = mvn_tril.MultivariateNormalTriL(
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
    predicted_state = ekf.ensemble_kalman_filter_predict(
        enkf_params.state,
        enkf_params.transition_fn,
        seed=seed_stream(),
        **predict_kwargs)
    updated_state = ekf.ensemble_kalman_filter_update(
        predicted_state,
        observation,
        enkf_params.observation_fn,
        seed=seed_stream(),
        **update_kwargs)
    log_marginal_likelihood = ekf.ensemble_kalman_filter_log_marginal_likelihood(
        predicted_state,
        observation,
        enkf_params.observation_fn,
        seed=seed_stream(),
        **log_marginal_likelihood_kwargs)

    return dict(
        predictive_mean=tf.reduce_mean(predicted_state.particles, axis=0),
        predictive_cov=sample_stats.covariance(predicted_state.particles),
        predictive_stddev=sample_stats.stddev(predicted_state.particles),
        updated_mean=tf.reduce_mean(updated_state.particles, axis=0),
        updated_cov=sample_stats.covariance(updated_state.particles),
        updated_stddev=sample_stats.stddev(updated_state.particles),
        log_marginal_likelihood=log_marginal_likelihood,
    )

  @test_combinations.generate(
      test_combinations.combine(
          noise_level=[0.001, 0.1, 1.0],
          n_states=[2, 5],
          n_observations=[2, 5],
          perturbed_observations=[False, True],
      ))
  def test_kf_vs_enkf(
      self,
      noise_level,
      n_states,
      n_observations,
      perturbed_observations,
  ):
    """Check that the KF and EnKF solutions are the same."""
    # Tests pass with n_ensemble = 1e7. The KF vs. EnKF tolerance is
    # proportional to 1 / sqrt(n_ensemble), so this shows good agreement.
    n_ensemble = int(1e4) if NUMPY_MODE else int(1e5)

    salt = str(noise_level) + str(n_states) + str(n_observations)
    seed_stream = test_util.test_seed_stream(salt)
    dtype = tf.float64
    predict_kwargs = {}
    update_kwargs = {}
    log_marginal_likelihood_kwargs = {
        'perturbed_observations': perturbed_observations,
    }

    linear_model_params = self._get_linear_model_params(
        noise_level=noise_level,
        n_states=n_states,
        n_observations=n_observations,
        seed_stream=seed_stream,
        dtype=dtype)

    # Ensure that our observation comes from a state that ~ prior.
    prior_dist = mvn_tril.MultivariateNormalTriL(
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
    tol_scale = 1 / np.sqrt(n_ensemble)  # 1 / Sqrt(1e5) ~= 0.003
    self.assertAllCloseNested(
        kf_soln, enkf_soln, atol=20 * tol_scale, rtol=50 * tol_scale)

  @parameterized.named_parameters(
      dict(
          testcase_name='low_rank_ensemble',
          kwargs_1=dict(
              predict={},
              update={
                  'low_rank_ensemble': False,
              },
              log_marginal_likelihood={
                  'low_rank_ensemble': False,
                  'perturbed_observations': False
              },
          ),
          kwargs_2=dict(
              predict={},
              update={
                  'low_rank_ensemble': True,
              },
              log_marginal_likelihood={
                  'low_rank_ensemble': True,
                  'perturbed_observations': False
              },
          ),
      ),
      dict(
          testcase_name='low_rank_ensemble_1d_obs',
          # n_observations = 1 invokes a special code path.
          n_observations=1,
          kwargs_1=dict(
              predict={},
              update={
                  'low_rank_ensemble': False,
              },
              log_marginal_likelihood={
                  'low_rank_ensemble': False,
                  'perturbed_observations': False
              },
          ),
          kwargs_2=dict(
              predict={},
              update={
                  'low_rank_ensemble': True,
              },
              log_marginal_likelihood={
                  'low_rank_ensemble': True,
                  'perturbed_observations': False
              },
          ),
      ),
  )
  def test_cases_where_different_kwargs_give_same_enkf_result(
      self,
      kwargs_1,
      kwargs_2,
      n_states=5,
      n_observations=5,
      n_ensemble=10,
  ):
    """Check that two sets of kwargs give same result."""
    # In most cases, `test_kf_vs_enkf` is more complete, since it tests
    # correctness. However, `test_kf_vs_enkf` requires a huge ensemble.
    # This test is useful when you cannot use a huge ensemble and/or you want to
    # compare to a method already checked for correctness by `test_kf_vs_enkf`.
    salt = str(n_ensemble) + str(n_states) + str(n_observations)
    seed_stream = test_util.test_seed_stream(salt)
    dtype = tf.float64

    linear_model_params = self._get_linear_model_params(
        noise_level=0.1,
        n_states=n_states,
        n_observations=n_observations,
        seed_stream=seed_stream,
        dtype=dtype)

    # Ensure that our observation comes from a state that ~ prior.
    prior_dist = mvn_tril.MultivariateNormalTriL(
        loc=linear_model_params.prior_mean,
        scale_tril=tf.linalg.cholesky(linear_model_params.prior_cov))
    true_state = prior_dist.sample(seed=seed_stream())
    observation = tf.linalg.matvec(linear_model_params.observation_mat,
                                   true_state)

    enkf_params = self._get_enkf_params(n_ensemble, linear_model_params,
                                        prior_dist, seed_stream, dtype)

    # Use the exact same seeds for each.
    enkf_soln_1 = self._enkf_solve(observation, enkf_params,
                                   kwargs_1['predict'], kwargs_1['update'],
                                   kwargs_1['log_marginal_likelihood'],
                                   test_util.test_seed_stream(salt))
    enkf_soln_2 = self._enkf_solve(observation, enkf_params,
                                   kwargs_2['predict'], kwargs_2['update'],
                                   kwargs_2['log_marginal_likelihood'],
                                   test_util.test_seed_stream(salt))

    # Evaluate at the same time, so both use the same randomness!
    # Do not use anything that was not evaluated here!
    enkf_soln_1, enkf_soln_2 = self.evaluate([enkf_soln_1, enkf_soln_2])

    # We used the same seed, so solutions should be identical up to tolerance of
    # different solver methods.
    self.assertAllCloseNested(enkf_soln_1, enkf_soln_2)


if __name__ == '__main__':
  test_util.main()
