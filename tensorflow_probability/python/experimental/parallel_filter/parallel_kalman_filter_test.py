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
"""Tests for parallel Kalman filter library."""
import collections

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.parallel_filter import parallel_kalman_filter_lib
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


def _random_variance(dim, batch, dtype, seed):
  w = tf.random.stateless_normal(batch + (dim, dim), dtype=dtype, seed=seed)
  return tf.eye(
      dim, dtype=dtype) + 0.1 * tf.linalg.matmul(w, w, transpose_a=True)


def _random_vector(dim, batch, dtype, seed):
  return tf.random.stateless_normal(batch + (dim,), dtype=dtype, seed=seed)


def _random_mask(batch, dtype, seed):
  return tf.cast(
      tf.random.stateless_uniform(batch,
                                  maxval=2,
                                  dtype=tf.int32,
                                  seed=seed),
      dtype=dtype)


def _random_matrix(dim0, dim1, batch, dtype, seed):
  return tf.random.stateless_normal(batch + (dim0, dim1),
                                    dtype=dtype, seed=seed)


class _KalmanFilterTest(test_util.TestCase):

  def test_basic_example_time_dependent(self):
    nsteps = 7
    initial_mean = np.array([0.], dtype=self.dtype)
    initial_cov = np.eye(1, dtype=self.dtype)
    transition_matrix = np.array(nsteps * [[[1.]]], dtype=self.dtype)
    transition_mean = np.array(nsteps * [[0.]], dtype=self.dtype)
    transition_cov = np.array(nsteps * [[[1.]]], dtype=self.dtype)
    observation_matrix = np.array(nsteps * [[[1.]]], dtype=self.dtype)
    observation_mean = np.array(nsteps * [[0.]], dtype=self.dtype)
    observation_cov = np.array(nsteps * [[[1.]]], dtype=self.dtype)

    _, y = parallel_kalman_filter_lib.sample_walk(
        transition_matrix=transition_matrix,
        transition_mean=transition_mean,
        transition_scale_tril=tf.linalg.cholesky(transition_cov),
        observation_matrix=observation_matrix,
        observation_mean=observation_mean,
        observation_scale_tril=tf.linalg.cholesky(observation_cov),
        initial_mean=initial_mean,
        initial_scale_tril=tf.linalg.cholesky(initial_cov),
        seed=test_util.test_seed())

    (_, _, filtered_covs, _, _, _, _) = (
        parallel_kalman_filter_lib.kalman_filter(
            transition_matrix=transition_matrix,
            transition_mean=transition_mean,
            transition_cov=transition_cov,
            observation_matrix=observation_matrix,
            observation_mean=observation_mean,
            observation_cov=observation_cov,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
            y=y,
            mask=None))

    fibonacci = np.array(
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],
        dtype=self.dtype)
    approximants = fibonacci[::2] / fibonacci[1::2]
    self.assertAllClose(filtered_covs[..., 0, 0], approximants)

  @test_util.disable_test_for_backend(disable_numpy=True, reason='Test compile')
  def test_basic_example_time_dependent_compiled(self):
    nsteps = 7
    initial_mean = np.array([0.], dtype=self.dtype)
    initial_cov = np.eye(1, dtype=self.dtype)
    transition_matrix = np.array(nsteps * [[[1.]]], dtype=self.dtype)
    transition_mean = np.array(nsteps * [[0.]], dtype=self.dtype)
    transition_cov = np.array(nsteps * [[[1.]]], dtype=self.dtype)
    observation_matrix = np.array(nsteps * [[[1.]]], dtype=self.dtype)
    observation_mean = np.array(nsteps * [[0.]], dtype=self.dtype)
    observation_cov = np.array(nsteps * [[[1.]]], dtype=self.dtype)
    seed = test_util.test_seed()

    def sample_walk():
      _, y = parallel_kalman_filter_lib.sample_walk(
          transition_matrix=transition_matrix,
          transition_mean=transition_mean,
          transition_scale_tril=tf.linalg.cholesky(transition_cov),
          observation_matrix=observation_matrix,
          observation_mean=observation_mean,
          observation_scale_tril=tf.linalg.cholesky(observation_cov),
          initial_mean=initial_mean,
          initial_scale_tril=tf.linalg.cholesky(initial_cov),
          seed=seed)
      return y

    def kalman_filter():
      (_, _, filtered_covs, _, _, _, _) = (
          parallel_kalman_filter_lib.kalman_filter(
              transition_matrix=transition_matrix,
              transition_mean=transition_mean,
              transition_cov=transition_cov,
              observation_matrix=observation_matrix,
              observation_mean=observation_mean,
              observation_cov=observation_cov,
              initial_mean=initial_mean,
              initial_cov=initial_cov,
              y=y,
              mask=None))
      return filtered_covs

    y = tf.function(sample_walk, jit_compile=True)()
    filtered_covs = tf.function(kalman_filter, jit_compile=True)()

    fibonacci = np.array(
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],
        dtype=self.dtype)
    approximants = fibonacci[::2] / fibonacci[1::2]
    self.assertAllClose(filtered_covs[..., 0, 0], approximants)

  def test_basic_example_time_dependent_batched(self):
    batch_shape = (2, 3)
    ndim = 7  # Dimension of latent space
    mdim = 5  # Dimension of observation space
    nsteps = 9

    Batches = collections.namedtuple(
        'Batches', ['initial_mean',
                    'initial_cov',
                    'transition_matrix',
                    'transition_mean',
                    'transition_cov',
                    'observation_matrix',
                    'observation_mean',
                    'observation_cov',
                    'mask'])

    def batch_generator():
      # Skipping 'mask' case because it isn't used in sample generation.
      for skip in range(8):
        batch_list = skip * [()] + [batch_shape] + (9 - skip - 1) * [()]
        yield Batches(*batch_list)

    # Test the broadcasting by ensuring each parameter individually
    # can be broadcast up to the full batch size.
    seed = test_util.test_seed(sampler_type='stateless')
    for batches in batch_generator():
      iter_seed, seed = samplers.split_seed(seed, n=2, salt='')
      s = samplers.split_seed(iter_seed, n=10, salt='')
      initial_mean = _random_vector(
          ndim, batches.initial_mean, dtype=self.dtype, seed=s[0])
      initial_cov = _random_variance(
          ndim, batches.initial_cov, dtype=self.dtype, seed=s[1])
      transition_matrix = 0.2 * _random_matrix(  # Avoid blowup (eigvals > 1).
          ndim, ndim, (nsteps,) + batches.transition_matrix,
          dtype=self.dtype, seed=s[2])
      transition_mean = _random_vector(
          ndim, (nsteps,) + batches.transition_mean,
          dtype=self.dtype, seed=s[3])
      transition_cov = _random_variance(
          ndim, (nsteps,) + batches.transition_cov,
          dtype=self.dtype, seed=s[4])
      observation_matrix = _random_matrix(
          mdim, ndim, (nsteps,) + batches.observation_matrix,
          dtype=self.dtype, seed=s[5])
      observation_mean = _random_vector(
          mdim, (nsteps,) + batches.observation_mean,
          dtype=self.dtype, seed=s[6])
      observation_cov = _random_variance(
          mdim, (nsteps,) + batches.observation_cov,
          dtype=self.dtype, seed=s[7])
      mask = _random_mask(
          (nsteps,) + batches.mask, dtype=tf.bool, seed=s[8])

      _, y = parallel_kalman_filter_lib.sample_walk(
          transition_matrix=transition_matrix,
          transition_mean=transition_mean,
          transition_scale_tril=tf.linalg.cholesky(transition_cov),
          observation_matrix=observation_matrix,
          observation_mean=observation_mean,
          observation_scale_tril=tf.linalg.cholesky(observation_cov),
          initial_mean=initial_mean,
          initial_scale_tril=tf.linalg.cholesky(initial_cov),
          seed=s[9])

      my_filter_results = parallel_kalman_filter_lib.kalman_filter(
          transition_matrix=transition_matrix,
          transition_mean=transition_mean,
          transition_cov=transition_cov,
          observation_matrix=observation_matrix,
          observation_mean=observation_mean,
          observation_cov=observation_cov,
          initial_mean=initial_mean,
          initial_cov=initial_cov,
          y=y,
          mask=mask)
      ((my_log_likelihoods,
        my_filtered_means, my_filtered_covs, my_predicted_means,
        my_predicted_covs, my_observation_means,
        my_observation_covs),
       y, mask) = tf.nest.map_structure(
           lambda x, r: distribution_util.move_dimension(x, 0, -r),
           (my_filter_results, y, mask),
           (type(my_filter_results)(1, 2, 3, 2, 3, 2, 3), 2, 1))

      # pylint: disable=g-long-lambda,cell-var-from-loop
      mvn = tfd.MultivariateNormalFullCovariance
      dist = tfd.LinearGaussianStateSpaceModel(
          num_timesteps=nsteps,
          transition_matrix=lambda t: tf.linalg.LinearOperatorFullMatrix(
              tf.gather(transition_matrix, t, axis=0)),
          transition_noise=lambda t: mvn(
              loc=tf.gather(transition_mean, t, axis=0),
              covariance_matrix=tf.gather(transition_cov, t, axis=0)),
          observation_matrix=lambda t: tf.linalg.LinearOperatorFullMatrix(
              tf.gather(observation_matrix, t, axis=0)),
          observation_noise=lambda t: mvn(
              loc=tf.gather(observation_mean, t, axis=0),
              covariance_matrix=tf.gather(observation_cov, t, axis=0)),
          initial_state_prior=mvn(loc=initial_mean,
                                  covariance_matrix=initial_cov),
          experimental_parallelize=False)  # Compare against sequential filter.
      # pylint: enable=g-long-lambda,cell-var-from-loop

      (log_likelihoods,
       filtered_means,
       filtered_covs,
       predicted_means,
       predicted_covs,
       observation_means,
       observation_covs) = dist.forward_filter(y, mask)

      rtol = (1e-6 if self.dtype == np.float64 else 1e-1)
      atol = (1e-6 if self.dtype == np.float64 else 1e-3)
      self.assertAllClose(log_likelihoods, my_log_likelihoods,
                          rtol=rtol, atol=atol)

      rtol = (1e-6 if self.dtype == np.float64 else 1e-3)
      atol = (1e-6 if self.dtype == np.float64 else 1e-3)
      self.assertAllClose(filtered_means, my_filtered_means,
                          rtol=rtol, atol=atol)
      self.assertAllClose(filtered_covs, my_filtered_covs,
                          rtol=rtol, atol=atol)
      self.assertAllClose(predicted_means, my_predicted_means,
                          rtol=rtol, atol=atol)
      self.assertAllClose(predicted_covs, my_predicted_covs,
                          rtol=rtol, atol=atol)
      self.assertAllClose(observation_means, my_observation_means,
                          rtol=rtol, atol=atol)
      self.assertAllClose(observation_covs, my_observation_covs,
                          rtol=rtol, atol=atol)

  @test_util.numpy_disable_gradient_test
  def test_log_prob_gradients_exist(self):
    nsteps = 7
    initial_mean = tf.constant([0.], dtype=self.dtype)
    initial_cov = tf.eye(1, dtype=self.dtype)
    transition_matrix = tf.constant(nsteps * [[[1.]]], dtype=self.dtype)
    transition_mean = tf.constant(nsteps * [[0.]], dtype=self.dtype)
    transition_cov = tf.constant(nsteps * [[[1.]]], dtype=self.dtype)
    observation_matrix = tf.constant(nsteps * [[[1.]]], dtype=self.dtype)
    observation_mean = tf.constant(nsteps * [[0.]], dtype=self.dtype)
    observation_cov = tf.constant(nsteps * [[[1.]]], dtype=self.dtype)
    y = tf.ones([nsteps, 1], dtype=self.dtype)

    def lp(initial_mean, initial_cov,
           transition_matrix, transition_mean, transition_cov,
           observation_matrix, observation_mean, observation_cov, y):
      (log_likelihoods, _, _, _, _, _, _) = (
          parallel_kalman_filter_lib.kalman_filter(
              transition_matrix=transition_matrix,
              transition_mean=transition_mean,
              transition_cov=transition_cov,
              observation_matrix=observation_matrix,
              observation_mean=observation_mean,
              observation_cov=observation_cov,
              initial_mean=initial_mean,
              initial_cov=initial_cov,
              y=y,
              mask=None))
      return tf.reduce_sum(log_likelihoods)
    grads = tfp.math.value_and_gradient(lp, (initial_mean,
                                             initial_cov,
                                             transition_matrix,
                                             transition_mean,
                                             transition_cov,
                                             observation_matrix,
                                             observation_mean,
                                             observation_cov,
                                             y))
    for g in grads:
      self.assertIsNotNone(g)

  def test_sample_stats(self):
    ndim = 3
    mdim = 2
    nsteps = 3
    nsamples = 10000

    seed = test_util.test_seed()
    _, seed = samplers.split_seed(seed, n=2, salt='')
    s = samplers.split_seed(seed, n=9, salt='')
    initial_mean = self.evaluate(
        _random_vector(ndim, (), dtype=self.dtype, seed=s[0]))
    initial_cov = self.evaluate(
        _random_variance(ndim, (), dtype=self.dtype, seed=s[1]))
    transition_matrix = self.evaluate(
        0.5 * _random_matrix(ndim, ndim, (nsteps,),
                             dtype=self.dtype, seed=s[2]))
    transition_mean = self.evaluate(
        _random_vector(ndim, (nsteps,), dtype=self.dtype, seed=s[3]))
    transition_cov = self.evaluate(
        0.5 * _random_variance(ndim, (nsteps,), dtype=self.dtype, seed=s[4]))
    observation_matrix = self.evaluate(
        _random_matrix(mdim, ndim, (nsteps,), dtype=self.dtype, seed=s[5]))
    observation_mean = self.evaluate(
        _random_vector(mdim, (nsteps,), dtype=self.dtype, seed=s[6]))
    observation_cov = self.evaluate(
        _random_variance(mdim, (nsteps,), dtype=self.dtype, seed=s[7]))

    x, y = self.evaluate(
        parallel_kalman_filter_lib.sample_walk(
            transition_matrix=transition_matrix,
            transition_scale_tril=tf.linalg.cholesky(transition_cov),
            observation_matrix=observation_matrix,
            observation_scale_tril=tf.linalg.cholesky(observation_cov),
            initial_mean=tf.broadcast_to(initial_mean,
                                         ps.concat([[nsamples],
                                                    ps.shape(initial_mean)],
                                                   axis=0)),
            initial_scale_tril=tf.linalg.cholesky(initial_cov),
            transition_mean=transition_mean,
            observation_mean=observation_mean,
            seed=s[8]))
    empirical_initial_mean = np.mean(x[0], axis=0)
    empirical_initial_cov = self.evaluate(tfp.stats.covariance(x[0]))
    self.assertAllClose(empirical_initial_mean, initial_mean, atol=0.1)
    self.assertAllClose(empirical_initial_cov, initial_cov, atol=0.1)

    transition_residuals = self.evaluate(x[1:] - tf.linalg.matvec(
        transition_matrix[:-1, tf.newaxis, ...], x[:-1]))
    empirical_transition_mean = np.mean(transition_residuals, axis=1)
    empirical_transition_cov = self.evaluate(
        tfp.stats.covariance(transition_residuals, sample_axis=1))
    # Checking the mean and cov of residuals implicitly also
    # checks the `transition_matrix` used to compute them.
    self.assertAllClose(empirical_transition_mean,
                        transition_mean[:-1],
                        atol=0.1)  # Typical error with 10k samples ~= 0.01-0.03
    self.assertAllClose(empirical_transition_cov, transition_cov[:-1], atol=0.1)

    observation_residuals = self.evaluate(
        y - tf.linalg.matvec(observation_matrix[:, tf.newaxis, ...], x))
    empirical_observation_mean = np.mean(observation_residuals, axis=1)
    empirical_observation_cov = self.evaluate(
        tfp.stats.covariance(observation_residuals,
                             sample_axis=1))
    self.assertAllClose(empirical_observation_mean,
                        observation_mean, atol=0.1)
    self.assertAllClose(empirical_observation_cov,
                        observation_cov, atol=0.1)


@test_util.test_all_tf_execution_regimes
class KalmanFilterTestFloat32(_KalmanFilterTest):
  dtype = np.float32


class KalmanFilterTestFloat64(_KalmanFilterTest):
  dtype = np.float64

del _KalmanFilterTest  # Don't run base class tests.

if __name__ == '__main__':
  test_util.main()
