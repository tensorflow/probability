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
"""Tests for VariationalGaussianProcess."""

from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import variational_gaussian_process
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic as tfpk


def _np_kernel_matrix_fn(amp, length_scale, x, y):
  x = np.expand_dims(x, -2)[..., 0]
  y = np.expand_dims(y, -3)[..., 0]
  return amp ** 2 * np.exp(-.5 * ((x - y)**2) / (length_scale**2))


# TODO(b/127523126): Figure out good tests for correctness for VGP, and add
# them here.
# Potential start is constructing kernels for which the Nystrom approximation is
# almost exact. This imples the VGP replicates the GP.
@test_util.test_all_tf_execution_regimes
class VariationalGaussianProcessTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_static', is_static=True),
      dict(testcase_name='_dyanmic', is_static=False))
  def testShapes(self, is_static):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [25, 2]
    batched_index_points = np.expand_dims(np.stack([index_points]*6), -3)
    # ==> shape = [6, 1, 25, 2]

    # 9 inducing index points in R^2
    inducing_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    inducing_index_points = np.stack(np.meshgrid(inducing_index_points,
                                                 inducing_index_points),
                                     axis=-1)
    inducing_index_points = np.reshape(inducing_index_points, [-1, 2])
    # ==> shape = [9, 2]

    variational_inducing_observations_loc = np.zeros([3, 9], dtype=np.float64)
    variational_inducing_observations_scale = np.eye(9, dtype=np.float64)

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape([1, 4, 1, 1])

    jitter = np.float64(1e-6)
    observation_noise_variance = np.float64(1e-2)

    if not is_static:
      amplitude = tf1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf1.placeholder_with_default(length_scale, shape=None)
      batched_index_points = tf1.placeholder_with_default(
          batched_index_points, shape=None)

      inducing_index_points = tf1.placeholder_with_default(
          inducing_index_points, shape=None)
      variational_inducing_observations_loc = tf1.placeholder_with_default(
          variational_inducing_observations_loc, shape=None)
      variational_inducing_observations_scale = tf1.placeholder_with_default(
          variational_inducing_observations_scale, shape=None)

    kernel = tfpk.ExponentiatedQuadratic(amplitude, length_scale)

    vgp = variational_gaussian_process.VariationalGaussianProcess(
        kernel=kernel,
        index_points=batched_index_points,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=(
            variational_inducing_observations_loc),
        variational_inducing_observations_scale=(
            variational_inducing_observations_scale),
        observation_noise_variance=observation_noise_variance,
        jitter=jitter)

    batch_shape = [2, 4, 6, 3]
    event_shape = [25]
    sample_shape = [9, 3]

    samples = vgp.sample(
        sample_shape, seed=test_util.test_seed())

    if is_static or tf.executing_eagerly():
      self.assertAllEqual(
          vgp.observation_noise_variance.shape, tf.TensorShape([]))
      self.assertAllEqual(
          vgp.predictive_noise_variance.shape, tf.TensorShape([]))
      self.assertAllEqual(vgp.batch_shape_tensor(), batch_shape)
      self.assertAllEqual(vgp.event_shape_tensor(), event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(vgp.batch_shape, batch_shape)
      self.assertAllEqual(vgp.event_shape, event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
    else:
      self.assertAllEqual(
          self.evaluate(tf.shape(vgp.observation_noise_variance)), [])
      self.assertAllEqual(
          self.evaluate(tf.shape(vgp.predictive_noise_variance)), [])
      self.assertAllEqual(self.evaluate(vgp.batch_shape_tensor()), batch_shape)
      self.assertAllEqual(self.evaluate(vgp.event_shape_tensor()), event_shape)
      self.assertAllEqual(self.evaluate(samples).shape,
                          sample_shape + batch_shape + event_shape)
      self.assertIsNone(tensorshape_util.rank(samples.shape))
      self.assertIsNone(tensorshape_util.rank(vgp.batch_shape))
      self.assertEqual(tensorshape_util.rank(vgp.event_shape), 1)
      self.assertIsNone(
          tf.compat.dimension_value(tensorshape_util.dims(vgp.event_shape)[0]))

  @parameterized.named_parameters(
      dict(testcase_name='_static', is_static=True),
      dict(testcase_name='_dyanmic', is_static=False))
  def testOptimalVariationalShapes(self, is_static):
    # 5x5 grid of observation index points in R^2 and flatten to 25x2
    observation_index_points = np.linspace(-4., 4., 5, dtype=np.float64)
    observation_index_points = np.stack(
        np.meshgrid(
            observation_index_points, observation_index_points), axis=-1)
    observation_index_points = np.reshape(
        observation_index_points, [-1, 2])
    # ==> shape = [25, 2]
    observation_index_points = np.expand_dims(
        np.stack([observation_index_points]*6), -3)
    # ==> shape = [6, 1, 25, 2]
    observations = np.sin(observation_index_points[..., 0])
    # ==> shape = [6, 1, 25]

    # 9 inducing index points in R^2
    inducing_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    inducing_index_points = np.stack(np.meshgrid(inducing_index_points,
                                                 inducing_index_points),
                                     axis=-1)
    inducing_index_points = np.reshape(inducing_index_points, [-1, 2])
    # ==> shape = [9, 2]

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape([1, 4, 1, 1])

    jitter = np.float64(1e-6)
    observation_noise_variance = np.float64(1e-2)

    if not is_static:
      amplitude = tf1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf1.placeholder_with_default(length_scale, shape=None)
      observation_index_points = tf1.placeholder_with_default(
          observation_index_points, shape=None)

      inducing_index_points = tf1.placeholder_with_default(
          inducing_index_points, shape=None)
    kernel = tfpk.ExponentiatedQuadratic(amplitude, length_scale)

    loc, scale = variational_gaussian_process.VariationalGaussianProcess.optimal_variational_posterior(
        kernel=kernel,
        inducing_index_points=inducing_index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter,
    )
    # We should expect that loc has shape [2, 4, 6, 1, 9]. This is because:
    # * [2, 4] comes from the batch shape of the kernel.
    # * [6, 1] comes from the batch shape of the observations / observation
    # index points.
    # * [9] comes from the number of inducing points.
    # Similar reasoning applies to scale.
    self.assertAllEqual([2, 4, 6, 1, 9], tf.shape(loc))
    self.assertAllEqual([2, 4, 6, 1, 9, 9], tf.shape(scale))

  @parameterized.named_parameters(
      dict(testcase_name='_static', is_static=True),
      dict(testcase_name='_dyanmic', is_static=False))
  def testVariationalLossShapes(self, is_static):
    # 2x2 grid of index points in R^2 and flatten to 4x2
    index_points = np.linspace(-4., 4., 2, dtype=np.float64)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2])
    # ==> shape = [4, 2]
    batched_index_points = np.expand_dims(np.stack([index_points]*6), -3)
    # ==> shape = [6, 1, 4, 2]

    # 3x3 grid of index points in R^2 and flatten to 9x2
    observation_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    observation_index_points = np.stack(
        np.meshgrid(
            observation_index_points, observation_index_points), axis=-1)
    observation_index_points = np.reshape(
        observation_index_points, [-1, 2])
    # ==> shape = [9, 2]
    observation_index_points = np.expand_dims(
        np.stack([observation_index_points]*6), -3)
    # ==> shape = [6, 1, 9, 2]
    observations = np.sin(observation_index_points[..., 0])
    # ==> shape = [6, 1, 9]

    # 9 inducing index points in R^2
    inducing_index_points = np.linspace(-4., 4., 3, dtype=np.float64)
    inducing_index_points = np.stack(np.meshgrid(inducing_index_points,
                                                 inducing_index_points),
                                     axis=-1)
    inducing_index_points = np.reshape(inducing_index_points, [-1, 2])
    # ==> shape = [9, 2]

    variational_inducing_observations_loc = np.zeros([3, 9], dtype=np.float64)
    variational_inducing_observations_scale = np.eye(9, dtype=np.float64)

    # Kernel with batch_shape [2, 4, 1, 1]
    amplitude = np.array([1., 2.], np.float64).reshape([2, 1, 1, 1])
    length_scale = np.array([.1, .2, .3, .4], np.float64).reshape([1, 4, 1, 1])

    jitter = np.float64(1e-6)
    observation_noise_variance = np.float64(1e-2)

    if not is_static:
      amplitude = tf1.placeholder_with_default(amplitude, shape=None)
      length_scale = tf1.placeholder_with_default(length_scale, shape=None)
      batched_index_points = tf1.placeholder_with_default(
          batched_index_points, shape=None)

      observations = tf1.placeholder_with_default(observations, shape=None)
      observation_index_points = tf1.placeholder_with_default(
          observation_index_points, shape=None)
      inducing_index_points = tf1.placeholder_with_default(
          inducing_index_points, shape=None)
      variational_inducing_observations_loc = tf1.placeholder_with_default(
          variational_inducing_observations_loc, shape=None)
      variational_inducing_observations_scale = tf1.placeholder_with_default(
          variational_inducing_observations_scale, shape=None)

    kernel = tfpk.ExponentiatedQuadratic(amplitude, length_scale)

    vgp = variational_gaussian_process.VariationalGaussianProcess(
        kernel=kernel,
        index_points=batched_index_points,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=(
            variational_inducing_observations_loc),
        variational_inducing_observations_scale=(
            variational_inducing_observations_scale),
        observation_noise_variance=observation_noise_variance,
        jitter=jitter,
        validate_args=True)

    loss = vgp.variational_loss(
        observations=observations,
        observation_index_points=observation_index_points)
    self.assertAllEqual(vgp.batch_shape_tensor(), tf.shape(loss))

  def testCustomCholeskyFn(self):
    def test_cholesky(x):
      test_cholesky.cholesky_count += 1
      return tf.linalg.cholesky(tf.linalg.set_diag(
          x, tf.linalg.diag_part(x) + 3.))
    test_cholesky.cholesky_count = 0

    index_points = np.linspace(-4., 4., 5, dtype=np.float64)[..., np.newaxis]
    inducing_index_points = np.linspace(-4., 4., 3, dtype=np.float64)[
        ..., np.newaxis]

    variational_inducing_observations_loc = np.zeros([3], dtype=np.float64)
    variational_inducing_observations_scale = np.eye(3, dtype=np.float64)

    amplitude = np.float64(1.)
    length_scale = np.float64(1.)

    jitter = np.float64(1e-6)
    kernel = tfpk.ExponentiatedQuadratic(amplitude, length_scale)

    vgp = variational_gaussian_process.VariationalGaussianProcess(
        kernel=kernel,
        index_points=index_points,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=(
            variational_inducing_observations_loc),
        variational_inducing_observations_scale=(
            variational_inducing_observations_scale),
        observation_noise_variance=1e-6,
        cholesky_fn=test_cholesky,
        jitter=jitter)
    self.evaluate(vgp.get_marginal_distribution().stddev())
    # Assert that the custom cholesky function is called at least once.
    self.assertGreaterEqual(test_cholesky.cholesky_count, 1)

  def testBernoulliLikelihood(self):
    kernel = tfpk.ExponentiatedQuadratic()
    num_predictive_points = 10
    num_inducing_points = 5
    num_observations = 15
    vgp = variational_gaussian_process.VariationalGaussianProcess(
        kernel=kernel,
        # 10 predictive locations, 5 inducing points both in 1-d
        index_points=np.random.uniform(size=(num_predictive_points, 1)),
        inducing_index_points=np.random.uniform(size=(num_inducing_points, 1)),
        # Variational inducing observations posterior normal with mean zero and
        # identity scale.
        variational_inducing_observations_loc=np.zeros(
            num_inducing_points, dtype=np.float64),
        variational_inducing_observations_scale=np.eye(
            num_inducing_points, dtype=np.float64),
        # No observation noise
        observation_noise_variance=0.)

    def log_likelihood_fn(observations, gp_events):
      bern = independent.Independent(
          bernoulli.Bernoulli(logits=gp_events), reinterpreted_batch_ndims=1)
      return bern.log_prob(observations)

    observations = np.random.randint(2, size=num_observations)
    observation_index_points = np.random.uniform(size=(num_observations, 1))

    # We really just check that this goes through without any shape errors.
    # TODO(b/127523126): Add more principled tests that this actually computes
    # what we expect it to (manual experimentation and testing has been done and
    # convincing results observed, but no clear strategy for automated tests
    # jumps out as terribly obvious).
    vgp.surrogate_posterior_expected_log_likelihood(
        observations,
        observation_index_points=observation_index_points,
        log_likelihood_fn=log_likelihood_fn,
        quadrature_size=20)

  def testMulticlassClassificationLikelihood(self):
    kernel = tfpk.ExponentiatedQuadratic()
    num_predictive_points = 10
    num_inducing_points = 5
    num_observations = 15
    num_classes = 10

    # To handle the multiclass classification task, we just need to make sure we
    # model a batch of `num_classes` independent GPs, whose values are the
    # logits of a Categorical distribution over observations.
    vgp = variational_gaussian_process.VariationalGaussianProcess(
        kernel=kernel,
        index_points=np.random.uniform(size=(num_predictive_points, 1)),

        # Batch inducing points of size `num_classes`.
        inducing_index_points=np.random.uniform(
            size=(num_classes, num_inducing_points, 1)),
        # Variational inducing observations posterior normal with mean zero and
        # identity scale. These are also in batches of size `num_classes`.
        variational_inducing_observations_loc=np.zeros(
            (num_classes, num_inducing_points), dtype=np.float64),
        variational_inducing_observations_scale=(
            np.ones((num_classes, 1, 1)) *
            np.eye(num_inducing_points, dtype=np.float64)),
        # No observation noise
        observation_noise_variance=0.)

    def log_likelihood_fn(observations, gp_events):
      # The shape of `gp_events` will be of the form
      #   [num_classes, quadrature_size, num_obserations]
      #
      # We want a batch of `Independent(Categorical)` distributions with
      #
      #   independent_categorical.batch_shape = [quadrature_size]
      #   independent_categorical.event_shape = [num_observations]
      #
      # and number of categories equal to `num_classes`. So we need to permute
      # the shape of  `gp_events` from the above to
      #
      #   [quadrature_size, num_observations, num_classes].
      logits = tf.transpose(gp_events, [1, 2, 0])
      independent_categorical = independent.Independent(
          categorical.Categorical(logits=logits),
          # Reinterpret `num_observations` as part of the event shape.
          reinterpreted_batch_ndims=1)
      return independent_categorical.log_prob(observations)

    observations = np.random.randint(num_classes, size=num_observations)
    observation_index_points = np.random.uniform(size=(num_observations, 1))

    # We really just check that this goes through without any shape errors.
    # TODO(b/127523126): Add more principled tests that this actually computes
    # what we expect it to (manual experimentation and testing has been done and
    # convincing results observed, but no clear strategy for automated tests
    # jumps out as terribly obvious).
    vgp.surrogate_posterior_expected_log_likelihood(
        observations,
        observation_index_points=observation_index_points,
        log_likelihood_fn=log_likelihood_fn,
        quadrature_size=20)

  def testWhiteningTransformationMatches(self):
    kernel = tfpk.ExponentiatedQuadratic()
    num_predictive_points = 10
    num_inducing_points = 5

    index_points = np.random.uniform(size=(num_predictive_points, 3))
    inducing_index_points = np.random.uniform(size=(num_inducing_points, 3))
    variational_loc = np.random.uniform(size=(num_inducing_points,))
    # Positive random triangular matrix.
    variational_scale = np.tril(
        np.random.uniform(size=(num_inducing_points, num_inducing_points)))**2
    observation_noise_variance = np.float64(1e-5)

    vgp1 = variational_gaussian_process.VariationalGaussianProcess(
        kernel=kernel,
        index_points=index_points,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=variational_loc,
        variational_inducing_observations_scale=variational_scale,
        observation_noise_variance=observation_noise_variance)

    chol_kzz = tf.linalg.cholesky(
        kernel.matrix(inducing_index_points, inducing_index_points))
    chol_kzz_linop = tf.linalg.LinearOperatorLowerTriangular(
        chol_kzz)

    vgp2 = variational_gaussian_process.VariationalGaussianProcess(
        kernel=kernel,
        index_points=index_points,
        inducing_index_points=inducing_index_points,
        # NOTE: When using whitening, one does not need to do this
        # transformation. This is purely for checking comparable numerics
        # when whitening is on and off.
        variational_inducing_observations_loc=chol_kzz_linop.solvevec(
            variational_loc),
        variational_inducing_observations_scale=chol_kzz_linop.solve(
            variational_scale),
        observation_noise_variance=observation_noise_variance,
        use_whitening_transform=True)

    mean_vgp1, mean_vgp2 = self.evaluate([vgp1.mean(), vgp2.mean()])
    self.assertAllClose(mean_vgp1, mean_vgp2, rtol=3e-5)

    stddev_vgp1, stddev_vgp2 = self.evaluate([vgp1.stddev(), vgp2.stddev()])
    self.assertAllClose(stddev_vgp1, stddev_vgp2, rtol=3e-5)

    loss_vgp1, loss_vgp2 = self.evaluate([
        vgp1.surrogate_posterior_kl_divergence_prior(),
        vgp2.surrogate_posterior_kl_divergence_prior()])

    self.assertAllClose(loss_vgp1, loss_vgp2, rtol=7e-5)


if __name__ == '__main__':
  test_util.main()
