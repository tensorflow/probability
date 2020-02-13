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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


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

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale)

    vgp = tfd.VariationalGaussianProcess(
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
      self.assertAllEqual(vgp.batch_shape_tensor(), batch_shape)
      self.assertAllEqual(vgp.event_shape_tensor(), event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(vgp.batch_shape, batch_shape)
      self.assertAllEqual(vgp.event_shape, event_shape)
      self.assertAllEqual(samples.shape,
                          sample_shape + batch_shape + event_shape)
    else:
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
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale)

    loc, scale = tfd.VariationalGaussianProcess.optimal_variational_posterior(
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

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale)

    vgp = tfd.VariationalGaussianProcess(
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

  def testBernoulliLikelihood(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    num_predictive_points = 10
    num_inducing_points = 5
    num_observations = 15
    vgp = tfd.VariationalGaussianProcess(
        kernel=kernel,
        # 10 predictive locations, 5 inducing points both in 1-d
        index_points=np.random.uniform(size=(num_predictive_points, 1)),
        inducing_index_points=np.random.uniform(size=(num_inducing_points, 1)),
        # Variational inducing observations posterior normal with mean zero and
        # identity scale.
        variational_inducing_observations_loc=np.zeros(num_inducing_points,
                                                       dtype=np.float64),
        variational_inducing_observations_scale=np.eye(num_inducing_points,
                                                       dtype=np.float64),
        # No observation noise
        observation_noise_variance=0.)

    def log_likelihood_fn(observations, gp_events):
      bernoulli = tfd.Independent(tfd.Bernoulli(logits=gp_events),
                                  reinterpreted_batch_ndims=1)
      return bernoulli.log_prob(observations)

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
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    num_predictive_points = 10
    num_inducing_points = 5
    num_observations = 15
    num_classes = 10

    # To handle the multiclass classification task, we just need to make sure we
    # model a batch of `num_classes` independent GPs, whose values are the
    # logits of a Categorical distribution over observations.
    vgp = tfd.VariationalGaussianProcess(
        kernel=kernel,
        index_points=np.random.uniform(size=(num_predictive_points, 1)),

        # Batch inducing points of size `num_classes`.
        inducing_index_points=np.random.uniform(size=(num_classes,
                                                      num_inducing_points,
                                                      1)),
        # Variational inducing observations posterior normal with mean zero and
        # identity scale. These are also in batches of size `num_classes`.
        variational_inducing_observations_loc=np.zeros((num_classes,
                                                        num_inducing_points),
                                                       dtype=np.float64),
        variational_inducing_observations_scale=(np.ones((num_classes, 1, 1)) *
                                                 np.eye(num_inducing_points,
                                                        dtype=np.float64)),
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
      independent_categorical = tfd.Independent(
          tfd.Categorical(logits=logits),
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

if __name__ == '__main__':
  tf.test.main()
