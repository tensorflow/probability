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
"""Tests NeuTra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from discussion import neutra
from tensorflow_probability.python.internal import test_util as tfp_test_util

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class NeutraKernelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((tfb.Identity(), tf.float32),
                            (tfb.Identity(), tf.float64),
                            (tfb.Softplus(), tf.float32))
  def testSingleTensor(self, bijector, dtype):
    if not tf.executing_eagerly():
      return
    base_mean = tf.convert_to_tensor(value=[1., 0], dtype=dtype)
    base_cov = tf.convert_to_tensor(value=[[1, 0.5], [0.5, 1]], dtype=dtype)

    base_dist = tfd.MultivariateNormalFullCovariance(
        loc=base_mean, covariance_matrix=base_cov)
    target_dist = bijector(base_dist)

    def debug_fn(*args):
      del args
      debug_fn.count += 1

    debug_fn.count = 0

    kernel = neutra.NeuTra(
        target_log_prob_fn=target_dist.log_prob,
        state_shape=2,
        num_step_size_adaptation_steps=800,
        num_train_steps=1000,
        train_batch_size=64,
        learning_rate=tf.convert_to_tensor(value=1e-2, dtype=dtype),
        seed=tfp_test_util.test_seed(),
        train_debug_fn=debug_fn,
        unconstraining_bijector=bijector,
    )

    chain = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=1000,
        current_state=tf.zeros([16, 2], dtype),
        kernel=kernel,
        trace_fn=None,
        parallel_iterations=1)
    self.assertEqual(1000, debug_fn.count)

    sample_mean = tf.reduce_mean(input_tensor=chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_samples = target_dist.sample(4096, seed=tfp_test_util.test_seed())

    true_mean = tf.reduce_mean(input_tensor=true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_cov, sample_cov, rtol=0.1, atol=0.1)

  @parameterized.parameters([tfb.Identity(), tfb.Softplus()])
  def testNested(self, bijector):
    if not tf.executing_eagerly():
      return
    base_mean = tf.constant([1., 0])
    base_cov = tf.constant([[1, 0.5], [0.5, 1]])

    dist_2d = tfd.MultivariateNormalFullCovariance(
        loc=base_mean, covariance_matrix=base_cov)
    dist_4d = tfd.MultivariateNormalDiag(scale_diag=tf.ones(4))

    target_dist = tfd.JointDistributionSequential([
        bijector(dist_2d),
        tfb.Reshape([2, 2])(dist_4d),
    ])

    kernel = neutra.NeuTra(
        target_log_prob_fn=lambda x, y: target_dist.log_prob((x, y)),
        state_shape=target_dist.event_shape,
        num_step_size_adaptation_steps=800,
        num_train_steps=1000,
        train_batch_size=64,
        seed=tfp_test_util.test_seed(),
        unconstraining_bijector=[bijector, tfb.Identity()],
    )

    chain_2d, chain_4d = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=1000,
        current_state=tf.nest.map_structure(
            lambda s: tf.zeros([16] + s.as_list()), target_dist.event_shape),
        kernel=kernel,
        trace_fn=None,
        parallel_iterations=1)

    sample_mean_2d = tf.reduce_mean(input_tensor=chain_2d, axis=[0, 1])
    sample_mean_4d = tf.reduce_mean(input_tensor=chain_4d, axis=[0, 1])

    true_samples_2d, true_samples_4d = target_dist.sample(
        4096, seed=tfp_test_util.test_seed())

    true_mean_2d = tf.reduce_mean(input_tensor=true_samples_2d, axis=0)
    true_mean_4d = tf.reduce_mean(input_tensor=true_samples_4d, axis=0)

    self.assertAllClose(true_mean_2d, sample_mean_2d, rtol=0.1, atol=0.1)
    self.assertAllClose(true_mean_4d, sample_mean_4d, rtol=0.1, atol=0.1)


if __name__ == '__main__':
  tf.test.main()
