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
"""Tests for ReplicaExchangeMC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tf.contrib.distributions


class REMCTest(tf.test.TestCase):

  def testRWM1DNNormal(self):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32

    with self.test_session(graph=tf.Graph()) as sess:
      target = tfd.Normal(loc=dtype(0), scale=dtype(1))

      remc = tfp.mcmc.ReplicaExchangeMC(
          target_log_prob_fn=target.log_prob,
          inverse_temperatures=10.**tf.linspace(0., -2., 5),
          make_kernel_fn=tfp.mcmc.HamiltonianMonteCarlo,
          step_size=1.0,
          num_leapfrog_steps=3,
          seed=42)

      samples, _ = tfp.mcmc.sample_chain(
          num_results=1000,
          current_state=dtype(1),
          kernel=remc,
          num_burnin_steps=500,
          parallel_iterations=1)  # For determinism.

      sample_mean = tf.reduce_mean(samples, axis=0)
      sample_std = tf.sqrt(
          tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                         axis=0))
      [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, 0., atol=0.1, rtol=0.1)
    self.assertAllClose(sample_std_, 1., atol=0.1, rtol=0.1)

  def testRWM2DMixNormal(self):
    """Sampling from a 2-D Mixture Normal Distribution."""
    dtype = np.float32

    with self.test_session(graph=tf.Graph()) as sess:
      target = tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
          components_distribution=tfd.MultivariateNormalDiag(
              loc=[[-1., -1], [1., 1.]],
              scale_identity_multiplier=[0.1, 0.1]))

      remc = tfp.mcmc.ReplicaExchangeMC(
          target_log_prob_fn=target.log_prob,
          inverse_temperatures=10.**tf.linspace(0., -2., 5),
          make_kernel_fn=tfp.mcmc.HamiltonianMonteCarlo,
          step_size=0.3,
          num_leapfrog_steps=3,
          seed=42)

      samples, _ = tfp.mcmc.sample_chain(
          num_results=5000,
          current_state=np.zeros(2, dtype=dtype),
          kernel=remc,
          num_burnin_steps=2000,
          parallel_iterations=1)  # For determinism.

      sample_mean = tf.reduce_mean(samples, axis=0)
      sample_std = tf.sqrt(
          tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                         axis=0))
      [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, [0., 0.], atol=0.3, rtol=0.3)
    self.assertAllClose(sample_std_, [1., 1.], atol=0.1, rtol=0.1)

  def testInverseTemperaturesValueError(self):
    """Using invalid `inverse_temperatures`."""
    dtype = np.float32

    with self.assertRaises(ValueError) as cm:
      target = tfd.Normal(loc=dtype(0), scale=dtype(1))

      tfp.mcmc.ReplicaExchangeMC(
          target_log_prob_fn=target.log_prob,
          inverse_temperatures=10.**tf.linspace(
              0., -2., tf.random_uniform([], maxval=10, dtype=tf.int32)),
          make_kernel_fn=tfp.mcmc.HamiltonianMonteCarlo,
          step_size=1.0,
          num_leapfrog_steps=3,
          seed=42)
    the_exception = cm.exception
    tf.equal(the_exception.args[0],
             '"inverse_temperatures" must have statically known rank '
             'and statically known leading shape')


if __name__ == '__main__':
  tf.test.main()
