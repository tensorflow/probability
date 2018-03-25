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


class ReplicaExchangeMCTest(tf.test.TestCase):
  def testDocstringExample(self):
    """Tests the simplified docstring example."""
    # Tuning acceptance rates:
    dtype = np.float32
    num_warmup_iter = 1000
    num_chain_iter = 1000

    x = tf.get_variable(name='x', initializer=np.zeros(2, dtype=dtype))

    # Target distribution is Mixture Normal.
    target = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-5., -5], [5., 5.]],
            scale_identity_multiplier=[1., 1.]))

    # Initialize the ReplicaExchangeMC sampler.
    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target.log_prob,
        inverse_temperatures=tf.pow(10., tf.linspace(0., -2., 5)),
        replica_kernel_class=tfp.mcmc.HamiltonianMonteCarlo,
        step_size=0.5,
        num_leapfrog_steps=3)

    # One iteration of the ReplicaExchangeMC
    init_results = remc.bootstrap_results(x)
    next_x, other_results = remc.one_step(
        current_state=x,
        previous_kernel_results=init_results)

    x_update = x.assign(next_x)
    replica_update = [init_results.replica_states[i].assign(
        other_results.replica_states[i]) for i in range(remc.n_replica)]

    warmup = tf.group([x_update, replica_update])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      # Warm up the sampler
      for _ in range(num_warmup_iter):
        sess.run(warmup)
      # Collect samples
      samples = np.zeros([num_chain_iter, 2])
      replica_samples = np.zeros([num_chain_iter, 5, 2])
      for i in range(num_chain_iter):
        _, x_, replica_x_ = sess.run([x_update, x, replica_update])
        samples[i] = x_
        replica_samples[i] = replica_x_


if __name__ == '__main__':
  tf.test.main()
