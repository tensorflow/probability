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
"""Tests of the No U-Turn Sampler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions

flags.DEFINE_string('test_device', None,
                    'TensorFlow device on which to place operators under test')
FLAGS = flags.FLAGS


def run_nuts_chain(event_size, batch_size, num_steps):
  def f():
    def target_log_prob_fn(event):
      return tfd.MultivariateNormalDiag(
          tf.zeros(event_size),
          scale_identity_multiplier=1.).log_prob(event)

    state = tf.zeros([batch_size, event_size])
    chain_state, extra = tfp.mcmc.sample_chain(
        num_results=num_steps,
        num_burnin_steps=0,
        current_state=[state],
        kernel=tfp.experimental.mcmc.NoUTurnSampler(
            target_log_prob_fn,
            step_size=[0.3],
            use_auto_batching=True,
            seed=1,
            backend=tf_backend.TensorFlowBackend(
                safety_checks=False, while_parallel_iterations=1)),
        parallel_iterations=1)
    return chain_state, extra.leapfrogs_taken
  return f


class NutsXLATest(test_util.TestCase):

  def testMultivariateNormalNd(self, event_size=32, batch_size=8, num_steps=2):
    tf.set_random_seed(3)
    with tf.device(FLAGS.test_device):
      f = run_nuts_chain(event_size, batch_size, num_steps)
      f = tf.function(f, autograph=False, experimental_compile=True)
      samples, leapfrogs = self.evaluate(f())

    # TODO(axch) Figure out what the right thing to test about the leapfrog
    # count really is and test it, instead of just flailing around like this
    # does.
    print(type(samples), type(leapfrogs))
    print(samples, leapfrogs)
    ev_leapfrogs = leapfrogs[0]
    self.assertGreater(len(set(ev_leapfrogs.tolist())), 1)
    self.assertTrue(all(ev_leapfrogs > 1))

  def test_xla_compile_and_correctness(self):
    tf.set_random_seed(3)
    nsamples = 1000
    nchains = 10
    nd = 5
    theta0 = np.zeros((nchains, nd))
    mu = np.arange(nd)
    stddev = np.arange(nd) + 1.
    rng = np.random.RandomState(seed=4)
    step_size = rng.rand(nchains, 1)*.1 + 1

    num_steps = nsamples
    initial_state = tf.cast(theta0, dtype=tf.float32)
    unrolled_leapfrog_steps = 2

    @tf.function(autograph=False)
    def target_log_prob_fn(event):
      with tf.name_scope('nuts_test_target_log_prob'):
        return tfd.MultivariateNormalDiag(
            loc=tf.cast(mu, dtype=tf.float32),
            scale_diag=tf.cast(stddev, dtype=tf.float32)).log_prob(event)

    @tf.function(autograph=False, experimental_compile=True)
    def _run_nuts_chain():
      kernel = tfp.mcmc.NoUTurnSampler(
          target_log_prob_fn,
          step_size=[tf.cast(step_size, dtype=tf.float32)],
          seed=9,
          unrolled_leapfrog_steps=unrolled_leapfrog_steps)
      [x], (is_accepted, leapfrogs_taken) = tfp.mcmc.sample_chain(
          num_results=num_steps,
          num_burnin_steps=0,
          current_state=[initial_state],
          kernel=kernel,
          trace_fn=lambda _, pkr: (pkr.is_accepted, pkr.leapfrogs_taken),
          parallel_iterations=1)
      return (
          tf.reduce_mean(x, axis=[0, 1]),
          tf.math.reduce_std(x, axis=[0, 1]),
          is_accepted,
          leapfrogs_taken
          )

    with tf.device(FLAGS.test_device):
      [
          sample_mean, sample_stddev, is_accepted, leapfrogs_taken
      ] = self.evaluate(_run_nuts_chain())

    self.assertAllClose(mu, sample_mean, atol=0.1, rtol=0.1)
    self.assertAllClose(stddev, sample_stddev, atol=0.15, rtol=0.15)
    # Test early stopping in tree building
    self.assertTrue(np.any(np.isin(
        np.asarray([5, 9, 11, 13]) * unrolled_leapfrog_steps,
        np.unique(leapfrogs_taken[is_accepted]))))


if __name__ == '__main__':
  tf.test.main()
