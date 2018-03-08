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
"""Tests for `MetropolisHastings` `TransitionKernel`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import random_seed


class MetropolisHastingsTest(tf.test.TestCase):

  def setUp(self):
    random_seed.set_random_seed(10003)
    np.random.seed(10003)

  # TODO(b/74154679): Create Fake TransitionKernel and not rely on HMC tests.

  def testCorrectlyQueriesInnerKernel(self):
    with self.test_session(graph=tf.Graph()) as sess:
      x0 = np.float32([-1., 0., 1])
      hmc = tfp.mcmc.MetropolisHastings(
          tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
              target_log_prob_fn=lambda x: -x - x**2,
              step_size=0.1,
              num_leapfrog_steps=3,
              seed=1),
          seed=1)
      uncal_hmc = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
          target_log_prob_fn=lambda x: -x - x**2,
          step_size=0.1,
          num_leapfrog_steps=3,
          seed=1)
      _, pkr1 = hmc.one_step(x0, hmc.bootstrap_results(x0))
      _, pkr2 = uncal_hmc.one_step(x0, uncal_hmc.bootstrap_results(x0))
      [pkr1_, pkr2_] = sess.run([pkr1, pkr2])
      self.assertAllClose(pkr1_.proposed_results, pkr2_)


if __name__ == '__main__':
  tf.test.main()
