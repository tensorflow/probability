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
"""Tests for `leapfrog_integrator.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl


@test_util.test_all_tf_execution_regimes
class LeapfrogIntegratorTest(test_util.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.

    tf.random.set_seed(10003)
    np.random.seed(10003)

  def assertAllFinite(self, x):
    self.assertAllEqual(np.ones_like(x).astype(bool), np.isfinite(x))

  def _log_gamma_log_prob(self, x, event_dims=()):
    """Computes log-pdf of a log-gamma random variable.

    Args:
      x: Value of the random variable.
      event_dims: Dimensions not to treat as independent.

    Returns:
      log_prob: The log-pdf up to a normalizing constant.
    """
    return tf.reduce_sum(
        self._shape_param * x - self._rate_param * tf.exp(x),
        axis=event_dims)

  def _integrator_conserves_energy(self, x, independent_chain_ndims, seed):
    event_dims = tf.range(independent_chain_ndims, tf.rank(x))

    target_fn = lambda x: self._log_gamma_log_prob(x, event_dims)

    m = tf.random.normal(tf.shape(x), seed=seed)
    log_prob_0 = target_fn(x)
    old_energy = -log_prob_0 + 0.5 * tf.reduce_sum(m**2., axis=event_dims)

    event_size = np.prod(
        self.evaluate(x).shape[independent_chain_ndims:])

    integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
        target_fn,
        step_sizes=[0.09 / event_size],
        num_steps=1000)

    [[new_m], [_], log_prob_1, [_]] = integrator([m], [x])

    new_energy = -log_prob_1 + 0.5 * tf.reduce_sum(new_m**2., axis=event_dims)

    old_energy_, new_energy_ = self.evaluate([old_energy, new_energy])
    tf1.logging.vlog(
        1, 'average energy relative change: {}'.format(
            (1. - new_energy_ / old_energy_).mean()))
    self.assertAllClose(old_energy_, new_energy_, atol=0., rtol=0.02)

  def _integrator_conserves_energy_wrapper(self, independent_chain_ndims):
    """Tests the long-term energy conservation of the leapfrog integrator.

    The leapfrog integrator is symplectic, so for sufficiently small step
    sizes it should be possible to run it more or less indefinitely without
    the energy of the system blowing up or collapsing.

    Args:
      independent_chain_ndims: Python `int` scalar representing the number of
        dims associated with independent chains.
    """
    seed_stream = test_util.test_seed_stream()
    x = self.evaluate(0.1 * tf.random.normal(
        shape=(50, 10, 2), seed=seed_stream()))
    x = tf.constant(x)
    self._integrator_conserves_energy(
        x, independent_chain_ndims, seed=seed_stream())

  def testIntegratorEnergyConservationNullShape(self):
    self._integrator_conserves_energy_wrapper(0)

  def testIntegratorEnergyConservation1(self):
    self._integrator_conserves_energy_wrapper(1)

  def testIntegratorEnergyConservation2(self):
    self._integrator_conserves_energy_wrapper(2)

  def testIntegratorEnergyConservation3(self):
    self._integrator_conserves_energy_wrapper(3)


if __name__ == '__main__':
  tf.test.main()
