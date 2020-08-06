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
"""Tests for CovarianceReducer and VarianceReducer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class CovarianceReducersTest(test_util.TestCase):

  def test_zero_covariance(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    state = cov_reducer.initialize(0.)
    for _ in range(2):
      state = cov_reducer.one_step(0., state)
    final_num_samples, final_mean, final_cov = self.evaluate([
        state.cov_state.num_samples,
        state.cov_state.mean,
        cov_reducer.finalize(state)])
    self.assertEqual(2, final_num_samples)
    self.assertEqual(0, final_mean)
    self.assertEqual(0, final_cov)

  def test_random_sanity_check(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100)
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    state = cov_reducer.initialize(0.)
    for sample in x:
      state = cov_reducer.one_step(sample, state)
    final_mean, final_cov = self.evaluate([
        state.cov_state.mean,
        cov_reducer.finalize(state)])
    self.assertNear(np.mean(x), final_mean, err=1e-6)
    self.assertNear(np.var(x, ddof=0), final_cov, err=1e-6)

  def test_covariance_shape(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(event_ndims=1)
    state = cov_reducer.initialize(tf.ones((9, 3)))
    for _ in range(2):
      state = cov_reducer.one_step(
          tf.zeros((5, 9, 3)), state, axis=0)
    final_mean, final_cov = self.evaluate([
        state.cov_state.mean,
        cov_reducer.finalize(state)])
    self.assertEqual((9, 3), final_mean.shape)
    self.assertEqual((9, 3, 3), final_cov.shape)

  def test_variance_shape(self):
    var_reducer = tfp.experimental.mcmc.VarianceReducer()
    state = var_reducer.initialize(tf.ones((9, 3)))
    for _ in range(2):
      state = var_reducer.one_step(
          tf.zeros((5, 9, 3)), state, axis=0)
    final_mean, final_var = self.evaluate([
        state.cov_state.mean,
        var_reducer.finalize(state)])
    self.assertEqual((9, 3), final_mean.shape)
    self.assertEqual((9, 3), final_var.shape)

  def test_attributes(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(
        event_ndims=1, ddof=1)
    state = cov_reducer.initialize(tf.ones((2, 3), dtype=tf.float64))

    # check attributes are correct right after initialization
    parameters = dict(
        event_ndims=1, ddof=1, name='covariance_reducer'
    )
    self.assertEqual(1, cov_reducer.event_ndims)
    self.assertEqual(1, cov_reducer.ddof)
    self.assertEqual(parameters, cov_reducer.parameters)
    for _ in range(2):
      state = cov_reducer.one_step(
          tf.zeros((2, 3), dtype=tf.float64), state)

    # check attributes don't change after stepping through
    self.assertEqual(1, cov_reducer.event_ndims)
    self.assertEqual(1, cov_reducer.ddof)
    self.assertEqual(parameters, cov_reducer.parameters)

  def test_tf_while(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    state = cov_reducer.initialize(tf.ones((2, 3)))
    _, state = tf.while_loop(
        lambda i, _: i < 100,
        lambda i, state: (i + 1, cov_reducer.one_step(tf.ones((2, 3)), state)),
        (0., state)
    )
    final_cov = self.evaluate(cov_reducer.finalize(state))
    self.assertAllClose(tf.zeros((2, 3, 2, 3)), final_cov, rtol=1e-6)

  def test_nested_chain_state(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(event_ndims=0)
    chain_state = ({'one': tf.ones((2, 3)), 'zero': tf.zeros((2, 3))},
                   {'two': tf.ones((2, 3)) * 2})
    state = cov_reducer.initialize(chain_state)
    _, state = tf.while_loop(
        lambda i, _: i < 10,
        lambda i, state: (i + 1, cov_reducer.one_step(chain_state, state)),
        (0., state)
    )
    final_cov = self.evaluate(cov_reducer.finalize(state))
    self.assertAllEqualNested(
        final_cov, ({'one': tf.zeros((2, 3)), 'zero': tf.zeros((2, 3))},
                    {'two': tf.zeros((2, 3))}))

  def test_nested_with_batching_and_chunking(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(event_ndims=1)
    chain_state = ({'one': tf.ones((3, 4)), 'zero': tf.zeros((3, 4))},
                   {'two': tf.ones((3, 4)) * 2})
    state = cov_reducer.initialize(chain_state)
    _, state = tf.while_loop(
        lambda i, _: i < 10,
        lambda i, state: (i + 1, cov_reducer.one_step(chain_state, state, 0)),
        (0., state)
    )
    final_cov = self.evaluate(cov_reducer.finalize(state))
    self.assertAllEqualNested(
        final_cov, ({'one': tf.zeros((3, 4, 4)), 'zero': tf.zeros((3, 4, 4))},
                    {'two': tf.zeros((3, 4, 4))}))


if __name__ == '__main__':
  tf.test.main()
