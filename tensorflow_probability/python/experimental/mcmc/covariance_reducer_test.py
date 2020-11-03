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

import collections

# Dependency imports

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


FakeKernelResults = collections.namedtuple(
        'FakeKernelResults', 'value, inner_results')


FakeInnerResults = collections.namedtuple(
    'FakeInnerResults', 'value')


@test_util.test_all_tf_execution_regimes
class CovarianceReducersTest(test_util.TestCase):

  def test_zero_covariance(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = cov_reducer.initialize(0., fake_kr)
    for _ in range(2):
      state = cov_reducer.one_step(0., state, fake_kr)
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
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = cov_reducer.initialize(0., fake_kr)
    for sample in x:
      state = cov_reducer.one_step(sample, state, fake_kr)
    final_mean, final_cov = self.evaluate([
        state.cov_state.mean,
        cov_reducer.finalize(state)])
    self.assertNear(np.mean(x), final_mean, err=1e-6)
    self.assertNear(np.var(x, ddof=0), final_cov, err=1e-6)

  def test_covariance_shape(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(event_ndims=1)
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = cov_reducer.initialize(tf.ones((9, 3)), fake_kr)
    for _ in range(2):
      state = cov_reducer.one_step(
          tf.zeros((5, 9, 3)), state, fake_kr, axis=0)
    final_mean, final_cov = self.evaluate([
        state.cov_state.mean,
        cov_reducer.finalize(state)])
    self.assertEqual((9, 3), final_mean.shape)
    self.assertEqual((9, 3, 3), final_cov.shape)

  def test_variance_shape(self):
    var_reducer = tfp.experimental.mcmc.VarianceReducer()
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = var_reducer.initialize(tf.ones((9, 3)), fake_kr)
    for _ in range(2):
      state = var_reducer.one_step(
          tf.zeros((5, 9, 3)), state, fake_kr, axis=0)
    final_mean, final_var = self.evaluate([
        state.cov_state.mean,
        var_reducer.finalize(state)])
    self.assertEqual((9, 3), final_mean.shape)
    self.assertEqual((9, 3), final_var.shape)

  def test_attributes(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(
        event_ndims=1, ddof=1)
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = cov_reducer.initialize(
        tf.ones((2, 3), dtype=tf.float64), fake_kr)

    # check attributes are correct right after initialization
    self.assertEqual(1, cov_reducer.event_ndims)
    self.assertEqual(1, cov_reducer.ddof)
    for _ in range(2):
      state = cov_reducer.one_step(
          tf.zeros((2, 3), dtype=tf.float64), state, fake_kr)

    # check attributes don't change after stepping through
    self.assertEqual(1, cov_reducer.event_ndims)
    self.assertEqual(1, cov_reducer.ddof)

  def test_tf_while(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = cov_reducer.initialize(tf.ones((2, 3)), fake_kr)
    print(state)
    _, state = tf.while_loop(
        lambda i, _: i < 100,
        lambda i, s: (i + 1, cov_reducer.one_step(tf.ones((2, 3)), s, fake_kr)),
        (0., state)
    )
    final_cov = self.evaluate(cov_reducer.finalize(state))
    self.assertAllClose(tf.zeros((2, 3, 2, 3)), final_cov, rtol=1e-6)

  def test_nested_chain_state(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(event_ndims=0)
    chain_state = ({'one': tf.ones((2, 3)), 'zero': tf.zeros((2, 3))},
                   {'two': tf.ones((2, 3)) * 2})
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = cov_reducer.initialize(chain_state, fake_kr)
    _, state = tf.while_loop(
        lambda i, _: i < 10,
        lambda i, s: (i + 1, cov_reducer.one_step(chain_state, s, fake_kr)),
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
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = cov_reducer.initialize(chain_state, fake_kr)
    _, state = tf.while_loop(
        lambda i, _: i < 10,
        lambda i, s: (i + 1, cov_reducer.one_step(chain_state, s, fake_kr, 0)),
        (0., state)
    )
    final_cov = self.evaluate(cov_reducer.finalize(state))
    self.assertAllEqualNested(
        final_cov, ({'one': tf.zeros((3, 4, 4)), 'zero': tf.zeros((3, 4, 4))},
                    {'two': tf.zeros((3, 4, 4))}))

  def test_manual_variance_transform_fn(self):
    var_reducer = tfp.experimental.mcmc.VarianceReducer(
        transform_fn=lambda _, kr: kr.inner_results.value)
    fake_kr = FakeKernelResults(0., FakeInnerResults(
        tf.zeros((2, 3))))
    # chain state should be irrelevant
    state = var_reducer.initialize(0., fake_kr)
    for sample in range(5):
      fake_kr = FakeKernelResults(
          sample, FakeInnerResults(tf.ones((2, 3)) * sample))
      state = var_reducer.one_step(sample, state, fake_kr)
    final_mean, final_var = self.evaluate([
        state.cov_state.mean,
        var_reducer.finalize(state)])
    self.assertEqual((2, 3), final_mean.shape)
    self.assertAllEqual(np.ones((2, 3)) * 2, final_mean)
    self.assertEqual((2, 3), final_var.shape)
    self.assertAllEqual(np.ones((2, 3)) * 2, final_var)

  def test_manual_covariance_transform_fn_with_random_states(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 5, 2)
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(
        transform_fn=lambda _, kr: kr.inner_results.value)
    fake_kr = FakeKernelResults(0., FakeInnerResults(
        tf.zeros((5, 2))))
    state = cov_reducer.initialize(0., fake_kr)
    for sample in x:
      fake_kr = FakeKernelResults(0., FakeInnerResults(sample))
      state = cov_reducer.one_step(0., state, fake_kr)
    final_mean, final_cov = self.evaluate([
        state.cov_state.mean,
        cov_reducer.finalize(state)])

    # reshaping to be compatible with a check against numpy
    x_reshaped = x.reshape(100, 10)
    final_cov_reshaped = tf.reshape(final_cov, (10, 10))
    self.assertEqual((5, 2), final_mean.shape)
    self.assertAllClose(np.mean(x, axis=0), final_mean, rtol=1e-5)
    self.assertEqual((5, 2, 5, 2), final_cov.shape)
    self.assertAllClose(np.cov(x_reshaped.T, ddof=0),
                        final_cov_reshaped,
                        rtol=1e-5)

  def test_latent_state_with_multiple_transform_fn(self):
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(
        event_ndims=1,
        transform_fn=[
            lambda _, kr: kr.value,
            lambda _, kr: kr.inner_results.value
        ])
    latent_state = ({'one': tf.ones((3, 4)), 'zero': tf.zeros((3, 4))},
                    {'two': tf.ones((3, 4)) * 2})
    fake_kr = FakeKernelResults(latent_state, FakeInnerResults(latent_state))
    state = cov_reducer.initialize(latent_state, fake_kr)
    _, state = tf.while_loop(
        lambda i, _: i < 10,
        lambda i, state: (i + 1, cov_reducer.one_step(0., state, fake_kr)),
        (0., state)
    )
    final_cov = self.evaluate(cov_reducer.finalize(state))
    cov_latent = ({'one': tf.zeros((3, 4, 4)), 'zero': tf.zeros((3, 4, 4))},
                  {'two': tf.zeros((3, 4, 4))})
    self.assertAllEqualNested(
        final_cov[0], cov_latent)
    self.assertAllEqualNested(
        final_cov[1], cov_latent)

  def test_transform_fn_with_nested_return(self):
    cov_red = tfp.experimental.mcmc.CovarianceReducer(
        transform_fn=lambda sample, _: [sample, sample + 1])
    fake_kr = FakeKernelResults(0., FakeInnerResults(0))
    state = cov_red.initialize(tf.zeros((2,)), fake_kr)
    _, state = tf.while_loop(
        lambda sample, _: sample < 5,
        lambda s, st: (s + 1, cov_red.one_step(tf.ones((2,)) * s, st, fake_kr)),
        (0., state)
    )
    final_cov = self.evaluate(cov_red.finalize(state))
    self.assertEqual(2, len(final_cov))
    self.assertAllEqual(np.ones((2, 2, 2)) * 2, final_cov)


if __name__ == '__main__':
  tf.test.main()
