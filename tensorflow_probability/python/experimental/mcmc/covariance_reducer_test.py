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

import collections

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.mcmc import covariance_reducer
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import test_util


FakeKernelResults = collections.namedtuple(
        'FakeKernelResults', 'value, inner_results')


FakeInnerResults = collections.namedtuple(
    'FakeInnerResults', 'value')


@test_util.test_all_tf_execution_regimes
class CovarianceReducersTest(test_util.TestCase):

  def test_random_sanity_check(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100)
    cov_reducer = covariance_reducer.CovarianceReducer()
    final_cov = self.evaluate(test_fixtures.reduce(cov_reducer, x))
    self.assertNear(np.var(x, ddof=0), final_cov, err=1e-6)

  def test_covariance_shape_and_zero_covariance(self):
    cov_reducer = covariance_reducer.CovarianceReducer(event_ndims=1)
    state = cov_reducer.initialize(tf.ones((9, 3)))
    for _ in range(2):
      state = cov_reducer.one_step(tf.zeros((5, 9, 3)), state, axis=0)
    final_num_samples, final_mean, final_cov = self.evaluate([
        state.cov_state.num_samples,
        state.cov_state.mean,
        cov_reducer.finalize(state)])
    self.assertEqual((9, 3), final_mean.shape)
    self.assertEqual((9, 3, 3), final_cov.shape)
    self.assertEqual(10, final_num_samples)
    self.assertAllEqual(tf.zeros((9, 3)), final_mean)
    self.assertAllEqual(tf.zeros((9, 3, 3)), final_cov)

  def test_attributes(self):
    cov_reducer = covariance_reducer.CovarianceReducer(event_ndims=1, ddof=1)
    state = cov_reducer.initialize(tf.ones((2, 3), dtype=tf.float64))

    # check attributes are correct right after initialization
    self.assertEqual(1, cov_reducer.event_ndims)
    self.assertEqual(1, cov_reducer.ddof)
    for _ in range(2):
      state = cov_reducer.one_step(tf.zeros((2, 3), dtype=tf.float64), state)

    # check attributes don't change after stepping through
    self.assertEqual(1, cov_reducer.event_ndims)
    self.assertEqual(1, cov_reducer.ddof)

  def test_nested_with_batching_and_chunking(self):
    cov_reducer = covariance_reducer.CovarianceReducer(event_ndims=1)
    chain_state = ({'one': tf.ones((3, 4)), 'zero': tf.zeros((3, 4))},
                   {'two': tf.ones((3, 4)) * 2})
    state = cov_reducer.initialize(chain_state)
    _, state = tf.while_loop(
        lambda i, _: i < 10,
        lambda i, s: (i + 1, cov_reducer.one_step(chain_state, s, 0)),
        (0., state)
    )
    final_cov = self.evaluate(cov_reducer.finalize(state))
    self.assertAllEqualNested(
        final_cov, ({'one': tf.zeros((3, 4, 4)), 'zero': tf.zeros((3, 4, 4))},
                    {'two': tf.zeros((3, 4, 4))}))

  def test_latent_state_with_multiple_transform_fn(self):
    cov_reducer = covariance_reducer.CovarianceReducer(
        event_ndims=1,
        transform_fn=[
            lambda _, kr: kr.value, lambda _, kr: kr.inner_results.value
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
    self.assertAllEqualNested(final_cov[0], cov_latent)
    self.assertAllEqualNested(final_cov[1], cov_latent)

  def test_transform_fn_with_nested_return(self):
    cov_red = covariance_reducer.CovarianceReducer(
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

  @test_util.numpy_disable_test_missing_functionality('composite tensors')
  def test_composite_kernel_results(self):
    kr = normal.Normal(0., 1.)
    cov_red = covariance_reducer.CovarianceReducer()
    state = cov_red.initialize(tf.zeros((2,)), kr)
    state = cov_red.one_step(tf.ones((2,)), state, kr)
    cov_red.finalize(state)


if __name__ == '__main__':
  test_util.main()
