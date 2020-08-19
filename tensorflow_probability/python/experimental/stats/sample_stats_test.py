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
"""Tests for Sample Stats Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RunningCovarianceTest(test_util.TestCase):

  def test_zero_running_variance(self):
    deterministic_samples = [0., 0., 0., 0.]
    running_var = tfp.experimental.stats.RunningVariance()
    state = running_var.initialize()
    for sample in deterministic_samples:
      state = running_var.update(state, sample)
    final_mean, final_var = self.evaluate([
        state.mean,
        running_var.finalize(state)])
    self.assertEqual(final_mean, 0.)
    self.assertEqual(final_var, 0.)

  @parameterized.parameters(0, 1)
  def test_running_variance(self, ddof):
    rng = test_util.test_np_rng()
    x = rng.rand(100)
    running_var = tfp.experimental.stats.RunningVariance()
    state = running_var.initialize()
    for sample in x:
      state = running_var.update(state, sample)
    final_mean, final_var = self.evaluate([
        state.mean,
        running_var.finalize(state, ddof=ddof)])
    self.assertNear(np.mean(x), final_mean, err=1e-6)
    self.assertNear(np.var(x, ddof=ddof), final_var, err=1e-6)

  def test_integer_running_covariance(self):
    running_cov = tfp.experimental.stats.RunningCovariance(
        shape=(), dtype=tf.int32
    )
    state = running_cov.initialize()
    for sample in range(5):
      state = running_cov.update(state, sample)
    final_cov = running_cov.finalize(state)
    # all int dtypes are converted to `tf.float32`
    self.assertEqual(tf.float32, state.mean.dtype)
    self.assertEqual(tf.float32, final_cov.dtype)
    final_mean, final_cov = self.evaluate([state.mean, final_cov])
    self.assertNear(2, final_mean, err=1e-6)
    self.assertNear(2, final_cov, err=1e-6)

  def test_higher_rank_running_variance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 5, 2)
    running_var = tfp.experimental.stats.RunningVariance(tf.TensorShape([5, 2]))
    state = running_var.initialize()
    for sample in x:
      state = running_var.update(state, sample)
    final_mean, final_var = self.evaluate([
        state.mean,
        running_var.finalize(state)])
    self.assertAllClose(np.mean(x, axis=0), final_mean, rtol=1e-5)
    self.assertEqual(final_var.shape, (5, 2))

    # reshaping to be compatible with a check against numpy
    x_reshaped = x.reshape(100, 10)
    final_var_reshape = tf.reshape(final_var, (10,))
    self.assertAllClose(np.var(x_reshaped, axis=0),
                        final_var_reshape,
                        rtol=1e-5)

  def test_chunked_running_variance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 2, 5)
    running_var = tfp.experimental.stats.RunningVariance((5,))
    state = running_var.initialize()
    for sample in x:
      state = running_var.update(state, sample, axis=0)
    final_mean, final_var = self.evaluate([
        state.mean,
        running_var.finalize(state)])
    self.assertAllClose(
        np.mean(x.reshape(200, 5), axis=0),
        final_mean,
        rtol=1e-5)
    self.assertEqual(final_var.shape, (5,))

    # reshaping to be compatible with a check against numpy
    x_reshaped = x.reshape(200, 5)
    self.assertAllClose(np.var(x_reshaped, axis=0),
                        final_var,
                        rtol=1e-5)

  def test_dynamic_shape_running_variance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 2, 5)
    running_var = tfp.experimental.stats.RunningVariance((5,), tf.float64)
    state = running_var.initialize()
    for sample in x:
      if not tf.executing_eagerly():
        sample = tf1.placeholder_with_default(sample, shape=None)
      state = running_var.update(state, sample, axis=0)
    final_var = self.evaluate(running_var.finalize(state))
    x_reshaped = x.reshape(200, 5)
    self.assertEqual(final_var.shape, (5,))
    self.assertAllClose(np.var(x_reshaped, axis=0),
                        final_var,
                        rtol=1e-5)

  def test_running_covariance_as_variance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 5, 2)
    running_cov = tfp.experimental.stats.RunningCovariance(
        tf.TensorShape([5, 2]),
        event_ndims=0)
    running_var = tfp.experimental.stats.RunningVariance(
        tf.TensorShape([5, 2])
    )
    cov_state = running_cov.initialize()
    var_state = running_var.initialize()
    for sample in x:
      cov_state = running_cov.update(cov_state, sample)
      var_state = running_var.update(var_state, sample)
    final_cov, final_var = self.evaluate([
        running_cov.finalize(cov_state),
        running_var.finalize(var_state)])
    self.assertEqual(final_cov.shape, (5, 2))
    self.assertAllClose(final_cov, final_var, rtol=1e-5)

  def test_zero_running_covariance(self):
    fake_samples = [[0., 0.] for _ in range(2)]
    running_cov = tfp.experimental.stats.RunningCovariance((2,))
    state = running_cov.initialize()
    for sample in fake_samples:
      state = running_cov.update(state, sample)
    final_mean, final_cov = self.evaluate([
        state.mean,
        running_cov.finalize(state)])
    self.assertAllClose([0., 0.], final_mean, rtol=1e-5)
    self.assertAllCloseNested(np.zeros((2, 2)), final_cov)

  @parameterized.parameters(0, 1)
  def test_running_covariance(self, ddof):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    running_cov = tfp.experimental.stats.RunningCovariance((10,))
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample)
    final_mean, final_cov = self.evaluate([
        state.mean,
        running_cov.finalize(state, ddof=ddof)])
    self.assertAllClose(np.mean(x, axis=0), final_mean, rtol=1e-5)
    self.assertAllClose(np.cov(x.T, ddof=ddof), final_cov, rtol=1e-5)

  def test_higher_rank_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 5, 2)
    running_cov = tfp.experimental.stats.RunningCovariance(
        tf.TensorShape([5, 2]))
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample)
    final_mean, final_cov = self.evaluate([
        state.mean,
        running_cov.finalize(state)])
    self.assertAllClose(np.mean(x, axis=0), final_mean, rtol=1e-5)
    self.assertEqual(final_cov.shape, (5, 2, 5, 2))

    # reshaping to be compatible with a check against numpy
    x_reshaped = x.reshape(100, 10)
    final_cov_reshaped = tf.reshape(final_cov, (10, 10))
    self.assertAllClose(np.cov(x_reshaped.T, ddof=0),
                        final_cov_reshaped,
                        rtol=1e-5)

  def test_chunked_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 2, 3, 5)
    running_cov = tfp.experimental.stats.RunningCovariance((3, 5))
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample, axis=0)
    final_mean, final_cov = self.evaluate([
        state.mean,
        running_cov.finalize(state)])
    self.assertAllClose(
        np.mean(x.reshape((200, 3, 5)), axis=0),
        final_mean,
        rtol=1e-5)
    self.assertEqual(final_cov.shape, (3, 5, 3, 5))

    # reshaping to be compatible with a check against numpy
    x_reshaped = x.reshape(200, 15)
    final_cov_reshaped = tf.reshape(final_cov, (15, 15))
    self.assertAllClose(np.cov(x_reshaped.T, ddof=0),
                        final_cov_reshaped,
                        rtol=1e-5)

  def test_running_covariance_with_event_ndims(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 3, 5, 2)
    running_cov = tfp.experimental.stats.RunningCovariance(
        tf.TensorShape([5, 2]),
        event_ndims=1)
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample, axis=0)
    final_mean, final_cov = self.evaluate([
        state.mean,
        running_cov.finalize(state)])
    self.assertAllClose(
        np.mean(x.reshape(300, 5, 2), axis=0),
        final_mean,
        rtol=1e-5)
    self.assertEqual(final_cov.shape, (5, 2, 2))

    # manual computation with loops
    manual_cov = np.zeros((5, 2, 2))
    x_reshaped = x.reshape((300, 5, 2))
    delta_mean = x_reshaped - np.mean(x_reshaped, axis=0)
    for residual in delta_mean:
      for i in range(5):
        for j in range(2):
          for k in range(2):
            manual_cov[i][j][k] += residual[i][j] * residual[i][k]
    manual_cov /= 300
    self.assertAllClose(manual_cov, final_cov, rtol=1e-5)

  def test_batched_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 3, 5, 2)
    running_cov = tfp.experimental.stats.RunningCovariance(
        tf.TensorShape([3, 5, 2]),
        event_ndims=2)
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample)
    final_mean, final_cov = self.evaluate([
        state.mean,
        running_cov.finalize(state)])
    self.assertAllClose(np.mean(x, axis=0), final_mean, rtol=1e-5)
    self.assertEqual(final_cov.shape, (3, 5, 2, 5, 2))

    # check against numpy
    x_reshaped = x.reshape((100, 3, 10))
    for i in range(x_reshaped.shape[1]):
      np_cov = np.cov(x_reshaped[:, i, :].T, ddof=0).reshape((5, 2, 5, 2))
      self.assertAllClose(np_cov, final_cov[i], rtol=1e-5)

  def test_dynamic_shape_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 3, 5, 2)
    running_cov = tfp.experimental.stats.RunningCovariance(
        tf.TensorShape([5, 2]),
        event_ndims=1)
    state = running_cov.initialize()
    for sample in x:
      if not tf.executing_eagerly():
        sample = tf1.placeholder_with_default(sample, shape=None)
      state = running_cov.update(state, sample, axis=0)
    final_mean, final_cov = self.evaluate([
        state.mean,
        running_cov.finalize(state)])
    self.assertAllClose(
        np.mean(x.reshape(300, 5, 2), axis=0),
        final_mean,
        rtol=1e-5)
    self.assertEqual(final_cov.shape, (5, 2, 2))

    # manual computation with loops
    manual_cov = np.zeros((5, 2, 2))
    x_reshaped = x.reshape((300, 5, 2))
    delta_mean = x_reshaped - np.mean(x_reshaped, axis=0)
    for residual in delta_mean:
      for i in range(5):
        for j in range(2):
          for k in range(2):
            manual_cov[i][j][k] += residual[i][j] * residual[i][k]
    manual_cov /= 300
    self.assertAllClose(manual_cov, final_cov, rtol=1e-5)

  def test_manual_dtype(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    running_cov = tfp.experimental.stats.RunningCovariance(
        (10,), dtype=tf.float64)
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample)
    final_cov = running_cov.finalize(state)
    self.assertEqual(final_cov.dtype, tf.float64)

  def test_shift_in_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10) * 10
    running_cov = tfp.experimental.stats.RunningCovariance((10,))
    cov_state = running_cov.initialize()
    running_shifted_cov = tfp.experimental.stats.RunningCovariance((10,))
    shifted_cov_state = running_shifted_cov.initialize()
    for sample in x:
      cov_state = running_cov.update(cov_state, sample)
      shifted_cov_state = running_shifted_cov.update(
          shifted_cov_state, sample + 1e4)
    final_cov, final_shifted_cov = self.evaluate([
        running_cov.finalize(cov_state),
        running_shifted_cov.finalize(shifted_cov_state)])
    self.assertAllClose(final_cov, np.cov(x.T, ddof=0), rtol=1e-5)
    self.assertAllClose(
        final_shifted_cov, np.cov(x.T, ddof=0), rtol=1e-1)

  def test_sorted_ascending_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    x.sort(axis=0)
    running_cov = tfp.experimental.stats.RunningCovariance((10,))
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample)
    final_cov = self.evaluate(
        running_cov.finalize(state))
    self.assertAllClose(final_cov, np.cov(x.T, ddof=0), rtol=1e-5)

  def test_sorted_descending_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    x[::-1].sort(axis=0)  # sorts in descending order
    running_cov = tfp.experimental.stats.RunningCovariance((10,))
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample)
    final_cov = self.evaluate(
        running_cov.finalize(state))
    self.assertAllClose(final_cov, np.cov(x.T, ddof=0), rtol=1e-5)

  def test_attributes(self):
    rng = test_util.test_np_rng()
    x = rng.rand(2, 3, 10)
    running_cov = tfp.experimental.stats.RunningCovariance(
        (3, 10,), event_ndims=1)
    state = running_cov.initialize()
    for sample in x:
      state = running_cov.update(state, sample)
    # the state holds information relevant to it including `num_samples`
    self.assertEqual(
        self.evaluate(state.num_samples), 2.)
    # metadata is stored in the `RunningCovariance` object
    self.assertEqual(running_cov.event_ndims, 1)
    self.assertEqual(running_cov.dtype, tf.float32)

  def test_tf_while(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)
    running_cov = tfp.experimental.stats.RunningCovariance((10,))
    running_var = tfp.experimental.stats.RunningVariance((10,))
    _, cov_state = tf.while_loop(
        lambda i, _: i < 100,
        lambda i, state: (i + 1, running_cov.update(state, tensor_x[i])),
        (0, running_cov.initialize()))
    final_cov = running_cov.finalize(cov_state)
    _, var_state = tf.while_loop(
        lambda i, _: i < 100,
        lambda i, state: (i + 1, running_var.update(state, tensor_x[i])),
        (0, running_var.initialize()))
    final_var = running_var.finalize(var_state)
    self.assertAllClose(final_cov, np.cov(x.T, ddof=0), rtol=1e-5)
    self.assertAllClose(final_var, np.var(x, axis=0), rtol=1e-5)

  def test_tf_while_cov_with_dynamic_shape(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)
    if not tf.executing_eagerly():
      tensor_x = tf1.placeholder_with_default(tensor_x, shape=None)
    running_cov = tfp.experimental.stats.RunningCovariance(shape=(10,))
    def _loop_body(i, state):
      sample = tensor_x[i]
      return (i + 1, running_cov.update(state, sample))
    _, state = tf.while_loop(
        lambda i, _: i < 100,
        _loop_body,
        (tf.constant(0, dtype=tf.int32), running_cov.initialize()),
        shape_invariants=(
            None, tfp.experimental.stats.RunningCovarianceState(
                None,
                tf.TensorShape(None),
                tf.TensorShape(None)
            ))
    )
    final_mean, final_cov = self.evaluate([
        state.mean, running_cov.finalize(state)])
    self.assertEqual(final_mean.shape, (10,))
    self.assertEqual(final_cov.shape, (10, 10))
    self.assertAllClose(final_cov, np.cov(x.T, ddof=0), rtol=1e-5)

  def test_tf_while_var_with_dynamic_shape(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)
    running_var = tfp.experimental.stats.RunningVariance((10,))
    def _loop_body(i, state):
      if not tf.executing_eagerly():
        sample = tf1.placeholder_with_default(tensor_x[i], shape=None)
      else:
        sample = tensor_x[i]
      return (i + 1, running_var.update(state, sample))
    _, state = tf.while_loop(
        lambda i, _: i < 100,
        _loop_body,
        (tf.constant(0, dtype=tf.int32), running_var.initialize()),
        shape_invariants=(
            None, tfp.experimental.stats.RunningCovarianceState(
                None,
                tf.TensorShape(None),
                tf.TensorShape(None)
            ))
    )
    final_mean, final_var = self.evaluate([
        state.mean, running_var.finalize(state)])
    self.assertEqual(final_mean.shape, (10,))
    self.assertEqual(final_var.shape, (10,))
    self.assertAllClose(final_var, np.var(x, axis=0), rtol=1e-5)


@test_util.test_all_tf_execution_regimes
class RunningRhatTest(test_util.TestCase):

  def test_simple_operation(self):
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction(
        shape=(),
        num_chains=3,
    )
    state = running_rhat.initialize()
    # 5 samples from 3 independent Markov chains
    x = np.arange(15, dtype=np.float32).reshape((5, 3))
    for sample in x:
      state = running_rhat.update(state, sample)
    rhat = running_rhat.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x,
        independent_chain_ndims=1,
    )
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertNear(true_rhat, rhat, err=1e-6)

  def test_random_scalar_computation(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10) * 100
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction(
        shape=(),
        num_chains=10,
    )
    state = running_rhat.initialize()
    for sample in x:
      state = running_rhat.update(state, sample)
    rhat = running_rhat.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x,
        independent_chain_ndims=1,
    )
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertNear(true_rhat, rhat, err=1e-6)

  def test_non_scalar_samples(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 2, 2, 3, 5) * 100
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction(
        shape=(2, 3, 5),
        num_chains=2,
    )
    state = running_rhat.initialize()
    for sample in x:
      state = running_rhat.update(state, sample)
    rhat = running_rhat.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x,
        independent_chain_ndims=1,
    )
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_chunking(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10, 2, 5) * 100
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction(
        shape=(5,),
        num_chains=2,
    )
    state = running_rhat.initialize()
    for sample in x:
      state = running_rhat.update(
          state, sample, axis=0)
    rhat = running_rhat.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x.reshape(1000, 2, 5),
        independent_chain_ndims=1,
    )
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_batching(self):
    n_samples = 100
    # state_0 is two scalar chains taken from iid Normal(0, 1).
    state_0 = np.random.randn(n_samples, 2)

    # state_1 is three 4-variate chains taken from Normal(0, 1) that have been
    # shifted.
    offset = np.array([1., -1., 2.]).reshape(3, 1)
    state_1 = np.random.randn(n_samples, 3, 4) + offset
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction(
        shape=[(), (4,)],
        num_chains=[2, 3]
    )
    state = running_rhat.initialize()
    for sample in zip(state_0, state_1):
      state = running_rhat.update(state, sample)
    rhat = self.evaluate(running_rhat.finalize(state))
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=[state_0, state_1], independent_chain_ndims=1)
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_tf_while(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10) * 100
    tensor_x = tf.convert_to_tensor(x)
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction(
        shape=(),
        num_chains=10,
    )
    state = running_rhat.initialize()
    def _loop_body(i, state):
      new_state = running_rhat.update(state, tensor_x[i])
      return i + 1, new_state
    _, state = tf.while_loop(
        lambda i, _: i < 100,
        _loop_body,
        (0, state)
    )
    rhat = running_rhat.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x,
        independent_chain_ndims=1,
    )
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertNear(true_rhat, rhat, err=1e-6)


@test_util.test_all_tf_execution_regimes
class RunningMeanTest(test_util.TestCase):

  def test_zero_mean(self):
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(),
    )
    state = running_mean.initialize()
    for _ in range(6):
      state = running_mean.update(state, 0)
    mean = self.evaluate(running_mean.finalize(state))
    self.assertEqual(0, mean)

  def test_higher_rank_shape(self):
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(5, 3),
    )
    state = running_mean.initialize()
    for sample in range(6):
      state = running_mean.update(state, tf.ones((5, 3)) * sample)
    mean = self.evaluate(running_mean.finalize(state))
    self.assertAllEqual(np.ones((5, 3)) * 2.5, mean)

  def test_manual_dtype(self):
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(),
        dtype=tf.float64,
    )
    state = running_mean.initialize()
    for _ in range(6):
      state = running_mean.update(state, 0)
    mean = running_mean.finalize(state)
    self.assertEqual(tf.float64, mean.dtype)

  def test_integer_dtype(self):
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(),
        dtype=tf.int32,
    )
    state = running_mean.initialize()
    for sample in range(6):
      state = running_mean.update(state, sample)
    mean = running_mean.finalize(state)
    self.assertEqual(tf.float32, mean.dtype)
    mean = self.evaluate(mean)
    self.assertEqual(2.5, mean)

  def test_random_mean(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100)
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(),
    )
    state = running_mean.initialize()
    for sample in x:
      state = running_mean.update(state, sample)
    mean = self.evaluate(running_mean.finalize(state))
    self.assertAllClose(np.mean(x), mean, rtol=1e-6)

  def test_chunking(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10, 5)
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(5,),
    )
    state = running_mean.initialize()
    for sample in x:
      state = running_mean.update(state, sample, axis=0)
    mean = self.evaluate(running_mean.finalize(state))
    self.assertAllClose(np.mean(x.reshape(1000, 5), axis=0), mean, rtol=1e-6)

  def test_tf_while(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(10,))
    _, state = tf.while_loop(
        lambda i, _: i < 100,
        lambda i, state: (i + 1, running_mean.update(state, tensor_x[i])),
        (0, running_mean.initialize()))
    mean = self.evaluate(running_mean.finalize(state))
    self.assertAllClose(np.mean(x, axis=0), mean, rtol=1e-6)

  def test_tf_while_with_dynamic_shape(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(10,))

    def _loop_body(i, state):
      if not tf.executing_eagerly():
        sample = tf1.placeholder_with_default(tensor_x[i], shape=None)
      else:
        sample = tensor_x[i]
      return (i + 1, running_mean.update(state, sample))

    _, state = tf.while_loop(
        lambda i, _: i < 100,
        _loop_body,
        (tf.constant(0, dtype=tf.int32), running_mean.initialize()),
        shape_invariants=(
            None, tfp.experimental.stats.RunningMeanState(
                None,
                tf.TensorShape(None),
            )))
    mean = self.evaluate(running_mean.finalize(state))
    self.assertAllClose(np.mean(x, axis=0), mean, rtol=1e-6)

  def test_no_inputs(self):
    running_mean = tfp.experimental.stats.RunningMean(
        shape=(),
    )
    state = running_mean.initialize()
    mean = self.evaluate(running_mean.finalize(state))
    self.assertEqual(0, mean)


if __name__ == '__main__':
  tf.test.main()
