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

# Dependency imports

from absl.testing import parameterized

import numpy as np
import scipy.stats as stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


def consume(running_stat, elems, chunk_axis=None):
  def body(running_stat, elem):
    if chunk_axis is None:
      return running_stat.update(elem)
    else:
      return running_stat.update(elem, axis=chunk_axis)
  return tf.foldl(body, elems, running_stat)


@test_util.test_all_tf_execution_regimes
class RunningCovarianceTest(test_util.TestCase):

  def test_from_stats(self):
    num_counts = 10.
    mean = 1.
    variance = 3.
    var = tfp.experimental.stats.RunningVariance.from_stats(
        num_counts, mean, variance)
    self.assertEqual(self.evaluate(var.mean), mean)
    self.assertEqual(self.evaluate(var.variance()), variance)

  @parameterized.parameters(0, 1)
  def test_running_variance(self, ddof):
    rng = test_util.test_np_rng()
    x = rng.rand(100)
    var = tfp.experimental.stats.RunningVariance.from_shape()
    var = consume(var, x)
    final_mean, final_var = self.evaluate([var.mean, var.variance(ddof=ddof)])
    self.assertNear(np.mean(x), final_mean, err=1e-6)
    self.assertNear(np.var(x, ddof=ddof), final_var, err=1e-6)

  def test_integer_running_covariance(self):
    cov = tfp.experimental.stats.RunningCovariance.from_shape(
        shape=(), dtype=tf.int32)
    for sample in range(5):
      cov = cov.update(sample)
    final_cov = cov.covariance()
    # all int dtypes are converted to `tf.float32`
    self.assertEqual(tf.float32, cov.mean.dtype)
    self.assertEqual(tf.float32, final_cov.dtype)
    final_mean, final_cov = self.evaluate([cov.mean, final_cov])
    self.assertNear(2, final_mean, err=1e-6)
    self.assertNear(2, final_cov, err=1e-6)

  def test_chunked_higher_rank_running_variance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 2, 5)
    var = tfp.experimental.stats.RunningVariance.from_shape((5,))
    var = consume(var, x, chunk_axis=0)
    final_mean, final_var = self.evaluate([var.mean, var.variance()])
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

  def test_zero_running_covariance(self):
    fake_samples = [[0., 0.] for _ in range(2)]
    cov = tfp.experimental.stats.RunningCovariance.from_shape((2,))
    for sample in fake_samples:
      cov = cov.update(sample)
    final_mean, final_cov = self.evaluate([cov.mean, cov.covariance()])
    self.assertAllClose([0., 0.], final_mean, rtol=1e-5)
    self.assertAllCloseNested(np.zeros((2, 2)), final_cov)

  @parameterized.parameters(0, 1)
  def test_running_covariance(self, ddof):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10)
    cov = tfp.experimental.stats.RunningCovariance.from_shape((10,))
    cov = consume(cov, x)
    final_mean, final_cov = self.evaluate([cov.mean, cov.covariance(ddof=ddof)])
    self.assertAllClose(np.mean(x, axis=0), final_mean, rtol=1e-5)
    self.assertAllClose(np.cov(x.T, ddof=ddof), final_cov, rtol=1e-5)
    self.assertEqual(cov.event_ndims, 1)
    self.assertEqual(cov.mean.dtype, tf.float32)

  def test_chunked_high_rank_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 2, 3, 5)
    cov = tfp.experimental.stats.RunningCovariance.from_shape((3, 5))
    cov = consume(cov, x, chunk_axis=0)
    final_mean, final_cov = self.evaluate([cov.mean, cov.covariance()])
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

  def test_running_covariance_with_event_ndims_2(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 3, 5, 2)
    cov = tfp.experimental.stats.RunningCovariance.from_shape(
        tf.TensorShape([3, 5, 2]),
        event_ndims=2)
    cov = consume(cov, x)
    final_mean, final_cov = self.evaluate([cov.mean, cov.covariance()])
    self.assertAllClose(np.mean(x, axis=0), final_mean, rtol=1e-5)
    self.assertEqual(final_cov.shape, (3, 5, 2, 5, 2))
    self.assertEqual(cov.event_ndims, 2)

    # check against numpy
    x_reshaped = x.reshape((100, 3, 10))
    for i in range(x_reshaped.shape[1]):
      np_cov = np.cov(x_reshaped[:, i, :].T, ddof=0).reshape((5, 2, 5, 2))
      self.assertAllClose(np_cov, final_cov[i], rtol=1e-5)

  def test_manual_dtype(self):
    rng = test_util.test_np_rng()
    x = rng.rand(3, 10)
    cov = tfp.experimental.stats.RunningCovariance.from_shape(
        (10,), dtype=tf.float64)
    cov = consume(cov, x)
    final_cov = cov.covariance()
    self.assertEqual(final_cov.dtype, tf.float64)

  def test_shift_in_running_covariance(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10) * 10
    shifted_x = x + 1e4
    cov = tfp.experimental.stats.RunningCovariance.from_shape((10,))
    cov = consume(cov, x)
    shifted_cov = tfp.experimental.stats.RunningCovariance.from_shape((10,))
    shifted_cov = consume(shifted_cov, shifted_x)
    final_cov, final_shifted_cov = self.evaluate([
        cov.covariance(), shifted_cov.covariance()])
    self.assertAllClose(final_cov, np.cov(x.T, ddof=0), rtol=1e-5)
    self.assertAllClose(
        final_shifted_cov, np.cov(x.T, ddof=0), rtol=1e-1)


@test_util.test_all_tf_execution_regimes
class RunningPotentialScaleReductionTest(test_util.TestCase):

  def test_random_scalar_computation(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10) * 100
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction.from_shape(
        shape=(10,))
    running_rhat = consume(running_rhat, x)
    rhat = running_rhat.potential_scale_reduction()
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x,
        independent_chain_ndims=1)
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertNear(true_rhat, rhat, err=1e-6)

  def test_non_scalar_samples(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 2, 2, 3, 5) * 100
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction.from_shape(
        shape=(2, 2, 3, 5))
    running_rhat = consume(running_rhat, x)
    rhat = running_rhat.potential_scale_reduction()
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x,
        independent_chain_ndims=1)
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_multistate(self):
    n_samples = 5
    # state_0 is two scalar chains taken from iid Normal(0, 1).
    state_0 = np.random.randn(n_samples, 2)

    # state_1 is three 4-variate chains taken from Normal(0, 1) that have been
    # shifted.
    offset = np.array([1., -1., 2.]).reshape(3, 1)
    state_1 = np.random.randn(n_samples, 3, 4) + offset
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction.from_shape(
        shape=[(2,), (3, 4)],
        independent_chain_ndims=[1, 1])
    for sample in zip(state_0, state_1):
      running_rhat = running_rhat.update(sample)
    rhat = self.evaluate(running_rhat.potential_scale_reduction())
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=[state_0, state_1], independent_chain_ndims=1)
    self.assertAllCloseNested(true_rhat, rhat, rtol=1e-6)

  def test_independent_chain_ndims(self):
    running_rhat = tfp.experimental.stats.RunningPotentialScaleReduction.from_shape(
        shape=(5, 3),
        independent_chain_ndims=2)
    x = np.arange(30, dtype=np.float32).reshape((2, 5, 3))
    running_rhat = consume(running_rhat, x)
    rhat = running_rhat.potential_scale_reduction()
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=x,
        independent_chain_ndims=2)
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)


@test_util.test_all_tf_execution_regimes
class RunningMeanTest(test_util.TestCase):

  def test_higher_rank_shape(self):
    running_mean = tfp.experimental.stats.RunningMean.from_shape(
        shape=(5, 3))
    for sample in range(6):
      running_mean = running_mean.update(tf.ones((5, 3)) * sample)
    mean = self.evaluate(running_mean.mean)
    self.assertAllEqual(np.ones((5, 3)) * 2.5, mean)

  def test_zero_mean_and_manual_dtype(self):
    running_mean = tfp.experimental.stats.RunningMean.from_shape(
        shape=(),
        dtype=tf.float64)
    for _ in range(6):
      running_mean = running_mean.update(0)
    mean = running_mean.mean
    self.assertEqual(tf.float64, mean.dtype)

  def test_integer_dtype(self):
    running_mean = tfp.experimental.stats.RunningMean.from_shape(
        shape=(),
        dtype=tf.int32)
    for sample in range(6):
      running_mean = running_mean.update(sample)
    mean = running_mean.mean
    self.assertEqual(tf.float32, mean.dtype)
    mean = self.evaluate(mean)
    self.assertEqual(2.5, mean)

  def test_random_mean(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100)
    running_mean = tfp.experimental.stats.RunningMean.from_shape(
        shape=())
    running_mean = consume(running_mean, x)
    mean = self.evaluate(running_mean.mean)
    self.assertAllClose(np.mean(x), mean, rtol=1e-6)

  def test_chunking(self):
    rng = test_util.test_np_rng()
    x = rng.rand(100, 10, 5)
    running_mean = tfp.experimental.stats.RunningMean.from_shape(
        shape=(5,))
    running_mean = consume(running_mean, x, chunk_axis=0)
    mean = self.evaluate(running_mean.mean)
    self.assertAllClose(np.mean(x.reshape(1000, 5), axis=0), mean, rtol=1e-6)

  def test_no_inputs(self):
    running_mean = tfp.experimental.stats.RunningMean.from_shape(
        shape=())
    mean = self.evaluate(running_mean.mean)
    self.assertEqual(0, mean)


@test_util.test_all_tf_execution_regimes
class RunningCentralMomentsTest(test_util.TestCase):

  def test_first_five_moments(self):
    running_moments = tfp.experimental.stats.RunningCentralMoments.from_shape(
        shape=(),
        moment=np.arange(5) + 1)
    for sample in range(5):
      running_moments = running_moments.update(sample)
    zeroth_moment, var, skew, kur, fifth_moment = self.evaluate(
        running_moments.moments())
    self.assertNear(0, zeroth_moment, err=1e-6)
    self.assertNear(2, var, err=1e-6)
    self.assertNear(0, skew, err=1e-6)
    self.assertNear(6.8, kur, err=1e-6)
    self.assertNear(0, fifth_moment, err=1e-6)

  def test_specific_moments(self):
    running_moments = tfp.experimental.stats.RunningCentralMoments.from_shape(
        shape=(),
        moment=[5, 3])
    for sample in range(5):
      running_moments = running_moments.update(sample)
    fifth_moment, skew = self.evaluate(running_moments.moments())
    self.assertNear(0, skew, err=1e-6)
    self.assertNear(0, fifth_moment, err=1e-6)

  def test_very_high_moments(self):
    running_moments = tfp.experimental.stats.RunningCentralMoments.from_shape(
        shape=(),
        moment=np.arange(15) + 1)
    for sample in range(5):
      running_moments = running_moments.update(sample)
    moments = self.evaluate(running_moments.moments())
    self.assertAllClose(
        stats.moment(np.arange(5), moment=np.arange(15) + 1),
        moments,
        atol=5e-5,
        rtol=1e-6)

  def test_higher_rank_samples(self):
    running_moments = tfp.experimental.stats.RunningCentralMoments.from_shape(
        shape=(2, 2),
        moment=np.arange(5) + 1)
    for sample in range(5):
      running_moments = running_moments.update(tf.ones((2, 2)) * sample)
    zeroth_moment, var, skew, kur, fifth_moment = self.evaluate(
        running_moments.moments())
    self.assertAllClose(tf.zeros((2, 2)), zeroth_moment, rtol=1e-6)
    self.assertAllClose(tf.ones((2, 2)) * 2, var, rtol=1e-6)
    self.assertAllClose(tf.zeros((2, 2)), skew, rtol=1e-6)
    self.assertAllClose(tf.ones((2, 2)) * 6.8, kur, rtol=1e-6)
    self.assertAllClose(tf.zeros((2, 2)), fifth_moment, rtol=1e-6)

  def test_random_higher_rank_samples(self):
    rng = test_util.test_np_rng()
    x_orig = rng.rand(100, 10)
    x = tf.convert_to_tensor(x_orig, dtype=tf.float32)
    running_moments = tfp.experimental.stats.RunningCentralMoments.from_shape(
        shape=(10,),
        moment=np.arange(5) + 1)
    running_moments = consume(running_moments, x)
    moments = self.evaluate(running_moments.moments())
    self.assertAllClose(
        stats.moment(x_orig, moment=[1, 2, 3, 4, 5]), moments, rtol=1e-6)

  def test_manual_dtype(self):
    running_moments = tfp.experimental.stats.RunningCentralMoments.from_shape(
        shape=(),
        moment=1,
        dtype=tf.float64)
    running_moments = running_moments.update(0)
    moment = running_moments.moments()
    self.assertEqual(tf.float64, moment.dtype)

  def test_int_dtype_casts(self):
    running_moments = tfp.experimental.stats.RunningCentralMoments.from_shape(
        shape=(),
        moment=1,
        dtype=tf.int32)
    running_moments = running_moments.update(0)
    moment = running_moments.moments()
    self.assertEqual(tf.float32, moment.dtype)


if __name__ == '__main__':
  test_util.main()
