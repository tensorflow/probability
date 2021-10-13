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
# Dependency imports
import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class _SkellamTest(object):

  def _make_skellam(self,
                    rate1,
                    rate2,
                    validate_args=True,
                    force_probs_to_zero_outside_support=False):
    return tfd.Skellam(
        rate1=rate1,
        rate2=rate2,
        validate_args=validate_args,
        force_probs_to_zero_outside_support=force_probs_to_zero_outside_support)

  def testSkellamShape(self):
    rate1 = tf.constant([3.0] * 5, dtype=self.dtype)
    rate2 = tf.constant([3.0] * 4, dtype=self.dtype)[..., tf.newaxis]
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)

    self.assertAllEqual(self.evaluate(skellam.batch_shape_tensor()), (4, 5))
    self.assertEqual(skellam.batch_shape, tf.TensorShape([4, 5]))
    self.assertAllEqual(self.evaluate(skellam.event_shape_tensor()), [])
    self.assertEqual(skellam.event_shape, tf.TensorShape([]))

  def testInvalidLam(self):
    invalid_rate = self.dtype([-.01, 1., 2.])
    valid_rate = self.dtype([1., 2., 3.])
    with self.assertRaisesOpError('Argument `rate1` must be non-negative.'):
      skellam = self._make_skellam(rate1=invalid_rate, rate2=valid_rate)
      self.evaluate(skellam.rate1_parameter())

    with self.assertRaisesOpError('Argument `rate2` must be non-negative.'):
      skellam = self._make_skellam(rate1=valid_rate, rate2=invalid_rate)
      self.evaluate(skellam.rate2_parameter())

  def testZeroRate(self):
    lam = self.dtype(0.)
    skellam = tfd.Skellam(rate1=lam, rate2=lam, validate_args=True)
    self.assertAllClose(lam, self.evaluate(skellam.rate1))
    self.assertAllClose(lam, self.evaluate(skellam.rate2))
    self.assertAllClose(0., skellam.prob(3.))
    self.assertAllClose(1., skellam.prob(0.))
    self.assertAllClose(0., skellam.log_prob(0.))

  def testSkellamLogPmfDiscreteMatchesScipy(self):
    batch_size = 12
    rate1 = np.linspace(1, 12, 12).astype(self.dtype)
    rate2 = np.array([[1.2], [2.3]]).astype(self.dtype)
    x = np.array([-3., -1., 0., 2., 4., 3., 7., 4., 8., 9., 6., 7.],
                 dtype=self.dtype)
    skellam = self._make_skellam(
        rate1=rate1, rate2=rate2,
        force_probs_to_zero_outside_support=True, validate_args=False)
    log_pmf = skellam.log_prob(x)
    self.assertEqual(log_pmf.shape, (2, batch_size))
    self.assertAllClose(
        self.evaluate(log_pmf),
        stats.skellam.logpmf(x, rate1, rate2))

    pmf = skellam.prob(x)
    self.assertEqual(pmf.shape, (2, batch_size,))
    self.assertAllClose(
        self.evaluate(pmf),
        stats.skellam.pmf(x, rate1, rate2))

  @test_util.numpy_disable_gradient_test
  def testSkellamLogPmfGradient(self):
    batch_size = 6
    rate1 = tf.constant([3.] * batch_size, dtype=self.dtype)
    rate2 = tf.constant([2.7] * batch_size, dtype=self.dtype)
    x = np.array([-1., 2., 3., 4., 5., 6.], dtype=self.dtype)

    err = self.compute_max_gradient_error(
        lambda lam: self._make_skellam(  # pylint:disable=g-long-lambda
            rate1=lam, rate2=rate2).log_prob(x), [rate1])
    self.assertLess(err, 7e-4)

    err = self.compute_max_gradient_error(
        lambda lam: self._make_skellam(  # pylint:disable=g-long-lambda
            rate1=rate1, rate2=lam).log_prob(x), [rate2])
    self.assertLess(err, 7e-4)

  @test_util.numpy_disable_gradient_test
  def testSkellamLogPmfGradientAtZeroPmf(self):
    # Check that the derivative wrt parameter at the zero-prob points is zero.
    batch_size = 6
    rate1 = tf.constant(np.linspace(1, 7, 6), dtype=self.dtype)
    rate2 = tf.constant(np.linspace(9.1, 12.1, 6), dtype=self.dtype)
    x = tf.constant([-2.1, -1.3, -0.5, 0.2, 1.5, 10.5], dtype=self.dtype)

    def make_skellam_log_prob(apply_to_second_rate=False):
      def skellam_log_prob(lam):
        return self._make_skellam(
            rate1=rate1 if apply_to_second_rate else lam,
            rate2=lam if apply_to_second_rate else rate2,
            force_probs_to_zero_outside_support=True,
            validate_args=False).log_prob(x)
      return skellam_log_prob
    _, dlog_pmf_dlam = self.evaluate(tfp.math.value_and_gradient(
        make_skellam_log_prob(), rate1))

    self.assertEqual(dlog_pmf_dlam.shape, (batch_size,))
    self.assertAllClose(dlog_pmf_dlam, np.zeros([batch_size]))

    _, dlog_pmf_dlam = self.evaluate(tfp.math.value_and_gradient(
        make_skellam_log_prob(True), rate2))

    self.assertEqual(dlog_pmf_dlam.shape, (batch_size,))
    self.assertAllClose(dlog_pmf_dlam, np.zeros([batch_size]))

  def testSkellamMean(self):
    rate1 = np.array([1.0, 3.0, 2.5], dtype=self.dtype)
    rate2 = np.array([5.0, 7.13, 2.56, 41.], dtype=self.dtype)[..., np.newaxis]
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)
    self.assertEqual(skellam.mean().shape, (4, 3))
    self.assertAllClose(
        self.evaluate(skellam.mean()), stats.skellam.mean(rate1, rate2))
    self.assertAllClose(self.evaluate(skellam.mean()), rate1 - rate2)

  def testSkellamVariance(self):
    rate1 = np.array([1.0, 3.0, 2.5], dtype=self.dtype)
    rate2 = np.array([5.0, 7.13, 2.56, 41.], dtype=self.dtype)[..., np.newaxis]
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)
    self.assertEqual(skellam.variance().shape, (4, 3))
    self.assertAllClose(
        self.evaluate(skellam.variance()), stats.skellam.var(rate1, rate2))
    self.assertAllClose(self.evaluate(skellam.variance()), rate1 + rate2)

  def testSkellamStd(self):
    rate1 = np.array([1.0, 3.0, 2.5], dtype=self.dtype)
    rate2 = np.array([5.0, 7.13, 2.56, 41.], dtype=self.dtype)[..., np.newaxis]
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)
    self.assertEqual(skellam.stddev().shape, (4, 3))
    self.assertAllClose(
        self.evaluate(skellam.stddev()), stats.skellam.std(rate1, rate2))
    self.assertAllClose(self.evaluate(skellam.stddev()), np.sqrt(rate1 + rate2))

  def testSkellamSample(self):
    rate1 = self.dtype([2., 3., 4.])
    rate2 = self.dtype([7.1, 3.2])[..., np.newaxis]
    n = int(2e5)
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)
    samples = skellam.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 2, 3))
    self.assertEqual(sample_values.shape, (n, 2, 3))
    self.assertAllClose(
        sample_values.mean(axis=0), stats.skellam.mean(rate1, rate2), rtol=.03)
    self.assertAllClose(
        sample_values.var(axis=0), stats.skellam.var(rate1, rate2), rtol=.03)

  def testAssertValidSample(self):
    rate1 = np.array([1.0, 3.0, 2.5], dtype=self.dtype)
    rate2 = np.array([2.1, 7.0, 42.5], dtype=self.dtype)
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)
    with self.assertRaisesOpError('has non-integer components'):
      self.evaluate(skellam.prob([-1.2, 3., 4.2]))

  def testSkellamSampleMultidimensionalMean(self):
    rate1 = self.dtype([2., 3., 4., 5., 6.])
    rate2 = self.dtype([7.1, 3.2, 10., 9.])[..., np.newaxis]
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)
    n = int(2e5)
    samples = skellam.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 4, 5))
    self.assertEqual(sample_values.shape, (n, 4, 5))
    self.assertAllClose(
        sample_values.mean(axis=0),
        stats.skellam.mean(rate1, rate2), rtol=.04, atol=0)

  def testSkellamSampleMultidimensionalVariance(self):
    rate1 = self.dtype([2., 3., 4., 5., 6.])
    rate2 = self.dtype([7.1, 3.2, 10., 9.])[..., np.newaxis]
    skellam = self._make_skellam(rate1=rate1, rate2=rate2)
    n = int(1e5)
    samples = skellam.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 4, 5))
    self.assertEqual(sample_values.shape, (n, 4, 5))

    self.assertAllClose(
        sample_values.var(axis=0),
        stats.skellam.var(rate1, rate2), rtol=.03, atol=0)

  @test_util.tf_tape_safety_test
  def testGradientThroughRate(self):
    rate1 = tf.Variable(3.)
    rate2 = tf.Variable(4.)
    dist = self._make_skellam(rate1=rate1, rate2=rate2)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)

  def testAssertsNonNegativeRate(self):
    rate1 = tf.Variable([-1., 2., -3.])
    rate2 = tf.Variable([1., 2., 3.])
    self.evaluate([rate1.initializer, rate2.initializer])
    with self.assertRaisesOpError('Argument `rate1` must be non-negative.'):
      dist = self._make_skellam(
          rate1=rate1, rate2=rate2, validate_args=True)
      self.evaluate(dist.sample(seed=test_util.test_seed()))

    rate1 = tf.Variable([1., 2., 3.])
    rate2 = tf.Variable([-1., 2., -3.])
    self.evaluate([rate1.initializer, rate2.initializer])

    with self.assertRaisesOpError('Argument `rate2` must be non-negative.'):
      dist = self._make_skellam(
          rate1=rate1, rate2=rate2, validate_args=True)
      self.evaluate(dist.sample(seed=test_util.test_seed()))

  def testAssertsNonNegativeRateAfterMutation(self):
    rate1 = tf.Variable([1., 2., 3.])
    rate2 = tf.Variable([1., 2., 3.])
    self.evaluate([rate1.initializer, rate2.initializer])
    dist = self._make_skellam(
        rate1=rate1, rate2=rate2, validate_args=True)
    self.evaluate(dist.mean())
    with self.assertRaisesOpError('Argument `rate1` must be non-negative.'):
      with tf.control_dependencies([rate1.assign([1., 2., -3.])]):
        self.evaluate(dist.sample(seed=test_util.test_seed()))

    rate1 = tf.Variable([1., 2., 3.])
    rate2 = tf.Variable([1., 2., 3.])
    self.evaluate([rate1.initializer, rate2.initializer])
    dist = self._make_skellam(
        rate1=rate1, rate2=rate2, validate_args=True)
    self.evaluate(dist.mean())

    with self.assertRaisesOpError('Argument `rate2` must be non-negative.'):
      with tf.control_dependencies([rate2.assign([1., 2., -3.])]):
        self.evaluate(dist.sample(seed=test_util.test_seed()))


@test_util.test_all_tf_execution_regimes
class SkellamTestFloat32(test_util.TestCase, _SkellamTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class SkellamTestFloat64(test_util.TestCase, _SkellamTest):
  dtype = np.float64


@test_util.test_all_tf_execution_regimes
class SkellamLogRateTest(_SkellamTest):

  def _make_skellam(self,
                    rate1,
                    rate2,
                    validate_args=True,
                    force_probs_to_zero_outside_support=False):
    return tfd.Skellam(
        log_rate1=tf.math.log(rate1),
        log_rate2=tf.math.log(rate2),
        validate_args=validate_args,
        force_probs_to_zero_outside_support=force_probs_to_zero_outside_support)

  # No need to worry about the non-negativity of `rate` when using the
  # `log_rate` parameterization.
  def testInvalidLam(self):
    pass

  def testAssertsNonNegativeRate(self):
    pass

  def testAssertsNonNegativeRateAfterMutation(self):
    pass

  # The gradient is not tracked through tf.math.log(rate) in _make_skellam(),
  # so log_rate needs to be defined as a Variable and passed directly.
  @test_util.tf_tape_safety_test
  def testGradientThroughRate(self):
    log_rate1 = tf.Variable(3.)
    log_rate2 = tf.Variable(4.)
    dist = tfd.Skellam(
        log_rate1=log_rate1, log_rate2=log_rate2, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)


if __name__ == '__main__':
  test_util.main()
