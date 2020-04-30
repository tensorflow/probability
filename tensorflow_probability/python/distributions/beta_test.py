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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports
from absl.testing import parameterized
import numpy as np
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class BetaTest(test_util.TestCase):

  def testSimpleShapes(self):
    a = np.random.rand(3)
    b = np.random.rand(3)
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)

  def testComplexShapes(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(3, 2, 2)
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testComplexShapesBroadcast(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(2, 2)
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testAlphaProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertEqual([1, 3], dist.concentration1.shape)
    self.assertAllClose(a, self.evaluate(dist.concentration1))

  def testBetaProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertEqual([1, 3], dist.concentration0.shape)
    self.assertAllClose(b, self.evaluate(dist.concentration0))

  def testPdfXProper(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Beta(a, b, validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(dist.prob([-1., 0.1, 0.5]))
    with self.assertRaisesOpError('Sample must be less than or equal to `1`.'):
      self.evaluate(dist.prob([.1, .2, 1.2]))

  def testPdfTwoBatches(self):
    a = [1., 2]
    b = [1., 2]
    x = [.5, .5]
    dist = tfd.Beta(a, b, validate_args=True)
    pdf = dist.prob(x)
    self.assertAllClose([1., 3. / 2], self.evaluate(pdf))
    self.assertEqual((2,), pdf.shape)

  def testPdfTwoBatchesNontrivialX(self):
    a = [1., 2]
    b = [1., 2]
    x = [.3, .7]
    dist = tfd.Beta(a, b, validate_args=True)
    pdf = dist.prob(x)
    self.assertAllClose([1, 63. / 50], self.evaluate(pdf))
    self.assertEqual((2,), pdf.shape)

  def testPdfUniformZeroBatch(self):
    # This is equivalent to a uniform distribution
    a = 1.
    b = 1.
    x = np.array([.1, .2, .3, .5, .8], dtype=np.float32)
    dist = tfd.Beta(a, b, validate_args=True)
    pdf = dist.prob(x)
    self.assertAllClose([1.] * 5, self.evaluate(pdf))
    self.assertEqual((5,), pdf.shape)

  def testPdfAlphaStretchedInBroadcastWhenSameRank(self):
    a = [[1., 2]]
    b = [[1., 2]]
    x = [[.5, .5], [.3, .7]]
    dist = tfd.Beta(a, b, validate_args=True)
    pdf = dist.prob(x)
    self.assertAllClose([[1., 3. / 2], [1., 63. / 50]], self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfAlphaStretchedInBroadcastWhenLowerRank(self):
    a = [1., 2]
    b = [1., 2]
    x = [[.5, .5], [.2, .8]]
    pdf = tfd.Beta(a, b, validate_args=True).prob(x)
    self.assertAllClose([[1., 3. / 2], [1., 24. / 25]], self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [[.5, .5]]
    pdf = tfd.Beta(a, b, validate_args=True).prob(x)
    self.assertAllClose([[1., 3. / 2], [3. / 2, 15. / 8]], self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [.5, .5]
    pdf = tfd.Beta(a, b, validate_args=True).prob(x)
    self.assertAllClose([[1., 3. / 2], [3. / 2, 15. / 8]], self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testLogPdfOnBoundaryIsFiniteWhenAlphaIsOne(self):
    b = [[0.01, 0.1, 1., 2], [5., 10., 2., 3]]
    pdf = self.evaluate(tfd.Beta(1., b, validate_args=True).prob(0.))
    self.assertAllEqual(np.ones_like(pdf, dtype=np.bool), np.isfinite(pdf))

  def testBetaMean(self):
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertEqual(dist.mean().shape, (3,))
    expected_mean = sp_stats.beta.mean(a, b)
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

  def testBetaVariance(self):
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertEqual(dist.variance().shape, (3,))
    expected_variance = sp_stats.beta.var(a, b)
    self.assertAllClose(expected_variance, self.evaluate(dist.variance()))

  def testBetaMode(self):
    a = np.array([1.1, 2, 3])
    b = np.array([2., 4, 1.2])
    expected_mode = (a - 1) / (a + b - 2)
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertEqual(dist.mode().shape, (3,))
    self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testBetaModeInvalid(self):
    a = np.array([1., 2, 3])
    b = np.array([2., 4, 1.2])
    dist = tfd.Beta(a, b, allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('Condition x < y.*'):
      self.evaluate(dist.mode())

    a = np.array([2., 2, 3])
    b = np.array([1., 4, 1.2])
    dist = tfd.Beta(a, b, allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('Condition x < y.*'):
      self.evaluate(dist.mode())

  def testBetaModeEnableAllowNanStats(self):
    a = np.array([1., 2, 3])
    b = np.array([2., 4, 1.2])
    dist = tfd.Beta(a, b, allow_nan_stats=True, validate_args=True)

    expected_mode = (a - 1) / (a + b - 2)
    expected_mode[0] = np.nan
    self.assertEqual((3,), dist.mode().shape)
    self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

    a = np.array([2., 2, 3])
    b = np.array([1., 4, 1.2])
    dist = tfd.Beta(a, b, allow_nan_stats=True, validate_args=True)

    expected_mode = (a - 1) / (a + b - 2)
    expected_mode[0] = np.nan
    self.assertEqual((3,), dist.mode().shape)
    self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testBetaEntropy(self):
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = tfd.Beta(a, b, validate_args=True)
    self.assertEqual(dist.entropy().shape, (3,))
    expected_entropy = sp_stats.beta.entropy(a, b)
    self.assertAllClose(expected_entropy, self.evaluate(dist.entropy()))

  def testBetaSample(self):
    a = 1.
    b = 2.
    beta = tfd.Beta(a, b, validate_args=True)
    n = tf.constant(100000)
    samples = beta.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000,))
    self.assertFalse(np.any(sample_values < 0.0))
    self.assertLess(
        sp_stats.kstest(
            # Beta is a univariate distribution.
            sample_values,
            sp_stats.beta(a=1., b=2.).cdf)[0],
        0.01)
    # The standard error of the sample mean is 1 / (sqrt(18 * n))
    self.assertAllClose(
        sample_values.mean(axis=0), sp_stats.beta.mean(a, b), atol=1e-2)
    self.assertAllClose(
        np.cov(sample_values, rowvar=0), sp_stats.beta.var(a, b), atol=1e-1)

  def testBetaFullyReparameterized(self):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    _, [grad_a, grad_b] = tfp.math.value_and_gradient(
        lambda a_, b_: tfd.Beta(a, b, validate_args=True).sample(  # pylint: disable=g-long-lambda
            100, seed=test_util.test_seed()), [a, b])
    self.assertIsNotNone(grad_a)
    self.assertIsNotNone(grad_b)

  # Test that sampling with the same seed twice gives the same results.
  def testBetaSampleMultipleTimes(self):
    a_val = 1.
    b_val = 2.
    n_val = 100
    seed = test_util.test_seed()

    tf.random.set_seed(seed)
    beta1 = tfd.Beta(
        concentration1=a_val,
        concentration0=b_val,
        name='beta1',
        validate_args=True)
    samples1 = self.evaluate(beta1.sample(n_val, seed=seed))

    tf.random.set_seed(seed)
    beta2 = tfd.Beta(
        concentration1=a_val,
        concentration0=b_val,
        name='beta2',
        validate_args=True)
    samples2 = self.evaluate(beta2.sample(n_val, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testBetaSampleMultidimensional(self):
    a = np.random.rand(3, 2, 2).astype(np.float32)
    b = np.random.rand(3, 2, 2).astype(np.float32)
    beta = tfd.Beta(a, b, validate_args=True)
    n = tf.constant(100000)
    samples = beta.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 3, 2, 2))
    self.assertFalse(np.any(sample_values < 0.0))
    self.assertAllClose(
        sample_values[:, 1, :].mean(axis=0),
        sp_stats.beta.mean(a, b)[1, :],
        atol=1e-1)

  @parameterized.parameters((np.float32, 5e-3), (np.float64, 1e-4))
  def testBetaCdf(self, dt, rtol):
    shape = (30, 40, 50)
    a = 10. * np.random.random(shape).astype(dt)
    b = 10. * np.random.random(shape).astype(dt)
    x = np.random.random(shape).astype(dt)
    actual = self.evaluate(tfd.Beta(a, b, validate_args=True).cdf(x))
    self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
    self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
    self.assertAllClose(sp_stats.beta.cdf(x, a, b), actual, rtol=rtol, atol=0)

  @parameterized.parameters((np.float32, 5e-3), (np.float64, 1e-4))
  def testBetaLogCdf(self, dt, rtol):
    shape = (30, 40, 50)
    a = 10. * np.random.random(shape).astype(dt)
    b = 10. * np.random.random(shape).astype(dt)
    x = np.random.random(shape).astype(dt)
    actual = self.evaluate(
        tf.exp(tfd.Beta(a, b, validate_args=True).log_cdf(x)))
    self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
    self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
    self.assertAllClose(sp_stats.beta.cdf(x, a, b), actual, rtol=rtol, atol=0)

  def testBetaBetaKL(self):
    for shape in [(10,), (4, 5)]:
      a1 = 6.0 * np.random.random(size=shape) + 1e-4
      b1 = 6.0 * np.random.random(size=shape) + 1e-4
      a2 = 6.0 * np.random.random(size=shape) + 1e-4
      b2 = 6.0 * np.random.random(size=shape) + 1e-4

      d1 = tfd.Beta(concentration1=a1, concentration0=b1, validate_args=True)
      d2 = tfd.Beta(concentration1=a2, concentration0=b2, validate_args=True)

      kl_expected = (sp_special.betaln(a2, b2) - sp_special.betaln(a1, b1) +
                     (a1 - a2) * sp_special.digamma(a1) +
                     (b1 - b2) * sp_special.digamma(b1) +
                     (a2 - a1 + b2 - b1) * sp_special.digamma(a1 + b1))

      kl = tfd.kl_divergence(d1, d2)
      kl_val = self.evaluate(kl)
      self.assertEqual(kl.shape, shape)
      self.assertAllClose(kl_val, kl_expected)

      # Make sure KL(d1||d1) is 0
      kl_same = self.evaluate(tfd.kl_divergence(d1, d1))
      self.assertAllClose(kl_same, np.zeros_like(kl_expected))

  def testBetaMeanAfterMutation(self):
    concentration1 = tf.Variable(2.)
    concentration0 = tf.Variable(3.)
    self.evaluate(concentration1.initializer)
    self.evaluate(concentration0.initializer)
    dist = tfd.Beta(
        concentration1=concentration1,
        concentration0=concentration0,
        validate_args=True)
    with tf.control_dependencies([concentration0.assign(6.)]):
      mean = self.evaluate(dist.mean())
      self.assertEqual(mean, 0.25)

  @test_util.tf_tape_safety_test
  def testGradientThroughConcentration1(self):
    concentration1 = tf.Variable(3.)
    d = tfd.Beta(
        concentration1=concentration1, concentration0=5., validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0.1, 0.2, 0.85])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveConcentration1(self):
    concentration1 = tf.Variable([1., 2., -3.])
    self.evaluate(concentration1.initializer)
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      d = tfd.Beta(
          concentration1=concentration1, concentration0=[5.],
          validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentration1AfterMutation(self):
    concentration1 = tf.Variable([1., 2., 3.])
    self.evaluate(concentration1.initializer)
    d = tfd.Beta(concentration1=concentration1, concentration0=[5.],
                 validate_args=True)
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      with tf.control_dependencies([concentration1.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testGradientThroughConcentration0(self):
    concentration0 = tf.Variable(3.)
    d = tfd.Beta(
        concentration0=concentration0, concentration1=5., validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0.25, 0.5, 0.9])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveConcentration0(self):
    concentration0 = tf.Variable([1., 2., -3.])
    self.evaluate(concentration0.initializer)
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      d = tfd.Beta(concentration0=concentration0, concentration1=[5.],
                   validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentration0AfterMutation(self):
    concentration0 = tf.Variable([1., 2., 3.])
    self.evaluate(concentration0.initializer)
    d = tfd.Beta(concentration0=concentration0, concentration1=[5.],
                 validate_args=True)
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      with tf.control_dependencies([concentration0.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testLogProbInfinityAtBoundary(self):
    d = tfd.Beta(concentration0=[5., 0.5], concentration1=[5., 0.5],
                 validate_args=True)
    log_prob = self.evaluate(d.log_prob([[0.], [1.]]))
    self.assertAllNegativeInf(log_prob[:, 0])
    self.assertAllPositiveInf(log_prob[:, 1])

  def testSupportBijectorOutsideRange(self):
    a = np.array([1., 2., 3.])
    b = np.array([2., 4., 1.2])
    dist = tfd.Beta(a, b, validate_args=True)
    eps = 1e-6
    x = np.array([-2.3, -eps, 1. + eps, 1.4])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

  def testGradientOfLogProbEvalutates(self):
    def f(a):
      return tfd.Beta(a, 10).log_prob(.5)
    self.evaluate(tfp.math.value_and_gradient(f, [100.0]))


if __name__ == '__main__':
  tf.test.main()
