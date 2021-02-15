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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class SigmoidBetaTest(test_util.TestCase):

  def testSimpleShapes(self):
    a = np.random.rand(3)
    b = np.random.rand(3)
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)

  def testComplexShapes(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(3, 2, 2)
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testComplexShapesBroadcast(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(2, 2)
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testAlphaProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    self.assertEqual([1, 3], dist.concentration1.shape)
    self.assertAllClose(a, self.evaluate(dist.concentration1))

  def testBetaProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    self.assertEqual([1, 3], dist.concentration0.shape)
    self.assertAllClose(b, self.evaluate(dist.concentration0))

  def testPDF(self):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    dist = tfd.SigmoidBeta(a, b, validate_args=True)

    oracle = tfd.TransformedDistribution(distribution=tfd.Beta(a, b),
                                         bijector=tfb.Invert(tfb.Sigmoid()))

    x = np.array([0.0, 0.5, 0.75, 1.0])
    log_pdf = dist.log_prob(x)
    pdf = dist.prob(x)
    expected_log_pdf = oracle.log_prob(x)
    expected_pdf = oracle.prob(x)
    self.assertAllClose(log_pdf, expected_log_pdf)
    self.assertAllClose(pdf, expected_pdf)

  def testPdfTwoBatches(self):
    a = [1., 2]
    b = [1., 2]
    x = [0., 0.]
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    pdf = dist.prob(x)
    self.assertAllClose([1. / 4., 3. / 8.], self.evaluate(pdf))
    self.assertEqual((2,), pdf.shape)

  def testLogPDF(self):
    a = [1., 2]
    b = [1., 2]
    x = [0., 0.]
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    logpdf = dist.log_prob(x)

    oracle = tfd.TransformedDistribution(distribution=tfd.Beta(a, b),
                                         bijector=tfb.Invert(tfb.Sigmoid()))
    expected_logpdf = oracle.log_prob(x)

    self.assertAllClose(expected_logpdf, logpdf)
    self.assertEqual((2,), logpdf.shape)

    x = [1., 1.]
    logpdf = dist.log_prob(x)
    expected_logpdf = oracle.log_prob(x)

    self.assertAllClose(expected_logpdf, logpdf)
    self.assertEqual((2,), logpdf.shape)

    a = [1., 2.]
    b = [2., 1.]
    x = [1., 1.]
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    oracle = tfd.TransformedDistribution(distribution=tfd.Beta(a, b),
                                         bijector=tfb.Invert(tfb.Sigmoid()))

    logpdf = dist.log_prob(x)
    expected_logpdf = oracle.log_prob(x)
    self.assertAllClose(expected_logpdf, logpdf)
    self.assertEqual((2,), logpdf.shape)

  def testSampleWithPartiallyDefinedShapeEndingInOne(self):
    b = [[0.01, 0.1, 1., 2], [5., 10., 2., 3]]
    pdf = self.evaluate(tfd.SigmoidBeta(1., b, validate_args=True).prob(0.))
    self.assertAllEqual(np.ones_like(pdf, dtype=np.bool), np.isfinite(pdf))

  def testIsFiniteLargeX(self):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    x = [1e10]
    expected = [-2*1e10]
    log_pdf = dist.log_prob(x)
    self.assertAllClose(log_pdf, expected)

  def testLogPDFMultidimensional(self):
    batch_size = 6
    a = tf.constant(1.0)
    b = tf.constant([[1.0, 2.0]] * batch_size)
    x = np.array([[0.0, 0.1, 0.2, 0.4, 0.5, 1.0]], dtype=np.float32).T
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    oracle = tfd.TransformedDistribution(distribution=tfd.Beta(a, b),
                                         bijector=tfb.Invert(tfb.Sigmoid()))
    log_pdf = dist.log_prob(x)
    expected_logpdf = oracle.log_prob(x)
    self.assertEqual(log_pdf.shape, (6, 2))
    self.assertAllClose(log_pdf, expected_logpdf)
    pdf = dist.prob(x)
    expected_pdf = oracle.prob(x)
    self.assertEqual(pdf.shape, (6, 2))
    self.assertAllClose(pdf, expected_pdf)

  def testCDF(self):
    batch_size = 6
    a = tf.constant([2.0] * batch_size)
    b = tf.constant([3.0] * batch_size)
    x = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 1.0], dtype=np.float32)
    dist = tfd.SigmoidBeta(a, b, validate_args=True)
    cdf = dist.cdf(x)
    self.assertEqual(cdf.shape, (6,))

    expected_cdf = tfd.TransformedDistribution(distribution=tfd.Beta(a, b),
                                               bijector=tfb.Invert(
                                                   tfb.Sigmoid())).cdf(x)
    self.assertAllClose(cdf, expected_cdf)

  def testMode(self):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    dist = tfd.SigmoidBeta(a, b, validate_args=True)

    self.assertAllClose(self.evaluate(tf.math.log(a / b)),
                        self.evaluate(dist.mode()))

  def testAssertsPositiveConcentration1(self):
    concentration1 = tf.Variable([1., 2., -3.])
    self.evaluate(concentration1.initializer)
    with self.assertRaisesOpError('Concentration1 parameter must be positive.'):
      d = tfd.SigmoidBeta(concentration1=concentration1,
                          concentration0=[5.],
                          validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentration1AfterMutation(self):
    concentration1 = tf.Variable([1., 2., 3.])
    self.evaluate(concentration1.initializer)
    d = tfd.SigmoidBeta(concentration1=concentration1,
                        concentration0=[5.],
                        validate_args=True)
    with self.assertRaisesOpError('Concentration1 parameter must be positive.'):
      with tf.control_dependencies([concentration1.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testGradientThroughConcentration0(self):
    concentration0 = tf.Variable(3.)
    d = tfd.SigmoidBeta(concentration0=concentration0,
                        concentration1=5.,
                        validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0.25, 0.5, 0.9])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveConcentration0(self):
    concentration0 = tf.Variable([1., 2., -3.])
    self.evaluate(concentration0.initializer)
    with self.assertRaisesOpError('Concentration0 parameter must be positive.'):
      d = tfd.SigmoidBeta(concentration0=concentration0,
                          concentration1=[5.],
                          validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentration0AfterMutation(self):
    concentration0 = tf.Variable([1., 2., 3.])
    self.evaluate(concentration0.initializer)
    d = tfd.SigmoidBeta(concentration0=concentration0,
                        concentration1=[5.],
                        validate_args=True)
    with self.assertRaisesOpError('Concentration0 parameter must be positive.'):
      with tf.control_dependencies([concentration0.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def _kstest(self, a, b, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    oracle_dist = tfd.SigmoidBeta(concentration0=a, concentration1=b)
    ks, _ = sp_stats.kstest(samples,
                            lambda x: self.evaluate(oracle_dist.cdf(x)))
    # Return True when the test passes.
    return ks < 0.02

  def testSample(self):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    n = 500000
    d = tfd.SigmoidBeta(concentration0=a, concentration1=b, validate_args=True)
    samples = d.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertTrue(self._kstest(a, b, sample_values))

    check_cdf_agrees = st.assert_true_cdf_equal_by_dkwm(
        samples, d.cdf, false_fail_rate=1e-6)
    self.evaluate(check_cdf_agrees)
    check_enough_power = assert_util.assert_less(
        st.min_discrepancy_of_true_cdfs_detectable_by_dkwm(
            n, false_fail_rate=1e-6, false_pass_rate=1e-6), 0.01)
    self.evaluate(check_enough_power)

if __name__ == '__main__':
  tf.test.main()
