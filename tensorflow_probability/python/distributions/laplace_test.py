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
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class LaplaceTest(test_util.TestCase):

  def testLaplaceShape(self):
    loc = tf.constant([3.0] * 5)
    scale = tf.constant(11.0)
    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)

    self.assertEqual(self.evaluate(laplace.batch_shape_tensor()), (5,))
    self.assertEqual(laplace.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(laplace.event_shape_tensor()), [])
    self.assertEqual(laplace.event_shape, tf.TensorShape([]))

  def testLaplaceLogPDF(self):
    batch_size = 6
    loc = tf.constant([2.0] * batch_size)
    scale = tf.constant([3.0] * batch_size)
    loc_v = 2.0
    scale_v = 3.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)
    log_pdf = laplace.log_prob(x)
    self.assertEqual(log_pdf.shape, (6,))
    expected_log_pdf = sp_stats.laplace.logpdf(x, loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

    pdf = laplace.prob(x)
    self.assertEqual(pdf.shape, (6,))
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testLaplaceLogPDFMultidimensional(self):
    batch_size = 6
    loc = tf.constant([[2.0, 4.0]] * batch_size)
    scale = tf.constant([[3.0, 4.0]] * batch_size)
    loc_v = np.array([2.0, 4.0])
    scale_v = np.array([3.0, 4.0])
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)
    log_pdf = laplace.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))

    pdf = laplace.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    expected_log_pdf = sp_stats.laplace.logpdf(x, loc_v, scale=scale_v)
    self.assertAllClose(log_pdf_values, expected_log_pdf)
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testLaplaceLogPDFMultidimensionalBroadcasting(self):
    batch_size = 6
    loc = tf.constant([[2.0, 4.0]] * batch_size)
    scale = tf.constant(3.0)
    loc_v = np.array([2.0, 4.0])
    scale_v = 3.0
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)
    log_pdf = laplace.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))

    pdf = laplace.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    expected_log_pdf = sp_stats.laplace.logpdf(x, loc_v, scale=scale_v)
    self.assertAllClose(log_pdf_values, expected_log_pdf)
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testLaplaceCDF(self):
    batch_size = 6
    loc = tf.constant([2.0] * batch_size)
    scale = tf.constant([3.0] * batch_size)
    loc_v = 2.0
    scale_v = 3.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)

    cdf = laplace.cdf(x)
    self.assertEqual(cdf.shape, (6,))
    expected_cdf = sp_stats.laplace.cdf(x, loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testLaplaceLogCDF(self):
    batch_size = 6
    loc = tf.constant([2.0] * batch_size)
    scale = tf.constant([3.0] * batch_size)
    loc_v = 2.0
    scale_v = 3.0
    x = np.array([-2.5, 2.5, -4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)

    cdf = laplace.log_cdf(x)
    self.assertEqual(cdf.shape, (6,))
    expected_cdf = sp_stats.laplace.logcdf(x, loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testLaplaceQuantile(self):
    qs = self.evaluate(
        tf.concat(
            [[0., 1],
             samplers.uniform([10], minval=.1, maxval=.9,
                              seed=test_util.test_seed())],
            axis=0))
    d = tfd.Laplace(loc=1., scale=1.3, validate_args=True)
    vals = d.quantile(qs)
    self.assertAllClose([-np.inf, np.inf], vals[:2])
    self.assertAllClose(qs[2:], d.cdf(vals[2:]))

  def testLaplaceLogSurvivalFunction(self):
    batch_size = 6
    loc = tf.constant([2.0] * batch_size)
    scale = tf.constant([3.0] * batch_size)
    loc_v = 2.0
    scale_v = 3.0
    x = np.array([-2.5, 2.5, -4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)

    sf = laplace.log_survival_function(x)
    self.assertEqual(sf.shape, (6,))
    expected_sf = sp_stats.laplace.logsf(x, loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(sf), expected_sf)

  def testLaplaceMean(self):
    loc_v = np.array([1.0, 3.0, 2.5])
    scale_v = np.array([1.0, 4.0, 5.0])
    laplace = tfd.Laplace(loc=loc_v, scale=scale_v, validate_args=True)
    self.assertEqual(laplace.mean().shape, (3,))
    expected_means = sp_stats.laplace.mean(loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(laplace.mean()), expected_means)

  def testLaplaceMode(self):
    loc_v = np.array([0.5, 3.0, 2.5])
    scale_v = np.array([1.0, 4.0, 5.0])
    laplace = tfd.Laplace(loc=loc_v, scale=scale_v, validate_args=True)
    self.assertEqual(laplace.mode().shape, (3,))
    self.assertAllClose(self.evaluate(laplace.mode()), loc_v)

  def testLaplaceVariance(self):
    loc_v = np.array([1.0, 3.0, 2.5])
    scale_v = np.array([1.0, 4.0, 5.0])
    laplace = tfd.Laplace(loc=loc_v, scale=scale_v, validate_args=True)
    self.assertEqual(laplace.variance().shape, (3,))
    expected_variances = sp_stats.laplace.var(loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(laplace.variance()), expected_variances)

  def testLaplaceStd(self):
    loc_v = np.array([1.0, 3.0, 2.5])
    scale_v = np.array([1.0, 4.0, 5.0])
    laplace = tfd.Laplace(loc=loc_v, scale=scale_v, validate_args=True)
    self.assertEqual(laplace.stddev().shape, (3,))
    expected_stddev = sp_stats.laplace.std(loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(laplace.stddev()), expected_stddev)

  def testLaplaceEntropy(self):
    loc_v = np.array([1.0, 3.0, 2.5])
    scale_v = np.array([1.0, 4.0, 5.0])
    laplace = tfd.Laplace(loc=loc_v, scale=scale_v, validate_args=True)
    self.assertEqual(laplace.entropy().shape, (3,))
    expected_entropy = sp_stats.laplace.entropy(loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(laplace.entropy()), expected_entropy)

  def testLaplaceSample(self):
    loc_v = 4.0
    scale_v = 3.0
    loc = tf.constant(loc_v)
    scale = tf.constant(scale_v)
    n = 100000
    laplace = tfd.Laplace(loc=loc, scale=scale, validate_args=True)
    samples = laplace.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(),
        sp_stats.laplace.mean(loc_v, scale=scale_v),
        rtol=0.05,
        atol=0.)
    self.assertAllClose(
        sample_values.var(),
        sp_stats.laplace.var(loc_v, scale=scale_v),
        rtol=0.05,
        atol=0.)
    self.assertTrue(self._kstest(loc_v, scale_v, sample_values))

  def testLaplaceFullyReparameterized(self):
    loc = tf.constant(4.0)
    scale = tf.constant(3.0)
    _, [grad_loc, grad_scale] = tfp.math.value_and_gradient(
        lambda l, s: tfd.Laplace(loc=l, scale=s, validate_args=True).sample(  # pylint: disable=g-long-lambda
            100, seed=test_util.test_seed()), [loc, scale])
    self.assertIsNotNone(grad_loc)
    self.assertIsNotNone(grad_scale)

  def testLaplaceSampleMultiDimensional(self):
    loc_v = np.array([np.arange(1, 101, dtype=np.float32)])  # 1 x 100
    scale_v = np.array([np.arange(1, 11, dtype=np.float32)]).T  # 10 x 1
    laplace = tfd.Laplace(loc=loc_v, scale=scale_v, validate_args=True)
    n = 10000
    samples = laplace.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 10, 100))
    self.assertEqual(sample_values.shape, (n, 10, 100))
    zeros = np.zeros_like(loc_v + scale_v)  # 10 x 100
    loc_bc = loc_v + zeros
    scale_bc = scale_v + zeros
    self.assertAllClose(
        sample_values.mean(axis=0),
        sp_stats.laplace.mean(loc_bc, scale=scale_bc),
        rtol=0.35,
        atol=0.)
    self.assertAllClose(
        sample_values.var(axis=0),
        sp_stats.laplace.var(loc_bc, scale=scale_bc),
        rtol=0.10,
        atol=0.)
    fails = 0
    trials = 0
    for ai, a in enumerate(np.reshape(loc_v, [-1])):
      for bi, b in enumerate(np.reshape(scale_v, [-1])):
        s = sample_values[:, bi, ai]
        trials += 1
        fails += 0 if self._kstest(a, b, s) else 1
    self.assertLess(fails, trials * 0.03)

  def _kstest(self, loc, scale, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    ks, _ = sp_stats.kstest(samples, sp_stats.laplace(loc, scale=scale).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testLaplacePdfOfSampleMultiDims(self):
    laplace = tfd.Laplace(loc=[7., 11.], scale=[[5.], [6.]], validate_args=True)
    num = 50000
    samples = laplace.sample(num, seed=test_util.test_seed())
    pdfs = laplace.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.shape, (num, 2, 2))
    self.assertEqual(pdfs.shape, (num, 2, 2))
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)
    self.assertAllClose(
        sp_stats.laplace.mean(
            [[7., 11.], [7., 11.]], scale=np.array([[5., 5.], [6., 6.]])),
        sample_vals.mean(axis=0),
        rtol=0.05,
        atol=0.)
    self.assertAllClose(
        sp_stats.laplace.var([[7., 11.], [7., 11.]],
                             scale=np.array([[5., 5.], [6., 6.]])),
        sample_vals.var(axis=0),
        rtol=0.05,
        atol=0.)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (0, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testLaplaceNonPositiveInitializationParamsRaises(self):
    loc_v = tf.constant(0.0, name='loc')
    scale_v = tf.constant(-1.0, name='scale')
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      laplace = tfd.Laplace(
          loc=loc_v, scale=scale_v, validate_args=True)
      self.evaluate(laplace.mean())

    loc_v = tf.constant(1.0, name='loc')
    scale_v = tf.constant(0.0, name='scale')
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      laplace = tfd.Laplace(
          loc=loc_v, scale=scale_v, validate_args=True)
      self.evaluate(laplace.mean())

    scale = tf.Variable([1., 2., -3.])
    self.evaluate(scale.initializer)
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      d = tfd.Laplace(loc=0, scale=scale, validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testLaplaceLaplaceKL(self):
    batch_size = 6
    event_size = 3

    a_loc = np.array([[0.5] * event_size] * batch_size, dtype=np.float32)
    a_scale = np.array([[0.1] * event_size] * batch_size, dtype=np.float32)
    b_loc = np.array([[0.4] * event_size] * batch_size, dtype=np.float32)
    b_scale = np.array([[0.2] * event_size] * batch_size, dtype=np.float32)

    a = tfd.Laplace(loc=a_loc, scale=a_scale, validate_args=True)
    b = tfd.Laplace(loc=b_loc, scale=b_scale, validate_args=True)

    distance = tf.abs(a_loc - b_loc)
    ratio = a_scale / b_scale
    true_kl = (-tf.math.log(ratio) - 1 + distance / b_scale +
               ratio * tf.exp(-distance / a_scale))

    kl = tfd.kl_divergence(a, b)

    x = a.sample(int(1e4), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)

    true_kl_, kl_, kl_sample_ = self.evaluate([true_kl, kl, kl_sample])
    self.assertAllClose(true_kl_, kl_, atol=1e-5, rtol=1e-5)
    self.assertAllClose(true_kl_, kl_sample_, atol=0., rtol=1e-1)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(true_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    loc = tf.Variable([-5., 0., 5.])
    scale = tf.Variable(2.)
    d = tfd.Laplace(loc=loc, scale=scale, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 3.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)

  def testAssertsPositiveScaleAfterMutation(self):
    scale = tf.Variable([1., 2., 3.])
    d = tfd.Laplace(loc=0., scale=scale, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([scale.assign([1., 2., -3.])]):
        self.evaluate(tfd.Laplace(loc=0., scale=1.).kl_divergence(d))

  def testAssertParamsAreFloats(self):
    loc = tf.convert_to_tensor(0, dtype=tf.int32)
    scale = tf.convert_to_tensor(1, dtype=tf.int32)
    with self.assertRaisesRegexp(ValueError, 'Expected floating point'):
      tfd.Laplace(loc=loc, scale=scale)


if __name__ == '__main__':
  tf.test.main()
