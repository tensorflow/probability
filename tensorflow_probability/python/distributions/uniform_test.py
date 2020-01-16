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
"""Tests for Uniform distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy import stats as sp_stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class UniformTest(test_util.TestCase):

  def testUniformRange(self):
    a = 3.0
    b = 10.0
    uniform = tfd.Uniform(low=a, high=b, validate_args=True)
    self.assertAllClose(a, self.evaluate(uniform.low))
    self.assertAllClose(b, self.evaluate(uniform.high))
    self.assertAllClose(b - a, self.evaluate(uniform.range()))

  def testUniformPDF(self):
    a = tf.constant([-3.0] * 5 + [15.0])
    b = tf.constant([11.0] * 5 + [20.0])
    uniform = tfd.Uniform(low=a, high=b, validate_args=False)

    a_v = -3.0
    b_v = 11.0
    x = np.array([-10.5, 4.0, 0.0, 10.99, 11.3, 17.0], dtype=np.float32)

    def _expected_pdf():
      pdf = np.zeros_like(x) + 1.0 / (b_v - a_v)
      pdf[x > b_v] = 0.0
      pdf[x < a_v] = 0.0
      pdf[5] = 1.0 / (20.0 - 15.0)
      return pdf

    expected_pdf = _expected_pdf()

    pdf = uniform.prob(x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

    log_pdf = uniform.log_prob(x)
    self.assertAllClose(np.log(expected_pdf), self.evaluate(log_pdf))

  def testUniformShape(self):
    a = tf.constant([-3.0] * 5)
    b = tf.constant(11.0)
    uniform = tfd.Uniform(low=a, high=b, validate_args=True)

    self.assertEqual(self.evaluate(uniform.batch_shape_tensor()), (5,))
    self.assertEqual(uniform.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(uniform.event_shape_tensor()), [])
    self.assertEqual(uniform.event_shape, tf.TensorShape([]))

  def testUniformPDFWithScalarEndpoint(self):
    a = tf.constant([0.0, 5.0])
    b = tf.constant(10.0)
    uniform = tfd.Uniform(low=a, high=b, validate_args=True)

    x = np.array([0.0, 8.0], dtype=np.float32)
    expected_pdf = np.array([1.0 / (10.0 - 0.0), 1.0 / (10.0 - 5.0)])

    pdf = uniform.prob(x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testUniformCDF(self):
    batch_size = 6
    a = tf.constant([1.0] * batch_size)
    b = tf.constant([11.0] * batch_size)
    a_v = 1.0
    b_v = 11.0
    x = np.array([-2.5, 2.5, 4.0, 0.0, 10.99, 12.0], dtype=np.float32)

    uniform = tfd.Uniform(low=a, high=b, validate_args=False)

    def _expected_cdf():
      cdf = (x - a_v) / (b_v - a_v)
      cdf[x >= b_v] = 1
      cdf[x < a_v] = 0
      return cdf

    cdf = uniform.cdf(x)
    self.assertAllClose(_expected_cdf(), self.evaluate(cdf))

    log_cdf = uniform.log_cdf(x)
    self.assertAllClose(np.log(_expected_cdf()), self.evaluate(log_cdf))

  def testUniformQuantile(self):
    low = tf.reshape(tf.linspace(0., 1., 6), [2, 1, 3])
    high = tf.reshape(tf.linspace(1.5, 2.5, 6), [1, 2, 3])
    uniform = tfd.Uniform(low=low, high=high, validate_args=True)
    expected_quantiles = tf.reshape(tf.linspace(1.01, 1.49, 24), [2, 2, 2, 3])
    cumulative_densities = uniform.cdf(expected_quantiles)
    actual_quantiles = uniform.quantile(cumulative_densities)
    self.assertAllClose(self.evaluate(expected_quantiles),
                        self.evaluate(actual_quantiles))

  def testUniformEntropy(self):
    a_v = np.array([1.0, 1.0, 1.0])
    b_v = np.array([[1.5, 2.0, 3.0]])
    uniform = tfd.Uniform(low=a_v, high=b_v, validate_args=True)

    expected_entropy = np.log(b_v - a_v)
    self.assertAllClose(expected_entropy, self.evaluate(uniform.entropy()))

  def testUniformAssertMaxGtMin(self):
    a_v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    b_v = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with self.assertRaisesOpError('not defined when `low` >= `high`'):
      uniform = tfd.Uniform(low=a_v, high=b_v, validate_args=True)
      self.evaluate(uniform.mean())

  def testUniformSample(self):
    a = tf.constant([3.0, 4.0])
    b = tf.constant(13.0)
    a1_v = 3.0
    a2_v = 4.0
    b_v = 13.0
    n = tf.constant(100000)
    uniform = tfd.Uniform(low=a, high=b, validate_args=True)

    samples = uniform.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 2))
    self.assertAllClose(
        sample_values[::, 0].mean(), (b_v + a1_v) / 2, atol=1e-1, rtol=0.)
    self.assertAllClose(
        sample_values[::, 1].mean(), (b_v + a2_v) / 2, atol=1e-1, rtol=0.)
    self.assertFalse(
        np.any(sample_values[::, 0] < a1_v) or np.any(sample_values >= b_v))
    self.assertFalse(
        np.any(sample_values[::, 1] < a2_v) or np.any(sample_values >= b_v))

  def _testUniformSampleMultiDimensional(self):
    # DISABLED: Please enable this test once b/issues/30149644 is resolved.
    batch_size = 2
    a_v = [3.0, 22.0]
    b_v = [13.0, 35.0]
    a = tf.constant([a_v] * batch_size)
    b = tf.constant([b_v] * batch_size)

    uniform = tfd.Uniform(low=a, high=b, validate_args=True)

    n_v = 100000
    n = tf.constant(n_v)
    samples = uniform.sample(n, seed=test_util.test_seed())
    self.assertEqual(samples.shape, (n_v, batch_size, 2))

    sample_values = self.evaluate(samples)

    self.assertFalse(
        np.any(sample_values[:, 0, 0] < a_v[0]) or
        np.any(sample_values[:, 0, 0] >= b_v[0]))
    self.assertFalse(
        np.any(sample_values[:, 0, 1] < a_v[1]) or
        np.any(sample_values[:, 0, 1] >= b_v[1]))

    self.assertAllClose(
        sample_values[:, 0, 0].mean(), (a_v[0] + b_v[0]) / 2, atol=1e-2)
    self.assertAllClose(
        sample_values[:, 0, 1].mean(), (a_v[1] + b_v[1]) / 2, atol=1e-2)

  def testUniformMean(self):
    a = 10.0
    b = 100.0
    uniform = tfd.Uniform(low=a, high=b, validate_args=True)
    s_uniform = sp_stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.mean()), s_uniform.mean())

  def testUniformVariance(self):
    a = 10.0
    b = 100.0
    uniform = tfd.Uniform(low=a, high=b, validate_args=True)
    s_uniform = sp_stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.variance()), s_uniform.var())

  def testUniformStd(self):
    a = 10.0
    b = 100.0
    uniform = tfd.Uniform(low=a, high=b, validate_args=True)
    s_uniform = sp_stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.stddev()), s_uniform.std())

  def testUniformNans(self):
    a = 10.0
    b = [11.0, 100.0]
    uniform = tfd.Uniform(low=a, high=b, validate_args=False)

    no_nans = tf.constant(1.0)
    nans = tf.constant(0.0) / tf.constant(0.0)
    self.assertTrue(self.evaluate(tf.math.is_nan(nans)))
    with_nans = tf.stack([no_nans, nans])

    pdf = uniform.prob(with_nans)

    is_nan = self.evaluate(tf.math.is_nan(pdf))
    self.assertFalse(is_nan[0])
    self.assertTrue(is_nan[1])

  def testUniformSamplePdf(self):
    a = 10.0
    b = [11.0, 100.0]
    uniform = tfd.Uniform(a, b, validate_args=True)
    samps = uniform.sample(10, seed=test_util.test_seed())
    self.assertTrue(self.evaluate(tf.reduce_all(uniform.prob(samps) > 0)))

  def testUniformBroadcasting(self):
    a = 10.0
    b = [11.0, 20.0]
    uniform = tfd.Uniform(a, b, validate_args=False)

    pdf = uniform.prob([[10.5, 11.5], [9.0, 19.0], [10.5, 21.0]])
    expected_pdf = np.array([[1.0, 0.1], [0.0, 0.1], [1.0, 0.0]])
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testUniformSampleWithShape(self):
    a = 10.0
    b = [11.0, 20.0]
    uniform = tfd.Uniform(a, b, validate_args=True)

    pdf = uniform.prob(uniform.sample((2, 3), seed=test_util.test_seed()))
    # pylint: disable=bad-continuation
    expected_pdf = [
        [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
        [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
    ]
    # pylint: enable=bad-continuation
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

    pdf = uniform.prob(uniform.sample(seed=test_util.test_seed()))
    expected_pdf = [1.0, 0.1]
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testFullyReparameterized(self):
    a = tf.constant(0.1)
    b = tf.constant(0.8)
    _, [grad_a, grad_b] = tfp.math.value_and_gradient(
        lambda a_, b_: (  # pylint: disable=g-long-lambda
            tfd.Uniform(a_, b, validate_args=True).sample(
                100, seed=test_util.test_seed())),
        [a, b])
    self.assertIsNotNone(grad_a)
    self.assertIsNotNone(grad_b)

  def testUniformFloat64(self):
    uniform = tfd.Uniform(
        low=np.float64(0.), high=np.float64(1.), validate_args=True)

    self.assertAllClose(
        [1., 1.],
        self.evaluate(uniform.prob(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(
        [0.5, 0.6],
        self.evaluate(uniform.cdf(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(0.5, self.evaluate(uniform.mean()))
    self.assertAllClose(1 / 12., self.evaluate(uniform.variance()))
    self.assertAllClose(0., self.evaluate(uniform.entropy()))

  def testUniformUniformKLFinite(self):
    batch_size = 6

    a_low = -1.0 * np.arange(1, batch_size + 1)
    a_high = np.array([1.0] * batch_size)
    b_low = -2.0 * np.arange(1, batch_size + 1)
    b_high = np.array([2.0] * batch_size)
    a = tfd.Uniform(low=a_low, high=a_high, validate_args=True)
    b = tfd.Uniform(low=b_low, high=b_high, validate_args=True)

    true_kl = np.log(b_high - b_low) - np.log(a_high - a_low)

    kl = tfd.kl_divergence(a, b)

    # This is essentially an approximated integral from the direct definition
    # of KL divergence.
    x = a.sample(int(1e4), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)

    kl_, kl_sample_ = self.evaluate([kl, kl_sample])
    self.assertAllEqual(true_kl, kl_)
    self.assertAllClose(true_kl, kl_sample_, atol=0.0, rtol=1e-1)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(true_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

  def testUniformUniformKLInfinite(self):

    # This covers three cases:
    # - a.low < b.low,
    # - a.high > b.high, and
    # - both.
    a_low = np.array([-1.0, 0.0, -1.0])
    a_high = np.array([1.0, 2.0, 2.0])
    b_low = np.array([0.0] * 3)
    b_high = np.array([1.0] * 3)
    a = tfd.Uniform(low=a_low, high=a_high, validate_args=True)
    b = tfd.Uniform(low=b_low, high=b_high, validate_args=True)

    # Since 'a' can be sampled to give points outside the support of 'b',
    # the KL Divergence is infinite.
    true_kl = tf.convert_to_tensor(value=np.array([np.inf] * 3))

    kl = tfd.kl_divergence(a, b)

    true_kl_, kl_ = self.evaluate([true_kl, kl])
    self.assertAllEqual(true_kl_, kl_)

  def testPdfAtBoundary(self):
    dist = tfd.Uniform(low=[-2., 3.], high=4., validate_args=True)
    pdf_at_boundary = self.evaluate(dist.prob([[-2., 3.], [3.5, 4.]]))
    self.assertAllFinite(pdf_at_boundary)

  def testAssertValidSample(self):
    dist = tfd.Uniform(low=2., high=5., validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to `low`'):
      self.evaluate(dist.cdf([2.3, 1.7, 4.]))
    with self.assertRaisesOpError('must be less than or equal to `high`'):
      self.evaluate(dist.survival_function([2.3, 5.2, 4.]))

  def testModifiedVariableAssertion(self):
    low = tf.Variable(0.)
    high = tf.Variable(1.)
    self.evaluate([low.initializer, high.initializer])
    uniform = tfd.Uniform(low=low, high=high, validate_args=True)
    with self.assertRaisesOpError('not defined when `low` >= `high`'):
      with tf.control_dependencies([low.assign(2.)]):
        self.evaluate(uniform.mean())

  def testModifiedVariableAssertionSingleVar(self):
    low = tf.Variable(0.)
    high = 1.
    self.evaluate(low.initializer)
    uniform = tfd.Uniform(low=low, high=high, validate_args=True)
    with self.assertRaisesOpError('not defined when `low` >= `high`'):
      with tf.control_dependencies([low.assign(2.)]):
        self.evaluate(uniform.mean())

  def testSupportBijectorOutsideRange(self):
    low = np.array([1., 2., 3., -5.])
    high = np.array([6., 7., 6., 1.])
    dist = tfd.Uniform(low=low, high=high, validate_args=False)
    eps = 1e-6
    x = np.array([1. - eps, 1.5, 6. + eps, -5. - eps])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


if __name__ == '__main__':
  tf.test.main()
