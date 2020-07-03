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
"""Tests for Cauchy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class CauchyTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, array):
    is_finite = np.isfinite(array)
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def _testParamShapes(self, sample_shape, expected):
    param_shapes = tfd.Cauchy.param_shapes(sample_shape)
    loc_shape, scale_shape = param_shapes['loc'], param_shapes['scale']
    self.assertAllEqual(expected, self.evaluate(loc_shape))
    self.assertAllEqual(expected, self.evaluate(scale_shape))
    loc = tf.zeros(loc_shape)
    scale = tf.ones(scale_shape)
    self.assertAllEqual(
        expected,
        self.evaluate(
            tf.shape(
                tfd.Cauchy(loc, scale, validate_args=True).sample(
                    seed=test_util.test_seed()))))

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = tfd.Cauchy.param_static_shapes(sample_shape)
    loc_shape, scale_shape = param_shapes['loc'], param_shapes['scale']
    self.assertEqual(expected, loc_shape)
    self.assertEqual(expected, scale_shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(tf.TensorShape(sample_shape), sample_shape)

  def testCauchyLogPDF(self):
    batch_size = 6
    loc = tf.constant([3.0] * batch_size, dtype=tf.float32)
    scale = tf.constant([np.sqrt(10.0, dtype=np.float32)] * batch_size)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    log_pdf = cauchy.log_prob(x)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(cauchy.batch_shape, log_pdf.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(log_pdf).shape)

    pdf = cauchy.prob(x)
    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(pdf).shape)
    self.assertAllEqual(cauchy.batch_shape, pdf.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(pdf).shape)

    expected_log_pdf = sp_stats.cauchy(self.evaluate(loc),
                                       self.evaluate(scale)).logpdf(x)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testCauchyLogPDFMultidimensional(self):
    batch_size = 6
    loc = tf.constant([[3.0, -3.0]] * batch_size, dtype=tf.float32)
    scale = tf.constant([
        [np.sqrt(10.0, dtype=np.float32),
         np.sqrt(15.0, dtype=np.float32)]] * batch_size)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    log_pdf = cauchy.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(cauchy.batch_shape, log_pdf.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(log_pdf).shape)

    pdf = cauchy.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()), pdf_values.shape)
    self.assertAllEqual(cauchy.batch_shape, pdf.shape)
    self.assertAllEqual(cauchy.batch_shape, pdf_values.shape)

    expected_log_pdf = sp_stats.cauchy(self.evaluate(loc),
                                       self.evaluate(scale)).logpdf(x)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testCauchyCDF(self):
    batch_size = 50
    loc = self._rng.randn(batch_size)
    scale = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)
    cdf = cauchy.cdf(x)
    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(cauchy.batch_shape, cdf.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(cdf).shape)
    expected_cdf = sp_stats.cauchy(loc, scale).cdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0)

  def testCauchySurvivalFunction(self):
    batch_size = 50
    loc = self._rng.randn(batch_size)
    scale = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    sf = cauchy.survival_function(x)
    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), sf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(cauchy.batch_shape, sf.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(sf).shape)
    expected_sf = sp_stats.cauchy(loc, scale).sf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0)

  def testCauchyLogCDF(self):
    batch_size = 50
    loc = self._rng.randn(batch_size)
    scale = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-100.0, 10.0, batch_size).astype(np.float64)

    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    cdf = cauchy.log_cdf(x)
    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(cauchy.batch_shape, cdf.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(cdf).shape)

    expected_cdf = sp_stats.cauchy(loc, scale).logcdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0, rtol=1e-5)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_variable_test
  def testFiniteGradientAtDifficultPoints(self):
    for dtype in [np.float32, np.float64]:
      loc = tf.Variable(dtype(0.0))
      scale = tf.Variable(dtype(1.0))
      x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
      def cauchy_function(name, x):
        def cauchy(loc, scale):
          return getattr(
              tfd.Cauchy(loc=loc, scale=scale, validate_args=True), name)(
                  x)

        return cauchy

      self.evaluate([loc.initializer, scale.initializer])
      for func_name in [
          'cdf', 'log_cdf', 'survival_function',
          'log_survival_function', 'log_prob', 'prob'
      ]:
        value, grads = self.evaluate(tfp.math.value_and_gradient(
            cauchy_function(func_name, x), [loc, scale]))
        self.assertAllFinite(value)
        self.assertAllFinite(grads[0])
        self.assertAllFinite(grads[1])

  def testCauchyLogSurvivalFunction(self):
    batch_size = 50
    loc = self._rng.randn(batch_size)
    scale = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-10.0, 100.0, batch_size).astype(np.float64)

    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    sf = cauchy.log_survival_function(x)
    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), sf.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(cauchy.batch_shape, sf.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(sf).shape)

    expected_sf = sp_stats.cauchy(loc, scale).logsf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0, rtol=1e-5)

  def testCauchyEntropy(self):
    loc = np.array([1.0, 1.0, 1.0])
    scale = np.array([[1.0, 2.0, 3.0]])
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    entropy = cauchy.entropy()
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()), entropy.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(entropy).shape)
    self.assertAllEqual(cauchy.batch_shape, entropy.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(entropy).shape)

    expected_entropy = sp_stats.cauchy(loc, scale[0]).entropy().reshape((1, 3))
    self.assertAllClose(expected_entropy, self.evaluate(entropy))

  def testCauchyMode(self):
    # Mu will be broadcast to [7, 7, 7].
    loc = [7.]
    scale = [11., 12., 13.]

    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    self.assertAllEqual((3,), cauchy.mode().shape)
    self.assertAllEqual([7., 7, 7], self.evaluate(cauchy.mode()))

  def testCauchyMean(self):
    loc = [1., 2., 3.]
    scale = [7.]
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    self.assertAllEqual((3,), cauchy.mean().shape)
    self.assertAllEqual([np.nan] * 3, self.evaluate(cauchy.mean()))

  def testCauchyNanMean(self):
    loc = [1., 2., 3.]
    scale = [7.]
    cauchy = tfd.Cauchy(
        loc=loc, scale=scale, allow_nan_stats=False, validate_args=True)

    with self.assertRaises(ValueError):
      self.evaluate(cauchy.mean())

  def testCauchyQuantile(self):
    batch_size = 50
    loc = self._rng.randn(batch_size)
    scale = self._rng.rand(batch_size) + 1.0
    p = np.linspace(0.000001, 0.999999, batch_size).astype(np.float64)

    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)
    x = cauchy.quantile(p)

    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), x.shape)
    self.assertAllEqual(
        self.evaluate(cauchy.batch_shape_tensor()),
        self.evaluate(x).shape)
    self.assertAllEqual(cauchy.batch_shape, x.shape)
    self.assertAllEqual(cauchy.batch_shape, self.evaluate(x).shape)

    expected_x = sp_stats.cauchy(loc, scale).ppf(p)
    self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  def testCauchyVariance(self):
    # scale will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    scale = [7.]
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    self.assertAllEqual((3,), cauchy.variance().shape)
    self.assertAllEqual([np.nan] * 3, self.evaluate(cauchy.variance()))

  def testCauchyNanVariance(self):
    # scale will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    scale = [7.]
    cauchy = tfd.Cauchy(
        loc=loc, scale=scale, allow_nan_stats=False, validate_args=True)

    with self.assertRaises(ValueError):
      self.evaluate(cauchy.variance())

  def testCauchyStandardDeviation(self):
    # scale will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    scale = [7.]
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    self.assertAllEqual((3,), cauchy.stddev().shape)
    self.assertAllEqual([np.nan] * 3, self.evaluate(cauchy.stddev()))

  def testCauchyNanStandardDeviation(self):
    # scale will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    scale = [7.]
    cauchy = tfd.Cauchy(
        loc=loc, scale=scale, allow_nan_stats=False, validate_args=True)

    with self.assertRaises(ValueError):
      self.evaluate(cauchy.stddev())

  def testCauchySample(self):
    loc = tf.constant(3.0)
    scale = tf.constant(1.0)
    loc_v = 3.0
    n = tf.constant(100000)
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)
    samples = cauchy.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(sample_values.shape, (100000,))
    self.assertAllClose(np.median(sample_values), loc_v, atol=1e-1)

    expected_shape = tf.TensorShape([self.evaluate(n)]).concatenate(
        tf.TensorShape(self.evaluate(cauchy.batch_shape_tensor())))

    self.assertAllEqual(expected_shape, samples.shape)
    self.assertAllEqual(expected_shape, sample_values.shape)

    expected_shape = (
        tf.TensorShape([self.evaluate(n)]).concatenate(cauchy.batch_shape))

    self.assertAllEqual(expected_shape, samples.shape)
    self.assertAllEqual(expected_shape, sample_values.shape)

  def testCauchySampleMultiDimensional(self):
    batch_size = 2
    loc = tf.constant([[3.0, -3.0]] * batch_size)
    scale = tf.constant([[0.5, 1.0]] * batch_size)
    loc_v = [3.0, -3.0]
    n = tf.constant(100000)
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)
    samples = cauchy.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (100000, batch_size, 2))
    self.assertAllClose(np.median(sample_values[:, 0, 0]), loc_v[0], atol=1e-1)
    self.assertAllClose(np.median(sample_values[:, 0, 1]), loc_v[1], atol=1e-1)

    expected_shape = tf.TensorShape([self.evaluate(n)]).concatenate(
        tf.TensorShape(self.evaluate(cauchy.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, samples.shape)
    self.assertAllEqual(expected_shape, sample_values.shape)

    expected_shape = (
        tf.TensorShape([self.evaluate(n)]).concatenate(cauchy.batch_shape))
    self.assertAllEqual(expected_shape, samples.shape)
    self.assertAllEqual(expected_shape, sample_values.shape)

  def testCauchyNegativeScaleFails(self):
    with self.assertRaisesOpError('`scale` must be positive.'):
      cauchy = tfd.Cauchy(loc=[1.], scale=[-5.], validate_args=True)
      self.evaluate(cauchy.mode())

  def testCauchyShape(self):
    loc = tf.constant([-3.0] * 5)
    scale = tf.constant(11.0)
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    self.assertEqual(self.evaluate(cauchy.batch_shape_tensor()), [5])
    self.assertEqual(cauchy.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(cauchy.event_shape_tensor()), [])
    self.assertEqual(cauchy.event_shape, tf.TensorShape([]))

  def testCauchyShapeWithPlaceholders(self):
    if tf.executing_eagerly():
      return
    loc = tf1.placeholder_with_default(5., shape=[])
    scale = tf1.placeholder_with_default([1., 2], shape=None)
    cauchy = tfd.Cauchy(loc=loc, scale=scale, validate_args=True)

    # get_batch_shape should return an '<unknown>' tensor.
    self.assertEqual(cauchy.batch_shape, tf.TensorShape(None))
    self.assertEqual(cauchy.event_shape, ())
    self.assertAllEqual(self.evaluate(cauchy.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(cauchy.batch_shape_tensor()), [2])

  def testAssertsPositiveScale(self):
    loc, scale = (
        tf.Variable([-1, 0, 1], dtype=tf.float64),
        tf.Variable([-1, 2, 3], dtype=tf.float64)
    )
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      d = tfd.Cauchy(loc, scale, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveScaleAfterMutation(self):
    loc, scale = (
        tf.Variable([-1., 0., 1.]),
        tf.Variable([1., 2., 3.])
    )
    d = tfd.Cauchy(loc, scale, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([scale.assign([1., 2., -3.])]):
        self.evaluate(d.mean())

  def testCauchyCauchyKLWithAnalytic(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    gamma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    gamma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    c_a = tfd.Cauchy(loc=mu_a, scale=gamma_a, validate_args=True)
    c_b = tfd.Cauchy(loc=mu_b, scale=gamma_b, validate_args=True)

    kl = tfd.kl_divergence(c_a, c_b)
    reverse_kl = tfd.kl_divergence(c_b, c_a)
    kl_val = self.evaluate(kl)
    reverse_kl_val = self.evaluate(reverse_kl)

    kl_expected = np.log(
        (gamma_a + gamma_b)**2 + (mu_a - mu_b)**2
        ) - np.log(4. * gamma_a * gamma_b)

    self.assertEqual(kl.shape, (batch_size,))

    # The Cauchy KL is symmetric
    self.assertAllClose(kl_val, kl_expected)
    self.assertAllClose(reverse_kl_val, kl_expected)

  def testCauchyCauchyKLWithMC(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    gamma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    gamma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    c_a = tfd.Cauchy(loc=mu_a, scale=gamma_a, validate_args=True)
    c_b = tfd.Cauchy(loc=mu_b, scale=gamma_b, validate_args=True)

    kl = tfd.kl_divergence(c_a, c_b)
    reverse_kl = tfd.kl_divergence(c_b, c_a)
    kl_val = self.evaluate(kl)
    reverse_kl_val = self.evaluate(reverse_kl)

    x = c_a.sample(int(1e5), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(c_a.log_prob(x) - c_b.log_prob(x), axis=0)
    kl_sample_val = self.evaluate(kl_sample)

    self.assertEqual(kl.shape, (batch_size,))

    # The Cauchy KL is symmetric
    self.assertAllClose(kl_val, kl_sample_val, atol=0.0, rtol=1e-2)
    self.assertAllClose(reverse_kl_val, kl_sample_val, atol=0.0, rtol=1e-2)

if __name__ == '__main__':
  tf.test.main()
