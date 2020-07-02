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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class HalfNormalTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, array):
    is_finite = np.isfinite(array)
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def assertRaisesError(self, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(Exception, msg)
    return self.assertRaisesOpError(msg)

  def _testParamShapes(self, sample_shape, expected):
    param_shapes = tfd.HalfNormal.param_shapes(sample_shape)
    scale_shape = param_shapes['scale']
    self.assertAllEqual(expected, self.evaluate(scale_shape))
    scale = tf.ones(scale_shape)
    self.assertAllEqual(
        expected,
        self.evaluate(
            tf.shape(tfd.HalfNormal(scale, validate_args=True).sample(
                seed=test_util.test_seed()))))

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = tfd.HalfNormal.param_static_shapes(sample_shape)
    scale_shape = param_shapes['scale']
    self.assertEqual(expected, scale_shape)

  def _testBatchShapes(self, dist, tensor):
    self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), tensor.shape)
    self.assertAllEqual(
        self.evaluate(dist.batch_shape_tensor()), self.evaluate(tensor).shape)
    self.assertAllEqual(dist.batch_shape, tensor.shape)
    self.assertAllEqual(dist.batch_shape, self.evaluate(tensor).shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(tf.TensorShape(sample_shape), sample_shape)

  def testHalfNormalLogPDF(self):
    batch_size = 6
    scale = tf.constant([3.0] * batch_size)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=False)

    log_pdf = halfnorm.log_prob(x)
    self._testBatchShapes(halfnorm, log_pdf)

    pdf = halfnorm.prob(x)
    self._testBatchShapes(halfnorm, pdf)

    expected_log_pdf = sp_stats.halfnorm(scale=self.evaluate(scale)).logpdf(x)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testHalfNormalLogPDFMultidimensional(self):
    batch_size = 6
    scale = tf.constant([[3.0, 1.0]] * batch_size)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=False)

    log_pdf = halfnorm.log_prob(x)
    self._testBatchShapes(halfnorm, log_pdf)

    pdf = halfnorm.prob(x)
    self._testBatchShapes(halfnorm, pdf)

    expected_log_pdf = sp_stats.halfnorm(scale=self.evaluate(scale)).logpdf(x)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testHalfNormalCDF(self):
    batch_size = 50
    scale = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=False)

    cdf = halfnorm.cdf(x)
    self._testBatchShapes(halfnorm, cdf)

    log_cdf = halfnorm.log_cdf(x)
    self._testBatchShapes(halfnorm, log_cdf)

    expected_logcdf = sp_stats.halfnorm(scale=scale).logcdf(x)
    self.assertAllClose(expected_logcdf, self.evaluate(log_cdf), atol=0)
    self.assertAllClose(np.exp(expected_logcdf), self.evaluate(cdf), atol=0)

  def testHalfNormalSurvivalFunction(self):
    batch_size = 50
    scale = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=False)

    sf = halfnorm.survival_function(x)
    self._testBatchShapes(halfnorm, sf)

    log_sf = halfnorm.log_survival_function(x)
    self._testBatchShapes(halfnorm, log_sf)

    expected_logsf = sp_stats.halfnorm(scale=scale).logsf(x)
    self.assertAllClose(expected_logsf, self.evaluate(log_sf), atol=0)
    self.assertAllClose(np.exp(expected_logsf), self.evaluate(sf), atol=0)

  def testHalfNormalQuantile(self):
    batch_size = 50
    scale = self._rng.rand(batch_size) + 1.0
    p = np.linspace(0., 1.0, batch_size).astype(np.float64)

    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)
    x = halfnorm.quantile(p)
    self._testBatchShapes(halfnorm, x)

    expected_x = sp_stats.halfnorm(scale=scale).ppf(p)
    self.assertAllClose(expected_x, self.evaluate(x), atol=0)

  @parameterized.parameters(np.float32, np.float64)
  def testFiniteGradients(self, dtype):
    scale = tf.constant(dtype(3.0))
    x = np.array([0.01, 0.1, 1., 5., 10.]).astype(dtype)
    def half_normal_function(name, x):
      def half_normal(scale):
        return getattr(tfd.HalfNormal(scale=scale, validate_args=True), name)(
            x)

      return half_normal

    for func_name in [
        'cdf', 'log_cdf', 'survival_function',
        'log_prob', 'prob', 'log_survival_function',
    ]:
      print(func_name)
      value, grads = self.evaluate(tfp.math.value_and_gradient(
          half_normal_function(func_name, x), scale))
      self.assertAllFinite(value)
      self.assertAllFinite(grads)

  def testHalfNormalEntropy(self):
    scale = np.array([[1.0, 2.0, 3.0]])
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)

    # See https://en.wikipedia.org/wiki/Half-normal_distribution for the
    # entropy formula used here.
    expected_entropy = 0.5 * np.log(np.pi * scale**2.0 / 2.0) + 0.5

    entropy = halfnorm.entropy()
    self._testBatchShapes(halfnorm, entropy)
    self.assertAllClose(expected_entropy, self.evaluate(entropy))

  def testHalfNormalMeanAndMode(self):
    scale = np.array([11., 12., 13.])

    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)
    expected_mean = scale * np.sqrt(2.0) / np.sqrt(np.pi)

    self.assertAllEqual((3,), self.evaluate(halfnorm.mean()).shape)
    self.assertAllEqual(expected_mean, self.evaluate(halfnorm.mean()))

    self.assertAllEqual((3,), self.evaluate(halfnorm.mode()).shape)
    self.assertAllEqual([0., 0., 0.], self.evaluate(halfnorm.mode()))

  def testHalfNormalVariance(self):
    scale = np.array([7., 7., 7.])
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)
    expected_variance = scale**2.0 * (1.0 - 2.0 / np.pi)

    self.assertAllEqual((3,), self.evaluate(halfnorm.variance()).shape)
    self.assertAllEqual(expected_variance, self.evaluate(halfnorm.variance()))

  def testHalfNormalStandardDeviation(self):
    scale = np.array([7., 7., 7.])
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)
    expected_variance = scale**2.0 * (1.0 - 2.0 / np.pi)

    self.assertAllEqual((3,), halfnorm.stddev().shape)
    self.assertAllEqual(
        np.sqrt(expected_variance), self.evaluate(halfnorm.stddev()))

  def testHalfNormalSample(self):
    scale = tf.constant(3.0)
    n = tf.constant(100000)
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)

    sample = halfnorm.sample(n, seed=test_util.test_seed())

    self.assertEqual(self.evaluate(sample).shape, (100000,))
    self.assertAllClose(
        self.evaluate(sample).mean(),
        3.0 * np.sqrt(2.0) / np.sqrt(np.pi),
        atol=1e-1)

    expected_shape = tf.TensorShape([self.evaluate(n)]).concatenate(
        tf.TensorShape(self.evaluate(halfnorm.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, sample.shape)
    self.assertAllEqual(expected_shape, self.evaluate(sample).shape)

    expected_shape_static = (
        tf.TensorShape([self.evaluate(n)]).concatenate(halfnorm.batch_shape))
    self.assertAllEqual(expected_shape_static, sample.shape)
    self.assertAllEqual(expected_shape_static, self.evaluate(sample).shape)

  def testHalfNormalSampleMultiDimensional(self):
    batch_size = 2
    scale = tf.constant([[2.0, 3.0]] * batch_size)
    n = tf.constant(100000)
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)

    sample = halfnorm.sample(n, seed=test_util.test_seed())
    self.assertEqual(sample.shape, (100000, batch_size, 2))
    self.assertAllClose(
        self.evaluate(sample)[:, 0, 0].mean(),
        2.0 * np.sqrt(2.0) / np.sqrt(np.pi),
        atol=1e-1)
    self.assertAllClose(
        self.evaluate(sample)[:, 0, 1].mean(),
        3.0 * np.sqrt(2.0) / np.sqrt(np.pi),
        atol=1e-1)

    expected_shape = tf.TensorShape([self.evaluate(n)]).concatenate(
        tf.TensorShape(self.evaluate(halfnorm.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, sample.shape)
    self.assertAllEqual(expected_shape, self.evaluate(sample).shape)

    expected_shape_static = (
        tf.TensorShape([self.evaluate(n)]).concatenate(halfnorm.batch_shape))
    self.assertAllEqual(expected_shape_static, sample.shape)
    self.assertAllEqual(expected_shape_static, self.evaluate(sample).shape)

  def testAssertValidSample(self):
    d = tfd.HalfNormal(scale=[2., 3.], validate_args=True)
    with self.assertRaisesOpError('Sample must be non-negative.'):
      d.log_prob(-0.2)

  def testPdfAtBoundary(self):
    d = tfd.HalfNormal(scale=[4., 6., 1.3], validate_args=True)
    log_pdf_at_boundary = self.evaluate(d.log_prob(0.))
    self.assertAllFinite(log_pdf_at_boundary)

  def testNegativeSigmaFails(self):
    with self.assertRaisesError('Argument `scale` must be positive.'):
      halfnorm = tfd.HalfNormal(scale=[-5.], validate_args=True, name='G')
      self.evaluate(halfnorm.mean())

  def testHalfNormalShape(self):
    scale = tf.constant([6.0] * 5)
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)

    self.assertEqual(self.evaluate(halfnorm.batch_shape_tensor()), [5])
    self.assertEqual(halfnorm.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(halfnorm.event_shape_tensor()), [])
    self.assertEqual(halfnorm.event_shape, tf.TensorShape([]))

  def testHalfNormalShapeWithPlaceholders(self):
    if tf.executing_eagerly():
      return
    scale = tf1.placeholder_with_default([1., 2], shape=None)
    halfnorm = tfd.HalfNormal(scale=scale, validate_args=True)

    # get_batch_shape should return an '<unknown>' tensor.
    self.assertEqual(halfnorm.batch_shape, tf.TensorShape(None))
    self.assertEqual(halfnorm.event_shape, ())
    self.assertAllEqual(self.evaluate(halfnorm.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(halfnorm.batch_shape_tensor()), [2])

  def testHalfNormalHalfNormalKL(self):
    a_scale = np.arange(0.5, 1.6, 0.1)
    b_scale = np.arange(0.5, 1.6, 0.1)

    # This reshape is intended to expand the number of test cases.
    a_scale = a_scale.reshape((len(a_scale), 1))
    b_scale = b_scale.reshape((1, len(b_scale)))

    a = tfd.HalfNormal(scale=a_scale, validate_args=True)
    b = tfd.HalfNormal(scale=b_scale, validate_args=True)

    true_kl = (np.log(b_scale) - np.log(a_scale) +
               (a_scale ** 2 - b_scale ** 2) / (2 * b_scale ** 2))

    kl = tfd.kl_divergence(a, b)

    x = a.sample(int(4e5), seed=test_util.test_seed(hardcoded_seed=0))
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)

    kl_, kl_sample_ = self.evaluate([kl, kl_sample])
    self.assertAllEqual(true_kl, kl_)
    self.assertAllClose(true_kl, kl_sample_, atol=0., rtol=5e-2)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(zero_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

  @test_util.tf_tape_safety_test
  def testGradientThroughScale(self):
    scale = tf.Variable(2.)
    d = tfd.HalfNormal(scale=scale, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 3.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveScale(self):
    scale = tf.Variable([1., 2., -3.])
    with self.assertRaisesError('Argument `scale` must be positive.'):
      d = tfd.HalfNormal(scale=scale, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveScaleAfterMutation(self):
    scale = tf.Variable([1., 2., 3.])
    d = tfd.HalfNormal(scale=scale, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesError('Argument `scale` must be positive.'):
      with tf.control_dependencies([scale.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testSupportBijectorOutsideRange(self):
    dist = tfd.HalfNormal(scale=[3.1, 2., 5.4], validate_args=True)
    x = np.array([-4.2, -1e-6, -1.3])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  tf.test.main()
