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
"""Tests for ExponentiallyModifiedGaussian Distribution."""

import math

# Dependency imports

import numpy as np
from scipy import stats as sp_stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import exponentially_modified_gaussian as emg
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


class _ExponentiallyModifiedGaussianTest(object):

  def _test_param_shapes(self, sample_shape, expected):
    param_shapes = emg.ExponentiallyModifiedGaussian.param_shapes(sample_shape)
    mu_shape, sigma_shape, lambda_shape = param_shapes['loc'], param_shapes[
        'scale'], param_shapes['rate']
    self.assertAllEqual(expected, self.evaluate(mu_shape))
    self.assertAllEqual(expected, self.evaluate(sigma_shape))
    self.assertAllEqual(expected, self.evaluate(lambda_shape))
    mu = tf.zeros(mu_shape, dtype=self.dtype)
    sigma = tf.ones(sigma_shape, dtype=self.dtype)
    rate = tf.ones(lambda_shape, dtype=self.dtype)
    self.assertAllEqual(
        expected,
        self.evaluate(
            tf.shape(
                emg.ExponentiallyModifiedGaussian(
                    mu, sigma, rate,
                    validate_args=True).sample(seed=test_util.test_seed()))))

  def _test_param_static_shapes(self, sample_shape, expected):
    param_shapes = emg.ExponentiallyModifiedGaussian.param_static_shapes(
        sample_shape)
    mu_shape, sigma_shape, lambda_shape = param_shapes['loc'], param_shapes[
        'scale'], param_shapes['rate']
    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)
    self.assertEqual(expected, lambda_shape)

  # Currently fails for numpy due to a bug in the types returned by
  # special_math.ndtr
  # As of now, numpy testing is disabled in the BUILD file
  def testSampleLikeArgsGetDistDType(self):
    zero = dtype_util.as_numpy_dtype(self.dtype)(0.)
    one = dtype_util.as_numpy_dtype(self.dtype)(1.)
    dist = emg.ExponentiallyModifiedGaussian(zero, one, one)
    self.assertEqual(self.dtype, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf',
                   'log_survival_function', 'survival_function'):
      self.assertEqual(self.dtype, getattr(dist, method)(one).dtype, msg=method)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._test_param_shapes(sample_shape, sample_shape)
    self._test_param_shapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._test_param_static_shapes(sample_shape, sample_shape)
    self._test_param_static_shapes(tf.TensorShape(sample_shape), sample_shape)

  def testExponentiallyModifiedGaussianLogPDF(self):
    batch_size = 6
    mu = tf.constant([3.0] * batch_size, dtype=self.dtype)
    sigma = tf.constant([math.sqrt(10.0)] * batch_size, dtype=self.dtype)
    rate = tf.constant([2.] * batch_size, dtype=self.dtype)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=self.dtype)
    exgaussian = emg.ExponentiallyModifiedGaussian(
        loc=mu, scale=sigma, rate=rate, validate_args=True)

    log_pdf = exgaussian.log_prob(x)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(exgaussian.batch_shape, log_pdf.shape)
    self.assertAllEqual(exgaussian.batch_shape, self.evaluate(log_pdf).shape)

    pdf = exgaussian.prob(x)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()),
        self.evaluate(pdf).shape)
    self.assertAllEqual(exgaussian.batch_shape, pdf.shape)
    self.assertAllEqual(exgaussian.batch_shape, self.evaluate(pdf).shape)

    expected_log_pdf = sp_stats.exponnorm(
        1. / (self.evaluate(rate) * self.evaluate(sigma)),
        loc=self.evaluate(mu),
        scale=self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(
        expected_log_pdf, self.evaluate(log_pdf), atol=1e-5, rtol=1e-5)
    self.assertAllClose(
        np.exp(expected_log_pdf), self.evaluate(pdf), atol=1e-5, rtol=1e-5)

  def testExponentiallyModifiedGaussianLogPDFMultidimensional(self):
    batch_size = 6
    mu = tf.constant([[3.0, -3.0]] * batch_size, dtype=self.dtype)
    sigma = tf.constant(
        [[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size, dtype=self.dtype)
    rate = tf.constant([[2., 3.]] * batch_size, dtype=self.dtype)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=self.dtype).T
    exgaussian = emg.ExponentiallyModifiedGaussian(
        loc=mu, scale=sigma, rate=rate, validate_args=True)

    log_pdf = exgaussian.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(exgaussian.batch_shape, log_pdf.shape)
    self.assertAllEqual(exgaussian.batch_shape, self.evaluate(log_pdf).shape)

    pdf = exgaussian.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()), pdf_values.shape)
    self.assertAllEqual(exgaussian.batch_shape, pdf.shape)
    self.assertAllEqual(exgaussian.batch_shape, pdf_values.shape)

    expected_log_pdf = sp_stats.exponnorm(
        1. / (self.evaluate(rate) * self.evaluate(sigma)),
        loc=self.evaluate(mu),
        scale=self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(expected_log_pdf, log_pdf_values, atol=1e-5, rtol=1e-5)
    self.assertAllClose(
        np.exp(expected_log_pdf), pdf_values, atol=1e-5, rtol=1e-5)

  def testExponentiallyModifiedGaussianLogCDF(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    rate = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(self.dtype)

    exgaussian = emg.ExponentiallyModifiedGaussian(
        loc=mu, scale=sigma, rate=rate, validate_args=True)
    cdf = exgaussian.log_cdf(x)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(exgaussian.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(exgaussian.batch_shape, cdf.shape)
    self.assertAllEqual(exgaussian.batch_shape, self.evaluate(cdf).shape)
    expected_cdf = sp_stats.exponnorm(
        1. / (rate * sigma), loc=mu, scale=sigma).logcdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0)

  @test_util.numpy_disable_gradient_test
  def testFiniteGradientAtDifficultPoints(self):

    def make_fn(attr):
      x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(self.dtype)
      return lambda m, s, l: getattr(  # pylint: disable=g-long-lambda
          emg.ExponentiallyModifiedGaussian(
              loc=m, scale=s, rate=l, validate_args=True), attr)(
                  x)

    for attr in ['cdf', 'log_prob']:
      value, grads = self.evaluate(
          gradient.value_and_gradient(
              make_fn(attr), [
                  tf.constant(0, self.dtype),
                  tf.constant(1, self.dtype),
                  tf.constant(1, self.dtype)
              ]))
      self.assertAllFinite(value)
      self.assertAllFinite(grads[0])
      self.assertAllFinite(grads[1])

  def testNegativeSigmaFails(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      exgaussian = emg.ExponentiallyModifiedGaussian(
          loc=[tf.constant(1., dtype=self.dtype)],
          scale=[tf.constant(-5., dtype=self.dtype)],
          rate=[tf.constant(1., dtype=self.dtype)],
          validate_args=True,
          name='G')
      self.evaluate(exgaussian.mean())

  def testExponentiallyModifiedGaussianShape(self):
    mu = tf.constant([-3.0] * 5, dtype=self.dtype)
    sigma = tf.constant(11.0, dtype=self.dtype)
    rate = tf.constant(6.0, dtype=self.dtype)
    exgaussian = emg.ExponentiallyModifiedGaussian(
        loc=mu, scale=sigma, rate=rate, validate_args=True)

    self.assertEqual(self.evaluate(exgaussian.batch_shape_tensor()), [5])
    self.assertEqual(exgaussian.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(exgaussian.event_shape_tensor()), [])
    self.assertEqual(exgaussian.event_shape, tf.TensorShape([]))

  def testVariableScale(self):
    x = tf.Variable(1., dtype=self.dtype)
    d = emg.ExponentiallyModifiedGaussian(
        loc=tf.constant(0., dtype=self.dtype),
        scale=x,
        rate=tf.constant(1., dtype=self.dtype),
        validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.assertIs(x, d.scale)
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(d.mean())

  def testIncompatibleArgShapes(self):
    with self.assertRaisesRegex(Exception, r'compatible shapes'):
      d = emg.ExponentiallyModifiedGaussian(
          loc=tf.zeros([2, 3], dtype=self.dtype),
          scale=tf.ones([4, 1], dtype=self.dtype),
          rate=tf.ones([2, 3], dtype=self.dtype),
          validate_args=True)
      self.evaluate(d.mean())


@test_util.test_all_tf_execution_regimes
class ExponentiallyModifiedGaussianTestFloat32(
    test_util.TestCase, _ExponentiallyModifiedGaussianTest):
  dtype = np.float32

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(ExponentiallyModifiedGaussianTestFloat32, self).setUp()


@test_util.test_all_tf_execution_regimes
class ExponentiallyModifiedGaussianTestFloat64(
    test_util.TestCase, _ExponentiallyModifiedGaussianTest):
  dtype = np.float64

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(ExponentiallyModifiedGaussianTestFloat64, self).setUp()


if __name__ == '__main__':
  test_util.main()
