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

import math

# Dependency imports

import numpy as np
from scipy import stats as sp_stats
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util

from tensorflow.python.framework import test_util as tf_test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.test_all_tf_execution_regimes
class NormalTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(NormalTest, self).setUp()

  def _testParamShapes(self, sample_shape, expected):
    param_shapes = tfd.Normal.param_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes['loc'], param_shapes['scale']
    self.assertAllEqual(expected, self.evaluate(mu_shape))
    self.assertAllEqual(expected, self.evaluate(sigma_shape))
    mu = tf.zeros(mu_shape)
    sigma = tf.ones(sigma_shape)
    self.assertAllEqual(
        expected,
        self.evaluate(
            tf.shape(tfd.Normal(mu, sigma, validate_args=True).sample(
                seed=test_util.test_seed()))))

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = tfd.Normal.param_static_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes['loc'], param_shapes['scale']
    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)

  def testSampleLikeArgsGetDistDType(self):
    dist = tfd.Normal(0., 1.)
    self.assertEqual(tf.float32, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf',
                   'log_survival_function', 'survival_function', 'quantile'):
      self.assertEqual(tf.float32, getattr(dist, method)(1).dtype)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(
        tf.TensorShape(sample_shape), sample_shape)

  def testNormalLogPDF(self):
    batch_size = 6
    mu = tf.constant([3.0] * batch_size)
    sigma = tf.constant([math.sqrt(10.0)] * batch_size)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    log_pdf = normal.log_prob(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(normal.batch_shape, log_pdf.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(log_pdf).shape)

    pdf = normal.prob(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(pdf).shape)
    self.assertAllEqual(normal.batch_shape, pdf.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(pdf).shape)

    expected_log_pdf = sp_stats.norm(self.evaluate(mu),
                                     self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testNormalLogPDFMultidimensional(self):
    batch_size = 6
    mu = tf.constant([[3.0, -3.0]] * batch_size)
    sigma = tf.constant(
        [[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    log_pdf = normal.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(normal.batch_shape, log_pdf.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(log_pdf).shape)

    pdf = normal.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), pdf_values.shape)
    self.assertAllEqual(normal.batch_shape, pdf.shape)
    self.assertAllEqual(normal.batch_shape, pdf_values.shape)

    expected_log_pdf = sp_stats.norm(self.evaluate(mu),
                                     self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testNormalCDF(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)
    cdf = normal.cdf(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(normal.batch_shape, cdf.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(cdf).shape)
    expected_cdf = sp_stats.norm(mu, sigma).cdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0)

  def testNormalSurvivalFunction(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    sf = normal.survival_function(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), sf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(normal.batch_shape, sf.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(sf).shape)
    expected_sf = sp_stats.norm(mu, sigma).sf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0)

  def testNormalLogCDF(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-100.0, 10.0, batch_size).astype(np.float64)

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    cdf = normal.log_cdf(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(normal.batch_shape, cdf.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(cdf).shape)

    expected_cdf = sp_stats.norm(mu, sigma).logcdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0, rtol=1e-3)

  def testFiniteGradientAtDifficultPoints(self):
    def make_fn(dtype, attr):
      x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
      return lambda m, s: getattr(  # pylint: disable=g-long-lambda
          tfd.Normal(loc=m, scale=s, validate_args=True), attr)(
              x)

    for dtype in np.float32, np.float64:
      for attr in ['cdf', 'log_cdf', 'survival_function',
                   'log_survival_function', 'log_prob', 'prob']:
        value, grads = self.evaluate(tfp.math.value_and_gradient(
            make_fn(dtype, attr),
            [tf.constant(0, dtype), tf.constant(1, dtype)]))
        self.assertAllFinite(value)
        self.assertAllFinite(grads[0])
        self.assertAllFinite(grads[1])

  def testNormalLogSurvivalFunction(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-10.0, 100.0, batch_size).astype(np.float64)

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    sf = normal.log_survival_function(x)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), sf.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(normal.batch_shape, sf.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(sf).shape)

    expected_sf = sp_stats.norm(mu, sigma).logsf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0, rtol=1e-5)

  def testNormalEntropyWithScalarInputs(self):
    # Scipy.sp_stats.norm cannot deal with the shapes in the other test.
    mu_v = 2.34
    sigma_v = 4.56
    normal = tfd.Normal(loc=mu_v, scale=sigma_v, validate_args=True)

    entropy = normal.entropy()
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), entropy.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(entropy).shape)
    self.assertAllEqual(normal.batch_shape, entropy.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(entropy).shape)
    # scipy.sp_stats.norm cannot deal with these shapes.
    expected_entropy = sp_stats.norm(mu_v, sigma_v).entropy()
    self.assertAllClose(expected_entropy, self.evaluate(entropy))

  def testNormalEntropy(self):
    mu_v = np.array([1.0, 1.0, 1.0])
    sigma_v = np.array([[1.0, 2.0, 3.0]]).T
    normal = tfd.Normal(loc=mu_v, scale=sigma_v, validate_args=True)

    # scipy.sp_stats.norm cannot deal with these shapes.
    sigma_broadcast = mu_v * sigma_v
    expected_entropy = 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_broadcast**2)
    entropy = normal.entropy()
    np.testing.assert_allclose(expected_entropy, self.evaluate(entropy))
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), entropy.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(entropy).shape)
    self.assertAllEqual(normal.batch_shape, entropy.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(entropy).shape)

  def testNormalMeanAndMode(self):
    # Mu will be broadcast to [7, 7, 7].
    mu = [7.]
    sigma = [11., 12., 13.]

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    self.assertAllEqual((3,), normal.mean().shape)
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.mean()))

    self.assertAllEqual((3,), normal.mode().shape)
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.mode()))

  def testNormalQuantile(self):
    batch_size = 52
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    p = np.linspace(0., 1.0, batch_size - 2).astype(np.float64)
    # Quantile performs piecewise rational approximation so adding some
    # special input values to make sure we hit all the pieces.
    p = np.hstack((p, np.exp(-33), 1. - np.exp(-33)))

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)
    x = normal.quantile(p)

    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), x.shape)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(x).shape)
    self.assertAllEqual(normal.batch_shape, x.shape)
    self.assertAllEqual(normal.batch_shape, self.evaluate(x).shape)

    expected_x = sp_stats.norm(mu, sigma).ppf(p)
    self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  def _baseQuantileFiniteGradientAtDifficultPoints(self, dtype):
    mu = tf.constant(dtype(0))
    sigma = tf.constant(dtype(1))
    p = tf.constant(dtype([np.exp(-32.), np.exp(-2.),
                           1. - np.exp(-2.), 1. - np.exp(-8.)]))
    value, grads = tfp.math.value_and_gradient(
        lambda m, p_: tfd.Normal(loc=m, scale=sigma, validate_args=True).  # pylint: disable=g-long-lambda
        quantile(p_), [mu, p])
    value, grads = self.evaluate([value, grads])
    self.assertAllFinite(grads[0])
    self.assertAllFinite(grads[1])

  def testQuantileFiniteGradientAtDifficultPointsFloat32(self):
    self._baseQuantileFiniteGradientAtDifficultPoints(np.float32)

  def testQuantileFiniteGradientAtDifficultPointsFloat64(self):
    self._baseQuantileFiniteGradientAtDifficultPoints(np.float64)

  def testNormalVariance(self):
    # sigma will be broadcast to [7, 7, 7]
    mu = [1., 2., 3.]
    sigma = [7.]

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    self.assertAllEqual((3,), normal.variance().shape)
    self.assertAllEqual([49., 49, 49], self.evaluate(normal.variance()))

  def testNormalStandardDeviation(self):
    # sigma will be broadcast to [7, 7, 7]
    mu = [1., 2., 3.]
    sigma = [7.]

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    self.assertAllEqual((3,), normal.stddev().shape)
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.stddev()))

  def testNormalSample(self):
    mu = tf.constant(3.0)
    sigma = tf.constant(math.sqrt(3.0))
    mu_v = 3.0
    sigma_v = np.sqrt(3.0)
    n = tf.constant(100000)
    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)
    samples = normal.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
    # The sample variance similarly is dependent on sigma and n.
    # Thus, the tolerances below are very sensitive to number of samples
    # as well as the variances chosen.
    self.assertEqual(sample_values.shape, (100000,))
    self.assertAllClose(sample_values.mean(), mu_v, atol=1e-1)
    self.assertAllClose(sample_values.std(), sigma_v, atol=1e-1)

    expected_samples_shape = tf.TensorShape(
        [self.evaluate(n)]).concatenate(
            tf.TensorShape(
                self.evaluate(normal.batch_shape_tensor())))

    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

    expected_samples_shape = (
        tf.TensorShape([self.evaluate(n)]).concatenate(
            normal.batch_shape))

    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNormalFullyReparameterized(self):
    mu = tf.constant(4.0)
    sigma = tf.constant(3.0)
    _, [grad_mu, grad_sigma] = tfp.math.value_and_gradient(
        lambda m, s: tfd.Normal(loc=m, scale=s, validate_args=True).sample(  # pylint: disable=g-long-lambda
            100, seed=test_util.test_seed()),
        [mu, sigma])
    grad_mu, grad_sigma = self.evaluate([grad_mu, grad_sigma])
    self.assertIsNotNone(grad_mu)
    self.assertIsNotNone(grad_sigma)

  def testNormalSampleMultiDimensional(self):
    batch_size = 2
    mu = tf.constant([[3.0, -3.0]] * batch_size)
    sigma = tf.constant(
        [[math.sqrt(2.0), math.sqrt(3.0)]] * batch_size)
    mu_v = [3.0, -3.0]
    sigma_v = [np.sqrt(2.0), np.sqrt(3.0)]
    n = tf.constant(100000)
    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)
    samples = normal.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
    # The sample variance similarly is dependent on sigma and n.
    # Thus, the tolerances below are very sensitive to number of samples
    # as well as the variances chosen.
    self.assertEqual(samples.shape, (100000, batch_size, 2))
    self.assertAllClose(sample_values[:, 0, 0].mean(), mu_v[0], atol=1e-1)
    self.assertAllClose(sample_values[:, 0, 0].std(), sigma_v[0], atol=1e-1)
    self.assertAllClose(sample_values[:, 0, 1].mean(), mu_v[1], atol=1e-1)
    self.assertAllClose(sample_values[:, 0, 1].std(), sigma_v[1], atol=1e-1)

    expected_samples_shape = tf.TensorShape(
        [self.evaluate(n)]).concatenate(
            tf.TensorShape(
                self.evaluate(normal.batch_shape_tensor())))
    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

    expected_samples_shape = (
        tf.TensorShape([self.evaluate(n)]).concatenate(
            normal.batch_shape))
    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNegativeSigmaFails(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      normal = tfd.Normal(loc=[1.], scale=[-5.], validate_args=True, name='G')
      self.evaluate(normal.mean())

  def testNormalShape(self):
    mu = tf.constant([-3.0] * 5)
    sigma = tf.constant(11.0)
    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    self.assertEqual(self.evaluate(normal.batch_shape_tensor()), [5])
    self.assertEqual(normal.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(normal.event_shape_tensor()), [])
    self.assertEqual(normal.event_shape, tf.TensorShape([]))

  def testNormalShapeWithPlaceholders(self):
    mu = tf1.placeholder_with_default(np.float32(5), shape=None)
    sigma = tf1.placeholder_with_default(
        np.float32([1.0, 2.0]), shape=None)
    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    # get_batch_shape should return an '<unknown>' tensor (graph mode only).
    self.assertEqual(normal.event_shape, ())
    self.assertEqual(normal.batch_shape,
                     tf.TensorShape([2] if tf.executing_eagerly() else None))
    self.assertAllEqual(self.evaluate(normal.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(normal.batch_shape_tensor()), [2])

  def testNormalNormalKL(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    n_a = tfd.Normal(loc=mu_a, scale=sigma_a, validate_args=True)
    n_b = tfd.Normal(loc=mu_b, scale=sigma_b, validate_args=True)

    kl = tfd.kl_divergence(n_a, n_b)
    kl_val = self.evaluate(kl)

    kl_expected = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
        (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))

    x = n_a.sample(int(1e5), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(n_a.log_prob(x) - n_b.log_prob(x), axis=0)
    kl_sample_ = self.evaluate(kl_sample)

    self.assertEqual(kl.shape, (batch_size,))
    self.assertAllClose(kl_val, kl_expected)
    self.assertAllClose(kl_expected, kl_sample_, atol=0.0, rtol=1e-2)

  def testVariableScale(self):
    x = tf.Variable(1.)
    d = tfd.Normal(loc=0., scale=x, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.assertIs(x, d.scale)
    self.assertEqual(0., self.evaluate(d.mean()))
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(d.mean())

  def testIncompatibleArgShapesGraph(self):
    if tf.executing_eagerly(): return
    scale = tf1.placeholder_with_default(tf.ones([4, 1]), shape=None)
    with self.assertRaisesRegexp(tf.errors.OpError, r'Incompatible shapes'):
      d = tfd.Normal(loc=tf.zeros([2, 3]), scale=scale, validate_args=True)
      self.evaluate(d.mean())

  def testIncompatibleArgShapesEager(self):
    if not tf.executing_eagerly(): return
    scale = tf1.placeholder_with_default(tf.ones([4, 1]), shape=None)
    with self.assertRaisesRegexp(
        ValueError,
        r'Arguments `loc` and `scale` must have compatible shapes.'):
      tfd.Normal(loc=tf.zeros([2, 3]), scale=scale, validate_args=True)


class NormalEagerGCTest(test_util.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNormalMeanAndMode(self):
    # Mu will be broadcast to [7, 7, 7].
    mu = [7.]
    sigma = [11., 12., 13.]

    normal = tfd.Normal(loc=mu, scale=sigma, validate_args=True)

    self.assertAllEqual((3,), normal.mean().shape)
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.mean()))

    self.assertAllEqual((3,), normal.mode().shape)
    self.assertAllEqual([7., 7, 7], self.evaluate(normal.mode()))


if __name__ == '__main__':
  tf.test.main()
