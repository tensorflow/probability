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


@test_util.test_all_tf_execution_regimes
class JohnsonSUTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(JohnsonSUTest, self).setUp()

  def _testParamShapes(self, sample_shape, expected):
    param_shapes = tfd.JohnsonSU.param_shapes(sample_shape)
    skewness_shape, tailweight_shape, mu_shape, sigma_shape = \
      param_shapes['skewness'], param_shapes['tailweight'], \
      param_shapes['loc'], param_shapes['scale']
    self.assertAllEqual(expected, self.evaluate(skewness_shape))
    self.assertAllEqual(expected, self.evaluate(tailweight_shape))
    self.assertAllEqual(expected, self.evaluate(mu_shape))
    self.assertAllEqual(expected, self.evaluate(sigma_shape))
    skewness = tf.zeros(skewness_shape)
    tailweight = tf.ones(tailweight_shape)
    mu = tf.zeros(mu_shape)
    sigma = tf.ones(sigma_shape)
    self.assertAllEqual(
        expected,
        self.evaluate(
            tf.shape(tfd.JohnsonSU(skewness, tailweight, mu, sigma,
                                   validate_args=True)
                     .sample(seed=test_util.test_seed()))))

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = tfd.JohnsonSU.param_static_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes['loc'], param_shapes['scale']
    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)

  def testSampleLikeArgsGetDistDType(self):
    dist = tfd.JohnsonSU(1., 2., 0., 1.)
    self.assertEqual(tf.float32, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf',
                   'log_survival_function', 'survival_function', 'quantile'):
      self.assertEqual(tf.float32, getattr(dist, method)(1).dtype, method)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(
        tf.TensorShape(sample_shape), sample_shape)

  def testJohnsonSULogPDF(self):
    batch_size = 6
    skewness = tf.constant([1.0] * batch_size)
    tailweight = tf.constant([2.0] * batch_size)
    mu = tf.constant([3.0] * batch_size)
    sigma = tf.constant([math.sqrt(10.0)] * batch_size)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    log_pdf = johnson_su.log_prob(x)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(johnson_su.batch_shape, log_pdf.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(log_pdf).shape)

    pdf = johnson_su.prob(x)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(pdf).shape)
    self.assertAllEqual(johnson_su.batch_shape, pdf.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(pdf).shape)

    expected_log_pdf = sp_stats.johnsonsu(self.evaluate(skewness),
                                          self.evaluate(tailweight),
                                          self.evaluate(mu),
                                          self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testJohnsonSULogPDFMultidimensional(self):
    batch_size = 6
    skewness = tf.constant([[1.0, -1.0]] * batch_size)
    tailweight = tf.constant([[1.0, 2.0]] * batch_size)
    mu = tf.constant([[3.0, -3.0]] * batch_size)
    sigma = tf.constant(
        [[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    log_pdf = johnson_su.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(johnson_su.batch_shape, log_pdf.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(log_pdf).shape)

    pdf = johnson_su.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), pdf_values.shape)
    self.assertAllEqual(johnson_su.batch_shape, pdf.shape)
    self.assertAllEqual(johnson_su.batch_shape, pdf_values.shape)

    expected_log_pdf = sp_stats.johnsonsu.logpdf(x,
                                                 self.evaluate(skewness),
                                                 self.evaluate(tailweight),
                                                 self.evaluate(mu),
                                                 self.evaluate(sigma))
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testJohnsonSUCDF(self):
    batch_size = 50
    skewness = self._rng.randn(batch_size)
    tailweight = self._rng.rand(batch_size) + 1.0
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)
    cdf = johnson_su.cdf(x)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(johnson_su.batch_shape, cdf.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(cdf).shape)
    expected_cdf = sp_stats.johnsonsu.cdf(x, skewness, tailweight, mu, sigma)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0)

  def testJohnsonSUSurvivalFunction(self):
    batch_size = 50
    skewness = self._rng.randn(batch_size)
    tailweight = self._rng.rand(batch_size) + 1.0
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    sf = johnson_su.survival_function(x)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), sf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(johnson_su.batch_shape, sf.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(sf).shape)
    expected_sf = sp_stats.johnsonsu.sf(x, skewness, tailweight, mu, sigma)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0)

  def testJohnsonSULogCDF(self):
    batch_size = 50
    skewness = self._rng.randn(batch_size)
    tailweight = self._rng.rand(batch_size) + 1.0
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-100.0, 10.0, batch_size).astype(np.float64)

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    cdf = johnson_su.log_cdf(x)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(johnson_su.batch_shape, cdf.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(cdf).shape)

    expected_cdf = sp_stats.johnsonsu.logcdf(x, skewness, tailweight, mu, sigma)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0, rtol=1e-3)

  def testFiniteGradientAtDifficultPoints(self):
    def make_fn(dtype, attr):
      x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
      return lambda g, d, m, s: getattr(  # pylint: disable=g-long-lambda
          tfd.JohnsonSU(skewness=g, tailweight=d, loc=m, scale=s,
                        validate_args=True),
          attr)(x)

    for dtype in np.float32, np.float64:
      for attr in ['cdf', 'log_cdf', 'survival_function',
                   'log_survival_function', 'log_prob', 'prob']:
        value, grads = self.evaluate(tfp.math.value_and_gradient(
            make_fn(dtype, attr),
            [tf.constant(0, dtype), tf.constant(1, dtype),
             tf.constant(2, dtype), tf.constant(3, dtype)]))
        self.assertAllFinite(value)
        self.assertAllFinite(grads[0])
        self.assertAllFinite(grads[1])
        self.assertAllFinite(grads[2])
        self.assertAllFinite(grads[3])

  def testJohnsonSULogSurvivalFunction(self):
    batch_size = 50
    skewness = self._rng.randn(batch_size)
    tailweight = self._rng.rand(batch_size) + 1.0
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-10.0, 10.0, batch_size).astype(np.float64)

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    sf = johnson_su.log_survival_function(x)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), sf.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(johnson_su.batch_shape, sf.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(sf).shape)

    expected_sf = sp_stats.johnsonsu.logsf(x, skewness, tailweight, mu, sigma)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0, rtol=1e-5)

  def testJohnsonSUMean(self):
    skewness = [1.]
    tailweight = [2.]
    # Mu will be broadcast to [7, 7, 7].
    mu = [7.]
    sigma = [11., 12., 13.]

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)
    self.assertAllEqual((3,), johnson_su.mean().shape)
    # sp_stats doesn't work with array tailweight
    expected_mean = sp_stats.johnsonsu.mean(skewness, tailweight[0], mu, sigma)
    self.assertAllClose(expected_mean, self.evaluate(johnson_su.mean()))

  def testJohnsonSUQuantile(self):
    batch_size = 52
    skewness = self._rng.randn(batch_size)
    tailweight = self._rng.rand(batch_size) + 1.0
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    p = np.linspace(0., 1.0, batch_size - 2).astype(np.float64)
    # Quantile performs piecewise rational approximation so adding some
    # special input values to make sure we hit all the pieces.
    p = np.hstack((p, np.exp(-33), 1. - np.exp(-33)))

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)
    x = johnson_su.quantile(p)

    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()), x.shape)
    self.assertAllEqual(
        self.evaluate(johnson_su.batch_shape_tensor()),
        self.evaluate(x).shape)
    self.assertAllEqual(johnson_su.batch_shape, x.shape)
    self.assertAllEqual(johnson_su.batch_shape, self.evaluate(x).shape)

    expected_x = sp_stats.johnsonsu.ppf(p, skewness, tailweight, mu, sigma)
    self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  def _testQuantileFiniteGradientAtDifficultPoints(self, dtype):
    skewness = tf.constant(dtype(0))
    tailweight = tf.constant(dtype(1))
    mu = tf.constant(dtype(0))
    sigma = tf.constant(dtype(1))
    p = tf.constant(dtype([np.exp(-32.), np.exp(-2.),
                           1. - np.exp(-2.), 1. - np.exp(-8.)]))
    value, grads = tfp.math.value_and_gradient(
        lambda m, p_: tfd.JohnsonSU(skewness=skewness, tailweight=tailweight,  # pylint:disable=g-long-lambda
                                    loc=m, scale=sigma, validate_args=True).
        quantile(p_), [mu, p])
    value, grads = self.evaluate([value, grads])
    self.assertAllFinite(grads[0])
    self.assertAllFinite(grads[1])

  def testQuantileFiniteGradientAtDifficultPointsFloat32(self):
    self._testQuantileFiniteGradientAtDifficultPoints(np.float32)

  def testQuantileFiniteGradientAtDifficultPointsFloat64(self):
    self._testQuantileFiniteGradientAtDifficultPoints(np.float64)

  def testJohnsonSUVariance(self):
    skewness = [1.]
    tailweight = [2.]
    # sigma will be broadcast to [7, 7, 7]
    mu = [1., 2., 3.]
    sigma = [7.]

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    self.assertAllEqual((3,), johnson_su.variance().shape)
    expected_v = sp_stats.johnsonsu.var(skewness[0], tailweight[0], mu[0],
                                        sigma[0])
    self.assertAllClose([expected_v] * 3, self.evaluate(johnson_su.variance()))

  def testJohnsonSUStandardDeviation(self):
    skewness = [1.]
    tailweight = [2.]
    # sigma will be broadcast to [7, 7, 7]
    mu = [1., 2., 3.]
    sigma = [7.]

    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    self.assertAllEqual((3,), johnson_su.stddev().shape)
    expected_d = sp_stats.johnsonsu.std(skewness[0], tailweight[0], mu[0],
                                        sigma[0])
    self.assertAllClose([expected_d] * 3, self.evaluate(johnson_su.stddev()))

  def testJohnsonSUSample(self):
    skewness = tf.constant(1.0)
    tailweight = tf.constant(2.0)
    mu = tf.constant(3.0)
    sigma = tf.constant(math.sqrt(3.0))
    mu_v = sp_stats.johnsonsu.mean(1, 2, 3, math.sqrt(3.0))
    sigma_v = sp_stats.johnsonsu.std(1, 2, 3, math.sqrt(3.0))
    n = tf.constant(100000)
    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)
    samples = johnson_su.sample(n, seed=test_util.test_seed())
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
                self.evaluate(johnson_su.batch_shape_tensor())))

    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

    expected_samples_shape = (
        tf.TensorShape([self.evaluate(n)]).concatenate(
            johnson_su.batch_shape))

    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testJohnsonSUFullyReparameterized(self):
    skewness = tf.constant(1.0)
    tailweight = tf.constant(2.0)
    mu = tf.constant(4.0)
    sigma = tf.constant(3.0)
    _, [grad_skewness, grad_tailweight, grad_mu, grad_sigma] = (
        tfp.math.value_and_gradient(
            lambda g, d, m, s: tfd.JohnsonSU(skewness=g, tailweight=d, loc=m,  # pylint:disable=g-long-lambda
                                             scale=s, validate_args=True)
            .sample(100, seed=test_util.test_seed()),
            [skewness, tailweight, mu, sigma]))
    grad_skewness, grad_tailweight, grad_mu, grad_sigma = self.evaluate(
        [grad_skewness, grad_tailweight, grad_mu, grad_sigma])
    self.assertIsNotNone(grad_skewness)
    self.assertIsNotNone(grad_tailweight)
    self.assertIsNotNone(grad_mu)
    self.assertIsNotNone(grad_sigma)

  def testJohnsonSUSampleMultiDimensional(self):
    batch_size = 2
    skewness = tf.constant([[1.0, -1.0]] * batch_size)
    tailweight = tf.constant([[2.0, 3.0]] * batch_size)
    mu = tf.constant([[3.0, -3.0]] * batch_size)
    sigma = tf.constant(
        [[math.sqrt(2.0), math.sqrt(3.0)]] * batch_size)

    sp_stats_params = [
        (1, 2, 3, math.sqrt(2.)),
        (-1, 3, -3, math.sqrt(3.))
    ]
    mu_v = [sp_stats.johnsonsu.mean(*params) for params in sp_stats_params]
    sigma_v = [sp_stats.johnsonsu.std(*params) for params in sp_stats_params]
    n = tf.constant(100000)
    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)
    samples = johnson_su.sample(n, seed=test_util.test_seed())
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
                self.evaluate(johnson_su.batch_shape_tensor())))
    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

    expected_samples_shape = (
        tf.TensorShape([self.evaluate(n)]).concatenate(
            johnson_su.batch_shape))
    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNegativetailweightFails(self):
    with self.assertRaisesOpError('Argument `tailweight` must be positive.'):
      johnson_su = tfd.JohnsonSU(skewness=[1.], tailweight=[-1.], loc=[1.],
                                 scale=[5.], validate_args=True, name='D')
      self.evaluate(johnson_su.mean())

  def testNegativeScaleFails(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      johnson_su = tfd.JohnsonSU(skewness=[1.], tailweight=[1.], loc=[1.],
                                 scale=[-5.], validate_args=True, name='S')
      self.evaluate(johnson_su.mean())

  def testJohnsonSUShape(self):
    skewness = tf.constant(1.0)
    tailweight = tf.constant(2.0)
    mu = tf.constant([-3.0] * 5)
    sigma = tf.constant(11.0)
    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    self.assertEqual(self.evaluate(johnson_su.batch_shape_tensor()), [5])
    self.assertEqual(johnson_su.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(johnson_su.event_shape_tensor()), [])
    self.assertEqual(johnson_su.event_shape, tf.TensorShape([]))

  def testJohnsonSUShapeWithPlaceholders(self):
    skewness = tf1.placeholder_with_default(np.float32(5), shape=None)
    tailweight = tf1.placeholder_with_default(np.float32(5), shape=None)
    mu = tf1.placeholder_with_default(np.float32(5), shape=None)
    sigma = tf1.placeholder_with_default(
        np.float32([1.0, 2.0]), shape=None)
    johnson_su = tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=mu,
                               scale=sigma, validate_args=True)

    # get_batch_shape should return an '<unknown>' tensor (graph mode only).
    self.assertEqual(johnson_su.event_shape, ())
    self.assertEqual(johnson_su.batch_shape,
                     tf.TensorShape([2] if tf.executing_eagerly() else None))
    self.assertAllEqual(self.evaluate(johnson_su.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(johnson_su.batch_shape_tensor()), [2])

  def testVariableScale(self):
    x = tf.Variable(1.)
    d = tfd.JohnsonSU(skewness=0., tailweight=2., loc=0., scale=x,
                      validate_args=True)
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
      d = tfd.JohnsonSU(skewness=1., tailweight=2., loc=tf.zeros([2, 3]),
                        scale=scale, validate_args=True)
      self.evaluate(d.mean())

  def testIncompatibleArgShapesEager(self):
    if not tf.executing_eagerly(): return
    scale = tf1.placeholder_with_default(tf.ones([4, 1]), shape=None)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Incompatible shapes for broadcasting: (2, 3) and (4, 1)'):
      tfd.JohnsonSU(skewness=1., tailweight=2., loc=tf.zeros([2, 3]),
                    scale=scale, validate_args=True)


if __name__ == '__main__':
  tf.test.main()
