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
import math

# Dependency imports

import numpy as np
from scipy import stats as sp_stats
from scipy.special import gamma as sp_gamma
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import generalized_normal
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class _GeneralizedNormalTest(object):

  def helper_param_shapes(self, sample_shape, expected):
    param_shapes = generalized_normal.GeneralizedNormal.param_shapes(
        sample_shape)
    mu_shape = param_shapes['loc']
    sigma_shape = param_shapes['scale']
    power_shape = param_shapes['power']
    self.assertAllEqual(expected, self.evaluate(mu_shape))
    self.assertAllEqual(expected, self.evaluate(sigma_shape))
    self.assertAllEqual(expected, self.evaluate(power_shape))
    mu = tf.zeros(mu_shape)
    sigma = tf.ones(sigma_shape)
    power = tf.ones(power_shape)
    samples = generalized_normal.GeneralizedNormal(
        mu, sigma, power, validate_args=True).sample(seed=test_util.test_seed())
    self.assertAllEqual(expected, self.evaluate(tf.shape(samples)))

  def helper_param_static_shapes(self, sample_shape, expected):
    param_shapes = generalized_normal.GeneralizedNormal.param_static_shapes(
        sample_shape)
    mu_shape = param_shapes['loc']
    sigma_shape = param_shapes['scale']
    power_shape = param_shapes['power']
    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)
    self.assertEqual(expected, power_shape)

  def testSampleLikeArgsGetDistDType(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as fp32.
      dist = generalized_normal.GeneralizedNormal(0., 1., 2.)
    elif self.dtype is np.float64:
      # The make_input function will cast them to self.dtype
      dist = generalized_normal.GeneralizedNormal(
          self.make_input(0.), self.make_input(1.), self.make_input(2.))
    self.assertEqual(self.dtype, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf'):
      self.assertEqual(self.dtype, getattr(dist, method)(1).dtype)
    for method in ('entropy', 'mean', 'variance'):
      self.assertEqual(self.dtype, getattr(dist, method)().dtype)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self.helper_param_shapes(sample_shape, sample_shape)
    self.helper_param_shapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self.helper_param_static_shapes(sample_shape, sample_shape)
    self.helper_param_static_shapes(
        tf.TensorShape(sample_shape), sample_shape)

  def testGeneralizedNormalLogPDF(self):
    batch_size = 6
    mu = tf.constant([3.] * batch_size, dtype=self.dtype)
    sigma = tf.constant([math.sqrt(10.)] * batch_size, dtype=self.dtype)
    power = tf.constant([4.] * batch_size, dtype=self.dtype)
    x = np.array([-2.5, 2.5, 4., 0., -1., 2.], dtype=np.float32)
    gnormal = generalized_normal.GeneralizedNormal(
        loc=mu, scale=sigma, power=power, validate_args=True)
    log_pdf = gnormal.log_prob(x)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(gnormal.batch_shape, log_pdf.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(log_pdf).shape)

    pdf = gnormal.prob(x)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), pdf.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(pdf).shape)
    self.assertAllEqual(gnormal.batch_shape, pdf.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(pdf).shape)

    expected_log_pdf = sp_stats.gennorm(self.evaluate(power),
                                        loc=self.evaluate(mu),
                                        scale=self.evaluate(sigma)).logpdf(x)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testGeneralizedNormalCDF(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.
    power = self._rng.rand(batch_size) + 1.
    x = np.linspace(-8., 8., batch_size).astype(np.float64)

    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(mu),
        scale=self.make_input(sigma),
        power=self.make_input(power),
        validate_args=True)
    cdf = gnormal.cdf(x)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(gnormal.batch_shape, cdf.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(cdf).shape)
    expected_cdf = sp_stats.gennorm(power, loc=mu, scale=sigma).cdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0, rtol=1e-5)

  def testGeneralizedNormalSF(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.
    power = self._rng.rand(batch_size) + 1.
    x = np.linspace(-8., 8., batch_size).astype(np.float64)

    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(mu),
        scale=self.make_input(sigma),
        power=self.make_input(power),
        validate_args=True)
    sf = gnormal.survival_function(x)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), sf.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(gnormal.batch_shape, sf.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(sf).shape)
    expected_sf = sp_stats.gennorm(power, loc=mu, scale=sigma).sf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0, rtol=1e-5)

  def testGeneralizedNormalLogCDF(self):
    if self.dtype is np.float32:
      self.skipTest('32-bit precision not sufficient for LogCDF')
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.
    power = self._rng.rand(batch_size) + 1.
    x = np.linspace(-100., 10., batch_size).astype(np.float64)

    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(mu),
        scale=self.make_input(sigma),
        power=self.make_input(power),
        validate_args=True)

    cdf = gnormal.log_cdf(x)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), cdf.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(cdf).shape)
    self.assertAllEqual(gnormal.batch_shape, cdf.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(cdf).shape)
    expected_cdf = sp_stats.gennorm(power, loc=mu, scale=sigma).logcdf(x)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0, rtol=1e-3)

  def testGeneralizedNormalQuantile(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.
    power = self._rng.rand(batch_size) + 1.
    p = np.linspace(0., 1., batch_size).astype(np.float64)
    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(mu),
        scale=self.make_input(sigma),
        power=self.make_input(power),
        validate_args=True)
    quantile = gnormal.quantile(p)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), quantile.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(quantile).shape)
    self.assertAllEqual(gnormal.batch_shape, quantile.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(quantile).shape)
    expected_quantile = sp_stats.gennorm(power, loc=mu, scale=sigma).ppf(p)
    self.assertAllClose(
        expected_quantile, self.evaluate(quantile), atol=0, rtol=1e-4)

  @test_util.numpy_disable_gradient_test
  def testFiniteGradientAtDifficultPoints(self):
    def make_fn(dtype, attr):
      x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
      return lambda m, s, p: getattr(  # pylint: disable=g-long-lambda
          generalized_normal.GeneralizedNormal(
              loc=m, scale=s, power=p, validate_args=True), attr)(
                  x)

    # TODO(b/157524947): add 'log_cdf', currently fails at -100, -20, in fp32.
    for attr in ['log_prob', 'prob', 'cdf', 'survival_function']:
      value, grads = self.evaluate(
          gradient.value_and_gradient(
              make_fn(self.dtype, attr),
              [
                  tf.constant(0, self.dtype),  # mu
                  tf.constant(1, self.dtype),  # scale
                  tf.constant(2.1, self.dtype)
              ]))  # power
      self.assertAllFinite(value)
      self.assertAllFinite(grads[0])  # d/d mu
      self.assertAllFinite(grads[1])  # d/d scale
      self.assertAllFinite(grads[2])  # d/d power

  def testGeneralizedNormalEntropyWithScalarInputs(self):
    loc_v = 2.34
    scale_v = 4.56
    power_v = 7.89

    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(loc_v),
        scale=self.make_input(scale_v),
        power=self.make_input(power_v),
        validate_args=True)
    entropy = gnormal.entropy()
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), entropy.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(entropy).shape)
    self.assertAllEqual(gnormal.batch_shape, entropy.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(entropy).shape)
    expected_entropy = sp_stats.gennorm(
        power_v, loc=loc_v, scale=scale_v).entropy()
    self.assertAllClose(expected_entropy, self.evaluate(entropy))

  def testGeneralizedNormalEntropy(self):
    loc_v = np.array([1., 1., 1.])
    scale_v = np.array([[1., 2., 3.]]).T
    power_v = np.array([1.])
    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(loc_v),
        scale=self.make_input(scale_v),
        power=self.make_input(power_v),
        validate_args=True)

    # scipy.sp_stats.norm cannot deal with these shapes.
    scale_broadcast = loc_v * scale_v
    expected_entropy = 1. / power_v - np.log(power_v/(
        2. * scale_broadcast * sp_gamma(1. / power_v)))
    entropy = gnormal.entropy()
    np.testing.assert_allclose(expected_entropy, self.evaluate(entropy))
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()), entropy.shape)
    self.assertAllEqual(
        self.evaluate(gnormal.batch_shape_tensor()),
        self.evaluate(entropy).shape)
    self.assertAllEqual(gnormal.batch_shape, entropy.shape)
    self.assertAllEqual(gnormal.batch_shape, self.evaluate(entropy).shape)

  def testGeneralizedNormalMeanAndMode(self):
    # loc will be broadcast to [[7, 7, 7], [7, 7, 7], [7, 7, 7]].
    loc = [7.]
    scale = [11., 12., 13.]
    power = [[1.], [2.], [3.]]

    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(loc),
        scale=self.make_input(scale),
        power=self.make_input(power),
        validate_args=True)

    self.assertAllEqual((3, 3), gnormal.mean().shape)
    self.assertAllEqual([[7, 7, 7.],
                         [7, 7., 7.],
                         [7., 7., 7.]], self.evaluate(gnormal.mean()))

    self.assertAllEqual((3, 3), gnormal.mode().shape)
    self.assertAllEqual([[7, 7, 7.],
                         [7, 7., 7],
                         [7., 7, 7]], self.evaluate(gnormal.mode()))

  def testGeneralizedNormalVariance(self):
    # scale will be broadcast to [[7, 7, 7], [7, 7, 7], [7, 7, 7]].
    loc = np.array([[1., 2., 3.]]).T
    scale = [7.]
    power = np.array([.5, 1., 3.])

    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(loc),
        scale=self.make_input(scale),
        power=self.make_input(power),
        validate_args=True)

    self.assertAllEqual((3, 3), gnormal.variance().shape)
    scale_broadcast = scale * np.ones_like(loc)
    reference = np.square(scale_broadcast) * (sp_gamma(3. / power) /
                                              sp_gamma(1. / power))
    self.assertAllClose(reference, self.evaluate(gnormal.variance()),
                        atol=0, rtol=1e-5)  # relaxed tol for fp32 in JAX

  def testNormalStandardDeviation(self):
    # scale will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    scale = [7.]
    power = [1.]

    gnormal = generalized_normal.GeneralizedNormal(
        loc=self.make_input(loc),
        scale=self.make_input(scale),
        power=self.make_input(power),
        validate_args=True)

    self.assertAllEqual((3,), gnormal.stddev().shape)
    reference = 7. * np.sqrt(2.) * np.ones_like(loc)
    self.assertAllClose(reference, self.evaluate(gnormal.stddev()))

  def testGeneralizedNormalSample(self):
    loc = tf.constant(3., self.dtype)
    scale = tf.constant(math.sqrt(3.), self.dtype)
    power = tf.constant(3., self.dtype)
    loc_v = 3.
    scale_v = np.sqrt(3. / sp_gamma(1. / 3.))
    n = tf.constant(100000)
    gnormal = generalized_normal.GeneralizedNormal(
        loc=loc, scale=scale, power=power, validate_args=True)
    samples = gnormal.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
    # The sample variance similarly is dependent on sigma and n.
    # Thus, the tolerances below are very sensitive to number of samples
    # as well as the variances chosen.
    self.assertEqual(sample_values.shape, (100000,))
    self.assertAllClose(sample_values.mean(), loc_v, atol=1e-1)
    self.assertAllClose(sample_values.std(), scale_v, atol=1e-1)

    expected_samples_shape = tf.TensorShape(
        [self.evaluate(n)]).concatenate(
            tf.TensorShape(
                self.evaluate(gnormal.batch_shape_tensor())))

    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

    expected_samples_shape = (
        tf.TensorShape([self.evaluate(n)]).concatenate(
            gnormal.batch_shape))

    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  @test_util.jax_disable_test_missing_functionality(
      '`jax.raw_ops` has no attribute `RandomGammaGrad`')
  @test_util.numpy_disable_gradient_test
  def testGeneralizedNormalFullyReparameterized(self):
    loc = tf.constant(4., self.dtype)
    scale = tf.constant(3., self.dtype)
    power = tf.constant(5., self.dtype)

    def sample_fn(m, s, p):
      gnormal = generalized_normal.GeneralizedNormal(
          loc=m, scale=s, power=p, validate_args=True)
      return gnormal.sample(100, seed=test_util.test_seed())

    _, [grad_loc, grad_scale,
        grad_power] = gradient.value_and_gradient(sample_fn,
                                                  [loc, scale, power])
    grad_loc, grad_scale, grad_power = self.evaluate([grad_loc, grad_scale,
                                                      grad_power])
    self.assertIsNotNone(grad_loc)
    self.assertIsNotNone(grad_scale)
    self.assertIsNotNone(grad_power)

  def testNegativeSigmaPowerFails(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      normal = generalized_normal.GeneralizedNormal(
          loc=[1.], scale=[-5.], power=[1.], validate_args=True, name='G')
      self.evaluate(normal.mean())
    with self.assertRaisesOpError('Argument `power` must be positive.'):
      normal = generalized_normal.GeneralizedNormal(
          self.make_input(1.),
          self.make_input(5.),
          self.make_input(-1.),
          validate_args=True)
      self.evaluate(normal.mean())

  def testGeneralizedNormalShape(self):
    mu = tf.constant([-3.] * 5, self.dtype)
    sigma = tf.constant(11., self.dtype)
    power = tf.constant(1., self.dtype)
    normal = generalized_normal.GeneralizedNormal(
        loc=mu, scale=sigma, power=power, validate_args=True)

    self.assertEqual(self.evaluate(normal.batch_shape_tensor()), [5])
    self.assertEqual(normal.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(normal.event_shape_tensor()), [])
    self.assertEqual(normal.event_shape, tf.TensorShape([]))

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_test_missing_functionality(
      'NumpyVariable does not handle unknown shapes')
  def testGeneralizedNormalShapeWithPlaceholders(self):
    mu = tf.Variable(np.float32(5), shape=tf.TensorShape(None))
    scale = tf.Variable(np.float32([1., 2.]), shape=tf.TensorShape(None))
    power = tf.Variable(np.float32([[1., 2.]]).T, shape=tf.TensorShape(None))
    self.evaluate([mu.initializer, scale.initializer, power.initializer])
    gnormal = generalized_normal.GeneralizedNormal(
        loc=mu, scale=scale, power=power, validate_args=True)

    # get_batch_shape should return an '<unknown>' tensor (graph mode only).
    self.assertEqual(gnormal.event_shape, ())
    self.assertEqual(gnormal.batch_shape, tf.TensorShape(None))
    self.assertAllEqual(self.evaluate(gnormal.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(gnormal.batch_shape_tensor()), [2, 2])

  def testVariableScale(self):
    x = tf.Variable(1., dtype=self.dtype)
    d = generalized_normal.GeneralizedNormal(
        loc=self.make_input(0.),
        scale=x,
        power=self.make_input(3.),
        validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.assertIs(x, d.scale)
    self.assertEqual(0., self.evaluate(d.mean()))
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(d.mean())

  def testIncompatibleArgShapesGraph(self):
    power = tf.Variable(tf.ones([2, 3], dtype=self.dtype),
                        shape=tf.TensorShape(None), name='pow')
    self.evaluate(power.initializer)
    with self.assertRaisesRegex(Exception, r'compatible shapes'):
      d = generalized_normal.GeneralizedNormal(
          loc=tf.zeros([4, 1], dtype=self.dtype),
          scale=tf.ones([4, 1], dtype=self.dtype),
          power=power,
          validate_args=True)
      self.evaluate(d.mean())


@test_util.test_all_tf_execution_regimes
class GeneralizedNormalTestStaticShapeFloat32(test_util.TestCase,
                                              _GeneralizedNormalTest):
  dtype = np.float32
  use_static_shape = True

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(GeneralizedNormalTestStaticShapeFloat32, self).setUp()


@test_util.test_all_tf_execution_regimes
class GeneralizedNormalTestDynamicShapeFloat32(test_util.TestCase,
                                               _GeneralizedNormalTest):
  dtype = np.float32
  use_static_shape = False

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(GeneralizedNormalTestDynamicShapeFloat32, self).setUp()


@test_util.test_all_tf_execution_regimes
class GeneralizedNormalTestStaticShapeFloat64(test_util.TestCase,
                                              _GeneralizedNormalTest):
  dtype = np.float64
  use_static_shape = True

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(GeneralizedNormalTestStaticShapeFloat64, self).setUp()


@test_util.test_all_tf_execution_regimes
class GeneralizedNormalTestDynamicShapeFloat64(test_util.TestCase,
                                               _GeneralizedNormalTest):
  dtype = np.float64
  use_static_shape = False

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(GeneralizedNormalTestDynamicShapeFloat64, self).setUp()


if __name__ == '__main__':
  test_util.main()
