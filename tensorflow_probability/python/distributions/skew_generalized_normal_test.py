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
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import skew_generalized_normal
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient

from tensorflow.python.framework import test_util as tf_test_util  # pylint: disable=g-direct-tensorflow-import


# Short alias for the distribution under test.
SGN = skew_generalized_normal.SkewGeneralizedNormal


# NumPy reference implementations of the skew generalized normal (generalized
# normal v2). No reference distribution exists in scipy (`scipy.stats.gennorm`
# is the *symmetric* v1), so we validate directly against the closed-form
# formulas, against `Normal(loc, scale)` in the `peak -> 0` limit, and against
# numerical moments.
def _np_y(x, loc, scale, peak):
  return -np.log1p(-peak * (x - loc) / scale) / peak


def _np_log_prob(x, loc, scale, peak):
  y = _np_y(x, loc, scale, peak)
  return (-0.5 * y**2 - 0.5 * np.log(2. * np.pi)
          - np.log(scale - peak * (x - loc)))


def _np_mean(loc, scale, peak):
  return loc - scale * np.expm1(0.5 * peak**2) / peak


def _np_stddev(scale, peak):
  return (scale / np.abs(peak)) * np.exp(0.5 * peak**2) * np.sqrt(
      np.expm1(peak**2))


def _np_mode(loc, scale, peak):
  return loc - scale * np.expm1(-peak**2) / peak


def _np_entropy(scale):
  return np.log(scale) + 0.5 * (1. + np.log(2. * np.pi))


@test_util.test_all_tf_execution_regimes
class _SkewGeneralizedNormalTest(object):

  def testSampleLikeArgsGetDistDType(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as fp32.
      dist = SGN(0., 1., 2.)
    else:
      dist = SGN(self.make_input(0.), self.make_input(1.), self.make_input(2.))
    self.assertEqual(self.dtype, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf'):
      self.assertEqual(self.dtype, getattr(dist, method)(1).dtype)
    for method in ('entropy', 'mean', 'variance'):
      self.assertEqual(self.dtype, getattr(dist, method)().dtype)

  def testSkewGeneralizedNormalLogPDF(self):
    batch_size = 6
    mu = tf.constant([3.] * batch_size, dtype=self.dtype)
    sigma = tf.constant([math.sqrt(10.)] * batch_size, dtype=self.dtype)
    peak = tf.constant([1.] * batch_size, dtype=self.dtype)
    # All points lie inside the support `x < loc + scale / peak ~= 6.16`.
    x = np.array([-2.5, 2.5, 4., 0., -1., 2.], dtype=self.dtype)
    sgn = SGN(loc=mu, scale=sigma, peak=peak, validate_args=True)
    log_pdf = sgn.log_prob(x)
    self.assertAllEqual(
        self.evaluate(sgn.batch_shape_tensor()), log_pdf.shape)
    self.assertAllEqual(
        self.evaluate(sgn.batch_shape_tensor()),
        self.evaluate(log_pdf).shape)
    self.assertAllEqual(sgn.batch_shape, log_pdf.shape)
    self.assertAllEqual(sgn.batch_shape, self.evaluate(log_pdf).shape)

    pdf = sgn.prob(x)
    self.assertAllEqual(sgn.batch_shape, pdf.shape)

    expected_log_pdf = _np_log_prob(x, 3., math.sqrt(10.), 1.)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testSkewGeneralizedNormalProbIsZeroOutsideSupport(self):
    # peak > 0 => support is x < loc + scale / peak = 2.0.
    sgn = SGN(loc=tf.constant(0., self.dtype),
              scale=tf.constant(2., self.dtype),
              peak=tf.constant(1., self.dtype),
              validate_args=False)
    x = np.array([5., 10., 100.], dtype=self.dtype)  # all > edge (2.0)
    self.assertAllEqual(np.zeros([3]), self.evaluate(sgn.prob(x)))

  def testSkewGeneralizedNormalCDF(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.
    peak = self._rng.rand(batch_size) + 0.5
    # Generate in-support points by pushing a standard-normal grid through the
    # quantile transform: x = loc + scale * (1 - exp(-peak * z)) / peak.
    z = np.linspace(-3., 3., batch_size)
    x = (mu + sigma * (1. - np.exp(-peak * z)) / peak).astype(np.float64)

    sgn = SGN(loc=self.make_input(mu),
              scale=self.make_input(sigma),
              peak=self.make_input(peak),
              validate_args=True)
    cdf = sgn.cdf(x)
    self.assertAllEqual(sgn.batch_shape, cdf.shape)
    expected_cdf = sp_stats.norm.cdf(_np_y(x, mu, sigma, peak))
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=0, rtol=1e-5)

  def testSkewGeneralizedNormalLogCDF(self):
    if self.dtype is np.float32:
      self.skipTest('32-bit precision not sufficient for LogCDF')
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.
    peak = self._rng.rand(batch_size) + 0.5
    z = np.linspace(-10., 3., batch_size)
    x = (mu + sigma * (1. - np.exp(-peak * z)) / peak).astype(np.float64)

    sgn = SGN(loc=self.make_input(mu),
              scale=self.make_input(sigma),
              peak=self.make_input(peak),
              validate_args=True)
    log_cdf = sgn.log_cdf(x)
    self.assertAllEqual(sgn.batch_shape, log_cdf.shape)
    expected_log_cdf = sp_stats.norm.logcdf(_np_y(x, mu, sigma, peak))
    self.assertAllClose(expected_log_cdf, self.evaluate(log_cdf),
                        atol=0, rtol=1e-3)

  @test_util.numpy_disable_gradient_test
  def testFiniteGradientAtDifficultPoints(self):
    def make_fn(dtype, attr):
      # peak = 2.1 => support is x < loc + scale / peak ~= 0.476. Probe the
      # long negative tail and points right up against the boundary.
      x = np.array([-100., -20., -5., 0., 0.4, 0.47]).astype(dtype)
      return lambda m, s, p: getattr(  # pylint: disable=g-long-lambda
          SGN(loc=m, scale=s, peak=p, validate_args=True), attr)(x)

    for attr in ['log_prob', 'prob', 'cdf']:
      value, grads = self.evaluate(gradient.value_and_gradient(
          make_fn(self.dtype, attr),
          [tf.constant(0, self.dtype),  # loc
           tf.constant(1, self.dtype),  # scale
           tf.constant(2.1, self.dtype)]))  # peak
      self.assertAllFinite(value)
      self.assertAllFinite(grads[0])  # d/d loc
      self.assertAllFinite(grads[1])  # d/d scale
      self.assertAllFinite(grads[2])  # d/d peak

  def testSkewGeneralizedNormalEntropyWithScalarInputs(self):
    loc_v = 2.34
    scale_v = 4.56
    peak_v = 7.89

    sgn = SGN(loc=self.make_input(loc_v),
              scale=self.make_input(scale_v),
              peak=self.make_input(peak_v),
              validate_args=True)
    entropy = sgn.entropy()
    self.assertAllEqual(sgn.batch_shape, entropy.shape)
    # Entropy depends only on `scale`.
    self.assertAllClose(_np_entropy(scale_v), self.evaluate(entropy))

  def testSkewGeneralizedNormalEntropy(self):
    loc_v = np.array([1., 1., 1.])
    scale_v = np.array([[1., 2., 3.]]).T
    peak_v = np.array([2.])
    sgn = SGN(loc=self.make_input(loc_v),
              scale=self.make_input(scale_v),
              peak=self.make_input(peak_v),
              validate_args=True)
    expected_entropy = _np_entropy(scale_v) * np.ones_like(loc_v)
    entropy = sgn.entropy()
    self.assertAllClose(expected_entropy, self.evaluate(entropy))
    self.assertAllEqual(sgn.batch_shape, entropy.shape)

  def testSkewGeneralizedNormalMeanAndMode(self):
    loc = np.array([7.], dtype=np.float64)
    scale = np.array([11., 12., 13.], dtype=np.float64)
    peak = np.array([[1.], [2.], [3.]], dtype=np.float64)

    sgn = SGN(loc=self.make_input(loc),
              scale=self.make_input(scale),
              peak=self.make_input(peak),
              validate_args=True)

    self.assertAllEqual((3, 3), sgn.mean().shape)
    self.assertAllClose(_np_mean(loc, scale, peak), self.evaluate(sgn.mean()))

    self.assertAllEqual((3, 3), sgn.mode().shape)
    self.assertAllClose(_np_mode(loc, scale, peak), self.evaluate(sgn.mode()))

  def testSkewGeneralizedNormalVariance(self):
    loc = np.array([[1., 2., 3.]]).T
    scale = np.array([7.], dtype=np.float64)
    peak = np.array([.5, 1., 3.], dtype=np.float64)

    sgn = SGN(loc=self.make_input(loc),
              scale=self.make_input(scale),
              peak=self.make_input(peak),
              validate_args=True)

    self.assertAllEqual((3, 3), sgn.variance().shape)
    reference = (_np_stddev(scale, peak)**2) * np.ones_like(loc)
    self.assertAllClose(reference, self.evaluate(sgn.variance()),
                        atol=0, rtol=1e-5)

  def testSkewGeneralizedNormalStandardDeviation(self):
    loc = np.array([1., 2., 3.], dtype=np.float64)
    scale = np.array([7.], dtype=np.float64)
    peak = np.array([1.5], dtype=np.float64)

    sgn = SGN(loc=self.make_input(loc),
              scale=self.make_input(scale),
              peak=self.make_input(peak),
              validate_args=True)

    self.assertAllEqual((3,), sgn.stddev().shape)
    reference = _np_stddev(scale, peak) * np.ones_like(loc)
    self.assertAllClose(reference, self.evaluate(sgn.stddev()))

  def testSkewGeneralizedNormalSample(self):
    loc = tf.constant(3., self.dtype)
    scale = tf.constant(math.sqrt(3.), self.dtype)
    peak = tf.constant(0.5, self.dtype)  # mild skew, light tails for MC checks.
    expected_mean = _np_mean(3., math.sqrt(3.), 0.5)
    expected_std = _np_stddev(math.sqrt(3.), 0.5)
    n = tf.constant(100000)
    sgn = SGN(loc=loc, scale=scale, peak=peak, validate_args=True)
    samples = sgn.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000,))
    self.assertAllClose(sample_values.mean(), expected_mean, atol=1e-1)
    self.assertAllClose(sample_values.std(), expected_std, atol=2e-1)

    expected_samples_shape = tf.TensorShape(
        [self.evaluate(n)]).concatenate(
            tf.TensorShape(
                self.evaluate(sgn.batch_shape_tensor())))
    self.assertAllEqual(expected_samples_shape, samples.shape)
    self.assertAllEqual(expected_samples_shape, sample_values.shape)

  @test_util.numpy_disable_gradient_test
  def testSkewGeneralizedNormalFullyReparameterized(self):
    loc = tf.constant(4., self.dtype)
    scale = tf.constant(3., self.dtype)
    peak = tf.constant(1.5, self.dtype)

    def sample_fn(m, s, p):
      sgn = SGN(loc=m, scale=s, peak=p, validate_args=True)
      return sgn.sample(100, seed=test_util.test_seed())

    _, [grad_loc, grad_scale, grad_peak] = gradient.value_and_gradient(
        sample_fn, [loc, scale, peak])
    grad_loc, grad_scale, grad_peak = self.evaluate([grad_loc, grad_scale,
                                                     grad_peak])
    self.assertIsNotNone(grad_loc)
    self.assertIsNotNone(grad_scale)
    self.assertIsNotNone(grad_peak)

  def testNegativePeakIsValid(self):
    # peak < 0 mirrors the distribution: support is x > loc + scale / peak,
    # i.e. x > -0.5 for the parameters below.
    loc, scale, peak = 0., 1., -2.
    sgn = SGN(loc=tf.constant(loc, self.dtype),
              scale=tf.constant(scale, self.dtype),
              peak=tf.constant(peak, self.dtype),
              validate_args=True)
    # In-support points (x > -0.5).
    x_in = np.array([-0.4, 0., 1., 5.], dtype=self.dtype)
    log_prob = self.evaluate(sgn.log_prob(x_in))
    self.assertAllFinite(log_prob)
    self.assertAllClose(_np_log_prob(x_in, loc, scale, peak), log_prob)
    # Out-of-support points (x < -0.5) => prob 0.
    x_out = np.array([-1., -5., -100.], dtype=self.dtype)
    self.assertAllEqual(np.zeros([3]), self.evaluate(sgn.prob(x_out)))
    # CDF is monotone increasing over the support.
    x_sorted = np.array([-0.49, -0.2, 0.5, 2., 20.], dtype=self.dtype)
    cdf = self.evaluate(sgn.cdf(x_sorted))
    self.assertAllEqual(np.ones([4], dtype=bool), np.diff(cdf) >= 0.)

  def testNegativeScaleFails(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      sgn = SGN(loc=[1.], scale=[-5.], peak=[1.],
                validate_args=True, name='G')
      self.evaluate(sgn.mean())

  def testZeroPeakFails(self):
    with self.assertRaisesOpError('Argument `peak` must be nonzero.'):
      sgn = SGN(self.make_input(1.),
                self.make_input(5.),
                self.make_input(0.),
                validate_args=True)
      self.evaluate(sgn.mean())

  def testSkewGeneralizedNormalShape(self):
    mu = tf.constant([-3.] * 5, self.dtype)
    sigma = tf.constant(11., self.dtype)
    peak = tf.constant(1., self.dtype)
    sgn = SGN(loc=mu, scale=sigma, peak=peak, validate_args=True)

    self.assertEqual(self.evaluate(sgn.batch_shape_tensor()), [5])
    self.assertEqual(sgn.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(sgn.event_shape_tensor()), [])
    self.assertEqual(sgn.event_shape, tf.TensorShape([]))

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_test_missing_functionality(
      'NumpyVariable does not handle unknown shapes')
  def testSkewGeneralizedNormalShapeWithPlaceholders(self):
    mu = tf.Variable(np.float32(5), shape=tf.TensorShape(None))
    scale = tf.Variable(np.float32([1., 2.]), shape=tf.TensorShape(None))
    peak = tf.Variable(np.float32([[1., 2.]]).T, shape=tf.TensorShape(None))
    self.evaluate([mu.initializer, scale.initializer, peak.initializer])
    sgn = SGN(loc=mu, scale=scale, peak=peak, validate_args=True)

    self.assertEqual(sgn.event_shape, ())
    self.assertEqual(sgn.batch_shape, tf.TensorShape(None))
    self.assertAllEqual(self.evaluate(sgn.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(sgn.batch_shape_tensor()), [2, 2])

  def testVariableScale(self):
    x = tf.Variable(1., dtype=self.dtype)
    d = SGN(loc=self.make_input(0.),
            scale=x,
            peak=self.make_input(3.),
            validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.assertIs(x, d.scale)
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(d.mean())

  def testIncompatibleArgShapesGraph(self):
    peak = tf.Variable(tf.ones([2, 3], dtype=self.dtype),
                       shape=tf.TensorShape(None), name='peak')
    self.evaluate(peak.initializer)
    with self.assertRaisesRegexp(Exception, r'compatible shapes'):
      d = SGN(loc=tf.zeros([4, 1], dtype=self.dtype),
              scale=tf.ones([4, 1], dtype=self.dtype),
              peak=peak, validate_args=True)
      self.evaluate(d.mean())


class SkewGeneralizedNormalEagerGCTest(test_util.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testSkewGeneralizedNormalMeanAndMode(self):
    loc = np.array([7.], dtype=np.float64)
    scale = np.array([11., 12., 13.], dtype=np.float64)
    peak = np.array([1., 2., 3.], dtype=np.float64)

    sgn = SGN(loc=loc, scale=scale, peak=peak, validate_args=True)

    self.assertAllEqual((3,), sgn.mean().shape)
    self.assertAllClose(_np_mean(loc, scale, peak), self.evaluate(sgn.mean()))

    self.assertAllEqual((3,), sgn.mode().shape)
    self.assertAllClose(_np_mode(loc, scale, peak), self.evaluate(sgn.mode()))


@test_util.test_all_tf_execution_regimes
class SkewGeneralizedNormalTestStaticShapeFloat32(test_util.TestCase,
                                                  _SkewGeneralizedNormalTest):
  dtype = np.float32
  use_static_shape = True

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(SkewGeneralizedNormalTestStaticShapeFloat32, self).setUp()


@test_util.test_all_tf_execution_regimes
class SkewGeneralizedNormalTestDynamicShapeFloat32(test_util.TestCase,
                                                   _SkewGeneralizedNormalTest):
  dtype = np.float32
  use_static_shape = False

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(SkewGeneralizedNormalTestDynamicShapeFloat32, self).setUp()


@test_util.test_all_tf_execution_regimes
class SkewGeneralizedNormalTestStaticShapeFloat64(test_util.TestCase,
                                                  _SkewGeneralizedNormalTest):
  dtype = np.float64
  use_static_shape = True

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(SkewGeneralizedNormalTestStaticShapeFloat64, self).setUp()


@test_util.test_all_tf_execution_regimes
class SkewGeneralizedNormalTestDynamicShapeFloat64(test_util.TestCase,
                                                   _SkewGeneralizedNormalTest):
  dtype = np.float64
  use_static_shape = False

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(SkewGeneralizedNormalTestDynamicShapeFloat64, self).setUp()


if __name__ == '__main__':
  tf.test.main()
