# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for Two-Piece Normal distribution."""

import itertools

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import two_piece_normal
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow.python.framework import test_util as tf_test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.test_all_tf_execution_regimes
class _TwoPieceNormalTest(object):

  def make_two_piece_normal(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as float32.
      dist = two_piece_normal.TwoPieceNormal(
          loc=3., scale=10., skewness=0.75, validate_args=True)
    elif self.dtype is np.float64:
      dist = two_piece_normal.TwoPieceNormal(
          loc=tf.constant(3., dtype=self.dtype),
          scale=tf.constant(10., dtype=self.dtype),
          skewness=tf.constant(0.75, dtype=self.dtype),
          validate_args=True)

    return dist

  def make_two_piece_normals(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as float32.
      dist = two_piece_normal.TwoPieceNormal(
          loc=3., scale=10., skewness=[0.75, 1., 1.33], validate_args=True)
    elif self.dtype is np.float64:
      dist = two_piece_normal.TwoPieceNormal(
          loc=tf.constant(3., dtype=self.dtype),
          scale=tf.constant(10., dtype=self.dtype),
          skewness=tf.constant([0.75, 1., 1.33], dtype=self.dtype),
          validate_args=True)

    return dist

  def helper_param_shapes(self, sample_shape, expected):
    param_shapes = two_piece_normal.TwoPieceNormal.param_shapes(sample_shape)
    mu_shape = param_shapes['loc']
    sigma_shape = param_shapes['scale']
    skewness_shape = param_shapes['skewness']

    self.assertAllEqual(expected, self.evaluate(mu_shape))
    self.assertAllEqual(expected, self.evaluate(sigma_shape))
    self.assertAllEqual(expected, self.evaluate(skewness_shape))

    mu = tf.zeros(mu_shape)
    sigma = tf.ones(sigma_shape)
    skewness = tf.ones(skewness_shape)
    seed = test_util.test_seed()
    samples = two_piece_normal.TwoPieceNormal(
        mu, sigma, skewness, validate_args=True).sample(seed=seed)

    self.assertAllEqual(expected, self.evaluate(tf.shape(samples)))

  def helper_param_static_shapes(self, sample_shape, expected):
    param_shapes = two_piece_normal.TwoPieceNormal.param_static_shapes(
        sample_shape)
    mu_shape = param_shapes['loc']
    sigma_shape = param_shapes['scale']
    skewness_shape = param_shapes['skewness']

    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)
    self.assertEqual(expected, skewness_shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self.helper_param_shapes(sample_shape, sample_shape)
    self.helper_param_shapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self.helper_param_static_shapes(sample_shape, sample_shape)
    self.helper_param_static_shapes(
        tf.TensorShape(sample_shape), sample_shape)

  def testSampleLikeArgsGetDistDType(self):
    dist = self.make_two_piece_normal()

    self.assertEqual(self.dtype, dist.dtype)

    seed = test_util.test_seed()
    self.assertEqual(self.dtype, dist.sample(1, seed=seed).dtype)

    for method in ('prob', 'cdf', 'survival_function', 'quantile',
                   'log_prob', 'log_cdf', 'log_survival_function'):
      self.assertEqual(self.dtype, getattr(dist, method)(1).dtype, msg=method)

    for method in ('mean', 'variance', 'mode'):
      self.assertEqual(self.dtype, getattr(dist, method)().dtype, msg=method)

  def testShape(self):
    for dist in (self.make_two_piece_normal(), self.make_two_piece_normals()):
      expected_batch_shape = dist.skewness.shape
      self.assertEqual(
          list(self.evaluate(dist.batch_shape_tensor())),
          list(expected_batch_shape))
      self.assertEqual(dist.batch_shape, expected_batch_shape)
      self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
      self.assertEqual(dist.event_shape, tf.TensorShape([]))

      n = 10
      sample = dist.sample(n, seed=test_util.test_seed())
      results = [sample]

      for method in ('prob', 'cdf', 'survival_function', 'log_prob', 'log_cdf',
                     'log_survival_function'):
        results.append(getattr(dist, method)(sample))

      probs = dist.cdf(sample)
      results.append(dist.quantile(probs))

      for result in results:
        self.assertAllEqual(
            [n] + list(self.evaluate(dist.batch_shape_tensor())), result.shape)
        self.assertAllEqual([n] +
                            list(self.evaluate(dist.batch_shape_tensor())),
                            self.evaluate(result).shape)
        self.assertAllEqual([n] + dist.batch_shape, result.shape)
        self.assertAllEqual(
            [n] + dist.batch_shape, self.evaluate(result).shape)

      for method in ('mean', 'variance', 'mode'):
        result = getattr(dist, method)()
        self.assertAllEqual(
            self.evaluate(dist.batch_shape_tensor()), result.shape)
        self.assertAllEqual(
            self.evaluate(dist.batch_shape_tensor()),
            self.evaluate(result).shape)
        self.assertAllEqual(dist.batch_shape, result.shape)
        self.assertAllEqual(dist.batch_shape, self.evaluate(result).shape)

  def testSample(self):
    one = tf.constant(1., dtype=self.dtype)
    seed_stream = test_util.test_seed_stream()

    dists = (self.make_two_piece_normal(), self.make_two_piece_normals())
    n = int(3e2)
    sample_shapes = ([n, n], [n * n])

    for dist, sample_shape in itertools.product(dists, sample_shapes):
      sample = dist.sample(sample_shape, seed=seed_stream())

      uniform_sample = tf.random.uniform(
          sample.shape, maxval=1., dtype=self.dtype, seed=seed_stream())
      sign = tf.where(uniform_sample < 0.5, -one, one)
      normal_sample = self.evaluate(sign * two_piece_normal.standardize(
          sample, loc=dist.loc, scale=dist.scale, skewness=dist.skewness))

      # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
      # The sample variance similarly is dependent on scale and n.
      # Thus, the tolerances below are very sensitive to number of samples
      # as well as the variances chosen.
      self.assertAllEqual(normal_sample.shape, sample_shape + dist.batch_shape)
      self.assertAllClose(np.mean(normal_sample), 0.0, atol=0.1)
      self.assertAllClose(np.std(normal_sample), 1.0, atol=0.1)

  def testLogPDF(self):
    dist = self.make_two_piece_normals()

    x = np.array([[-35.], [3.], [20.]], dtype=self.dtype)

    log_pdf = self.evaluate(dist.log_prob(x))
    # The following values were calculated using the R package gamlss.dist.
    # Package version: 5.3-2. Distribution: SN2.
    expected_log_pdf = np.array([
        [-7.323596, -10.44152, -16.03311],
        [-3.262346, -3.221524, -3.261648],
        [-5.831235, -4.666524, -4.078539],
    ], dtype=self.dtype)

    self.assertAllEqual(log_pdf.shape, expected_log_pdf.shape)
    self.assertAllClose(log_pdf, expected_log_pdf)

  def testCDF(self):
    dist = self.make_two_piece_normals()

    x = np.array([[-35.], [3.], [20.]], dtype=self.dtype)

    cdf = self.evaluate(dist.cdf(x))
    # The following values were calculated using the R package 'gamlss.dist'.
    # Package version: 5.3-2. Distribution: SN2.
    expected_cdf = np.array([
        [2.798031e-03, 7.234804e-05, 1.562540e-07],
        [6.400000e-01, 5.000000e-01, 3.611542e-01],
        [9.915722e-01, 9.554345e-01, 8.714767e-01],
    ], dtype=self.dtype)

    self.assertAllEqual(cdf.shape, expected_cdf.shape)
    self.assertAllClose(cdf, expected_cdf)

  def testSurvivalFunction(self):
    dist = self.make_two_piece_normals()

    x = np.array([[-35.], [3.], [20.]], dtype=self.dtype)

    sf = self.evaluate(dist.survival_function(x))
    # The following values were calculated using the R package 'gamlss.dist'.
    # Package version: 5.3-2. Distribution: SN2.
    expected_sf = np.array([
        [9.972020e-01, 9.999277e-01, 9.999998e-01],
        [3.600000e-01, 5.000000e-01, 6.388458e-01],
        [8.427815e-03, 4.456546e-02, 1.285233e-01],
    ], dtype=self.dtype)

    self.assertAllEqual(sf.shape, expected_sf.shape)
    self.assertAllClose(sf, expected_sf)

  def testQuantile(self):
    dist = self.make_two_piece_normals()

    x = np.array([[0.000001], [0.5], [0.999999]], dtype=self.dtype)

    quantile = self.evaluate(dist.quantile(x))
    # The following values were calculated using the R package 'gamlss.dist'.
    # Package version: 5.3-2. Distribution: SN2.
    expected_quantile = np.array([
        [-61.040950, -44.53424, -32.24254],
        [-0.7025392, 3.0000000, 6.6688350],
        [38.1495200, 50.534240, 66.876050],
    ], dtype=self.dtype)

    self.assertAllEqual(quantile.shape, expected_quantile.shape)
    self.assertAllClose(
        a=quantile,
        b=expected_quantile,
        rtol=1e-03 if self.dtype == np.float32 else 1e-06)

  def testMean(self):
    dist = self.make_two_piece_normals()

    mean = self.evaluate(dist.mean())
    expected_mean = np.array([-1.6543264, 3., 7.612733], dtype=self.dtype)

    self.assertAllEqual(mean.shape, expected_mean.shape)
    self.assertAllClose(mean, expected_mean)

  def testVariance(self):
    dist = self.make_two_piece_normals()

    variance = self.evaluate(dist.variance())
    expected_variance = np.array(
        [112.365005, 100., 112.14502], dtype=self.dtype)

    self.assertAllEqual(variance.shape, expected_variance.shape)
    self.assertAllClose(variance, expected_variance)

  def testMode(self):
    dist = self.make_two_piece_normals()

    mode = self.evaluate(dist.mode())
    expected_mode = np.array([3., 3., 3.], dtype=self.dtype)

    self.assertAllEqual(mode.shape, expected_mode.shape)
    self.assertAllClose(mode, expected_mode)

  @test_util.numpy_disable_gradient_test
  def testFiniteGradientAtDifficultPoints(self):
    def make_fn(attr):
      x = np.array([-100, -20, -5., 0., 5., 20, 100]).astype(self.dtype)
      return lambda m, s, g: getattr(  # pylint: disable=g-long-lambda
          two_piece_normal.TwoPieceNormal(
              m, scale=s, skewness=g, validate_args=True), attr)(
                  x)

    loc = tf.constant(0., self.dtype)
    scale = tf.constant(1., self.dtype)

    # 'log_cdf' currently fails at -100 in fp64 and at -100, -20 in fp32.
    # 'log_survival_function' currently fails at 100, 20 in fp64 and at 100,
    #   20, 5 in fp32.
    # We've already tried the following ideas to solve these problems:
    # * Implementing the log_cdf method directly using the Log Normal
    #   distribution function (log_ndtr) when value < loc;
    # * Implementing the cdf method using the Gamma distribution function; and
    # * Implementing the cdf method using the Student's t distribution function
    #   when value < loc.
    for skewness in [0.75, 1., 1.33]:
      for attr in ('prob', 'cdf', 'survival_function', 'log_prob'):
        value, grads = self.evaluate(
            gradient.value_and_gradient(
                make_fn(attr),
                [loc, scale, tf.constant(skewness, self.dtype)]))
        self.assertAllFinite(value)
        self.assertAllFinite(grads[0])  # d/d loc
        self.assertAllFinite(grads[1])  # d/d scale
        self.assertAllFinite(grads[2])  # d/d skewness

  @test_util.numpy_disable_gradient_test
  def testQuantileFiniteGradientAtDifficultPoints(self):
    def quantile(loc, scale, skewness, probs):
      dist = two_piece_normal.TwoPieceNormal(
          loc, scale=scale, skewness=skewness, validate_args=True)
      return dist.quantile(probs)

    x = -17. if self.dtype == np.float32 else -33.
    loc = tf.constant(0., self.dtype)
    scale = tf.constant(1., self.dtype)
    probs = tf.constant(
        [np.exp(x), np.exp(-2.), 1. - np.exp(-2.), 1. - np.exp(x)],
        dtype=self.dtype)

    for skewness in [0.75, 1., 1.33]:
      value, grads = gradient.value_and_gradient(
          quantile,
          [loc, scale, tf.constant(skewness, self.dtype), probs])
      self.assertAllFinite(value)
      self.assertAllFinite(grads[0])  # d/d loc
      self.assertAllFinite(grads[1])  # d/d scale
      self.assertAllFinite(grads[2])  # d/d skewness
      self.assertAllFinite(grads[3])  # d/d probs

  @test_util.numpy_disable_gradient_test
  def testFullyReparameterized(self):
    n = 100
    def sampler(loc, scale, skewness):
      dist = two_piece_normal.TwoPieceNormal(
          loc, scale=scale, skewness=skewness, validate_args=True)
      return dist.sample(n, seed=test_util.test_seed())

    loc = tf.constant(0., self.dtype)
    scale = tf.constant(1., self.dtype)

    for skewness in [0.75, 1., 1.33]:
      _, grads = gradient.value_and_gradient(
          sampler, [loc, scale, tf.constant(skewness, self.dtype)])
      self.assertIsNotNone(grads[0])  # d/d loc
      self.assertIsNotNone(grads[1])  # d/d scale
      self.assertIsNotNone(grads[2])  # d/d skewness

  @test_util.numpy_disable_gradient_test
  def testDifferentiableSampleNumerically(self):
    """Test the gradients of the samples w.r.t. skewness."""
    sample_shape = [int(2e5)]
    seed = test_util.test_seed()

    def get_abs_sample_mean(skewness):
      loc = tf.constant(0., self.dtype)
      scale = tf.constant(1., self.dtype)
      dist = two_piece_normal.TwoPieceNormal(
          loc, scale=scale, skewness=skewness, validate_args=True)
      return tf.reduce_mean(tf.abs(dist.sample(sample_shape, seed=seed)))

    for skewness in [0.75, 1., 1.33]:
      err = self.compute_max_gradient_error(
          get_abs_sample_mean, [tf.constant(skewness, self.dtype)], delta=0.1)
      self.assertLess(err, 0.05)

  @test_util.numpy_disable_gradient_test
  def testDifferentiableSampleAnalytically(self):
    """Test the gradients of the samples w.r.t. loc and scale."""
    n = 100
    sample_shape = [n, n]
    n_samples = np.prod(sample_shape)

    n_params = 20
    loc = tf.constant(
        np.linspace(-3., stop=3., num=n_params), dtype=self.dtype)
    scale = tf.constant(
        np.linspace(0.1, stop=10., num=n_params), dtype=self.dtype)
    skewness = tf.constant(
        np.linspace(0.75, stop=1.33, num=n_params), dtype=self.dtype)

    seed = test_util.test_seed()

    def get_exp_samples(loc, scale):
      dist = two_piece_normal.TwoPieceNormal(
          loc=loc, scale=scale, skewness=skewness, validate_args=True)
      return tf.math.exp(dist.sample(sample_shape, seed=seed))

    exp_samples, dsamples = gradient.value_and_gradient(get_exp_samples,
                                                        [loc, scale])
    dloc_auto, dscale_auto = [grad / n_samples for grad in dsamples]

    dloc_calc = tf.reduce_mean(exp_samples, axis=[0, 1])
    dscale_calc = tf.reduce_mean(
        (tf.math.log(exp_samples) - loc) / scale * exp_samples, axis=[0, 1])

    self.assertAllClose(dloc_auto, dloc_calc)
    self.assertAllClose(dscale_auto, dscale_calc)

  def testNegativeScaleSkewnessFails(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      dist = two_piece_normal.TwoPieceNormal(
          loc=[0.], scale=[-1.], skewness=[1.], validate_args=True)
      self.evaluate(dist.mean())

    with self.assertRaisesOpError('Argument `skewness` must be positive.'):
      dist = two_piece_normal.TwoPieceNormal(
          loc=[0.], scale=[1.], skewness=[-1.], validate_args=True)
      self.evaluate(dist.mean())

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_test_missing_functionality(
      'NumpyVariable does not handle unknown shapes')
  def testShapeWithPlaceholders(self):
    loc = tf.Variable(self.dtype(0), shape=tf.TensorShape(None))
    scale = tf.Variable(self.dtype([1., 2.]), shape=tf.TensorShape(None))
    skewness = tf.Variable(
        self.dtype([[0.75, 1.33]]).T, shape=tf.TensorShape(None))
    self.evaluate([loc.initializer, scale.initializer, skewness.initializer])
    dist = two_piece_normal.TwoPieceNormal(
        loc=loc, scale=scale, skewness=skewness, validate_args=True)

    # get_batch_shape should return an '<unknown>' tensor (graph mode only).
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, tf.TensorShape(None))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), [2, 2])

  def testVariableSkewness(self):
    loc = tf.constant(0., self.dtype)
    scale = tf.constant(1., self.dtype)
    skewness = tf.Variable(1., dtype=self.dtype)
    dist = two_piece_normal.TwoPieceNormal(
        loc=loc, scale=scale, skewness=skewness, validate_args=True)

    self.evaluate([v.initializer for v in dist.variables])
    self.assertIs(skewness, dist.skewness)
    self.assertEqual(0., self.evaluate(dist.mean()))

    with self.assertRaisesOpError('Argument `skewness` must be positive.'):
      with tf.control_dependencies([skewness.assign(-1.)]):
        self.evaluate(dist.mean())

  def testIncompatibleArgShapesGraph(self):
    skewness = tf.Variable(
        tf.ones([2, 3], dtype=self.dtype), shape=tf.TensorShape(None))
    self.evaluate(skewness.initializer)

    with self.assertRaisesRegexp(Exception, r'compatible shapes'):
      dist = two_piece_normal.TwoPieceNormal(
          loc=tf.zeros([4, 1], dtype=self.dtype),
          scale=tf.ones([4, 1], dtype=self.dtype),
          skewness=skewness,
          validate_args=True)
      self.evaluate(dist.mean())


class TwoPieceNormalEagerGCTest(test_util.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testMeanAndMode(self):
    dist = two_piece_normal.TwoPieceNormal(
        loc=3., scale=10., skewness=[0.75, 1., 1.33], validate_args=True)

    self.assertAllEqual((3,), dist.mean().shape)
    expected_mean = np.array([-1.6543264, 3., 7.612733], dtype=np.float32)
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

    self.assertAllEqual((3,), dist.mode().shape)
    expected_mode = np.array([3., 3., 3.], dtype=np.float32)
    self.assertAllEqual(expected_mode, self.evaluate(dist.mode()))


@test_util.test_all_tf_execution_regimes
class TwoPieceNormalTestStaticShapeFloat32(test_util.TestCase,
                                           _TwoPieceNormalTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class TwoPieceNormalTestDynamicShapeFloat32(test_util.TestCase,
                                            _TwoPieceNormalTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class TwoPieceNormalTestStaticShapeFloat64(test_util.TestCase,
                                           _TwoPieceNormalTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class TwoPieceNormalTestDynamicShapeFloat64(test_util.TestCase,
                                            _TwoPieceNormalTest):
  dtype = np.float64
  use_static_shape = False


if __name__ == '__main__':
  test_util.main()
