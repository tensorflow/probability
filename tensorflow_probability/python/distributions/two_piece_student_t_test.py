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
"""Tests for Two-Piece Student's t-distribution."""

import itertools

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import two_piece_student_t
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class _TwoPieceStudentTTest(parameterized.TestCase):

  def make_two_piece_student_t(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as float32.
      dist = two_piece_student_t.TwoPieceStudentT(
          df=6., loc=3., scale=10., skewness=0.75, validate_args=True)
    else:
      dist = two_piece_student_t.TwoPieceStudentT(
          df=self.dtype(6.),
          loc=self.dtype(3.),
          scale=self.dtype(10.),
          skewness=self.dtype(0.75),
          validate_args=True)

    return dist

  def make_two_piece_student_ts(self):
    skewness = [0.75, 1., 1.33]

    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as float32.
      dist = two_piece_student_t.TwoPieceStudentT(
          df=6., loc=3., scale=10., skewness=skewness, validate_args=True)
    else:
      dist = two_piece_student_t.TwoPieceStudentT(
          df=self.dtype(6.),
          loc=self.dtype(3.),
          scale=self.dtype(10.),
          skewness=self.dtype(skewness),
          validate_args=True)

    return dist

  def helper_param_shapes(self, sample_shape, expected):
    param_shapes = two_piece_student_t.TwoPieceStudentT.param_shapes(
        sample_shape)
    df_shape = param_shapes['df']
    loc_shape = param_shapes['loc']
    scale_shape = param_shapes['scale']
    skewness_shape = param_shapes['skewness']

    self.assertAllEqual(expected, self.evaluate(df_shape))
    self.assertAllEqual(expected, self.evaluate(loc_shape))
    self.assertAllEqual(expected, self.evaluate(scale_shape))
    self.assertAllEqual(expected, self.evaluate(skewness_shape))

    df = tf.ones(df_shape)
    loc = tf.zeros(loc_shape)
    scale = tf.ones(scale_shape)
    skewness = tf.ones(skewness_shape)
    seed = test_util.test_seed()
    samples = two_piece_student_t.TwoPieceStudentT(
        df, loc, scale, skewness, validate_args=True).sample(seed=seed)

    self.assertAllEqual(expected, self.evaluate(tf.shape(samples)))

  def helper_param_static_shapes(self, sample_shape, expected):
    param_shapes = two_piece_student_t.TwoPieceStudentT.param_static_shapes(
        sample_shape)
    df_shape = param_shapes['df']
    loc_shape = param_shapes['loc']
    scale_shape = param_shapes['scale']
    skewness_shape = param_shapes['skewness']

    self.assertEqual(expected, df_shape)
    self.assertEqual(expected, loc_shape)
    self.assertEqual(expected, scale_shape)
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

  def testShape(self):
    dists = (self.make_two_piece_student_t(), self.make_two_piece_student_ts())

    for dist in dists:
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

      for method in ('log_prob', 'cdf', 'survival_function'):
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
    one = self.dtype(1.)
    two = self.dtype(2.)
    seed_stream = test_util.test_seed_stream()

    dists = (self.make_two_piece_student_t(), self.make_two_piece_student_ts())
    n = int(1e2)
    sample_shapes = ([n, n], [n * n])

    for dist, sample_shape in itertools.product(dists, sample_shapes):
      sample = dist.sample(sample_shape, seed=seed_stream())

      uniform_sample = tf.random.uniform(
          sample.shape, maxval=one, dtype=self.dtype, seed=seed_stream())
      sign = tf.where(uniform_sample < 0.5, -one, one)
      student_t_sample = self.evaluate(sign * two_piece_student_t.standardize(
          sample, loc=dist.loc, scale=dist.scale, skewness=dist.skewness))

      # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
      # The sample variance similarly is dependent on scale and n.
      # Thus, the tolerances below are very sensitive to number of samples
      # as well as the variances chosen.
      self.assertDTypeEqual(sample, self.dtype)
      self.assertAllEqual(
          student_t_sample.shape, sample_shape + dist.batch_shape)
      self.assertAllClose(np.mean(student_t_sample), 0.0, atol=0.1)
      self.assertAllClose(
          np.std(student_t_sample),
          self.evaluate(tf.math.sqrt(dist.df / (dist.df - two))),
          atol=0.1)

  def testLogPDF(self):
    dist = self.make_two_piece_student_ts()

    x = np.array([[-123.], [-4.], [3.], [10.], [88.]], dtype=self.dtype)

    log_pdf = self.evaluate(dist.log_prob(x))
    # The following values were calculated using the R package gamlss.dist.
    # Package version: 5.3-2. Distribution: ST3.
    expected_log_pdf = np.array([
        [-12.982363352394, -14.857559749935, -16.838090065368],
        [ -3.461022986748,  -3.537764063953,  -3.775393050210],
        [ -3.303825343266,  -3.263003348746,  -3.303127355368],
        [ -3.778307595268,  -3.537764063953,  -3.461096404463],
        [-14.186695915359, -12.251526109418, -10.495893758498],
    ], dtype=self.dtype)

    self.assertDTypeEqual(log_pdf, self.dtype)
    self.assertShapeEqual(log_pdf, expected_log_pdf)
    self.assertAllClose(
        log_pdf,
        expected_log_pdf,
        atol=0.,
        rtol=1e-6 if self.dtype == np.float32 else 2e-9)

  def testCDF(self):
    dist = self.make_two_piece_student_ts()

    x = np.array([[-123.], [-4.], [3.], [10.], [88.]], dtype=self.dtype)

    cdf = self.evaluate(dist.cdf(x))
    # The following values were calculated using the R package gamlss.dist.
    # Package version: 5.3-2. Distribution: ST3.
    expected_cdf = np.array([
        [5.11329136655268e-5, 7.65138872532116e-6, 1.04125412928787e-6],
        [3.95781464247676e-1, 2.55071690475380e-1, 1.40048351817201e-1],
        [6.40000000000000e-1, 5.00000000000000e-1, 3.61154248979739e-1],
        [8.60799760188126e-1, 7.44928309524620e-1, 6.05481963344001e-1],
        [9.99989828542182e-1, 9.99927435887923e-1, 9.99558660780453e-1],
    ], dtype=self.dtype)

    self.assertDTypeEqual(cdf, self.dtype)
    self.assertShapeEqual(cdf, expected_cdf)
    self.assertAllClose(
        cdf,
        expected_cdf,
        atol=0.,
        rtol=2e-6 if self.dtype == np.float32 else 1e-12)

  def testSurvivalFunction(self):
    dist = self.make_two_piece_student_ts()

    x = np.array([[-123.], [-4.], [3.], [10.], [88.]], dtype=self.dtype)

    sf = self.evaluate(dist.survival_function(x))
    # The following values were calculated using the following relation:
    #   expected_sf = 1 - expected_cdf
    expected_sf = np.array([
        [9.999488670863345e-01, 9.999923486112747e-01, 9.999989587458707e-01],
        [6.042185357523240e-01, 7.449283095246200e-01, 8.599516481827990e-01],
        [3.600000000000000e-01, 5.000000000000000e-01, 6.388457510202610e-01],
        [1.392002398118740e-01, 2.550716904753800e-01, 3.945180366559990e-01],
        [1.017145781800899e-05, 7.256411207701152e-05, 4.413392195470323e-04],
    ], dtype=self.dtype)

    self.assertDTypeEqual(sf, self.dtype)
    self.assertShapeEqual(sf, expected_sf)
    self.assertAllClose(
        sf,
        expected_sf,
        atol=0.,
        rtol=1e-6 if self.dtype == np.float32 else 5e-11)

  def testQuantile(self):
    dist = self.make_two_piece_student_ts()

    p = np.array(
        [[1e-6], [0.25], [0.5], [0.75], [1. - 1e-5]], dtype=self.dtype)

    quantile = self.evaluate(dist.quantile(p))
    # The following values were calculated using the R package gamlss.dist.
    # Package version: 5.3-2. Distribution: ST3.
    expected_quantile = np.array([
        [-244.884076670630000, -175.303145006555, -123.86764610382800],
        [  -9.334084600342130,   -4.175581964914,   -0.12407156338927],
        [  -0.872658556456447,    3.000000000000,    6.83723135012289],
        [   6.092263521199590,   10.175581964914,   15.28370839341050],
        [  88.250979391306600,  123.316531781276,  169.92793695323400],
    ], dtype=self.dtype)

    self.assertDTypeEqual(quantile, self.dtype)
    self.assertShapeEqual(quantile, expected_quantile)
    self.assertAllClose(
        quantile,
        expected_quantile,
        atol=0.,
        rtol=3e-4 if self.dtype == np.float32 else 1e-12)

  def testMean(self):
    dist = self.make_two_piece_student_ts()

    mean = self.evaluate(dist.mean())
    # The following values were calculated using the R package gamlss.dist.
    # Package version: 5.3-2. Distribution: ST3.
    expected_mean = np.array([-2.35825881, 3., 8.31037405], dtype=self.dtype)

    self.assertDTypeEqual(mean, self.dtype)
    self.assertShapeEqual(mean, expected_mean)
    self.assertAllClose(mean, expected_mean)

    dist = two_piece_student_t.TwoPieceStudentT(
        df=self.dtype(1.),
        loc=self.dtype(0.),
        scale=self.dtype(1.),
        skewness=self.dtype(1.),
        validate_args=True)

    mean = self.evaluate(dist.mean())
    self.assertDTypeEqual(mean, self.dtype)
    self.assertAllNan(mean)

    with self.assertRaisesOpError('mean not defined for '
                                  'components of df <= 1'):
      dist = two_piece_student_t.TwoPieceStudentT(
          df=self.dtype(1.),
          loc=self.dtype(0.),
          scale=self.dtype(1.),
          skewness=self.dtype(1.),
          allow_nan_stats=False,
          validate_args=True)
      self.evaluate(dist.mean())

  def testVariance(self):
    dist = self.make_two_piece_student_ts()

    var = self.evaluate(dist.variance())
    # The following values were calculated using the R package gamlss.dist.
    # Package version: 5.3-2. Distribution: ST3.
    expected_var = np.array(
        [172.33072917, 150., 171.93338977], dtype=self.dtype)

    self.assertDTypeEqual(var, self.dtype)
    self.assertShapeEqual(var, expected_var)
    self.assertAllClose(var, expected_var)

    dist = two_piece_student_t.TwoPieceStudentT(
        df=self.dtype([1., 2.]),
        loc=self.dtype(0.),
        scale=self.dtype(1.),
        skewness=self.dtype(1.),
        validate_args=True)

    var = self.evaluate(dist.variance())
    expected_var = np.array([np.nan, np.inf], dtype=self.dtype)

    self.assertDTypeEqual(var, self.dtype)
    self.assertShapeEqual(var, expected_var)
    self.assertAllEqual(var, expected_var)

    with self.assertRaisesOpError('variance not defined for '
                                  'components of df <= 1'):
      dist = two_piece_student_t.TwoPieceStudentT(
          df=self.dtype(1.),
          loc=self.dtype(0.),
          scale=self.dtype(1.),
          skewness=self.dtype(1.),
          allow_nan_stats=False,
          validate_args=True)
      self.evaluate(dist.variance())

  def testMode(self):
    dist = self.make_two_piece_student_ts()

    mode = self.evaluate(dist.mode())
    expected_mode = np.array([3., 3., 3.], dtype=self.dtype)

    self.assertDTypeEqual(mode, self.dtype)
    self.assertShapeEqual(mode, expected_mode)
    self.assertAllClose(mode, expected_mode)

  @test_util.numpy_disable_gradient_test
  @parameterized.parameters(0.75, 1., 1.33)
  def testFiniteGradientAtDifficultPoints(self, skewness):
    eps = np.finfo(self.dtype).eps
    large = 1. / np.sqrt(eps)

    def make_fn(attr):
      x = np.array([-large, -eps, 0., eps, large], dtype=self.dtype)
      return lambda d, m, s, g: getattr(  # pylint: disable=g-long-lambda
          two_piece_student_t.TwoPieceStudentT(
              df=d, loc=m, scale=s, skewness=g, validate_args=True), attr)(x)

    df = tf.constant(6., self.dtype)
    loc = tf.constant(0., self.dtype)
    scale = tf.constant(1., self.dtype)

    for attr in ('log_prob', 'cdf', 'survival_function'):
      value, grads = self.evaluate(
          gradient.value_and_gradient(
              make_fn(attr),
              [df, loc, scale, tf.constant(skewness, self.dtype)]))
      self.assertAllFinite(value)
      self.assertAllFinite(grads[0])  # d/d df
      self.assertAllFinite(grads[1])  # d/d loc
      self.assertAllFinite(grads[2])  # d/d scale
      self.assertAllFinite(grads[3])  # d/d skewness

  @test_util.numpy_disable_gradient_test
  @parameterized.parameters(0.75, 1., 1.33)
  def testQuantileFiniteGradientAtDifficultPoints(self, skewness):
    eps = np.finfo(self.dtype).eps

    def quantile(df, loc, scale, skewness, probs):
      dist = two_piece_student_t.TwoPieceStudentT(
          df=df, loc=loc, scale=scale, skewness=skewness, validate_args=True)
      return dist.quantile(probs)

    df = tf.constant(6., self.dtype)
    loc = tf.constant(0., self.dtype)
    scale = tf.constant(1., self.dtype)
    probs = tf.constant(
        [np.square(eps), 0.5 - eps, 0.5, 0.5 + eps, 1. - eps],
        dtype=self.dtype)

    value, grads = self.evaluate(
        gradient.value_and_gradient(
            quantile,
            [df, loc, scale, tf.constant(skewness, self.dtype), probs]))
    self.assertAllFinite(value)
    self.assertAllFinite(grads[0])  # d/d df
    self.assertAllFinite(grads[1])  # d/d loc
    self.assertAllFinite(grads[2])  # d/d scale
    self.assertAllFinite(grads[3])  # d/d skewness

  @test_util.numpy_disable_gradient_test
  def testDifferentiableSample(self):
    """Test the gradients of the samples w.r.t. `df` and `skewness`."""
    sample_shape = [int(2e3)]
    seed = test_util.test_seed(sampler_type='stateless')

    space_df = np.linspace(6., 30., num=5).tolist()
    space_skewness = np.linspace(0.75, 1.33, num=5).tolist()
    df, skewness = [
        tf.constant(z, dtype=self.dtype)
        for z in zip(*list(itertools.product(space_df, space_skewness)))]

    def get_abs_sample_mean(df, skewness):
      loc = tf.constant(0., self.dtype)
      scale = tf.constant(1., self.dtype)
      dist = two_piece_student_t.TwoPieceStudentT(
          df=df, loc=loc, scale=scale, skewness=skewness, validate_args=True)
      return tf.reduce_mean(
          tf.abs(dist.sample(sample_shape, seed=seed)), axis=0)

    err = self.compute_max_gradient_error(
        lambda _df: get_abs_sample_mean(_df, skewness), [df], delta=1e-4)
    self.assertLess(err, 8e-3)

    err = self.compute_max_gradient_error(
        lambda _skewness: get_abs_sample_mean(df, _skewness),
        [skewness],
        delta=5e-2)
    self.assertLess(err, 0.11)

  def testNonPositiveDfScaleSkewnessFails(self):
    with self.assertRaisesOpError('Argument `df` must be positive.'):
      dist = two_piece_student_t.TwoPieceStudentT(
          df=0., loc=0., scale=1., skewness=1., validate_args=True)
      self.evaluate(dist.mean())

    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      dist = two_piece_student_t.TwoPieceStudentT(
          df=2., loc=0., scale=0., skewness=1., validate_args=True)
      self.evaluate(dist.mean())

    with self.assertRaisesOpError('Argument `skewness` must be positive.'):
      dist = two_piece_student_t.TwoPieceStudentT(
          df=2., loc=0., scale=1., skewness=0., validate_args=True)
      self.evaluate(dist.mean())

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_test_missing_functionality(
      'NumpyVariable does not handle unknown shapes')
  def testShapeWithPlaceholders(self):
    df = tf.Variable(self.dtype(2.), shape=tf.TensorShape(None))
    loc = tf.Variable(self.dtype([0., 1.]), shape=tf.TensorShape(None))
    scale = tf.Variable(self.dtype([[1., 2.]]), shape=tf.TensorShape(None))
    skewness = tf.Variable(
        self.dtype([[0.75, 1.], [1., 1.33]]).T, shape=tf.TensorShape(None))

    dist = two_piece_student_t.TwoPieceStudentT(
        df=df, loc=loc, scale=scale, skewness=skewness, validate_args=True)
    self.evaluate([v.initializer for v in dist.variables])

    # get_batch_shape should return an '<unknown>' tensor (graph mode only).
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, tf.TensorShape(None))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), [2, 2])

  def testVariableParameters(self):
    df = tf.Variable(2., dtype=self.dtype)
    loc = tf.Variable(0., dtype=self.dtype)
    scale = tf.Variable(1., dtype=self.dtype)
    skewness = tf.Variable(1., dtype=self.dtype)
    dist = two_piece_student_t.TwoPieceStudentT(
        df=df, loc=loc, scale=scale, skewness=skewness, validate_args=True)

    self.evaluate([v.initializer for v in dist.variables])
    self.assertIs(df, dist.df)
    self.assertIs(loc, dist.loc)
    self.assertIs(scale, dist.scale)
    self.assertIs(skewness, dist.skewness)
    self.assertEqual(0., self.evaluate(dist.mean()))

    with self.assertRaisesOpError('Argument `df` must be positive.'):
      with tf.control_dependencies([df.assign(0.)]):
        self.evaluate(dist.mean())

    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([df.assign(2.), scale.assign(0.)]):
        self.evaluate(dist.mean())

    with self.assertRaisesOpError('Argument `skewness` must be positive.'):
      with tf.control_dependencies([scale.assign(1.), skewness.assign(0.)]):
        self.evaluate(dist.mean())

  def testIncompatibleArgShapesGraph(self):
    skewness = tf.Variable(
        tf.ones([2, 3], dtype=self.dtype), shape=tf.TensorShape(None))
    self.evaluate(skewness.initializer)

    with self.assertRaisesRegex(Exception, r'compatible shapes'):
      dist = two_piece_student_t.TwoPieceStudentT(
          df=self.dtype(2.) * tf.ones([4, 1], dtype=self.dtype),
          loc=tf.zeros([4, 1], dtype=self.dtype),
          scale=tf.ones([4, 1], dtype=self.dtype),
          skewness=skewness,
          validate_args=True)
      self.evaluate(dist.mean())


@test_util.test_all_tf_execution_regimes
class TwoPieceStudentTTestStaticShapeFloat32(test_util.TestCase,
                                             _TwoPieceStudentTTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class TwoPieceStudentTTestDynamicShapeFloat32(test_util.TestCase,
                                              _TwoPieceStudentTTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class TwoPieceStudentTTestStaticShapeFloat64(test_util.TestCase,
                                             _TwoPieceStudentTTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class TwoPieceStudentTTestDynamicShapeFloat64(test_util.TestCase,
                                              _TwoPieceStudentTTest):
  dtype = np.float64
  use_static_shape = False


del _TwoPieceStudentTTest


if __name__ == '__main__':
  test_util.main()
