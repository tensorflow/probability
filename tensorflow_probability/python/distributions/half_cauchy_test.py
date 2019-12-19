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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


class _HalfCauchyTest(object):

  def _create_placeholder_with_default(self, default, name=None):
    default_ = tf.convert_to_tensor(default, dtype=self.dtype)
    return tf1.placeholder_with_default(
        default_,
        shape=default_.shape if self.use_static_shape else None,
        name=name)

  def _test_param_shapes(self, sample_shape, expected):
    param_shapes = tfd.HalfCauchy.param_shapes(sample_shape)
    loc_shape, scale_shape = param_shapes['loc'], param_shapes['scale']
    self.assertAllEqual(expected, self.evaluate(loc_shape))
    self.assertAllEqual(expected, self.evaluate(scale_shape))
    loc = tf.zeros(loc_shape)
    scale = tf.ones(scale_shape)
    self.assertAllEqual(
        expected,
        self.evaluate(
            tf.shape(tfd.HalfCauchy(loc, scale, validate_args=True).sample(
                seed=test_util.test_seed()))))

  def _test_param_static_shapes(self, sample_shape, expected):
    param_shapes = tfd.HalfCauchy.param_static_shapes(sample_shape)
    loc_shape, scale_shape = param_shapes['loc'], param_shapes['scale']
    self.assertEqual(expected, loc_shape)
    self.assertEqual(expected, scale_shape)

  def testHalfCauchyParamShapes(self):
    sample_shape = [10, 3, 4]
    self._test_param_shapes(sample_shape, sample_shape)
    self._test_param_shapes(tf.constant(sample_shape), sample_shape)

  def testHalfCauchyParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._test_param_static_shapes(sample_shape, sample_shape)
    self._test_param_static_shapes(tf.TensorShape(sample_shape), sample_shape)

  def testHalfCauchyShape(self):
    batch_size = 6
    loc = self._create_placeholder_with_default([0.] * batch_size, name='loc')
    scale = self._create_placeholder_with_default(
        [1.] * batch_size, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)

    if self.use_static_shape or tf.executing_eagerly():
      expected_batch_shape = tf.TensorShape([batch_size])
    else:
      expected_batch_shape = tf.TensorShape(None)

    self.assertEqual(
        self.evaluate(half_cauchy.batch_shape_tensor()), (batch_size,))
    self.assertEqual(half_cauchy.batch_shape, expected_batch_shape)
    self.assertAllEqual(self.evaluate(half_cauchy.event_shape_tensor()), [])
    self.assertEqual(half_cauchy.event_shape, tf.TensorShape([]))

  def testHalfCauchyShapeBroadcast(self):
    loc = self._create_placeholder_with_default([0., 1.], name='loc')
    scale = self._create_placeholder_with_default(
        [[1.], [2.], [3.]], name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)

    if self.use_static_shape or tf.executing_eagerly():
      expected_batch_shape = tf.TensorShape([3, 2])
    else:
      expected_batch_shape = tf.TensorShape(None)

    self.assertAllEqual(self.evaluate(half_cauchy.batch_shape_tensor()), (3, 2))
    self.assertAllEqual(half_cauchy.batch_shape, expected_batch_shape)
    self.assertAllEqual(self.evaluate(half_cauchy.event_shape_tensor()), [])
    self.assertEqual(half_cauchy.event_shape, tf.TensorShape([]))

  def testHalfCauchyInvalidScale(self):
    invalid_scales = [0., -0.01, -2.]
    loc = self._create_placeholder_with_default(0., name='loc')
    for scale_ in invalid_scales:
      scale = self._create_placeholder_with_default(scale_, name='scale')
      with self.assertRaisesOpError('Condition x > 0'):
        half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
        self.evaluate(half_cauchy.entropy())

  def testHalfCauchyPdf(self):
    batch_size = 6
    loc_ = 2.
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale_ = 3.
    scale = self._create_placeholder_with_default(
        [scale_] * batch_size, name='scale')
    x_ = [2., 3., 3.1, 4., 5., 6.]
    x = self._create_placeholder_with_default(x_, name='x')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    for tfp_f, scipy_f in [
        (half_cauchy.prob, stats.halfcauchy.pdf),
        (half_cauchy.log_prob, stats.halfcauchy.logpdf)]:
      tfp_res = tfp_f(x)
      if self.use_static_shape or tf.executing_eagerly():
        expected_shape = tf.TensorShape((batch_size,))
      else:
        expected_shape = tf.TensorShape(None)
      self.assertEqual(tfp_res.shape, expected_shape)
      self.assertAllEqual(self.evaluate(tf.shape(tfp_res)), (batch_size,))
      self.assertAllClose(
          self.evaluate(tfp_res),
          scipy_f(x_, loc_, scale_))

  def testHalfCauchyPdfValidateArgs(self):
    batch_size = 3
    loc = self._create_placeholder_with_default([-1, 0., 1.1], name='loc')
    scale = self._create_placeholder_with_default(
        [1.] * batch_size, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    x_vals = [[-1.1, 1., 2.],
              [0., -1., 2.],
              [0., 1., 1.09]]
    for x_ in x_vals:
      for f in [half_cauchy.prob, half_cauchy.log_prob]:
        with self.assertRaisesOpError('must be greater than'):
          x = self._create_placeholder_with_default(x_, name='x')
          self.evaluate(f(x))

  def testHalfCauchyPdfMultidimensional(self):
    batch_size = 6
    loc_ = [-1, 0., 1.1]
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale_ = [0.1, 1., 2.5]
    scale = self._create_placeholder_with_default(
        [scale_] * batch_size, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    x_ = [[2.], [3.1], [4.], [5.], [6.], [7.]]
    x = self._create_placeholder_with_default(x_, name='x')
    for tfp_f, scipy_f in [
        (half_cauchy.prob, stats.halfcauchy.pdf),
        (half_cauchy.log_prob, stats.halfcauchy.logpdf)]:
      tfp_res = tfp_f(x)
      if self.use_static_shape or tf.executing_eagerly():
        expected_shape = tf.TensorShape((batch_size, 3))
      else:
        expected_shape = tf.TensorShape(None)
      self.assertEqual(tfp_res.shape, expected_shape)
      self.assertAllEqual(
          self.evaluate(tf.shape(tfp_res)), (batch_size, 3))
      self.assertAllClose(
          self.evaluate(tfp_res),
          scipy_f(x_, loc_, scale_))

  def testHalfCauchyPdfBroadcast(self):
    loc_ = [-1, 0., 1.1]
    loc = self._create_placeholder_with_default(loc_, name='loc')
    scale_ = [0.1]
    scale = self._create_placeholder_with_default(scale_, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    x_ = [[2.], [3.1], [4.], [5.], [6.], [7.]]
    x = self._create_placeholder_with_default(x_, name='x')
    for tfp_f, scipy_f in [
        (half_cauchy.prob, stats.halfcauchy.pdf),
        (half_cauchy.log_prob, stats.halfcauchy.logpdf)]:
      tfp_res = tfp_f(x)
      if self.use_static_shape or tf.executing_eagerly():
        expected_shape = tf.TensorShape((6, 3))
      else:
        expected_shape = tf.TensorShape(None)
      self.assertEqual(tfp_res.shape, expected_shape)
      self.assertAllEqual(self.evaluate(tf.shape(tfp_res)), (6, 3))
      self.assertAllClose(
          self.evaluate(tfp_res),
          scipy_f(x_, loc_, scale_))

  def testHalfCauchyCdf(self):
    batch_size = 6
    loc_ = 2.
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale_ = 3.
    scale = self._create_placeholder_with_default(
        [scale_] * batch_size, name='scale')
    x_ = [2., 3., 3.1, 4., 5., 6.]
    x = self._create_placeholder_with_default(x_, name='x')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    for tfp_f, scipy_f in [
        (half_cauchy.cdf, stats.halfcauchy.cdf),
        (half_cauchy.log_cdf, stats.halfcauchy.logcdf)]:
      tfp_res = tfp_f(x)
      if self.use_static_shape or tf.executing_eagerly():
        expected_shape = tf.TensorShape((batch_size,))
      else:
        expected_shape = tf.TensorShape(None)
      self.assertEqual(tfp_res.shape, expected_shape)
      self.assertAllEqual(self.evaluate(tf.shape(tfp_res)), (batch_size,))
      self.assertAllClose(
          self.evaluate(tfp_res),
          scipy_f(x_, loc_, scale_))

  def testHalfCauchyCdfValidateArgs(self):
    batch_size = 3
    loc = self._create_placeholder_with_default([-1, 0., 1.1], name='loc')
    scale = self._create_placeholder_with_default(
        [1.] * batch_size, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    x_vals = [[-1.1, 1., 2.],
              [0., -1., 2.],
              [0., 1., 1.09]]
    for x_ in x_vals:
      for f in [half_cauchy.cdf, half_cauchy.log_cdf]:
        with self.assertRaisesOpError('must be greater than'):
          x = self._create_placeholder_with_default(x_, name='x')
          self.evaluate(f(x))

  def testHalfCauchyCdfMultidimensional(self):
    batch_size = 6
    loc_ = [-1, 0., 1.1]
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale_ = [0.1, 1., 2.5]
    scale = self._create_placeholder_with_default(
        [scale_] * batch_size, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    x_ = [[2.], [3.1], [4.], [5.], [6.], [7.]]
    x = self._create_placeholder_with_default(x_, name='x')
    for tfp_f, scipy_f in [
        (half_cauchy.cdf, stats.halfcauchy.cdf),
        (half_cauchy.log_cdf, stats.halfcauchy.logcdf)]:
      tfp_res = tfp_f(x)
      if self.use_static_shape or tf.executing_eagerly():
        expected_shape = tf.TensorShape((batch_size, 3))
      else:
        expected_shape = tf.TensorShape(None)
      self.assertEqual(tfp_res.shape, expected_shape)
      self.assertAllEqual(
          self.evaluate(tf.shape(tfp_res)), (batch_size, 3))
      self.assertAllClose(
          self.evaluate(tfp_res),
          scipy_f(x_, loc_, scale_))

  def testHalfCauchyCdfBroadcast(self):
    loc_ = [-1, 0., 1.1]
    loc = self._create_placeholder_with_default(loc_, name='loc')
    scale_ = [0.1]
    scale = self._create_placeholder_with_default(scale_, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    x_ = [[2.], [3.1], [4.], [5.], [6.], [7.]]
    x = self._create_placeholder_with_default(x_, name='x')
    for tfp_f, scipy_f in [
        (half_cauchy.cdf, stats.halfcauchy.cdf),
        (half_cauchy.log_cdf, stats.halfcauchy.logcdf)]:
      tfp_res = tfp_f(x)
      if self.use_static_shape or tf.executing_eagerly():
        expected_shape = tf.TensorShape((6, 3))
      else:
        expected_shape = tf.TensorShape(None)
      self.assertEqual(tfp_res.shape, expected_shape)
      self.assertAllEqual(self.evaluate(tf.shape(tfp_res)), (6, 3))
      self.assertAllClose(
          self.evaluate(tfp_res),
          scipy_f(x_, loc_, scale_))

  def testHalfCauchyMean(self):
    batch_size = 3
    loc = self._create_placeholder_with_default([0.] * batch_size, name='loc')
    scale = self._create_placeholder_with_default(1., name='scale')
    half_cauchy = tfd.HalfCauchy(
        loc, scale, allow_nan_stats=False, validate_args=True)
    with self.assertRaisesRegexp(ValueError, 'is undefined'):
      self.evaluate(half_cauchy.mean())

    half_cauchy = tfd.HalfCauchy(
        loc, scale, allow_nan_stats=True, validate_args=True)
    self.assertAllNan(half_cauchy.mean())

  def testHalfCauchyVariance(self):
    batch_size = 3
    loc = self._create_placeholder_with_default([0.] * batch_size, name='loc')
    scale = self._create_placeholder_with_default(1., name='scale')

    half_cauchy = tfd.HalfCauchy(
        loc, scale, allow_nan_stats=False, validate_args=True)
    with self.assertRaisesRegexp(ValueError, 'is undefined'):
      self.evaluate(half_cauchy.variance())

    half_cauchy = tfd.HalfCauchy(
        loc, scale, allow_nan_stats=True, validate_args=True)
    self.assertAllNan(half_cauchy.variance())

  def testHalfCauchyStddev(self):
    batch_size = 3
    loc = self._create_placeholder_with_default([0.] * batch_size, name='loc')
    scale = self._create_placeholder_with_default(1., name='scale')

    half_cauchy = tfd.HalfCauchy(
        loc, scale, allow_nan_stats=False, validate_args=True)
    with self.assertRaisesRegexp(ValueError, 'is undefined'):
      self.evaluate(half_cauchy.stddev())

    half_cauchy = tfd.HalfCauchy(
        loc, scale, allow_nan_stats=True, validate_args=True)
    self.assertAllNan(half_cauchy.stddev())

  def testHalfCauchyEntropy(self):
    batch_size = 6
    loc_ = 2.
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale_ = 3.
    scale = self._create_placeholder_with_default(
        [scale_] * batch_size, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    entropy = half_cauchy.entropy()
    if self.use_static_shape or tf.executing_eagerly():
      expected_shape = tf.TensorShape((batch_size,))
    else:
      expected_shape = tf.TensorShape(None)
    self.assertEqual(entropy.shape, expected_shape)
    self.assertAllEqual(self.evaluate(tf.shape(entropy)), (batch_size,))
    self.assertAllClose(
        self.evaluate(entropy),
        [stats.halfcauchy.entropy(loc_, scale_)] * batch_size)

  def testHalfCauchyQuantile(self):
    batch_size = 6
    loc_ = 2.
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale_ = 3.
    scale = self._create_placeholder_with_default(
        [scale_] * batch_size, name='scale')
    half_cauchy = tfd.HalfCauchy(loc, scale, validate_args=True)
    p_ = np.linspace(0.000001, 0.999999, batch_size).astype(self.dtype)
    p = self._create_placeholder_with_default(p_, name='prob')
    quantile = half_cauchy.quantile(p)
    if self.use_static_shape or tf.executing_eagerly():
      expected_shape = tf.TensorShape((batch_size,))
    else:
      expected_shape = tf.TensorShape(None)
    self.assertEqual(quantile.shape, expected_shape)
    self.assertAllEqual(self.evaluate(tf.shape(quantile)), (batch_size,))
    self.assertAllClose(
        self.evaluate(quantile),
        stats.halfcauchy.ppf(p_, loc_, scale_))

  def testHalfCauchySampleMedian(self):
    batch_size = 2
    loc_ = 3.
    scale_ = 1.
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale = self._create_placeholder_with_default(scale_, name='scale')
    n = int(1e5)
    half_cauchy = tfd.HalfCauchy(loc=loc, scale=scale, validate_args=True)
    samples = half_cauchy.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(sample_values.shape, (n, batch_size))
    self.assertAllClose(np.median(sample_values),
                        stats.halfcauchy.median(loc_, scale_),
                        atol=0., rtol=1e-2)

    expected_shape = tf.TensorShape([n]).concatenate(
        tf.TensorShape(self.evaluate(half_cauchy.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, sample_values.shape)

    expected_shape = tf.TensorShape([n]).concatenate(half_cauchy.batch_shape)
    self.assertEqual(expected_shape, samples.shape)

  def testHalfCauchySampleMultidimensionalMedian(self):
    batch_size = 2
    loc_ = [3., -3.]
    scale_ = [0.5, 1.]
    loc = self._create_placeholder_with_default([loc_] * batch_size, name='loc')
    scale = self._create_placeholder_with_default(
        [scale_] * batch_size, name='scale')
    n_ = [int(1e5), 2]
    n = tf.convert_to_tensor(n_, dtype=tf.int32, name='n')
    half_cauchy = tfd.HalfCauchy(loc=loc, scale=scale, validate_args=True)
    samples = half_cauchy.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertAllEqual(sample_values.shape, n_ + [batch_size, 2])
    self.assertAllClose(np.median(sample_values[:, :, 0, 0]),
                        stats.halfcauchy.median(loc_[0], scale_[0]),
                        atol=1e-1)
    self.assertAllClose(np.median(sample_values[:, :, 0, 1]),
                        stats.halfcauchy.median(loc_[1], scale_[1]),
                        atol=1e-1)

    expected_shape = tf.TensorShape(n_).concatenate(
        tf.TensorShape(self.evaluate(half_cauchy.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, sample_values.shape)

    expected_shape = (tf.TensorShape(n_).concatenate(half_cauchy.batch_shape))
    self.assertAllEqual(expected_shape, samples.shape)

  def testHalfCauchyPdfGradientZeroOutsideSupport(self):
    loc_ = [-3.1, -2., 0., 1.1]
    loc = self._create_placeholder_with_default(loc_, name='loc')
    scale = self._create_placeholder_with_default(2., name='scale')
    x = loc - 0.1
    _, grads = self.evaluate(tfp.math.value_and_gradient(
        lambda loc, scale, x: tfd.HalfCauchy(loc, scale).prob(x),
        [loc, scale, x]))
    self.assertAllClose(
        grads,
        [np.zeros_like(loc_), 0., np.zeros_like(loc_)])

    _, grads = self.evaluate(tfp.math.value_and_gradient(
        lambda loc, scale, x: tfd.HalfCauchy(loc, scale).log_prob(x),
        [loc, scale, x]))
    self.assertAllClose(
        grads,
        [np.zeros_like(loc_), 0., np.zeros_like(loc_)])

  def testHalfCauchyCdfGradientZeroOutsideSupport(self):
    loc_ = [-3.1, -2., 0., 1.1]
    loc = self._create_placeholder_with_default(loc_, name='loc')
    scale = self._create_placeholder_with_default(2., name='scale')
    x = loc - 0.1
    _, grads = self.evaluate(tfp.math.value_and_gradient(
        lambda loc, scale, x: tfd.HalfCauchy(loc, scale).cdf(x),
        [loc, scale, x]))
    self.assertAllClose(
        grads,
        [np.zeros_like(loc_), 0., np.zeros_like(loc_)])

    _, grads = self.evaluate(tfp.math.value_and_gradient(
        lambda loc, scale, x: tfd.HalfCauchy(loc, scale).log_cdf(x),
        [loc, scale, x]))
    self.assertAllClose(
        grads,
        [np.zeros_like(loc_), 0., np.zeros_like(loc_)])

  def testHalfCauchyGradientsAndValueFiniteAtLoc(self):
    batch_size = 1000
    loc_ = np.linspace(0., 100., batch_size)
    loc = self._create_placeholder_with_default(loc_, name='loc')
    scale = self._create_placeholder_with_default([1.], name='scale')
    x = self._create_placeholder_with_default(loc_, name='x')
    # log_cdf does not have a finite gradient at `x = loc` and cdf,
    # survival_function, log_survival_function are all computed based on
    # log_cdf. So none of these functions have a finite gradient at `x = loc`.
    for func in [
        lambda loc, scale, x: tfd.HalfCauchy(  # pylint: disable=g-long-lambda
            loc, scale, validate_args=True).prob(x),
        lambda loc, scale, x: tfd.HalfCauchy(  # pylint: disable=g-long-lambda
            loc, scale, validate_args=True).log_prob(x),
    ]:
      value, grads = self.evaluate(
          tfp.math.value_and_gradient(func, [loc, scale, x]))
      self.assertAllFinite(value)
      for grad in grads:
        self.assertAllFinite(grad)

  def testHalfCauchyGradientsAndValueFiniteAtGreaterThanLoc(self):
    batch_size = 1000
    loc = self._create_placeholder_with_default([0.] * batch_size, name='loc')
    scale = self._create_placeholder_with_default([1.], name='scale')
    x_ = np.linspace(1e-3, 100., batch_size)
    x = self._create_placeholder_with_default(x_, name='x')

    def get_half_cauchy_func(func_name):
      def half_cauchy_func(loc, scale, x):
        return getattr(
            tfd.HalfCauchy(loc, scale, validate_args=True), func_name)(
                x)

      return half_cauchy_func

    for func_name in [
        'prob',
        'log_prob',
        'cdf',
        'log_cdf',
        'survival_function',
        'log_survival_function',
    ]:
      func = get_half_cauchy_func(func_name)
      value, grads = self.evaluate(
          tfp.math.value_and_gradient(func, [loc, scale, x]))
      self.assertAllFinite(value)
      for grad in grads:
        self.assertAllFinite(grad)

  def testSupportBijectorOutsideRange(self):
    dist = tfd.HalfCauchy(loc=[-3., 2., 5.4], scale=2., validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to 0'):
      self.evaluate(dist._experimental_default_event_space_bijector().inverse(
          [-4.2, 2. - 1e-6, 5.1]))


@test_util.test_all_tf_execution_regimes
class HalfCauchyTestStaticShapeFloat32(test_util.TestCase, _HalfCauchyTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class HalfCauchyTestDynamicShapeFloat32(test_util.TestCase, _HalfCauchyTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class HalfCauchyTestStaticShapeFloat64(test_util.TestCase, _HalfCauchyTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class HalfCauchyTestDynamicShapeFloat64(test_util.TestCase, _HalfCauchyTest):
  dtype = np.float64
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
