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
"""Tests for GEV."""

# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gev
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


class _GEVTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self._dtype)
    return tf1.placeholder_with_default(
        x, shape=x.shape if self._use_static_shape else None)

  def testGEVShape(self):
    loc = np.array([3.0] * 5, dtype=self._dtype)
    scale = np.array([3.0] * 5, dtype=self._dtype)
    conc = np.array([3.0] * 5, dtype=self._dtype)
    dist = gev.GeneralizedExtremeValue(
        loc=loc, scale=scale, concentration=conc, validate_args=True)

    self.assertEqual((5,), self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([5]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testInvalidScale(self):
    scale = [-.01, 0., 2.]
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      dist = gev.GeneralizedExtremeValue(
          loc=0., scale=scale, concentration=1., validate_args=True)
      self.evaluate(dist.mean())

    scale = tf.Variable([.01])
    self.evaluate(scale.initializer)
    dist = gev.GeneralizedExtremeValue(
        loc=0., scale=scale, concentration=1., validate_args=True)
    self.assertIs(scale, dist.scale)
    self.evaluate(dist.mean())
    with tf.control_dependencies([scale.assign([-.01])]):
      with self.assertRaisesOpError('Argument `scale` must be positive.'):
        self.evaluate(dist.mean())

  def testGEVLogPdf(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    conc = np.array([2.] * batch_size, dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)
    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)
    log_pdf = dist.log_prob(self.make_tensor(x))
    self.assertAllClose(
        gev_dist.logpdf(x),
        self.evaluate(log_pdf))

    pdf = dist.prob(x)
    self.assertAllClose(
        gev_dist.pdf(x), self.evaluate(pdf))

  def testGEVLogPdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[-2.0, -4.0, -5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    conc = np.array([[0.0, 1.0, 2.0]] * batch_size, dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)
    log_pdf = dist.log_prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_pdf), gev_dist.logpdf(x))

    pdf = dist.prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(pdf), gev_dist.pdf(x))

  def testGEVCDF(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    conc = np.array([2.] * batch_size, dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    log_cdf = dist.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf), gev_dist.logcdf(x))

    cdf = dist.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf), gev_dist.cdf(x))

  def testGEVCdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[-2.0, -4.0, -5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    conc = np.array([[0.0, 1.0, 2.0]] * batch_size, dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    log_cdf = dist.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
        gev_dist.logcdf(x))

    cdf = dist.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf),
        gev_dist.cdf(x))

  def testGEVMean(self):
    loc = np.array([2.0], dtype=self._dtype)
    scale = np.array([1.5], dtype=self._dtype)
    conc = np.array([-0.9, 0.0], dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)
    self.assertAllClose(self.evaluate(dist.mean()), gev_dist.mean())

    conc_with_inf_mean = np.array([2.], dtype=self._dtype)
    gev_with_inf_mean = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc_with_inf_mean),
        validate_args=True)
    self.assertAllClose(self.evaluate(gev_with_inf_mean.mean()),
                        [np.inf])

  def testGEVVariance(self):
    loc = np.array([2.0], dtype=self._dtype)
    scale = np.array([1.5], dtype=self._dtype)
    conc = np.array([-0.9, 0.0], dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    self.assertAllClose(self.evaluate(dist.variance()), gev_dist.var())

    conc_with_inf_var = np.array([1.5], dtype=self._dtype)
    gev_with_inf_var = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc_with_inf_var),
        validate_args=True)
    self.assertAllClose(self.evaluate(gev_with_inf_var.variance()),
                        [np.inf])

  def testGEVStd(self):
    loc = np.array([2.0], dtype=self._dtype)
    scale = np.array([1.5], dtype=self._dtype)
    conc = np.array([-0.9, 0.0], dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    self.assertAllClose(self.evaluate(dist.stddev()), gev_dist.std())

    conc_with_inf_std = np.array([1.5], dtype=self._dtype)
    gev_with_inf_std = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc_with_inf_std),
        validate_args=True)
    self.assertAllClose(self.evaluate(gev_with_inf_std.stddev()),
                        [np.inf])

  def testGEVMode(self):
    loc = np.array([2.0], dtype=self._dtype)
    scale = np.array([1.5], dtype=self._dtype)
    conc = np.array([-0.9, 0.0, 1.5], dtype=self._dtype)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    np_mode_z = np.where(conc == 0., 0., ((conc+1)**(-conc) - 1.) / conc)
    np_mode = loc + np_mode_z * scale
    self.assertAllClose(self.evaluate(dist.mode()), np_mode)

  def testGEVSample(self):
    loc = self._dtype(4.0)
    scale = self._dtype(1.0)
    conc = self._dtype(0.2)
    n = int(1e6)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n,), sample_values.shape)
    self.assertAllClose(
        gev_dist.mean(),
        sample_values.mean(), rtol=.01)
    self.assertAllClose(
        gev_dist.var(),
        sample_values.var(), rtol=.01)

  def testGEVSampleMultidimensionalMean(self):
    loc = np.array([2.0, 4.0, 5.0], dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    conc = np.array([0.2], dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)
    n = int(1e6)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        gev_dist.mean(),
        sample_values.mean(axis=0),
        rtol=.03,
        atol=0)

  def testGEVSampleMultidimensionalVar(self):
    loc = np.array([2.0, 4.0, 5.0], dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    conc = np.array([0.2], dtype=self._dtype)
    gev_dist = stats.genextreme(-conc, loc=loc, scale=scale)
    n = int(1e6)

    dist = gev.GeneralizedExtremeValue(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        concentration=self.make_tensor(conc),
        validate_args=True)

    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        gev_dist.var(),
        sample_values.var(axis=0),
        rtol=.03,
        atol=0)

  @test_util.numpy_disable_gradient_test
  def testFiniteGradientAtDifficultPoints(self):
    def make_fn(dtype, attr):
      x = np.array([1.]).astype(dtype)
      return lambda m, s, p: getattr(  # pylint: disable=g-long-lambda
          gev.GeneralizedExtremeValue(
              loc=m, scale=s, concentration=p, validate_args=True), attr)(
                  x)

    loc = np.array([1.0], dtype=self._dtype)
    scale = np.array([1.5], dtype=self._dtype)
    conc = np.array([-0.7, 0.0, 0.5, 1.], dtype=self._dtype)

    for attr in ['log_prob', 'prob', 'cdf', 'log_cdf']:
      value, grads = self.evaluate(
          gradient.value_and_gradient(
              make_fn(self._dtype, attr),
              [
                  self.make_tensor(loc),  # loc
                  self.make_tensor(scale),  # scale
                  self.make_tensor(conc)
              ]))  # conc
      self.assertAllFinite(value)
      self.assertAllFinite(grads[0])  # d/d loc
      self.assertAllFinite(grads[1])  # d/d scale
      self.assertAllFinite(grads[2])  # d/d conc

  def testBroadcastingParams(self):

    def _check(gev_dist):
      self.assertEqual(gev_dist.mean().shape, (3,))
      self.assertEqual(gev_dist.variance().shape, (3,))
      self.assertEqual(gev_dist.entropy().shape, (3,))
      self.assertEqual(gev_dist.log_prob(6.).shape, (3,))
      self.assertEqual(gev_dist.prob(6.).shape, (3,))
      self.assertEqual(gev_dist.sample(
          37, seed=test_util.test_seed()).shape, (37, 3,))

    _check(
        gev.GeneralizedExtremeValue(
            loc=[
                2.,
                3.,
                4.,
            ], scale=2., concentration=1., validate_args=True))
    _check(
        gev.GeneralizedExtremeValue(
            loc=3., scale=[
                2.,
                3.,
                4.,
            ], concentration=1., validate_args=True))
    _check(
        gev.GeneralizedExtremeValue(
            loc=3., scale=3., concentration=[
                2.,
                3.,
                4.,
            ], validate_args=True))

  def testBroadcastingPdfArgs(self):

    def _assert_shape(gev_dist, arg, shape):
      self.assertEqual(gev_dist.log_prob(arg).shape, shape)
      self.assertEqual(gev_dist.prob(arg).shape, shape)

    def _check(gev_dist):
      _assert_shape(gev_dist, 5., (3,))
      xs = np.array([5., 6., 7.], dtype=np.float32)
      _assert_shape(gev_dist, xs, (3,))
      xs = np.array([xs])
      _assert_shape(gev_dist, xs, (1, 3))
      xs = xs.T
      _assert_shape(gev_dist, xs, (3, 3))

    _check(
        gev.GeneralizedExtremeValue(
            loc=[
                -2.,
                -3.,
                -4.,
            ],
            scale=2.,
            concentration=1.,
            validate_args=True))
    _check(
        gev.GeneralizedExtremeValue(
            loc=-6., scale=[
                2.,
                3.,
                4.,
            ], concentration=1., validate_args=True))
    _check(
        gev.GeneralizedExtremeValue(
            loc=-7., scale=3., concentration=[
                2.,
                3.,
                4.,
            ], validate_args=True))

    def _check2d(gev_dist):
      _assert_shape(gev_dist, 5., (1, 3))
      xs = np.array([5., 6., 7.], dtype=np.float32)
      _assert_shape(gev_dist, xs, (1, 3))
      xs = np.array([xs])
      _assert_shape(gev_dist, xs, (1, 3))
      xs = xs.T
      _assert_shape(gev_dist, xs, (3, 3))

    _check2d(
        gev.GeneralizedExtremeValue(
            loc=[[
                -2.,
                -3.,
                -4.,
            ]],
            scale=2.,
            concentration=1.,
            validate_args=True))
    _check2d(
        gev.GeneralizedExtremeValue(
            loc=-7.,
            scale=[[
                2.,
                3.,
                4.,
            ]],
            concentration=1.,
            validate_args=True))
    _check2d(
        gev.GeneralizedExtremeValue(
            loc=-7.,
            scale=3.,
            concentration=[[
                2.,
                3.,
                4.,
            ]],
            validate_args=True))

    def _check2d_rows(gev_dist):
      _assert_shape(gev_dist, 5., (3, 1))
      xs = np.array([5., 6., 7.], dtype=np.float32)  # (3,)
      _assert_shape(gev_dist, xs, (3, 3))
      xs = np.array([xs])  # (1,3)
      _assert_shape(gev_dist, xs, (3, 3))
      xs = xs.T  # (3,1)
      _assert_shape(gev_dist, xs, (3, 1))

    _check2d_rows(
        gev.GeneralizedExtremeValue(
            loc=[[-2.], [-3.], [-4.]],
            scale=2.,
            concentration=1.,
            validate_args=True))
    _check2d_rows(
        gev.GeneralizedExtremeValue(
            loc=-7.,
            scale=[[2.], [3.], [4.]],
            concentration=1.,
            validate_args=True))
    _check2d_rows(
        gev.GeneralizedExtremeValue(
            loc=-7.,
            scale=3.,
            concentration=[[2.], [3.], [4.]],
            validate_args=True))


@test_util.test_all_tf_execution_regimes
class GEVTestStaticShape(test_util.TestCase, _GEVTest):
  _dtype = np.float32
  _use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GEVTestFloat64StaticShape(test_util.TestCase, _GEVTest):
  _dtype = np.float64
  _use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GEVTestDynamicShape(test_util.TestCase, _GEVTest):
  _dtype = np.float32
  _use_static_shape = False


if __name__ == '__main__':
  test_util.main()
