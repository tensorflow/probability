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
"""Tests for Bates distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fractions
import math
import sys

# Dependency imports
from absl.testing import parameterized
import numpy as np
import scipy
import scipy.integrate
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import bates
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class BatesTest(test_util.TestCase):

  def testBatesParamsNoBatch(self):
    n = 8.
    l = -11.
    h = -5.
    b = tfd.Bates(total_count=n, low=l, high=h, validate_args=True)
    self.assertAllClose(n, self.evaluate(b.total_count))
    self.assertAllClose(l, self.evaluate(b.low))
    self.assertAllClose(h, self.evaluate(b.high))
    self.assertAllEqual([], self.evaluate(b.batch_shape_tensor()))

  def testBatesParamsBatch(self):
    n = tf.ones((2, 1, 3), dtype=tf.float32)
    l = tf.zeros((2, 2, 1), dtype=tf.float32)
    h = tf.constant(3, dtype=tf.float32)
    b = tfd.Bates(total_count=n, low=l, high=h, validate_args=True)
    self.assertAllClose(n, self.evaluate(b.total_count))
    self.assertAllClose(l, self.evaluate(b.low))
    self.assertAllClose(h, self.evaluate(b.high))
    self.assertAllEqual([2, 2, 3], self.evaluate(b.batch_shape_tensor()))

  def testBatesInvalidShapes(self):
    n = np.ones((2, 3))
    l = np.zeros((2, 2))
    with self.assertRaisesRegex(
        ValueError,
        'Arguments `total_count`, `low` and `high` must have compatible shapes'
    ):
      d = tfd.Bates(total_count=n, low=l, validate_args=True)
      self.evaluate(d.prob(1.))

  def testBatesNonNegTotalCount(self):
    ns = [0., -1.]
    for n in ns:
      with self.assertRaisesOpError(
          '`total_count` must be positive.'):
        d = tfd.Bates(total_count=n, validate_args=True)
        self.evaluate(d.prob(1.))

  def testBatesIntegralTotalCount(self):
    with self.assertRaisesOpError('`total_count` must be integer-valued.'):
      d = tfd.Bates(total_count=1.5, validate_args=True)
      self.evaluate(d.prob(1.))

  def testBatesStableTotalCount(self):
    bad = max(bates.BATES_TOTAL_COUNT_STABILITY_LIMITS.values()) + 10.
    with self.assertRaisesOpError('`total_count` > .* is numerically unstable'):
      d = tfd.Bates(total_count=bad, validate_args=True)
      self.evaluate(d.prob(1.))

  # TODO(b/157665671): Figure out a way to capture output in all modes.
  @test_util.test_graph_mode_only
  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='Unable to capture stderr in numpy / JAX tests.')
  def testBatesStableTotalCountWarning(self):
    bad = max(bates.BATES_TOTAL_COUNT_STABILITY_LIMITS.values()) + 10.
    d = tfd.Bates(total_count=bad, validate_args=False)
    with self.captureWritesToStream(sys.stderr) as captured:
      self.evaluate(d.prob(1.))
    self.assertRegex(
        captured.contents(),
        'Bates PDF/CDF is unstable for `total_count` >')
    with self.captureWritesToStream(sys.stderr) as captured:
      self.evaluate(d.cdf(1.))
    self.assertRegex(
        captured.contents(),
        'Bates PDF/CDF is unstable for `total_count` >')

  def testBatesLowLtHigh(self):
    bounds = [(-1., -1.), (0., 0.), (-1., -1.1), (1.1, 1.)]
    for l, h in bounds:
      with self.assertRaisesOpError('`low` must be less than `high`'):
        d = tfd.Bates(total_count=1, low=l, high=h, validate_args=True)
        self.evaluate(d.prob(1.))

  def testBatesVariables(self):
    n0 = np.array([1., 2.])
    l0 = np.array([0., 1.])
    h0 = np.array([1., 11.])
    n = tf.Variable(n0)
    l = tf.Variable(l0)
    h = tf.Variable(h0)
    d = tfd.Bates(total_count=n, low=l, high=h, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.evaluate(d.prob([.5, 1.]))

    self.evaluate(n.assign(-n0))
    with self.assertRaisesOpError('`total_count` must be positive.'):
      self.evaluate(d.prob([.5, 1.]))
    self.evaluate(n.assign(n0))
    self.evaluate(d.prob([.5, 1.]))

    self.evaluate(n.assign(n0 / 2.))
    with self.assertRaisesOpError('`total_count` must be integer-valued.'):
      self.evaluate(d.prob([.5, 1.]))
    self.evaluate(n.assign(n0))
    self.evaluate(d.prob([.5, 1.]))

    self.evaluate(n.assign([1000., 2000.]))
    with self.assertRaisesOpError(
        '`total_count` > .* is numerically unstable.'):
      self.evaluate(d.prob([.5, 1.]))
    self.evaluate(n.assign(n0))
    self.evaluate(d.prob([.5, 1.]))

    self.evaluate(l.assign(h0))
    with self.assertRaisesOpError('`low` must be less than `high`'):
      self.evaluate(d.prob([.5, 1.]))
    self.evaluate(l.assign(l0))
    self.evaluate(d.prob([.5, 1.]))

  def shapeless(self, val):
    var = tf.Variable(val, shape=tf.TensorShape(None))
    self.evaluate(tf1.global_variables_initializer())
    return var

  def make_shapeless_bates(self, total_count, low, high):
    return tfd.Bates(total_count=self.shapeless(total_count),
                     low=self.shapeless(low),
                     high=self.shapeless(high))

  def testBatesDynamicShapeTensor(self):
    dist = self.make_shapeless_bates(1., 0., [0.5, 1.])
    self.assertAllEqual([2], dist.batch_shape_tensor())
    dist = self.make_shapeless_bates(1., 0., [0.5, 1.])
    self.evaluate(dist.sample(self.shapeless(10), seed=test_util.test_seed()))
    dist = self.make_shapeless_bates(1., 0., [0.5, 1.])
    self.assertAllEqual(self.evaluate(dist.prob(self.shapeless([-1., 2.]))),
                        [0., 0.])
    dist = self.make_shapeless_bates(1., 0., [0.5, 1.])
    self.assertAllEqual(self.evaluate(dist.cdf(self.shapeless([-1., 2.]))),
                        [0., 1.])

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='Shapeless Variables not supported in numpy / JAX.')
  def testBatesDynamicShape(self):
    dist = self.make_shapeless_bates(1., 0., [0.5, 1.])
    self.assertEqual(tf.TensorShape(None), dist.batch_shape)

  @test_util.numpy_disable_test_missing_functionality('tf.ragged.range')
  @test_util.jax_disable_test_missing_functionality('tf.ragged.range')
  def testSegmentedRange(self):
    # TODO(b/157666350): Turn `testSegmentedRange` into a hypothesis test.
    limits = np.array([3, 1, 4, 5, 4, 3, 10, 1])
    self.assertAllEqual(
        self.evaluate(tf.ragged.range(limits).flat_values),
        bates._segmented_range(limits))

  def testBatesPDFLowTotalCount(self):
    ns = np.array([1., 2.])
    lss = np.array([[0., -1.], [-10., -1.]])
    hss = np.array([[1., 3.], [-9., 0.]])
    b = tfd.Bates(
        total_count=tf.reshape(ns, (2, 1, 1)), low=lss, high=hss,
        validate_args=True)
    self.assertAllEqual([2, 2, 2], self.evaluate(b.batch_shape_tensor()))
    xs = np.array([0., .25, .5, 1.1, 1.5, 2.])
    probs = b.prob(tf.reshape(xs, (6, 1, 1, 1)))
    self.assertAllEqual([6, 2, 2, 2], self.evaluate(tf.shape(probs)))

    def expected_pdf(n, l, h, x):
      if n == 1:
        left = right = 1. / (h - l)
      elif n == 2:
        left = np.power(2 / (h - l), 2) * (x - l)
        right = np.power(2 / (h - l), 2) * (h - x)
      else:
        raise ValueError('Compute your own damn pdfs')
      return np.where(x < l, 0, np.where(x > h, 0, np.where(
          x < (l + h) / 2., left, right)))

    expected = [[[[expected_pdf(n, l, h, x) for l, h in zip(ls, hs)]
                  for ls, hs in zip(lss, hss)]
                 for n in ns] for x in xs]
    self.assertAllClose(expected, self.evaluate(probs))

  def testBatesPDFHighTotalCount(self):
    # Compute with exact integer arithmetic.
    def exact(n, nx):
      fractional = sum(
          fractions.Fraction((-1)**k * (nx - k)**(n-1) * math.factorial(n),
                             math.factorial(n-k) * math.factorial(k))
          for k in range(nx+1)) * fractions.Fraction(n, math.factorial(n-1))
      return fractional.numerator / fractional.denominator

    tests = [
        (48, .25), (48, .5), (48, .75),
        (50, 0.02), (50, .48), (50, .52), (50, .98)]
    for n, x in tests:
      nx_ = n*x
      nx = int(nx_)
      self.assertAllClose(nx_, nx)

      b = tfd.Bates(total_count=n, low=tf.cast(0, tf.float64))
      val = b.prob(tf.cast(x, tf.float64))
      self.assertAllEqual([], self.evaluate(tf.shape(val)))
      self.assertAllClose(self.evaluate(val), exact(n, nx))

  @parameterized.parameters(
      *((total_count, bounds)  # pylint: disable=g-complex-comprehension
        for total_count in [1., 2., 3., 10.]
        for bounds in [(-20., 0.1), (-0.1, 1.), (0., 20.)])
  )
  def testBatesPDFisNormalized(self, total_count, bounds):
    low, high = tf.cast(bounds[0], tf.float64), tf.cast(bounds[1], tf.float64)
    d = tfd.Bates(total_count=total_count, low=low, high=high)
    # This is about as high as JAX can go and still finish in time.
    nx = 100
    x = tf.linspace(low, high, nx)
    y = self.evaluate(d.prob(x))
    dx = self.evaluate(x[1] - x[0])
    self.assertAllClose(scipy.integrate.simps(y=y, dx=dx), 1.,
                        atol=5e-05, rtol=5e-05)

  def testBatesPDFonNaNs(self):
    b = tfd.Bates(1, 0, 1)
    values_with_nans = [
        np.nan, -1., np.nan, 0., np.nan, .5, np.nan, 1., np.nan, 2., np.nan]
    values = [v if i % 2 != 0 else 0. for i, v in enumerate(values_with_nans)]
    probs = self.evaluate(b.log_prob(values))
    probs_with_nans = self.evaluate(b.log_prob(values_with_nans))
    should_be_nan = [probs_with_nans[i]
                     for i, v in enumerate(values_with_nans)
                     if np.isnan(v)]
    self.assertAllNan(should_be_nan)
    lhs = [probs[i] for i, v in enumerate(values_with_nans)
           if not np.isnan(v)]
    rhs = [probs_with_nans[i] for i, v in enumerate(values_with_nans)
           if not np.isnan(v)]
    self.assertAllEqual(lhs, rhs)

  def testBatesCDFLowTotalCount(self):
    ns = np.array([1., 2.])
    ls = np.array([0., 1.])
    hs = np.array([1., 3.])
    b = tfd.Bates(
        total_count=tf.expand_dims(ns, -1), low=ls, high=hs, validate_args=True)
    self.assertAllEqual([2, 2], self.evaluate(b.batch_shape_tensor()))
    xs = np.array([0., .25, .5, 1.1, 1.5, 2.])
    cdfs = b.cdf(tf.reshape(xs, (6, 1, 1)))
    self.assertAllEqual([6, 2, 2], self.evaluate(tf.shape(cdfs)))

    def expected_cdf(n, l, h, x):
      if n == 1:
        left = right = (x - l) / (h - l)
      elif n == 2:
        left = 2 * np.power((x - l) / (h - l), 2)
        right = 1 - 2 * np.power((h - x) / (h - l), 2)
      else:
        raise ValueError('Compute your own damn cdfs')
      return np.where(x < l, 0, np.where(x > h, 1, np.where(
          x < (l + h) / 2., left, right)))

    expected = [[[expected_cdf(n, l, h, x) for l, h in zip(ls, hs)]
                 for n in ns]
                for x in xs]
    self.assertAllClose(expected, self.evaluate(cdfs))

  def testBatesCDFHighTotalCount(self):
    # Compute with exact integer arithmetic.
    def exact(n, nx):
      fractional = sum(
          fractions.Fraction((-1)**k * (nx - k)**n * math.factorial(n),
                             math.factorial(n-k) * math.factorial(k))
          for k in range(nx+1)) * fractions.Fraction(1, math.factorial(n))
      return fractional.numerator / fractional.denominator

    tests = [
        (48, .25), (48, .5), (48, .75),
        (50, 0.02), (50, .48), (50, .52), (50, .98)]
    for n, x in tests:
      nx_ = n*x
      nx = int(nx_)
      self.assertAllClose(nx_, nx)

      b = tfd.Bates(total_count=n, low=tf.cast(0, tf.float64))
      val = b.cdf(tf.cast(x, tf.float64))
      self.assertAllEqual([], self.evaluate(tf.shape(val)))
      self.assertAllClose(self.evaluate(val), exact(n, nx))

  def testBatesMean(self):
    # TODO(b/157666350): Turn this into a hypothesis test.
    bounds = np.array([[0., 1.], [1., 2.], [-2., -1.], [10., 20.]])
    b = tfd.Bates(total_count=10., low=bounds[..., 0], high=bounds[..., 1],
                  validate_args=True)
    self.assertAllClose(
        bounds.mean(1),
        self.evaluate(b.mean()))

  def testBatesVariance(self):
    ns = np.array([1., 2., 10.])
    lss = np.array([[-10., -2.], [-10., 0.]])
    hss = np.array([[-1., 0.], [10., 100.]])
    b = tfd.Bates(total_count=tf.reshape(ns, (3, 1, 1)), low=lss, high=hss,
                  validate_args=True)
    expected = [[[np.power(h - l, 2) / (12 * n) for l, h in zip(ls, hs)]
                 for ls, hs in zip(lss, hss)]
                for n in ns]
    self.assertAllClose(
        self.evaluate(b.variance()),
        expected)

  def testBatesSampleStatistics(self):
    # TODO(b/157666350): Turn this into a hypothesis test.
    bounds = np.array([[0., 1.], [1., 2.], [-2., -1.], [10., 20.]])
    b = tfd.Bates(total_count=10., low=bounds[..., 0], high=bounds[..., 1],
                  validate_args=True)
    samples = b.sample(1e6, seed=test_util.test_seed())
    self.assertAllClose(
        self.evaluate(b.mean()),
        np.mean(self.evaluate(samples), axis=0),
        atol=1e-03, rtol=1e-03
    )
    self.assertAllClose(
        self.evaluate(b.variance()),
        np.var(self.evaluate(samples), axis=0),
        atol=1e-03, rtol=1e-03
    )


if __name__ == '__main__':
  tf.test.main()
