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
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


tfd = tfp.distributions


def _kumaraswamy_mode(a, b):
  a = np.asarray(a)
  b = np.asarray(b)
  return ((a - 1) / (a * b - 1))**(1 / a)


def _kumaraswamy_moment(a, b, n):
  a = np.asarray(a)
  b = np.asarray(b)
  return b * sp_special.beta(1.0 + n / a, b)


def _harmonic_number(b):
  b = np.asarray(b)
  return sp_special.psi(b + 1) - sp_special.psi(1)


def _kumaraswamy_cdf(a, b, x):
  a = np.asarray(a)
  b = np.asarray(b)
  x = np.asarray(x)
  return 1 - (1 - x**a)**b


def _kumaraswamy_pdf(a, b, x):
  a = np.asarray(a)
  b = np.asarray(b)
  x = np.asarray(x)
  return a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)


@test_util.run_all_in_graph_and_eager_modes
class KumaraswamyTest(tf.test.TestCase):

  def testSimpleShapes(self):
    a = np.random.rand(3)
    b = np.random.rand(3)
    dist = tfd.Kumaraswamy(a, b)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)

  def testComplexShapes(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(3, 2, 2)
    dist = tfd.Kumaraswamy(a, b)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testComplexShapesBroadcast(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(2, 2)
    dist = tfd.Kumaraswamy(a, b)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testAProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Kumaraswamy(a, b)
    self.assertEqual([1, 3], dist.concentration1.shape)
    self.assertAllClose(a, self.evaluate(dist.concentration1))

  def testBProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Kumaraswamy(a, b)
    self.assertEqual([1, 3], dist.concentration0.shape)
    self.assertAllClose(b, self.evaluate(dist.concentration0))

  def testPdfXProper(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError("sample must be non-negative"):
      self.evaluate(dist.prob([-1., 0.1, 0.5]))
    with self.assertRaisesOpError("sample must be no larger than `1`"):
      self.evaluate(dist.prob([.1, .2, 1.2]))

  def testPdfTwoBatches(self):
    a = [1., 2]
    b = [1., 2]
    x = [.5, .5]
    dist = tfd.Kumaraswamy(a, b)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2,), pdf.shape)

  def testPdfTwoBatchesNontrivialX(self):
    a = [1., 2]
    b = [1., 2]
    x = [.3, .7]
    dist = tfd.Kumaraswamy(a, b)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2,), pdf.shape)

  def testPdfUniformZeroBatch(self):
    # This is equivalent to a uniform distribution
    a = 1.
    b = 1.
    x = np.array([.1, .2, .3, .5, .8], dtype=np.float32)
    dist = tfd.Kumaraswamy(a, b)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((5,), pdf.shape)

  def testPdfAStretchedInBroadcastWhenSameRank(self):
    a = [[1., 2]]
    b = [[1., 2]]
    x = [[.5, .5], [.3, .7]]
    dist = tfd.Kumaraswamy(a, b)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfAStretchedInBroadcastWhenLowerRank(self):
    a = [1., 2]
    b = [1., 2]
    x = [[.5, .5], [.2, .8]]
    pdf = tfd.Kumaraswamy(a, b).prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [[.5, .5]]
    pdf = tfd.Kumaraswamy(a, b).prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [.5, .5]
    pdf = tfd.Kumaraswamy(a, b).prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testKumaraswamyMean(self):
    with tf.compat.v1.Session():
      a = [1., 2, 3]
      b = [2., 4, 1.2]
      dist = tfd.Kumaraswamy(a, b)
      self.assertEqual(dist.mean().shape, (3,))
      expected_mean = _kumaraswamy_moment(a, b, 1)
      self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

  def testKumaraswamyVariance(self):
    with tf.compat.v1.Session():
      a = [1., 2, 3]
      b = [2., 4, 1.2]
      dist = tfd.Kumaraswamy(a, b)
      self.assertEqual(dist.variance().shape, (3,))
      expected_variance = _kumaraswamy_moment(a, b, 2) - _kumaraswamy_moment(
          a, b, 1)**2
      self.assertAllClose(expected_variance, self.evaluate(dist.variance()))

  def testKumaraswamyMode(self):
    with tf.compat.v1.Session():
      a = np.array([1.1, 2, 3])
      b = np.array([2., 4, 1.2])
      expected_mode = _kumaraswamy_mode(a, b)
      dist = tfd.Kumaraswamy(a, b)
      self.assertEqual(dist.mode().shape, (3,))
      self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testKumaraswamyModeInvalid(self):
    with tf.compat.v1.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=False)
      with self.assertRaisesOpError("Mode undefined for concentration1 <= 1."):
        self.evaluate(dist.mode())

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=False)
      with self.assertRaisesOpError("Mode undefined for concentration0 <= 1."):
        self.evaluate(dist.mode())

  def testKumaraswamyModeEnableAllowNanStats(self):
    with tf.compat.v1.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=True)

      expected_mode = _kumaraswamy_mode(a, b)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().shape)
      self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=True)

      expected_mode = _kumaraswamy_mode(a, b)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().shape)
      self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testKumaraswamyEntropy(self):
    with tf.compat.v1.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b)
      self.assertEqual(dist.entropy().shape, (3,))
      expected_entropy = (1 - 1. / b) + (
          1 - 1. / a) * _harmonic_number(b) - np.log(a * b)
      self.assertAllClose(expected_entropy, self.evaluate(dist.entropy()))

  def testKumaraswamySample(self):
    a = 1.
    b = 2.
    kumaraswamy = tfd.Kumaraswamy(a, b)
    n = tf.constant(100000)
    samples = kumaraswamy.sample(n)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000,))
    self.assertFalse(np.any(sample_values < 0.0))
    self.assertLess(
        sp_stats.kstest(
            # Kumaraswamy is a univariate distribution.
            sample_values,
            lambda x: _kumaraswamy_cdf(1., 2., x))[0],
        0.01)
    # The standard error of the sample mean is 1 / (sqrt(18 * n))
    expected_mean = _kumaraswamy_moment(a, b, 1)
    self.assertAllClose(sample_values.mean(axis=0), expected_mean, atol=1e-2)
    expected_variance = _kumaraswamy_moment(a, b, 2) - _kumaraswamy_moment(
        a, b, 1)**2
    self.assertAllClose(
        np.cov(sample_values, rowvar=0), expected_variance, atol=1e-1)

  # Test that sampling with the same seed twice gives the same results.
  def testKumaraswamySampleMultipleTimes(self):
    a_val = 1.
    b_val = 2.
    n_val = 100
    seed = tfp_test_util.test_seed()

    tf.compat.v1.set_random_seed(seed)
    kumaraswamy1 = tfd.Kumaraswamy(
        concentration1=a_val, concentration0=b_val, name="kumaraswamy1")
    samples1 = self.evaluate(kumaraswamy1.sample(n_val, seed=seed))

    tf.compat.v1.set_random_seed(seed)
    kumaraswamy2 = tfd.Kumaraswamy(
        concentration1=a_val, concentration0=b_val, name="kumaraswamy2")
    samples2 = self.evaluate(kumaraswamy2.sample(n_val, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testKumaraswamySampleMultidimensional(self):
    a = np.random.rand(3, 2, 2).astype(np.float32)
    b = np.random.rand(3, 2, 2).astype(np.float32)
    kumaraswamy = tfd.Kumaraswamy(a, b)
    n = tf.constant(100000)
    samples = kumaraswamy.sample(n)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 3, 2, 2))
    self.assertFalse(np.any(sample_values < 0.0))
    self.assertAllClose(
        sample_values[:, 1, :].mean(axis=0),
        _kumaraswamy_moment(a, b, 1)[1, :],
        atol=1e-1)

  def testKumaraswamyCdf(self):
    shape = (30, 40, 50)
    for dt in (np.float32, np.float64):
      a = 10. * np.random.random(shape).astype(dt)
      b = 10. * np.random.random(shape).astype(dt)
      x = np.random.random(shape).astype(dt)
      actual = self.evaluate(tfd.Kumaraswamy(a, b).cdf(x))
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
      self.assertAllClose(_kumaraswamy_cdf(a, b, x), actual, rtol=1e-4, atol=0)

  def testKumaraswamyLogCdf(self):
    shape = (30, 40, 50)
    for dt in (np.float32, np.float64):
      a = 10. * np.random.random(shape).astype(dt)
      b = 10. * np.random.random(shape).astype(dt)
      x = np.random.random(shape).astype(dt)
      actual = self.evaluate(tf.exp(tfd.Kumaraswamy(a, b).log_cdf(x)))
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
      self.assertAllClose(_kumaraswamy_cdf(a, b, x), actual, rtol=1e-4, atol=0)


if __name__ == "__main__":
  tf.test.main()
