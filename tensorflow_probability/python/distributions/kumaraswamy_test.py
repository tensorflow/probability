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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


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
  # The CDF is 1. - (1 - x ** a) ** b
  # We write this in a numerically stable way.
  return -np.expm1(b * np.log1p(-x ** a))


def _kumaraswamy_pdf(a, b, x):
  a = np.asarray(a)
  b = np.asarray(b)
  x = np.asarray(x)
  return a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)


@test_util.test_all_tf_execution_regimes
class KumaraswamyTest(test_util.TestCase):

  def testSimpleShapes(self):
    a = np.random.rand(3)
    b = np.random.rand(3)
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)

  def testComplexShapes(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(3, 2, 2)
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testComplexShapesBroadcast(self):
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(2, 2)
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 2]), dist.batch_shape)

  def testAProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.assertEqual([1, 3], dist.concentration1.shape)
    self.assertAllClose(a, self.evaluate(dist.concentration1))

  def testBProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.assertEqual([1, 3], dist.concentration0.shape)
    self.assertAllClose(b, self.evaluate(dist.concentration0))

  def testPdfXProper(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError('Sample must be non-negative'):
      self.evaluate(dist.prob([-1., 0.1, 0.5]))
    with self.assertRaisesOpError('Sample must be less than or equal to `1`'):
      self.evaluate(dist.prob([.1, .2, 1.2]))

  def testPdfTwoBatches(self):
    a = [1., 2]
    b = [1., 2]
    x = [.5, .5]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2,), pdf.shape)

  def testPdfTwoBatchesNontrivialX(self):
    a = [1., 2]
    b = [1., 2]
    x = [.3, .7]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2,), pdf.shape)

  def testPdfUniformZeroBatch(self):
    # This is equivalent to a uniform distribution
    a = 1.
    b = 1.
    x = np.array([.1, .2, .3, .5, .8], dtype=np.float32)
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((5,), pdf.shape)

  def testPdfAStretchedInBroadcastWhenSameRank(self):
    a = [[1., 2]]
    b = [[1., 2]]
    x = [[.5, .5], [.3, .7]]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    pdf = dist.prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfAStretchedInBroadcastWhenLowerRank(self):
    a = [1., 2]
    b = [1., 2]
    x = [[.5, .5], [.2, .8]]
    pdf = tfd.Kumaraswamy(a, b, validate_args=True).prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [[.5, .5]]
    pdf = tfd.Kumaraswamy(a, b, validate_args=True).prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))
    self.assertEqual((2, 2), pdf.shape)

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [.5, .5]
    pdf = tfd.Kumaraswamy(a, b, validate_args=True).prob(x)
    expected_pdf = _kumaraswamy_pdf(a, b, x)
    self.assertEqual((2, 2), pdf.shape)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testKumaraswamyMean(self):
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    expected_mean = _kumaraswamy_moment(a, b, 1)
    self.assertEqual((3,), dist.mean().shape)
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

  def testKumaraswamyVariance(self):
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    expected_variance = _kumaraswamy_moment(a, b, 2) - _kumaraswamy_moment(
        a, b, 1)**2
    self.assertEqual((3,), dist.variance().shape)
    self.assertAllClose(expected_variance, self.evaluate(dist.variance()))

  def testKumaraswamyMode(self):
    a = np.array([1.1, 2, 3])
    b = np.array([2., 4, 1.2])
    expected_mode = _kumaraswamy_mode(a, b)
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.assertEqual((3,), dist.mode().shape)
    self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testKumaraswamyModeInvalid(self):
    with tf1.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=False, validate_args=True)
      with self.assertRaisesOpError('Mode undefined for concentration1 <= 1.'):
        self.evaluate(dist.mode())

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=False, validate_args=True)
      with self.assertRaisesOpError('Mode undefined for concentration0 <= 1.'):
        self.evaluate(dist.mode())

  def testKumaraswamyModeEnableAllowNanStats(self):
    with tf1.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=True, validate_args=True)

      expected_mode = _kumaraswamy_mode(a, b)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().shape)
      self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, allow_nan_stats=True, validate_args=True)

      expected_mode = _kumaraswamy_mode(a, b)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().shape)
      self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testKumaraswamyEntropy(self):
    with tf1.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tfd.Kumaraswamy(a, b, validate_args=True)
      self.assertEqual(dist.entropy().shape, (3,))
      expected_entropy = (1 - 1. / b) + (
          1 - 1. / a) * _harmonic_number(b) - np.log(a * b)
      self.assertAllClose(expected_entropy, self.evaluate(dist.entropy()))

  def testKumaraswamySample(self):
    a = 1.
    b = 2.
    kumaraswamy = tfd.Kumaraswamy(a, b, validate_args=True)
    n = tf.constant(100000)
    samples = kumaraswamy.sample(n, seed=test_util.test_seed())
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
    seed = test_util.test_seed()

    tf.random.set_seed(seed)
    kumaraswamy1 = tfd.Kumaraswamy(
        concentration1=a_val,
        concentration0=b_val,
        name='kumaraswamy1',
        validate_args=True)
    samples1 = self.evaluate(kumaraswamy1.sample(n_val, seed=seed))

    tf.random.set_seed(seed)
    kumaraswamy2 = tfd.Kumaraswamy(
        concentration1=a_val,
        concentration0=b_val,
        name='kumaraswamy2',
        validate_args=True)
    samples2 = self.evaluate(kumaraswamy2.sample(n_val, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testKumaraswamySampleMultidimensional(self):
    a = np.random.rand(3, 2, 2).astype(np.float32)
    b = np.random.rand(3, 2, 2).astype(np.float32)
    kumaraswamy = tfd.Kumaraswamy(a, b, validate_args=True)
    n = tf.constant(100000)
    samples = kumaraswamy.sample(n, seed=test_util.test_seed())
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
      actual = self.evaluate(tfd.Kumaraswamy(a, b, validate_args=True).cdf(x))
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
      self.assertAllClose(_kumaraswamy_cdf(a, b, x), actual)

  def testKumaraswamyLogCdf(self):
    shape = (30, 40, 50)
    for dt in (np.float32, np.float64):
      a = 10. * np.random.random(shape).astype(dt)
      b = 10. * np.random.random(shape).astype(dt)
      x = np.random.random(shape).astype(dt)
      actual = self.evaluate(
          tf.exp(tfd.Kumaraswamy(a, b, validate_args=True).log_cdf(x)))
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
      self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
      self.assertAllClose(_kumaraswamy_cdf(a, b, x), actual)

  def testPdfAtBoundary(self):
    a = [0.5, 2.]
    b = [0.5, 5.]
    x = [[0.], [1.]]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    pdf = self.evaluate(dist.prob(x))
    log_pdf = self.evaluate(dist.log_prob(x))
    self.assertAllPositiveInf(pdf[:, 0])
    self.assertAllFinite(pdf[:, 1])
    self.assertAllPositiveInf(log_pdf[:, 0])
    self.assertAllNegativeInf(log_pdf[:, 1])

  def testAssertValidSample(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(dist.prob([-1., 0.1, 0.5]))
    with self.assertRaisesOpError('Sample must be less than or equal to `1`.'):
      self.evaluate(dist.prob([.1, .2, 1.2]))

  def testInvalidConcentration1(self):
    x = tf.Variable(1.)
    dist = tfd.Kumaraswamy(
        concentration0=1., concentration1=x, validate_args=True)
    self.evaluate(x.initializer)
    self.assertIs(x, dist.concentration1)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    with self.assertRaisesOpError(
        'Argument `concentration1` must be positive.'):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(dist.event_shape_tensor())

  def testInvalidConcentration0(self):
    x = tf.Variable(1.)
    dist = tfd.Kumaraswamy(
        concentration0=x, concentration1=1., validate_args=True)
    self.evaluate(x.initializer)
    self.assertIs(x, dist.concentration0)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    with self.assertRaisesOpError(
        'Argument `concentration0` must be positive.'):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(dist.event_shape_tensor())

  def testSupportBijectorOutsideRange(self):
    a = np.array([1., 2., 3.])
    b = np.array([2., 4., 1.2])
    dist = tfd.Kumaraswamy(a, b, validate_args=True)
    eps = 1e-6
    x = np.array([-2.3, -eps, 1. + eps, 1.4])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  tf.test.main()
