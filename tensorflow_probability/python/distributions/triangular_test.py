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

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


class _TriangularTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self._dtype)
    return tf1.placeholder_with_default(
        x, shape=x.shape if self._use_static_shape else None)

  def _create_triangular_dist(self, low, high, peak):
    return tfd.Triangular(
        low=self.make_tensor(low),
        high=self.make_tensor(high),
        peak=self.make_tensor(peak),
        validate_args=True)

  def _scipy_triangular(self, low, high, peak):
    # Scipy triangular specifies a triangular distribution
    # from loc to loc + scale, with loc + c * scale as the
    # peak, giving:
    # loc = low, loc + scale = high, loc + c * scale = peak.
    # We invert this mapping here.
    return stats.triang(
        c=(peak - low) / (high - low),
        loc=low,
        scale=(high - low))

  def testShapes(self):
    low = self._dtype(0.)
    high = self._dtype(1.)
    peak = self._dtype(0.5)
    tri = tfd.Triangular(low=low, high=high, peak=peak, validate_args=True)
    self.assertAllEqual([], self.evaluate(tri.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(tri.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), tri.event_shape)
    self.assertEqual(tf.TensorShape([]), tri.batch_shape)

  def testShapesBroadcast(self):
    low = np.zeros(shape=(2, 3), dtype=self._dtype)
    high = np.ones(shape=(1, 3), dtype=self._dtype)
    peak = self._dtype(0.5)
    tri = tfd.Triangular(low=low, high=high, peak=peak, validate_args=True)
    self.assertAllEqual([], self.evaluate(tri.event_shape_tensor()))
    self.assertAllEqual([2, 3], self.evaluate(tri.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), tri.event_shape)
    self.assertEqual(tf.TensorShape([2, 3]), tri.batch_shape)

  def testProperties(self):
    low = np.zeros(shape=(3), dtype=self._dtype)
    high = np.ones(shape=(1), dtype=self._dtype)
    peak = self._dtype(0.5)
    tri = self._create_triangular_dist(low, high, peak)
    self.assertAllClose(low, self.evaluate(tri.low))
    self.assertAllClose(high, self.evaluate(tri.high))
    self.assertAllClose(peak, self.evaluate(tri.peak))

  def testInvalidDistribution(self):
    with self.assertRaisesOpError('(low >= high||low >= peak)'):
      tri = tfd.Triangular(
          low=2., high=1., peak=0.5, validate_args=True)
      self.evaluate(tri.mean())
    with self.assertRaisesOpError('low > peak'):
      tri = tfd.Triangular(
          low=0., high=1., peak=-1., validate_args=True)
      self.evaluate(tri.mean())
    with self.assertRaisesOpError('peak > high'):
      tri = tfd.Triangular(
          low=0., high=1., peak=2., validate_args=True)
      self.evaluate(tri.mean())

  def testAssertValidSample(self):
    tri = tfd.Triangular(low=2., high=5., peak=3.3, validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to `low`'):
      self.evaluate(tri.cdf([2.3, 1.7, 4.]))
    with self.assertRaisesOpError('must be less than or equal to `high`'):
      self.evaluate(tri.survival_function([2.3, 5.2, 4.]))

  def testTriangularPDF(self):
    low = np.arange(1.0, 5.0, dtype=self._dtype)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.arange(3., 7.0, dtype=self._dtype)
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_pdf = self.evaluate(tri.prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).pdf(x),
        tri_pdf)

    tri_log_pdf = self.evaluate(tri.log_prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logpdf(x),
        tri_log_pdf)
    self.assertAllClose(tri_pdf, np.exp(tri_log_pdf))

  def testTriangularPDFPeakEqualsLow(self):
    low = np.arange(1.0, 5.0, dtype=self._dtype)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = low
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_pdf = self.evaluate(tri.prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).pdf(x),
        tri_pdf)

    tri_log_pdf = self.evaluate(
        tri.log_prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logpdf(x),
        tri_log_pdf)
    self.assertAllClose(tri_pdf, np.exp(tri_log_pdf))

  def testTriangularPDFPeakEqualsHigh(self):
    low = np.arange(1.0, 5.0, dtype=self._dtype)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = high
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_pdf = self.evaluate(tri.prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).pdf(x),
        tri_pdf)

    tri_log_pdf = self.evaluate(
        tri.log_prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logpdf(x),
        tri_log_pdf)
    self.assertAllClose(tri_pdf, np.exp(tri_log_pdf))

  def testTriangularPDFBroadcast(self):
    low = self._dtype(-3.)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.array([[0., 0., 0., 0.]] * 3, dtype=self._dtype)
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_pdf = self.evaluate(tri.prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).pdf(x),
        tri_pdf)

    tri_log_pdf = self.evaluate(
        tri.log_prob(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logpdf(x),
        tri_log_pdf)
    self.assertAllClose(tri_pdf, np.exp(tri_log_pdf))

  def testTriangularCDF(self):
    low = np.arange(1.0, 5.0, dtype=self._dtype)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.arange(3., 7.0, dtype=self._dtype)
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_cdf = self.evaluate(tri.cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).cdf(x),
        tri_cdf)

    tri_log_cdf = self.evaluate(tri.log_cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logcdf(x),
        tri_log_cdf)
    self.assertAllClose(tri_cdf, np.exp(tri_log_cdf))

  def testTriangularCDFPeakEqualsLow(self):
    low = np.arange(1.0, 5.0, dtype=self._dtype)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = low
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_cdf = self.evaluate(tri.cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).cdf(x),
        tri_cdf)

    tri_log_cdf = self.evaluate(tri.log_cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logcdf(x),
        tri_log_cdf)
    self.assertAllClose(tri_cdf, np.exp(tri_log_cdf))

  def testTriangularCDFPeakEqualsHigh(self):
    low = np.arange(1.0, 5.0, dtype=self._dtype)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = high
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_cdf = self.evaluate(tri.cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).cdf(x),
        tri_cdf)

    tri_log_cdf = self.evaluate(tri.log_cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logcdf(x),
        tri_log_cdf)
    self.assertAllClose(tri_cdf, np.exp(tri_log_cdf))

  def testTriangularCDFBroadcast(self):
    low = self._dtype(-3.)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.array([[0., 0., 0., 0.]] * 3, dtype=self._dtype)
    tri = self._create_triangular_dist(low, high, peak)

    x = self._rng.uniform(low=low, high=high).astype(self._dtype)

    tri_cdf = self.evaluate(tri.cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).cdf(x),
        tri_cdf)

    tri_log_cdf = self.evaluate(tri.log_cdf(self.make_tensor(x)))
    self.assertAllClose(
        self._scipy_triangular(low, high, peak).logcdf(x),
        tri_log_cdf)
    self.assertAllClose(tri_cdf, np.exp(tri_log_cdf))

  def testTriangularEntropy(self):
    low = self._dtype(-3.)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.arange(3., 7., dtype=self._dtype)

    tri = self._create_triangular_dist(low, high, peak)

    self.assertAllClose(
        self._scipy_triangular(low, high, peak).entropy(),
        self.evaluate(tri.entropy()))

  def testTriangularMean(self):
    low = self._dtype(-3.)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.arange(3., 7., dtype=self._dtype)
    tri = self._create_triangular_dist(low, high, peak)

    self.assertAllClose(
        self._scipy_triangular(low, high, peak).mean(),
        self.evaluate(tri.mean()))

  def testTriangularVariance(self):
    low = self._dtype(0.)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.arange(3., 7., dtype=self._dtype)
    tri = self._create_triangular_dist(low, high, peak)

    self.assertAllClose(
        self._scipy_triangular(low, high, peak).var(),
        self.evaluate(tri.variance()))

  def testTriangularSample(self):
    low = self._dtype([-3.] * 4)
    high = np.arange(7., 11., dtype=self._dtype)
    peak = np.array([0.] * 4, dtype=self._dtype)
    tri = self._create_triangular_dist(low, high, peak)
    num_samples = int(3e6)
    samples = tri.sample(num_samples, seed=test_util.test_seed())

    detectable_discrepancies = self.evaluate(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples, low, high,
            false_fail_rate=self._dtype(1e-6),
            false_pass_rate=self._dtype(1e-6)))
    below_threshold = detectable_discrepancies <= 0.05
    self.assertTrue(np.all(below_threshold))

    self.evaluate(
        st.assert_true_mean_equal_by_dkwm(
            samples, low=low, high=high, expected=tri.mean(),
            false_fail_rate=self._dtype(1e-6)))

  def testTriangularSampleMultidimensionalMean(self):
    low = np.array([np.arange(1, 21, dtype=self._dtype)])
    high = low + 3.
    peak = (high - low) / 3 + low
    tri = tfd.Triangular(low=low, high=high, peak=peak, validate_args=True)
    n = int(100e3)
    samples = tri.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 20))
    self.assertEqual(sample_values.shape, (n, 1, 20))
    self.assertAllClose(
        sample_values.mean(axis=0),
        self._scipy_triangular(low, high, peak).mean(),
        rtol=.01,
        atol=0)

  def testTriangularSampleMultidimensionalVariance(self):
    low = np.array([np.arange(1, 21, dtype=self._dtype)])
    high = low + 3.
    peak = (high - low) / 3 + low
    tri = tfd.Triangular(low=low, high=high, peak=peak, validate_args=True)
    n = int(100e3)
    samples = tri.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 20))
    self.assertEqual(sample_values.shape, (n, 1, 20))
    self.assertAllClose(
        sample_values.var(axis=0),
        self._scipy_triangular(low, high, peak).var(),
        rtol=.03,
        atol=0)

  def testTriangularExtrema(self):
    low = self._dtype(0.)
    peak = self._dtype(1.)
    high = self._dtype(4.)
    tri = tfd.Triangular(low=low, peak=peak, high=high, validate_args=True)
    self.assertAllClose(self.evaluate(tri.prob([0., 1., 4.])), [0, 0.5, 0])

  def testModifiedVariableAssertion(self):
    low = tf.Variable(0.)
    peak = tf.Variable(0.5)
    high = tf.Variable(1.)
    self.evaluate([low.initializer, peak.initializer, high.initializer])
    triangular = tfd.Triangular(
        low=low, peak=peak, high=high, validate_args=True)
    with self.assertRaisesOpError('low > peak'):
      with tf.control_dependencies([low.assign(0.6)]):
        self.evaluate(triangular.mean())
    with self.assertRaisesOpError('peak > high'):
      with tf.control_dependencies([low.assign(0.), peak.assign(1.2)]):
        self.evaluate(triangular.mean())

  def testSupportBijectorOutsideRange(self):
    low = np.array([1., 2., 3.])
    peak = np.array([4., 4., 4.])
    high = np.array([6., 7., 6.])
    dist = tfd.Triangular(low, high, peak, validate_args=False)
    eps = 1e-6
    x = np.array([1. - eps, 1.5, 6. + eps])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


@test_util.test_all_tf_execution_regimes
class TriangularTestStaticShape(test_util.TestCase, _TriangularTest):
  _dtype = np.float32
  _use_static_shape = True

  def setUp(self):
    self._rng = np.random.RandomState(123)


@test_util.test_all_tf_execution_regimes
class TriangularTestFloat64StaticShape(test_util.TestCase, _TriangularTest):
  _dtype = np.float64
  _use_static_shape = True

  def setUp(self):
    self._rng = np.random.RandomState(123)


@test_util.test_all_tf_execution_regimes
class TriangularTestDynamicShape(test_util.TestCase, _TriangularTest):
  _dtype = np.float32
  _use_static_shape = False

  def setUp(self):
    self._rng = np.random.RandomState(123)


if __name__ == '__main__':
  tf.test.main()
