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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class ZipfTest(test_util.TestCase):

  def assertBetween(self, x, minimum, maximum):
    self.assertGreaterEqual(x, minimum)
    self.assertLessEqual(x, maximum)

  def assertAllBetween(self, a, minval, maxval, atol=1e-6):
    a = self._GetNdArray(a)
    minval = self._GetNdArray(minval)
    maxval = self._GetNdArray(maxval)

    self.assertEqual(a.shape, minval.shape)
    self.assertEqual(a.shape, maxval.shape)

    for idx, _ in np.ndenumerate(a):
      self.assertBetween(a[idx], minval[idx] - atol, maxval[idx] + atol)

  def testZipfShape(self):
    power = tf.constant([3.0] * 5)
    zipf = tfd.Zipf(power=power, validate_args=True)

    self.assertEqual(self.evaluate(zipf.batch_shape_tensor()), (5,))
    self.assertEqual(zipf.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(zipf.event_shape_tensor()), [])
    self.assertEqual(zipf.event_shape, tf.TensorShape([]))

  def testInvalidPower(self):
    invalid_powers = [-.02, 0.5, -2., .99, 1.]
    for power in invalid_powers:
      with self.assertRaisesOpError("Condition x > y"):
        zipf = tfd.Zipf(power=power, validate_args=True)
        self.evaluate(zipf.mean())

  def testNanPower(self):
    zipf = tfd.Zipf(power=np.nan, validate_args=False)
    self.assertAllNan(self.evaluate(zipf.power))

  def testValidPower_ImplicitlyConvertsToFloat32(self):
    powers = [2, 10, 1.1]
    for power in powers:
      zipf = tfd.Zipf(power=power, validate_args=True)
      self.assertEqual(zipf.power.dtype, tf.float32)

  def testEventDtype(self):
    for power_dtype in [tf.float32, tf.float64]:
      for event_dtype in [tf.int32, tf.int64, tf.float32, tf.float64]:
        power_dtype = tf.float32
        event_dtype = tf.int32
        power = tf.constant(5., dtype=power_dtype)
        zipf = tfd.Zipf(power=power, dtype=event_dtype, validate_args=True)
        self.assertEqual(zipf.dtype, event_dtype)
        self.assertEqual(
            zipf.dtype, zipf.sample(10, seed=test_util.test_seed()).dtype)
        self.assertEqual(
            zipf.dtype, zipf.sample(1, seed=test_util.test_seed()).dtype)
        self.assertEqual(zipf.dtype, zipf.mode().dtype)

  def testInvalidEventDtype(self):
    with self.assertRaisesWithPredicateMatch(
        TypeError, "power.dtype .* not a supported .* type"):
      power = tf.constant(5., dtype=tf.float16)
      zipf = tfd.Zipf(power=power, dtype=tf.int32, validate_args=True)
      self.evaluate(zipf.sample(seed=test_util.test_seed()))

  def testZipfLogPmf_InvalidArgs(self):
    power = tf.constant([4.0])
    # Non-integer samples are rejected if validate_args is True and
    # interpolate_nondiscrete is False.
    non_integer_samples = [0.99, 4.5, 5.001, 1e-6, -3, -2, -1]
    for x in non_integer_samples:
      zipf = tfd.Zipf(
          power=power, interpolate_nondiscrete=False, validate_args=True)

      with self.assertRaisesOpError("Condition (x == y|x >= 0)"):
        self.evaluate(zipf.log_prob(x))

      with self.assertRaisesOpError("Condition (x == y|x >= 0)"):
        self.evaluate(zipf.prob(x))

  def testZipfLogPmf_IntegerArgs(self):
    batch_size = 9
    power = tf.constant([3.0] * batch_size)
    power_v = 3.0
    x = np.array([-3., -0., 0., 2., 3., 4., 5., 6., 7.], dtype=np.float32)
    zipf = tfd.Zipf(power=power, validate_args=False)
    log_pmf = zipf.log_prob(x)
    self.assertEqual((batch_size,), log_pmf.shape)
    self.assertAllClose(self.evaluate(log_pmf), stats.zipf.logpmf(x, power_v))

    pmf = zipf.prob(x)
    self.assertEqual((batch_size,), pmf.shape)
    self.assertAllClose(self.evaluate(pmf), stats.zipf.pmf(x, power_v))

  def testZipfLogPmf_NonIntegerArgs(self):
    batch_size = 12
    power = tf.constant([3.0] * batch_size)
    power_v = 3.0
    x = [-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.2]

    zipf = tfd.Zipf(power=power, validate_args=False)
    log_pmf = zipf.log_prob(x)
    self.assertEqual((batch_size,), log_pmf.shape)

    # Check that log_pmf(x) of tfd.Zipf is between the values of
    # stats.zipf.logpmf for ceil(x) and floor(x).
    log_pmf_values = self.evaluate(log_pmf)
    floor_x = np.floor(x)
    ceil_x = np.ceil(x)
    self.assertAllBetween(log_pmf_values, stats.zipf.logpmf(ceil_x, power_v),
                          stats.zipf.logpmf(floor_x, power_v))

    # Check that pmf(x) of tfd.Zipf is between the values of stats.zipf.pmf for
    # ceil(x) and floor(x).
    pmf = zipf.prob(x)
    self.assertEqual((batch_size,), pmf.shape)

    pmf_values = self.evaluate(pmf)
    self.assertAllBetween(pmf_values, stats.zipf.pmf(ceil_x, power_v),
                          stats.zipf.pmf(floor_x, power_v))

  def testZipfLogPmf_NonIntegerArgsNoInterpolation(self):
    batch_size = 12
    power = tf.constant([3.0] * batch_size)
    power_v = 3.0
    x = [-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.2]

    zipf = tfd.Zipf(
        power=power, interpolate_nondiscrete=False, validate_args=False)
    log_pmf = zipf.log_prob(x)
    self.assertEqual((batch_size,), log_pmf.shape)

    log_pmf_values = self.evaluate(log_pmf)
    self.assertAllClose(log_pmf_values, stats.zipf.logpmf(x, power_v))

    pmf = zipf.prob(x)
    self.assertEqual((batch_size,), pmf.shape)

    pmf_values = self.evaluate(pmf)
    self.assertAllClose(pmf_values, stats.zipf.pmf(x, power_v))

  def testZipfLogPmfMultidimensional_IntegerArgs(self):
    batch_size = 6
    power = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    power_v = [2.0, 4.0, 5.0]
    x = np.array([[2.1, 3.5, 4.9, 5., 6.6, 7.]], dtype=np.int32).T

    zipf = tfd.Zipf(power=power, validate_args=True)
    log_pmf = zipf.log_prob(x)
    self.assertEqual((6, 3), log_pmf.shape)
    self.assertAllClose(self.evaluate(log_pmf), stats.zipf.logpmf(x, power_v))

    pmf = zipf.prob(x)
    self.assertEqual((6, 3), pmf.shape)
    self.assertAllClose(self.evaluate(pmf), stats.zipf.pmf(x, power_v))

  def testZipfLogPmfMultidimensional_NonIntegerArgs(self):
    batch_size = 6
    power = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    power_v = [2.0, 4.0, 5.0]
    x = np.array([[2., 3.2, 4.3, 5.5, 6.9, 7.]], dtype=np.float32).T
    floor_x = np.floor(x)
    ceil_x = np.ceil(x)

    zipf = tfd.Zipf(power=power, validate_args=True)
    log_pmf = zipf.log_prob(x)
    self.assertEqual((6, 3), log_pmf.shape)
    self.assertAllBetween(
        self.evaluate(log_pmf), stats.zipf.logpmf(ceil_x, power_v),
        stats.zipf.logpmf(floor_x, power_v))

    pmf = zipf.prob(x)
    self.assertEqual((6, 3), pmf.shape)
    self.assertAllBetween(
        self.evaluate(pmf), stats.zipf.pmf(ceil_x, power_v),
        stats.zipf.pmf(floor_x, power_v))

  def testZipfCdf_IntegerArgs(self):
    batch_size = 12
    power = tf.constant([3.0] * batch_size)
    power_v = 3.0
    x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    zipf = tfd.Zipf(power=power, validate_args=False)
    log_cdf = zipf.log_cdf(x)
    self.assertEqual((batch_size,), log_cdf.shape)
    self.assertAllClose(self.evaluate(log_cdf), stats.zipf.logcdf(x, power_v))

    cdf = zipf.cdf(x)
    self.assertEqual((batch_size,), cdf.shape)
    self.assertAllClose(self.evaluate(cdf), stats.zipf.cdf(x, power_v))

  def testZipfCdf_NonIntegerArgsNoInterpolation(self):
    batch_size = 12
    power = tf.constant([3.0] * batch_size)
    power_v = 3.0
    x = [-3.5, -0.5, 0., 1, 1.1, 2.2, 3.1, 4., 5., 5.5, 6.4, 7.8]

    zipf = tfd.Zipf(
        power=power, interpolate_nondiscrete=False, validate_args=False)
    log_cdf = zipf.log_cdf(x)
    self.assertEqual((batch_size,), log_cdf.shape)
    self.assertAllClose(self.evaluate(log_cdf), stats.zipf.logcdf(x, power_v))

    cdf = zipf.cdf(x)
    self.assertEqual((batch_size,), cdf.shape)
    self.assertAllClose(self.evaluate(cdf), stats.zipf.cdf(x, power_v))

  def testZipfCdf_NonIntegerArgsInterpolated(self):
    batch_size = 12
    power = tf.constant([3.0] * batch_size)
    power_v = 3.0
    x = [-3.5, -0.5, 0., 1, 1.1, 2.2, 3.1, 4., 5., 5.5, 6.4, 7.8]
    floor_x = np.floor(x)
    ceil_x = np.ceil(x)

    zipf = tfd.Zipf(power=power, validate_args=False)
    log_cdf = zipf.log_cdf(x)
    self.assertEqual((batch_size,), log_cdf.shape)
    self.assertAllBetween(
        self.evaluate(log_cdf), stats.zipf.logcdf(floor_x, power_v),
        stats.zipf.logcdf(ceil_x, power_v))

    cdf = zipf.cdf(x)
    self.assertEqual((batch_size,), cdf.shape)
    self.assertAllBetween(
        self.evaluate(cdf), stats.zipf.cdf(floor_x, power_v),
        stats.zipf.cdf(ceil_x, power_v))

  def testZipfCdf_NonIntegerArgs(self):
    batch_size = 12
    power = tf.constant([3.0] * batch_size)
    power_v = 3.0
    x = [-3.5, -0.5, 0., 1, 1.1, 2.2, 3.1, 4., 5., 5.5, 6.4, 7.8]
    floor_x = np.floor(x)
    ceil_x = np.ceil(x)

    zipf = tfd.Zipf(power=power, validate_args=False)
    log_cdf = zipf.log_cdf(x)
    self.assertEqual((batch_size,), log_cdf.shape)
    self.assertAllBetween(
        self.evaluate(log_cdf), stats.zipf.logcdf(floor_x, power_v),
        stats.zipf.logcdf(ceil_x, power_v))

    cdf = zipf.cdf(x)
    self.assertEqual((batch_size,), cdf.shape)
    self.assertAllBetween(
        self.evaluate(cdf), stats.zipf.cdf(floor_x, power_v),
        stats.zipf.cdf(ceil_x, power_v))

  def testZipfCdfMultidimensional_IntegerArgs(self):
    batch_size = 6
    power = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    power_v = [2.0, 4.0, 5.0]
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    zipf = tfd.Zipf(power=power, validate_args=True)
    log_cdf = zipf.log_cdf(x)
    self.assertEqual((6, 3), log_cdf.shape)
    self.assertAllClose(self.evaluate(log_cdf), stats.zipf.logcdf(x, power_v))

    cdf = zipf.cdf(x)
    self.assertEqual((6, 3), cdf.shape)
    self.assertAllClose(self.evaluate(cdf), stats.zipf.cdf(x, power_v))

  def testZipfCdfMultidimensional_NonIntegerArgs(self):
    batch_size = 6
    power = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    power_v = [2.0, 4.0, 5.0]
    x = np.array([[2.3, 3.5, 4.1, 5.5, 6.8, 7.9]], dtype=np.float32).T
    floor_x = np.floor(x)
    ceil_x = np.ceil(x)

    zipf = tfd.Zipf(power=power, validate_args=True)
    log_cdf = zipf.log_cdf(x)
    self.assertEqual((6, 3), log_cdf.shape)
    self.assertAllBetween(
        self.evaluate(log_cdf), stats.zipf.logcdf(floor_x, power_v),
        stats.zipf.logcdf(ceil_x, power_v))

    cdf = zipf.cdf(x)
    self.assertEqual((6, 3), cdf.shape)
    self.assertAllBetween(
        self.evaluate(cdf), stats.zipf.cdf(floor_x, power_v),
        stats.zipf.cdf(ceil_x, power_v))

  def testZipfMean(self):
    power_v = [2.0, 3.0, 2.5]
    zipf = tfd.Zipf(power=power_v, validate_args=True)
    self.assertEqual((3,), zipf.mean().shape)
    self.assertAllClose(self.evaluate(zipf.mean()), stats.zipf.mean(power_v))

  def testZipfVariance(self):
    power_v = [4.0, 3.0, 5.5]  # var is undefined for power <= 3
    zipf = tfd.Zipf(power=power_v, validate_args=True)
    self.assertEqual((3,), zipf.variance().shape)
    stat_vars = np.vectorize(stats.zipf.var)(power_v)
    self.assertAllClose(self.evaluate(zipf.variance()), stat_vars)

  def testZipfStd(self):
    power_v = [4.0, 3.5, 4.5]
    zipf = tfd.Zipf(power=power_v, validate_args=True)
    self.assertEqual((3,), zipf.stddev().shape)
    stat_stddevs = np.vectorize(stats.zipf.std)(power_v)
    self.assertAllClose(self.evaluate(zipf.stddev()), stat_stddevs)

  def testZipfMode(self):
    power_v = [10.0, 3.0, 2.5, 3.2, 1.1, 0.05]
    zipf = tfd.Zipf(power=power_v, validate_args=False)
    self.assertEqual((6,), zipf.mode().shape)
    self.assertAllClose(self.evaluate(zipf.mode()), np.ones_like(power_v))

  def testZipfSample(self):
    power_v = 5.
    n = int(500e4)

    for power_dtype in [tf.float32, tf.float64]:
      power = tf.constant(power_v, dtype=power_dtype)
      for dtype in [tf.int32, tf.int64, tf.float32, tf.float64]:
        zipf = tfd.Zipf(power=power, dtype=dtype, validate_args=True)
        samples = zipf.sample(n, seed=test_util.test_seed())
        sample_values = self.evaluate(samples)
        self.assertEqual((n,), samples.shape)
        self.assertEqual((n,), sample_values.shape)
        self.assertAllClose(
            sample_values.mean(), stats.zipf.mean(power_v), rtol=.01)
        self.assertAllClose(
            sample_values.std(), stats.zipf.std(power_v), rtol=.03)

  def testZipfSample_ValidateArgs(self):
    power_v = 3.
    n = int(100e3)

    for power_dtype in [tf.float32, tf.float64]:
      power = tf.constant(power_v, dtype=power_dtype)

      for dtype in [tf.int32, tf.int64, tf.float32, tf.float64]:
        zipf = tfd.Zipf(power=power, dtype=dtype, validate_args=True)
        samples = zipf.sample(n, seed=test_util.test_seed())
        self.evaluate(samples)

  def testZipfSampleMultidimensionalMean(self):
    power_v = np.array([np.arange(5, 15, dtype=np.float32)])  # 1 x 10
    zipf = tfd.Zipf(power=power_v, validate_args=True)
    n = int(100e3)
    samples = zipf.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n, 1, 10,), samples.shape)
    self.assertEqual((n, 1, 10,), sample_values.shape)

    # stats.zipf wants float64 params.
    stats_mean = np.vectorize(stats.zipf.mean)(power_v.astype(np.float64))
    self.assertAllClose(sample_values.mean(axis=0), stats_mean, rtol=.01)

  def testZipfSampleMultidimensionalStd(self):
    power_v = np.array([np.arange(5, 10, dtype=np.float32)])  # 1 x 5
    zipf = tfd.Zipf(power=power_v, validate_args=True)
    n = int(100e4)
    samples = zipf.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n, 1, 5), samples.shape)
    self.assertEqual((n, 1, 5), sample_values.shape)

    # stats.zipf wants float64 params.
    stats_std = np.vectorize(stats.zipf.std)(power_v.astype(np.float64))
    self.assertAllClose(sample_values.std(axis=0), stats_std, rtol=.04)

  # Test that sampling with the same seed twice gives the same results.
  def testZipfSampleMultipleTimes(self):
    n = 1000
    seed = test_util.test_seed()
    power = 1.5

    zipf1 = tfd.Zipf(power=power, name="zipf1", validate_args=True)
    tf.random.set_seed(seed)
    samples1 = self.evaluate(zipf1.sample(n, seed=seed))

    zipf2 = tfd.Zipf(power=power, name="zipf2", validate_args=True)
    tf.random.set_seed(seed)
    samples2 = self.evaluate(zipf2.sample(n, seed=seed))

    self.assertAllEqual(samples1, samples2)

  def testZipfSample_AvoidsInfiniteLoop(self):
    zipf = tfd.Zipf(power=1., validate_args=False)
    n = 1000
    self.evaluate(zipf.sample(n, seed=test_util.test_seed()))


if __name__ == "__main__":
  tf.test.main()
