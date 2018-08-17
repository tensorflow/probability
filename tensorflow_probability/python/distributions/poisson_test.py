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
from scipy import special
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class PoissonTest(tf.test.TestCase):

  def _make_poisson(self, rate, validate_args=False):
    return tfd.Poisson(rate=rate, validate_args=validate_args)

  def testPoissonShape(self):
    lam = tf.constant([3.0] * 5)
    poisson = self._make_poisson(rate=lam)

    self.assertEqual(self.evaluate(poisson.batch_shape_tensor()), (5,))
    self.assertEqual(poisson.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(poisson.event_shape_tensor()), [])
    self.assertEqual(poisson.event_shape, tf.TensorShape([]))

  def testInvalidLam(self):
    invalid_lams = [-.01, 0., -2.]
    for lam in invalid_lams:
      with self.assertRaisesOpError("Condition x > 0"):
        poisson = self._make_poisson(rate=lam, validate_args=True)
        self.evaluate(poisson.rate)

  def testPoissonLogPmf(self):
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = [2., 3., 4., 5., 6., 7.]
    poisson = self._make_poisson(rate=lam)
    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.get_shape(), (6,))
    self.assertAllClose(self.evaluate(log_pmf), stats.poisson.logpmf(x, lam_v))

    pmf = poisson.prob(x)
    self.assertEqual(pmf.get_shape(), (6,))
    self.assertAllClose(self.evaluate(pmf), stats.poisson.pmf(x, lam_v))

  def testPoissonLogPmfValidateArgs(self):
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    x = tf.placeholder_with_default(
        input=[2.5, 3.2, 4.3, 5.1, 6., 7.], shape=[6])
    poisson = self._make_poisson(rate=lam, validate_args=True)

    # Non-integer
    with self.assertRaisesOpError("cannot contain fractional components"):
      self.evaluate(poisson.log_prob(x))

    with self.assertRaisesOpError("Condition x >= 0"):
      self.evaluate(poisson.log_prob([-1.]))

    poisson = self._make_poisson(rate=lam, validate_args=False)
    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.get_shape(), (6,))
    pmf = poisson.prob(x)
    self.assertEqual(pmf.get_shape(), (6,))

  def testPoissonLogPmfMultidimensional(self):
    batch_size = 6
    lam = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    lam_v = [2.0, 4.0, 5.0]
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    poisson = self._make_poisson(rate=lam)
    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.get_shape(), (6, 3))
    self.assertAllClose(self.evaluate(log_pmf), stats.poisson.logpmf(x, lam_v))

    pmf = poisson.prob(x)
    self.assertEqual(pmf.get_shape(), (6, 3))
    self.assertAllClose(self.evaluate(pmf), stats.poisson.pmf(x, lam_v))

  def testPoissonCDF(self):
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = [2., 3., 4., 5., 6., 7.]

    poisson = self._make_poisson(rate=lam)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.get_shape(), (6,))
    self.assertAllClose(self.evaluate(log_cdf), stats.poisson.logcdf(x, lam_v))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.get_shape(), (6,))
    self.assertAllClose(self.evaluate(cdf), stats.poisson.cdf(x, lam_v))

  def testPoissonCDFNonIntegerValues(self):
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = np.array([2.2, 3.1, 4., 5.5, 6., 7.], dtype=np.float32)

    poisson = self._make_poisson(rate=lam)
    cdf = poisson.cdf(x)
    self.assertEqual(cdf.get_shape(), (6,))

    # The Poisson CDF should be valid on these non-integer values, and
    # equal to igammac(1 + x, rate).
    self.assertAllClose(self.evaluate(cdf), special.gammaincc(1. + x, lam_v))

    with self.assertRaisesOpError("cannot contain fractional components"):
      poisson_validate = self._make_poisson(rate=lam, validate_args=True)
      self.evaluate(poisson_validate.cdf(x))

  def testPoissonCdfMultidimensional(self):
    batch_size = 6
    lam = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    lam_v = [2.0, 4.0, 5.0]
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    poisson = self._make_poisson(rate=lam)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.get_shape(), (6, 3))
    self.assertAllClose(self.evaluate(log_cdf), stats.poisson.logcdf(x, lam_v))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.get_shape(), (6, 3))
    self.assertAllClose(self.evaluate(cdf), stats.poisson.cdf(x, lam_v))

  def testPoissonMean(self):
    lam_v = [1.0, 3.0, 2.5]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.mean().get_shape(), (3,))
    self.assertAllClose(
        self.evaluate(poisson.mean()), stats.poisson.mean(lam_v))
    self.assertAllClose(self.evaluate(poisson.mean()), lam_v)

  def testPoissonVariance(self):
    lam_v = [1.0, 3.0, 2.5]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.variance().get_shape(), (3,))
    self.assertAllClose(
        self.evaluate(poisson.variance()), stats.poisson.var(lam_v))
    self.assertAllClose(self.evaluate(poisson.variance()), lam_v)

  def testPoissonStd(self):
    lam_v = [1.0, 3.0, 2.5]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.stddev().get_shape(), (3,))
    self.assertAllClose(
        self.evaluate(poisson.stddev()), stats.poisson.std(lam_v))
    self.assertAllClose(self.evaluate(poisson.stddev()), np.sqrt(lam_v))

  def testPoissonMode(self):
    lam_v = [1.0, 3.0, 2.5, 3.2, 1.1, 0.05]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.mode().get_shape(), (6,))
    self.assertAllClose(self.evaluate(poisson.mode()), np.floor(lam_v))

  def testPoissonMultipleMode(self):
    lam_v = [1.0, 3.0, 2.0, 4.0, 5.0, 10.0]
    poisson = self._make_poisson(rate=lam_v)
    # For the case where lam is an integer, the modes are: lam and lam - 1.
    # In this case, we get back the larger of the two modes.
    self.assertEqual((6,), poisson.mode().get_shape())
    self.assertAllClose(lam_v, self.evaluate(poisson.mode()))

  def testPoissonSample(self):
    lam_v = 4.0
    lam = tf.constant(lam_v)
    # Choosing `n >= (k/rtol)**2, roughly ensures our sample mean should be
    # within `k` std. deviations of actual up to rtol precision.
    n = int(100e3)
    poisson = self._make_poisson(rate=lam)
    samples = poisson.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.get_shape(), (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(), stats.poisson.mean(lam_v), rtol=.01)
    self.assertAllClose(sample_values.var(), stats.poisson.var(lam_v), rtol=.01)

  def testPoissonSampleMultidimensionalMean(self):
    lam_v = np.array([np.arange(1, 51, dtype=np.float32)])  # 1 x 50
    poisson = self._make_poisson(rate=lam_v)
    # Choosing `n >= (k/rtol)**2, roughly ensures our sample mean should be
    # within `k` std. deviations of actual up to rtol precision.
    n = int(100e3)
    samples = poisson.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.get_shape(), (n, 1, 50))
    self.assertEqual(sample_values.shape, (n, 1, 50))
    self.assertAllClose(
        sample_values.mean(axis=0), stats.poisson.mean(lam_v), rtol=.01, atol=0)

  def testPoissonSampleMultidimensionalVariance(self):
    lam_v = np.array([np.arange(5, 15, dtype=np.float32)])  # 1 x 10
    poisson = self._make_poisson(rate=lam_v)
    # Choosing `n >= 2 * lam * (k/rtol)**2, roughly ensures our sample
    # variance should be within `k` std. deviations of actual up to rtol
    # precision.
    n = int(300e3)
    samples = poisson.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.get_shape(), (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))

    self.assertAllClose(
        sample_values.var(axis=0), stats.poisson.var(lam_v), rtol=.03, atol=0)


@test_util.run_all_in_graph_and_eager_modes
class PoissonLogRateTest(PoissonTest):

  def _make_poisson(self, rate, validate_args=False):
    return tfd.Poisson(log_rate=tf.log(rate), validate_args=validate_args)

  def testInvalidLam(self):
    # No need to worry about the non-negativity of `rate` when using the
    # `log_rate` parameterization.
    pass


if __name__ == "__main__":
  tf.test.main()
