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
"""Tests for Moyal."""

# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import moyal
from tensorflow_probability.python.internal import test_util


class _MoyalTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self.dtype)
    return tf1.placeholder_with_default(
        x, shape=x.shape if self.use_static_shape else None)

  def testMoyalShape(self):
    loc = np.array([3.0] * 5, dtype=self.dtype)
    scale = np.array([3.0] * 5, dtype=self.dtype)
    dist = moyal.Moyal(loc=loc, scale=scale, validate_args=True)

    self.assertEqual((5,), self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([5]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testInvalidScale(self):
    scale = [-.01, 0., 2.]
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      dist = moyal.Moyal(loc=0., scale=scale, validate_args=True)
      self.evaluate(dist.mean())

    scale = tf.Variable([.01])
    self.evaluate(scale.initializer)
    dist = moyal.Moyal(loc=0., scale=scale, validate_args=True)
    self.assertIs(scale, dist.scale)
    self.evaluate(dist.mean())
    with tf.control_dependencies([scale.assign([-.01])]):
      with self.assertRaisesOpError('Argument `scale` must be positive.'):
        self.evaluate(dist.mean())

  def testMoyalLogPdf(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self.dtype)
    scale = np.array([3.] * batch_size, dtype=self.dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self.dtype)
    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)
    log_pdf = dist.log_prob(self.make_tensor(x))
    self.assertAllClose(
        stats.moyal.logpdf(x, loc=loc, scale=scale),
        self.evaluate(log_pdf))

    pdf = dist.prob(x)
    self.assertAllClose(
        stats.moyal.pdf(x, loc=loc, scale=scale), self.evaluate(pdf))

  def testMoyalLogPdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0], dtype=self.dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self.dtype).T

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)
    log_pdf = dist.log_prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_pdf), stats.moyal.logpdf(x, loc=loc, scale=scale))

    pdf = dist.prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(pdf), stats.moyal.pdf(x, loc=loc, scale=scale))

  def testMoyalCDF(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self.dtype)
    scale = np.array([3.] * batch_size, dtype=self.dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self.dtype)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    log_cdf = dist.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf), stats.moyal.logcdf(x, loc=loc, scale=scale))

    cdf = dist.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf), stats.moyal.cdf(x, loc=loc, scale=scale))

  def testMoyalCdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0], dtype=self.dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self.dtype).T

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    log_cdf = dist.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.moyal.logcdf(x, loc=loc, scale=scale))

    cdf = dist.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf),
        stats.moyal.cdf(x, loc=loc, scale=scale))

  def testMoyalMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0], dtype=self.dtype)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)
    self.assertAllClose(
        self.evaluate(dist.mean()), stats.moyal.mean(loc=loc, scale=scale))

  def testMoyalVariance(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0], dtype=self.dtype)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    self.assertAllClose(
        self.evaluate(dist.variance()), stats.moyal.var(loc=loc, scale=scale))

  def testMoyalStd(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0], dtype=self.dtype)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    self.assertAllClose(
        self.evaluate(dist.stddev()), stats.moyal.std(loc=loc, scale=scale))

  def testMoyalMode(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0], dtype=self.dtype)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    self.assertAllClose(self.evaluate(dist.mode()), self.evaluate(dist.loc))

  def testMoyalSample(self):
    loc = self.dtype(4.0)
    scale = self.dtype(1.0)
    n = int(3e5)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n,), sample_values.shape)
    self.assertAllClose(
        stats.moyal.mean(loc=loc, scale=scale),
        sample_values.mean(), rtol=.01)
    self.assertAllClose(
        stats.moyal.var(loc=loc, scale=scale),
        sample_values.var(), rtol=.01)

  def testMoyalSampleMultidimensionalMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self.dtype)
    n = int(2e5)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    # TODO(b/157561663): Remove the masking once tf.math.special.erfcinv exists.
    sample_values = np.ma.masked_invalid(sample_values)
    self.assertAllClose(
        stats.moyal.mean(loc=loc, scale=scale),
        sample_values.mean(axis=0),
        rtol=.03,
        atol=0)

  def testMoyalSampleMultidimensionalVar(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self.dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self.dtype)
    n = int(1e5)

    dist = moyal.Moyal(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    # TODO(b/157561663): Remove the masking once tf.math.special.erfcinv exists.
    sample_values = np.ma.masked_invalid(sample_values)
    self.assertAllClose(
        stats.moyal.var(loc=loc, scale=scale),
        sample_values.var(axis=0),
        rtol=.03,
        atol=0)

  def testMoyalMoyalKL(self):
    a_loc = np.arange(-2.0, 3.0, 1.0)
    a_scale = np.arange(0.5, 2.5, 0.5)
    b_loc = 2 * np.arange(-2.0, 3.0, 1.0)
    b_scale = np.arange(0.5, 2.5, 0.5)

    # This reshape is intended to expand the number of test cases.
    a_loc = a_loc.reshape((len(a_loc), 1, 1, 1))
    a_scale = a_scale.reshape((1, len(a_scale), 1, 1))
    b_loc = b_loc.reshape((1, 1, len(b_loc), 1))
    b_scale = b_scale.reshape((1, 1, 1, len(b_scale)))

    a = moyal.Moyal(loc=a_loc, scale=a_scale, validate_args=True)
    b = moyal.Moyal(loc=b_loc, scale=b_scale, validate_args=True)

    kl = kullback_leibler.kl_divergence(a, b)

    x = a.sample(int(3e5), seed=test_util.test_seed())
    kl_samples = a.log_prob(x) - b.log_prob(x)
    kl_, kl_samples_ = self.evaluate([kl, kl_samples])

    self.assertAllMeansClose(kl_samples_, kl_, axis=0, atol=1e-15, rtol=1e-1)

    zero_kl = kullback_leibler.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(zero_kl), zero_kl])
    self.assertAllClose(true_zero_kl_, zero_kl_)


@test_util.test_all_tf_execution_regimes
class MoyalTestStaticShape(test_util.TestCase, _MoyalTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class MoyalTestDynamicShape(test_util.TestCase, _MoyalTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class MoyalTestFloat64StaticShape(test_util.TestCase, _MoyalTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class MoyalTestFloat64DynamicShape(test_util.TestCase, _MoyalTest):
  dtype = np.float64
  use_static_shape = False


if __name__ == '__main__':
  test_util.main()
