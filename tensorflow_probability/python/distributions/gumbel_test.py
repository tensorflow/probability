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
"""Tests for Gumbel."""

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


class _GumbelTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self._dtype)
    return tf1.placeholder_with_default(
        x, shape=x.shape if self._use_static_shape else None)

  def testGumbelShape(self):
    loc = np.array([3.0] * 5, dtype=self._dtype)
    scale = np.array([3.0] * 5, dtype=self._dtype)
    gumbel = tfd.Gumbel(loc=loc, scale=scale, validate_args=True)

    self.assertEqual((5,), self.evaluate(gumbel.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([5]), gumbel.batch_shape)
    self.assertAllEqual([], self.evaluate(gumbel.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), gumbel.event_shape)

  def testInvalidScale(self):
    scale = [-.01, 0., 2.]
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      gumbel = tfd.Gumbel(loc=0., scale=scale, validate_args=True)
      self.evaluate(gumbel.mean())

    scale = tf.Variable([.01])
    self.evaluate(scale.initializer)
    gumbel = tfd.Gumbel(loc=0., scale=scale, validate_args=True)
    self.assertIs(scale, gumbel.scale)
    self.evaluate(gumbel.mean())
    with tf.control_dependencies([scale.assign([-.01])]):
      with self.assertRaisesOpError('Argument `scale` must be positive.'):
        self.evaluate(gumbel.mean())

  def testGumbelLogPdf(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)
    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)
    log_pdf = gumbel.log_prob(self.make_tensor(x))
    self.assertAllClose(
        stats.gumbel_r.logpdf(x, loc=loc, scale=scale),
        self.evaluate(log_pdf))

    pdf = gumbel.prob(x)
    self.assertAllClose(
        stats.gumbel_r.pdf(x, loc=loc, scale=scale), self.evaluate(pdf))

  def testGumbelLogPdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)
    log_pdf = gumbel.log_prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_pdf), stats.gumbel_r.logpdf(x, loc=loc, scale=scale))

    pdf = gumbel.prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(pdf), stats.gumbel_r.pdf(x, loc=loc, scale=scale))

  def testGumbelCDF(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    log_cdf = gumbel.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf), stats.gumbel_r.logcdf(x, loc=loc, scale=scale))

    cdf = gumbel.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf), stats.gumbel_r.cdf(x, loc=loc, scale=scale))

  def testGumbelCdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    log_cdf = gumbel.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.gumbel_r.logcdf(x, loc=loc, scale=scale))

    cdf = gumbel.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf),
        stats.gumbel_r.cdf(x, loc=loc, scale=scale))

  def testGumbelMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)
    self.assertAllClose(self.evaluate(gumbel.mean()),
                        stats.gumbel_r.mean(loc=loc, scale=scale))

  def testGumbelVariance(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    self.assertAllClose(self.evaluate(gumbel.variance()),
                        stats.gumbel_r.var(loc=loc, scale=scale))

  def testGumbelStd(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    self.assertAllClose(self.evaluate(gumbel.stddev()),
                        stats.gumbel_r.std(loc=loc, scale=scale))

  def testGumbelMode(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    self.assertAllClose(self.evaluate(gumbel.mode()), self.evaluate(gumbel.loc))

  def testGumbelSample(self):
    loc = self._dtype(4.0)
    scale = self._dtype(1.0)
    n = int(100e3)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    samples = gumbel.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n,), sample_values.shape)
    self.assertAllClose(
        stats.gumbel_r.mean(loc=loc, scale=scale),
        sample_values.mean(), rtol=.01)
    self.assertAllClose(
        stats.gumbel_r.var(loc=loc, scale=scale),
        sample_values.var(), rtol=.01)

  def testGumbelSampleMultidimensionalMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    n = int(1e5)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    samples = gumbel.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        stats.gumbel_r.mean(loc=loc, scale=scale),
        sample_values.mean(axis=0),
        rtol=.03,
        atol=0)

  def testGumbelSampleMultidimensionalVar(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    n = int(1e5)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale),
        validate_args=True)

    samples = gumbel.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        stats.gumbel_r.var(loc=loc, scale=scale),
        sample_values.var(axis=0),
        rtol=.03,
        atol=0)

  def testGumbelGumbelKL(self):
    a_loc = np.arange(-2.0, 3.0, 1.0)
    a_scale = np.arange(0.5, 2.5, 0.5)
    b_loc = 2 * np.arange(-2.0, 3.0, 1.0)
    b_scale = np.arange(0.5, 2.5, 0.5)

    # This reshape is intended to expand the number of test cases.
    a_loc = a_loc.reshape((len(a_loc), 1, 1, 1))
    a_scale = a_scale.reshape((1, len(a_scale), 1, 1))
    b_loc = b_loc.reshape((1, 1, len(b_loc), 1))
    b_scale = b_scale.reshape((1, 1, 1, len(b_scale)))

    a = tfd.Gumbel(loc=a_loc, scale=a_scale, validate_args=True)
    b = tfd.Gumbel(loc=b_loc, scale=b_scale, validate_args=True)

    true_kl = (np.log(b_scale) - np.log(a_scale)
               + np.euler_gamma * (a_scale / b_scale - 1.)
               + np.expm1((b_loc - a_loc) / b_scale
                          + np.vectorize(np.math.lgamma)(a_scale / b_scale
                                                         + 1.))
               + (a_loc - b_loc) / b_scale)

    kl = tfd.kl_divergence(a, b)

    x = a.sample(int(1e5), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)

    # As noted in the Gumbel-Gumbel KL divergence implementation, there is an
    # error in the reference paper we use to implement our divergence. This
    # error is a missing summand, (a.loc - b.loc) / b.scale. To ensure that we
    # are adequately testing this difference in the below tests, we compute the
    # relative error between kl_sample_ and kl_ and check that it is "much less"
    # than this missing summand.
    summand = (a_loc - b_loc) / b_scale
    relative_error = (tf.abs(kl - kl_sample) /
                      tf.minimum(tf.abs(kl), tf.abs(kl_sample)))
    exists_missing_summand_test = tf.reduce_any(
        summand > 2 * relative_error)
    exists_missing_summand_test_ = self.evaluate(exists_missing_summand_test)
    self.assertTrue(exists_missing_summand_test_,
                    msg=('No test case exists where (a.loc - b.loc) / b.scale '
                         'is much less than the relative error between kl as '
                         'computed in closed form, and kl as computed by '
                         'sampling. Failing to include such a test case makes '
                         'it difficult to detect regressions where this '
                         'summand (which is missing in our reference paper) '
                         'is omitted.'))

    kl_, kl_sample_ = self.evaluate([kl, kl_sample])
    self.assertAllClose(true_kl, kl_, atol=0.0, rtol=1e-12)
    self.assertAllClose(true_kl, kl_sample_, atol=0.0, rtol=1e-1)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(zero_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)


@test_util.test_all_tf_execution_regimes
class GumbelTestStaticShape(test_util.TestCase, _GumbelTest):
  _dtype = np.float32
  _use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GumbelTestFloat64StaticShape(test_util.TestCase, _GumbelTest):
  _dtype = np.float64
  _use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GumbelTestDynamicShape(test_util.TestCase, _GumbelTest):
  _dtype = np.float32
  _use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
