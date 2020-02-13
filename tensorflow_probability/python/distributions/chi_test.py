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
"""Tests for Chi distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import special
from scipy import stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class ChiTest(test_util.TestCase):

  def testChiLogPDF(self):
    df = np.arange(1, 6, dtype=np.float64)
    x = np.arange(1, 6, dtype=np.float64)

    df = df.reshape((len(df), 1))
    x = x.reshape((1, len(x)))

    chi = tfd.Chi(df=df, validate_args=True)
    expected_log_pdf = stats.chi.logpdf(x, df)

    log_pdf = chi.log_prob(x)
    self.assertEqual(log_pdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

    pdf = chi.prob(x)
    self.assertEqual(pdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testLogPdfAssertsOnInvalidSample(self):
    d = tfd.Chi(df=13.37, validate_args=True)
    with self.assertRaisesOpError('Condition x >= 0'):
      self.evaluate(d.log_prob([14.2, -5.3]))

  def testPdfOnBoundary(self):
    d = tfd.Chi(df=[1., 3.], validate_args=True)
    log_prob_boundary = self.evaluate(d.log_prob(0.))

    self.assertAllNegativeInf(log_prob_boundary[1])

    # TODO(b/144948687) Avoid `nan` log_prob on the boundary when `df==1`.
    # Ideally we'd do this test:
    # self.assertTrue(np.isfinite(log_prob_boundary[0]))
    # prob_boundary = self.evaluate(d.prob(0.))
    # self.assertAllFinite(prob_boundary))

  def testChiCDF(self):
    df = np.arange(1, 6, dtype=np.float64)
    x = np.arange(1, 6, dtype=np.float64)

    df = df.reshape((len(df), 1))
    x = x.reshape((1, len(x)))

    chi = tfd.Chi(df=df, validate_args=True)
    expected_cdf = stats.chi.cdf(x, df)

    cdf = chi.cdf(x)
    self.assertEqual(cdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testChiMean(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_mean = stats.chi.mean(df)
    chi = tfd.Chi(df=df, validate_args=True)
    self.assertEqual(chi.mean().shape, df.shape)
    self.assertAllClose(self.evaluate(chi.mean()), expected_mean)

  def testChiVariance(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_variances = stats.chi.var(df)
    chi = tfd.Chi(df=df, validate_args=True)
    self.assertEqual(chi.variance().shape, df.shape)
    self.assertAllClose(self.evaluate(chi.variance()), expected_variances)

  def testChiEntropy(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_entropy = stats.chi.entropy(df)
    chi = tfd.Chi(df=df, validate_args=True)
    self.assertEqual(chi.entropy().shape, df.shape)
    self.assertAllClose(self.evaluate(chi.entropy()), expected_entropy)

  def testChiChiKL(self):
    # We make sure a_df and b_df don't have any overlap. If this is not done,
    # then the check for true_kl vs kl_sample_ ends up failing because the
    # true_kl is zero and the sample is nonzero (though very small).
    a_df = np.arange(1, 6, step=2, dtype=np.float64)
    b_df = np.arange(2, 7, step=2, dtype=np.float64)

    a_df = a_df.reshape((len(a_df), 1))
    b_df = b_df.reshape((1, len(b_df)))

    a = tfd.Chi(df=a_df, validate_args=True)
    b = tfd.Chi(df=b_df, validate_args=True)

    true_kl = (0.5 * special.digamma(0.5 * a_df) * (a_df - b_df) +
               special.gammaln(0.5 * b_df) - special.gammaln(0.5 * a_df))

    kl = tfd.kl_divergence(a, b)

    x = a.sample(
        int(8e5),
        seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)

    kl_, kl_sample_ = self.evaluate([kl, kl_sample])
    self.assertAllClose(true_kl, kl_, atol=0., rtol=1e-12)
    self.assertAllClose(true_kl, kl_sample_, atol=0., rtol=5e-2)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(zero_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    df = tf.Variable(19.43, dtype=tf.float64)
    d = tfd.Chi(df, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 3.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveDf(self):
    df = tf.Variable([1., 2., -3.])
    with self.assertRaisesOpError('Argument `df` must be positive.'):
      d = tfd.Chi(df, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.entropy())

  def testAssertsPositiveDfAfterMutation(self):
    df = tf.Variable([1., 2., 3.])
    d = tfd.Chi(df, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Argument `df` must be positive.'):
      with tf.control_dependencies([df.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testSupportBijectorOutsideRange(self):
    df = np.array([2., 4., 7.])
    dist = tfd.Chi(df, validate_args=True)
    x = np.array([-8.3, -0.4, -1e-6])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  tf.test.main()
