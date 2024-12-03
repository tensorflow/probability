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

import numpy as np
from scipy import special
from scipy import stats

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import chi
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ChiTest(test_util.TestCase):

  def testChiLogPDF(self):
    df = np.arange(1, 6, dtype=np.float64)
    x = np.arange(1, 6, dtype=np.float64)

    df = df.reshape((len(df), 1))
    x = x.reshape((1, len(x)))

    dist = chi.Chi(df=df, validate_args=True)
    expected_log_pdf = stats.chi.logpdf(x, df)

    log_pdf = dist.log_prob(x)
    self.assertEqual(log_pdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

    pdf = dist.prob(x)
    self.assertEqual(pdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testLogPdfAssertsOnInvalidSample(self):
    d = chi.Chi(df=13.37, validate_args=True)
    with self.assertRaisesOpError('Condition x >= 0'):
      self.evaluate(d.log_prob([14.2, -5.3]))

  def testPdfOnBoundary(self):
    d = chi.Chi(df=[1., 3.], validate_args=True)
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

    dist = chi.Chi(df=df, validate_args=True)
    expected_cdf = stats.chi.cdf(x, df)

    cdf = dist.cdf(x)
    self.assertEqual(cdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testChiMean(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_mean = stats.chi.mean(df)
    dist = chi.Chi(df=df, validate_args=True)
    self.assertEqual(dist.mean().shape, df.shape)
    self.assertAllClose(self.evaluate(dist.mean()), expected_mean)

  def testChiVariance(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_variances = stats.chi.var(df)
    dist = chi.Chi(df=df, validate_args=True)
    self.assertEqual(dist.variance().shape, df.shape)
    self.assertAllClose(self.evaluate(dist.variance()), expected_variances)

  def testChiEntropy(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_entropy = stats.chi.entropy(df)
    dist = chi.Chi(df=df, validate_args=True)
    self.assertEqual(dist.entropy().shape, df.shape)
    self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testChiChiKL(self):
    # We make sure a_df and b_df don't have any overlap. If this is not done,
    # then the check for true_kl vs kl_sample_ ends up failing because the
    # true_kl is zero and the sample is nonzero (though very small).
    a_df = np.arange(1, 6, step=2, dtype=np.float64)
    b_df = np.arange(2, 7, step=2, dtype=np.float64)

    a_df = a_df.reshape((len(a_df), 1))
    b_df = b_df.reshape((1, len(b_df)))

    a = chi.Chi(df=a_df, validate_args=True)
    b = chi.Chi(df=b_df, validate_args=True)

    true_kl = (0.5 * special.digamma(0.5 * a_df) * (a_df - b_df) +
               special.gammaln(0.5 * b_df) - special.gammaln(0.5 * a_df))

    kl = kullback_leibler.kl_divergence(a, b)

    x = a.sample(
        int(8e5),
        seed=test_util.test_seed())
    kl_samples = a.log_prob(x) - b.log_prob(x)

    kl_, kl_samples_ = self.evaluate([kl, kl_samples])
    self.assertAllClose(kl_, true_kl, atol=0., rtol=1e-12)
    self.assertAllMeansClose(kl_samples_, true_kl, axis=0, atol=0., rtol=5e-2)

    zero_kl = kullback_leibler.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(zero_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    df = tf.Variable(19.43, dtype=tf.float64)
    d = chi.Chi(df, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 3.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveDf(self):
    df = tf.Variable([1., 2., -3.])
    with self.assertRaisesOpError('Argument `df` must be positive.'):
      d = chi.Chi(df, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.entropy())

  def testAssertsPositiveDfAfterMutation(self):
    df = tf.Variable([1., 2., 3.])
    d = chi.Chi(df, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Argument `df` must be positive.'):
      with tf.control_dependencies([df.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testSupportBijectorOutsideRange(self):
    df = np.array([2., 4., 7.])
    dist = chi.Chi(df, validate_args=True)
    x = np.array([-8.3, -0.4, -1e-6])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  test_util.main()
