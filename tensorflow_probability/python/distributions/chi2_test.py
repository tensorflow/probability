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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import special
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class Chi2Test(tf.test.TestCase):

  def testChi2LogPDF(self):
    batch_size = 6
    df = tf.constant([2.0] * batch_size, dtype=np.float64)
    df_v = 2.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float64)
    chi2 = tfd.Chi2(df=df)
    expected_log_pdf = stats.chi2.logpdf(x, df_v)

    log_pdf = chi2.log_prob(x)
    self.assertEqual(log_pdf.shape, (6,))
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

    pdf = chi2.prob(x)
    self.assertEqual(pdf.shape, (6,))
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testChi2CDF(self):
    batch_size = 6
    df = tf.constant([2.0] * batch_size, dtype=np.float64)
    df_v = 2.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float64)

    chi2 = tfd.Chi2(df=df)
    expected_cdf = stats.chi2.cdf(x, df_v)

    cdf = chi2.cdf(x)
    self.assertEqual(cdf.shape, (6,))
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testChi2Mean(self):
    df_v = np.array([1., 3, 5], dtype=np.float64)
    expected_mean = stats.chi2.mean(df_v)
    chi2 = tfd.Chi2(df=df_v)
    self.assertEqual(chi2.mean().shape, (3,))
    self.assertAllClose(self.evaluate(chi2.mean()), expected_mean)

  def testChi2Variance(self):
    df_v = np.array([1., 3, 5], np.float64)
    expected_variances = stats.chi2.var(df_v)
    chi2 = tfd.Chi2(df=df_v)
    self.assertEqual(chi2.variance().shape, (3,))
    self.assertAllClose(self.evaluate(chi2.variance()), expected_variances)

  def testChi2Entropy(self):
    df_v = np.array([1., 3, 5], dtype=np.float64)
    expected_entropy = stats.chi2.entropy(df_v)
    chi2 = tfd.Chi2(df=df_v)
    self.assertEqual(chi2.entropy().shape, (3,))
    self.assertAllClose(self.evaluate(chi2.entropy()), expected_entropy)

  def testChi2WithAbsDf(self):
    df_v = np.array([-1.3, -3.2, 5], dtype=np.float64)
    chi2 = tfd.Chi2WithAbsDf(df=df_v)
    self.assertAllClose(
        self.evaluate(tf.floor(tf.abs(df_v))), self.evaluate(chi2.df))

  def testChi2Chi2KL(self):
    a_df = np.arange(1.0, 10.0)
    b_df = np.arange(1.0, 10.0)

    # This reshape is intended to expand the number of test cases.
    a_df = a_df.reshape((len(a_df), 1))
    b_df = b_df.reshape((1, len(b_df)))

    a = tfd.Chi2(df=a_df)
    b = tfd.Chi2(df=b_df)

    # Consistent with
    # http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 110
    true_kl = (special.gammaln(b_df / 2.0) - special.gammaln(a_df / 2.0) +
               (a_df - b_df) / 2.0 * special.digamma(a_df / 2.0))

    kl = tfd.kl_divergence(a, b)

    x = a.sample(int(1e5), seed=0)
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)

    kl_, kl_sample_ = self.evaluate([kl, kl_sample])
    self.assertAllClose(true_kl, kl_, atol=0., rtol=5e-13)
    self.assertAllClose(true_kl, kl_sample_, atol=0., rtol=5e-2)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(zero_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

if __name__ == "__main__":
  tf.test.main()
