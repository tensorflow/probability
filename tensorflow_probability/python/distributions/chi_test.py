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
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class ChiTest(tf.test.TestCase):

  def testChiLogPDF(self):
    df = np.arange(1, 6, dtype=np.float64)
    x = np.arange(1, 6, dtype=np.float64)

    df = df.reshape((len(df), 1))
    x = x.reshape((1, len(x)))

    chi = tfd.Chi(df=df)
    expected_log_pdf = stats.chi.logpdf(x, df)

    log_pdf = chi.log_prob(x)
    self.assertEqual(log_pdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

    pdf = chi.prob(x)
    self.assertEqual(pdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testChiCDF(self):
    df = np.arange(1, 6, dtype=np.float64)
    x = np.arange(1, 6, dtype=np.float64)

    df = df.reshape((len(df), 1))
    x = x.reshape((1, len(x)))

    chi = tfd.Chi(df=df)
    expected_cdf = stats.chi.cdf(x, df)

    cdf = chi.cdf(x)
    self.assertEqual(cdf.shape, np.broadcast(df, x).shape)
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testChiMean(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_mean = stats.chi.mean(df)
    chi = tfd.Chi(df=df)
    self.assertEqual(chi.mean().shape, df.shape)
    self.assertAllClose(self.evaluate(chi.mean()), expected_mean)

  def testChiVariance(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_variances = stats.chi.var(df)
    chi = tfd.Chi(df=df)
    self.assertEqual(chi.variance().shape, df.shape)
    self.assertAllClose(self.evaluate(chi.variance()), expected_variances)

  def testChiEntropy(self):
    df = np.arange(1, 6, dtype=np.float64)
    expected_entropy = stats.chi.entropy(df)
    chi = tfd.Chi(df=df)
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

    a = tfd.Chi(df=a_df)
    b = tfd.Chi(df=b_df)

    true_kl = (0.5 * special.digamma(0.5 * a_df) * (a_df - b_df) +
               special.gammaln(0.5 * b_df) - special.gammaln(0.5 * a_df))

    kl = tfd.kl_divergence(a, b)

    x = a.sample(int(8e5), seed=0)
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)

    kl_, kl_sample_ = self.evaluate([kl, kl_sample])
    self.assertAllClose(true_kl, kl_, atol=0., rtol=1e-14)
    self.assertAllClose(true_kl, kl_sample_, atol=0., rtol=5e-3)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(zero_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

if __name__ == '__main__':
  tf.test.main()
