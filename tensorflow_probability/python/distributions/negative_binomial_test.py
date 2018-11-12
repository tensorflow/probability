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
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util

tfd = tfp.distributions


# In all tests that follow, we use scipy.stats.nbinom, which
# represents a Negative Binomial distribution, with success and failure
# probabilities flipped.
@test_util.run_all_in_graph_and_eager_modes
class NegativeBinomialTest(tf.test.TestCase):

  def testNegativeBinomialShape(self):
    probs = [.1] * 5
    total_count = [2.0] * 5
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)

    self.assertEqual([5], self.evaluate(negbinom.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([5]), negbinom.batch_shape)
    self.assertAllEqual([], self.evaluate(negbinom.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), negbinom.event_shape)

  def testNegativeBinomialShapeBroadcast(self):
    probs = [[.1, .2, .3]] * 5
    total_count = [[2.]] * 5
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)

    self.assertAllEqual([5, 3], self.evaluate(negbinom.batch_shape_tensor()))
    self.assertAllEqual(tf.TensorShape([5, 3]), negbinom.batch_shape)
    self.assertAllEqual([], self.evaluate(negbinom.event_shape_tensor()))
    self.assertAllEqual(tf.TensorShape([]), negbinom.event_shape)

  def testLogits(self):
    logits = [[0., 9., -0.5]]
    negbinom = tfd.NegativeBinomial(total_count=3., logits=logits)
    self.assertEqual([1, 3], negbinom.probs.shape)
    self.assertEqual([1, 3], negbinom.logits.shape)
    self.assertAllClose(logits, self.evaluate(negbinom.logits))

  def testInvalidP(self):
    invalid_ps = [-.01, 0., -2.,]
    with self.assertRaisesOpError("Condition x >= 0"):
      negbinom = tfd.NegativeBinomial(5., probs=invalid_ps, validate_args=True)
      self.evaluate(negbinom.probs)

    invalid_ps = [1.01, 2., 1.001,]
    with self.assertRaisesOpError("probs has components greater than 1."):
      negbinom = tfd.NegativeBinomial(5., probs=invalid_ps, validate_args=True)
      self.evaluate(negbinom.probs)

  def testInvalidNegativeCount(self):
    invalid_rs = [-.01, 0., -2.,]
    with self.assertRaisesOpError("Condition x > 0"):
      negbinom = tfd.NegativeBinomial(
          total_count=invalid_rs, probs=0.1, validate_args=True)
      self.evaluate(negbinom.total_count)

  def testNegativeBinomialLogCdf(self):
    batch_size = 6
    probs = [.2] * batch_size
    probs_v = .2
    total_count = 5.
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=np.float32)
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)
    expected_log_cdf = stats.nbinom.logcdf(x, n=total_count, p=1 - probs_v)
    log_cdf = negbinom.log_cdf(x)
    self.assertEqual([6], log_cdf.shape)
    self.assertAllClose(expected_log_cdf, self.evaluate(log_cdf))

    cdf = negbinom.cdf(x)
    self.assertEqual([6], cdf.shape)
    self.assertAllClose(np.exp(expected_log_cdf), self.evaluate(cdf))

  def testNegativeBinomialLogCdfValidateArgs(self):
    batch_size = 6
    probs = [.9] * batch_size
    total_count = 5.
    with self.assertRaisesOpError("Condition x >= 0"):
      negbinom = tfd.NegativeBinomial(
          total_count=total_count, probs=probs, validate_args=True)
      self.evaluate(negbinom.log_cdf(-1.))

  def testNegativeBinomialLogPmf(self):
    batch_size = 6
    probs = [.2] * batch_size
    probs_v = .2
    total_count = 5.
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=np.float32)
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)
    expected_log_pmf = stats.nbinom.logpmf(x, n=total_count, p=1 - probs_v)
    log_pmf = negbinom.log_prob(x)
    self.assertEqual([6], log_pmf.shape)
    self.assertAllClose(expected_log_pmf, self.evaluate(log_pmf))

    pmf = negbinom.prob(x)
    self.assertEqual([6], pmf.shape)
    self.assertAllClose(np.exp(expected_log_pmf), self.evaluate(pmf))

  def testNegativeBinomialLogPmfValidateArgs(self):
    batch_size = 6
    probs = [.9] * batch_size
    total_count = 5.
    x = tf.placeholder_with_default(
        input=[2.5, 3.2, 4.3, 5.1, 6., 7.], shape=[6])
    negbinom = tfd.NegativeBinomial(
        total_count=total_count, probs=probs, validate_args=True)

    with self.assertRaisesOpError("Condition x == y"):
      self.evaluate(negbinom.log_prob(x))

    with self.assertRaisesOpError("Condition x >= 0"):
      self.evaluate(negbinom.log_prob([-1.]))

    negbinom = tfd.NegativeBinomial(
        total_count=total_count, probs=probs, validate_args=False)
    log_pmf = negbinom.log_prob(x)
    self.assertEqual([6], log_pmf.shape)
    pmf = negbinom.prob(x)
    self.assertEqual([6], pmf.shape)

  def testNegativeBinomialLogPmfMultidimensional(self):
    batch_size = 6
    probs = tf.constant([[.2, .3, .5]] * batch_size)
    probs_v = np.array([.2, .3, .5])
    total_count = 5.
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)
    expected_log_pmf = stats.nbinom.logpmf(x, n=total_count, p=1 - probs_v)
    log_pmf = negbinom.log_prob(x)
    log_pmf_values = self.evaluate(log_pmf)
    self.assertEqual([6, 3], log_pmf.shape)
    self.assertAllClose(expected_log_pmf, log_pmf_values)

    pmf = negbinom.prob(x)
    pmf_values = self.evaluate(pmf)
    self.assertEqual([6, 3], pmf.shape)
    self.assertAllClose(np.exp(expected_log_pmf), pmf_values)

  def testNegativeBinomialMean(self):
    total_count = 5.
    probs = np.array([.1, .3, .25], dtype=np.float32)
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)
    expected_means = stats.nbinom.mean(n=total_count, p=1 - probs)
    self.assertEqual([3], negbinom.mean().shape)
    self.assertAllClose(expected_means, self.evaluate(negbinom.mean()))

  def testNegativeBinomialVariance(self):
    total_count = 5.
    probs = np.array([.1, .3, .25], dtype=np.float32)
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)
    expected_vars = stats.nbinom.var(n=total_count, p=1 - probs)
    self.assertEqual([3], negbinom.variance().shape)
    self.assertAllClose(expected_vars, self.evaluate(negbinom.variance()))

  def testNegativeBinomialStddev(self):
    total_count = 5.
    probs = np.array([.1, .3, .25], dtype=np.float32)
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)
    expected_stds = stats.nbinom.std(n=total_count, p=1 - probs)
    self.assertEqual([3], negbinom.stddev().shape)
    self.assertAllClose(expected_stds, self.evaluate(negbinom.stddev()))

  def testNegativeBinomialSample(self):
    probs = [.3, .9]
    total_count = [4., 11.]
    n = int(100e3)
    negbinom = tfd.NegativeBinomial(total_count=total_count, probs=probs)

    samples = negbinom.sample(n, seed=12345)
    self.assertEqual([n, 2], samples.shape)

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_var = tf.reduce_mean(
        (samples - sample_mean[tf.newaxis, ...])**2., axis=0)
    sample_min = tf.reduce_min(samples)
    [sample_mean_, sample_var_,
     sample_min_] = self.evaluate([sample_mean, sample_var, sample_min])
    self.assertAllEqual(
        np.ones(sample_min_.shape, dtype=np.bool), sample_min_ >= 0.0)
    for i in range(2):
      self.assertAllClose(
          sample_mean_[i],
          stats.nbinom.mean(total_count[i], 1 - probs[i]),
          atol=0.,
          rtol=.02)
      self.assertAllClose(
          sample_var_[i],
          stats.nbinom.var(total_count[i], 1 - probs[i]),
          atol=0.,
          rtol=.02)

  def testLogProbOverflow(self):
    logits = np.float32([20., 30., 40.])
    total_count = np.float32(1.)
    x = np.float32(0.)
    nb = tfd.NegativeBinomial(total_count=total_count, logits=logits)
    log_prob_ = self.evaluate(nb.log_prob(x))
    self.assertAllEqual(
        np.ones_like(log_prob_, dtype=np.bool), np.isfinite(log_prob_))

  def testLogProbUnderflow(self):
    logits = np.float32([-90, -100, -110])
    total_count = np.float32(1.)
    x = np.float32(0.)
    nb = tfd.NegativeBinomial(total_count=total_count, logits=logits)
    log_prob_ = self.evaluate(nb.log_prob(x))
    self.assertAllEqual(
        np.ones_like(log_prob_, dtype=np.bool), np.isfinite(log_prob_))


if __name__ == "__main__":
  tf.test.main()
