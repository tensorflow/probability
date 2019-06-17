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
from tensorflow_probability.python.internal import test_util as tfp_test_util

tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class BinomialTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testSimpleShapes(self):
    p = np.float32(np.random.beta(1, 1))
    binom = tfd.Binomial(total_count=1., probs=p)
    self.assertAllEqual([], self.evaluate(binom.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(binom.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), binom.event_shape)
    self.assertEqual(tf.TensorShape([]), binom.batch_shape)

  def testComplexShapes(self):
    p = np.random.beta(1, 1, size=(3, 2)).astype(np.float32)
    n = [[3., 2], [4, 5], [6, 7]]
    binom = tfd.Binomial(total_count=n, probs=p)
    self.assertAllEqual([], self.evaluate(binom.event_shape_tensor()))
    self.assertAllEqual([3, 2], self.evaluate(binom.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), binom.event_shape)
    self.assertEqual(tf.TensorShape([3, 2]), binom.batch_shape)

  def testNProperty(self):
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    binom = tfd.Binomial(total_count=n, probs=p)
    self.assertEqual((2, 1), binom.total_count.shape)
    self.assertAllClose(n, self.evaluate(binom.total_count))

  def testPProperty(self):
    p = [[0.1, 0.2, 0.7]]
    binom = tfd.Binomial(total_count=3., probs=p)
    self.assertEqual((1, 3), binom.probs.shape)
    self.assertEqual((1, 3), binom.logits.shape)
    self.assertAllClose(p, self.evaluate(binom.probs))

  def testLogitsProperty(self):
    logits = [[0., 9., -0.5]]
    binom = tfd.Binomial(total_count=3., logits=logits)
    self.assertEqual((1, 3), binom.probs.shape)
    self.assertEqual((1, 3), binom.logits.shape)
    self.assertAllClose(logits, self.evaluate(binom.logits))

  def testPmfAndCdfNandCountsAgree(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    self.evaluate(binom.prob([2., 3, 2]))
    self.evaluate(binom.prob([3., 1, 2]))
    self.evaluate(binom.cdf([2., 3, 2]))
    self.evaluate(binom.cdf([3., 1, 2]))
    with self.assertRaisesOpError("Condition x >= 0.*"):
      self.evaluate(binom.prob([-1., 4, 2]))
    with self.assertRaisesOpError("Condition x <= y.*"):
      self.evaluate(binom.prob([7., 3, 0]))
    with self.assertRaisesOpError("Condition x >= 0.*"):
      self.evaluate(binom.cdf([-1., 4, 2]))
    with self.assertRaisesOpError("Condition x <= y.*"):
      self.evaluate(binom.cdf([7., 3, 0]))

  def testPmfAndCdfNonIntegerCounts(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    # No errors with integer n.
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    self.evaluate(binom.prob([2., 3, 2]))
    self.evaluate(binom.prob([3., 1, 2]))
    self.evaluate(binom.cdf([2., 3, 2]))
    self.evaluate(binom.cdf([3., 1, 2]))
    placeholder = tf.compat.v1.placeholder_with_default(
        input=[1.0, 2.5, 1.5], shape=[3])
    # Both equality and integer checking fail.
    with self.assertRaisesOpError("cannot contain fractional components."):
      self.evaluate(binom.prob(placeholder))
    with self.assertRaisesOpError("cannot contain fractional components."):
      self.evaluate(binom.cdf(placeholder))

    binom = tfd.Binomial(total_count=n, probs=p, validate_args=False)
    self.evaluate(binom.prob([1., 2., 3.]))
    self.evaluate(binom.cdf([1., 2., 3.]))
    # Non-integer arguments work.
    self.evaluate(binom.prob([1.0, 2.5, 1.5]))
    self.evaluate(binom.cdf([1.0, 2.5, 1.5]))

  def testPmfAndCdfBothZeroBatches(self):
    # Both zero-batches.  No broadcast
    p = 0.5
    counts = 1.
    binom = tfd.Binomial(total_count=1., probs=p)
    pmf = binom.prob(counts)
    cdf = binom.cdf(counts)
    self.assertAllClose(0.5, self.evaluate(pmf))
    self.assertAllClose(stats.binom.cdf(counts, n=1, p=p), self.evaluate(cdf))
    self.assertEqual((), pmf.shape)
    self.assertEqual((), cdf.shape)

  def testPmfAndCdfBothZeroBatchesNontrivialN(self):
    # Both zero-batches.  No broadcast
    p = 0.1
    counts = 3.
    binom = tfd.Binomial(total_count=5., probs=p)
    pmf = binom.prob(counts)
    cdf = binom.cdf(counts)
    self.assertAllClose(stats.binom.pmf(counts, n=5., p=p), self.evaluate(pmf))
    self.assertAllClose(stats.binom.cdf(counts, n=5., p=p), self.evaluate(cdf))
    self.assertEqual((), pmf.shape)
    self.assertEqual((), cdf.shape)

  def testPmfAndCdfPStretchedInBroadcastWhenSameRank(self):
    p = [[0.1, 0.9]]
    counts = [[1., 2.]]
    binom = tfd.Binomial(total_count=3., probs=p)
    pmf = binom.prob(counts)
    cdf = binom.cdf(counts)
    self.assertAllClose(stats.binom.pmf(counts, n=3., p=p), self.evaluate(pmf))
    self.assertAllClose(stats.binom.cdf(counts, n=3., p=p), self.evaluate(cdf))
    self.assertEqual((1, 2), pmf.shape)
    self.assertEqual((1, 2), cdf.shape)

  def testPmfAndCdfPStretchedInBroadcastWhenLowerRank(self):
    p = [0.1, 0.4]
    counts = [[1.], [0.]]
    binom = tfd.Binomial(total_count=1., probs=p)
    pmf = binom.prob(counts)
    cdf = binom.cdf(counts)
    self.assertAllClose([[0.1, 0.4], [0.9, 0.6]], self.evaluate(pmf))
    self.assertAllClose([[1.0, 1.0], [0.9, 0.6]], self.evaluate(cdf))
    self.assertEqual((2, 2), pmf.shape)
    self.assertEqual((2, 2), cdf.shape)

  def testBinomialMean(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p)
    expected_means = stats.binom.mean(n, p)
    self.assertEqual((3,), binom.mean().shape)
    self.assertAllClose(expected_means, self.evaluate(binom.mean()))

  def testBinomialVariance(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p)
    expected_variances = stats.binom.var(n, p)
    self.assertEqual((3,), binom.variance().shape)
    self.assertAllClose(expected_variances, self.evaluate(binom.variance()))

  def testBinomialMode(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p)
    expected_modes = [0., 1, 4]
    self.assertEqual((3,), binom.mode().shape)
    self.assertAllClose(expected_modes, self.evaluate(binom.mode()))

  def testBinomialMultipleMode(self):
    n = 9.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p)
    # For the case where (n + 1) * p is an integer, the modes are:
    # (n + 1) * p and (n + 1) * p - 1. In this case, we get back
    # the larger of the two modes.
    expected_modes = [1., 2, 7]
    self.assertEqual((3,), binom.mode().shape)
    self.assertAllClose(expected_modes, self.evaluate(binom.mode()))

  def testSampleUnbiasedNonScalarBatch(self):
    probs = self._rng.rand(4, 3).astype(np.float32)
    counts = np.float32([4, 11., 20.])
    dist = tfd.Binomial(total_count=counts, probs=probs)
    n = int(1e5)
    x = dist.sample(n, seed=tfp_test_util.test_seed())
    sample_mean, sample_variance = tf.nn.moments(x=x, axes=0)
    [
        sample_mean_,
        sample_variance_,
    ] = self.evaluate([
        sample_mean,
        sample_variance,
    ])
    self.assertAllEqual([4, 3], sample_mean.shape)
    self.assertAllClose(
        stats.binom.mean(counts, probs), sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4, 3], sample_variance.shape)
    self.assertAllClose(
        stats.binom.var(counts, probs), sample_variance_, atol=0., rtol=0.20)

  def testSampleUnbiasedScalarBatch(self):
    counts = np.float32(5.)
    probs = self._rng.rand(4).astype(np.float32)
    dist = tfd.Binomial(total_count=counts, probs=probs)
    n = int(1e5)
    x = dist.sample(n, seed=tfp_test_util.test_seed())
    sample_mean, sample_variance = tf.nn.moments(x=x, axes=0)
    [
        sample_mean_,
        sample_variance_,
    ] = self.evaluate([
        sample_mean,
        sample_variance,
    ])
    self.assertAllEqual([4], sample_mean.shape)
    self.assertAllClose(
        stats.binom.mean(counts, probs), sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4], sample_variance.shape)
    self.assertAllClose(
        stats.binom.var(counts, probs), sample_variance_, atol=0., rtol=0.20)

  def testParamTensorFromLogits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = tfd.Binomial(total_count=1, logits=x, validate_args=True)
    logit = lambda x: tf.math.log(x) - tf.math.log1p(-x)
    self.assertAllClose(
        *self.evaluate([logit(d.prob(1.)), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([d.prob(1.), d.probs_parameter()]),
        atol=0, rtol=1e-4)

  def testParamTensorFromProbs(self):
    x = tf.constant([0.1, 0.5, 0.4])
    d = tfd.Binomial(total_count=1, probs=x, validate_args=True)
    logit = lambda x: tf.math.log(x) - tf.math.log1p(-x)
    self.assertAllClose(
        *self.evaluate([logit(d.prob(1.)), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([d.prob(1.), d.probs_parameter()]),
        atol=0, rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()
