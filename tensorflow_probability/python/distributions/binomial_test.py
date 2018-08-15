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

tfd = tfp.distributions


class BinomialTest(tf.test.TestCase):

  def testSimpleShapes(self):
    with self.test_session():
      p = np.float32(np.random.beta(1, 1))
      binom = tfd.Binomial(total_count=1., probs=p)
      self.assertAllEqual([], self.evaluate(binom.event_shape_tensor()))
      self.assertAllEqual([], self.evaluate(binom.batch_shape_tensor()))
      self.assertEqual(tf.TensorShape([]), binom.event_shape)
      self.assertEqual(tf.TensorShape([]), binom.batch_shape)

  def testComplexShapes(self):
    with self.test_session():
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
    with self.test_session():
      binom = tfd.Binomial(total_count=n, probs=p)
      self.assertEqual((2, 1), binom.total_count.get_shape())
      self.assertAllClose(n, self.evaluate(binom.total_count))

  def testPProperty(self):
    p = [[0.1, 0.2, 0.7]]
    with self.test_session():
      binom = tfd.Binomial(total_count=3., probs=p)
      self.assertEqual((1, 3), binom.probs.get_shape())
      self.assertEqual((1, 3), binom.logits.get_shape())
      self.assertAllClose(p, self.evaluate(binom.probs))

  def testLogitsProperty(self):
    logits = [[0., 9., -0.5]]
    with self.test_session():
      binom = tfd.Binomial(total_count=3., logits=logits)
      self.assertEqual((1, 3), binom.probs.get_shape())
      self.assertEqual((1, 3), binom.logits.get_shape())
      self.assertAllClose(logits, self.evaluate(binom.logits))

  def testPmfAndCdfNandCountsAgree(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.test_session():
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
    with self.test_session():
      # No errors with integer n.
      binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
      self.evaluate(binom.prob([2., 3, 2]))
      self.evaluate(binom.prob([3., 1, 2]))
      self.evaluate(binom.cdf([2., 3, 2]))
      self.evaluate(binom.cdf([3., 1, 2]))
      placeholder = tf.placeholder(tf.float32)
      # Both equality and integer checking fail.
      with self.assertRaisesOpError(
          "cannot contain fractional components."):
        binom.prob(placeholder).eval(feed_dict={placeholder: [1.0, 2.5, 1.5]})
      with self.assertRaisesOpError(
          "cannot contain fractional components."):
        binom.cdf(placeholder).eval(feed_dict={placeholder: [1.0, 2.5, 1.5]})

      binom = tfd.Binomial(total_count=n, probs=p, validate_args=False)
      self.evaluate(binom.prob([1., 2., 3.]))
      self.evaluate(binom.cdf([1., 2., 3.]))
      # Non-integer arguments work.
      self.evaluate(binom.prob([1.0, 2.5, 1.5]))
      self.evaluate(binom.cdf([1.0, 2.5, 1.5]))

  def testPmfAndCdfBothZeroBatches(self):
    with self.test_session():
      # Both zero-batches.  No broadcast
      p = 0.5
      counts = 1.
      binom = tfd.Binomial(total_count=1., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose(0.5, self.evaluate(pmf))
      self.assertAllClose(stats.binom.cdf(counts, n=1, p=p), self.evaluate(cdf))
      self.assertEqual((), pmf.get_shape())
      self.assertEqual((), cdf.get_shape())

  def testPmfAndCdfBothZeroBatchesNontrivialN(self):
    with self.test_session():
      # Both zero-batches.  No broadcast
      p = 0.1
      counts = 3.
      binom = tfd.Binomial(total_count=5., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose(
          stats.binom.pmf(counts, n=5., p=p), self.evaluate(pmf))
      self.assertAllClose(
          stats.binom.cdf(counts, n=5., p=p), self.evaluate(cdf))
      self.assertEqual((), pmf.get_shape())
      self.assertEqual((), cdf.get_shape())

  def testPmfAndCdfPStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      p = [[0.1, 0.9]]
      counts = [[1., 2.]]
      binom = tfd.Binomial(total_count=3., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose(
          stats.binom.pmf(counts, n=3., p=p), self.evaluate(pmf))
      self.assertAllClose(
          stats.binom.cdf(counts, n=3., p=p), self.evaluate(cdf))
      self.assertEqual((1, 2), pmf.get_shape())
      self.assertEqual((1, 2), cdf.get_shape())

  def testPmfAndCdfPStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      p = [0.1, 0.4]
      counts = [[1.], [0.]]
      binom = tfd.Binomial(total_count=1., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose([[0.1, 0.4], [0.9, 0.6]], self.evaluate(pmf))
      self.assertAllClose([[1.0, 1.0], [0.9, 0.6]], self.evaluate(cdf))
      self.assertEqual((2, 2), pmf.get_shape())
      self.assertEqual((2, 2), cdf.get_shape())

  def testBinomialMean(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = tfd.Binomial(total_count=n, probs=p)
      expected_means = stats.binom.mean(n, p)
      self.assertEqual((3,), binom.mean().get_shape())
      self.assertAllClose(expected_means, self.evaluate(binom.mean()))

  def testBinomialVariance(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = tfd.Binomial(total_count=n, probs=p)
      expected_variances = stats.binom.var(n, p)
      self.assertEqual((3,), binom.variance().get_shape())
      self.assertAllClose(expected_variances, self.evaluate(binom.variance()))

  def testBinomialMode(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = tfd.Binomial(total_count=n, probs=p)
      expected_modes = [0., 1, 4]
      self.assertEqual((3,), binom.mode().get_shape())
      self.assertAllClose(expected_modes, self.evaluate(binom.mode()))

  def testBinomialMultipleMode(self):
    with self.test_session():
      n = 9.
      p = [0.1, 0.2, 0.7]
      binom = tfd.Binomial(total_count=n, probs=p)
      # For the case where (n + 1) * p is an integer, the modes are:
      # (n + 1) * p and (n + 1) * p - 1. In this case, we get back
      # the larger of the two modes.
      expected_modes = [1., 2, 7]
      self.assertEqual((3,), binom.mode().get_shape())
      self.assertAllClose(expected_modes, self.evaluate(binom.mode()))


if __name__ == "__main__":
  tf.test.main()
