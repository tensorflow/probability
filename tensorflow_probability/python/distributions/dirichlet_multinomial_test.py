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
import tensorflow as tf

from tensorflow_probability.python.distributions import dirichlet_multinomial
from tensorflow.python.eager import backprop


ds = dirichlet_multinomial


class DirichletMultinomialTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testSimpleShapes(self):
    with self.cached_session():
      alpha = np.random.rand(3)
      dist = ds.DirichletMultinomial(1., alpha)
      self.assertEqual(3, dist.event_shape_tensor().eval())
      self.assertAllEqual([], dist.batch_shape_tensor().eval())
      self.assertEqual(tf.TensorShape([3]), dist.event_shape)
      self.assertEqual(tf.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    with self.cached_session():
      alpha = np.random.rand(3, 2, 2)
      n = [[3., 2], [4, 5], [6, 7]]
      dist = ds.DirichletMultinomial(n, alpha)
      self.assertEqual(2, dist.event_shape_tensor().eval())
      self.assertAllEqual([3, 2], dist.batch_shape_tensor().eval())
      self.assertEqual(tf.TensorShape([2]), dist.event_shape)
      self.assertEqual(tf.TensorShape([3, 2]), dist.batch_shape)

  def testNproperty(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(n, alpha)
      self.assertEqual([1, 1], dist.total_count.shape)
      self.assertAllClose(n, dist.total_count.eval())

  def testAlphaProperty(self):
    alpha = [[1., 2, 3]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(1, alpha)
      self.assertEqual([1, 3], dist.concentration.shape)
      self.assertAllClose(alpha, dist.concentration.eval())

  def testPmfNandCountsAgree(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
      dist.prob([2., 3, 0]).eval()
      dist.prob([3., 0, 2]).eval()
      with self.assertRaisesOpError("must be non-negative"):
        dist.prob([-1., 4, 2]).eval()
      with self.assertRaisesOpError(
          "last-dimension must sum to `self.total_count`"):
        dist.prob([3., 3, 0]).eval()

  def testPmfNonIntegerCounts(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
      dist.prob([2., 3, 0]).eval()
      dist.prob([3., 0, 2]).eval()
      dist.prob([3.0, 0, 2.0]).eval()
      # Both equality and integer checking fail.
      placeholder = tf.placeholder(tf.float32)
      with self.assertRaisesOpError(
          "cannot contain fractional components"):
        dist.prob(placeholder).eval(feed_dict={placeholder: [1.0, 2.5, 1.5]})
      dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
      dist.prob([1., 2., 3.]).eval()
      # Non-integer arguments work.
      dist.prob([1.0, 2.5, 1.5]).eval()

  def testPmfBothZeroBatches(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      # Both zero-batches.  No broadcast
      alpha = [1., 2]
      counts = [1., 0]
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1 / 3., pmf.eval())
      self.assertEqual((), pmf.shape)

  def testPmfBothZeroBatchesNontrivialN(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      # Both zero-batches.  No broadcast
      alpha = [1., 2]
      counts = [3., 2]
      dist = ds.DirichletMultinomial(5., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1 / 7., pmf.eval())
      self.assertEqual((), pmf.shape)

  def testPmfBothZeroBatchesMultidimensionalN(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [1., 2]
      counts = [3., 2]
      n = np.full([4, 3], 5., dtype=np.float32)
      dist = ds.DirichletMultinomial(n, alpha)
      pmf = dist.prob(counts)
      self.assertAllClose([[1 / 7., 1 / 7., 1 / 7.]] * 4, pmf.eval())
      self.assertEqual((4, 3), pmf.shape)

  def testPmfAlphaStretchedInBroadcastWhenSameRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [[1., 2]]
      counts = [[1., 0], [0., 1]]
      dist = ds.DirichletMultinomial([1.], alpha)
      pmf = dist.prob(counts)
      self.assertAllClose([1 / 3., 2 / 3.], pmf.eval())
      self.assertAllEqual([2], pmf.shape)

  def testPmfAlphaStretchedInBroadcastWhenLowerRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [1., 2]
      counts = [[1., 0], [0., 1]]
      pmf = ds.DirichletMultinomial(1., alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 3.], pmf.eval())
      self.assertAllEqual([2], pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [[1., 2], [2., 3]]
      counts = [[1., 0]]
      pmf = ds.DirichletMultinomial([1., 1.], alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 5.], pmf.eval())
      self.assertAllEqual([2], pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [[1., 2], [2., 3]]
      counts = [1., 0]
      pmf = ds.DirichletMultinomial(1., alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 5.], pmf.eval())
      self.assertAllEqual([2], pmf.shape)

  def testPmfForOneVoteIsTheMeanWithOneRecordInput(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [1., 2, 3]
    with self.cached_session():
      for class_num in range(3):
        counts = np.zeros([3], dtype=np.float32)
        counts[class_num] = 1
        dist = ds.DirichletMultinomial(1., alpha)
        mean = dist.mean().eval()
        pmf = dist.prob(counts).eval()

        self.assertAllClose(mean[class_num], pmf)
        self.assertAllEqual([3], mean.shape)
        self.assertAllEqual([], pmf.shape)

  def testMeanDoubleTwoVotes(self):
    # The probabilities of two votes falling into class k for
    # DirichletMultinomial(2, alpha) is twice as much as the probability of one
    # vote falling into class k for DirichletMultinomial(1, alpha)
    alpha = [1., 2, 3]
    with self.cached_session():
      for class_num in range(3):
        counts_one = np.zeros([3], dtype=np.float32)
        counts_one[class_num] = 1.
        counts_two = np.zeros([3], dtype=np.float32)
        counts_two[class_num] = 2

        dist1 = ds.DirichletMultinomial(1., alpha)
        dist2 = ds.DirichletMultinomial(2., alpha)

        mean1 = dist1.mean().eval()
        mean2 = dist2.mean().eval()

        self.assertAllClose(mean2[class_num], 2 * mean1[class_num])
        self.assertAllEqual([3], mean1.shape)

  def testCovarianceFromSampling(self):
    # We will test mean, cov, var, stddev on a DirichletMultinomial constructed
    # via broadcast between alpha, n.
    alpha = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    # Ideally we'd be able to test broadcasting but, the multinomial sampler
    # doesn't support different total counts.
    n = np.float32(5)
    with self.cached_session() as sess:
      # batch_shape=[2], event_shape=[3]
      dist = ds.DirichletMultinomial(n, alpha)
      x = dist.sample(int(250e3), seed=1)
      sample_mean = tf.reduce_mean(x, 0)
      x_centered = x - sample_mean[tf.newaxis, ...]
      sample_cov = tf.reduce_mean(tf.matmul(
          x_centered[..., tf.newaxis],
          x_centered[..., tf.newaxis, :]), 0)
      sample_var = tf.matrix_diag_part(sample_cov)
      sample_stddev = tf.sqrt(sample_var)
      [
          sample_mean_,
          sample_cov_,
          sample_var_,
          sample_stddev_,
          analytic_mean,
          analytic_cov,
          analytic_var,
          analytic_stddev,
      ] = sess.run([
          sample_mean,
          sample_cov,
          sample_var,
          sample_stddev,
          dist.mean(),
          dist.covariance(),
          dist.variance(),
          dist.stddev(),
      ])
      self.assertAllClose(sample_mean_, analytic_mean, atol=0.04, rtol=0.)
      self.assertAllClose(sample_cov_, analytic_cov, atol=0.05, rtol=0.)
      self.assertAllClose(sample_var_, analytic_var, atol=0.05, rtol=0.)
      self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.02, rtol=0.)

  def testCovariance(self):
    # Shape [2]
    alpha = [1., 2]
    ns = [2., 3., 4., 5.]
    alpha_0 = np.sum(alpha)

    # Diagonal entries are of the form:
    # Var(X_i) = n * alpha_i / alpha_sum * (1 - alpha_i / alpha_sum) *
    # (alpha_sum + n) / (alpha_sum + 1)
    variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
    # Off diagonal entries are of the form:
    # Cov(X_i, X_j) = -n * alpha_i * alpha_j / (alpha_sum ** 2) *
    # (alpha_sum + n) / (alpha_sum + 1)
    covariance_entry = lambda a, b, a_sum: -a * b / a_sum**2
    # Shape [2, 2].
    shared_matrix = np.array([[
        variance_entry(alpha[0], alpha_0),
        covariance_entry(alpha[0], alpha[1], alpha_0)
    ], [
        covariance_entry(alpha[1], alpha[0], alpha_0),
        variance_entry(alpha[1], alpha_0)
    ]])

    with self.cached_session():
      for n in ns:
        # n is shape [] and alpha is shape [2].
        dist = ds.DirichletMultinomial(n, alpha)
        covariance = dist.covariance()
        expected_covariance = n * (n + alpha_0) / (1 + alpha_0) * shared_matrix

        self.assertEqual([2, 2], covariance.shape)
        self.assertAllClose(expected_covariance, covariance.eval())

  def testCovarianceNAlphaBroadcast(self):
    alpha_v = [1., 2, 3]
    alpha_0 = 6.

    # Shape [4, 3]
    alpha = np.array(4 * [alpha_v], dtype=np.float32)
    # Shape [4, 1]
    ns = np.array([[2.], [3.], [4.], [5.]], dtype=np.float32)

    variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
    covariance_entry = lambda a, b, a_sum: -a * b / a_sum**2
    # Shape [4, 3, 3]
    shared_matrix = np.array(
        4 * [[[
            variance_entry(alpha_v[0], alpha_0),
            covariance_entry(alpha_v[0], alpha_v[1], alpha_0),
            covariance_entry(alpha_v[0], alpha_v[2], alpha_0)
        ], [
            covariance_entry(alpha_v[1], alpha_v[0], alpha_0),
            variance_entry(alpha_v[1], alpha_0),
            covariance_entry(alpha_v[1], alpha_v[2], alpha_0)
        ], [
            covariance_entry(alpha_v[2], alpha_v[0], alpha_0),
            covariance_entry(alpha_v[2], alpha_v[1], alpha_0),
            variance_entry(alpha_v[2], alpha_0)
        ]]],
        dtype=np.float32)

    with self.cached_session():
      # ns is shape [4, 1], and alpha is shape [4, 3].
      dist = ds.DirichletMultinomial(ns, alpha)
      covariance = dist.covariance()
      expected_covariance = shared_matrix * (
          ns * (ns + alpha_0) / (1 + alpha_0))[..., tf.newaxis]

      self.assertEqual([4, 3, 3], covariance.shape)
      self.assertAllClose(expected_covariance, covariance.eval())

  def testCovarianceMultidimensional(self):
    alpha = np.random.rand(3, 5, 4).astype(np.float32)
    alpha2 = np.random.rand(6, 3, 3).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5, 1]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1, 1]).astype(np.float32)

    with self.cached_session():
      dist = ds.DirichletMultinomial(ns, alpha)
      dist2 = ds.DirichletMultinomial(ns2, alpha2)

      covariance = dist.covariance()
      covariance2 = dist2.covariance()
      self.assertEqual([3, 5, 4, 4], covariance.shape)
      self.assertEqual([6, 3, 3, 3], covariance2.shape)

  def testZeroCountsResultsInPmfEqualToOne(self):
    # There is only one way for zero items to be selected, and this happens with
    # probability 1.
    alpha = [5, 0.5]
    counts = [0., 0]
    with self.cached_session():
      dist = ds.DirichletMultinomial(0., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1.0, pmf.eval())
      self.assertEqual((), pmf.shape)

  def testLargeTauGivesPreciseProbabilities(self):
    # If tau is large, we are doing coin flips with probability mu.
    mu = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    tau = np.array([100.], dtype=np.float32)
    alpha = tau * mu

    # One (three sided) coin flip.  Prob[coin 3] = 0.8.
    # Note that since it was one flip, value of tau didn't matter.
    counts = [0., 0, 1]
    with self.cached_session():
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.8, pmf.eval(), atol=1e-4)
      self.assertEqual((), pmf.shape)

    # Two (three sided) coin flips.  Prob[coin 3] = 0.8.
    counts = [0., 0, 2]
    with self.cached_session():
      dist = ds.DirichletMultinomial(2., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.8**2, pmf.eval(), atol=1e-2)
      self.assertEqual((), pmf.shape)

    # Three (three sided) coin flips.
    counts = [1., 0, 2]
    with self.cached_session():
      dist = ds.DirichletMultinomial(3., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(3 * 0.1 * 0.8 * 0.8, pmf.eval(), atol=1e-2)
      self.assertEqual((), pmf.shape)

  def testSmallTauPrefersCorrelatedResults(self):
    # If tau is small, then correlation between draws is large, so draws that
    # are both of the same class are more likely.
    mu = np.array([0.5, 0.5], dtype=np.float32)
    tau = np.array([0.1], dtype=np.float32)
    alpha = tau * mu

    # If there is only one draw, it is still a coin flip, even with small tau.
    counts = [1., 0]
    with self.cached_session():
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.5, pmf.eval())
      self.assertEqual((), pmf.shape)

    # If there are two draws, it is much more likely that they are the same.
    counts_same = [2., 0]
    counts_different = [1, 1.]
    with self.cached_session():
      dist = ds.DirichletMultinomial(2., alpha)
      pmf_same = dist.prob(counts_same)
      pmf_different = dist.prob(counts_different)
      self.assertLess(5 * pmf_different.eval(), pmf_same.eval())
      self.assertEqual((), pmf_same.shape)

  def testNonStrictTurnsOffAllChecks(self):
    # Make totally invalid input.
    with self.cached_session():
      alpha = [[-1., 2]]  # alpha should be positive.
      counts = [[1., 0], [0., -1]]  # counts should be non-negative.
      n = [-5.3]  # n should be a non negative integer equal to counts.sum.
      dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
      dist.prob(counts).eval()  # Should not raise.

  def testSampleUnbiasedNonScalarBatch(self):
    with self.cached_session() as sess:
      dist = ds.DirichletMultinomial(
          total_count=5.,
          concentration=1. + 2. * self._rng.rand(4, 3, 2).astype(np.float32))
      n = int(3e3)
      x = dist.sample(n, seed=0)
      sample_mean = tf.reduce_mean(x, 0)
      # Cyclically rotate event dims left.
      x_centered = tf.transpose(x - sample_mean, [1, 2, 3, 0])
      sample_covariance = tf.matmul(
          x_centered, x_centered, adjoint_b=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.mean(),
          dist.covariance(),
      ])
      self.assertAllEqual([4, 3, 2], sample_mean.shape)
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.20)
      self.assertAllEqual([4, 3, 2, 2], sample_covariance.shape)
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testSampleUnbiasedScalarBatch(self):
    with self.cached_session() as sess:
      dist = ds.DirichletMultinomial(
          total_count=5.,
          concentration=1. + 2. * self._rng.rand(4).astype(np.float32))
      n = int(5e3)
      x = dist.sample(n, seed=0)
      sample_mean = tf.reduce_mean(x, 0)
      x_centered = x - sample_mean  # Already transposed to [n, 2].
      sample_covariance = tf.matmul(
          x_centered, x_centered, adjoint_a=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.mean(),
          dist.covariance(),
      ])
      self.assertAllEqual([4], sample_mean.shape)
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.20)
      self.assertAllEqual([4, 4], sample_covariance.shape)
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testNotReparameterized(self):
    total_count = tf.constant(5.0)
    concentration = tf.constant([0.1, 0.1, 0.1])
    with backprop.GradientTape() as tape:
      tape.watch(total_count)
      tape.watch(concentration)
      dist = ds.DirichletMultinomial(
          total_count=total_count,
          concentration=concentration)
      samples = dist.sample(100)
    grad_total_count, grad_concentration = tape.gradient(
        samples, [total_count, concentration])
    self.assertIsNone(grad_total_count)
    self.assertIsNone(grad_concentration)


if __name__ == "__main__":
  tf.test.main()
