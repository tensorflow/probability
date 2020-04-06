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

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class DirichletMultinomialTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testSimpleShapes(self):
    alpha = np.random.rand(3)
    dist = tfd.DirichletMultinomial(1., alpha, validate_args=True)
    self.assertEqual(3, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([3]), dist.event_shape)
    self.assertEqual(tf.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    alpha = np.random.rand(3, 2, 2)
    n = np.array([[3., 2], [4, 5], [6, 7]], dtype=np.float64)
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=True)
    self.assertEqual(2, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([2]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2]), dist.batch_shape)
    self.assertEqual(
        tf.TensorShape([3, 2, 2]), dist.sample(
            seed=test_util.test_seed()).shape)

  def testNproperty(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=True)
    self.assertEqual([1, 1], dist.total_count.shape)
    self.assertAllClose(n, self.evaluate(dist.total_count))

  def testAlphaProperty(self):
    alpha = [[1., 2, 3]]
    dist = tfd.DirichletMultinomial(1, alpha, validate_args=True)
    self.assertEqual([1, 3], dist.concentration.shape)
    self.assertAllClose(alpha, self.evaluate(dist.concentration))

  def testPmfNandCountsAgree(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=True)
    self.evaluate(dist.prob([2., 3, 0]))
    self.evaluate(dist.prob([3., 0, 2]))
    with self.assertRaisesOpError('must be non-negative'):
      self.evaluate(dist.prob([-1., 4, 2]))
    with self.assertRaisesOpError(
        'last-dimension must sum to `self.total_count`'):
      self.evaluate(dist.prob([3., 3, 0]))

  def testPmfNonIntegerCounts(self):
    alpha = [[1., 2, 3]]
    n = [[5.]]
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=True)
    self.evaluate(dist.prob([2., 3, 0]))
    self.evaluate(dist.prob([3., 0, 2]))
    self.evaluate(dist.prob([3.0, 0, 2.0]))
    # Both equality and integer checking fail.
    placeholder = tf1.placeholder_with_default([1.0, 2.5, 1.5], shape=None)
    with self.assertRaisesOpError(
        'cannot contain fractional components'):
      self.evaluate(dist.prob(placeholder))
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=False)
    self.evaluate(dist.prob([1., 2., 3.]))
    # Non-integer arguments work.
    self.evaluate(dist.prob([1.0, 2.5, 1.5]))

  def testPmfBothZeroBatches(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    # Both zero-batches.  No broadcast
    alpha = [1., 2]
    counts = [1., 0]
    dist = tfd.DirichletMultinomial(1., alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose(1 / 3., self.evaluate(pmf))
    self.assertEqual((), pmf.shape)

  def testPmfBothZeroBatchesNontrivialN(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    # Both zero-batches.  No broadcast
    alpha = [1., 2]
    counts = [3., 2]
    dist = tfd.DirichletMultinomial(5., alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose(1 / 7., self.evaluate(pmf))
    self.assertEqual((), pmf.shape)

  def testPmfBothZeroBatchesMultidimensionalN(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [1., 2]
    counts = [3., 2]
    n = np.full([4, 3], 5., dtype=np.float32)
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose([[1 / 7., 1 / 7., 1 / 7.]] * 4, self.evaluate(pmf))
    self.assertEqual((4, 3), pmf.shape)

  def testPmfAlphaStretchedInBroadcastWhenSameRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [[1., 2]]
    counts = [[1., 0], [0., 1]]
    dist = tfd.DirichletMultinomial([1.], alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose([1 / 3., 2 / 3.], self.evaluate(pmf))
    self.assertAllEqual([2], pmf.shape)

  def testPmfAlphaStretchedInBroadcastWhenLowerRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [1., 2]
    counts = [[1., 0], [0., 1]]
    pmf = tfd.DirichletMultinomial(1., alpha, validate_args=True).prob(counts)
    self.assertAllClose([1 / 3., 2 / 3.], self.evaluate(pmf))
    self.assertAllEqual([2], pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [[1., 2], [2., 3]]
    counts = [[1., 0]]
    pmf = tfd.DirichletMultinomial([1., 1.], alpha,
                                   validate_args=True).prob(counts)
    self.assertAllClose([1 / 3., 2 / 5.], self.evaluate(pmf))
    self.assertAllEqual([2], pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [[1., 2], [2., 3]]
    counts = [1., 0]
    pmf = tfd.DirichletMultinomial(1., alpha, validate_args=True).prob(counts)
    self.assertAllClose([1 / 3., 2 / 5.], self.evaluate(pmf))
    self.assertAllEqual([2], pmf.shape)

  def testPmfForOneVoteIsTheMeanWithOneRecordInput(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [1., 2, 3]
    dist = tfd.DirichletMultinomial(1., alpha, validate_args=True)
    mean = self.evaluate(dist.mean())
    for class_num in range(3):
      counts = np.zeros([3], dtype=np.float32)
      counts[class_num] = 1
      pmf = self.evaluate(dist.prob(counts))

      self.assertAllClose(mean[class_num], pmf)
      self.assertAllEqual([3], mean.shape)
      self.assertAllEqual([], pmf.shape)

  def testMeanDoubleTwoVotes(self):
    # The probabilities of two votes falling into class k for
    # DirichletMultinomial(2, alpha) is twice as much as the probability of one
    # vote falling into class k for DirichletMultinomial(1, alpha)
    alpha = [1., 2, 3]
    dist1 = tfd.DirichletMultinomial(1., alpha, validate_args=True)
    dist2 = tfd.DirichletMultinomial(2., alpha, validate_args=True)

    mean1 = self.evaluate(dist1.mean())
    mean2 = self.evaluate(dist2.mean())
    self.assertAllEqual([3], mean1.shape)

    for class_num in range(3):
      self.assertAllClose(mean2[class_num], 2 * mean1[class_num])

  def testCovarianceFromSampling(self):
    alpha = np.array([[1., 2, 3],
                      [2.5, 4, 0.1]], dtype=np.float32)
    n = np.array([10, 30], dtype=np.float32)
    # batch_shape=[2], event_shape=[3]
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=True)
    # Sample count chosen based on what can be drawn in about 1 minute.
    x = dist.sample(int(25e3), seed=test_util.test_seed())
    sample_mean = tf.reduce_mean(x, axis=0)
    x_centered = x - sample_mean[tf.newaxis, ...]
    sample_cov = tf.reduce_mean(
        tf.matmul(x_centered[..., tf.newaxis], x_centered[..., tf.newaxis, :]),
        axis=0)
    sample_var = tf.linalg.diag_part(sample_cov)
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
    ] = self.evaluate([
        sample_mean,
        sample_cov,
        sample_var,
        sample_stddev,
        dist.mean(),
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
    ])
    # Tolerances tuned as follows:
    # - Run with 1000 independent seeds until < 5% fail
    # - Then double the relative tolerance
    # If the sampled quantities are normally distributed, a 5% failure rate
    # corresponds to a z-score of about 2; doubling this should give a z-score
    # of 4, which corresponds to a failure rate of 6.4e-5
    self.assertAllClose(sample_mean_, analytic_mean, rtol=0.1)
    self.assertAllClose(sample_cov_, analytic_cov, rtol=0.3)
    self.assertAllClose(sample_var_, analytic_var, rtol=0.2)
    self.assertAllClose(sample_stddev_, analytic_stddev, rtol=0.1)

  def testCovariance(self):
    # Shape [2]
    alpha = [1., 2]
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

    # ns is shape [4, 1] and alpha is shape [2].
    ns = [[2.], [3.], [4.], [5.]]
    dist = tfd.DirichletMultinomial(ns, alpha, validate_args=True)
    covariance = self.evaluate(dist.covariance())
    for i in range(len(ns)):
      n = ns[i][0]
      expected_covariance = n * (n + alpha_0) / (1 + alpha_0) * shared_matrix
      self.assertAllClose(expected_covariance, np.squeeze(covariance[i]))

  def testCovarianceNAlphaBroadcast(self):
    alpha_v = [1., 2, 3]
    alpha_0 = 6.

    # Shape [4, 3]
    alpha = np.array(4 * [alpha_v], dtype=np.float32)
    # Shape [4]
    ns = np.array([2., 3., 4., 5.], dtype=np.float32)

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

    # ns is shape [4], and alpha is shape [4, 3].
    dist = tfd.DirichletMultinomial(ns, alpha, validate_args=True)
    covariance = dist.covariance()
    expected_covariance = shared_matrix * (
        ns * (ns + alpha_0) / (1 + alpha_0))[..., tf.newaxis, tf.newaxis]

    self.assertEqual([4, 3, 3], covariance.shape)
    self.assertAllClose(expected_covariance, self.evaluate(covariance))

  def testCovarianceMultidimensional(self):
    alpha = np.random.rand(3, 1, 4).astype(np.float32)
    alpha2 = np.random.rand(6, 3, 3).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1]).astype(np.float32)

    dist = tfd.DirichletMultinomial(ns, alpha, validate_args=True)
    dist2 = tfd.DirichletMultinomial(ns2, alpha2, validate_args=True)

    covariance = dist.covariance()
    covariance2 = dist2.covariance()
    self.assertEqual([3, 5, 4, 4], covariance.shape)
    self.assertEqual([6, 3, 3, 3], covariance2.shape)

  def testZeroCountsResultsInPmfEqualToOne(self):
    # There is only one way for zero items to be selected, and this happens with
    # probability 1.
    alpha = [5, 0.5]
    counts = [0., 0]
    dist = tfd.DirichletMultinomial(0., alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose(1.0, self.evaluate(pmf))
    self.assertEqual((), pmf.shape)

  def testLargeTauGivesPreciseProbabilities(self):
    # If tau is large, we are doing coin flips with probability mu.
    mu = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    tau = np.array([100.], dtype=np.float32)
    alpha = tau * mu

    # One (three sided) coin flip.  Prob[coin 3] = 0.8.
    # Note that since it was one flip, value of tau didn't matter.
    counts = [0., 0, 1]
    dist = tfd.DirichletMultinomial(1., alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose(0.8, self.evaluate(pmf), atol=1e-4)
    self.assertEqual((), pmf.shape)

    # Two (three sided) coin flips.  Prob[coin 3] = 0.8.
    counts = [0., 0, 2]
    dist = tfd.DirichletMultinomial(2., alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose(0.8**2, self.evaluate(pmf), atol=1e-2)
    self.assertEqual((), pmf.shape)

    # Three (three sided) coin flips.
    counts = [1., 0, 2]
    dist = tfd.DirichletMultinomial(3., alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose(3 * 0.1 * 0.8 * 0.8, self.evaluate(pmf), atol=1e-2)
    self.assertEqual((), pmf.shape)

  def testSmallTauPrefersCorrelatedResults(self):
    # If tau is small, then correlation between draws is large, so draws that
    # are both of the same class are more likely.
    mu = np.array([0.5, 0.5], dtype=np.float32)
    tau = np.array([0.1], dtype=np.float32)
    alpha = tau * mu

    # If there is only one draw, it is still a coin flip, even with small tau.
    counts = [1., 0]
    dist = tfd.DirichletMultinomial(1., alpha, validate_args=True)
    pmf = dist.prob(counts)
    self.assertAllClose(0.5, self.evaluate(pmf))
    self.assertEqual((), pmf.shape)

    # If there are two draws, it is much more likely that they are the same.
    counts_same = [2., 0]
    counts_different = [1, 1.]
    dist = tfd.DirichletMultinomial(2., alpha, validate_args=True)
    pmf_same = dist.prob(counts_same)
    pmf_different = dist.prob(counts_different)
    self.assertLess(
        5 * self.evaluate(pmf_different),
        self.evaluate(pmf_same))
    self.assertEqual((), pmf_same.shape)

  def testNonStrictTurnsOffAllChecks(self):
    # Make totally invalid input.
    alpha = [[-1., 2]]  # alpha should be positive.
    counts = [[1., 0], [0., -1]]  # counts should be non-negative.
    n = [-5.3]  # n should be a non negative integer equal to counts.sum.
    dist = tfd.DirichletMultinomial(n, alpha, validate_args=False)
    self.evaluate(dist.prob(counts))  # Should not raise.

  def testSampleUnbiasedNonScalarBatch(self):
    seed_stream = test_util.test_seed_stream()
    concentration = 1. + 2. * tf.random.uniform(
        shape=[4, 3, 2], dtype=np.float32, seed=seed_stream())
    dist = tfd.DirichletMultinomial(
        total_count=5.,
        concentration=concentration,
        validate_args=True)
    n = int(5e3)
    x = dist.sample(n, seed=seed_stream())
    sample_mean = tf.reduce_mean(x, axis=0)
    # Cyclically rotate event dims left.
    x_centered = tf.transpose(a=x - sample_mean, perm=[1, 2, 3, 0])
    sample_covariance = tf.matmul(
        x_centered, x_centered, adjoint_b=True) / n
    [
        sample_mean_,
        sample_covariance_,
        actual_mean_,
        actual_covariance_,
    ] = self.evaluate([
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
    seed_stream = test_util.test_seed_stream()
    concentration = 1. + 2. * tf.random.uniform(
        shape=[4], dtype=np.float32, seed=seed_stream())
    dist = tfd.DirichletMultinomial(
        total_count=5.,
        concentration=concentration,
        validate_args=True)
    n = int(1e4)
    x = dist.sample(n, seed=seed_stream())
    sample_mean = tf.reduce_mean(x, axis=0)
    x_centered = x - sample_mean  # Already transposed to [n, 2].
    sample_covariance = tf.linalg.matmul(
        x_centered, x_centered, adjoint_a=True) / n
    [
        sample_mean_,
        sample_covariance_,
        actual_mean_,
        actual_covariance_,
    ] = self.evaluate([
        sample_mean,
        sample_covariance,
        dist.mean(),
        dist.covariance(),
    ])
    self.assertAllEqual([4], sample_mean.shape)
    self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.25)
    self.assertAllEqual([4, 4], sample_covariance.shape)
    self.assertAllClose(
        actual_covariance_, sample_covariance_, atol=0., rtol=0.25)

  def testNotReparameterized(self):
    if tf1.control_flow_v2_enabled():
      self.skipTest('b/138796859')
    total_count = tf.constant(5.0)
    concentration = tf.constant([0.1, 0.1, 0.1])
    _, [grad_total_count, grad_concentration] = tfp.math.value_and_gradient(
        lambda n, c: tfd.DirichletMultinomial(n, c, validate_args=True).sample(  # pylint: disable=g-long-lambda
            100, seed=test_util.test_seed()), [total_count, concentration])
    self.assertIsNone(grad_total_count)
    self.assertIsNone(grad_concentration)

  def testSamplesHaveCorrectTotalCounts(self):
    seed_stream = test_util.test_seed_stream()
    concentration = 1. + 2. * tf.random.uniform(
        shape=[4], dtype=np.float32, seed=seed_stream())
    total_count = tf.constant(list(range(int(1e4))), dtype=np.float32)
    dist = tfd.DirichletMultinomial(
        total_count=total_count,
        concentration=concentration,
        validate_args=True)
    x = dist.sample(seed=seed_stream())
    self.assertAllEqual(tf.reduce_sum(x, axis=-1), total_count)


@test_util.test_all_tf_execution_regimes
class DirichletMultinomialFromVariableTest(test_util.TestCase):

  def testAssertionCategoricalEventShape(self):
    total_count = tf.constant(10.0, dtype=tf.float16)
    too_many_classes = 2**11 + 1
    concentration = tf.Variable(tf.ones(too_many_classes, tf.float16))
    with self.assertRaisesRegexp(
        ValueError, 'Number of classes exceeds `dtype` precision'):
      tfd.DirichletMultinomial(
          total_count, concentration, validate_args=True)

  def testAssertionNonNegativeTotalCount(self):
    total_count = tf.Variable(-1.0)
    concentration = tf.constant([1., 1., 1.])

    with self.assertRaisesOpError('must be non-negative'):
      d = tfd.DirichletMultinomial(total_count, concentration,
                                   validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.mean())

  def testAssertionNonNegativeTotalCountAfterMutation(self):
    total_count = tf.Variable(0.0)
    concentration = tf.constant([1., 1., 1.])
    d = tfd.DirichletMultinomial(total_count, concentration,
                                 validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.evaluate(d.mean())
    self.evaluate(total_count.assign(-1.0))

    with self.assertRaisesOpError('must be non-negative'):
      self.evaluate(d.mean())

  def testAssertionIntegerFormTotalCount(self):
    total_count = tf.Variable(0.5)
    concentration = tf.constant([1., 1., 1.])

    with self.assertRaisesOpError('cannot contain fractional components'):
      d = tfd.DirichletMultinomial(total_count, concentration,
                                   validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.mean())

  def testAssertionIntegerFormTotalCountAfterMutation(self):
    total_count = tf.Variable(0.0)
    concentration = tf.constant([1., 1., 1.])
    d = tfd.DirichletMultinomial(total_count, concentration,
                                 validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.evaluate(d.mean())
    self.evaluate(total_count.assign(0.5))

    with self.assertRaisesOpError('cannot contain fractional components'):
      self.evaluate(d.mean())

  def testAssertionPositiveConcentration(self):
    total_count = tf.constant(10.0)
    concentration = tf.Variable([1., 1., -1.])

    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      d = tfd.DirichletMultinomial(total_count, concentration,
                                   validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.mean())

  def testAssertionPositiveConcentrationAfterMutation(self):
    total_count = tf.constant(10.0)
    concentration = tf.Variable([1., 1., 1.])

    d = tfd.DirichletMultinomial(total_count, concentration,
                                 validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.evaluate(d.mean())
    self.evaluate(concentration.assign([1., 1., -1.]))

    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      self.evaluate(d.mean())


if __name__ == '__main__':
  tf.test.main()
