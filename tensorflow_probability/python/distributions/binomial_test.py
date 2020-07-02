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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import binomial as binomial_lib
from tensorflow_probability.python.internal import implementation_selection
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class BinomialTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)
    super(BinomialTest, self).setUp()

  def testAssertionsOneOfProbsAndLogits(self):
    with self.assertRaisesRegex(
        ValueError, 'Construct `Binomial` with `probs` or `logits`'):
      self.evaluate(tfd.Binomial(total_count=10))

    with self.assertRaisesRegexp(ValueError, 'but not both'):
      self.evaluate(tfd.Binomial(total_count=10, probs=0.5, logits=0.))

  def testInvalidProbabilities(self):
    invalid_probabilities = [1.01, 2.]
    for p in invalid_probabilities:
      with self.assertRaisesOpError('probs has components greater than 1'):
        dist = tfd.Binomial(total_count=10, probs=p, validate_args=True)
        self.evaluate(dist.probs_parameter())

    invalid_probabilities = [-0.01, -3.]
    for p in invalid_probabilities:
      with self.assertRaisesOpError('x >= 0 did not hold'):
        dist = tfd.Binomial(total_count=10, probs=p, validate_args=True)
        self.evaluate(dist.probs_parameter())

  def testSimpleShapes(self):
    p = np.float32(np.random.beta(1, 1))
    binom = tfd.Binomial(total_count=1., probs=p, validate_args=True)
    self.assertAllEqual([], self.evaluate(binom.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(binom.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), binom.event_shape)
    self.assertEqual(tf.TensorShape([]), binom.batch_shape)

  def testComplexShapes(self):
    p = np.random.beta(1, 1, size=(3, 2)).astype(np.float32)
    n = [[3., 2], [4, 5], [6, 7]]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    self.assertAllEqual([], self.evaluate(binom.event_shape_tensor()))
    self.assertAllEqual([3, 2], self.evaluate(binom.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), binom.event_shape)
    self.assertEqual(tf.TensorShape([3, 2]), binom.batch_shape)

  def testNProperty(self):
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    self.assertEqual((2, 1), binom.total_count.shape)
    self.assertAllClose(n, self.evaluate(binom.total_count))

  def testPProperty(self):
    p = [[0.1, 0.2, 0.7]]
    binom = tfd.Binomial(total_count=3., probs=p, validate_args=True)
    self.assertEqual((1, 3), binom.probs.shape)
    self.assertEqual((1, 3), binom.logits_parameter().shape)
    self.assertAllClose(p, self.evaluate(binom.probs))

  def testLogitsProperty(self):
    logits = [[0., 9., -0.5]]
    binom = tfd.Binomial(total_count=3., logits=logits, validate_args=True)
    self.assertEqual((1, 3), binom.probs_parameter().shape)
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
    with self.assertRaisesOpError('Condition x >= 0.*'):
      self.evaluate(binom.prob([-1., 4, 2]))
    with self.assertRaisesOpError('Condition x <= y.*'):
      self.evaluate(binom.prob([7., 3, 0]))
    with self.assertRaisesOpError('Condition x >= 0.*'):
      self.evaluate(binom.cdf([-1., 4, 2]))
    with self.assertRaisesOpError('Condition x <= y.*'):
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
    placeholder = tf1.placeholder_with_default([1.0, 2.5, 1.5], shape=[3])
    # Both equality and integer checking fail.
    with self.assertRaisesOpError('cannot contain fractional components.'):
      self.evaluate(binom.prob(placeholder))
    with self.assertRaisesOpError('cannot contain fractional components.'):
      self.evaluate(binom.cdf(placeholder))

    binom = tfd.Binomial(total_count=n, probs=p, validate_args=False)
    self.evaluate(binom.prob([1., 2., 3.]))
    self.evaluate(binom.cdf([1., 2., 3.]))
    # Non-integer arguments work.
    self.evaluate(binom.prob([1.0, 2.5, 1.5]))
    self.evaluate(binom.cdf([1.0, 2.5, 1.5]))

  def testPmfAndCdfExtremeProbs(self):
    n = 5.
    p = [0., 1.]
    counts = [[0.], [3.], [5.]]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    self.assertAllClose(self.evaluate(binom.prob(counts)),
                        [[1., 0.], [0., 0.], [0., 1.]])
    self.assertAllClose(self.evaluate(binom.cdf(counts)),
                        [[1., 0.], [1., 0.], [1., 1.]])

  def testPmfAndCdfBothZeroBatches(self):
    # Both zero-batches.  No broadcast
    p = 0.5
    counts = 1.
    binom = tfd.Binomial(total_count=1., probs=p, validate_args=True)
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
    binom = tfd.Binomial(total_count=5., probs=p, validate_args=True)
    pmf = binom.prob(counts)
    cdf = binom.cdf(counts)
    self.assertAllClose(stats.binom.pmf(counts, n=5., p=p), self.evaluate(pmf))
    self.assertAllClose(stats.binom.cdf(counts, n=5., p=p), self.evaluate(cdf))
    self.assertEqual((), pmf.shape)
    self.assertEqual((), cdf.shape)

  def testPmfAndCdfPStretchedInBroadcastWhenSameRank(self):
    p = [[0.1, 0.9]]
    counts = [[1., 2.]]
    binom = tfd.Binomial(total_count=3., probs=p, validate_args=True)
    pmf = binom.prob(counts)
    cdf = binom.cdf(counts)
    self.assertAllClose(stats.binom.pmf(counts, n=3., p=p), self.evaluate(pmf))
    self.assertAllClose(stats.binom.cdf(counts, n=3., p=p), self.evaluate(cdf))
    self.assertEqual((1, 2), pmf.shape)
    self.assertEqual((1, 2), cdf.shape)

  def testPmfAndCdfPStretchedInBroadcastWhenLowerRank(self):
    p = [0.1, 0.4]
    counts = [[1.], [0.]]
    binom = tfd.Binomial(total_count=1., probs=p, validate_args=True)
    pmf = binom.prob(counts)
    cdf = binom.cdf(counts)
    self.assertAllClose([[0.1, 0.4], [0.9, 0.6]], self.evaluate(pmf))
    self.assertAllClose([[1.0, 1.0], [0.9, 0.6]], self.evaluate(cdf))
    self.assertEqual((2, 2), pmf.shape)
    self.assertEqual((2, 2), cdf.shape)

  def testBinomialMean(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    expected_means = stats.binom.mean(n, p)
    self.assertEqual((3,), binom.mean().shape)
    self.assertAllClose(expected_means, self.evaluate(binom.mean()))

  def testBinomialVariance(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    expected_variances = stats.binom.var(n, p)
    self.assertEqual((3,), binom.variance().shape)
    self.assertAllClose(expected_variances, self.evaluate(binom.variance()))

  def testBinomialMode(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    expected_modes = [0., 1, 4]
    self.assertEqual((3,), binom.mode().shape)
    self.assertAllClose(expected_modes, self.evaluate(binom.mode()))

  def testBinomialModeExtremeValues(self):
    n = [51., 52.]

    p = 1.
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    expected_modes = n
    self.assertEqual((2,), binom.mode().shape)
    self.assertAllClose(expected_modes, self.evaluate(binom.mode()))

    p = 0.
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    expected_modes = [0., 0.]
    self.assertEqual((2,), binom.mode().shape)
    self.assertAllClose(expected_modes, self.evaluate(binom.mode()))

  def testBinomialMultipleMode(self):
    n = 9.
    p = [0.1, 0.2, 0.7]
    binom = tfd.Binomial(total_count=n, probs=p, validate_args=True)
    # For the case where (n + 1) * p is an integer, the modes are:
    # (n + 1) * p and (n + 1) * p - 1. In this case, we get back
    # the larger of the two modes.
    expected_modes = [1., 2, 7]
    self.assertEqual((3,), binom.mode().shape)
    self.assertAllClose(expected_modes, self.evaluate(binom.mode()))

  def testUnivariateLogConcaveDistributionRejectionSamplerGeometric(self):
    seed = test_util.test_seed()
    n = int(5e5)

    probs = np.float32([0.7, 0.8, 0.3, 0.2])
    geometric = tfd.Geometric(probs=probs)
    x = binomial_lib._log_concave_rejection_sampler(
        mode=geometric.mode(), prob_fn=geometric.prob, dtype=geometric.dtype,
        sample_shape=[n], distribution_minimum=0, seed=seed)

    x = x + 1  ## scipy.stats.geom is 1-indexed instead of 0-indexed.
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
        stats.geom.mean(probs), sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4], sample_variance.shape)
    self.assertAllClose(
        stats.geom.var(probs), sample_variance_, atol=0., rtol=0.20)

  def testSampleUnbiasedNonScalarBatch(self):
    probs = self._rng.rand(4, 3).astype(np.float32)
    counts = np.float32([4, 11., 20.])
    dist = tfd.Binomial(total_count=counts, probs=probs, validate_args=True)
    n = int(1e5)
    x = dist.sample(n, seed=test_util.test_seed())
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
    dist = tfd.Binomial(total_count=counts, probs=probs, validate_args=True)
    n = int(1e5)
    x = dist.sample(n, seed=test_util.test_seed())
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

  def testSampleExtremeValues(self):
    total_count = tf.constant(17., dtype=tf.float32)
    probs = tf.constant([0., 1.], dtype=tf.float32)
    dist = tfd.Binomial(
        total_count=total_count, probs=probs, validate_args=True)
    samples, expected_samples = self.evaluate(
        (dist.sample(seed=test_util.test_seed()),
         total_count * probs))
    self.assertAllEqual(samples, expected_samples)

  def testSampleWithLargeCountAndSmallProbs(self):
    total_count = tf.constant(1e6, dtype=tf.float32)
    probs = tf.constant(1e-6, dtype=tf.float32)
    dist = tfd.Binomial(total_count=total_count,
                        probs=probs, validate_args=True)
    small_probs = self.evaluate(dist.prob([0., 1., 2.]))
    # Approximate analytic probabilities for very small counts for this
    # binomial:
    # - p(0) = (1 - 1e-6)**1e6 \approx 1/e
    # - p(1) = p(0) * 1e6 * 1e-6 * 1/(1 - 1e-6)
    #        \approx 1/e * 1/(1 - 1e-6) \approx 1/e
    # - p(2) = p(0) * 1e6**2 * choose(2, 1e6) / (1 - 1e-6)**2
    #        \approx 1/e * 1e6 * 1/(1e6 - 1) * 1/2 * 1/(1 - 1e-6)**2
    #        \approx 1/(2*e)
    expected_probs = [np.exp(-1.), np.exp(-1.), np.exp(-1.) / 2.]
    self.assertAllClose(small_probs, expected_probs)
    small_log_probs = self.evaluate(dist.log_prob([0., 1., 2.]))
    expected_log_probs = [-1., -1., -1. - np.log(2)]
    self.assertAllClose(small_log_probs, expected_log_probs, rtol=5e-6)

    n = 1000
    sample_avg = self.evaluate(
        tf.reduce_mean(dist.sample(n, seed=test_util.test_seed())))
    # `sample_avg` has mean 1 and variance ~1/n.  So `sample_avg` should be
    # within `4.9 / sqrt(n)` of 1 with probability `1 - 1e-6`.
    self.assertAllClose(sample_avg, 1., atol=4.9/np.sqrt(n))

  def testSampleWithZeroCountsAndNanSuccessProbability(self):
    # With zero counts, should sample 0 successes even if success probability is
    # nan; and should not interfere with the rest of the batch.
    total_count = tf.constant([23., 0.], dtype=tf.float32)
    probs = tf.constant([1., np.nan], dtype=tf.float32)
    dist = tfd.Binomial(
        total_count=total_count, probs=probs, validate_args=False)
    samples, expected_samples = self.evaluate(
        (dist.sample(seed=test_util.test_seed()),
         total_count))
    self.assertAllEqual(samples, expected_samples)

  def testSampleInBounds(self):
    # Regression test for b/152660887.
    # This number of samples suffices to make the test fail about 45 times out
    # of 100 under --vary_seed (before applying the fix, that is).
    num_samples = 1000000
    total_count = tf.constant([1.], dtype=tf.float32)
    probs = tf.constant([0.], dtype=tf.float32)
    dist = tfd.Binomial(
        total_count=total_count, probs=probs, validate_args=True)
    samples = self.evaluate(
        dist.sample(num_samples, seed=test_util.test_seed()))
    self.assertAllEqual(samples, tf.zeros_like(samples))

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
        # Set atol because logit(0.5) == 0.
        atol=1e-6, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([d.prob(1.), d.probs_parameter()]),
        atol=0, rtol=1e-4)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      'No gradient for while loops in JAX backend.')
  def testNotReparameterized(self):
    def f(n, p):
      b = tfd.Binomial(n, p)
      return b.sample(5, seed=test_util.test_seed())

    _, [grad_n, grad_p] = tfp.math.value_and_gradient(f, [10., 0.5])
    self.assertIsNone(grad_n)
    self.assertIsNone(grad_p)


@test_util.test_all_tf_execution_regimes
class BinomialFromVariableTest(test_util.TestCase):

  def testAssertionsTotalCount(self):
    x = tf.Variable([-1.0, 4.0, 1.0])
    d = tfd.Binomial(total_count=x, logits=[0.0, 0.0, 0.0], validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('`total_count` must be non-negative.'):
      self.evaluate(d.mean())

  def testAssertionsTotalCountMutation(self):
    x = tf.Variable([1.0, 4.0, 1.0])
    d = tfd.Binomial(total_count=x, logits=[0.0, 0.0, 0.0], validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.evaluate(d.mean())
    self.evaluate(x.assign(-x))
    with self.assertRaisesOpError('`total_count` must be non-negative.'):
      self.evaluate(d.mean())

  def testAssertionsProbsMutation(self):
    probs = tf.Variable([0.5])

    d = tfd.Binomial(total_count=10, probs=probs, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.evaluate(d.probs_parameter())

    self.evaluate(probs.assign([1.25]))
    with self.assertRaisesOpError('probs has components greater than 1'):
      self.evaluate(d.probs_parameter())

    self.evaluate(probs.assign([-0.5]))
    with self.assertRaisesOpError('x >= 0 did not hold'):
      self.evaluate(d.probs_parameter())

    self.evaluate(probs.assign([0.75]))
    self.evaluate(d.probs_parameter())


@test_util.test_graph_and_eager_modes
class BinomialSamplingTest(test_util.TestCase):

  @test_util.jax_disable_test_missing_functionality('tf stateless_binomial')
  def testSampleCPU(self):
    with tf.device('CPU'):
      _, runtime = self.evaluate(
          binomial_lib._random_binomial(
              shape=tf.constant([], dtype=tf.int32),
              counts=tf.constant(10.), probs=tf.constant(.5),
              seed=test_util.test_seed()))
    self.assertEqual(implementation_selection._RUNTIME_CPU, runtime)

  def testSampleGPU(self):
    # Debugging tip: --vmodule=implementation_selector=2,function_api_info=3
    # helps debug errors like: Skipping optimization due to error while loading
    # function libraries: Invalid argument: Functions
    # '__inference__binomial_cpu_1184' and '__inference__binomial_noncpu_2305'
    # both implement 'binomial_76a77701-978a-4299-ad23-ff0d5c7598cb' but their
    # signatures do not match.
    if not tf.test.is_gpu_available():
      self.skipTest('no GPU')
    with tf.device('GPU'):
      _, runtime = self.evaluate(binomial_lib._random_binomial(
          shape=tf.constant([], dtype=tf.int32),
          counts=tf.constant(10.), probs=tf.constant(.5),
          seed=test_util.test_seed()))
    self.assertEqual(implementation_selection._RUNTIME_DEFAULT, runtime)

  def testSampleXLA(self):
    self.skip_if_no_xla()
    if not tf.executing_eagerly(): return  # experimental_compile is eager-only.
    probs = np.random.rand(4, 3).astype(np.float32)
    counts = np.float32([4, 11., 20.])
    dist = tfd.Binomial(total_count=counts, probs=probs, validate_args=True)
    # Verify the compile succeeds going all the way through the distribution.
    self.evaluate(
        tf.function(lambda: dist.sample(5, seed=test_util.test_seed()),
                    experimental_compile=True)())
    # Also test the low-level sampler and verify the XLA-friendly variant.
    _, runtime = self.evaluate(
        tf.function(binomial_lib._random_binomial, experimental_compile=True)(
            shape=tf.constant([], dtype=tf.int32),
            counts=tf.constant(10.), probs=tf.constant(.5),
            seed=test_util.test_seed()))
    self.assertEqual(implementation_selection._RUNTIME_DEFAULT, runtime)


if __name__ == '__main__':
  tf.test.main()
