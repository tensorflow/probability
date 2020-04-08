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

import unittest

# Dependency imports
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import hypothesis_testlib as dhps
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class MultinomialTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)
    super(MultinomialTest, self).setUp()

  def testSimpleShapes(self):
    p = [.1, .3, .6]
    dist = tfd.Multinomial(total_count=1., probs=p, validate_args=True)
    self.assertEqual(3, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([3]), dist.event_shape)
    self.assertEqual(tf.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    p = 0.5 * np.ones([3, 2, 2], dtype=np.float32)
    n = [[3., 2], [4, 5], [6, 7]]
    dist = tfd.Multinomial(total_count=n, probs=p, validate_args=True)
    self.assertEqual(2, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([2]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2]), dist.batch_shape)
    self.assertEqual(tf.TensorShape([17, 3, 2, 2]), dist.sample(
        17, seed=test_util.test_seed()).shape)

  def testN(self):
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    dist = tfd.Multinomial(total_count=n, probs=p, validate_args=True)
    self.assertAllEqual([2, 1], dist.total_count.shape)
    self.assertAllClose(n, self.evaluate(dist.total_count))

  def testP(self):
    p = [[0.1, 0.2, 0.7]]
    dist = tfd.Multinomial(total_count=3., probs=p, validate_args=True)
    self.assertAllEqual([1, 3], dist.probs_parameter().shape)
    self.assertAllEqual([1, 3], dist.logits_parameter().shape)
    self.assertAllClose(p, self.evaluate(dist.probs))

  def testLogits(self):
    p = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    logits = np.log(p) - 50.
    multinom = tfd.Multinomial(
        total_count=3., logits=logits, validate_args=True)
    self.assertAllEqual([1, 3], multinom.probs_parameter().shape)
    self.assertAllEqual([1, 3], multinom.logits_parameter().shape)
    self.assertAllClose(p, self.evaluate(multinom.probs_parameter()))
    self.assertAllClose(logits, self.evaluate(multinom.logits_parameter()))

  def testPmfUnderflow(self):
    logits = np.array([[-200, 0]], dtype=np.float32)
    dist = tfd.Multinomial(total_count=1., logits=logits, validate_args=True)
    lp = self.evaluate(dist.log_prob([1., 0.]))[0]
    self.assertAllClose(-200, lp, atol=0, rtol=1e-6)

  def testPmfandCountsAgree(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    dist = tfd.Multinomial(total_count=n, probs=p, validate_args=True)
    self.evaluate(dist.prob([2., 3, 0]))
    self.evaluate(dist.prob([3., 0, 2]))
    with self.assertRaisesOpError('must be non-negative'):
      self.evaluate(dist.prob([-1., 4, 2]))
    with self.assertRaisesOpError('counts must sum to `self.total_count`'):
      self.evaluate(dist.prob([3., 3, 0]))

  def testPmfNonIntegerCounts(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    # No errors with integer n.
    multinom = tfd.Multinomial(
        total_count=n, probs=p, validate_args=True)
    self.evaluate(multinom.prob([2., 1, 2]))
    self.evaluate(multinom.prob([3., 0, 2]))
    # Counts don't sum to n.
    with self.assertRaisesOpError('counts must sum to `self.total_count`'):
      self.evaluate(multinom.prob([2., 3, 2]))
    # Counts are non-integers.
    x = tf1.placeholder_with_default([1., 2.5, 1.5], shape=None)
    with self.assertRaisesOpError(
        'cannot contain fractional components.'):
      self.evaluate(multinom.prob(x))

    multinom = tfd.Multinomial(
        total_count=n, probs=p, validate_args=False)
    self.evaluate(multinom.prob([1., 2., 2.]))
    # Non-integer arguments work.
    self.evaluate(multinom.prob([1.0, 2.5, 1.5]))

  def testPmfBothZeroBatches(self):
    # Both zero-batches.  No broadcast
    p = [0.5, 0.5]
    counts = [1., 0]
    pmf = tfd.Multinomial(
        total_count=1., probs=p, validate_args=True).prob(counts)
    self.assertAllClose(0.5, self.evaluate(pmf))
    self.assertAllEqual([], pmf.shape)

  def testPmfBothZeroBatchesNontrivialN(self):
    # Both zero-batches.  No broadcast
    p = [0.1, 0.9]
    counts = [3., 2]
    dist = tfd.Multinomial(total_count=5., probs=p, validate_args=True)
    pmf = dist.prob(counts)
    # 5 choose 3 = 5 choose 2 = 10. 10 * (.9)^2 * (.1)^3 = 81/10000.
    self.assertAllClose(81. / 10000, self.evaluate(pmf))
    self.assertAllEqual([], pmf.shape)

  def testPmfPStretchedInBroadcastWhenSameRank(self):
    p = [[0.1, 0.9]]
    counts = [[1., 0], [0, 1]]
    pmf = tfd.Multinomial(
        total_count=1., probs=p, validate_args=True).prob(counts)
    self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
    self.assertAllEqual([2], pmf.shape)

  def testPmfPStretchedInBroadcastWhenLowerRank(self):
    p = [0.1, 0.9]
    counts = [[1., 0], [0, 1]]
    pmf = tfd.Multinomial(
        total_count=1., probs=p, validate_args=True).prob(counts)
    self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
    self.assertAllEqual([2], pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    p = [[0.1, 0.9], [0.7, 0.3]]
    counts = [[1., 0]]
    pmf = tfd.Multinomial(
        total_count=1., probs=p, validate_args=True).prob(counts)
    self.assertAllClose(self.evaluate(pmf), [0.1, 0.7])
    self.assertAllEqual([2], pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    p = [[0.1, 0.9], [0.7, 0.3]]
    counts = [1., 0]
    pmf = tfd.Multinomial(
        total_count=1., probs=p, validate_args=True).prob(counts)
    self.assertAllClose(self.evaluate(pmf), [0.1, 0.7])
    self.assertAllEqual([2], pmf.shape)

  def testPmfShapeCountsStretchedN(self):
    # [2, 2, 2]
    p = [[[0.1, 0.9], [0.1, 0.9]], [[0.7, 0.3], [0.7, 0.3]]]
    # [2, 2]
    n = [[3., 3], [3, 3]]
    # [2]
    counts = [2., 1]
    pmf = tfd.Multinomial(
        total_count=n, probs=p, validate_args=True).prob(counts)
    self.evaluate(pmf)
    self.assertAllEqual([2, 2], pmf.shape)

  def testPmfShapeCountsPStretchedN(self):
    p = [0.1, 0.9]
    counts = [3., 2]
    n = np.full([4, 3], 5., dtype=np.float32)
    pmf = tfd.Multinomial(
        total_count=n, probs=p, validate_args=True).prob(counts)
    self.evaluate(pmf)
    self.assertEqual((4, 3), pmf.shape)

  def testPmfZeros(self):
    dist = tfd.Multinomial(1000, probs=[0.7, 0.0, 0.3], validate_args=True)
    x = [489, 0, 511]
    dist2 = tfd.Binomial(1000, probs=0.7)
    self.assertAllClose(dist2.log_prob(x[0]), dist.log_prob(x), rtol=1e-5)

  def testMultinomialMean(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    dist = tfd.Multinomial(total_count=n, probs=p, validate_args=True)
    expected_means = 5 * np.array(p, dtype=np.float32)
    self.assertEqual((3,), dist.mean().shape)
    self.assertAllClose(expected_means, self.evaluate(dist.mean()))

  def testMultinomialCovariance(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    dist = tfd.Multinomial(total_count=n, probs=p, validate_args=True)
    expected_covariances = [[9. / 20, -1 / 10, -7 / 20],
                            [-1 / 10, 4 / 5, -7 / 10],
                            [-7 / 20, -7 / 10, 21 / 20]]
    self.assertEqual((3, 3), dist.covariance().shape)
    self.assertAllClose(
        expected_covariances, self.evaluate(dist.covariance()))

  def testMultinomialCovarianceBatch(self):
    # Shape [2]
    n = [5.] * 2
    # Shape [4, 1, 2]
    p = [[[0.1, 0.9]], [[0.1, 0.9]]] * 2
    dist = tfd.Multinomial(total_count=n, probs=p, validate_args=True)
    # Shape [2, 2]
    inner_var = [[9. / 20, -9 / 20], [-9 / 20, 9 / 20]]
    # Shape [4, 2, 2, 2]
    expected_covariances = [[inner_var, inner_var]] * 4
    self.assertEqual((4, 2, 2, 2), dist.covariance().shape)
    self.assertAllClose(
        expected_covariances, self.evaluate(dist.covariance()))

  def testCovarianceMultidimensional(self):
    # Shape [3, 5, 4]
    p = np.random.dirichlet([.25, .25, .25, .25], [3, 5]).astype(np.float32)
    # Shape [6, 3, 3]
    p2 = np.random.dirichlet([.3, .3, .4], [6, 3]).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1]).astype(np.float32)

    dist = tfd.Multinomial(ns, p, validate_args=True)
    dist2 = tfd.Multinomial(ns2, p2, validate_args=True)

    covariance = dist.covariance()
    covariance2 = dist2.covariance()
    self.assertEqual((3, 5, 4, 4), covariance.shape)
    self.assertEqual((6, 3, 3, 3), covariance2.shape)

  def testCovarianceFromSampling(self):
    if tf.executing_eagerly():
      raise unittest.SkipTest(
          'testCovarianceFromSampling is too slow to test in Eager mode')
    # We will test mean, cov, var, stddev on a Multinomial constructed via
    # broadcast between alpha, n.
    theta = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    theta /= np.sum(theta, 1)[..., tf.newaxis]
    n = np.array([[10., 9.], [8., 7.], [6., 5.]], dtype=np.float32)
    # batch_shape=[3, 2], event_shape=[3]
    dist = tfd.Multinomial(n, theta, validate_args=True)
    x = dist.sample(int(100e3), seed=test_util.test_seed())
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
    self.assertAllClose(sample_mean_, analytic_mean, atol=0.1, rtol=0.1)
    self.assertAllClose(sample_cov_, analytic_cov, atol=0.1, rtol=0.1)
    self.assertAllClose(sample_var_, analytic_var, atol=0.1, rtol=0.1)
    self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.1, rtol=0.1)

  def testSampleUnbiasedNonScalarBatch(self):
    dist = tfd.Multinomial(
        total_count=[7., 6., 5.],
        logits=tf.math.log(2. * self._rng.rand(4, 3, 2).astype(np.float32)),
        validate_args=True)
    n = int(3e4)
    x = dist.sample(n, seed=test_util.test_seed())
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
    self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4, 3, 2, 2], sample_covariance.shape)
    self.assertAllClose(
        actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testSampleUnbiasedScalarBatch(self):
    dist = tfd.Multinomial(
        total_count=5.,
        logits=tf.math.log(2. * self._rng.rand(4).astype(np.float32)),
        validate_args=True)
    n = int(3e4)
    x = dist.sample(n, seed=test_util.test_seed())
    sample_mean = tf.reduce_mean(x, axis=0)
    x_centered = x - sample_mean  # Already transposed to [n, 2].
    sample_covariance = tf.matmul(
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
    self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4, 4], sample_covariance.shape)
    self.assertAllClose(
        actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def propSampleCorrectMarginals(
      self, dist, special_class, under_hypothesis=False):
    # Property: When projected on one class, multinomial should sample the
    # binomial distribution.
    seed = test_util.test_seed()
    num_samples = 120000
    needed = self.evaluate(st.min_num_samples_for_dkwm_cdf_test(
        0.02, false_fail_rate=1e-9, false_pass_rate=1e-9))
    self.assertGreater(num_samples, needed)
    samples = dist.sample(num_samples, seed=seed)
    successes = samples[..., special_class]
    prob_success = dist._probs_parameter_no_checks()[..., special_class]
    if under_hypothesis:
      hp.note('Expected probability of success {}'.format(prob_success))
      hp.note('Successes obtained {}'.format(successes))
    expected_dist = tfd.Binomial(dist.total_count, probs=prob_success)
    self.evaluate(st.assert_true_cdf_equal_by_dkwm(
        successes, expected_dist.cdf,
        st.left_continuous_cdf_discrete_distribution(expected_dist),
        false_fail_rate=1e-9))

  @parameterized.parameters(
      (50., [0.25, 0.25, 0.25, 0.25], 2),
      ([0., 1., 25., 100.], [0., 0.1, 0.35, 0.55], 0),
      ([0., 1., 25., 100.], [0., 0.1, 0.35, 0.55], 1),
      ([0., 1., 25., 100.], [0., 0.1, 0.35, 0.55], 2),
  )
  def testSampleCorrectMarginals(self, total_counts, probs, index):
    if tf.executing_eagerly():
      raise unittest.SkipTest(
          'testSampleCorrectMarginals is too slow to test in Eager mode')
    dist = tfd.Multinomial(total_counts, probs=probs)
    self.propSampleCorrectMarginals(dist, index)

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def manual_testSampleCorrectMarginalsWithHypothesis(self, data):
    # You probably want --test_timeout=300 for this one
    dist = data.draw(dhps.distributions(dist_name='Multinomial'))
    special_class = data.draw(
        hps.sampled_from(range(dist._probs_parameter_no_checks().shape[-1])))
    # TODO(axch): Drawing the test seed inside the property will interact poorly
    # with --vary_seed under Hypothesis, because the seed will change as
    # Hypothesis tries to shrink failing examples.  It would be better to
    # compute the test seed outside the test method somehow, so each example
    # gets the same one.
    # TODO(axch): Not sure how to think about the false positive rate of this
    # test.  Hypothesis will try an adversarial search to make the statistical
    # assertion fail, which seems like a multiple comparisons problem with a
    # combinatorially large space of alternatives?
    self.propSampleCorrectMarginals(dist, special_class, under_hypothesis=True)

  def testNotReparameterized(self):
    if tf1.control_flow_v2_enabled():
      self.skipTest('b/138796859')
    total_count = tf.constant(5.0)
    probs = tf.constant([0.4, 0.6])
    _, [grad_total_count, grad_probs] = tfp.math.value_and_gradient(
        lambda n, p: tfd.Multinomial(  # pylint: disable=g-long-lambda
            total_count=n,
            probs=p,
            validate_args=True).sample(
                100, seed=test_util.test_seed()),
        [total_count, probs])
    self.assertIsNone(grad_total_count)
    self.assertIsNone(grad_probs)

  def testParamTensorFromLogits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = tfd.Multinomial(total_count=1, logits=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([x, d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([tf.math.softmax(x),
                        d.probs_parameter()]),
        atol=0,
        rtol=1e-4)

  def testParamTensorFromProbs(self):
    x = tf.constant([0.1, 0.5, 0.4])
    d = tfd.Multinomial(total_count=1, probs=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([tf.math.log(x), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([x, d.probs_parameter()]),
        atol=0, rtol=1e-4)


@test_util.test_all_tf_execution_regimes
class MultinomialFromVariableTest(test_util.TestCase):

  @test_util.jax_disable_variable_test
  def testGradientLogits(self):
    x = tf.Variable([-1., 0., 1])
    d = tfd.Multinomial(total_count=2., logits=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0, 0, 2])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  @test_util.jax_disable_variable_test
  def testGradientProbs(self):
    x = tf.Variable([0.1, 0.7, 0.2])
    d = tfd.Multinomial(total_count=2., probs=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0, 1, 1])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  def testAssertionsProbs(self):
    x = tf.Variable([0.1, 0.7, 0.0])
    with self.assertRaisesOpError('Argument `probs` must sum to 1.'):
      d = tfd.Multinomial(total_count=2., probs=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.mean())

  def testAssertionsLogits(self):
    x = tfp.util.TransformedVariable(0., tfb.Identity(), shape=None)
    with self.assertRaisesRegexp(
        ValueError, 'Argument `logits` must have rank at least 1.'):
      d = tfd.Multinomial(total_count=2., logits=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.mean())


if __name__ == '__main__':
  tf.test.main()
