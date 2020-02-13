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
"""Tests for the ProbitBernoulli distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports
import numpy as np
from scipy import special as sp_special

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


def make_bernoulli(batch_shape, dtype=tf.int32):
  p = np.random.uniform(size=list(batch_shape))
  p = tf.constant(p, dtype=tf.float32)
  return tfd.ProbitBernoulli(probs=p, dtype=dtype, validate_args=True)


def entropy(p):
  q = 1. - p
  return -q * np.log(q) - p * np.log(p)


@test_util.test_all_tf_execution_regimes
class ProbitBernoulliTest(test_util.TestCase):

  def testP(self):
    p = [0.2, 0.4]
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    self.assertAllClose(p, self.evaluate(dist.probs))

  def testProbits(self):
    probits = [-42., 42.]
    dist = tfd.ProbitBernoulli(probits=probits, validate_args=True)
    self.assertAllClose(probits, self.evaluate(dist.probits))
    self.assertAllClose(
        sp_special.ndtr(probits), self.evaluate(dist.probs_parameter()))

    p = [0.01, 0.99, 0.42]
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    self.assertAllClose(
        sp_special.ndtri(p), self.evaluate(dist.probits_parameter()))

  def testInvalidP(self):
    invalid_ps = [1.01, 2.]
    for p in invalid_ps:
      with self.assertRaisesOpError('probs has components greater than 1'):
        dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
        self.evaluate(dist.probs_parameter())

    invalid_ps = [-0.01, -3.]
    for p in invalid_ps:
      with self.assertRaisesOpError('x >= 0 did not hold'):
        dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
        self.evaluate(dist.probs_parameter())

    valid_ps = [0.0, 0.5, 1.0]
    for p in valid_ps:
      dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
      self.assertEqual(p, self.evaluate(dist.probs))  # Should not fail

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_bernoulli(batch_shape)
      self.assertAllEqual(batch_shape,
                          tensorshape_util.as_list(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], tensorshape_util.as_list(dist.event_shape))
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))

  def testDtype(self):
    dist = make_bernoulli([])
    self.assertEqual(dist.dtype, tf.int32)
    self.assertEqual(dist.dtype, dist.sample(
        5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs.dtype, dist.mean().dtype)
    self.assertEqual(dist.probs.dtype, dist.variance().dtype)
    self.assertEqual(dist.probs.dtype, dist.stddev().dtype)
    self.assertEqual(dist.probs.dtype, dist.entropy().dtype)
    self.assertEqual(dist.probs.dtype, dist.prob(0).dtype)
    self.assertEqual(dist.probs.dtype, dist.prob(1).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob(0).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob(1).dtype)

    dist64 = make_bernoulli([], tf.int64)
    self.assertEqual(dist64.dtype, tf.int64)
    self.assertEqual(dist64.dtype, dist64.sample(
        5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist64.dtype, dist64.mode().dtype)

  def testFloatMode(self):
    dist = tfd.ProbitBernoulli(probs=.6, dtype=tf.float32, validate_args=True)
    self.assertEqual(np.float32(1), self.evaluate(dist.mode()))

  def _testPmf(self, **kwargs):
    dist = tfd.ProbitBernoulli(**kwargs)
    # pylint: disable=bad-continuation
    xs = [
        0,
        [1],
        [1, 0],
        [[1, 0]],
        [[1, 0], [1, 1]],
    ]
    expected_pmfs = [
        [[0.8, 0.6], [0.7, 0.4]],
        [[0.2, 0.4], [0.3, 0.6]],
        [[0.2, 0.6], [0.3, 0.4]],
        [[0.2, 0.6], [0.3, 0.4]],
        [[0.2, 0.6], [0.3, 0.6]],
    ]
    # pylint: enable=bad-continuation

    for x, expected_pmf in zip(xs, expected_pmfs):
      self.assertAllClose(self.evaluate(dist.prob(x)), expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(x)), np.log(expected_pmf))

  def testPmfCorrectBroadcastDynamicShape(self):
    p = tf1.placeholder_with_default([0.2, 0.3, 0.4], shape=None)
    dist = tfd.ProbitBernoulli(probs=p)
    event1 = [1, 0, 1]
    event2 = [[1, 0, 1]]
    self.assertAllClose(
        [0.2, 0.7, 0.4], self.evaluate(dist.prob(event1)))
    self.assertAllClose(
        [[0.2, 0.7, 0.4]], self.evaluate(dist.prob(event2)))

  def testPmfInvalid(self):
    p = [0.1, 0.2, 0.7]
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    with self.assertRaisesOpError('must be non-negative.'):
      self.evaluate(dist.prob([1, 1, -1]))
    with self.assertRaisesOpError('Elements cannot exceed 1.'):
      self.evaluate(dist.prob([2, 0, 1]))

  def testPmfWithP(self):
    p = [[0.2, 0.4], [0.3, 0.6]]
    self._testPmf(probs=p, validate_args=True)
    self._testPmf(probits=sp_special.ndtri(p), validate_args=True)

  def testPmfWithFloatArgReturnsXEntropy(self):
    p = [[0.2], [0.4], [0.3], [0.6]]
    samps = [0, 0.1, 0.8]
    self.assertAllClose(
        np.float32(samps) * np.log(np.float32(p)) +
        (1 - np.float32(samps)) * np.log(1 - np.float32(p)),
        self.evaluate(
            tfd.ProbitBernoulli(probs=p, validate_args=False).log_prob(samps)))

  def testBroadcasting(self):
    probs = lambda p: tf1.placeholder_with_default(p, shape=None)
    dist = lambda p: tfd.ProbitBernoulli(probs=probs(p), validate_args=True)
    self.assertAllClose(np.log(0.5), self.evaluate(dist(0.5).log_prob(1)))
    self.assertAllClose(
        np.log([0.5, 0.5, 0.5]), self.evaluate(dist(0.5).log_prob([1, 1, 1])))
    self.assertAllClose(np.log([0.5, 0.5, 0.5]),
                        self.evaluate(dist([0.5, 0.5, 0.5]).log_prob(1)))

  def testPmfShapes(self):
    probs = lambda p: tf1.placeholder_with_default(p, shape=None)
    dist = lambda p: tfd.ProbitBernoulli(probs=probs(p), validate_args=True)
    self.assertEqual(
        2, len(self.evaluate(dist([[0.5], [0.5]]).log_prob(1)).shape))

    dist = tfd.ProbitBernoulli(probs=0.5, validate_args=True)
    self.assertEqual(2, len(self.evaluate(dist.log_prob([[1], [1]])).shape))

    dist = tfd.ProbitBernoulli(probs=0.5, validate_args=True)
    self.assertAllEqual([], dist.log_prob(1).shape)
    self.assertAllEqual([1], dist.log_prob([1]).shape)
    self.assertAllEqual([2, 1], dist.log_prob([[1], [1]]).shape)

    dist = tfd.ProbitBernoulli(probs=[[0.5], [0.5]], validate_args=True)
    self.assertAllEqual([2, 1], dist.log_prob(1).shape)

  def testBoundaryConditions(self):
    dist = tfd.ProbitBernoulli(probs=1.0, validate_args=True)
    self.assertAllClose(np.nan, self.evaluate(dist.log_prob(0)))
    self.assertAllClose([np.nan], [self.evaluate(dist.log_prob(1))])

  def testEntropyNoBatch(self):
    p = 0.2
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    self.assertAllClose(self.evaluate(dist.entropy()), entropy(p))

  def testEntropyWithBatch(self):
    p = [[0.1, 0.7], [0.2, 0.6]]
    dist = tfd.ProbitBernoulli(probs=p, validate_args=False)
    self.assertAllClose(
        self.evaluate(dist.entropy()),
        [[entropy(0.1), entropy(0.7)], [entropy(0.2),
                                        entropy(0.6)]])

  def testSampleN(self):
    p = [0.2, 0.6]
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    n = 100000
    samples = dist.sample(n, seed=test_util.test_seed())
    tensorshape_util.set_shape(samples, [n, 2])
    self.assertEqual(samples.dtype, tf.int32)
    sample_values = self.evaluate(samples)
    self.assertTrue(np.all(sample_values >= 0))
    self.assertTrue(np.all(sample_values <= 1))
    # Note that the standard error for the sample mean is ~ sqrt(p * (1 - p) /
    # n). This means that the tolerance is very sensitive to the value of p
    # as well as n.
    self.assertAllClose(p, np.mean(sample_values, axis=0), atol=1e-2)
    self.assertEqual(set([0, 1]), set(sample_values.flatten()))
    # In this test we're just interested in verifying there isn't a crash
    # owing to mismatched types. b/30940152
    dist = tfd.ProbitBernoulli(np.log([.2, .4]), validate_args=True)
    x = dist.sample(1, seed=test_util.test_seed())
    self.assertAllEqual((1, 2), tensorshape_util.as_list(x.shape))

  @test_util.jax_disable_test_missing_functionality(
      'JAX does not return None for gradients.')
  def testNotReparameterized(self):
    p = tf.constant([0.2, 0.6])
    _, grad_p = tfp.math.value_and_gradient(
        lambda x: tfd.ProbitBernoulli(  # pylint: disable=g-long-lambda
            probs=x,
            validate_args=True).sample(
                100, seed=test_util.test_seed()),
        p)
    self.assertIsNone(grad_p)

  def testSampleDeterministicScalarVsVector(self):
    p = [0.2, 0.6]
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    n = 1000
    def _seed(seed=None):
      seed = test_util.test_seed() if seed is None else seed
      if tf.executing_eagerly():
        tf.random.set_seed(seed)
      return seed
    seed = _seed()
    self.assertAllEqual(
        self.evaluate(dist.sample(n, seed)),
        self.evaluate(dist.sample([n], _seed(seed))))
    n = tf1.placeholder_with_default(np.int32(1000), shape=None)
    seed = _seed()
    sample1 = dist.sample(n, seed)
    sample2 = dist.sample([n], _seed(seed=seed))
    sample1, sample2 = self.evaluate([sample1, sample2])
    self.assertAllEqual(sample1, sample2)

  def testMean(self):
    p = np.array([[0.2, 0.7], [0.5, 0.4]], dtype=np.float32)
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    self.assertAllEqual(self.evaluate(dist.mean()), p)

  def testVarianceAndStd(self):
    var = lambda p: p * (1. - p)
    p = [[0.2, 0.7], [0.5, 0.4]]
    dist = tfd.ProbitBernoulli(probs=p, validate_args=True)
    self.assertAllClose(
        self.evaluate(dist.variance()),
        np.array([[var(0.2), var(0.7)], [var(0.5), var(0.4)]],
                 dtype=np.float32))
    self.assertAllClose(
        self.evaluate(dist.stddev()),
        np.array([[np.sqrt(var(0.2)), np.sqrt(var(0.7))],
                  [np.sqrt(var(0.5)), np.sqrt(var(0.4))]],
                 dtype=np.float32))

  def testProbitBernoulliProbitBernoulliKL(self):
    batch_size = 6
    a_p = np.array([0.6] * batch_size, dtype=np.float32)
    b_p = np.array([0.4] * batch_size, dtype=np.float32)

    a = tfd.ProbitBernoulli(probs=a_p, validate_args=True)
    b = tfd.ProbitBernoulli(probs=b_p, validate_args=True)

    kl = tfd.kl_divergence(a, b)
    kl_val = self.evaluate(kl)

    kl_expected = (a_p * np.log(a_p / b_p) + (1. - a_p) * np.log(
        (1. - a_p) / (1. - b_p)))

    self.assertEqual(kl.shape, (batch_size,))
    self.assertAllClose(kl_val, kl_expected)

  def testParamTensorFromProbits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = tfd.ProbitBernoulli(probits=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([tf.math.ndtri(d.prob(1.)),
                        d.probits_parameter()]),
        atol=0,
        rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([d.prob(1.), d.probs_parameter()]), atol=0, rtol=1e-4)

  def testParamTensorFromProbs(self):
    x = tf.constant([0.1, 0.5, 0.4])
    d = tfd.ProbitBernoulli(probs=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([tf.math.ndtri(d.prob(1.)),
                        d.probits_parameter()]),
        atol=0,
        rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([d.prob(1.), d.probs_parameter()]), atol=0, rtol=1e-4)


@test_util.test_all_tf_execution_regimes
class ProbitBernoulliFromVariableTest(test_util.TestCase):

  @test_util.tf_tape_safety_test
  def testGradientProbits(self):
    x = tf.Variable([-1., 1])
    self.evaluate(x.initializer)
    d = tfd.ProbitBernoulli(probits=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0, 1])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  @test_util.tf_tape_safety_test
  def testGradientProbs(self):
    x = tf.Variable([0.1, 0.7])
    self.evaluate(x.initializer)
    d = tfd.ProbitBernoulli(probs=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0, 1])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  def testAssertionsProbs(self):
    x = tf.Variable([0.1, 0.7, 0.0])
    d = tfd.ProbitBernoulli(probs=x, validate_args=True)
    self.evaluate(x.initializer)
    self.evaluate(d.entropy())
    with tf.control_dependencies([x.assign([0.1, -0.7, 0.0])]):
      with self.assertRaisesOpError('x >= 0 did not hold'):
        self.evaluate(d.entropy())


if __name__ == '__main__':
  tf.test.main()
