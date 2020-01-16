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

rng = np.random.RandomState(0)
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class DeterministicTest(test_util.TestCase):

  def testShape(self):
    loc = rng.rand(2, 3, 4)
    deterministic = tfd.Deterministic(loc, validate_args=True)

    self.assertAllEqual(
        self.evaluate(deterministic.batch_shape_tensor()), (2, 3, 4))
    self.assertAllEqual(deterministic.batch_shape, (2, 3, 4))
    self.assertAllEqual(self.evaluate(deterministic.event_shape_tensor()), [])
    self.assertEqual(deterministic.event_shape, tf.TensorShape([]))

  def testInvalidTolRaises(self):
    loc = rng.rand(2, 3, 4).astype(np.float32)
    with self.assertRaisesOpError('Condition x >= 0'):
      deterministic = tfd.Deterministic(loc, atol=-1, validate_args=True)
      self.evaluate(deterministic.prob(0.))

  def testProbWithNoBatchDimsIntegerType(self):
    deterministic = tfd.Deterministic(0, validate_args=True)
    self.assertAllClose(1, self.evaluate(deterministic.prob(0)))
    self.assertAllClose(0, self.evaluate(deterministic.prob(2)))
    self.assertAllClose([1, 0], self.evaluate(deterministic.prob([0, 2])))

  def testProbWithNoBatchDims(self):
    deterministic = tfd.Deterministic(0., validate_args=True)
    self.assertAllClose(1., self.evaluate(deterministic.prob(0.)))
    self.assertAllClose(0., self.evaluate(deterministic.prob(2.)))
    self.assertAllClose([1., 0.], self.evaluate(deterministic.prob([0., 2.])))

  def testProbWithDefaultTol(self):
    loc = [[0., 1.], [2., 3.]]
    x = [[0., 1.1], [1.99, 3.]]
    deterministic = tfd.Deterministic(loc, validate_args=True)
    expected_prob = [[1., 0.], [0., 1.]]
    prob = deterministic.prob(x)
    self.assertAllEqual((2, 2), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testLogProbWithDefaultTolInt32Loc(self):
    loc = [[0, 1], [2, 3]]
    x = [[0, 2], [1, 3]]
    deterministic = tfd.Deterministic(
        tf.constant(loc, dtype=tf.int32), validate_args=True)
    expected_logprob = [[0., float('-inf')], [float('-inf'), 0.]]
    prob = deterministic.log_prob(x)
    self.assertAllEqual((2, 2), prob.shape)
    self.assertAllEqual(expected_logprob, self.evaluate(prob))

  def testProbWithNonzeroATol(self):
    loc = [[0., 1.], [2., 3.]]
    x = [[0., 1.1], [1.99, 3.]]
    deterministic = tfd.Deterministic(loc, atol=0.05, validate_args=True)
    expected_prob = [[1., 0.], [1., 1.]]
    prob = deterministic.prob(x)
    self.assertAllEqual((2, 2), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbWithNonzeroATolIntegerType(self):
    loc = [[0, 1], [2, 3]]
    x = [[0, 2], [4, 2]]
    deterministic = tfd.Deterministic(loc, atol=1, validate_args=True)
    expected_prob = [[1, 1], [0, 1]]
    prob = deterministic.prob(x)
    self.assertAllEqual((2, 2), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbWithNonzeroRTol(self):
    loc = [[0., 1.], [100., 100.]]
    x = [[0., 1.1], [100.1, 103.]]
    deterministic = tfd.Deterministic(loc, rtol=0.01, validate_args=True)
    expected_prob = [[1., 0.], [1., 0.]]
    prob = deterministic.prob(x)
    self.assertAllEqual((2, 2), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbWithRTolBcastsLoc(self):
    loc = [100., 200.]
    rtol = [[0.01], [0.02]]
    x = [[102., 201.9], [99.1, 205.]]
    deterministic = tfd.Deterministic(loc, rtol=rtol, validate_args=True)
    expected_prob = [[0., 1.], [1., 0.]]
    prob = deterministic.prob(x)
    self.assertAllEqual((2, 2), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbWithNonzeroRTolIntegerType(self):
    loc = [[10, 10, 10], [10, 10, 10]]
    x = [[10, 20, 30], [10, 20, 30]]
    # Batch 0 will have rtol = 0
    # Batch 1 will have rtol = 1 (100% slack allowed)
    deterministic = tfd.Deterministic(loc, rtol=[[0], [1]], validate_args=True)
    expected_prob = [[1, 0, 0], [1, 1, 0]]
    prob = deterministic.prob(x)
    self.assertAllEqual((2, 3), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testCdfWithDefaultTol(self):
    loc = [[0., 0.], [0., 0.]]
    x = [[-1., -0.1], [-0.01, 1.000001]]
    deterministic = tfd.Deterministic(loc, validate_args=True)
    expected_cdf = [[0., 0.], [0., 1.]]
    cdf = deterministic.cdf(x)
    self.assertAllEqual((2, 2), cdf.shape)
    self.assertAllEqual(expected_cdf, self.evaluate(cdf))

  def testCdfWithNonzeroATol(self):
    loc = [[0., 0.], [0., 0.]]
    x = [[-1., -0.1], [-0.01, 1.000001]]
    deterministic = tfd.Deterministic(loc, atol=0.05, validate_args=True)
    expected_cdf = [[0., 0.], [1., 1.]]
    cdf = deterministic.cdf(x)
    self.assertAllEqual((2, 2), cdf.shape)
    self.assertAllEqual(expected_cdf, self.evaluate(cdf))

  def testCdfWithNonzeroRTol(self):
    loc = [[1., 1.], [100., 100.]]
    x = [[0.9, 1.], [99.9, 97]]
    deterministic = tfd.Deterministic(loc, rtol=0.01, validate_args=True)
    expected_cdf = [[0., 1.], [1., 0.]]
    cdf = deterministic.cdf(x)
    self.assertAllEqual((2, 2), cdf.shape)
    self.assertAllEqual(expected_cdf, self.evaluate(cdf))

  def testSampleNoBatchDims(self):
    deterministic = tfd.Deterministic(0., validate_args=True)
    for sample_shape in [(), (4,)]:
      sample = deterministic.sample(
          sample_shape, seed=test_util.test_seed())
      self.assertAllEqual(sample_shape, sample.shape)
      self.assertAllClose(
          np.zeros(sample_shape).astype(np.float32), self.evaluate(sample))

  def testSampleWithBatchDims(self):
    deterministic = tfd.Deterministic([0., 0.], validate_args=True)
    for sample_shape in [(), (4,)]:
      sample = deterministic.sample(sample_shape, seed=test_util.test_seed())
      self.assertAllEqual(sample_shape + (2,), sample.shape)
      self.assertAllClose(
          np.zeros(sample_shape + (2,)).astype(np.float32),
          self.evaluate(sample))

  def testSampleWithBatchAtol(self):
    deterministic = tfd.Deterministic(0., atol=[.1, .1], validate_args=True)
    for sample_shape in [(), (4,)]:
      sample = deterministic.sample(sample_shape, seed=test_util.test_seed())
      self.assertAllEqual(sample_shape + (2,), sample.shape)
      self.assertAllClose(
          np.zeros(sample_shape + (2,)).astype(np.float32),
          self.evaluate(sample))

  def testSampleDynamicWithBatchDims(self):
    loc = tf1.placeholder_with_default([0., 0], shape=[2])

    deterministic = tfd.Deterministic(loc, validate_args=True)
    for sample_shape_ in [(), (4,)]:
      sample_shape = tf1.placeholder_with_default(
          np.array(sample_shape_, dtype=np.int32), shape=None)
      sample_ = self.evaluate(deterministic.sample(
          sample_shape, seed=test_util.test_seed()))
      self.assertAllClose(
          np.zeros(sample_shape_ + (2,)).astype(np.float32), sample_)

  def testEntropy(self):
    loc = np.array([-0.1, -3.2, 7.])
    deterministic = tfd.Deterministic(loc=loc, validate_args=True)
    entropy_ = self.evaluate(deterministic.entropy())
    self.assertAllEqual(np.zeros(3), entropy_)

  def testDeterministicDeterministicKL(self):
    batch_size = 6
    a_loc = np.array([0.5] * batch_size, dtype=np.float32)
    b_loc = np.array([0.4] * batch_size, dtype=np.float32)

    a = tfd.Deterministic(loc=a_loc, validate_args=True)
    b = tfd.Deterministic(loc=b_loc, validate_args=True)

    kl = tfd.kl_divergence(a, b)
    kl_ = self.evaluate(kl)
    self.assertAllEqual(np.zeros(6) + np.inf, kl_)

  def testDeterministicGammaKL(self):
    batch_size = 2
    a_loc = np.array([0.5] * batch_size, dtype=np.float32)
    b_concentration = np.array([3.2] * batch_size, dtype=np.float32)
    b_rate = np.array([4.1] * batch_size, dtype=np.float32)

    a = tfd.Deterministic(loc=a_loc, validate_args=True)
    b = tfd.Gamma(concentration=b_concentration, rate=b_rate)

    expected_kl = -b.log_prob(a_loc)
    actual_kl = tfd.kl_divergence(a, b)
    expected_kl_, actual_kl_ = self.evaluate([expected_kl, actual_kl])
    self.assertAllEqual(expected_kl_, actual_kl_)

  @test_util.tf_tape_safety_test
  def testVariableGradients(self):
    loc = tf.Variable(1.)
    deterministic = tfd.Deterministic(loc=loc, validate_args=True)
    with tf.GradientTape() as tape:
      s = deterministic.sample(seed=test_util.test_seed())
    g = tape.gradient(s, deterministic.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  def testVariableAssertions(self):
    atol = tf.Variable(0.1)
    rtol = tf.Variable(0.1)
    deterministic = tfd.Deterministic(
        loc=0.1, atol=atol, rtol=rtol, validate_args=True)

    self.evaluate([v.initializer for v in deterministic.variables])
    self.evaluate(deterministic.log_prob(1.))

    self.evaluate(atol.assign(-1.))
    with self.assertRaisesRegexp((ValueError, tf.errors.InvalidArgumentError),
                                 'Condition x >= 0'):
      self.evaluate(deterministic.log_prob(1.))

    self.evaluate(atol.assign(0.1))
    self.evaluate(rtol.assign(-1.))
    with self.assertRaisesRegexp((ValueError, tf.errors.InvalidArgumentError),
                                 'Condition x >= 0'):
      self.evaluate(deterministic.log_prob(1.))


class VectorDeterministicTest(test_util.TestCase):

  def testParamBroadcasts(self):
    loc = rng.rand(2, 1, 4)
    atol = np.abs(rng.rand(2, 3, 1))
    rtol = np.abs(rng.rand(7, 2, 3, 1))
    deterministic = tfd.VectorDeterministic(
        loc, atol=atol, rtol=rtol, validate_args=True)

    self.assertAllEqual(
        self.evaluate(deterministic.batch_shape_tensor()), (7, 2, 3))
    self.assertAllEqual(deterministic.batch_shape, (7, 2, 3))
    self.assertAllEqual(self.evaluate(deterministic.event_shape_tensor()), [4])
    self.assertEqual(deterministic.event_shape, tf.TensorShape([4]))

  def testShape(self):
    loc = rng.rand(2, 3, 4)
    deterministic = tfd.VectorDeterministic(loc, validate_args=True)

    self.assertAllEqual(
        self.evaluate(deterministic.batch_shape_tensor()), (2, 3))
    self.assertAllEqual(deterministic.batch_shape, (2, 3))
    self.assertAllEqual(self.evaluate(deterministic.event_shape_tensor()), [4])
    self.assertEqual(deterministic.event_shape, tf.TensorShape([4]))

  def testShapeUknown(self):
    loc = tf1.placeholder_with_default(np.float32([0]), shape=[None])
    deterministic = tfd.VectorDeterministic(loc, validate_args=True)
    self.assertAllEqual(deterministic.event_shape_tensor().shape, [1])

  def testInvalidTolRaises(self):
    loc = rng.rand(2, 3, 4).astype(np.float32)
    with self.assertRaisesRegexp((ValueError, tf.errors.InvalidArgumentError),
                                 'Condition x >= 0'):
      deterministic = tfd.VectorDeterministic(loc, atol=-1, validate_args=True)
      self.evaluate(deterministic.prob(loc))

  def testInvalidXRaises(self):
    loc = rng.rand(2, 3, 4).astype(np.float32)
    with self.assertRaisesRegexp((ValueError, tf.errors.InvalidArgumentError),
                                 'rank at least 1'):
      deterministic = tfd.VectorDeterministic(loc, atol=1, validate_args=True)
      self.evaluate(deterministic.prob(0.))

  def testProbVectorDeterministicWithNoBatchDims(self):
    # 0 batch of deterministics on R^1.
    deterministic = tfd.VectorDeterministic([0.], validate_args=True)
    self.assertAllClose(1., self.evaluate(deterministic.prob([0.])))
    self.assertAllClose(0., self.evaluate(deterministic.prob([2.])))
    self.assertAllClose([1., 0.], self.evaluate(
        deterministic.prob([[0.], [2.]])))

  def testProbWithDefaultTol(self):
    # 3 batch of deterministics on R^2.
    loc = [[0., 1.], [2., 3.], [4., 5.]]
    x = [[0., 1.], [1.9, 3.], [3.99, 5.]]
    deterministic = tfd.VectorDeterministic(loc, validate_args=True)
    expected_prob = [1., 0., 0.]
    prob = deterministic.prob(x)
    self.assertAllEqual((3,), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbWithNonzeroATol(self):
    # 3 batch of deterministics on R^2.
    loc = [[0., 1.], [2., 3.], [4., 5.]]
    x = [[0., 1.], [1.9, 3.], [3.99, 5.]]
    deterministic = tfd.VectorDeterministic(loc, atol=0.05, validate_args=True)
    expected_prob = [1., 0., 1.]
    prob = deterministic.prob(x)
    self.assertAllEqual((3,), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbWithATolBcastsLoc(self):
    # 3 batch of deterministics on R^2.
    loc = [0., 1.]
    atol = [[0.02], [0.03], [0.05]]
    x = [[0., .99], [0.04, 1.], [0., 1.045]]
    deterministic = tfd.VectorDeterministic(loc, atol=atol, validate_args=True)
    expected_prob = [1., 0., 1.]
    prob = deterministic.prob(x)
    self.assertAllEqual((3,), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbWithNonzeroRTol(self):
    # 3 batch of deterministics on R^2.
    loc = [[0., 1.], [1., 1.], [100., 100.]]
    x = [[0., 1.], [0.9, 1.], [99.9, 100.1]]
    deterministic = tfd.VectorDeterministic(loc, rtol=0.01, validate_args=True)
    expected_prob = [1., 0., 1.]
    prob = deterministic.prob(x)
    self.assertAllEqual((3,), prob.shape)
    self.assertAllEqual(expected_prob, self.evaluate(prob))

  def testProbVectorDeterministicWithNoBatchDimsOnRZero(self):
    # 0 batch of deterministics on R^0.
    deterministic = tfd.VectorDeterministic([], validate_args=True)
    self.assertAllClose(1., self.evaluate(deterministic.prob([])))

  def testProbVectorDeterministicWithNoBatchDimsOnRZeroRaisesIfXNotInSameRk(
      self):
    # 0 batch of deterministics on R^0.
    deterministic = tfd.VectorDeterministic([], validate_args=True)
    with self.assertRaisesOpError('not defined in the same space'):
      self.evaluate(deterministic.prob([1.]))

  def testSampleNoBatchDims(self):
    deterministic = tfd.VectorDeterministic([0.], validate_args=True)
    for sample_shape in [(), (4,)]:
      sample = deterministic.sample(sample_shape, seed=test_util.test_seed())
      self.assertAllEqual(sample_shape + (1,), sample.shape)
      self.assertAllClose(
          np.zeros(sample_shape + (1,)).astype(np.float32),
          self.evaluate(sample))

  def testSampleWithBatchDims(self):
    deterministic = tfd.VectorDeterministic([[0.], [0.]], validate_args=True)
    for sample_shape in [(), (4,)]:
      sample = deterministic.sample(sample_shape, seed=test_util.test_seed())
      self.assertAllEqual(sample_shape + (2, 1), sample.shape)
      self.assertAllClose(
          np.zeros(sample_shape + (2, 1)).astype(np.float32),
          self.evaluate(sample))

  def testSampleDynamicWithBatchDims(self):
    loc = tf1.placeholder_with_default([[0.], [0.]], shape=[2, 1])

    deterministic = tfd.VectorDeterministic(loc, validate_args=True)
    for sample_shape_ in [(), (4,)]:
      sample_shape = tf1.placeholder_with_default(
          np.array(sample_shape_, dtype=np.int32), shape=None)
      sample_ = self.evaluate(deterministic.sample(
          sample_shape, seed=test_util.test_seed()))
      self.assertAllClose(
          np.zeros(sample_shape_ + (2, 1)).astype(np.float32), sample_)

  def testEntropy(self):
    loc = np.array([[8.3, 1.2, 3.3], [-0.1, -3.2, 7.]])
    deterministic = tfd.VectorDeterministic(loc=loc, validate_args=True)
    entropy_ = self.evaluate(deterministic.entropy())
    self.assertAllEqual(np.zeros(2), entropy_)

  def testVectorDeterministicVectorDeterministicKL(self):
    batch_size = 6
    event_size = 3
    a_loc = np.array([[0.5] * event_size] * batch_size, dtype=np.float32)
    b_loc = np.array([[0.4] * event_size] * batch_size, dtype=np.float32)

    a = tfd.VectorDeterministic(loc=a_loc, validate_args=True)
    b = tfd.VectorDeterministic(loc=b_loc, validate_args=True)

    kl = tfd.kl_divergence(a, b)
    kl_ = self.evaluate(kl)
    self.assertAllEqual(np.zeros(6) + np.inf, kl_)

  def testVectorDeterministicMultivariateNormalDiagKL(self):
    batch_size = 4
    event_size = 5
    a_loc = np.array([[0.5] * event_size] * batch_size, dtype=np.float32)
    b_loc = np.array([[0.4] * event_size] * batch_size, dtype=np.float32)
    b_scale_diag = np.array([[3.2] * event_size] * batch_size, dtype=np.float32)

    a = tfd.VectorDeterministic(loc=a_loc, validate_args=True)
    b = tfd.MultivariateNormalDiag(loc=b_loc, scale_diag=b_scale_diag)

    expected_kl = -b.log_prob(a_loc)
    actual_kl = tfd.kl_divergence(a, b)
    expected_kl_, actual_kl_ = self.evaluate([expected_kl, actual_kl])
    self.assertAllEqual(expected_kl_, actual_kl_)

  @test_util.tf_tape_safety_test
  def testVariableGradients(self):
    loc = tf.Variable([1., 2.])
    deterministic = tfd.VectorDeterministic(loc=loc, validate_args=True)
    with tf.GradientTape() as tape:
      s = deterministic.sample(seed=test_util.test_seed())
    g = tape.gradient(s, deterministic.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  def testVariableAssertions(self):
    atol = tf.Variable(0.1)
    rtol = tf.Variable(0.1)
    deterministic = tfd.VectorDeterministic(
        loc=[0.1], atol=atol, rtol=rtol, validate_args=True)

    self.evaluate([v.initializer for v in deterministic.variables])
    self.evaluate(deterministic.log_prob([1.]))

    self.evaluate(atol.assign(-1.))
    with self.assertRaisesRegexp((ValueError, tf.errors.InvalidArgumentError),
                                 'Condition x >= 0'):
      self.evaluate(deterministic.log_prob([1.]))

    self.evaluate(atol.assign(0.1))
    self.evaluate(rtol.assign(-1.))
    with self.assertRaisesRegexp((ValueError, tf.errors.InvalidArgumentError),
                                 'Condition x >= 0'):
      self.evaluate(deterministic.log_prob([1.]))


if __name__ == '__main__':
  tf.test.main()
