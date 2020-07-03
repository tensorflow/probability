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
"""Tests for OneHotCategorical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


def make_onehot_categorical(batch_shape, num_classes, dtype=tf.int32):
  logits = -50. + samplers.uniform(
      list(batch_shape) + [num_classes], -10, 10,
      dtype=tf.float32, seed=test_util.test_seed())
  return tfd.OneHotCategorical(logits, dtype=dtype, validate_args=True)


@test_util.test_all_tf_execution_regimes
class OneHotCategoricalTest(test_util.TestCase):

  def setUp(self):
    super(OneHotCategoricalTest, self).setUp()
    self._rng = np.random.RandomState(42)

  def assertRaises(self, error_class, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(error_class, msg)
    return self.assertRaisesOpError(msg)

  def testP(self):
    p = [0.2, 0.8]
    dist = tfd.OneHotCategorical(probs=p, validate_args=True)
    self.assertAllClose(p, self.evaluate(dist.probs))
    self.assertAllEqual([2], dist.logits_parameter().shape)

  def testLogits(self):
    p = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.log(p) - 50.
    dist = tfd.OneHotCategorical(logits=logits, validate_args=True)
    self.assertAllEqual([2], dist.probs_parameter().shape)
    self.assertAllEqual([2], dist.logits.shape)
    self.assertAllClose(self.evaluate(dist.probs_parameter()), p)
    self.assertAllClose(self.evaluate(dist.logits), logits)

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_onehot_categorical(batch_shape, 10)
      self.assertAllEqual(batch_shape,
                          tensorshape_util.as_list(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], tensorshape_util.as_list(dist.event_shape))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))
      # event_shape is available as a constant because the shape is
      # known at graph build time.
      self.assertEqual(10, dist.event_shape)

    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_onehot_categorical(batch_shape, tf.constant(
          10, dtype=tf.int32))
      self.assertAllEqual(
          len(batch_shape), tensorshape_util.rank(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], tensorshape_util.as_list(dist.event_shape))
      self.assertEqual(10, self.evaluate(dist.event_shape_tensor()))

  def testDtype(self):
    dist = make_onehot_categorical([], 5, dtype=tf.int32)
    self.assertEqual(dist.dtype, tf.int32)
    self.assertEqual(dist.dtype, dist.sample(
        5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_onehot_categorical([], 5, dtype=tf.int64)
    self.assertEqual(dist.dtype, tf.int64)
    self.assertEqual(dist.dtype, dist.sample(
        5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs_parameter().dtype, tf.float32)
    self.assertEqual(dist.logits.dtype, tf.float32)
    self.assertEqual(dist.logits.dtype, dist.entropy().dtype)
    self.assertEqual(dist.logits.dtype, dist.prob(
        np.array([1]+[0]*4, dtype=np.int64)).dtype)
    self.assertEqual(dist.logits.dtype, dist.log_prob(
        np.array([1]+[0]*4, dtype=np.int64)).dtype)

  def testUnknownShape(self):
    logits = tf1.placeholder_with_default(
        [[-1000.0, 1000.0], [1000.0, -1000.0]], shape=None)
    dist = tfd.OneHotCategorical(logits, validate_args=True)
    sample = dist.sample(seed=test_util.test_seed())
    # Batch entry 0 will sample class 1, batch entry 1 will sample class 0.
    sample_value_batch = self.evaluate(sample)
    self.assertAllEqual([[0, 1], [1, 0]], sample_value_batch)

  def testUnknownAndInvalidShape(self):
    logits = tf1.placeholder_with_default(19.84, shape=None)
    with self.assertRaises(
        ValueError, 'Argument `logits` must have rank at least 1.'):
      dist = tfd.OneHotCategorical(logits, validate_args=True)
      self.evaluate(dist.sample(seed=test_util.test_seed()))

    logits = tf1.placeholder_with_default([[], []], shape=None)
    with self.assertRaises(
        ValueError, 'Argument `logits` must have final dimension >= 1.'):
      dist = tfd.OneHotCategorical(logits, validate_args=True)
      self.evaluate(dist.sample(seed=test_util.test_seed()))

  def testAssertValidSample(self):
    logits = [0.1, 0.4, 0.5, 0.3, 0.1]
    dist = tfd.OneHotCategorical(logits, validate_args=True)
    with self.assertRaisesOpError('Condition x >= 0'):
      self.evaluate(dist.prob([-1., 0., 1., 1., 0.]))
    with self.assertRaisesOpError('Last dimension of sample must sum to 1.'):
      self.evaluate(dist.log_prob([1., 0., 1., 0., 0.]))

  def testEntropyNoBatch(self):
    logits = np.log([0.2, 0.8]) - 50.
    dist = tfd.OneHotCategorical(logits, validate_args=True)
    self.assertAllClose(self.evaluate(dist.entropy()),
                        -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                        atol=1e-5, rtol=1e-5)

  def testEntropyWithBatch(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = tfd.OneHotCategorical(logits, validate_args=True)
    self.assertAllClose(
        self.evaluate(dist.entropy()),
        [-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
         -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))],
        atol=1e-5, rtol=1e-5)

  def testPmf(self):
    # check that probability of samples correspond to their class probabilities
    logits = self._rng.random_sample(size=(8, 2, 10))
    prob = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    dist = tfd.OneHotCategorical(logits=logits, validate_args=True)
    np_sample = self.evaluate(dist.sample(seed=test_util.test_seed()))
    np_prob = self.evaluate(dist.prob(np_sample))
    expected_prob = prob[np_sample.astype(np.bool)]
    self.assertAllClose(expected_prob, np_prob.flatten())

  def testEventSizeOfOne(self):
    d = tfd.OneHotCategorical(
        probs=tf1.placeholder_with_default([1.], shape=None),
        validate_args=True)
    self.assertAllEqual(np.ones((5, 3, 1), dtype=np.int32),
                        self.evaluate(d.sample(
                            [5, 3], seed=test_util.test_seed())))
    self.assertAllClose([0., 0., 0., 0., 0.],
                        self.evaluate(d.log_prob(tf.ones((5, 1)))))
    self.assertAllClose(0., self.evaluate(d.entropy()))
    self.assertAllClose([0.], self.evaluate(d.variance()))
    self.assertAllClose([[0.]], self.evaluate(d.covariance()))
    self.assertAllClose([1.], self.evaluate(d.mode()))
    self.assertAllClose([1.], self.evaluate(d.mean()))

  def testSample(self):
    probs = [[[0.2, 0.8], [0.4, 0.6]]]
    dist = tfd.OneHotCategorical(np.log(probs) - 50., validate_args=True)
    n = 100
    samples = dist.sample(n, seed=test_util.test_seed())
    self.assertEqual(samples.dtype, tf.int32)
    sample_values = self.evaluate(samples)
    self.assertAllEqual([n, 1, 2, 2], sample_values.shape)
    self.assertFalse(np.any(sample_values < 0))
    self.assertFalse(np.any(sample_values > 1))

  def testSampleWithSampleShape(self):
    probs = [[[0.2, 0.8], [0.4, 0.6]]]
    dist = tfd.OneHotCategorical(np.log(probs) - 50., validate_args=True)
    samples = dist.sample((100, 100), seed=test_util.test_seed())
    prob = dist.prob(samples)
    prob_val = self.evaluate(prob)
    self.assertAllClose(
        [0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()], atol=1e-2)
    self.assertAllClose(
        [0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()], atol=1e-2)

  def testCategoricalCategoricalKL(self):
    def np_softmax(logits):
      exp_logits = np.exp(logits)
      return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    for categories in [2, 10]:
      for batch_size in [1, 2]:
        p_logits = self._rng.random_sample((batch_size, categories))
        q_logits = self._rng.random_sample((batch_size, categories))
        p = tfd.OneHotCategorical(logits=p_logits, validate_args=True)
        q = tfd.OneHotCategorical(logits=q_logits, validate_args=True)
        prob_p = np_softmax(p_logits)
        prob_q = np_softmax(q_logits)
        kl_expected = np.sum(
            prob_p * (np.log(prob_p) - np.log(prob_q)), axis=-1)

        kl_actual = tfd.kl_divergence(p, q)
        kl_same = tfd.kl_divergence(p, p)
        x = p.sample(int(2e4), seed=test_util.test_seed())
        x = tf.cast(x, dtype=tf.float32)
        # Compute empirical KL(p||q).
        kl_sample = tf.reduce_mean(
            p.log_prob(x) - q.log_prob(x), axis=0)

        [kl_sample_, kl_actual_,
         kl_same_] = self.evaluate([kl_sample, kl_actual, kl_same])
        self.assertEqual(kl_actual.shape, (batch_size,))
        self.assertAllClose(kl_same_, np.zeros_like(kl_expected))
        self.assertAllClose(kl_actual_, kl_expected, atol=0., rtol=1e-4)
        self.assertAllClose(kl_sample_, kl_expected, atol=1e-2, rtol=0.)

  def testSampleUnbiasedNonScalarBatch(self):
    logits = self._rng.rand(4, 3, 2).astype(np.float32)
    dist = tfd.OneHotCategorical(logits=logits, validate_args=True)
    n = int(3e3)
    x = dist.sample(n, seed=test_util.test_seed())
    x = tf.cast(x, dtype=tf.float32)
    sample_mean = tf.reduce_mean(x, axis=0)
    x_centered = tf.transpose(a=x - sample_mean, perm=[1, 2, 3, 0])
    sample_covariance = tf.matmul(x_centered, x_centered, adjoint_b=True) / n
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
    self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.07)
    self.assertAllEqual([4, 3, 2, 2], sample_covariance.shape)
    self.assertAllClose(
        actual_covariance_, sample_covariance_, atol=0., rtol=0.10)

  def testSampleUnbiasedScalarBatch(self):
    logits = self._rng.rand(3).astype(np.float32)
    dist = tfd.OneHotCategorical(logits=logits, validate_args=True)
    n = int(1e4)
    x = dist.sample(n, seed=test_util.test_seed())
    x = tf.cast(x, dtype=tf.float32)
    sample_mean = tf.reduce_mean(x, axis=0)  # elementwise mean
    x_centered = x - sample_mean
    sample_covariance = tf.matmul(x_centered, x_centered, adjoint_a=True) / n
    [
        sample_mean_,
        sample_covariance_,
        actual_mean_,
        actual_covariance_,
    ] = self.evaluate([
        sample_mean,
        sample_covariance,
        dist.probs_parameter(),
        dist.covariance(),
    ])
    self.assertAllEqual([3], sample_mean.shape)
    self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.1)
    self.assertAllEqual([3, 3], sample_covariance.shape)
    self.assertAllClose(
        actual_covariance_, sample_covariance_, atol=0., rtol=0.1)

  def testParamTensorFromLogits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = tfd.OneHotCategorical(logits=x, validate_args=True)
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
    d = tfd.OneHotCategorical(probs=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([tf.math.log(x), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([x, d.probs_parameter()]),
        atol=0, rtol=1e-4)


@test_util.test_all_tf_execution_regimes
class OneHotCategoricalFromVariableTest(test_util.TestCase):

  @test_util.tf_tape_safety_test
  def testGradientLogits(self):
    x = tf.Variable([-1., 0., 1])
    d = tfd.OneHotCategorical(logits=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([[1, 0, 0], [0, 0, 1]])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  @test_util.tf_tape_safety_test
  def testGradientProbs(self):
    x = tf.Variable([0.1, 0.7, 0.2])
    d = tfd.OneHotCategorical(probs=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([[1, 0, 0], [0, 0, 1]])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  def testAssertionsProbs(self):
    x = tf.Variable([0.1, 0.7, 0.0])
    with self.assertRaisesOpError('Argument `probs` must sum to 1.'):
      d = tfd.OneHotCategorical(probs=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.entropy())

  def testAssertionsProbsAfterMutation(self):
    x = tf.Variable([0.25, 0.25, 0.5])
    d = tfd.OneHotCategorical(probs=x, validate_args=True)
    with self.assertRaisesOpError('Condition x >= 0 did not hold element-wise'):
      self.evaluate([v.initializer for v in d.variables])
      with tf.control_dependencies([x.assign([-0.25, 0.75, 0.5])]):
        self.evaluate(d.logits_parameter())

  def testAssertionsLogits(self):
    x = tfp.util.TransformedVariable(0., tfb.Identity(), shape=None)
    with self.assertRaisesRegexp(
        ValueError, 'Argument `logits` must have rank at least 1.'):
      d = tfd.OneHotCategorical(logits=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.entropy())


if __name__ == '__main__':
  tf.test.main()
