# Copyright 2020 The TensorFlow Probability Authors.
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

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class BetaBinomialTest(test_util.TestCase):

  def testSimpleShapes(self):
    n = np.array((3, 4, 5)).astype(np.float64)
    c1 = np.random.rand(3)
    c0 = np.random.rand(3)
    dist = tfd.BetaBinomial(n, c1, c0, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)

  def testComplexShapes(self):
    n = np.random.randint(1, 10, size=(5, 4)).astype(np.float64)
    c1 = np.random.rand(2, 1, 1)
    c0 = np.random.rand(3, 1, 5, 1)
    dist = tfd.BetaBinomial(n, c1, c0, validate_args=True)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 5, 4], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2, 5, 4]), dist.batch_shape)

  def testProperties(self):
    n = [3., 4, 5]
    c1 = [[0.5, 1.0, 2.0]]
    c0 = [[[3.0, 2.0, 1.0]]]
    dist = tfd.BetaBinomial(n, c1, c0)
    self.assertEqual([3], dist.total_count.shape)
    self.assertEqual([1, 3], dist.concentration1.shape)
    self.assertEqual([1, 1, 3], dist.concentration0.shape)

  def testSampleAgainstMeanAndVariance(self):
    seed_stream = test_util.test_seed_stream()

    n = self.evaluate(
        tf.cast(
            tf.random.uniform(shape=[3, 1], minval=1, maxval=8193,
                              dtype=tf.int32, seed=seed_stream()),
            tf.float32))
    c1 = self.evaluate(
        1. + 2. * tf.random.uniform(
            shape=[4, 3, 2], dtype=tf.float32, seed=seed_stream()))
    c0 = self.evaluate(
        1. + 2. * tf.random.uniform(
            shape=[4, 3, 1], dtype=tf.float32, seed=seed_stream()))

    dist = tfd.BetaBinomial(n, c1, c0, validate_args=True)

    num_samples = int(1e4)
    x = dist.sample(num_samples, seed=seed_stream())
    sample_mean, sample_variance = self.evaluate(tf.nn.moments(x=x, axes=0))

    self.assertAllEqual([4, 3, 2], sample_mean.shape)
    self.assertAllClose(
        self.evaluate(dist.mean()), sample_mean, atol=0., rtol=0.10)
    self.assertAllEqual([4, 3, 2], sample_variance.shape)
    self.assertAllClose(
        dist.variance(), sample_variance, atol=0., rtol=0.10)

  def testMeanAndVarianceAgainstDirichletMultinomial(self):
    seed_stream = test_util.test_seed_stream()

    n = tf.constant([10., 20., 30.])
    c1 = self.evaluate(1. + 2. * tf.random.uniform(
        shape=[4, 3], dtype=tf.float32, seed=seed_stream()))
    c0 = self.evaluate(1. + 2. * tf.random.uniform(
        shape=[4, 3], dtype=tf.float32, seed=seed_stream()))

    beta_binomial = tfd.BetaBinomial(n, c1, c0, validate_args=True)
    dirichlet_multinomial = tfd.DirichletMultinomial(
        n, tf.stack([c1, c0], axis=-1), validate_args=True)

    beta_binomial_mean = self.evaluate(beta_binomial.mean())
    dirichlet_multinomial_mean = self.evaluate(dirichlet_multinomial.mean())

    self.assertEqual((4, 3), beta_binomial_mean.shape)
    self.assertEqual((4, 3, 2), dirichlet_multinomial_mean.shape)
    self.assertAllClose(beta_binomial_mean,
                        np.squeeze(dirichlet_multinomial_mean[..., 0]))
    self.assertAllClose(n - beta_binomial_mean,
                        np.squeeze(dirichlet_multinomial_mean[..., 1]))

    beta_binomial_variance = self.evaluate(beta_binomial.variance())
    dirichlet_multinomial_variance = self.evaluate(
        dirichlet_multinomial.variance())
    self.assertEqual((4, 3), beta_binomial_variance.shape)
    self.assertEqual((4, 3, 2), dirichlet_multinomial_variance.shape)
    self.assertAllClose(beta_binomial_variance,
                        np.squeeze(dirichlet_multinomial_variance[..., 0]))

  def testSampleAgainstProb(self):
    seed_stream = test_util.test_seed_stream()

    n = 4
    c1 = self.evaluate(
        1. + 2. * tf.random.uniform(
            shape=[4, 3, 2], dtype=tf.float32, seed=seed_stream()))
    c0 = self.evaluate(
        1. + 2. * tf.random.uniform(
            shape=[4, 3, 1], dtype=tf.float32, seed=seed_stream()))
    dist = tfd.BetaBinomial(
        tf.cast(n, dtype=tf.float32), c1, c0, validate_args=True)

    num_samples = int(1e4)
    x = self.evaluate(
        tf.cast(dist.sample(num_samples, seed=seed_stream()), tf.int32))

    for i in range(n + 1):
      self.assertAllClose(
          self.evaluate(dist.prob(i)),
          np.sum(x == i, axis=0) / (num_samples * 1.0),
          atol=0.01, rtol=0.1)

  def testEmpiricalCdfAgainstDirichletMultinomial(self):
    # This test is too slow for Eager mode.
    if tf.executing_eagerly():
      return

    seed_stream = test_util.test_seed_stream()

    n = 10
    c1 = self.evaluate(
        1. + 2. * tf.random.uniform(
            shape=[3], dtype=tf.float32, seed=seed_stream()))
    c0 = self.evaluate(
        1. + 2. * tf.random.uniform(
            shape=[3], dtype=tf.float32, seed=seed_stream()))

    beta_binomial = tfd.BetaBinomial(n, c1, c0, validate_args=True)
    dirichlet_multinomial = tfd.DirichletMultinomial(
        n, tf.stack([c1, c0], axis=-1), validate_args=True)

    num_samples_to_draw = tf.math.floor(
        1 + st.min_num_samples_for_dkwm_cdf_two_sample_test(.02)[0])

    beta_binomial_samples = beta_binomial.sample(num_samples_to_draw)

    dirichlet_multinomial_samples = dirichlet_multinomial.sample(
        num_samples_to_draw)
    dirichlet_multinomial_samples = tf.squeeze(
        dirichlet_multinomial_samples[..., 0])

    self.evaluate(st.assert_true_cdf_equal_by_dkwm_two_sample(
        beta_binomial_samples, dirichlet_multinomial_samples))

  def testLogProbCountsValid(self):
    d = tfd.BetaBinomial([10., 4.], 1., 1., validate_args=True)
    d.log_prob([0., 0.])
    d.log_prob([0., 2.])
    d.log_prob([2., 4.])
    d.log_prob([10., 4.])
    with self.assertRaisesOpError('must be non-negative.'):
      d.log_prob([-1., 3.])
    with self.assertRaisesOpError('cannot contain fractional components.'):
      d.log_prob([3., 3.5])
    with self.assertRaisesOpError(
        'must be itemwise less than or equal to `total_count` parameter.'):
      d.log_prob(5.)

  def testLogProbAgainstDirichletMultinomial(self):
    seed_stream = test_util.test_seed_stream()

    n = tf.constant([10., 20., 30.])
    c1 = self.evaluate(1. + 2. * tf.random.uniform(
        shape=[4, 3], dtype=tf.float32, seed=seed_stream()))
    c0 = self.evaluate(1. + 2. * tf.random.uniform(
        shape=[4, 3], dtype=tf.float32, seed=seed_stream()))

    beta_binomial = tfd.BetaBinomial(n, c1, c0, validate_args=True)
    dirichlet_multinomial = tfd.DirichletMultinomial(
        n, tf.stack([c1, c0], axis=-1), validate_args=True)

    num_samples = 3

    beta_binomial_sample = self.evaluate(
        beta_binomial.sample(num_samples, seed=seed_stream()))
    beta_binomial_log_prob = beta_binomial.log_prob(beta_binomial_sample)
    dirichlet_multinomial_log_prob = dirichlet_multinomial.log_prob(
        tf.stack([beta_binomial_sample, n - beta_binomial_sample], axis=-1))
    self.assertAllClose(self.evaluate(beta_binomial_log_prob),
                        self.evaluate(dirichlet_multinomial_log_prob),
                        rtol=1e-4, atol=1e-4)

    dirichlet_multinomial_sample = self.evaluate(
        dirichlet_multinomial.sample(num_samples, seed=seed_stream()))
    dirichlet_multinomial_log_prob = dirichlet_multinomial.log_prob(
        dirichlet_multinomial_sample)
    beta_binomial_log_prob = beta_binomial.log_prob(
        tf.squeeze(dirichlet_multinomial_sample[..., 0]))
    self.assertAllClose(self.evaluate(dirichlet_multinomial_log_prob),
                        self.evaluate(beta_binomial_log_prob),
                        rtol=1e-4, atol=1e-4)

  def testNotReparameterized(self):
    if tf1.control_flow_v2_enabled():
      self.skipTest('b/138796859')
    n = tf.constant(5.0)
    c1 = tf.constant([0.1, 0.1, 0.1])
    c0 = tf.constant([0.3, 0.3, 0.3])

    def f(n, c1, c0):
      dist = tfd.BetaBinomial(n, c1, c0, validate_args=True)
      return dist.sample(100, seed=test_util.test_seed())

    _, [grad_n, grad_c1, grad_c0] = tfp.math.value_and_gradient(f, [n, c1, c0])
    self.assertIsNone(grad_n)
    self.assertIsNone(grad_c1)
    self.assertIsNone(grad_c0)


@test_util.test_all_tf_execution_regimes
class BetaBinomialFromVariableTest(test_util.TestCase):

  def testAssertionsTotalCount(self):
    total_count = tf.Variable([-1.0, 4.0, 1.0])
    d = tfd.BetaBinomial(total_count, 1.0, 1.0, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('`total_count` must be non-negative.'):
      self.evaluate(d.mean())

    total_count = tf.Variable([0.5, 4.0, 1.0])
    d = tfd.BetaBinomial(total_count, 1.0, 1.0, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError(
        '`total_count` cannot contain fractional components.'):
      self.evaluate(d.mean())

  def testAssertionsTotalCountMutation(self):
    total_count = tf.Variable([1.0, 4.0, 1.0])
    d = tfd.BetaBinomial(total_count, 1.0, 1.0, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    self.evaluate(d.mean())

    self.evaluate(total_count.assign(-total_count))
    with self.assertRaisesOpError('`total_count` must be non-negative.'):
      self.evaluate(d.mean())

    self.evaluate(total_count.assign(0.5 * -total_count))
    with self.assertRaisesOpError(
        '`total_count` cannot contain fractional components.'):
      self.evaluate(d.mean())

  def testAssertsPositiveConcentration1(self):
    concentration1 = tf.Variable([1., 2., -3.])
    self.evaluate(concentration1.initializer)
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      d = tfd.BetaBinomial(
          total_count=10, concentration1=concentration1, concentration0=[5.],
          validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentration1AfterMutation(self):
    concentration1 = tf.Variable([1., 2., 3.])
    self.evaluate(concentration1.initializer)
    d = tfd.BetaBinomial(
        total_count=10, concentration1=concentration1, concentration0=[5.],
        validate_args=True)
    self.evaluate(concentration1.assign([1., 2., -3.]))
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testLogProbGradientThroughConcentration1(self):
    concentration1 = tf.Variable(3.)
    d = tfd.BetaBinomial(
        total_count=10, concentration1=concentration1, concentration0=5.,
        validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([3., 4., 5.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveConcentration0(self):
    concentration0 = tf.Variable([1., 2., -3.])
    self.evaluate(concentration0.initializer)
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      d = tfd.BetaBinomial(
          total_count=10, concentration1=[5.], concentration0=concentration0,
          validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentration0AfterMutation(self):
    concentration0 = tf.Variable([1., 2., 3.])
    self.evaluate(concentration0.initializer)
    d = tfd.BetaBinomial(
        total_count=10, concentration1=[5.], concentration0=concentration0,
        validate_args=True)
    self.evaluate(concentration0.assign([1., 2., -3.]))
    with self.assertRaisesOpError('Concentration parameter must be positive.'):
      self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testLogProbGradientThroughConcentration0(self):
    concentration0 = tf.Variable(3.)
    d = tfd.BetaBinomial(
        total_count=10, concentration1=0.5, concentration0=concentration0,
        validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([3., 4., 5.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

if __name__ == '__main__':
  tf.test.main()
