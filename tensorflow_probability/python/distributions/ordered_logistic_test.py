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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class OrderedLogisticTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(OrderedLogisticTest, self).setUp()

  def testBatchShapes(self):

    for test in ["cutpoints", "location", "both"]:

      for batch_shape in ([], [1], [1, 2, 3]):

        if test == "cutpoints":
          cutpoints = tf.broadcast_to(
              tf.constant([1., 2.], dtype=tf.float64), batch_shape + [2])
          location = tf.constant(0., dtype=tf.float64)
        elif test == "location":
          cutpoints = tf.constant([1., 2.], dtype=tf.float64)
          location = self._rng.randn(*batch_shape)
        elif test == "both":
          cutpoints = tf.broadcast_to(
              tf.constant([1., 2.], dtype=tf.float64), batch_shape + [2])
          location = self._rng.randn(*batch_shape)

        dist = tfd.OrderedLogistic(cutpoints, location)

        self.assertAllEqual(dist.batch_shape, batch_shape)
        self.assertAllEqual(
            self.evaluate(dist.batch_shape_tensor()), batch_shape)

        self.assertAllEqual(dist.event_shape, [])
        self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])

        log_probs_shape = tf.shape(dist.categorical_log_probs())
        self.assertAllEqual(self.evaluate(log_probs_shape), batch_shape + [3])

        sample_shape = tf.shape(dist.sample(seed=test_util.test_seed()))
        self.assertAllEqual(self.evaluate(sample_shape), batch_shape)

        sample_shape_n = tf.shape(
            dist.sample([4, 5], seed=test_util.test_seed()))
        self.assertAllEqual(self.evaluate(sample_shape_n), [4, 5] + batch_shape)

  def testProbs(self):

    # survival functions
    # P(Y > 0) = sigmoid(1) = 0.7310586
    # P(Y > 1) = sigmoid(0) = 0.5
    # P(Y > 2) = sigmoid(-1) = 0.26894143

    # probs
    # P(Y = 0) = 1. - sigmoid(1) = 0.2689414
    # P(Y = 1) = sigmoid(1) - sigmoid(0) = 0.2310586
    # P(Y = 2) = sigmoid(0) - sigmoid(-1) = 0.23105857
    # P(Y = 3) = sigmoid(-1) = 0.26894143
    expected_probs = [0.2689414, 0.2310586, 0.23105857, 0.26894143]
    dist = tfd.OrderedLogistic([-1., 0., 1.], 0.)

    categorical_probs = self.evaluate(dist.categorical_probs())
    self.assertAllClose(expected_probs, categorical_probs, atol=1e-6)

    probs = np.flip(self.evaluate(dist.prob([3, 2, 1, 0])))
    self.assertAllClose(expected_probs, probs, atol=1e-6)

  def testMode(self):
    # 2 cutpoints i.e. 3 possible outcomes. 3 "batched" distributions with the
    # logistic distribution location well within the large cutpoint spacing so
    # mode is obvious
    dist = tfd.OrderedLogistic([-10., 10.], [-20., 0., 20.])
    mode = self.evaluate(dist.mode())
    self.assertAllEqual([0, 1, 2], mode)

  def testSample(self):
    # as per `testProbs`
    dist = tfd.OrderedLogistic([-1., 0., 1.], 0.)
    samples = self.evaluate(dist.sample(100000, seed=test_util.test_seed()))
    expected_probs = [0.2689414, 0.2310586, 0.23105857, 0.26894143]
    for k, p in enumerate(expected_probs):
      self.assertAllClose(np.mean(samples == k), p, atol=0.01)

  def testEntropy(self):
    dist = tfd.OrderedLogistic([-4.2, 0., 0.123], [-3.14, 0.321])
    categorical_dist = tfd.Categorical(dist.categorical_log_probs())
    expected_entropy = self.evaluate(categorical_dist.entropy())
    entropy = self.evaluate(dist.entropy())
    self.assertAllClose(expected_entropy, entropy)

  def testKL(self):
    for batch_size in [1, 10]:
      cutpoints = tf.constant([-1., 1., 2.], dtype=tf.float64)
      a_location = self._rng.randn(batch_size)
      b_location = self._rng.randn(batch_size)

      a = tfd.OrderedLogistic(cutpoints, a_location, validate_args=True)
      b = tfd.OrderedLogistic(cutpoints, b_location, validate_args=True)

      a_cat = tfd.Categorical(
          logits=a.categorical_log_probs(), validate_args=True)
      b_cat = tfd.Categorical(
          logits=b.categorical_log_probs(), validate_args=True)

      kl = self.evaluate(tfd.kl_divergence(a, b))
      self.assertEqual(kl.shape, (batch_size,))

      kl_expected = self.evaluate(tfd.kl_divergence(a_cat, b_cat))
      self.assertAllClose(kl, kl_expected)

      kl_same = self.evaluate(tfd.kl_divergence(a, a))
      self.assertAllClose(kl_same, np.zeros_like(kl_expected))

  def testUnorderedCutpointsFails(self):
    with self.assertRaisesRegexp(
        ValueError, 'Argument `cutpoints` must be non-decreasing.'):
      dist = tfd.OrderedLogistic([1., 0.9], 0.0, validate_args=True)
      self.evaluate(dist.mode())

if __name__ == '__main__':
  tf.test.main()
