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

import itertools
# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfb = tfp.bijectors


@test_util.test_all_tf_execution_regimes
class OrderedLogisticTest(test_util.TestCase):

  def _random_cutpoints(self, shape):
    return self._ordered.inverse(self._rng.randn(*shape))

  def _random_location(self, shape):
    return self._rng.randn(*shape)

  def setUp(self):
    self._ordered = tfb.Ordered()
    self._rng = np.random.RandomState(test_util.test_seed())
    super(OrderedLogisticTest, self).setUp()

  @parameterized.parameters(
      itertools.product(['cutpoints', 'loc', 'both'], [[], [1], [1, 2, 3]])
  )
  def testBatchShapes(self, test, batch_shape):

    if test == 'cutpoints':
      cutpoints = self._random_cutpoints(batch_shape + [2])
      loc = tf.constant(0., dtype=tf.float64)
    elif test == 'loc':
      cutpoints = tf.constant([1., 2.], dtype=tf.float64)
      loc = self._random_location(batch_shape)
    elif test == 'both':
      cutpoints = self._random_cutpoints(batch_shape + [2])
      loc = self._random_location(batch_shape)

    dist = tfd.OrderedLogistic(cutpoints=cutpoints, loc=loc)

    self.assertAllEqual(dist.batch_shape, batch_shape)
    self.assertAllEqual(
        self.evaluate(dist.batch_shape_tensor()), batch_shape)

    self.assertAllEqual(dist.event_shape, [])
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])

    categorical_probs = dist.categorical_probs()
    categorical_probs_shape = tf.shape(categorical_probs)
    self.assertAllEqual(
        self.evaluate(categorical_probs_shape), batch_shape + [3])

    sample = dist.sample(seed=test_util.test_seed())
    sample_shape = tf.shape(sample)
    self.assertAllEqual(self.evaluate(sample_shape), batch_shape)

    prob_sample_shape = tf.shape(dist.prob(sample))
    survival_prob_sample_shape = tf.shape(dist.survival_function(sample))
    self.assertAllEqual(self.evaluate(prob_sample_shape), batch_shape)
    self.assertAllEqual(self.evaluate(survival_prob_sample_shape), batch_shape)

    n = [4, 5]
    sample_n = dist.sample(n, seed=test_util.test_seed())
    sample_n_shape = tf.shape(sample_n)
    self.assertAllEqual(self.evaluate(sample_n_shape), n + batch_shape)

    prob_sample_n_shape = tf.shape(dist.prob(sample_n))
    survival_prob_sample_n_shape = tf.shape(dist.survival_function(sample_n))
    self.assertAllEqual(self.evaluate(prob_sample_n_shape), n + batch_shape)
    self.assertAllEqual(
        self.evaluate(survival_prob_sample_n_shape), n + batch_shape)

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
    expected_survival_probs = 1. - np.cumsum(expected_probs)
    dist = tfd.OrderedLogistic(cutpoints=[-1., 0., 1.], loc=0.)

    categorical_probs = self.evaluate(dist.categorical_probs())
    self.assertAllClose(expected_probs, categorical_probs, atol=1e-6)

    probs = np.flip(self.evaluate(dist.prob([3, 2, 1, 0])))
    self.assertAllClose(expected_probs, probs, atol=1e-6)

    survival_probs = self.evaluate(dist.survival_function([0, 1, 2, 3]))
    self.assertAllClose(expected_survival_probs, survival_probs, atol=1e-6)

    zero_probs = self.evaluate(dist.prob([-1, 4]))
    self.assertAllClose([0., 0.], zero_probs, atol=1e-6)

    out_of_bounds_survival_probs = self.evaluate(
        dist.survival_function([-2, -1, 4, 5]))
    self.assertAllClose(
        [1., 1., 0., 0.], out_of_bounds_survival_probs, atol=1e-6)

  def testMode(self):
    # 2 cutpoints i.e. 3 possible outcomes. 3 "batched" distributions with the
    # logistic distribution location well within the large cutpoint spacing so
    # mode is obvious
    dist = tfd.OrderedLogistic(cutpoints=[-10., 10.], loc=[-20., 0., 20.])
    mode = self.evaluate(dist.mode())
    self.assertAllEqual([0, 1, 2], mode)

  def testSample(self):
    # as per `testProbs`
    dist = tfd.OrderedLogistic(cutpoints=[-1., 0., 1.], loc=0.)
    samples = self.evaluate(dist.sample(int(1e5), seed=test_util.test_seed()))
    expected_probs = [0.2689414, 0.2310586, 0.23105857, 0.26894143]
    for k, p in enumerate(expected_probs):
      self.assertAllClose(np.mean(samples == k), p, atol=0.01)

  def testEntropyAgainstCategoricalDistribution(self):
    cutpoints = self._random_cutpoints([3])
    loc = self._random_location([2])
    dist = tfd.OrderedLogistic(cutpoints=cutpoints, loc=loc)
    categorical_dist = tfd.Categorical(dist.categorical_log_probs())
    expected_entropy = self.evaluate(categorical_dist.entropy())
    entropy = self.evaluate(dist.entropy())
    self.assertAllClose(expected_entropy, entropy)

  def testEntropyAgainstSampling(self):
    cutpoints = self._random_cutpoints([4])
    loc = self._random_location([])
    dist = tfd.OrderedLogistic(cutpoints=cutpoints, loc=loc)
    samples = dist.sample(int(1e5), seed=test_util.test_seed())
    sampled_entropy = self.evaluate(-tf.reduce_mean(dist.log_prob(samples)))
    entropy = self.evaluate(dist.entropy())
    self.assertAllClose(sampled_entropy, entropy, atol=0.01)

  @parameterized.parameters(1, 10, 25)
  def testKLAgainstCategoricalDistribution(self, batch_size):
    cutpoints = self._random_cutpoints([100])
    a_loc = self._random_location([batch_size])
    b_loc = self._random_location([batch_size])

    a = tfd.OrderedLogistic(
        cutpoints=cutpoints, loc=a_loc, validate_args=True)
    b = tfd.OrderedLogistic(
        cutpoints=cutpoints, loc=b_loc, validate_args=True)

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

  def testKLAgainstSampling(self):
    a_cutpoints = self._random_cutpoints([4])
    b_cutpoints = self._random_cutpoints([4])
    loc = self._random_location([])

    a = tfd.OrderedLogistic(cutpoints=a_cutpoints, loc=loc)
    b = tfd.OrderedLogistic(cutpoints=b_cutpoints, loc=loc)

    samples = a.sample(int(1e5), seed=test_util.test_seed())
    sampled_kl = self.evaluate(
        tf.reduce_mean(a.log_prob(samples) - b.log_prob(samples)))
    kl = self.evaluate(tfd.kl_divergence(a, b))

    self.assertAllClose(sampled_kl, kl, atol=2e-2)

  def testLatentLogistic(self):
    loc = self._random_location([2])
    cutpoints = self._random_cutpoints([2])
    latent = tfd.Logistic(loc=loc, scale=1.)
    ordered = tfd.OrderedLogistic(cutpoints=cutpoints, loc=loc)
    ordered_cdf = self.evaluate(ordered.cdf([0, 1]))
    latent_cdf = self.evaluate(latent.cdf(cutpoints))
    self.assertAllClose(ordered_cdf, latent_cdf)

  def testUnorderedCutpointsFails(self):
    with self.assertRaisesRegexp(
        ValueError, 'Argument `cutpoints` must be non-decreasing.'):
      dist = tfd.OrderedLogistic(
          cutpoints=[1., 0.9], loc=0.0, validate_args=True)
      self.evaluate(dist.mode())

if __name__ == '__main__':
  tf.test.main()
