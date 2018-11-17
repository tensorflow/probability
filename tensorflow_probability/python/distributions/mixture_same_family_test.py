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
"""Tests for MixtureSameFamily distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
from tensorflow.python.framework import test_util as tf_test_util

tfd = tfp.distributions


@tf_test_util.run_all_in_graph_and_eager_modes
class MixtureSameFamilyTest(test_util.VectorDistributionTestHelpers,
                            tf.test.TestCase):

  def testSampleAndLogProbUnivariateShapes(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.Normal(
            loc=[-1., 1], scale=[0.1, 0.5]))
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertEqual([4, 5], x.shape)
    self.assertEqual([4, 5], log_prob_x.shape)

  def testSampleAndLogProbBatch(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[[0.3, 0.7]]),
        components_distribution=tfd.Normal(
            loc=[[-1., 1]], scale=[[0.1, 0.5]]))
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertEqual([4, 5, 1], x.shape)
    self.assertEqual([4, 5, 1], log_prob_x.shape)

  def testSampleAndLogProbShapesBroadcastMix(self):
    mix_probs = np.float32([.3, .7])
    bern_probs = np.float32([[.4, .6], [.25, .75]])
    bm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mix_probs),
        components_distribution=tfd.Bernoulli(probs=bern_probs))
    x = bm.sample([4, 5], seed=42)
    log_prob_x = bm.log_prob(x)
    x_ = self.evaluate(x)
    self.assertEqual([4, 5, 2], x.shape)
    self.assertEqual([4, 5, 2], log_prob_x.shape)
    self.assertAllEqual(
        np.ones_like(x_, dtype=np.bool), np.logical_or(x_ == 0., x_ == 1.))

  def testSampleAndLogProbMultivariateShapes(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-1., 1], [1, -1]], scale_identity_multiplier=[1., 0.5]))
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertEqual([4, 5, 2], x.shape)
    self.assertEqual([4, 5], log_prob_x.shape)

  def testSampleAndLogProbBatchMultivariateShapes(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[[-1., 1], [1, -1]], [[0., 1], [1, 0]]],
            scale_identity_multiplier=[1., 0.5]))
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertEqual([4, 5, 2, 2], x.shape)
    self.assertEqual([4, 5, 2], log_prob_x.shape)

  def testSampleConsistentLogProb(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-1., 1], [1, -1]], scale_identity_multiplier=[1., 0.5]))
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate, gm, radius=1., center=[-1., 1], rtol=0.02)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate, gm, radius=1., center=[1., -1], rtol=0.02)

  def testLogCdf(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.Normal(
            loc=[-1., 1], scale=[0.1, 0.5]))
    x = gm.sample(10, seed=42)
    actual_log_cdf = gm.log_cdf(x)
    expected_log_cdf = tf.reduce_logsumexp(
        (gm.mixture_distribution.logits + gm.components_distribution.log_cdf(
            x[..., tf.newaxis])),
        axis=1)
    actual_log_cdf_, expected_log_cdf_ = self.evaluate(
        [actual_log_cdf, expected_log_cdf])
    self.assertAllClose(actual_log_cdf_, expected_log_cdf_, rtol=1e-6, atol=0.0)

  def testSampleConsistentMeanCovariance(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-1., 1], [1, -1]], scale_identity_multiplier=[1., 0.5]))
    self.run_test_sample_consistent_mean_covariance(self.evaluate, gm)

  def testVarianceConsistentCovariance(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-1., 1], [1, -1]], scale_identity_multiplier=[1., 0.5]))
    cov_, var_ = self.evaluate([gm.covariance(), gm.variance()])
    self.assertAllClose(cov_.diagonal(), var_, atol=0.)


if __name__ == "__main__":
  tf.test.main()
