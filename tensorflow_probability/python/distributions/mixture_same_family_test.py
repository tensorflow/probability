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
tfd = tfp.distributions
tfe = tf.contrib.eager


class _MixtureSameFamilyTest(test_util.VectorDistributionTestHelpers):

  def testSampleAndLogProbUnivariateShapes(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([0.3, 0.7])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([-1., 1]),
            scale=self._build_tensor([0.1, 0.5])))
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5], self._shape(x))
    self.assertAllEqual([4, 5], self._shape(log_prob_x))

  def testSampleAndLogProbBatch(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([[0.3, 0.7]])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([[-1., 1]]),
            scale=self._build_tensor([[0.1, 0.5]])))
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 1], self._shape(x))
    self.assertAllEqual([4, 5, 1], self._shape(log_prob_x))

  def testSampleAndLogProbShapesBroadcastMix(self):
    mix_probs = self._build_tensor([.3, .7])
    bern_probs = self._build_tensor([[.4, .6], [.25, .75]])
    bm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mix_probs),
        components_distribution=tfd.Bernoulli(probs=bern_probs))
    x = bm.sample([4, 5], seed=42)
    log_prob_x = bm.log_prob(x)
    x_ = self.evaluate(x)
    self.assertAllEqual([4, 5, 2], self._shape(x))
    self.assertAllEqual([4, 5, 2], self._shape(log_prob_x))
    self.assertAllEqual(
        np.ones_like(x_, dtype=np.bool), np.logical_or(x_ == 0., x_ == 1.))

  def testSampleAndLogProbMultivariateShapes(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 2], self._shape(x))
    self.assertAllEqual([4, 5], self._shape(log_prob_x))

  def testSampleAndLogProbBatchMultivariateShapes(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[[-1., 1], [1, -1]], [[0., 1], [1, 0]]],
        scale_identity_multiplier=[1., 0.5])
    x = gm.sample([4, 5], seed=42)
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 2, 2], self._shape(x))
    self.assertAllEqual([4, 5, 2], self._shape(log_prob_x))

  def testSampleConsistentLogProb(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate, gm, radius=1., center=[-1., 1], rtol=0.02)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate, gm, radius=1., center=[1., -1], rtol=0.02)

  def testLogCdf(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([0.3, 0.7])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([-1., 1]),
            scale=self._build_tensor([0.1, 0.5])))
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
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    self.run_test_sample_consistent_mean_covariance(self.evaluate, gm)

  def testVarianceConsistentCovariance(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    cov_, var_ = self.evaluate([gm.covariance(), gm.variance()])
    self.assertAllClose(cov_.diagonal(), var_, atol=0.)

  def _shape(self, x):
    if self.use_static_shape:
      return x.shape.as_list()
    else:
      return self.evaluate(tf.shape(x))

  def _build_mvndiag_mixture(self, probs, loc, scale_identity_multiplier):
    components_distribution = tfd.MultivariateNormalDiag(
        loc=self._build_tensor(loc),
        scale_identity_multiplier=self._build_tensor(
            scale_identity_multiplier))

    # Use a no-op `Independent` wrapper to possibly create dynamic ndims.
    wrapped_components_distribution = tfd.Independent(
        components_distribution,
        reinterpreted_batch_ndims=self._build_tensor(0, dtype=np.int32))
    # Lambda ensures that the covariance fn sees `self=components_distribution`.
    wrapped_components_distribution._covariance = (
        lambda: components_distribution.covariance())  # pylint: disable=unnecessary-lambda

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor(probs)),
        components_distribution=wrapped_components_distribution)
    return gm

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    if self.use_static_shape:
      return tf.convert_to_tensor(ndarray)
    else:
      return tf.placeholder_with_default(
          input=ndarray, shape=None)


@tfe.run_all_tests_in_graph_and_eager_modes
class MixtureSameFamilyTestStatic32(
    _MixtureSameFamilyTest,
    tf.test.TestCase):
  use_static_shape = True
  dtype = np.float32


@tfe.run_all_tests_in_graph_and_eager_modes
class MixtureSameFamilyTestDynamic32(
    _MixtureSameFamilyTest,
    tf.test.TestCase):
  use_static_shape = False
  dtype = np.float32


@tfe.run_all_tests_in_graph_and_eager_modes
class MixtureSameFamilyTestStatic64(
    _MixtureSameFamilyTest,
    tf.test.TestCase):
  use_static_shape = True
  dtype = np.float64

if __name__ == "__main__":
  tf.test.main()
