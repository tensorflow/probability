# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for UnitNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from absl.testing import parameterized

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util

_MATCHING_METHODS = [
    "prob",
    "log_prob",
    "survival_function",
    "log_survival_function",
    "cdf",
    "log_cdf",
    "quantile",
]


@test_util.test_all_tf_execution_regimes
class UnitNormalTest(test_util.TestCase):

  @parameterized.parameters(_MATCHING_METHODS)
  def testMethodsMatchNormalScalarDistributions(self, method):
    unit_dist = tfd.UnitNormal()
    normal_dist = tfd.Normal(0.0, 1.0)
    x = [-np.inf, np.inf, np.nan, -1.0, 0.0, 1.0, 1.1234]
    f = getattr(unit_dist, method)
    g = getattr(normal_dist, method)
    self.assertAllClose(*self.evaluate([tf.identity(f(x)), g(x)]))

  @parameterized.parameters(_MATCHING_METHODS)
  def testMethodsMatchNormalBatchedDistributionsFloat64(self, method):
    unit_dist = tfd.UnitNormal(batch_shape=[1, 2, 3], dtype=tf.float64)
    normal_dist = tfd.Normal(tf.zeros([1, 2, 3], dtype=tf.float64), 1.0)
    x = np.random.randn(4, 1, 2, 3)
    f = getattr(unit_dist, method)
    g = getattr(normal_dist, method)
    self.assertAllClose(*self.evaluate([tf.identity(f(x)), g(x)]))

  def testDistributionPropertiesFloat64(self):
    dist = tfd.UnitNormal(batch_shape=[13], dtype=tf.float64)
    self.assertAllClose(self.evaluate(dist.loc), np.zeros([13]))
    self.assertAllClose(self.evaluate(dist.mean()), np.zeros([13]))
    self.assertAllClose(self.evaluate(dist.mode()), np.zeros([13]))
    self.assertAllClose(self.evaluate(dist.scale), np.ones([13]))
    self.assertAllClose(self.evaluate(dist.stddev()), np.ones([13]))

  def testSampleShapes(self):
    self.assertEqual(self._sample_shape([], []), [])
    self.assertEqual(self._sample_shape([2], []), [2])
    self.assertEqual(self._sample_shape([], [2]), [2])
    self.assertEqual(self._sample_shape([2], [3]), [3, 2])

  def testEntropyMatchesNormal(self):
    entropy_a = tfd.Normal(0.0, 1.0).entropy()
    entropy_b = tfd.UnitNormal().entropy()
    self.assertAllClose(*self.evaluate([entropy_a, entropy_b]))

  def testKLDivergenceIsZeroAndBroadcasts(self):
    a = tfd.UnitNormal(batch_shape=[2, 1])
    b = tfd.UnitNormal(batch_shape=[1, 2, 3])
    kl = self.evaluate(tfd.kl_divergence(a, b))
    self.assertAllClose(kl, np.zeros([1, 2, 3], dtype=np.float32))

  def _sample_shape(self, batch_shape, n):
    dist = tfd.UnitNormal(batch_shape=batch_shape)
    sample = dist.sample(n, seed=test_util.test_seed())
    return sample.shape


if __name__ == '__main__':
  tf.test.main()
