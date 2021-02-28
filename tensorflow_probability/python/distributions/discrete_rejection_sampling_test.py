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
"""Tests for discrete rejection sampling utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import discrete_rejection_sampling
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class DiscreteRejectionSamplingTest(test_util.TestCase):

  def testUnivariateLogConcaveDistributionRejectionSamplerGeometric(self):
    seed = test_util.test_seed()
    n = int(5e5)

    probs = np.float32([0.7, 0.8, 0.3, 0.2])
    geometric = tfd.Geometric(probs=probs)
    x = discrete_rejection_sampling.log_concave_rejection_sampler(
        mode=geometric.mode(), prob_fn=geometric.prob, dtype=geometric.dtype,
        sample_shape=[n], distribution_minimum=0, seed=seed)

    x = x + 1  ## scipy.stats.geom is 1-indexed instead of 0-indexed.
    sample_mean, sample_variance = tf.nn.moments(x=x, axes=0)
    [
        sample_mean_,
        sample_variance_,
    ] = self.evaluate([
        sample_mean,
        sample_variance,
    ])
    self.assertAllEqual([4], sample_mean.shape)
    self.assertAllClose(
        stats.geom.mean(probs), sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4], sample_variance.shape)
    self.assertAllClose(
        stats.geom.var(probs), sample_variance_, atol=0., rtol=0.20)


if __name__ == '__main__':
  tf.test.main()
