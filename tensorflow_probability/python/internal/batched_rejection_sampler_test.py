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

from absl.testing import parameterized
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import batched_rejection_sampler as brs
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class BatchedRejectionSamplerTest(test_util.TestCase):

  def testBatchedLasVegasAlgorithm(self):
    def uniform_less_than_point_five(seed):
      values = samplers.uniform([6], seed=seed)
      negative_values = -values
      good = tf.less(values, 0.5)

      return ((negative_values, values), good)

    ((negative_values, values), _) = self.evaluate(
        brs.batched_las_vegas_algorithm(
            uniform_less_than_point_five,
            seed=test_util.test_seed()))

    self.assertAllLess(values, 0.5)
    self.assertAllClose(-values, negative_values)

    # Check for reproducibility.
    ((negative_values_2, values_2), _) = self.evaluate(
        brs.batched_las_vegas_algorithm(
            uniform_less_than_point_five,
            seed=test_util.test_seed()))
    self.assertAllEqual(negative_values, negative_values_2)
    self.assertAllEqual(values, values_2)

  @parameterized.named_parameters(
      dict(testcase_name='_static_float32', is_static=True, dtype=tf.float32),
      dict(testcase_name='_static_float64', is_static=True, dtype=tf.float64),
      dict(testcase_name='_dynamic_float32', is_static=False, dtype=tf.float32),
      dict(testcase_name='_dynamic_float64', is_static=False, dtype=tf.float64))
  def testBatchedRejectionBetaSample(self, is_static, dtype):
    # We build a rejection sampler for two beta distributions (in a batch): a
    # beta(2, 5) and a beta(2, 2). Eyeballing an image on the wikipedia page,
    # these are upper bounded by rectangles of heights 2.5 and 1.6 respectively.
    numpy_dtype = dtype_util.as_numpy_dtype(dtype)
    alpha = np.array([2.], dtype=numpy_dtype)
    beta = np.array([5., 2.], dtype=numpy_dtype)
    upper_bounds = tf.constant([2.5, 1.6], dtype=dtype)
    samples_per_distribution = 10000

    target_fn = tfd.Beta(alpha, beta).prob

    def proposal_fn(seed):
      # Test static and dynamic shape of proposed samples.
      uniform_samples = self.maybe_static(
          samplers.uniform(
              [samples_per_distribution, 2], seed=seed, dtype=dtype),
          is_static)
      return uniform_samples, tf.ones_like(uniform_samples) * upper_bounds

    all_samples, _ = self.evaluate(brs.batched_rejection_sampler(
        proposal_fn, target_fn, seed=test_util.test_seed(), dtype=dtype))

    for i in range(beta.shape[0]):
      samples = all_samples[:, i]
      ks, _ = sp_stats.kstest(samples, sp_stats.beta(alpha, beta[i]).cdf)
      self.assertLess(ks, 0.02)

    # Check for reproducibility.
    all_samples_2, _ = self.evaluate(brs.batched_rejection_sampler(
        proposal_fn, target_fn, seed=test_util.test_seed(), dtype=dtype))
    self.assertAllEqual(all_samples, all_samples_2)

if __name__ == '__main__':
  tf.test.main()
